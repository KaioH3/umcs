package lexdb

import (
	"encoding/binary"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/kak/umcs/pkg/seed"
)

// Build reads seed roots and words, validates them, and writes a .lsdb binary file.
func Build(roots []seed.Root, words []seed.Word, outPath string) (*BuildStats, error) {
	if err := validate(roots, words); err != nil {
		return nil, err
	}

	// --- String heap ---
	heap := &stringHeap{}

	// Build root records
	rootByID := make(map[uint32]*seed.Root, len(roots))
	for i := range roots {
		rootByID[roots[i].RootID] = &roots[i]
	}

	// Count words per root and first word index (after sorting words by word_id)
	sort.Slice(words, func(i, j int) bool {
		return words[i].WordID < words[j].WordID
	})
	sort.Slice(roots, func(i, j int) bool {
		return roots[i].RootID < roots[j].RootID
	})

	type rootMeta struct {
		wordCount    uint32
		firstWordIdx uint32
		langCoverage uint32
	}
	meta := make(map[uint32]*rootMeta, len(roots))
	for _, r := range roots {
		meta[r.RootID] = &rootMeta{}
	}
	for i, w := range words {
		m, ok := meta[w.RootID]
		if !ok {
			return nil, fmt.Errorf("word %q references unknown root_id %d", w.Word, w.RootID)
		}
		if m.wordCount == 0 {
			m.firstWordIdx = uint32(i)
		}
		m.wordCount++
		langBit := langBitFromStr(w.Lang)
		m.langCoverage |= langBit
	}

	// Build RootRecords
	rootRecords := make([]RootRecord, len(roots))
	var globalLangFlags uint32
	for i, r := range roots {
		m := meta[r.RootID]
		globalLangFlags |= m.langCoverage
		rootRecords[i] = RootRecord{
			RootID:        r.RootID,
			WordCount:     m.wordCount,
			FirstWordIdx:  m.firstWordIdx,
			NameOffset:    heap.add(r.RootStr),
			LangCoverage:  m.langCoverage,
			ParentRootID:  r.ParentRootID,
			OriginOffset:  heap.add(r.Origin),
			MeaningOffset: heap.add(r.MeaningEN),
		}
	}

	// Build WordRecords
	wordRecords := make([]WordRecord, len(words))
	for i, w := range words {
		lang, ok := ParseLang(w.Lang)
		if !ok {
			return nil, fmt.Errorf("unknown lang %q for word %q", w.Lang, w.Word)
		}
		wordRecords[i] = WordRecord{
			WordID:     w.WordID,
			RootID:     w.RootID,
			Lang:       lang,
			Sentiment:  w.Sentiment,
			WordOffset: heap.add(w.Word),
			NormOffset: heap.add(normalize(w.Norm)),
			FreqRank:   w.FreqRank,
			Flags:      w.Flags,
		}
	}

	// Compute offsets
	rootTableOffset := uint32(HeaderSize)
	wordTableOffset := rootTableOffset + uint32(len(rootRecords))*RootRecordSize
	heapOffset := wordTableOffset + uint32(len(wordRecords))*WordRecordSize

	// Build the data bytes for checksum
	data := buildData(rootRecords, wordRecords, heap.bytes())
	checksum := fnv1a32(data)

	// Write file
	f, err := os.Create(outPath)
	if err != nil {
		return nil, fmt.Errorf("create %s: %w", outPath, err)
	}
	defer f.Close()

	write := func(v any) {
		if err != nil {
			return
		}
		err = binary.Write(f, ByteOrder, v)
	}

	// Header
	write(Magic)
	write(Version)
	write(uint32(len(rootRecords)))
	write(uint32(len(wordRecords)))
	write(uint32(len(heap.bytes())))
	write(rootTableOffset)
	write(wordTableOffset)
	write(heapOffset)
	write(globalLangFlags)
	write(checksum)
	// 6 reserved uint32s = 24 bytes
	for range 6 {
		write(uint32(0))
	}
	if err != nil {
		return nil, fmt.Errorf("write header: %w", err)
	}

	// Root table
	for _, rr := range rootRecords {
		write(rr.RootID)
		write(rr.WordCount)
		write(rr.FirstWordIdx)
		write(rr.NameOffset)
		write(rr.LangCoverage)
		write(rr.ParentRootID)
		write(rr.OriginOffset)
		write(rr.MeaningOffset)
		if err != nil {
			return nil, fmt.Errorf("write root table: %w", err)
		}
	}

	// Word table
	for _, wr := range wordRecords {
		write(wr.WordID)
		write(wr.RootID)
		write(wr.Lang)
		write(wr.Sentiment)
		write(wr.WordOffset)
		write(wr.NormOffset)
		write(wr.FreqRank)
		write(wr.Flags)
		if err != nil {
			return nil, fmt.Errorf("write word table: %w", err)
		}
	}

	// String heap
	if _, err = f.Write(heap.bytes()); err != nil {
		return nil, fmt.Errorf("write string heap: %w", err)
	}

	fi, _ := f.Stat()
	return &BuildStats{
		RootCount: len(rootRecords),
		WordCount: len(wordRecords),
		HeapSize:  len(heap.bytes()),
		FileSize:  fi.Size(),
		LangFlags: globalLangFlags,
	}, nil
}

// BuildStats contains summary information after a successful build.
type BuildStats struct {
	RootCount int
	WordCount int
	HeapSize  int
	FileSize  int64
	LangFlags uint32
}

func (s *BuildStats) Langs() string {
	var out []string
	if s.LangFlags&LangPT != 0 {
		out = append(out, "PT")
	}
	if s.LangFlags&LangEN != 0 {
		out = append(out, "EN")
	}
	if s.LangFlags&LangES != 0 {
		out = append(out, "ES")
	}
	if s.LangFlags&LangIT != 0 {
		out = append(out, "IT")
	}
	if s.LangFlags&LangDE != 0 {
		out = append(out, "DE")
	}
	return strings.Join(out, " ")
}

// validate checks consistency between roots and words.
func validate(roots []seed.Root, words []seed.Word) error {
	rootSet := make(map[uint32]bool, len(roots))
	for _, r := range roots {
		if r.RootID == 0 {
			return fmt.Errorf("root %q has root_id=0 (must start at 1)", r.RootStr)
		}
		if rootSet[r.RootID] {
			return fmt.Errorf("duplicate root_id %d", r.RootID)
		}
		rootSet[r.RootID] = true
	}
	// Validate etymology links
	for _, r := range roots {
		if r.ParentRootID != 0 && !rootSet[r.ParentRootID] {
			return fmt.Errorf("root %q parent_root_id %d not found", r.RootStr, r.ParentRootID)
		}
	}
	wordIDSet := make(map[uint32]string, len(words))
	normLangSet := make(map[string]string, len(words))
	for _, w := range words {
		if !rootSet[w.RootID] {
			return fmt.Errorf("word %q references unknown root_id %d", w.Word, w.RootID)
		}
		if _, ok := ParseLang(w.Lang); !ok {
			return fmt.Errorf("word %q has unknown lang %q", w.Word, w.Lang)
		}
		// Enforce word_id uniqueness — duplicate IDs cause silent data corruption.
		if prev, exists := wordIDSet[w.WordID]; exists {
			return fmt.Errorf("duplicate word_id %d: %q and %q", w.WordID, prev, w.Word)
		}
		wordIDSet[w.WordID] = w.Word
		// Enforce (norm, lang) uniqueness — same normalized form in same language
		// would make one word unreachable via LookupWord.
		normKey := strings.ToLower(strings.TrimSpace(w.Norm)) + "_" + w.Lang
		if prev, exists := normLangSet[normKey]; exists {
			return fmt.Errorf("duplicate normalized form %q in lang %s: %q and %q",
				w.Norm, w.Lang, prev, w.Word)
		}
		normLangSet[normKey] = w.Word
	}
	return nil
}

func langBitFromStr(lang string) uint32 {
	id, _ := ParseLang(lang)
	return LangBit(id)
}

// normalize lowercases an ASCII string (diacritics already stripped in CSV norm column).
func normalize(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

// buildData serializes all sections into a byte slice for checksumming.
func buildData(roots []RootRecord, words []WordRecord, heap []byte) []byte {
	var buf []byte
	appendU32 := func(v uint32) {
		var b [4]byte
		ByteOrder.PutUint32(b[:], v)
		buf = append(buf, b[:]...)
	}
	for _, r := range roots {
		appendU32(r.RootID)
		appendU32(r.WordCount)
		appendU32(r.FirstWordIdx)
		appendU32(r.NameOffset)
		appendU32(r.LangCoverage)
		appendU32(r.ParentRootID)
		appendU32(r.OriginOffset)
		appendU32(r.MeaningOffset)
	}
	for _, w := range words {
		appendU32(w.WordID)
		appendU32(w.RootID)
		appendU32(w.Lang)
		appendU32(w.Sentiment)
		appendU32(w.WordOffset)
		appendU32(w.NormOffset)
		appendU32(w.FreqRank)
		appendU32(w.Flags)
	}
	buf = append(buf, heap...)
	return buf
}

// stringHeap accumulates null-terminated strings and returns byte offsets.
type stringHeap struct {
	data    []byte
	offsets map[string]uint32
}

func (h *stringHeap) add(s string) uint32 {
	if h.offsets == nil {
		h.offsets = make(map[string]uint32)
	}
	if off, ok := h.offsets[s]; ok {
		return off
	}
	off := uint32(len(h.data))
	h.offsets[s] = off
	h.data = append(h.data, []byte(s)...)
	h.data = append(h.data, 0) // null terminator
	return off
}

func (h *stringHeap) bytes() []byte {
	return h.data
}
