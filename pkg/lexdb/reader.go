package lexdb

import (
	"encoding/binary"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

// Lexicon holds the fully loaded lexicon in memory.
// After Load, all lookups are O(1) (word) or O(log N) (root by ID).
type Lexicon struct {
	Roots     []RootRecord
	Words     []WordRecord
	Heap      []byte   // raw string heap
	wordIndex map[string]int // normalized form → index in Words
	Stats     LexiconStats
}

// LexiconStats contains summary information about a loaded lexicon.
type LexiconStats struct {
	RootCount int
	WordCount int
	HeapSize  int
	FileSize  int64
	LangFlags uint32
	Checksum  uint32
}

// Load reads a .lsdb file into memory and builds the lookup index.
// The entire file is read at once; for very large lexicons (>100MB) use LoadMmap.
func Load(path string) (*Lexicon, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	// Read entire file
	data := make([]byte, fi.Size())
	if _, err = f.Read(data); err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	return parse(data, fi.Size())
}

func parse(data []byte, fileSize int64) (*Lexicon, error) {
	if len(data) < HeaderSize {
		return nil, fmt.Errorf("file too small (%d bytes)", len(data))
	}

	// Parse header
	off := 0
	readU32 := func() uint32 {
		v := ByteOrder.Uint32(data[off:])
		off += 4
		return v
	}

	magic := readU32()
	if magic != Magic {
		return nil, fmt.Errorf("invalid magic 0x%08X (want 0x%08X)", magic, Magic)
	}
	version := readU32()
	if version != Version {
		return nil, fmt.Errorf("unsupported version %d", version)
	}
	rootCount := readU32()
	wordCount := readU32()
	heapSize := readU32()
	rootTableOffset := readU32()
	wordTableOffset := readU32()
	heapOffset := readU32()
	langFlags := readU32()
	checksum := readU32()

	// Validate checksum
	dataStart := uint32(HeaderSize)
	dataLen := uint32(len(data)) - dataStart
	_ = dataLen
	expectedEnd := heapOffset + heapSize
	if int(expectedEnd) > len(data) {
		return nil, fmt.Errorf("corrupt file: heap extends past end of file")
	}
	got := fnv1a32(data[dataStart:expectedEnd])
	if got != checksum {
		return nil, fmt.Errorf("checksum mismatch: got 0x%08X, want 0x%08X", got, checksum)
	}

	// Parse root table
	roots := make([]RootRecord, rootCount)
	roff := int(rootTableOffset)
	for i := range roots {
		roots[i] = RootRecord{
			RootID:        ByteOrder.Uint32(data[roff:]),
			WordCount:     ByteOrder.Uint32(data[roff+4:]),
			FirstWordIdx:  ByteOrder.Uint32(data[roff+8:]),
			NameOffset:    ByteOrder.Uint32(data[roff+12:]),
			LangCoverage:  ByteOrder.Uint32(data[roff+16:]),
			ParentRootID:  ByteOrder.Uint32(data[roff+20:]),
			OriginOffset:  ByteOrder.Uint32(data[roff+24:]),
			MeaningOffset: ByteOrder.Uint32(data[roff+28:]),
		}
		roff += RootRecordSize
	}

	// Parse word table
	words := make([]WordRecord, wordCount)
	woff := int(wordTableOffset)
	for i := range words {
		words[i] = WordRecord{
			WordID:     ByteOrder.Uint32(data[woff:]),
			RootID:     ByteOrder.Uint32(data[woff+4:]),
			Lang:       ByteOrder.Uint32(data[woff+8:]),
			Sentiment:  ByteOrder.Uint32(data[woff+12:]),
			WordOffset: ByteOrder.Uint32(data[woff+16:]),
			NormOffset: ByteOrder.Uint32(data[woff+20:]),
			FreqRank:   ByteOrder.Uint32(data[woff+24:]),
			Flags:      ByteOrder.Uint32(data[woff+28:]),
		}
		woff += WordRecordSize
	}

	heap := data[heapOffset : heapOffset+heapSize]

	lex := &Lexicon{
		Roots: roots,
		Words: words,
		Heap:  heap,
		Stats: LexiconStats{
			RootCount: int(rootCount),
			WordCount: int(wordCount),
			HeapSize:  int(heapSize),
			FileSize:  fileSize,
			LangFlags: langFlags,
			Checksum:  checksum,
		},
	}
	lex.buildIndex()
	return lex, nil
}

// buildIndex constructs the normalized-form → word index for O(1) lookups.
func (l *Lexicon) buildIndex() {
	l.wordIndex = make(map[string]int, len(l.Words))
	for i, w := range l.Words {
		norm := l.str(w.NormOffset)
		l.wordIndex[norm] = i
		// Also index by surface form (lowercased) if different from norm
		surface := strings.ToLower(l.str(w.WordOffset))
		if surface != norm {
			if _, exists := l.wordIndex[surface]; !exists {
				l.wordIndex[surface] = i
			}
		}
	}
}

// str reads a null-terminated string from the heap at the given offset.
func (l *Lexicon) str(offset uint32) string {
	start := int(offset)
	end := start
	for end < len(l.Heap) && l.Heap[end] != 0 {
		end++
	}
	return string(l.Heap[start:end])
}

// Normalize strips diacritics, lowercases, and trims a word for lookup.
func Normalize(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(strings.TrimSpace(s)) {
		switch r {
		case 'á', 'à', 'â', 'ã', 'ä', 'å':
			b.WriteByte('a')
		case 'é', 'è', 'ê', 'ë':
			b.WriteByte('e')
		case 'í', 'ì', 'î', 'ï':
			b.WriteByte('i')
		case 'ó', 'ò', 'ô', 'õ', 'ö':
			b.WriteByte('o')
		case 'ú', 'ù', 'û', 'ü':
			b.WriteByte('u')
		case 'ç':
			b.WriteByte('c')
		case 'ñ':
			b.WriteByte('n')
		case 'ß':
			b.WriteString("ss")
		default:
			if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' {
				b.WriteRune(r)
			}
		}
	}
	return b.String()
}

// LookupWord finds a word record by surface form (normalized internally).
// Returns nil if not found.
func (l *Lexicon) LookupWord(word string) *WordRecord {
	norm := Normalize(word)
	idx, ok := l.wordIndex[norm]
	if !ok {
		return nil
	}
	return &l.Words[idx]
}

// LookupRoot finds a root record by root_id using binary search (O(log N)).
func (l *Lexicon) LookupRoot(rootID uint32) *RootRecord {
	idx := sort.Search(len(l.Roots), func(i int) bool {
		return l.Roots[i].RootID >= rootID
	})
	if idx < len(l.Roots) && l.Roots[idx].RootID == rootID {
		return &l.Roots[idx]
	}
	return nil
}

// Cognates returns all word records sharing the same root_id as the given word.
// Words are sorted by word_id in the binary, so cognates are contiguous.
func (l *Lexicon) Cognates(wordID uint32) []WordRecord {
	rootID := wordID >> 12
	root := l.LookupRoot(rootID)
	if root == nil {
		return nil
	}
	start := int(root.FirstWordIdx)
	end := start + int(root.WordCount)
	if start >= len(l.Words) || end > len(l.Words) {
		return nil
	}
	return l.Words[start:end]
}

// WordStr returns the surface form of a word record.
func (l *Lexicon) WordStr(w *WordRecord) string {
	return l.str(w.WordOffset)
}

// RootStr returns the string form of a root record.
func (l *Lexicon) RootStr(r *RootRecord) string {
	return l.str(r.NameOffset)
}

// RootOrigin returns the origin language of a root (e.g. "LATIN").
func (l *Lexicon) RootOrigin(r *RootRecord) string {
	return l.str(r.OriginOffset)
}

// RootMeaning returns the English gloss of a root.
func (l *Lexicon) RootMeaning(r *RootRecord) string {
	return l.str(r.MeaningOffset)
}

// EtymologyChain traces the etymology from a root back to its ancestor.
// Returns the chain from root → parent → grandparent → ... (proto-language).
func (l *Lexicon) EtymologyChain(rootID uint32) []RootRecord {
	var chain []RootRecord
	seen := make(map[uint32]bool)
	for rootID != 0 {
		if seen[rootID] {
			break // cycle guard
		}
		seen[rootID] = true
		r := l.LookupRoot(rootID)
		if r == nil {
			break
		}
		chain = append(chain, *r)
		rootID = r.ParentRootID
	}
	return chain
}

// LangCoverage returns human-readable language names covered by a root.
func (l *Lexicon) LangCoverage(coverage uint32) []string {
	var langs []string
	for i := uint32(0); i < 11; i++ {
		if coverage&(1<<i) != 0 {
			langs = append(langs, LangName(i))
		}
	}
	return langs
}

// ensure encoding/binary is used (for format consistency in tests)
var _ = binary.LittleEndian
