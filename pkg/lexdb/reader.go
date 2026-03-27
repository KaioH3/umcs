package lexdb

import (
	"encoding/binary"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"

	"github.com/kak/umcs/pkg/morpheme"
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

// Load reads a .umcs file into memory and builds the lookup index.
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
	if version != 1 && version != Version {
		return nil, fmt.Errorf("unsupported version %d (supported: 1, %d)", version, Version)
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
	expectedEnd := heapOffset + heapSize
	if int(expectedEnd) > len(data) {
		return nil, fmt.Errorf("corrupt file: heap extends past end of file")
	}
	got := fnv1a32(data[dataStart:expectedEnd])
	if got != checksum {
		return nil, fmt.Errorf("checksum mismatch: got 0x%08X, want 0x%08X", got, checksum)
	}

	// Choose record sizes based on version for backward-compatible parsing.
	rootRecSize := uint64(RootRecordSize)
	wordRecSize := uint64(WordRecordSize)
	if version == 1 {
		rootRecSize = RootRecordSizeV1
		wordRecSize = WordRecordSizeV1
	}

	// Validate table bounds before parsing — guards against corrupt offsets.
	rootTableEnd := uint64(rootTableOffset) + uint64(rootCount)*rootRecSize
	wordTableEnd := uint64(wordTableOffset) + uint64(wordCount)*wordRecSize
	fileLen := uint64(len(data))
	if rootTableEnd > fileLen {
		return nil, fmt.Errorf("corrupt file: root table extends past end of file")
	}
	if wordTableEnd > fileLen {
		return nil, fmt.Errorf("corrupt file: word table extends past end of file")
	}

	// Parse root table
	roots := make([]RootRecord, rootCount)
	roff := int(rootTableOffset)
	for i := range roots {
		r := RootRecord{
			RootID:       ByteOrder.Uint32(data[roff:]),
			WordCount:    ByteOrder.Uint32(data[roff+4:]),
			FirstWordIdx: ByteOrder.Uint32(data[roff+8:]),
			NameOffset:   ByteOrder.Uint32(data[roff+12:]),
			LangCoverage: ByteOrder.Uint32(data[roff+16:]),
			ParentRootID: ByteOrder.Uint32(data[roff+20:]),
			OriginOffset: ByteOrder.Uint32(data[roff+24:]),
			MeaningOffset: ByteOrder.Uint32(data[roff+28:]),
		}
		if version >= 2 {
			r.HypernymRootID = ByteOrder.Uint32(data[roff+32:])
			r.AntonymRootID  = ByteOrder.Uint32(data[roff+36:])
			r.SynonymRootID  = ByteOrder.Uint32(data[roff+40:])
		}
		roots[i] = r
		roff += int(rootRecSize)
	}

	// Parse word table
	words := make([]WordRecord, wordCount)
	woff := int(wordTableOffset)
	for i := range words {
		w := WordRecord{
			WordID:     ByteOrder.Uint32(data[woff:]),
			RootID:     ByteOrder.Uint32(data[woff+4:]),
			Lang:       ByteOrder.Uint32(data[woff+8:]),
			Sentiment:  ByteOrder.Uint32(data[woff+12:]),
			WordOffset: ByteOrder.Uint32(data[woff+16:]),
			NormOffset: ByteOrder.Uint32(data[woff+20:]),
			FreqRank:   ByteOrder.Uint32(data[woff+24:]),
			Flags:      ByteOrder.Uint32(data[woff+28:]),
		}
		if version >= 2 {
			w.PronOffset = ByteOrder.Uint32(data[woff+32:])
		}
		words[i] = w
		woff += int(wordRecSize)
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
//
// Collision policy: keep-first (lowest word_id wins for any-language lookups).
// Words are sorted by word_id during Build, so the first occurrence encountered
// here is always the one with the lowest word_id.
//
// Lang-specific keys use the format "norm_LANGID" (e.g. "mais_0" for PT) to
// allow LookupWordInLang to disambiguate homographs across languages.
func (l *Lexicon) buildIndex() {
	l.wordIndex = make(map[string]int, len(l.Words)*2)
	for i, w := range l.Words {
		norm := l.str(w.NormOffset)
		// Any-language lookup: keep the first (lowest word_id) match only.
		if _, exists := l.wordIndex[norm]; !exists {
			l.wordIndex[norm] = i
		}
		// Lang-specific key: always stored (one entry per norm+lang pair).
		langKey := fmt.Sprintf("%s_%d", norm, w.Lang)
		l.wordIndex[langKey] = i
		// Also index by lowercased surface form if it differs from norm.
		surface := strings.ToLower(l.str(w.WordOffset))
		if surface != norm {
			if _, exists := l.wordIndex[surface]; !exists {
				l.wordIndex[surface] = i
			}
		}
	}
}

// str reads a null-terminated string from the heap at the given offset.
// Returns "" if offset is out of bounds (corrupted file guard).
func (l *Lexicon) str(offset uint32) string {
	start := int(offset)
	if start >= len(l.Heap) {
		return ""
	}
	end := start
	for end < len(l.Heap) && l.Heap[end] != 0 {
		end++
	}
	return string(l.Heap[start:end])
}

// Normalize is the canonical diacritic-stripping function for the UMCS lexicon.
//
// It lowercases, trims, and maps accented characters to their ASCII base form,
// preserving the semantic identity of a word across orthographic variants
// (e.g. "café" = "cafe", "naïve" = "naive", "über" = "uber").
//
// Non-Latin scripts (Arabic, Cyrillic, CJK, Hangul, Hebrew) are preserved
// unchanged — only Latin-script diacritics are stripped. CJK ideograms are
// each their own morpheme and must never be collapsed.
//
// This function is used by LookupWord and by the Build pipeline, so changing
// it changes the norm stored in .umcs files — rebuild required after any change.
func Normalize(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(strings.TrimSpace(s)) {
		switch r {
		// ── Vowel a ────────────────────────────────────────────────────────────
		case 'á', 'à', 'â', 'ã', 'ä', 'å', 'ā', 'ă', 'ą':
			b.WriteByte('a')
		// ── Vowel e ────────────────────────────────────────────────────────────
		case 'é', 'è', 'ê', 'ë', 'ē', 'ě', 'ę':
			b.WriteByte('e')
		// ── Vowel i ────────────────────────────────────────────────────────────
		case 'í', 'ì', 'î', 'ï', 'ī', 'ĭ', 'į':
			b.WriteByte('i')
		// ── Vowel o ────────────────────────────────────────────────────────────
		case 'ó', 'ò', 'ô', 'õ', 'ö', 'ō', 'ő', 'ø':
			b.WriteByte('o')
		// ── Vowel u ────────────────────────────────────────────────────────────
		case 'ú', 'ù', 'û', 'ü', 'ū', 'ű', 'ů':
			b.WriteByte('u')
		// ── Vowel y ────────────────────────────────────────────────────────────
		case 'ý', 'ÿ':
			b.WriteByte('y')
		// ── Consonant c ────────────────────────────────────────────────────────
		case 'ç', 'ć', 'č':
			b.WriteByte('c')
		// ── Consonant n ────────────────────────────────────────────────────────
		case 'ñ', 'ń', 'ň':
			b.WriteByte('n')
		// ── Consonant s ────────────────────────────────────────────────────────
		case 'š', 'ś', 'ş':
			b.WriteByte('s')
		// ── Consonant z ────────────────────────────────────────────────────────
		case 'ž', 'ź', 'ż':
			b.WriteByte('z')
		// ── Consonant d ────────────────────────────────────────────────────────
		case 'đ', 'ð':
			b.WriteByte('d')
		// ── Consonant t ────────────────────────────────────────────────────────
		case 'ț', 'ţ': // U+021B (comma below) and U+0163 (cedilla) — both Romanian
			b.WriteByte('t')
		// ── Consonant l ────────────────────────────────────────────────────────
		case 'ł':
			b.WriteByte('l')
		// ── Consonant r ────────────────────────────────────────────────────────
		case 'ř':
			b.WriteByte('r')
		// ── Consonant g ────────────────────────────────────────────────────────
		case 'ğ':
			b.WriteByte('g')
		// ── Multi-char expansions ──────────────────────────────────────────────
		case 'ß':
			b.WriteString("ss")
		case 'þ':
			b.WriteString("th")
		case 'æ':
			b.WriteString("ae")
		case 'œ':
			b.WriteString("oe")
		default:
			// Preserve: letters (non-Latin scripts, CJK, Cyrillic, Arabic, etc.),
			// digits, and hyphens. Strip punctuation and whitespace.
			if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' {
				b.WriteRune(r)
			}
		}
	}
	return b.String()
}

// LookupWord finds a word record by surface form (normalized internally).
// When multiple languages share the same normalized form, the word with the
// lowest word_id is returned. Use LookupWordInLang for language-specific lookup.
// Returns nil if not found.
func (l *Lexicon) LookupWord(word string) *WordRecord {
	norm := Normalize(word)
	idx, ok := l.wordIndex[norm]
	if !ok {
		return nil
	}
	return &l.Words[idx]
}

// LookupWordInLang finds a word by surface form in a specific language.
// This disambiguates homographs across languages (e.g. "mais" PT vs "maïs" FR).
// Falls back to LookupWord if the language is unknown or no lang-specific match exists.
func (l *Lexicon) LookupWordInLang(word, lang string) *WordRecord {
	langID, ok := ParseLang(strings.ToUpper(lang))
	if !ok {
		return l.LookupWord(word)
	}
	norm := Normalize(word)
	key := fmt.Sprintf("%s_%d", norm, langID)
	if idx, ok := l.wordIndex[key]; ok {
		return &l.Words[idx]
	}
	return l.LookupWord(word) // fallback: any language
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
	rootID := morpheme.RootOf(wordID)
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

// WordPron returns the IPA pronunciation string for a word record.
// Returns "" if no IPA was annotated (PronOffset == 0).
// IPA strings use standard Unicode IPA characters (e.g. "/ˈtɛr.ɪ.bəl/").
func (l *Lexicon) WordPron(w *WordRecord) string {
	if w.PronOffset == 0 {
		return ""
	}
	return l.str(w.PronOffset)
}

// Hypernym returns the hypernym root (is-a parent) of the given root, or nil.
// Example: "dog" root → hypernym root "animal".
func (l *Lexicon) Hypernym(r *RootRecord) *RootRecord {
	if r.HypernymRootID == 0 {
		return nil
	}
	return l.LookupRoot(r.HypernymRootID)
}

// Antonym returns the antonym root of the given root, or nil.
// Example: "good" root → antonym root "bad".
func (l *Lexicon) Antonym(r *RootRecord) *RootRecord {
	if r.AntonymRootID == 0 {
		return nil
	}
	return l.LookupRoot(r.AntonymRootID)
}

// Synonym returns the nearest synonym root, or nil.
// Example: "anger" root → synonym root "rage".
func (l *Lexicon) Synonym(r *RootRecord) *RootRecord {
	if r.SynonymRootID == 0 {
		return nil
	}
	return l.LookupRoot(r.SynonymRootID)
}

// RootCanonicalWord returns the first WordRecord for rootID by index order.
// This gives the "canonical" representative of a root family for feature
// extraction (e.g. looking up antonym or hypernym polarity). Returns nil if no
// words exist for that root.
func (l *Lexicon) RootCanonicalWord(rootID uint32) *WordRecord {
	root := l.LookupRoot(rootID)
	if root == nil || root.WordCount == 0 {
		return nil
	}
	idx := int(root.FirstWordIdx)
	if idx >= len(l.Words) {
		return nil
	}
	return &l.Words[idx]
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
