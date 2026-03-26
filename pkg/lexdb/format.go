// Package lexdb defines the .lsdb binary format and provides reader/writer.
//
// File layout:
//   [0..63]   Header (64 bytes)
//   [H..]     Root table  (RootRecordSize bytes per root, sorted by root_id)
//   [W..]     Word table  (WordRecordSize bytes per word, sorted by word_id)
//   [S..]     String heap (null-terminated UTF-8 strings, byte-addressed)
//
// All integers are little-endian uint32 (4 bytes).
package lexdb

import "encoding/binary"

var ByteOrder = binary.LittleEndian

const (
	Magic   uint32 = 0x4C534442 // "LSDB"
	Version uint32 = 1

	HeaderSize     = 64
	RootRecordSize = 32 // extended with etymology fields
	WordRecordSize = 32

	// Lang bitmask flags (bits in Header.LangFlags and Root.LangCoverage)
	LangPT uint32 = 1 << 0
	LangEN uint32 = 1 << 1
	LangES uint32 = 1 << 2
	LangIT uint32 = 1 << 3
	LangDE uint32 = 1 << 4
	LangFR uint32 = 1 << 5
	LangNL uint32 = 1 << 6
	LangAR uint32 = 1 << 7
	LangZH uint32 = 1 << 8
	LangJA uint32 = 1 << 9
	LangRU uint32 = 1 << 10
	LangKO uint32 = 1 << 11
	// bits 12..31 reserved for future languages

	// Word.Lang values
	WordLangPT uint32 = 0
	WordLangEN uint32 = 1
	WordLangES uint32 = 2
	WordLangIT uint32 = 3
	WordLangDE uint32 = 4
	WordLangFR uint32 = 5
	WordLangNL uint32 = 6
	WordLangAR uint32 = 7
	WordLangZH uint32 = 8
	WordLangJA uint32 = 9
	WordLangRU uint32 = 10
	WordLangKO uint32 = 11

	// Word.Flags bitmask
	WordFlagProper      uint32 = 1 << 0
	WordFlagArchaic     uint32 = 1 << 1
	WordFlagColloquial  uint32 = 1 << 2
	WordFlagDomain      uint32 = 1 << 3
	WordFlagFalseFriend uint32 = 1 << 4 // looks like a cognate but has different meaning
	WordFlagLoanword    uint32 = 1 << 5 // borrowed from another language
	WordFlagAllomorph   uint32 = 1 << 6 // phonological variant of a morpheme
	WordFlagOnomatopeia uint32 = 1 << 7 // word that sounds like what it means
)

// Header is the 64-byte file header.
type Header struct {
	Magic           uint32 // 0x4C534442
	Version         uint32
	RootCount       uint32
	WordCount       uint32
	StringHeapSize  uint32
	RootTableOffset uint32 // byte offset from file start
	WordTableOffset uint32
	HeapOffset      uint32
	LangFlags       uint32 // languages present
	Checksum        uint32 // FNV-1a of all data after header
	_               [6]uint32 // reserved, 24 bytes
}

// RootRecord is one 32-byte root entry in the root table.
// Etymology: ParentRootID links to the ancestral root (0 = no parent).
// Example: Latin "negare" → root_id=1 is parent of "negar" (PT), "negate" (EN).
type RootRecord struct {
	RootID       uint32
	WordCount    uint32 // number of word records with this root
	FirstWordIdx uint32 // index in word table of first cognate
	NameOffset   uint32 // offset into string heap (the root string, e.g. "negat")
	LangCoverage uint32 // bitmask of languages covered
	ParentRootID uint32 // etymology: ancestral root (0 = proto/no parent)
	OriginOffset uint32 // offset into string heap (origin lang/proto, e.g. "LATIN")
	MeaningOffset uint32 // offset into string heap (English gloss, e.g. "to deny")
}

// WordRecord is one 32-byte word entry in the word table.
type WordRecord struct {
	WordID     uint32
	RootID     uint32
	Lang       uint32    // WordLangPT/EN/ES/IT/DE
	Sentiment  uint32    // packed bitmask (see pkg/sentiment)
	WordOffset uint32    // offset into string heap (actual word, e.g. "negativo")
	NormOffset uint32    // offset into string heap (ASCII-normalized, e.g. "negativo")
	FreqRank   uint32    // corpus frequency rank (0=unknown)
	Flags      uint32    // WordFlag* bitmask
}

// LangName maps a lang ID to its ISO 639-1 code.
func LangName(lang uint32) string {
	names := []string{"PT", "EN", "ES", "IT", "DE", "FR", "NL", "AR", "ZH", "JA", "RU", "KO"}
	if int(lang) < len(names) {
		return names[lang]
	}
	return "??"
}

// ParseLang converts a language string to its lang ID.
func ParseLang(s string) (uint32, bool) {
	langs := map[string]uint32{
		"PT": WordLangPT,
		"EN": WordLangEN,
		"ES": WordLangES,
		"IT": WordLangIT,
		"DE": WordLangDE,
		"FR": WordLangFR,
		"NL": WordLangNL,
		"AR": WordLangAR,
		"ZH": WordLangZH,
		"JA": WordLangJA,
		"RU": WordLangRU,
		"KO": WordLangKO,
	}
	id, ok := langs[s]
	return id, ok
}

// LangBit converts a lang ID to its bitmask flag.
// Returns 0 for unknown lang IDs (> 31) to prevent undefined bit-shift behavior.
func LangBit(lang uint32) uint32 {
	if lang > 31 {
		return 0
	}
	return 1 << lang
}

// fnv1a32 computes a FNV-1a checksum over b.
func fnv1a32(b []byte) uint32 {
	h := uint32(2166136261)
	for _, c := range b {
		h ^= uint32(c)
		h *= 16777619
	}
	return h
}
