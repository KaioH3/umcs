// Package lexdb defines the .umcs binary format and provides reader/writer.
//
// File layout:
//   [0..63]   Header (64 bytes)
//   [H..]     Root table  (RootRecordSize bytes per root, sorted by root_id)
//   [W..]     Word table  (WordRecordSize bytes per word, sorted by word_id)
//   [S..]     String heap (null-terminated UTF-8 strings, byte-addressed)
//
// All integers are little-endian uint32 (4 bytes).
//
// # Format versions
//
//   Version 1 — RootRecord 32 bytes (8 fields), WordRecord 32 bytes (8 fields)
//   Version 2 — RootRecord 44 bytes (+HypernymRootID, AntonymRootID, SynonymRootID)
//               WordRecord 36 bytes (+PronOffset for IPA string)
//               Backward-compatible reader: v1 files load with new fields zeroed.
package lexdb

import "encoding/binary"

var ByteOrder = binary.LittleEndian

const (
	Magic   uint32 = 0x4C534442 // "LSDB"
	Version uint32 = 2

	// HeaderSize is always 64 bytes across all versions.
	HeaderSize = 64

	// V1 record sizes (for backward-compatible reading).
	RootRecordSizeV1 = 32 // 8 × uint32
	WordRecordSizeV1 = 32 // 8 × uint32

	// V2 record sizes (current format).
	// RootRecord gains 3 semantic-relation fields (+12 bytes).
	// WordRecord gains PronOffset for IPA pronunciation (+4 bytes).
	RootRecordSize = 44 // 11 × uint32
	WordRecordSize = 36 // 9 × uint32

	// Lang bitmask flags (bits in Header.LangFlags and Root.LangCoverage).
	// One bit per language; supports up to 32 languages in a uint32.
	LangPT uint32 = 1 << 0  // Portuguese
	LangEN uint32 = 1 << 1  // English
	LangES uint32 = 1 << 2  // Spanish
	LangIT uint32 = 1 << 3  // Italian
	LangDE uint32 = 1 << 4  // German
	LangFR uint32 = 1 << 5  // French
	LangNL uint32 = 1 << 6  // Dutch
	LangAR uint32 = 1 << 7  // Arabic (Semitic; RTL; trilateral root system)
	LangZH uint32 = 1 << 8  // Mandarin Chinese (CJK; logographic)
	LangJA uint32 = 1 << 9  // Japanese (CJK; mixed kana/kanji)
	LangRU uint32 = 1 << 10 // Russian (Slavic; Cyrillic script)
	LangKO uint32 = 1 << 11 // Korean (Hangul; Altaic family)
	LangTG uint32 = 1 << 12 // Tupi-Guarani (South American indigenous; Latin script)
	LangHI uint32 = 1 << 13 // Hindi (Indo-Aryan; Devanagari script)
	LangBN uint32 = 1 << 14 // Bengali (Indo-Aryan; Bengali script)
	LangID uint32 = 1 << 15 // Indonesian/Malay (Austronesian; Latin script)
	LangTR uint32 = 1 << 16 // Turkish (Turkic; Latin script)
	LangFA uint32 = 1 << 17 // Persian/Farsi (Iranian; Arabic script)
	LangSW uint32 = 1 << 18 // Swahili (Bantu; Latin script)
	LangUK uint32 = 1 << 19 // Ukrainian (Slavic; Cyrillic script)
	LangPL uint32 = 1 << 20 // Polish (Slavic; Latin script)
	LangSA uint32 = 1 << 21 // Sanskrit (Indo-Aryan; Devanagari; classical ancestor)
	LangTA uint32 = 1 << 22 // Tamil (Dravidian; Tamil script)
	LangHE uint32 = 1 << 23 // Hebrew (Semitic; RTL; trilateral root system)
	// Bits 24..31: overflow languages (top 8 by corpus size from imported datasets).
	LangLA uint32 = 1 << 24 // Latin (classical; Romance ancestor)
	LangFI uint32 = 1 << 25 // Finnish (Uralic; agglutinative)
	LangDA uint32 = 1 << 26 // Danish (Germanic; Scandinavian)
	LangHU uint32 = 1 << 27 // Hungarian (Uralic; agglutinative)
	LangSV uint32 = 1 << 28 // Swedish (Germanic; Scandinavian)
	LangCA uint32 = 1 << 29 // Catalan (Romance; Iberian)
	LangRO uint32 = 1 << 30 // Romanian (Romance; Balkan)
	LangCS uint32 = 1 << 31 // Czech (Slavic; Latin script)

	// Word.Lang values (IDs, not bitmasks — stored in WordRecord.Lang).
	// IDs 0..31 have corresponding LangCoverage bitmask bits.
	// IDs 32+ work for lookup but don't get a LangCoverage bit.
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
	WordLangTG uint32 = 12
	WordLangHI uint32 = 13
	WordLangBN uint32 = 14
	WordLangID uint32 = 15
	WordLangTR uint32 = 16
	WordLangFA uint32 = 17
	WordLangSW uint32 = 18
	WordLangUK uint32 = 19
	WordLangPL uint32 = 20
	WordLangSA uint32 = 21
	WordLangTA uint32 = 22
	WordLangHE uint32 = 23
	WordLangLA uint32 = 24 // Latin
	WordLangFI uint32 = 25 // Finnish
	WordLangDA uint32 = 26 // Danish
	WordLangHU uint32 = 27 // Hungarian
	WordLangSV uint32 = 28 // Swedish
	WordLangCA uint32 = 29 // Catalan
	WordLangRO uint32 = 30 // Romanian
	WordLangCS uint32 = 31 // Czech
	WordLangGL uint32 = 32 // Galician
	WordLangMS uint32 = 33 // Malay
	WordLangSK uint32 = 34 // Slovak
	WordLangSL uint32 = 35 // Slovenian
	WordLangHR uint32 = 36 // Croatian
	WordLangBG uint32 = 37 // Bulgarian
	WordLangGU uint32 = 38 // Gujarati
	WordLangEL uint32 = 39 // Greek
	WordLangIS uint32 = 40 // Icelandic
	WordLangPA uint32 = 41 // Punjabi
	WordLangKN uint32 = 42 // Kannada
	WordLangNE uint32 = 43 // Nepali
	WordLangEU uint32 = 44 // Basque
	WordLangTH uint32 = 45 // Thai
	WordLangVI uint32 = 46 // Vietnamese
	WordLangGA uint32 = 47 // Irish
	WordLangMK uint32 = 48 // Macedonian
	WordLangTE uint32 = 49 // Telugu
	WordLangAF uint32 = 50 // Afrikaans
	WordLangUR uint32 = 51 // Urdu
	WordLangNO uint32 = 52 // Norwegian
	WordLangSQ uint32 = 53 // Albanian
	WordLangML uint32 = 54 // Malayalam
	WordLangMR uint32 = 55 // Marathi
	WordLangKA uint32 = 56 // Georgian
	WordLangTL uint32 = 57 // Tagalog
	WordLangHY uint32 = 58 // Armenian
	WordLangCY uint32 = 59 // Welsh
	WordLangMT uint32 = 60 // Maltese
	WordLangBS uint32 = 61 // Bosnian
	WordLangAM uint32 = 62 // Amharic
	WordLangET uint32 = 63 // Estonian
	WordLangSR uint32 = 64 // Serbian
	WordLangLV uint32 = 65 // Latvian
	WordLangLT uint32 = 66 // Lithuanian
	WordLangAZ uint32 = 67 // Azerbaijani
	WordLangLB uint32 = 68 // Luxembourgish
	WordLangSI uint32 = 69 // Sinhala

	// Word.Flags bitmask.
	//
	// Low byte (bits 7..0): lexical type flags
	WordFlagProper      uint32 = 1 << 0 // proper noun / named entity
	WordFlagArchaic     uint32 = 1 << 1 // obsolete / historical usage
	WordFlagColloquial  uint32 = 1 << 2 // colloquial / spoken register
	WordFlagDomain      uint32 = 1 << 3 // domain-specific term
	WordFlagFalseFriend uint32 = 1 << 4 // looks like cognate but has different meaning
	WordFlagLoanword    uint32 = 1 << 5 // borrowed from another language
	WordFlagAllomorph   uint32 = 1 << 6 // phonological variant of a morpheme
	WordFlagOnomatopeia uint32 = 1 << 7 // word sounds like its referent

	// Bits 11..8: REGISTER (4-bit enum) — formality/register of the word.
	// Encode as (value << 8) and extract with (flags >> 8) & 0xF.
	RegisterNeutral    uint32 = 0 << 8
	RegisterFormal     uint32 = 1 << 8 // written/official language
	RegisterInformal   uint32 = 2 << 8 // conversational
	RegisterSlang      uint32 = 3 << 8 // slang / street language
	RegisterVulgar     uint32 = 4 << 8 // vulgar / taboo
	RegisterArchaic    uint32 = 5 << 8 // historical / obsolete register
	RegisterPoetic     uint32 = 6 << 8 // literary / poetic
	RegisterTechnical  uint32 = 7 << 8 // domain-specific technical
	RegisterScientific uint32 = 8 << 8 // scientific nomenclature
	RegisterChild      uint32 = 9 << 8 // child-directed speech (early AoA)
	RegisterRegional   uint32 = 10 << 8
	RegisterMask       uint32 = 0xF << 8

	// Bits 15..12: ONTOLOGICAL category (4-bit enum) — what kind of thing the word refers to.
	// Encode as (value << 12) and extract with (flags >> 12) & 0xF.
	OntoNone       uint32 = 0 << 12
	OntoPerson     uint32 = 1 << 12 // human / agent
	OntoPlace      uint32 = 2 << 12 // location / space
	OntoArtifact   uint32 = 3 << 12 // man-made object
	OntoNatural    uint32 = 4 << 12 // natural object (animal, plant, mineral)
	OntoEvent      uint32 = 5 << 12 // action / occurrence
	OntoState      uint32 = 6 << 12 // state of affairs / condition
	OntoProperty   uint32 = 7 << 12 // quality / attribute
	OntoQuantity   uint32 = 8 << 12 // number / amount / measure
	OntoRelation   uint32 = 9 << 12 // relation / connection
	OntoTemporal   uint32 = 10 << 12
	OntoBiological uint32 = 11 << 12 // biological / body
	OntoSocial     uint32 = 12 << 12 // social / institution
	OntoAbstract   uint32 = 13 << 12 // abstract concept
	OntoMask       uint32 = 0xF << 12

	// Bits 19..16: POLYSEMY tier (4-bit count) — how many distinct senses the word has.
	// 0=unknown, 1=monosemous, 2=2 senses, … max 15.
	PolysemyMask uint32 = 0xF << 16

	// Bit 20: CULTURAL_SPECIFIC — no equivalent in most other languages (e.g. "saudade", "schadenfreude").
	CulturalSpecific uint32 = 1 << 20

	// Bits 31..21: PHONOLOGY — packed into Flags (see pkg/phon for constants and helpers).
	// bit  21      neologism
	// bit  22      irony_capable
	// bits 25..23  valency (Tesnière: 0=N/A, 1=intrans, 2=trans, 3=ditrans, 4=copular, 5=modal)
	// bits 27..26  stress (0=unknown, 1=final/oxytone, 2=penult/paroxytone, 3=antepenult/proparoxytone)
	// bits 31..28  syllable count (0=unknown, 1-15)
	// See pkg/phon for SetSyllables, SetStress, SetValency, etc.
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

// RootRecord is one 44-byte root entry in the root table (format v2).
//
// Etymology: ParentRootID links to the ancestral root (0 = no parent).
// Semantic relations follow WordNet conventions:
//   - HypernymRootID: is-a / generalization (dog → animal)
//   - AntonymRootID:  direct antonym of opposite polarity (good → bad)
//   - SynonymRootID:  nearest synonym root (anger ≈ rage)
//
// Example: "negat" → ParentRootID=0 (Latin proto-root)
//          HypernymRootID=0 (grammatical scope marker, no hypernym)
//          AntonymRootID=2  (affirm is its antonym)
type RootRecord struct {
	RootID         uint32 // unique root identifier
	WordCount      uint32 // number of word records with this root
	FirstWordIdx   uint32 // index in word table of first cognate
	NameOffset     uint32 // offset into string heap (root string, e.g. "negat")
	LangCoverage   uint32 // bitmask of languages covered
	ParentRootID   uint32 // etymology: ancestral root (0 = proto/no parent)
	OriginOffset   uint32 // offset into string heap (e.g. "LATIN")
	MeaningOffset  uint32 // offset into string heap (English gloss)
	HypernymRootID uint32 // WordNet-style is-a parent root (0 = none) [v2]
	AntonymRootID  uint32 // primary antonym root (0 = none) [v2]
	SynonymRootID  uint32 // nearest synonym root (0 = none) [v2]
}

// WordRecord is one 36-byte word entry in the word table (format v2).
//
// Phonology is packed into the high bits of Flags (see pkg/phon).
// PronOffset points to an IPA pronunciation string in the heap (0 = not annotated).
// IPA strings use standard Unicode IPA characters (e.g. "/ˈtɛr.ɪ.bəl/").
type WordRecord struct {
	WordID     uint32 // packed (root_id<<12)|variant — primary LLM token
	RootID     uint32 // root family (for cross-linguistic embedding sharing)
	Lang       uint32 // WordLangPT/EN/ES/IT/DE (see format.go)
	Sentiment  uint32 // packed bitmask (see pkg/sentiment)
	WordOffset uint32 // offset into string heap (surface form, e.g. "negativo")
	NormOffset uint32 // offset into string heap (ASCII-normalized form)
	FreqRank   uint32 // corpus frequency rank (0=unknown)
	Flags      uint32 // WordFlag* | register | ontological | polysemy | phonology
	PronOffset uint32 // IPA pronunciation offset in heap (0 = none) [v2]
}

// langNames maps lang IDs (array index) to ISO 639-1 codes.
// IDs 0–31 have LangCoverage bitmask bits; 32+ are lookup-only.
var langNames = []string{
	"PT", "EN", "ES", "IT", "DE", "FR", "NL", "AR", // 0–7
	"ZH", "JA", "RU", "KO", "TG", "HI", "BN", "ID", // 8–15
	"TR", "FA", "SW", "UK", "PL", "SA", "TA", "HE", // 16–23
	"LA", "FI", "DA", "HU", "SV", "CA", "RO", "CS", // 24–31
	"GL", "MS", "SK", "SL", "HR", "BG", "GU", "EL", // 32–39
	"IS", "PA", "KN", "NE", "EU", "TH", "VI", "GA", // 40–47
	"MK", "TE", "AF", "UR", "NO", "SQ", "ML", "MR", // 48–55
	"KA", "TL", "HY", "CY", "MT", "BS", "AM", "ET", // 56–63
	"SR", "LV", "LT", "AZ", "LB", "SI",             // 64–69
}

// langIndex maps ISO 639-1 codes to lang IDs (built once from langNames).
var langIndex map[string]uint32

func init() {
	langIndex = make(map[string]uint32, len(langNames))
	for i, name := range langNames {
		langIndex[name] = uint32(i)
	}
}

// LangName maps a lang ID to its ISO 639-1/639-2 code.
func LangName(lang uint32) string {
	if int(lang) < len(langNames) {
		return langNames[lang]
	}
	return "??"
}

// ParseLang converts a language code to its lang ID.
func ParseLang(s string) (uint32, bool) {
	id, ok := langIndex[s]
	return id, ok
}

// LangCount returns the total number of supported languages.
func LangCount() int {
	return len(langNames)
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
