package lexdb_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
)

func buildTestLexicon(t *testing.T) (*lexdb.Lexicon, string) {
	t.Helper()
	dir := t.TempDir()
	outPath := filepath.Join(dir, "test.lsdb")

	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "to deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
		{RootID: 10, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear", ParentRootID: 0},
		{RootID: 32, RootStr: "terrib", Origin: "LATIN", MeaningEN: "terrible", ParentRootID: 10},
	}

	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: 0x00120180},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: 0x00120180},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "negativo", Lang: "ES", Norm: "negativo", Sentiment: 0x00120180},
		{WordID: 8193, RootID: 2, Variant: 1, Word: "bom", Lang: "PT", Norm: "bom", Sentiment: 0x00130140},
		{WordID: 8194, RootID: 2, Variant: 2, Word: "good", Lang: "EN", Norm: "good", Sentiment: 0x00130140},
		{WordID: 40961, RootID: 10, Variant: 1, Word: "terror", Lang: "EN", Norm: "terror", Sentiment: 0x00440180},
		{WordID: 131073, RootID: 32, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: 0x00330180},
	}

	stats, err := lexdb.Build(roots, words, outPath)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if stats.RootCount != 4 {
		t.Fatalf("want 4 roots, got %d", stats.RootCount)
	}
	if stats.WordCount != 7 {
		t.Fatalf("want 7 words, got %d", stats.WordCount)
	}

	lex, err := lexdb.Load(outPath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return lex, outPath
}

func TestBuildAndLoad(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	if lex.Stats.RootCount != 4 {
		t.Fatalf("want 4 roots after load, got %d", lex.Stats.RootCount)
	}
	if lex.Stats.WordCount != 7 {
		t.Fatalf("want 7 words after load, got %d", lex.Stats.WordCount)
	}
}

func TestLookupWord(t *testing.T) {
	lex, _ := buildTestLexicon(t)

	w := lex.LookupWord("negative")
	if w == nil {
		t.Fatal("expected to find 'negative'")
	}
	if w.WordID != 4097 {
		t.Fatalf("want word_id=4097, got %d", w.WordID)
	}
	if w.RootID != 1 {
		t.Fatalf("want root_id=1, got %d", w.RootID)
	}

	// Test normalization
	w2 := lex.LookupWord("NEGATIVE")
	if w2 == nil {
		t.Fatal("case-insensitive lookup failed")
	}
	if w2.WordID != w.WordID {
		t.Fatal("case-insensitive lookup returned different word")
	}
}

func TestLookupDiacritic(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	// "negativo" stored as PT — look up with diacritic-free form
	w := lex.LookupWord("negativo")
	if w == nil {
		t.Fatal("expected to find 'negativo'")
	}
}

func TestCognates(t *testing.T) {
	lex, _ := buildTestLexicon(t)

	cognates := lex.Cognates(4097) // "negative"
	if len(cognates) != 3 {
		t.Fatalf("want 3 cognates for root 'negat', got %d", len(cognates))
	}

	// All cognates should share root_id=1
	for _, c := range cognates {
		if c.RootID != 1 {
			t.Fatalf("cognate %q has wrong root_id %d", lex.WordStr(&c), c.RootID)
		}
	}
}

func TestLookupRoot(t *testing.T) {
	lex, _ := buildTestLexicon(t)

	root := lex.LookupRoot(1)
	if root == nil {
		t.Fatal("root_id=1 not found")
	}
	if lex.RootStr(root) != "negat" {
		t.Fatalf("want 'negat', got %q", lex.RootStr(root))
	}
	if root.WordCount != 3 {
		t.Fatalf("want 3 words for root 'negat', got %d", root.WordCount)
	}

	// Non-existent
	if lex.LookupRoot(999) != nil {
		t.Fatal("root_id=999 should not exist")
	}
}

func TestEtymologyChain(t *testing.T) {
	lex, _ := buildTestLexicon(t)

	// "terribile" → root_id=32 (terrib) → parent root_id=10 (terr)
	chain := lex.EtymologyChain(32)
	if len(chain) != 2 {
		t.Fatalf("want 2 in chain, got %d", len(chain))
	}
	if chain[0].RootID != 32 {
		t.Fatalf("chain[0] should be root 32, got %d", chain[0].RootID)
	}
	if chain[1].RootID != 10 {
		t.Fatalf("chain[1] should be root 10 (parent), got %d", chain[1].RootID)
	}
}

func TestNotFound(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	if lex.LookupWord("xyzzy") != nil {
		t.Fatal("xyzzy should not be found")
	}
}

func TestNormalize(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Negativo", "negativo"},
		{"TERRÍVEL", "terrivel"},
		{"excelência", "excelencia"},
		{"não", "nao"},
		{"müde", "mude"},
		{"straße", "strasse"},
	}
	for _, c := range cases {
		got := lexdb.Normalize(c.in)
		if got != c.want {
			t.Errorf("Normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestNormalizeComprehensive verifies the expanded diacritic coverage including
// characters from Czech, Polish, Turkish, Romanian, Icelandic, and Nordic scripts.
func TestNormalizeComprehensive(t *testing.T) {
	cases := []struct{ in, want string }{
		// Previously missing cases
		{"naïve", "naive"},         // French ï
		{"über", "uber"},           // German ü (extra test)
		{"André", "andre"},         // French é + capital
		{"Ångström", "angstrom"},   // Swedish Å
		{"café", "cafe"},           // French é
		{"résumé", "resume"},       // French é
		{"Čeština", "cestina"},     // Czech č
		{"Ångström", "angstrom"},   // å → a
		{"Søren", "soren"},         // Danish ø
		{"Ołówek", "olowek"},       // Polish ł
		{"Şeker", "seker"},         // Turkish ş (new)
		{"Ţară", "tara"},           // Romanian ț (new)
		{"Ðanish", "danish"},       // Icelandic ð → d (new)
		{"þorn", "thorn"},          // Icelandic þ → th (new)
		{"Æsop", "aesop"},          // Latin æ → ae
		{"Œuvre", "oeuvre"},        // French œ → oe
		// CJK must be preserved (not stripped)
		{"愛", "愛"},
		{"悲", "悲"},
		// Arabic preserved (Unicode letter, not Latin)
		{"حب", "حب"},
	}
	for _, c := range cases {
		got := lexdb.Normalize(c.in)
		if got != c.want {
			t.Errorf("Normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestBuildRejectsDuplicateWordID verifies that Build() returns an error
// when two words have the same word_id (CRITICAL: silent corruption otherwise).
func TestBuildRejectsDuplicateWordID(t *testing.T) {
	dir := t.TempDir()
	roots := []seed.Root{{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"}}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative"},
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negativ", Lang: "DE", Norm: "negativ"}, // same word_id!
	}
	_, err := lexdb.Build(roots, words, filepath.Join(dir, "dup_id.lsdb"))
	if err == nil {
		t.Fatal("duplicate word_id must be rejected by Build()")
	}
}

// TestBuildRejectsDuplicateNormLang verifies that Build() returns an error
// when two words have the same normalized form in the same language.
func TestBuildRejectsDuplicateNormLang(t *testing.T) {
	dir := t.TempDir()
	roots := []seed.Root{{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"}}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "cafe", Lang: "PT", Norm: "cafe"},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "café", Lang: "PT", Norm: "cafe"}, // same norm + lang!
	}
	_, err := lexdb.Build(roots, words, filepath.Join(dir, "dup_norm.lsdb"))
	if err == nil {
		t.Fatal("duplicate (norm, lang) pair must be rejected by Build()")
	}
}

// TestLookupWordInLang verifies that lang-specific lookup disambiguates
// homographs that share the same normalized form (e.g. "mais" PT vs FR).
func TestLookupWordInLang(t *testing.T) {
	dir := t.TempDir()
	roots := []seed.Root{
		{RootID: 1, RootStr: "neg", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	// "mais" means "but/more" in PT (root 1) and is the word for "corn" in FR (root 2)
	// After normalize: both become "mais"
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "mais", Lang: "PT", Norm: "mais", Sentiment: 0x00120180},
		{WordID: 8193, RootID: 2, Variant: 1, Word: "maïs", Lang: "FR", Norm: "mais", Sentiment: 0x00130140},
	}
	_, err := lexdb.Build(roots, words, filepath.Join(dir, "lang.lsdb"))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(filepath.Join(dir, "lang.lsdb"))
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	// Lang-specific lookups must return the correct word
	pt := lex.LookupWordInLang("mais", "PT")
	if pt == nil {
		t.Fatal("LookupWordInLang('mais', 'PT') returned nil")
	}
	if pt.WordID != 4097 {
		t.Fatalf("PT 'mais': want word_id=4097, got %d", pt.WordID)
	}

	fr := lex.LookupWordInLang("maïs", "FR")
	if fr == nil {
		t.Fatal("LookupWordInLang('maïs', 'FR') returned nil")
	}
	if fr.WordID != 8193 {
		t.Fatalf("FR 'maïs': want word_id=8193, got %d", fr.WordID)
	}

	// Any-language lookup returns first (lowest word_id) match
	any := lex.LookupWord("mais")
	if any == nil {
		t.Fatal("LookupWord('mais') returned nil")
	}
	if any.WordID != 4097 {
		t.Fatalf("any-language 'mais': want word_id=4097 (first), got %d", any.WordID)
	}
}

func TestChecksumValidation(t *testing.T) {
	_, path := buildTestLexicon(t)

	// Corrupt the file (flip a byte in the data section)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[100] ^= 0xFF
	os.WriteFile(path, data, 0644)

	_, err = lexdb.Load(path)
	if err == nil {
		t.Fatal("corrupted file should fail to load")
	}
}

func TestValidationErrors(t *testing.T) {
	dir := t.TempDir()

	// Duplicate root_id
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 1, RootStr: "duplicate", Origin: "LATIN", MeaningEN: "dup"},
	}
	_, err := lexdb.Build(roots, nil, filepath.Join(dir, "dup.lsdb"))
	if err == nil {
		t.Fatal("duplicate root_id must be rejected")
	}

	// Word referencing unknown root
	roots2 := []seed.Root{{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"}}
	words2 := []seed.Word{{WordID: 8193, RootID: 2, Variant: 1, Word: "bom", Lang: "PT", Norm: "bom"}}
	_, err = lexdb.Build(roots2, words2, filepath.Join(dir, "orphan.lsdb"))
	if err == nil {
		t.Fatal("word with unknown root_id must be rejected")
	}
}
