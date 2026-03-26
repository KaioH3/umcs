package lexdb_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/kak/lex-sentiment/pkg/lexdb"
	"github.com/kak/lex-sentiment/pkg/seed"
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
