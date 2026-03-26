package lexdb_test

// Destructive tests: corrupt files, edge cases, boundary values, overflow.

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kak/lex-sentiment/pkg/lexdb"
	"github.com/kak/lex-sentiment/pkg/morpheme"
	"github.com/kak/lex-sentiment/pkg/seed"
)

// --- Binary corruption ---

func TestTruncatedFile(t *testing.T) {
	_, path := buildTestLexicon(t)
	data, _ := os.ReadFile(path)

	// Try to load every possible truncation
	for i := 1; i < len(data); i += len(data) / 20 {
		os.WriteFile(path, data[:i], 0644)
		_, err := lexdb.Load(path)
		// Must either load cleanly or return an error — never panic
		_ = err
	}
}

func TestWrongMagic(t *testing.T) {
	_, path := buildTestLexicon(t)
	data, _ := os.ReadFile(path)
	copy(data[0:4], []byte("XXXX"))
	os.WriteFile(path, data, 0644)
	_, err := lexdb.Load(path)
	if err == nil {
		t.Fatal("wrong magic should fail")
	}
}

func TestWrongVersion(t *testing.T) {
	_, path := buildTestLexicon(t)
	data, _ := os.ReadFile(path)
	// Overwrite version field (bytes 4-7)
	lexdb.ByteOrder.PutUint32(data[4:], 99)
	// Recalculate checksum would be needed for this to pass checksumming,
	// but wrong version should be caught first
	os.WriteFile(path, data, 0644)
	_, err := lexdb.Load(path)
	if err == nil {
		t.Fatal("wrong version should fail")
	}
}

func TestEmptyFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.lsdb")
	os.WriteFile(path, []byte{}, 0644)
	_, err := lexdb.Load(path)
	if err == nil {
		t.Fatal("empty file should fail to load")
	}
}

func TestAllZeros(t *testing.T) {
	path := filepath.Join(t.TempDir(), "zeros.lsdb")
	os.WriteFile(path, bytes.Repeat([]byte{0}, 1024), 0644)
	_, err := lexdb.Load(path)
	if err == nil {
		t.Fatal("all-zeros file should fail (wrong magic)")
	}
}

func TestNonexistentFile(t *testing.T) {
	_, err := lexdb.Load("/tmp/doesnotexist_lexsent_test_12345.lsdb")
	if err == nil {
		t.Fatal("loading nonexistent file should fail")
	}
}

func TestFlipEveryByte(t *testing.T) {
	_, path := buildTestLexicon(t)
	data, _ := os.ReadFile(path)
	originalLen := len(data)

	// Flip one byte at a time after header — must not panic, must return error
	for i := lexdb.HeaderSize; i < originalLen; i += originalLen / 10 {
		modified := make([]byte, len(data))
		copy(modified, data)
		modified[i] ^= 0xFF
		os.WriteFile(path, modified, 0644)
		// Must not panic
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("panic at byte %d: %v", i, r)
				}
			}()
			lexdb.Load(path)
		}()
	}
}

// --- ID boundary tests ---

func TestMaxWordID(t *testing.T) {
	maxRoot := uint32(morpheme.MaxRootID)
	maxVar := uint32(morpheme.MaxVariant)
	id, err := morpheme.MakeWordID(maxRoot, maxVar)
	if err != nil {
		t.Fatal(err)
	}
	if morpheme.RootOf(id) != maxRoot {
		t.Fatalf("root extraction failed at max: want %d got %d", maxRoot, morpheme.RootOf(id))
	}
	if morpheme.VariantOf(id) != maxVar {
		t.Fatalf("variant extraction failed at max: want %d got %d", maxVar, morpheme.VariantOf(id))
	}
}

func TestRootIDZeroInvalid(t *testing.T) {
	_, err := morpheme.MakeWordID(0, 1)
	if err == nil {
		t.Fatal("root_id=0 must be invalid")
	}
}

func TestVariantZeroInvalid(t *testing.T) {
	_, err := morpheme.MakeWordID(1, 0)
	if err == nil {
		t.Fatal("variant=0 must be invalid")
	}
}

func TestOverflowRootID(t *testing.T) {
	_, err := morpheme.MakeWordID(morpheme.MaxRootID+1, 1)
	if err == nil {
		t.Fatal("root_id overflow must be rejected")
	}
}

// --- Build edge cases ---

func TestBuildNoRoots(t *testing.T) {
	path := filepath.Join(t.TempDir(), "empty.lsdb")
	stats, err := lexdb.Build(nil, nil, path)
	if err != nil {
		t.Fatal(err)
	}
	if stats.RootCount != 0 || stats.WordCount != 0 {
		t.Fatalf("empty build: want 0/0, got %d/%d", stats.RootCount, stats.WordCount)
	}
	lex, err := lexdb.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if lex.LookupWord("anything") != nil {
		t.Fatal("empty lexicon should find nothing")
	}
}

func TestBuildDuplicateWordID(t *testing.T) {
	// Two words with the same word_id — builder should accept it (same word, different record)
	// This tests that we don't crash (deduplication in index is OK)
	roots := []seed.Root{{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"}}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative"},
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative"}, // exact duplicate
	}
	path := filepath.Join(t.TempDir(), "dup.lsdb")
	_, err := lexdb.Build(roots, words, path)
	// Should not crash; may or may not error depending on policy
	_ = err
}

func TestBuildRootWithNoWords(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"}, // no words
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative"},
	}
	path := filepath.Join(t.TempDir(), "nowords.lsdb")
	stats, err := lexdb.Build(roots, words, path)
	if err != nil {
		t.Fatal(err)
	}
	if stats.RootCount != 2 {
		t.Fatalf("want 2 roots, got %d", stats.RootCount)
	}
	lex, _ := lexdb.Load(path)
	// Root "bon" exists but has no cognates
	root := lex.LookupRoot(2)
	if root == nil {
		t.Fatal("root 2 should exist")
	}
	if root.WordCount != 0 {
		t.Fatalf("root 'bon' has no words, want 0, got %d", root.WordCount)
	}
}

func TestBuildEtymologyCycle(t *testing.T) {
	// A → B → A (cycle in parent links) — builder validates parent exists, not cycles
	roots := []seed.Root{
		{RootID: 1, RootStr: "a", Origin: "TEST", MeaningEN: "a", ParentRootID: 2},
		{RootID: 2, RootStr: "b", Origin: "TEST", MeaningEN: "b", ParentRootID: 1},
	}
	path := filepath.Join(t.TempDir(), "cycle.lsdb")
	_, err := lexdb.Build(roots, nil, path)
	if err != nil {
		t.Fatal("builder accepts cycles (reader's EtymologyChain has cycle guard)")
	}
	lex, err := lexdb.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	// EtymologyChain must not infinite-loop
	chain := lex.EtymologyChain(1)
	if len(chain) > 10 {
		t.Fatalf("cycle guard failed, chain length=%d", len(chain))
	}
}

// --- Lookup edge cases ---

func TestLookupEmptyString(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	if lex.LookupWord("") != nil {
		t.Fatal("empty string lookup should return nil")
	}
}

func TestLookupPunctuation(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	if lex.LookupWord("!!!") != nil {
		t.Fatal("punctuation-only lookup should return nil")
	}
}

func TestLookupVeryLongWord(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	// 10KB word
	long := strings.Repeat("a", 10240)
	if lex.LookupWord(long) != nil {
		t.Fatal("very long word should not be found")
	}
}

func TestLookupUnicode(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	// Emoji, CJK, Arabic — should return nil, not panic
	for _, s := range []string{"😀", "日本語", "العربية", "🔥💯"} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("panic on Unicode input %q: %v", s, r)
				}
			}()
			lex.LookupWord(s)
		}()
	}
}

func TestCognatesUnknownWordID(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	// word_id for non-existent root_id → should return empty, not panic
	cognates := lex.Cognates(999 << 12)
	if cognates != nil && len(cognates) > 0 {
		t.Fatalf("non-existent root should return empty cognates, got %d", len(cognates))
	}
}

func TestCognatesZeroWordID(t *testing.T) {
	lex, _ := buildTestLexicon(t)
	// word_id=0 → root_id=0 (invalid)
	cognates := lex.Cognates(0)
	if len(cognates) > 0 {
		t.Fatal("word_id=0 should return empty cognates")
	}
}

// --- Normalize edge cases ---

func TestNormalizeEmpty(t *testing.T) {
	got := lexdb.Normalize("")
	if got != "" {
		t.Fatalf("normalize empty → %q", got)
	}
}

func TestNormalizeOnlySpaces(t *testing.T) {
	got := lexdb.Normalize("   ")
	if got != "" {
		t.Fatalf("normalize spaces → %q", got)
	}
}

func TestNormalizeGermanSS(t *testing.T) {
	got := lexdb.Normalize("Straße")
	if got != "strasse" {
		t.Fatalf("ß → ss, got %q", got)
	}
}

func TestNormalizeAllDiacritics(t *testing.T) {
	cases := map[string]string{
		"áàâãäå": "aaaaaa",
		"éèêë":   "eeee",
		"íìîï":   "iiii",
		"óòôõö":  "ooooo",
		"úùûü":   "uuuu",
		"ç":      "c",
		"ñ":      "n",
	}
	for in, want := range cases {
		got := lexdb.Normalize(in)
		if got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
}
