package analyze_test

import (
	"path/filepath"
	"testing"

	"github.com/kak/lex-sentiment/pkg/analyze"
	"github.com/kak/lex-sentiment/pkg/lexdb"
	"github.com/kak/lex-sentiment/pkg/seed"
	"github.com/kak/lex-sentiment/pkg/sentiment"
)

func buildTestLex(t *testing.T) *lexdb.Lexicon {
	t.Helper()
	dir := t.TempDir()

	terrSent, _ := sentiment.Pack("NEGATIVE", "STRONG", "EVALUATION", "GENERAL", nil)
	goodSent, _ := sentiment.Pack("POSITIVE", "MODERATE", "EVALUATION", "GENERAL", nil)
	notSent, _ := sentiment.Pack("NEUTRAL", "NONE", "NEGATION_MARKER", "GENERAL", []string{"NEGATION_MARKER"})
	verySent, _ := sentiment.Pack("NEUTRAL", "NONE", "INTENSIFIER", "GENERAL", []string{"INTENSIFIER"})

	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
		{RootID: 10, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
		{RootID: 61, RootStr: "ne", Origin: "PIE", MeaningEN: "negation"},
		{RootID: 62, RootStr: "vald", Origin: "PGmc", MeaningEN: "very"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: 0x00120180},
		{WordID: 8193, RootID: 2, Variant: 1, Word: "good", Lang: "EN", Norm: "good", Sentiment: goodSent},
		{WordID: 40961, RootID: 10, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: terrSent},
		{WordID: 249857, RootID: 61, Variant: 1, Word: "not", Lang: "EN", Norm: "not", Sentiment: notSent},
		{WordID: 253953, RootID: 62, Variant: 1, Word: "very", Lang: "EN", Norm: "very", Sentiment: verySent},
	}

	_, err := lexdb.Build(roots, words, filepath.Join(dir, "test.lsdb"))
	if err != nil {
		t.Fatal(err)
	}
	lex, err := lexdb.Load(filepath.Join(dir, "test.lsdb"))
	if err != nil {
		t.Fatal(err)
	}
	return lex
}

func TestNegation(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "not terrible")
	// "terrible" NEGATIVE negated → positive contribution
	if r.TotalScore <= 0 {
		t.Fatalf("'not terrible' should be positive, got score=%d verdict=%s", r.TotalScore, r.Verdict)
	}
}

func TestPositive(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "very good")
	// "good" POSITIVE amplified 2×
	if r.TotalScore < 3 {
		t.Fatalf("'very good' should have high positive score, got %d", r.TotalScore)
	}
}

func TestNegative(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "terrible negative")
	if r.Verdict != "NEGATIVE" {
		t.Fatalf("'terrible negative' should be NEGATIVE, got %s (score=%d)", r.Verdict, r.TotalScore)
	}
}

func TestNeutral(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "unknown words here")
	if r.Matched != 0 {
		t.Fatalf("all unknown words, matched=%d", r.Matched)
	}
	if r.Verdict != "NEUTRAL" {
		t.Fatalf("no matches → NEUTRAL, got %s", r.Verdict)
	}
}

func TestIntensifierAmplifies(t *testing.T) {
	lex := buildTestLex(t)
	r1 := analyze.Analyze(lex, "terrible")
	r2 := analyze.Analyze(lex, "very terrible")
	if r2.TotalScore >= r1.TotalScore {
		t.Fatalf("intensified score (%d) should be more negative than base (%d)", r2.TotalScore, r1.TotalScore)
	}
}

func TestDoubleNegation(t *testing.T) {
	lex := buildTestLex(t)
	// "not not terrible" → should be NEGATIVE again (double negation)
	r := analyze.Analyze(lex, "not not terrible")
	if r.TotalScore >= 0 {
		t.Fatalf("'not not terrible' double negation → negative, got score=%d", r.TotalScore)
	}
}

func TestPunctuationStrip(t *testing.T) {
	lex := buildTestLex(t)
	r1 := analyze.Analyze(lex, "terrible")
	r2 := analyze.Analyze(lex, "terrible!")
	if r1.TotalScore != r2.TotalScore {
		t.Fatalf("punctuation should be stripped: %d vs %d", r1.TotalScore, r2.TotalScore)
	}
}

func TestTokenCount(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "this is not terrible at all")
	if r.Total != 6 {
		t.Fatalf("want 6 tokens, got %d", r.Total)
	}
	if r.Matched != 2 { // "not" + "terrible"
		t.Fatalf("want 2 matched (not, terrible), got %d", r.Matched)
	}
}
