package analyze_test

import (
	"path/filepath"
	"testing"

	"github.com/kak/umcs/pkg/analyze"
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
)

func buildTestLex(t *testing.T) *lexdb.Lexicon {
	t.Helper()
	dir := t.TempDir()

	terrSent, _ := sentiment.Pack("NEGATIVE", "STRONG", "EVALUATION", "GENERAL", nil)
	goodSent, _ := sentiment.Pack("POSITIVE", "MODERATE", "EVALUATION", "GENERAL", nil)
	notSent, _ := sentiment.Pack("NEUTRAL", "NONE", "NEGATION_MARKER", "GENERAL", []string{"NEGATION_MARKER"})
	verySent, _ := sentiment.Pack("NEUTRAL", "NONE", "INTENSIFIER", "GENERAL", []string{"INTENSIFIER"})
	slightlySent, _ := sentiment.Pack("NEUTRAL", "NONE", "DOWNTONER", "GENERAL", []string{"DOWNTONER"})

	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
		{RootID: 10, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
		{RootID: 55, RootStr: "lent", Origin: "LATIN", MeaningEN: "slow"},
		{RootID: 61, RootStr: "ne", Origin: "PIE", MeaningEN: "negation"},
		{RootID: 62, RootStr: "vald", Origin: "PGmc", MeaningEN: "very"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: 0x00120180},
		{WordID: 8193, RootID: 2, Variant: 1, Word: "good", Lang: "EN", Norm: "good", Sentiment: goodSent},
		{WordID: 40961, RootID: 10, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: terrSent},
		{WordID: 225281, RootID: 55, Variant: 1, Word: "slightly", Lang: "EN", Norm: "slightly", Sentiment: slightlySent},
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

// TestIntensifierThenNegation verifies the intensifier carryover fix:
// "very not terrible" — "very" should NOT amplify "terrible" because "not"
// clears intensifyNext before "terrible" is processed.
func TestIntensifierThenNegation(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "very not terrible")
	// "not" clears intensifyNext → "terrible" is negated but NOT amplified.
	// Negated STRONG (3) → weight = +3, not +6.
	if r.TotalScore != 3 {
		t.Fatalf("'very not terrible': want score=3 (negated, not amplified), got %d", r.TotalScore)
	}
	// Verify the token-level state
	found := false
	for _, tok := range r.Tokens {
		if tok.Surface == "terrible" {
			if tok.Amplified {
				t.Fatal("'terrible' should NOT be amplified after 'very not'")
			}
			if !tok.Negated {
				t.Fatal("'terrible' should be negated")
			}
			found = true
		}
	}
	if !found {
		t.Fatal("'terrible' token not found in result")
	}
}

// TestDowntonerThenNegation verifies that a downtoner before "not" does not
// carry over to affect the word after "not".
func TestDowntonerThenNegation(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "slightly not good")
	// "not" clears downtonNext → "good" is negated but NOT downtoned.
	// Negated MODERATE (2) → weight = -2, not -1.
	if r.TotalScore != -2 {
		t.Fatalf("'slightly not good': want score=-2 (negated, not downtoned), got %d", r.TotalScore)
	}
}

// TestVerdictBoundary verifies that the threshold is strict (> 2, < -2).
// Score of exactly 2 or -2 → NEUTRAL.
func TestVerdictBoundary(t *testing.T) {
	lex := buildTestLex(t)

	cases := []struct {
		text    string
		verdict string
		desc    string
	}{
		// "good" MODERATE weight=2 → score=2 → NEUTRAL (not > 2)
		{"good", "NEUTRAL", "score=2 → NEUTRAL"},
		// "terrible" STRONG weight=-3 → score=-3 → NEGATIVE
		{"terrible", "NEGATIVE", "score=-3 → NEGATIVE"},
		// "negative" MODERATE(-2) → score=-2 → NEUTRAL (not < -2)
		{"negative", "NEUTRAL", "score=-2 → NEUTRAL"},
	}
	for _, c := range cases {
		r := analyze.Analyze(lex, c.text)
		if r.Verdict != c.verdict {
			t.Errorf("%s: got verdict=%s score=%d, want %s", c.desc, r.Verdict, r.TotalScore, c.verdict)
		}
	}
}

// TestNegationWindowExpiry verifies that negation expires after negationWindow tokens.
// "not [oov1] [oov2] [oov3] terrible" — 3 unknown tokens deplete the window
// before "terrible", so it should NOT be negated.
func TestNegationWindowExpiry(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "not xyz1 xyz2 xyz3 terrible")
	// All 3 unknown words consume the negation window.
	// "terrible" should be negative (not negated).
	for _, tok := range r.Tokens {
		if tok.Surface == "terrible" && tok.Negated {
			t.Fatal("'terrible' should NOT be negated — negation window expired through 3 unknown words")
		}
	}
	if r.TotalScore >= 0 {
		t.Fatalf("negation expired: 'terrible' should be NEGATIVE, got score=%d", r.TotalScore)
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
