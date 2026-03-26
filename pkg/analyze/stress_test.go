package analyze_test

// Stress tests: long texts, mixed languages, cascade negation, orphaned scope
// modifiers, and Unicode inputs. All use the buildTestLex fixture from analyze_test.go.

import (
	"strings"
	"testing"

	"github.com/kak/lex-sentiment/pkg/analyze"
)

func TestAnalyzeEmptyString(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "")
	if r.Total != 0 {
		t.Fatalf("empty string: want Total=0, got %d", r.Total)
	}
	if r.Matched != 0 {
		t.Fatalf("empty string: want Matched=0, got %d", r.Matched)
	}
	if r.Verdict != "NEUTRAL" {
		t.Fatalf("empty string: want NEUTRAL, got %s", r.Verdict)
	}
}

func TestAnalyzeOnlySpaces(t *testing.T) {
	lex := buildTestLex(t)
	r := analyze.Analyze(lex, "     \t\n   ")
	if r.Total != 0 {
		t.Fatalf("whitespace: want Total=0, got %d", r.Total)
	}
	if r.Verdict != "NEUTRAL" {
		t.Fatalf("whitespace: want NEUTRAL, got %s", r.Verdict)
	}
}

func TestAnalyzeLongText(t *testing.T) {
	lex := buildTestLex(t)
	// Repeat "terrible" 1000 times — should not hang, should be negative
	text := strings.Repeat("terrible ", 1000)
	r := analyze.Analyze(lex, text)
	if r.Total != 1000 {
		t.Fatalf("want 1000 tokens, got %d", r.Total)
	}
	if r.Matched != 1000 {
		t.Fatalf("want 1000 matched, got %d", r.Matched)
	}
	if r.Verdict != "NEGATIVE" {
		t.Fatalf("1000×terrible should be NEGATIVE, got %s (score=%d)", r.Verdict, r.TotalScore)
	}
}

func TestAnalyzeMixedLanguageNoOOVPanic(t *testing.T) {
	lex := buildTestLex(t)
	// Mix of known tokens + OOV (PT/DE words not in the EN-only fixture)
	r := analyze.Analyze(lex, "not terrible muito bom nicht gut")
	// Exact matches depend on fixture — just verify no panic and verdict is valid
	switch r.Verdict {
	case "POSITIVE", "NEGATIVE", "NEUTRAL":
		// all valid
	default:
		t.Fatalf("unexpected verdict: %q", r.Verdict)
	}
	if r.Total != 6 {
		t.Fatalf("want 6 tokens, got %d", r.Total)
	}
}

func TestAnalyzeCascadeNegationEven(t *testing.T) {
	lex := buildTestLex(t)
	// 4 negations (even) → cancel out → "terrible" not negated → NEGATIVE
	r := analyze.Analyze(lex, "not not not not terrible")
	if r.TotalScore >= 0 {
		t.Fatalf("4 negations cancel out: score should be negative, got %d", r.TotalScore)
	}
}

func TestAnalyzeCascadeNegationOdd(t *testing.T) {
	lex := buildTestLex(t)
	// 3 negations (odd) → "terrible" negated → POSITIVE
	r := analyze.Analyze(lex, "not not not terrible")
	if r.TotalScore <= 0 {
		t.Fatalf("3 negations (odd): score should be positive, got %d", r.TotalScore)
	}
}

func TestAnalyzeOrphanedIntensifier(t *testing.T) {
	lex := buildTestLex(t)
	// "very" at end of text — intensifyNext set but never applied
	// Must not panic, TotalScore should be 0 (very has NEUTRAL/NONE weight)
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("panic on orphaned intensifier: %v", r)
			}
		}()
		r := analyze.Analyze(lex, "very")
		if r.Verdict != "NEUTRAL" {
			t.Errorf("orphaned intensifier: want NEUTRAL, got %s (score=%d)", r.Verdict, r.TotalScore)
		}
	}()
}

func TestAnalyzeIntensifierThenOOV(t *testing.T) {
	lex := buildTestLex(t)
	// Intensifier followed by OOV word — intensifyNext must be cleared
	r1 := analyze.Analyze(lex, "terrible")
	r2 := analyze.Analyze(lex, "very unknown_xyz terrible")
	// "very" intensifies OOV (no-op), intensifyNext cleared;
	// "terrible" gets normal weight — not doubled
	if r2.TotalScore != r1.TotalScore {
		t.Fatalf("intensifier before OOV should not carry over to next sentence token: "+
			"base=%d, got=%d", r1.TotalScore, r2.TotalScore)
	}
}

func TestAnalyzeAllNeutralConnectors(t *testing.T) {
	lex := buildTestLex(t)
	// Text with only negation markers and unknown words → TotalScore=0
	r := analyze.Analyze(lex, "not not not")
	if r.TotalScore != 0 {
		t.Fatalf("only negation markers: want score=0, got %d", r.TotalScore)
	}
}

func TestAnalyzeUnicodeSafety(t *testing.T) {
	lex := buildTestLex(t)
	inputs := []string{
		"😀 terrible 💯",
		"日本語 good",
		"العربية negative",
		"très bien",
		"über alles",
		"naïve",
	}
	for _, text := range inputs {
		func(input string) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("panic on unicode input %q: %v", input, r)
				}
			}()
			r := analyze.Analyze(lex, input)
			// Verdict must always be one of the valid values
			switch r.Verdict {
			case "POSITIVE", "NEGATIVE", "NEUTRAL":
			default:
				t.Errorf("invalid verdict %q for input %q", r.Verdict, input)
			}
		}(text)
	}
}

func TestAnalyzeScoreProportional(t *testing.T) {
	lex := buildTestLex(t)
	r1 := analyze.Analyze(lex, "terrible")
	r2 := analyze.Analyze(lex, "terrible terrible terrible")
	// Score should scale with number of matched tokens
	if r2.TotalScore != r1.TotalScore*3 {
		t.Fatalf("3x terrible should give 3x score: want %d, got %d",
			r1.TotalScore*3, r2.TotalScore)
	}
}

func TestAnalyzePunctuationVariants(t *testing.T) {
	lex := buildTestLex(t)
	base := analyze.Analyze(lex, "terrible")
	variants := []string{
		"terrible!",
		"terrible.",
		"(terrible)",
		"terrible?",
		"\"terrible\"",
		"terrible...",
	}
	for _, v := range variants {
		r := analyze.Analyze(lex, v)
		if r.TotalScore != base.TotalScore {
			t.Errorf("punctuation variant %q: want score=%d, got %d", v, base.TotalScore, r.TotalScore)
		}
	}
}
