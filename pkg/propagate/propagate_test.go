package propagate_test

import (
	"path/filepath"
	"testing"

	"github.com/kak/lex-sentiment/pkg/lexdb"
	"github.com/kak/lex-sentiment/pkg/propagate"
	"github.com/kak/lex-sentiment/pkg/seed"
	"github.com/kak/lex-sentiment/pkg/sentiment"
)

func buildPropLex(t *testing.T, roots []seed.Root, words []seed.Word) *lexdb.Lexicon {
	t.Helper()
	dir := t.TempDir()
	_, err := lexdb.Build(roots, words, filepath.Join(dir, "prop.lsdb"))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(filepath.Join(dir, "prop.lsdb"))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return lex
}

const (
	negSent = uint32(0x00120180) // NEGATIVE/MODERATE/EVALUATION/GENERAL
	posSent = uint32(0x00130140) // POSITIVE/STRONG/EVALUATION/GENERAL
)

func TestPropagateBasic(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: negSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: 0}, // unannotated
	}
	lex := buildPropLex(t, roots, words)

	results := propagate.Run(lex)

	if len(results) != 1 {
		t.Fatalf("want 1 propagation result, got %d", len(results))
	}
	if results[0].WordID != 4098 {
		t.Fatalf("want propagation to word_id=4098, got %d", results[0].WordID)
	}
	if results[0].OldSent != 0 {
		t.Fatalf("want OldSent=0 (was unannotated), got 0x%08X", results[0].OldSent)
	}
	if results[0].NewSent == 0 {
		t.Fatal("NewSent should be non-zero after propagation")
	}
	if sentiment.Polarity(results[0].NewSent) != sentiment.PolarityNegative {
		t.Fatalf("propagated polarity should be NEGATIVE (0x%X), got 0x%X",
			sentiment.PolarityNegative, sentiment.Polarity(results[0].NewSent))
	}
}

func TestPropagateNoUnannotated(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: negSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: negSent},
	}
	lex := buildPropLex(t, roots, words)
	results := propagate.Run(lex)
	if len(results) != 0 {
		t.Fatalf("all annotated: want 0 results, got %d", len(results))
	}
}

func TestPropagateNoAnnotated(t *testing.T) {
	// No words have sentiment → no consensus possible → nothing propagated
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: 0},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)
	results := propagate.Run(lex)
	if len(results) != 0 {
		t.Fatalf("no annotated source: want 0 results, got %d", len(results))
	}
}

func TestPropagateMajorityVote(t *testing.T) {
	// 3 NEGATIVE + 1 POSITIVE + 1 unannotated → consensus must be NEGATIVE
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "a", Lang: "EN", Norm: "a", Sentiment: negSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "b", Lang: "PT", Norm: "b", Sentiment: negSent},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "c", Lang: "ES", Norm: "c", Sentiment: negSent},
		{WordID: 4100, RootID: 1, Variant: 4, Word: "d", Lang: "IT", Norm: "d", Sentiment: posSent},
		{WordID: 4101, RootID: 1, Variant: 5, Word: "e", Lang: "DE", Norm: "e", Sentiment: 0}, // unannotated
	}
	lex := buildPropLex(t, roots, words)

	results := propagate.Run(lex)
	if len(results) != 1 {
		t.Fatalf("want 1 propagation result, got %d", len(results))
	}
	if sentiment.Polarity(results[0].NewSent) != sentiment.PolarityNegative {
		t.Fatalf("majority vote (3 NEG vs 1 POS) should give NEGATIVE, got 0x%X",
			sentiment.Polarity(results[0].NewSent))
	}
}

// TestMajorityVoteDomainPreservation verifies the domain-loss bug fix:
// cognates with FINANCIAL domain should produce consensus with FINANCIAL, not GENERAL.
func TestMajorityVoteDomainPreservation(t *testing.T) {
	finNegSent, _ := sentiment.Pack("NEGATIVE", "MODERATE", "EVALUATION", "FINANCIAL", nil)

	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: finNegSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: finNegSent},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "negativo", Lang: "ES", Norm: "negativo", Sentiment: 0}, // unannotated
	}
	lex := buildPropLex(t, roots, words)

	results := propagate.Run(lex)
	if len(results) != 1 {
		t.Fatalf("want 1 propagation result, got %d", len(results))
	}

	got := results[0].NewSent
	gotDomain := sentiment.Domain(got)
	if gotDomain != sentiment.Domain(finNegSent) {
		t.Fatalf("domain should be preserved as FINANCIAL (0x%X), got 0x%X",
			sentiment.Domain(finNegSent), gotDomain)
	}
	if sentiment.Polarity(got) != sentiment.PolarityNegative {
		t.Fatalf("polarity should be NEGATIVE, got 0x%X", sentiment.Polarity(got))
	}
}

func TestMajorityVotePolarity(t *testing.T) {
	// 3 NEGATIVE + 1 POSITIVE → NEGATIVE wins (duplicates the vote-checking in TestPropagateMajorityVote
	// but here we verify the exact polarity, not just "not positive")
	roots := []seed.Root{
		{RootID: 5, RootStr: "fort", Origin: "LATIN", MeaningEN: "strong"},
	}
	posSent, _ := sentiment.Pack("POSITIVE", "STRONG", "EVALUATION", "GENERAL", nil)
	negSentLocal, _ := sentiment.Pack("NEGATIVE", "MODERATE", "EVALUATION", "GENERAL", nil)
	words := []seed.Word{
		{WordID: 20481, RootID: 5, Variant: 1, Word: "w1", Lang: "EN", Norm: "w1", Sentiment: negSentLocal},
		{WordID: 20482, RootID: 5, Variant: 2, Word: "w2", Lang: "PT", Norm: "w2", Sentiment: negSentLocal},
		{WordID: 20483, RootID: 5, Variant: 3, Word: "w3", Lang: "ES", Norm: "w3", Sentiment: negSentLocal},
		{WordID: 20484, RootID: 5, Variant: 4, Word: "w4", Lang: "IT", Norm: "w4", Sentiment: posSent},
		{WordID: 20485, RootID: 5, Variant: 5, Word: "w5", Lang: "DE", Norm: "w5", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)
	results := propagate.Run(lex)
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	if sentiment.Polarity(results[0].NewSent) != sentiment.PolarityNegative {
		t.Fatalf("3 NEGATIVE vs 1 POSITIVE → NEGATIVE should win")
	}
}

func TestPropagateEmptyLexicon(t *testing.T) {
	lex := buildPropLex(t, nil, nil)
	results := propagate.Run(lex)
	if len(results) != 0 {
		t.Fatalf("empty lexicon: want 0 results, got %d", len(results))
	}
}

func TestPropagateRootWithNoWords(t *testing.T) {
	// Root exists but has no words → skip cleanly, no panic
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"}, // no words
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: negSent},
	}
	lex := buildPropLex(t, roots, words)

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("panic on root with no words: %v", r)
			}
		}()
		propagate.Run(lex)
	}()
}

func TestPropagateTwoFamilies(t *testing.T) {
	// Two independent root families, each with one unannotated word
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: negSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: 0},
		{WordID: 8193, RootID: 2, Variant: 1, Word: "good", Lang: "EN", Norm: "good", Sentiment: posSent},
		{WordID: 8194, RootID: 2, Variant: 2, Word: "bom", Lang: "PT", Norm: "bom", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)

	results := propagate.Run(lex)
	if len(results) != 2 {
		t.Fatalf("want 2 propagation results (one per family), got %d", len(results))
	}
	// Verify polarity matches each family
	for _, r := range results {
		if r.WordID == 4098 && sentiment.Polarity(r.NewSent) != sentiment.PolarityNegative {
			t.Errorf("negativo: want NEGATIVE, got 0x%X", sentiment.Polarity(r.NewSent))
		}
		if r.WordID == 8194 && sentiment.Polarity(r.NewSent) != sentiment.PolarityPositive {
			t.Errorf("bom: want POSITIVE, got 0x%X", sentiment.Polarity(r.NewSent))
		}
	}
}
