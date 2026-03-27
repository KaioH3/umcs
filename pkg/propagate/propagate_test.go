package propagate_test

import (
	"path/filepath"
	"testing"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/propagate"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
)

func buildPropLex(t *testing.T, roots []seed.Root, words []seed.Word) *lexdb.Lexicon {
	t.Helper()
	dir := t.TempDir()
	_, err := lexdb.Build(roots, words, filepath.Join(dir, "prop.umcs"))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(filepath.Join(dir, "prop.umcs"))
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

// ── PropagateExtended tests ────────────────────────────────────────────────────

func TestPropagateExtended_FillsArousalForCognates(t *testing.T) {
	// EN word has arousal=HIGH, PT cognate has arousal=NONE → should be filled
	const arousalHigh = uint32(3 << 4) // ArousalHigh
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	baseSent := negSent | arousalHigh
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: baseSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: negSent}, // has polarity but no arousal
	}
	lex := buildPropLex(t, roots, words)

	n := propagate.PropagateExtended(lex)
	if n == 0 {
		t.Error("PropagateExtended should have updated at least one word")
	}
	// Check that terrível now has arousal
	w := lex.LookupWordInLang("terrível", "PT")
	if w == nil {
		w = lex.LookupWord("terrivel")
	}
	if w == nil {
		t.Skip("terrível not found in test lexicon")
	}
	ar := (w.Sentiment >> 4) & 0x3
	if ar != 3 {
		t.Errorf("terrível arousal = %d, want 3 (HIGH) after PropagateExtended", ar)
	}
}

func TestPropagateExtended_PreservesExistingAnnotations(t *testing.T) {
	// Word already has arousal → should NOT be overwritten
	const arousalHigh = uint32(3 << 4)
	const arousalLow = uint32(1 << 4)
	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "good", Lang: "EN", Norm: "good", Sentiment: posSent | arousalHigh},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "bom", Lang: "PT", Norm: "bom", Sentiment: posSent | arousalLow},
	}
	lex := buildPropLex(t, roots, words)
	propagate.PropagateExtended(lex)

	// "bom" already had arousalLow — must not be changed to arousalHigh
	w := lex.LookupWordInLang("bom", "PT")
	if w == nil {
		t.Skip("bom not found in test lexicon")
	}
	ar := (w.Sentiment >> 4) & 0x3
	if ar != 1 {
		t.Errorf("bom arousal = %d, want 1 (LOW, original value, not overwritten)", ar)
	}
}

func TestPropagateExtended_ReturnsUpdateCount(t *testing.T) {
	const arousalHigh = uint32(3 << 4)
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: negSent | arousalHigh},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: negSent},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "terrible_es", Lang: "ES", Norm: "terrible_es", Sentiment: negSent},
	}
	lex := buildPropLex(t, roots, words)
	n := propagate.PropagateExtended(lex)
	if n <= 0 {
		t.Errorf("PropagateExtended returned %d, want > 0 (2 words missing arousal)", n)
	}
}

func TestPropagateExtended_EmptyLexiconIsNoop(t *testing.T) {
	lex := buildPropLex(t, nil, nil)
	n := propagate.PropagateExtended(lex)
	if n != 0 {
		t.Errorf("empty lexicon: PropagateExtended returned %d, want 0", n)
	}
}

func TestPropagateExtended_SyllablesFromIpaAnnotatedCognate(t *testing.T) {
	// EN word has syllable count in flags; PT cognate has none → should be filled
	const syllables3 = uint32(3 << 28) // 3 syllables
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: negSent, Flags: syllables3},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: negSent, Flags: 0},
	}
	lex := buildPropLex(t, roots, words)
	propagate.PropagateExtended(lex)

	w := lex.LookupWordInLang("terrível", "PT")
	if w == nil {
		w = lex.LookupWord("terrivel")
	}
	if w == nil {
		t.Skip("terrível not found")
	}
	syl := (w.Flags >> 28) & 0xF
	if syl != 3 {
		t.Errorf("terrível syllables = %d, want 3 (propagated from EN cognate)", syl)
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

// ── Additional Run() edge case tests ──────────────────────────────────────────

func TestPropagateSingleWordFamily(t *testing.T) {
	// Root with exactly one word (annotated) → nothing to propagate
	roots := []seed.Root{
		{RootID: 1, RootStr: "unic", Origin: "LATIN", MeaningEN: "one"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "unique", Lang: "EN", Norm: "unique", Sentiment: posSent},
	}
	lex := buildPropLex(t, roots, words)
	results := propagate.Run(lex)
	if len(results) != 0 {
		t.Fatalf("single annotated word family: want 0 results, got %d", len(results))
	}
}

func TestPropagateSingleWordUnannotated(t *testing.T) {
	// Root with exactly one unannotated word → no consensus source → 0 results
	roots := []seed.Root{
		{RootID: 1, RootStr: "sol", Origin: "LATIN", MeaningEN: "alone"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "solo", Lang: "EN", Norm: "solo", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)
	results := propagate.Run(lex)
	if len(results) != 0 {
		t.Fatalf("single unannotated word: want 0 results, got %d", len(results))
	}
}

func TestPropagateMultipleUnannotated(t *testing.T) {
	// One annotated + multiple unannotated → all unannotated should get sentiment
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: negSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: 0},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "terrible_es", Lang: "ES", Norm: "terrible_es", Sentiment: 0},
		{WordID: 4100, RootID: 1, Variant: 4, Word: "terribile", Lang: "IT", Norm: "terribile", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)
	results := propagate.Run(lex)
	if len(results) != 3 {
		t.Fatalf("want 3 propagations (3 unannotated cognates), got %d", len(results))
	}
	for _, r := range results {
		if r.OldSent != 0 {
			t.Errorf("word %d: OldSent should be 0, got 0x%X", r.WordID, r.OldSent)
		}
		if sentiment.Polarity(r.NewSent) != sentiment.PolarityNegative {
			t.Errorf("word %d: propagated polarity should be NEGATIVE", r.WordID)
		}
	}
}

func TestPropagateAllPositive(t *testing.T) {
	// All annotated words are POSITIVE + one unannotated → should get POSITIVE
	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "bonus", Lang: "EN", Norm: "bonus", Sentiment: posSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "bon", Lang: "FR", Norm: "bon", Sentiment: posSent},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "bueno", Lang: "ES", Norm: "bueno", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)
	results := propagate.Run(lex)
	if len(results) != 1 {
		t.Fatalf("want 1 propagation, got %d", len(results))
	}
	if sentiment.Polarity(results[0].NewSent) != sentiment.PolarityPositive {
		t.Fatalf("all-positive consensus: want POSITIVE, got 0x%X", sentiment.Polarity(results[0].NewSent))
	}
}

func TestPropagateResultFields(t *testing.T) {
	// Verify all fields in Result struct are populated correctly
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: negSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)
	results := propagate.Run(lex)
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	r := results[0]
	if r.TargetWord == "" {
		t.Error("TargetWord should not be empty")
	}
	if r.TargetLang == "" {
		t.Error("TargetLang should not be empty")
	}
	if r.WordID == 0 {
		t.Error("WordID should not be zero")
	}
	if r.OldSent != 0 {
		t.Errorf("OldSent should be 0 (was unannotated), got 0x%X", r.OldSent)
	}
	if r.NewSent == 0 {
		t.Error("NewSent should not be zero after propagation")
	}
}

// ── Additional PropagateExtended tests ────────────────────────────────────────

func TestPropagateExtended_MixedVAD(t *testing.T) {
	// Test with mixed arousal, dominance, AoA values across cognates
	const (
		arousalHigh = uint32(3 << 4)  // HIGH
		arousalMed  = uint32(2 << 4)  // MED
		domHigh     = uint32(3 << 2)  // HIGH dominance
		aoaEarly    = uint32(1)       // early AoA
	)
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: negSent | arousalHigh | domHigh | aoaEarly},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: negSent | arousalMed | domHigh | aoaEarly},
		// Word with polarity set but missing all extended fields
		{WordID: 4099, RootID: 1, Variant: 3, Word: "terrible_es", Lang: "ES", Norm: "terrible_es", Sentiment: negSent},
	}
	lex := buildPropLex(t, roots, words)
	n := propagate.PropagateExtended(lex)
	if n == 0 {
		t.Error("should have updated at least one word with VAD values")
	}
	// Check ES word got dominance and AoA propagated
	for i := range lex.Words {
		w := &lex.Words[i]
		if lex.WordStr(w) == "terrible_es" {
			dom := (w.Sentiment >> 2) & 0x3
			if dom != 3 {
				t.Errorf("terrible_es dominance = %d, want 3 (HIGH)", dom)
			}
			aoa := w.Sentiment & 0x3
			if aoa != 1 {
				t.Errorf("terrible_es AoA = %d, want 1 (EARLY)", aoa)
			}
		}
	}
}

func TestPropagateExtended_POSConsensus_AllAgreeNoTarget(t *testing.T) {
	// When all annotated cognates have the same POS, posConsensus is set.
	// But POS propagation only fills words where POS bits are zero AND
	// the word itself is annotated (s!=0). Since ALL annotated words
	// contribute to posVals, a word with POS=0 makes the consensus fail.
	// This test verifies that when all words already have the same POS,
	// no spurious update occurs (no word has POS=0).
	const (
		posNoun     = uint32(1 << 29)
		arousalHigh = uint32(3 << 4)
	)
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terror", Lang: "EN", Norm: "terror", Sentiment: negSent | posNoun | arousalHigh},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terror_pt", Lang: "PT", Norm: "terror_pt", Sentiment: negSent | posNoun | arousalHigh},
	}
	lex := buildPropLex(t, roots, words)
	n := propagate.PropagateExtended(lex)
	// Both words already have arousal and POS → nothing to update
	if n != 0 {
		t.Errorf("all words have POS and arousal: want 0 updates, got %d", n)
	}
}

func TestPropagateExtended_POSConsensus_Disagree(t *testing.T) {
	// When POS values differ, POS should NOT be propagated
	const posNoun = uint32(1 << 29)
	const posVerb = uint32(2 << 29)
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terror", Lang: "EN", Norm: "terror", Sentiment: negSent | posNoun},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "aterrar", Lang: "PT", Norm: "aterrar", Sentiment: negSent | posVerb},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "terror_es", Lang: "ES", Norm: "terror_es", Sentiment: negSent},
	}
	lex := buildPropLex(t, roots, words)
	propagate.PropagateExtended(lex)
	for i := range lex.Words {
		w := &lex.Words[i]
		if lex.WordStr(w) == "terror_es" {
			pos := (w.Sentiment >> 29) & 0x7
			if pos != 0 {
				t.Errorf("terror_es POS = %d, want 0 (POS should not propagate when cognates disagree)", pos)
			}
		}
	}
}

func TestPropagateExtended_ConcreteMajority(t *testing.T) {
	// Concreteness bit: majority concrete → propagated
	// Need arousal to trigger the extended propagation path
	const (
		concreteBit = uint32(1 << 28)
		arousalHigh = uint32(3 << 4)
	)
	roots := []seed.Root{
		{RootID: 1, RootStr: "petr", Origin: "LATIN", MeaningEN: "stone"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "stone", Lang: "EN", Norm: "stone", Sentiment: posSent | concreteBit | arousalHigh},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "pedra", Lang: "PT", Norm: "pedra", Sentiment: posSent | concreteBit | arousalHigh},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "piedra", Lang: "ES", Norm: "piedra", Sentiment: posSent}, // missing concrete + arousal
	}
	lex := buildPropLex(t, roots, words)
	propagate.PropagateExtended(lex)
	for i := range lex.Words {
		w := &lex.Words[i]
		if lex.WordStr(w) == "piedra" {
			concrete := (w.Sentiment >> 28) & 1
			if concrete != 1 {
				t.Errorf("piedra concreteness = %d, want 1 (majority are concrete)", concrete)
			}
		}
	}
}

func TestPropagateExtended_ConcreteMinority(t *testing.T) {
	// Concreteness bit: minority concrete → NOT propagated
	const concreteBit = uint32(1 << 28)
	roots := []seed.Root{
		{RootID: 1, RootStr: "liber", Origin: "LATIN", MeaningEN: "free"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "free", Lang: "EN", Norm: "free", Sentiment: posSent | concreteBit}, // concrete
		{WordID: 4098, RootID: 1, Variant: 2, Word: "libre", Lang: "ES", Norm: "libre", Sentiment: posSent},             // not concrete
		{WordID: 4099, RootID: 1, Variant: 3, Word: "livre", Lang: "PT", Norm: "livre", Sentiment: posSent},              // not concrete
		{WordID: 4100, RootID: 1, Variant: 4, Word: "frei", Lang: "DE", Norm: "frei", Sentiment: posSent},                // target, no concrete
	}
	lex := buildPropLex(t, roots, words)
	propagate.PropagateExtended(lex)
	for i := range lex.Words {
		w := &lex.Words[i]
		if lex.WordStr(w) == "frei" {
			concrete := (w.Sentiment >> 28) & 1
			if concrete != 0 {
				t.Errorf("frei concreteness = %d, want 0 (minority are concrete, should not propagate)", concrete)
			}
		}
	}
}

func TestPropagateExtended_RegisterOntologyPolysemy(t *testing.T) {
	// Test propagation of register, ontological, and polysemy flags
	const (
		regFormal  = uint32(1 << 8)  // register=FORMAL
		ontoPlace  = uint32(2 << 12) // ontological=PLACE
		polyHigh   = uint32(5 << 16) // polysemy=5
	)
	roots := []seed.Root{
		{RootID: 1, RootStr: "circ", Origin: "LATIN", MeaningEN: "around"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "circuit", Lang: "EN", Norm: "circuit", Sentiment: negSent, Flags: regFormal | ontoPlace | polyHigh},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "circuito", Lang: "PT", Norm: "circuito", Sentiment: negSent, Flags: 0}, // all missing
	}
	lex := buildPropLex(t, roots, words)
	n := propagate.PropagateExtended(lex)
	if n == 0 {
		t.Error("should have updated circuito flags")
	}
	for i := range lex.Words {
		w := &lex.Words[i]
		if lex.WordStr(w) == "circuito" {
			reg := (w.Flags >> 8) & 0xF
			if reg != 1 {
				t.Errorf("circuito register = %d, want 1 (FORMAL)", reg)
			}
			onto := (w.Flags >> 12) & 0xF
			if onto != 2 {
				t.Errorf("circuito ontological = %d, want 2 (PLACE)", onto)
			}
			poly := (w.Flags >> 16) & 0xF
			if poly != 5 {
				t.Errorf("circuito polysemy = %d, want 5", poly)
			}
		}
	}
}

func TestPropagateExtended_SkipsUnannotatedSentiment(t *testing.T) {
	// Words with Sentiment=0 should NOT receive extended propagation
	const arousalHigh = uint32(3 << 4)
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: negSent | arousalHigh},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: 0}, // completely unannotated
	}
	lex := buildPropLex(t, roots, words)
	propagate.PropagateExtended(lex)
	for i := range lex.Words {
		w := &lex.Words[i]
		if lex.WordStr(w) == "terrivel" || lex.WordStr(w) == "terrível" {
			if w.Sentiment != 0 {
				// PropagateExtended skips words with Sentiment==0
				// (those need Run() first to get base sentiment)
				t.Errorf("word with Sentiment=0 should not be modified by PropagateExtended, got 0x%X", w.Sentiment)
			}
		}
	}
}

func TestPropagateExtended_PhonologySyllablesAndStress(t *testing.T) {
	// First cognate with syllables/stress → propagated to others missing those fields
	const (
		syl4    = uint32(4 << 28)
		stress2 = uint32(2 << 26) // PENULTIMATE
	)
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: negSent, Flags: syl4 | stress2},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: negSent, Flags: 0},
	}
	lex := buildPropLex(t, roots, words)
	n := propagate.PropagateExtended(lex)
	if n == 0 {
		t.Error("should have propagated syllable/stress")
	}
	for i := range lex.Words {
		w := &lex.Words[i]
		if lex.WordStr(w) == "terrivel" || lex.WordStr(w) == "terrível" {
			syl := (w.Flags >> 28) & 0xF
			str := (w.Flags >> 26) & 0x3
			if syl != 4 {
				t.Errorf("syllable count = %d, want 4", syl)
			}
			if str != 2 {
				t.Errorf("stress = %d, want 2 (PENULTIMATE)", str)
			}
		}
	}
}

func TestPropagate_Run_CognatesFallbackPath(t *testing.T) {
	// When Cognates() via MakeWordID returns empty, Run falls back to
	// using root.FirstWordIdx directly. We trigger this by building a valid
	// lexicon then corrupting root.RootID so MakeWordID produces a wordID
	// that doesn't match any word, but FirstWordIdx still points correctly.
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: negSent},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)
	// Corrupt root's RootID so MakeWordID produces a word_id for a different root,
	// but FirstWordIdx and WordCount still point to the correct words.
	if len(lex.Roots) > 0 {
		lex.Roots[0].RootID = 999 // Cognates(MakeWordID(999,1)) will return empty
	}
	results := propagate.Run(lex)
	// The fallback path uses FirstWordIdx directly
	if len(results) != 1 {
		t.Fatalf("fallback cognate path: want 1 result, got %d", len(results))
	}
}

func TestPropagate_Run_BoundsCheckInWordIteration(t *testing.T) {
	// Test the i >= len(lex.Words) bounds check during word iteration
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: negSent},
	}
	lex := buildPropLex(t, roots, words)
	// Inflate WordCount beyond actual words to trigger bounds check
	if len(lex.Roots) > 0 {
		lex.Roots[0].WordCount = 100
	}
	// Should not panic, should handle bounds gracefully
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("panic on bounds overflow: %v", r)
			}
		}()
		propagate.Run(lex)
	}()
}

func TestPropagateExtended_BoundsOutOfRange(t *testing.T) {
	// Root with FirstWordIdx or WordCount pointing beyond Words slice
	// This tests the bounds check: start >= len(lex.Words) || end > len(lex.Words)
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: negSent},
	}
	lex := buildPropLex(t, roots, words)
	// Corrupt the root's FirstWordIdx to be out of range
	if len(lex.Roots) > 0 {
		lex.Roots[0].FirstWordIdx = uint32(len(lex.Words) + 10)
	}
	// Should not panic
	n := propagate.PropagateExtended(lex)
	if n != 0 {
		t.Errorf("out-of-range root: want 0 updates, got %d", n)
	}
}

func TestPropagateExtended_NoAnnotatedWords(t *testing.T) {
	// All words have Sentiment=0 → nothing to compute consensus from
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: 0},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: 0},
	}
	lex := buildPropLex(t, roots, words)
	n := propagate.PropagateExtended(lex)
	if n != 0 {
		t.Errorf("no annotated words: want 0 updates, got %d", n)
	}
}

func TestPropagateExtended_AllAlreadyHaveValues(t *testing.T) {
	// All cognates already have all extended fields → nothing to update
	const (
		arousalHigh = uint32(3 << 4)
		domHigh     = uint32(3 << 2)
		aoaLate     = uint32(3)
		posNoun     = uint32(1 << 29)
		concBit     = uint32(1 << 28)
		regFormal   = uint32(1 << 8)
		ontoPlace   = uint32(2 << 12)
		poly3       = uint32(3 << 16)
		syl3        = uint32(3 << 28)
	)
	sent := negSent | arousalHigh | domHigh | aoaLate | posNoun | concBit
	fl := regFormal | ontoPlace | poly3 | syl3
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: sent, Flags: fl},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "terrível", Lang: "PT", Norm: "terrivel", Sentiment: sent, Flags: fl},
	}
	lex := buildPropLex(t, roots, words)
	n := propagate.PropagateExtended(lex)
	if n != 0 {
		t.Errorf("all fields already set: want 0 updates, got %d", n)
	}
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
