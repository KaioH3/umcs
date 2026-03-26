package discover

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
)

// ── IsCJK / hasCJK ───────────────────────────────────────────────────────────

func TestIsCJK(t *testing.T) {
	cases := []struct {
		r    rune
		want bool
	}{
		{'愛', true},   // CJK Unified
		{'悲', true},   // CJK Unified
		{'あ', true},   // Hiragana
		{'ア', true},   // Katakana
		{'한', true},   // Hangul
		{'a', false},  // ASCII
		{'é', false},  // Latin diacritic
		{' ', false},  // space
		{'1', false},  // digit
	}
	for _, c := range cases {
		if got := IsCJK(c.r); got != c.want {
			t.Errorf("IsCJK(%q) = %v, want %v", c.r, got, c.want)
		}
	}
}

func TestHasCJK(t *testing.T) {
	if !hasCJK("愛情") {
		t.Error("expected hasCJK=true for 愛情")
	}
	if !hasCJK("love愛") {
		t.Error("expected hasCJK=true for mixed")
	}
	if hasCJK("amor") {
		t.Error("expected hasCJK=false for ASCII-only")
	}
	if hasCJK("") {
		t.Error("expected hasCJK=false for empty string")
	}
}

// ── PhoneticNorm ─────────────────────────────────────────────────────────────

func TestPhoneticNorm_Latin(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Négation", "negation"},
		{"São", "sao"},
		{"ÜBER", "uber"},
		{"ñoño", "nono"},
		{"Straße", "strasse"},
		{"café", "cafe"},
		{"Ünlü", "unlu"},
		{"", ""},
		{"abc", "abc"},
	}
	for _, c := range cases {
		if got := PhoneticNorm(c.in); got != c.want {
			t.Errorf("PhoneticNorm(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestPhoneticNorm_CJK(t *testing.T) {
	// CJK characters must be preserved unchanged.
	cases := []struct{ in, want string }{
		{"愛", "愛"},
		{"希望", "希望"},
		{" 悲 ", "悲"},   // trim spaces
		{"愛情", "愛情"},
	}
	for _, c := range cases {
		if got := PhoneticNorm(c.in); got != c.want {
			t.Errorf("PhoneticNorm(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestPhoneticNormMatchesNormalize verifies that for non-CJK strings,
// PhoneticNorm() and lexdb.Normalize() produce identical results.
// This guards the invariant that a word accepted during discovery will
// always be found by LookupWord() after building the lexicon.
func TestPhoneticNormMatchesNormalize(t *testing.T) {
	cases := []string{
		"café", "naïve", "über", "André", "Ångström",
		"Søren", "Ołówek", "Şeker", "Ţară", "Ðanish", "þorn",
		"Æsop", "Œuvre", "straße", "negativo", "TERRÍVEL",
		"résumé", "Čeština", "feliz", "amor", "NEGATIVE",
	}
	for _, s := range cases {
		got := PhoneticNorm(s)
		want := lexdb.Normalize(s)
		if got != want {
			t.Errorf("PhoneticNorm(%q)=%q != lexdb.Normalize(%q)=%q", s, got, s, want)
		}
	}
}

// ── LevenshteinSim ───────────────────────────────────────────────────────────

func TestLevenshteinSim(t *testing.T) {
	cases := []struct {
		a, b string
		min  float64
		max  float64
	}{
		{"negat", "negat", 1.0, 1.0},           // identical
		{"", "", 1.0, 1.0},                     // both empty
		{"", "abc", 0.0, 0.0},                  // one empty
		{"abc", "", 0.0, 0.0},                  // other empty
		{"amor", "amor", 1.0, 1.0},             // identical
		{"color", "dolor", 0.75, 0.85},         // similar but different
		{"negat", "negatif", 0.7, 0.85},        // prefix match
		{"abc", "xyz", 0.0, 0.4},               // very different
		{"feliz", "felix", 0.7, 0.9},           // near cognate
	}
	for _, c := range cases {
		sim := LevenshteinSim(c.a, c.b)
		if sim < c.min || sim > c.max {
			t.Errorf("LevenshteinSim(%q, %q) = %.3f, want [%.2f, %.2f]", c.a, c.b, sim, c.min, c.max)
		}
	}
}

func TestLevenshteinSim_Symmetric(t *testing.T) {
	pairs := [][2]string{
		{"color", "dolor"},
		{"amor", "amour"},
		{"negat", "negacion"},
	}
	for _, p := range pairs {
		ab := LevenshteinSim(p[0], p[1])
		ba := LevenshteinSim(p[1], p[0])
		if ab != ba {
			t.Errorf("LevenshteinSim not symmetric: (%q,%q)=%.4f vs (%q,%q)=%.4f",
				p[0], p[1], ab, p[1], p[0], ba)
		}
	}
}

// ── IsCognate ────────────────────────────────────────────────────────────────

func TestIsCognate(t *testing.T) {
	cases := []struct {
		a, langA, b, langB string
		want               bool
	}{
		{"amor", "PT", "amor", "ES", true},       // identical
		{"feliz", "PT", "felice", "IT", true},    // close cognates
		{"vida", "PT", "vita", "IT", true},       // near match
		{"amor", "PT", "hate", "EN", false},      // unrelated
		{"bom", "PT", "gut", "DE", false},        // unrelated
		{"lieben", "DE", "love", "EN", false},    // too different
		{"愛", "ZH", "愛", "JA", true},            // identical CJK
	}
	for _, c := range cases {
		got := IsCognate(c.a, c.langA, c.b, c.langB)
		if got != c.want {
			t.Errorf("IsCognate(%q,%q,%q,%q) = %v, want %v",
				c.a, c.langA, c.b, c.langB, got, c.want)
		}
	}
}

// ── StemAncestor ─────────────────────────────────────────────────────────────

func TestStemAncestor(t *testing.T) {
	cases := []struct{ in, want string }{
		{"negare", "neg"},        // Latin -are stripped
		{"negation", "neg"},      // -ation stripped
		{"bonus", "bon"},         // Latin -us stripped
		{"tristis", "trist"},     // Latin -is stripped
		{"fortis", "fort"},       // Latin -is stripped
		{"amor", "amor"},         // no suffix → unchanged
		{"愛", "愛"},              // CJK → unchanged (ideogram IS the morpheme)
		{"希望", "希望"},           // CJK compound → unchanged
		{"", ""},                 // empty
	}
	for _, c := range cases {
		got := StemAncestor(c.in)
		if got != c.want {
			t.Errorf("StemAncestor(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ── Assign ───────────────────────────────────────────────────────────────────

func TestAssign_ExactMatch(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat"},
		{RootID: 2, RootStr: "affirm"},
		{RootID: 3, RootStr: "bon"},
	}
	id, isNew := Assign("negat", roots)
	if id != 1 || isNew {
		t.Errorf("Assign exact: got id=%d isNew=%v, want id=1 isNew=false", id, isNew)
	}
}

func TestAssign_FuzzyMatch(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "affirm"},
		{RootID: 2, RootStr: "negat"},
	}
	// "affirms" (7 chars) vs "affirm" (6 chars): dist=1, maxLen=7, sim=0.857 >= 0.85
	id, isNew := Assign("affirms", roots)
	if isNew {
		t.Errorf("Assign fuzzy: expected match to affirm, got isNew=true (id=%d)", id)
	}
	if id != 1 {
		t.Errorf("Assign fuzzy: expected id=1 (affirm), got %d", id)
	}
}

func TestAssign_NoFalsePositive_ColorDolor(t *testing.T) {
	// "color" must NOT match "dolor" — they look similar (Levenshtein=0.80)
	// but the threshold is 0.85, so this should be a new root.
	roots := []seed.Root{
		{RootID: 38, RootStr: "dolor"},
	}
	_, isNew := Assign("color", roots)
	if !isNew {
		t.Error("Assign: 'color' falsely matched 'dolor' — threshold too low")
	}
}

func TestAssign_ShortStringSkipped(t *testing.T) {
	// Short roots (< 4 chars) must not participate in fuzzy matching.
	roots := []seed.Root{
		{RootID: 7, RootStr: "am"},   // len=2 → must not fuzzy match anything
		{RootID: 8, RootStr: "mal"},  // len=3 → must not fuzzy match
	}
	// "ama" is close to "am" but "am" is only 2 chars — must not match
	_, isNew := Assign("ama", roots)
	if !isNew {
		t.Error("Assign: short root 'am' (2 chars) participated in fuzzy match — should not")
	}
}

func TestAssign_NewRoot(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "negat"},
		{RootID: 5, RootStr: "fort"},
	}
	id, isNew := Assign("bibliothek", roots)
	if !isNew {
		t.Error("Assign: expected new root for unrelated word")
	}
	if id != 6 { // max+1 = 5+1
		t.Errorf("Assign: expected new id=6, got %d", id)
	}
}

func TestAssign_EmptyRoots(t *testing.T) {
	id, isNew := Assign("negat", nil)
	if !isNew {
		t.Error("Assign: expected new root when existing list is empty")
	}
	if id != 1 {
		t.Errorf("Assign: expected id=1 for first root, got %d", id)
	}
}

// ── NextVariant ───────────────────────────────────────────────────────────────

func TestNextVariant(t *testing.T) {
	words := []seed.Word{
		{RootID: 1, Variant: 1},
		{RootID: 1, Variant: 3},
		{RootID: 1, Variant: 2},
		{RootID: 2, Variant: 5},
	}
	if v := NextVariant(1, words); v != 4 {
		t.Errorf("NextVariant(1): got %d, want 4", v)
	}
	if v := NextVariant(2, words); v != 6 {
		t.Errorf("NextVariant(2): got %d, want 6", v)
	}
	if v := NextVariant(99, words); v != 1 {
		t.Errorf("NextVariant(99 unknown): got %d, want 1", v)
	}
}

// ── isValidWord ───────────────────────────────────────────────────────────────

func TestIsValidWord(t *testing.T) {
	valid := []string{"hello", "café", "über", "愛", "悲", "negativo", "un-happy", "希望"}
	for _, w := range valid {
		if !isValidWord(w) {
			t.Errorf("isValidWord(%q) = false, want true", w)
		}
	}

	invalid := []string{
		"[[dar]] [[crédito]]", // wikitext link
		"{{template}}",        // wikitext template
		"en dehors de",        // multi-word phrase
		"faire un pet",        // multi-word phrase
		"a",                   // single char (too short)
		"",                    // empty
	}
	for _, w := range invalid {
		if isValidWord(w) {
			t.Errorf("isValidWord(%q) = true, want false", w)
		}
	}
}

// ── ScoreViaPropagation ───────────────────────────────────────────────────────

func makeSeedWord(rootID uint32, polarity, intensity string) seed.Word {
	pol := map[string]uint32{
		"POSITIVE": sentiment.PolarityPositive,
		"NEGATIVE": sentiment.PolarityNegative,
		"NEUTRAL":  sentiment.PolarityNeutral,
	}[polarity]
	inten := map[string]uint32{
		"WEAK":     sentiment.IntensityWeak,
		"MODERATE": sentiment.IntensityModerate,
		"STRONG":   sentiment.IntensityStrong,
		"EXTREME":  sentiment.IntensityExtreme,
		"NONE":     sentiment.IntensityNone,
	}[intensity]
	return seed.Word{
		RootID:    rootID,
		Sentiment: pol | inten | sentiment.RoleEvaluation | sentiment.DomainGeneral,
	}
}

func TestScoreViaPropagation_Unanimous(t *testing.T) {
	words := []seed.Word{
		makeSeedWord(1, "POSITIVE", "STRONG"),
		makeSeedWord(1, "POSITIVE", "STRONG"),
		makeSeedWord(1, "POSITIVE", "MODERATE"),
	}
	s := ScoreViaPropagation(1, words)
	if s.Polarity != "POSITIVE" {
		t.Errorf("expected POSITIVE, got %s", s.Polarity)
	}
	if s.Confidence != 1.0 {
		t.Errorf("unanimous vote should have confidence=1.0, got %.2f", s.Confidence)
	}
	if s.Source != "propagation" {
		t.Errorf("expected source=propagation, got %s", s.Source)
	}
}

func TestScoreViaPropagation_Majority(t *testing.T) {
	words := []seed.Word{
		makeSeedWord(2, "NEGATIVE", "STRONG"),
		makeSeedWord(2, "NEGATIVE", "STRONG"),
		makeSeedWord(2, "POSITIVE", "WEAK"),
	}
	s := ScoreViaPropagation(2, words)
	if s.Polarity != "NEGATIVE" {
		t.Errorf("majority should be NEGATIVE, got %s", s.Polarity)
	}
	want := 2.0 / 3.0
	if s.Confidence < want-0.01 || s.Confidence > want+0.01 {
		t.Errorf("confidence: want %.4f, got %.4f", want, s.Confidence)
	}
}

func TestScoreViaPropagation_NoWords(t *testing.T) {
	s := ScoreViaPropagation(999, nil)
	if s.Confidence != 0 {
		t.Errorf("no words: expected confidence=0, got %.2f", s.Confidence)
	}
}

func TestScoreViaPropagation_WrongRoot(t *testing.T) {
	words := []seed.Word{
		makeSeedWord(1, "POSITIVE", "STRONG"),
		makeSeedWord(1, "POSITIVE", "STRONG"),
	}
	s := ScoreViaPropagation(99, words) // different root_id
	if s.Confidence != 0 {
		t.Errorf("wrong root: expected confidence=0, got %.2f", s.Confidence)
	}
}

// ── ScoreViaDefinition ────────────────────────────────────────────────────────

func TestScoreViaDefinition_Positive(t *testing.T) {
	defs := []string{"Having great beauty and love", "wonderful and kind"}
	s := ScoreViaDefinition(defs)
	if s.Polarity != "POSITIVE" {
		t.Errorf("expected POSITIVE, got %s", s.Polarity)
	}
	if s.Confidence <= 0 {
		t.Errorf("expected positive confidence, got %.2f", s.Confidence)
	}
}

func TestScoreViaDefinition_Negative(t *testing.T) {
	defs := []string{"causing great pain and fear", "evil and cruel behaviour"}
	s := ScoreViaDefinition(defs)
	if s.Polarity != "NEGATIVE" {
		t.Errorf("expected NEGATIVE, got %s", s.Polarity)
	}
}

func TestScoreViaDefinition_Neutral(t *testing.T) {
	defs := []string{"a round object", "used for measurement"}
	s := ScoreViaDefinition(defs)
	// no strong positive/negative keywords → neutral
	if s.Confidence > 0.3 {
		t.Errorf("neutral definition should have low confidence, got %.2f", s.Confidence)
	}
}

func TestScoreViaDefinition_Empty(t *testing.T) {
	s := ScoreViaDefinition(nil)
	if s.Confidence < 0 {
		t.Error("empty defs: confidence should be >= 0")
	}
}

// ── ScoreViaMorphology ────────────────────────────────────────────────────────

func TestScoreViaMorphology_NegativePrefix(t *testing.T) {
	cases := []string{"unhappy", "dislike", "immoral", "illegal", "irresponsible", "misplace", "nonfiction", "desamor", "antiwar", "malheur"}
	for _, w := range cases {
		s := ScoreViaMorphology(w)
		if s.Polarity != "NEGATIVE" {
			t.Errorf("ScoreViaMorphology(%q): expected NEGATIVE, got %s", w, s.Polarity)
		}
	}
}

func TestScoreViaMorphology_Diminutive(t *testing.T) {
	cases := []string{"cãozinho", "mädchen", "amiguito", "maisonnette", "booklet"}
	for _, w := range cases {
		s := ScoreViaMorphology(w)
		if s.Intensity != "WEAK" {
			t.Errorf("ScoreViaMorphology(%q): expected WEAK intensity, got %s", w, s.Intensity)
		}
	}
}

func TestScoreViaMorphology_Superlative(t *testing.T) {
	cases := []string{"bellissimo", "grandissimo"}
	for _, w := range cases {
		s := ScoreViaMorphology(w)
		if s.Intensity != "EXTREME" {
			t.Errorf("ScoreViaMorphology(%q): expected EXTREME, got %s", w, s.Intensity)
		}
	}
}

func TestScoreViaMorphology_Neutral(t *testing.T) {
	s := ScoreViaMorphology("table")
	if s.Polarity != "NEUTRAL" {
		t.Errorf("expected NEUTRAL for 'table', got %s", s.Polarity)
	}
}

// ── BestScore ─────────────────────────────────────────────────────────────────

func TestBestScore(t *testing.T) {
	low := Score{Polarity: "NEUTRAL", Confidence: 0.1, Source: "morphology"}
	mid := Score{Polarity: "POSITIVE", Confidence: 0.5, Source: "definition"}
	high := Score{Polarity: "NEGATIVE", Confidence: 0.9, Source: "propagation"}

	best := BestScore(low, mid, high)
	if best.Polarity != "NEGATIVE" || best.Confidence != 0.9 {
		t.Errorf("BestScore: expected NEGATIVE/0.9, got %s/%.1f", best.Polarity, best.Confidence)
	}

	// Single score
	s := BestScore(mid)
	if s.Polarity != "POSITIVE" {
		t.Errorf("BestScore single: expected POSITIVE, got %s", s.Polarity)
	}

	// No scores
	empty := BestScore()
	if empty.Confidence < 0 {
		t.Errorf("BestScore empty: confidence should be >= 0 (fallback)")
	}
}

// ── ParseDumpPage ─────────────────────────────────────────────────────────────

var sampleWikitext = `
==English==
===Etymology===
From Latin ''{{m|la|amō}}'', to love.
===Verb===
# {{lb|en|transitive}} To have great affection for.
# {{lb|en|intransitive}} To feel love.

====Translations====
{{trans-top|to have affection for}}
* Portuguese: {{t+|pt|amar}}
* Spanish: {{t+|es|amar}}
* French: {{t+|fr|aimer}}
* Italian: {{t+|it|amare}}
* German: {{t+|de|lieben}}
{{trans-bottom}}

==Portuguese==
===Etymology===
From Latin ''amāre''.
===Verb===
# To love.
`

func TestParseDumpPage_English(t *testing.T) {
	page := WikiPage{Title: "love", Text: sampleWikitext}
	entries := ParseDumpPage(page, []string{"EN", "PT", "ES", "FR", "IT", "DE"})
	if len(entries) == 0 {
		t.Fatal("ParseDumpPage returned no entries")
	}

	// Should have an English entry with translations
	var enEntry *Entry
	for i := range entries {
		if entries[i].Lang == "EN" {
			enEntry = &entries[i]
			break
		}
	}
	if enEntry == nil {
		t.Fatal("no English entry found")
	}
	if len(enEntry.Translations) == 0 {
		t.Error("English entry should have translations from the Translations table")
	}
}

func TestParseDumpPage_EmptyText(t *testing.T) {
	page := WikiPage{Title: "test", Text: ""}
	entries := ParseDumpPage(page, []string{"EN", "PT"})
	if len(entries) != 0 {
		t.Errorf("expected 0 entries for empty text, got %d", len(entries))
	}
}

func TestParseDumpPage_UnknownLang(t *testing.T) {
	page := WikiPage{Title: "test", Text: sampleWikitext}
	// "XX" is not a valid lang code — should be silently skipped
	entries := ParseDumpPage(page, []string{"XX"})
	// Only English section is always parsed; XX section won't exist in wikitext
	_ = entries // must not panic
}

// ── ScoreViaDefinition — negation and intensifier ─────────────────────────────

func TestScoreViaDefinition_NegationFlips(t *testing.T) {
	// "not good" should read as negative, not positive
	s := ScoreViaDefinition([]string{"not good and not kind"})
	if s.Polarity == "POSITIVE" {
		t.Errorf("negated positive indicators should not yield POSITIVE, got %s conf=%.2f", s.Polarity, s.Confidence)
	}
}

func TestScoreViaDefinition_StrongIndicatorsReachThreshold(t *testing.T) {
	// Two strong positive indicators should yield confidence >= 0.60
	s := ScoreViaDefinition([]string{"feeling of wonderful and magnificent joy"})
	if s.Polarity != "POSITIVE" {
		t.Errorf("expected POSITIVE, got %s", s.Polarity)
	}
	if s.Confidence < 0.60 {
		t.Errorf("two strong indicators should reach threshold: got conf=%.2f", s.Confidence)
	}
}

func TestScoreViaDefinition_IntensifierRaisesConfidence(t *testing.T) {
	base := ScoreViaDefinition([]string{"good feeling"})
	intensified := ScoreViaDefinition([]string{"extremely good feeling"})
	if intensified.Confidence <= base.Confidence {
		t.Errorf("intensified (%.2f) should exceed base (%.2f)", intensified.Confidence, base.Confidence)
	}
}

func TestScoreViaDefinition_MixedSignalLowConfidence(t *testing.T) {
	// Balanced pos/neg → confidence should be low (signal is ambiguous)
	s := ScoreViaDefinition([]string{"wonderful but also terrible, a mix of good and bad"})
	if s.Confidence > 0.45 {
		t.Errorf("ambiguous definition should stay below 0.45, got %.2f", s.Confidence)
	}
}

// ── SenseCoherent ─────────────────────────────────────────────────────────────

func TestSenseCoherent_MatchingMeaning(t *testing.T) {
	// "gratitude" definition mentions "grateful" — overlaps with root meaning "grateful or pleasant"
	if !SenseCoherent([]string{"the quality of being thankful and grateful"}, "grateful or pleasant") {
		t.Error("overlapping definition should be coherent")
	}
}

func TestSenseCoherent_IncoherentPolysemy(t *testing.T) {
	// EN "gut" (intestine) definition vs root meaning "good"
	if SenseCoherent(
		[]string{"the intestinal tract; the stomach and bowels"},
		"good or pleasant (from Proto-Germanic)",
	) {
		t.Error("intestine definition should be incoherent with 'good' root meaning")
	}
}

func TestSenseCoherent_ShortDefLenient(t *testing.T) {
	// Short definitions get benefit of the doubt
	if !SenseCoherent([]string{"a type of food"}, "good or pleasant") {
		t.Error("short definition should be treated as coherent (lenient)")
	}
}

func TestSenseCoherent_EmptyRootMeaning(t *testing.T) {
	if !SenseCoherent([]string{"anything at all"}, "") {
		t.Error("empty root meaning should be treated as coherent")
	}
}

// ── StagedWriter ──────────────────────────────────────────────────────────────

func TestStagedWriter_Dedup(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "staged.csv")

	w := NewStagedWriter(path)
	words := []StagedWord{
		{Word: "hello", Lang: "EN", RootStr: "test", Score: Score{Polarity: "POSITIVE", Confidence: 0.3}},
		{Word: "hello", Lang: "EN", RootStr: "test", Score: Score{Polarity: "POSITIVE", Confidence: 0.3}}, // duplicate
		{Word: "mundo", Lang: "PT", RootStr: "test", Score: Score{Polarity: "NEUTRAL", Confidence: 0.2}},
	}
	if err := w.Write(words); err != nil {
		t.Fatalf("write: %v", err)
	}

	// Second writer reads same file — should not re-add existing entries.
	w2 := NewStagedWriter(path)
	if err := w2.Write(words); err != nil {
		t.Fatalf("write2: %v", err)
	}

	// File should have header + 2 unique rows (hello+mundo), not 4+ rows.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 { // header + 2 words
		t.Errorf("expected 3 lines (header+2 words), got %d:\n%s", len(lines), string(data))
	}
}
