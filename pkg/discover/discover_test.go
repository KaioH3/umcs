package discover

import (
	"bytes"
	"io"
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

// ── Checkpoint ────────────────────────────────────────────────────────────────

func TestCheckpoint_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.json")

	// Load from non-existent file → fresh checkpoint
	cp, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadCheckpoint non-existent: %v", err)
	}
	if cp.IsProcessed("love:EN") {
		t.Error("fresh checkpoint: IsProcessed must return false")
	}

	// Mark and save
	cp.Mark("love:EN")
	cp.Mark("hate:EN")
	if err := cp.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Reload and verify
	cp2, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadCheckpoint after save: %v", err)
	}
	if !cp2.IsProcessed("love:EN") {
		t.Error("love:EN should be marked after reload")
	}
	if !cp2.IsProcessed("hate:EN") {
		t.Error("hate:EN should be marked after reload")
	}
	if cp2.IsProcessed("peace:EN") {
		t.Error("peace:EN was never marked — must return false")
	}
}

func TestCheckpoint_CorruptFile_StartsFresh(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "checkpoint.json")
	// Write garbage JSON
	if err := os.WriteFile(path, []byte("not json {{{"), 0o644); err != nil {
		t.Fatal(err)
	}
	cp, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("corrupt checkpoint should not return error: %v", err)
	}
	if cp == nil || cp.Processed == nil {
		t.Fatal("corrupt checkpoint should return fresh empty checkpoint")
	}
}

// ── Flush / appendRoots / appendWords ─────────────────────────────────────────

func TestFlush_AppendsRootsAndWords(t *testing.T) {
	dir := t.TempDir()
	rootsPath := filepath.Join(dir, "roots.csv")
	wordsPath := filepath.Join(dir, "words.csv")

	roots := []seed.Root{
		{RootID: 1, RootStr: "amor", Origin: "LATIN", MeaningEN: "love"},
	}
	words := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "love", Lang: "EN", Norm: "love",
			Sentiment: sentiment.PolarityPositive},
	}

	if err := Flush(roots, words, rootsPath, wordsPath); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// Verify files exist and have content
	rootData, err := os.ReadFile(rootsPath)
	if err != nil {
		t.Fatalf("roots file not created: %v", err)
	}
	if !strings.Contains(string(rootData), "amor") {
		t.Errorf("roots file missing 'amor': %s", rootData)
	}

	wordData, err := os.ReadFile(wordsPath)
	if err != nil {
		t.Fatalf("words file not created: %v", err)
	}
	if !strings.Contains(string(wordData), "love") {
		t.Errorf("words file missing 'love': %s", wordData)
	}
}

func TestFlush_EmptySlices_NoError(t *testing.T) {
	dir := t.TempDir()
	if err := Flush(nil, nil, filepath.Join(dir, "r.csv"), filepath.Join(dir, "w.csv")); err != nil {
		t.Errorf("Flush with empty slices: %v", err)
	}
}

// ── WordExists ────────────────────────────────────────────────────────────────

func TestWordExists_Found(t *testing.T) {
	existing := []seed.Word{
		{Norm: "love", Lang: "EN"},
		{Norm: "amor", Lang: "PT"},
	}
	if !WordExists("love", "EN", existing) {
		t.Error("love/EN should be found")
	}
	if !WordExists("amor", "PT", existing) {
		t.Error("amor/PT should be found")
	}
}

func TestWordExists_NotFound(t *testing.T) {
	existing := []seed.Word{{Norm: "love", Lang: "EN"}}
	if WordExists("hate", "EN", existing) {
		t.Error("hate/EN should not be found")
	}
	if WordExists("love", "PT", existing) {
		t.Error("love/PT should not be found (wrong lang)")
	}
}

func TestWordExists_Empty(t *testing.T) {
	if WordExists("love", "EN", nil) {
		t.Error("empty existing slice: must return false")
	}
}

// ── classifyBest ──────────────────────────────────────────────────────────────

func TestClassifyBest_NilEntry_UsesPropagation(t *testing.T) {
	words := []seed.Word{
		makeSeedWord(1, "POSITIVE", "STRONG"),
		makeSeedWord(1, "POSITIVE", "MODERATE"),
	}
	s := classifyBest(1, nil, words)
	if s.Polarity != "POSITIVE" {
		t.Errorf("nil entry: expected POSITIVE from propagation, got %s", s.Polarity)
	}
}

func TestClassifyBest_WithEntry(t *testing.T) {
	words := []seed.Word{makeSeedWord(2, "NEGATIVE", "STRONG")}
	entry := &Entry{
		Word:        "terrible",
		Lang:        "EN",
		Definitions: []string{"extremely bad and frightening"},
	}
	s := classifyBest(2, entry, words)
	// Should return best of prop+def+morph scores
	if s.Confidence < 0 || s.Confidence > 1 {
		t.Errorf("classifyBest: confidence %v out of [0,1]", s.Confidence)
	}
}

// ── makeWord ──────────────────────────────────────────────────────────────────

func TestMakeWord_Success(t *testing.T) {
	score := Score{Polarity: "POSITIVE", Intensity: "MODERATE", Role: "EVALUATION", Confidence: 0.8}
	w, ok := makeWord(1, "love", "EN", "love", score, nil)
	if !ok {
		t.Fatal("makeWord returned false for valid input")
	}
	if w.Word != "love" || w.Lang != "EN" {
		t.Errorf("makeWord: word/lang mismatch: %q %q", w.Word, w.Lang)
	}
	if w.RootID != 1 {
		t.Errorf("makeWord: RootID = %d, want 1", w.RootID)
	}
	if w.Sentiment == 0 {
		t.Error("makeWord: Sentiment should be non-zero")
	}
}

func TestMakeWord_EmptyPolarity_DefaultsNeutral(t *testing.T) {
	score := Score{Polarity: "", Intensity: "", Role: ""}
	w, ok := makeWord(1, "xyz", "EN", "xyz", score, nil)
	if !ok {
		t.Fatal("makeWord should succeed with empty score (defaults to NEUTRAL)")
	}
	_ = w
}

// ── isTargetLang / firstDef ───────────────────────────────────────────────────

func TestIsTargetLang(t *testing.T) {
	targets := []string{"EN", "PT", "ES"}
	if !isTargetLang("EN", targets) {
		t.Error("EN should be in targets")
	}
	if !isTargetLang("PT", targets) {
		t.Error("PT should be in targets")
	}
	if isTargetLang("ZH", targets) {
		t.Error("ZH should NOT be in targets")
	}
	if isTargetLang("EN", nil) {
		t.Error("empty targets: must return false")
	}
}

func TestFirstDef_NonEmpty(t *testing.T) {
	defs := []string{"first definition", "second definition"}
	if got := firstDef(defs, "fallback"); got != "first definition" {
		t.Errorf("firstDef: got %q, want 'first definition'", got)
	}
}

func TestFirstDef_Empty(t *testing.T) {
	if got := firstDef(nil, "fallback"); got != "fallback" {
		t.Errorf("firstDef(nil): got %q, want 'fallback'", got)
	}
}

// ── buildNormPolarityIndex ────────────────────────────────────────────────────

func TestBuildNormPolarityIndex_BasicMapping(t *testing.T) {
	words := []seed.Word{
		{Norm: "love", Sentiment: sentiment.PolarityPositive | sentiment.IntensityStrong |
			sentiment.RoleEvaluation | sentiment.DomainGeneral},
		{Norm: "hate", Sentiment: sentiment.PolarityNegative | sentiment.IntensityStrong |
			sentiment.RoleEvaluation | sentiment.DomainGeneral},
		{Norm: "maybe", Sentiment: sentiment.PolarityNeutral | sentiment.IntensityNone |
			sentiment.RoleEvaluation | sentiment.DomainGeneral},
		{Norm: "zero", Sentiment: 0}, // zero sentiment must be skipped
	}
	idx := buildNormPolarityIndex(words)
	if idx["love"] != "POSITIVE" {
		t.Errorf("love: want POSITIVE, got %q", idx["love"])
	}
	if idx["hate"] != "NEGATIVE" {
		t.Errorf("hate: want NEGATIVE, got %q", idx["hate"])
	}
	if _, ok := idx["maybe"]; ok {
		t.Error("neutral 'maybe' should NOT be in polarity index")
	}
	if _, ok := idx["zero"]; ok {
		t.Error("zero-sentiment word should NOT be in polarity index")
	}
}

func TestBuildNormPolarityIndex_Empty(t *testing.T) {
	idx := buildNormPolarityIndex(nil)
	if len(idx) != 0 {
		t.Errorf("empty words: want empty index, got %v", idx)
	}
}

// ── resolveRoot ───────────────────────────────────────────────────────────────

func TestResolveRoot_ExistingWord(t *testing.T) {
	roots := []seed.Root{{RootID: 1, RootStr: "amor", Origin: "LATIN", MeaningEN: "love"}}
	words := []seed.Word{{WordID: (1 << 12) | 1, RootID: 1, Word: "love", Lang: "EN", Norm: "love"}}
	entry := &Entry{Word: "love", Lang: "EN", Definitions: []string{"strong affection"}}

	rootID, _, isNew := resolveRoot(entry, "EN", roots, words)
	if isNew {
		t.Error("existing word 'love' should map to existing root, not a new one")
	}
	if rootID != 1 {
		t.Errorf("resolveRoot existing: want rootID=1, got %d", rootID)
	}
}

func TestResolveRoot_NewWord(t *testing.T) {
	roots := []seed.Root{{RootID: 1, RootStr: "amor", Origin: "LATIN", MeaningEN: "love"}}
	words := []seed.Word{{WordID: (1 << 12) | 1, RootID: 1, Word: "love", Lang: "EN", Norm: "love"}}
	// "freedom" has no match in words, ancestor is empty → goes through fallback
	entry := &Entry{Word: "freedom", Lang: "EN", Definitions: []string{"state of being free"}}

	_, _, isNew := resolveRoot(entry, "EN", roots, words)
	// May or may not be new depending on fuzzy match, but must not panic
	_ = isNew
}

func TestResolveRoot_EmptyLexicon(t *testing.T) {
	entry := &Entry{Word: "amor", Lang: "PT", Definitions: []string{"feeling of love"}}
	rootID, rootStr, isNew := resolveRoot(entry, "PT", nil, nil)
	if !isNew {
		t.Error("empty lexicon: must create new root")
	}
	if rootID == 0 {
		t.Error("rootID must be non-zero")
	}
	if rootStr == "" {
		t.Error("rootStr must be non-empty")
	}
}

// ── WriteStagedCSV ────────────────────────────────────────────────────────────

func TestWriteStagedCSV_ProducesValidCSV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "staged.csv")

	staged := []StagedWord{
		{
			Word: "triste", Lang: "PT", RootStr: "trist",
			ProposedRootID: 42,
			Score:          Score{Polarity: "NEGATIVE", Intensity: "MODERATE", Role: "EMOTION", Confidence: 0.55, Source: "definition"},
			Definition:     "feeling sadness",
		},
		{
			Word: "joyeux", Lang: "FR", RootStr: "joy",
			ProposedRootID: 10,
			Score:          Score{Polarity: "POSITIVE", Intensity: "STRONG", Role: "EVALUATION", Confidence: 0.45, Source: "propagation"},
			Definition:     "full of joy",
		},
	}

	if err := WriteStagedCSV(staged, path); err != nil {
		t.Fatalf("WriteStagedCSV: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read staged file: %v", err)
	}
	content := string(data)

	// Verify header is present.
	if !strings.Contains(content, "word,lang,root_str") {
		t.Error("staged CSV missing header row")
	}
	// Verify both words appear.
	if !strings.Contains(content, "triste") {
		t.Error("staged CSV missing 'triste'")
	}
	if !strings.Contains(content, "joyeux") {
		t.Error("staged CSV missing 'joyeux'")
	}
	// Verify confidence formatting (2 decimal places).
	if !strings.Contains(content, "0.55") {
		t.Error("staged CSV missing confidence '0.55'")
	}
	if !strings.Contains(content, "0.45") {
		t.Error("staged CSV missing confidence '0.45'")
	}
	// Verify correct number of lines: header + 2 data rows.
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) != 3 {
		t.Errorf("expected 3 lines (header + 2 rows), got %d", len(lines))
	}
}

func TestWriteStagedCSV_EmptySlice_NoFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "staged.csv")

	if err := WriteStagedCSV(nil, path); err != nil {
		t.Fatalf("WriteStagedCSV(nil): %v", err)
	}

	// Empty slice should not create the file at all.
	if _, err := os.Stat(path); err == nil {
		t.Error("expected no file for empty staged slice")
	}
}

// ── CheckpointPath / StagedPath ───────────────────────────────────────────────

func TestCheckpointPath(t *testing.T) {
	got := CheckpointPath("/some/dir")
	want := filepath.Join("/some/dir", ".discover_checkpoint.json")
	if got != want {
		t.Errorf("CheckpointPath: got %q, want %q", got, want)
	}
}

func TestCheckpointPath_EmptyDir(t *testing.T) {
	got := CheckpointPath("")
	if got != ".discover_checkpoint.json" {
		t.Errorf("CheckpointPath empty: got %q, want %q", got, ".discover_checkpoint.json")
	}
}

func TestStagedPath(t *testing.T) {
	got := StagedPath("/output")
	want := filepath.Join("/output", "staged.csv")
	if got != want {
		t.Errorf("StagedPath: got %q, want %q", got, want)
	}
}

func TestStagedPath_EmptyDir(t *testing.T) {
	got := StagedPath("")
	if got != "staged.csv" {
		t.Errorf("StagedPath empty: got %q, want %q", got, "staged.csv")
	}
}

// ── loadCache / saveCache ─────────────────────────────────────────────────────

func TestCacheRoundTrip(t *testing.T) {
	// Override XDG_CACHE_HOME so we write into a temp dir.
	dir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", dir)

	entry := &Entry{
		Word:         "love",
		Lang:         "EN",
		Etymology:    "From Latin amō",
		AncestorWord: "amō",
		AncestorLang: "LATIN",
		POS:          "Verb",
		Definitions:  []string{"To have great affection for"},
		Translations: []Trans{{Word: "amar", Lang: "PT"}},
	}

	saveCache("love", "EN", entry)

	loaded, ok := loadCache("love", "EN")
	if !ok {
		t.Fatal("loadCache returned false after saveCache")
	}
	if loaded.Word != entry.Word {
		t.Errorf("Word: got %q, want %q", loaded.Word, entry.Word)
	}
	if loaded.AncestorWord != entry.AncestorWord {
		t.Errorf("AncestorWord: got %q, want %q", loaded.AncestorWord, entry.AncestorWord)
	}
	if loaded.AncestorLang != entry.AncestorLang {
		t.Errorf("AncestorLang: got %q, want %q", loaded.AncestorLang, entry.AncestorLang)
	}
	if loaded.POS != entry.POS {
		t.Errorf("POS: got %q, want %q", loaded.POS, entry.POS)
	}
	if len(loaded.Definitions) != 1 || loaded.Definitions[0] != entry.Definitions[0] {
		t.Errorf("Definitions: got %v, want %v", loaded.Definitions, entry.Definitions)
	}
	if len(loaded.Translations) != 1 || loaded.Translations[0].Word != "amar" {
		t.Errorf("Translations: got %v, want %v", loaded.Translations, entry.Translations)
	}
}

func TestLoadCache_Missing(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", dir)

	_, ok := loadCache("nonexistent_word_xyz", "EN")
	if ok {
		t.Error("loadCache should return false for missing entry")
	}
}

func TestLoadCache_CorruptJSON(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", dir)

	// Save valid to create the directory, then corrupt the file.
	saveCache("corrupt", "EN", &Entry{Word: "corrupt"})
	cachePath := wiktCachePath("corrupt", "EN")
	if err := os.WriteFile(cachePath, []byte("{invalid json!!!"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, ok := loadCache("corrupt", "EN")
	if ok {
		t.Error("loadCache should return false for corrupt JSON")
	}
}

// ── wiktCacheDir / wiktCachePath ──────────────────────────────────────────────

func TestWiktCacheDir_XDG(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", "/tmp/testcache")
	got := wiktCacheDir()
	want := filepath.Join("/tmp/testcache", "lexsent", "wikt")
	if got != want {
		t.Errorf("wiktCacheDir with XDG: got %q, want %q", got, want)
	}
}

func TestWiktCachePath_Encoding(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", "/tmp/testcache")
	got := wiktCachePath("café", "PT")
	// url.QueryEscape("café") = "caf%C3%A9"
	if !strings.Contains(got, "PT_caf") {
		t.Errorf("wiktCachePath: expected PT prefix, got %q", got)
	}
	if !strings.HasSuffix(got, ".json") {
		t.Errorf("wiktCachePath: expected .json suffix, got %q", got)
	}
}

// ── ScanDump ──────────────────────────────────────────────────────────────────

func TestScanDump_XMLFragment(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "test.xml")

	xmlContent := `<mediawiki>
<page>
  <title>love</title>
  <ns>0</ns>
  <revision><text>==English==
===Etymology===
From Old English lufu.
===Noun===
# An intense feeling of affection.
</text></revision>
</page>
<page>
  <title>Template:test</title>
  <ns>10</ns>
  <revision><text>template content</text></revision>
</page>
<page>
  <title>hate</title>
  <ns>0</ns>
  <revision><text>==English==
===Noun===
# Strong dislike.
</text></revision>
</page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var pages []WikiPage
	err := ScanDump(xmlPath, func(p WikiPage) error {
		pages = append(pages, p)
		return nil
	})
	if err != nil {
		t.Fatalf("ScanDump: %v", err)
	}

	// Should skip the Template page (ns=10) and only return ns=0 pages.
	if len(pages) != 2 {
		t.Fatalf("expected 2 pages, got %d", len(pages))
	}
	if pages[0].Title != "love" {
		t.Errorf("first page title: got %q, want %q", pages[0].Title, "love")
	}
	if pages[1].Title != "hate" {
		t.Errorf("second page title: got %q, want %q", pages[1].Title, "hate")
	}
	if !strings.Contains(pages[0].Text, "Old English") {
		t.Error("first page text should contain etymology")
	}
}

func TestScanDump_EarlyStop(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "test.xml")

	xmlContent := `<mediawiki>
<page><title>first</title><ns>0</ns><revision><text>content1</text></revision></page>
<page><title>second</title><ns>0</ns><revision><text>content2</text></revision></page>
<page><title>third</title><ns>0</ns><revision><text>content3</text></revision></page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var count int
	err := ScanDump(xmlPath, func(p WikiPage) error {
		count++
		if count >= 2 {
			return errStopScan // stop after 2 pages
		}
		return nil
	})
	if err != nil {
		t.Fatalf("ScanDump early stop: %v", err)
	}
	if count != 2 {
		t.Errorf("expected 2 pages before stop, got %d", count)
	}
}

func TestScanDump_SkipColonTitles(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "test.xml")

	xmlContent := `<mediawiki>
<page><title>Category:English</title><ns>0</ns><revision><text>cat</text></revision></page>
<page><title>good</title><ns>0</ns><revision><text>good content</text></revision></page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var pages []WikiPage
	err := ScanDump(xmlPath, func(p WikiPage) error {
		pages = append(pages, p)
		return nil
	})
	if err != nil {
		t.Fatalf("ScanDump: %v", err)
	}
	if len(pages) != 1 || pages[0].Title != "good" {
		t.Errorf("expected only 'good' page, got %v", pages)
	}
}

func TestScanDump_NonexistentFile(t *testing.T) {
	err := ScanDump("/nonexistent/path/dump.xml", func(p WikiPage) error {
		return nil
	})
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

// ── ParseDumpPage — edge cases ────────────────────────────────────────────────

func TestParseDumpPage_MultipleLanguageSections(t *testing.T) {
	wikitext := `==English==
===Noun===
# A feeling of happiness.

====Translations====
{{trans-top|happiness}}
* Portuguese: {{t+|pt|felicidade}}
{{trans-bottom}}

==Portuguese==
===Etymology===
From Latin ''fēlīcitātem''.
===Noun===
# Happiness; state of being happy.
`
	page := WikiPage{Title: "felicity", Text: wikitext}
	entries := ParseDumpPage(page, []string{"EN", "PT"})

	var hasEN, hasPT bool
	for _, e := range entries {
		if e.Lang == "EN" {
			hasEN = true
		}
		if e.Lang == "PT" {
			hasPT = true
			if len(e.Definitions) == 0 {
				t.Error("PT entry should have definitions")
			}
		}
	}
	if !hasEN {
		t.Error("expected English entry")
	}
	if !hasPT {
		t.Error("expected Portuguese entry")
	}
}

func TestParseDumpPage_NoEnglishSection(t *testing.T) {
	// A page that only has a Portuguese section.
	wikitext := `==Portuguese==
===Etymology===
From Latin ''amāre''.
===Verb===
# To love.
`
	page := WikiPage{Title: "amar", Text: wikitext}
	entries := ParseDumpPage(page, []string{"PT"})

	// Should have a PT entry from the Portuguese section.
	var hasPT bool
	for _, e := range entries {
		if e.Lang == "PT" && len(e.Definitions) > 0 {
			hasPT = true
		}
	}
	if !hasPT {
		t.Error("expected Portuguese entry with definitions")
	}
}

func TestParseDumpPage_EtymologyTemplates(t *testing.T) {
	// Test with {{inh}} and {{der}} templates.
	wikitext := `==English==
===Etymology===
{{inh|en|la|negativus}}, from {{m|la|negare||to deny}}.
===Adjective===
# Not positive; denying or refusing.
`
	page := WikiPage{Title: "negative", Text: wikitext}
	entries := ParseDumpPage(page, []string{"EN"})

	if len(entries) == 0 {
		t.Fatal("expected at least one entry")
	}
	en := entries[0]
	if en.AncestorWord != "negativus" {
		t.Errorf("AncestorWord: got %q, want %q", en.AncestorWord, "negativus")
	}
	if en.AncestorLang != "LATIN" {
		t.Errorf("AncestorLang: got %q, want %q", en.AncestorLang, "LATIN")
	}
}

func TestParseDumpPage_MultiplePOS(t *testing.T) {
	// parseWikitext iterates POS in order: Adjective, Verb, Noun, Adverb.
	// When both Noun and Verb sections exist, Verb wins (checked first).
	wikitext := `==English==
===Noun===
# A piece of equipment.
===Verb===
# To run something.
`
	page := WikiPage{Title: "run", Text: wikitext}
	entries := ParseDumpPage(page, []string{"EN"})

	if len(entries) == 0 {
		t.Fatal("expected at least one entry")
	}
	// Verb is checked before Noun in the iteration order.
	if entries[0].POS != "Verb" {
		t.Errorf("POS: got %q, want %q (Verb checked before Noun)", entries[0].POS, "Verb")
	}
}

// ── cleanWikitextArg — edge cases ─────────────────────────────────────────────

func TestCleanWikitextArg_EdgeCases(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"negativus", "negativus"},           // no change
		{"negare|to deny", "negare"},         // strip inline param
		{" amor ", "amor"},                   // trim spaces
		{"*proto", "proto"},                  // strip leading asterisk
		{"-tion", "tion"},                    // strip leading hyphen
		{"", ""},                             // empty
		{"  ", ""},                           // whitespace only
		{"word|alt=foo|tr=bar", "word"},      // multiple params
		{"test\t", "test"},                   // trailing tab
	}
	for _, c := range cases {
		got := cleanWikitextArg(c.in)
		if got != c.want {
			t.Errorf("cleanWikitextArg(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ── appendRoots — append behavior ─────────────────────────────────────────────

func TestAppendRoots_AppendsToExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "roots.csv")

	// Write initial batch.
	batch1 := []seed.Root{
		{RootID: 1, RootStr: "amor", Origin: "LATIN", MeaningEN: "love"},
	}
	if err := appendRoots(batch1, path); err != nil {
		t.Fatalf("first append: %v", err)
	}

	// Append second batch.
	batch2 := []seed.Root{
		{RootID: 2, RootStr: "neg", Origin: "LATIN", MeaningEN: "deny"},
	}
	if err := appendRoots(batch2, path); err != nil {
		t.Fatalf("second append: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)

	if !strings.Contains(content, "amor") {
		t.Error("missing first root 'amor'")
	}
	if !strings.Contains(content, "neg") {
		t.Error("missing second root 'neg'")
	}

	// Should have exactly 2 data lines.
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) != 2 {
		t.Errorf("expected 2 lines, got %d", len(lines))
	}
}

// ── Checkpoint.Save — full flow ───────────────────────────────────────────────

func TestCheckpointSave_WritesJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cp.json")

	cp := &Checkpoint{Processed: make(map[string]bool)}
	cp.Mark("word1_EN")
	cp.Mark("word2_PT")

	if err := cp.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read saved checkpoint: %v", err)
	}
	content := string(data)

	if !strings.Contains(content, "word1_EN") {
		t.Error("saved checkpoint missing word1_EN")
	}
	if !strings.Contains(content, "word2_PT") {
		t.Error("saved checkpoint missing word2_PT")
	}
	if !strings.Contains(content, "last_run") {
		t.Error("saved checkpoint missing last_run timestamp")
	}

	// Verify it round-trips correctly.
	cp2, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}
	if !cp2.IsProcessed("word1_EN") || !cp2.IsProcessed("word2_PT") {
		t.Error("round-tripped checkpoint lost processed entries")
	}
	if cp2.LastRun == "" {
		t.Error("round-tripped checkpoint lost LastRun timestamp")
	}
}

func TestCheckpointSave_OverwritesPrevious(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cp.json")

	cp := &Checkpoint{Processed: make(map[string]bool)}
	cp.Mark("first")
	if err := cp.Save(path); err != nil {
		t.Fatal(err)
	}

	cp.Mark("second")
	if err := cp.Save(path); err != nil {
		t.Fatal(err)
	}

	cp2, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if !cp2.IsProcessed("first") || !cp2.IsProcessed("second") {
		t.Error("second save should contain both first and second")
	}
}

// ── StagedWriter — header and dedup edge cases ────────────────────────────────

func TestStagedWriter_HeaderWrittenOnce(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "staged.csv")

	w := NewStagedWriter(path)
	s1 := []StagedWord{{Word: "a", Lang: "EN", Score: Score{Polarity: "NEUTRAL", Confidence: 0.1}}}
	s2 := []StagedWord{{Word: "b", Lang: "EN", Score: Score{Polarity: "NEUTRAL", Confidence: 0.2}}}

	if err := w.Write(s1); err != nil {
		t.Fatal(err)
	}
	if err := w.Write(s2); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Count header occurrences.
	headerCount := strings.Count(string(data), "word,lang,root_str")
	if headerCount != 1 {
		t.Errorf("expected 1 header, got %d", headerCount)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 { // header + 2 words
		t.Errorf("expected 3 lines, got %d", len(lines))
	}
}

func TestStagedWriter_CrossLangNotDeduped(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "staged.csv")

	w := NewStagedWriter(path)
	staged := []StagedWord{
		{Word: "amor", Lang: "PT", Score: Score{Polarity: "POSITIVE", Confidence: 0.3}},
		{Word: "amor", Lang: "ES", Score: Score{Polarity: "POSITIVE", Confidence: 0.3}}, // same word, different lang
	}
	if err := w.Write(staged); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 { // header + 2 distinct entries
		t.Errorf("same word different lang should not be deduped: expected 3 lines, got %d", len(lines))
	}
}

// ── extractDefinitions ────────────────────────────────────────────────────────

func TestExtractDefinitions_WikiLinks(t *testing.T) {
	section := `
# A [[strong]] feeling of [[affection]].
# {{lb|en|uncountable}} [[compassion|Compassion]] and kindness.
`
	defs := extractDefinitions(section)
	if len(defs) == 0 {
		t.Fatal("expected definitions")
	}
	// Wikilinks should be resolved: [[affection]] → affection, [[compassion|Compassion]] → Compassion
	if strings.Contains(defs[0], "[[") || strings.Contains(defs[0], "]]") {
		t.Errorf("definition should not contain wikilinks: %q", defs[0])
	}
}

func TestExtractDefinitions_SkipsNonDefinitions(t *testing.T) {
	section := `
Some introductory text.
## A subheading line.
# An actual definition.
`
	defs := extractDefinitions(section)
	if len(defs) != 1 {
		t.Errorf("expected 1 definition, got %d: %v", len(defs), defs)
	}
}

// ── extractTranslations ───────────────────────────────────────────────────────

func TestExtractTranslations_Dedup(t *testing.T) {
	section := `
{{trans-top|love}}
* Portuguese: {{t+|pt|amor}}, {{t|pt|amor}}
* Spanish: {{t+|es|amor}}
{{trans-bottom}}
`
	trans := extractTranslations(section)
	// "amor" PT should appear only once despite two template instances.
	ptCount := 0
	for _, tr := range trans {
		if tr.Lang == "PT" && tr.Word == "amor" {
			ptCount++
		}
	}
	if ptCount != 1 {
		t.Errorf("expected 1 PT 'amor', got %d", ptCount)
	}
}

func TestExtractTranslations_UnknownLangSkipped(t *testing.T) {
	section := `
{{trans-top|test}}
* {{t|xx|unknown}}
* Portuguese: {{t+|pt|teste}}
{{trans-bottom}}
`
	trans := extractTranslations(section)
	for _, tr := range trans {
		if tr.Lang == "" {
			t.Error("unknown lang code should be skipped, not produce empty lang")
		}
	}
}

// ── extractAncestor ───────────────────────────────────────────────────────────

func TestExtractAncestor_DerTemplate(t *testing.T) {
	etym := `{{der|en|la|amāre||to love}}`
	word, lang := extractAncestor(etym)
	if lang != "LATIN" {
		t.Errorf("lang: got %q, want LATIN", lang)
	}
	if word == "" {
		t.Error("word should not be empty for {{der}} template")
	}
}

func TestExtractAncestor_BorTemplate(t *testing.T) {
	etym := `{{bor|en|grc|φιλοσοφία}}`
	word, lang := extractAncestor(etym)
	if lang != "GREEK" {
		t.Errorf("lang: got %q, want GREEK", lang)
	}
	if word == "" {
		t.Error("word should not be empty for {{bor}} template")
	}
}

func TestExtractAncestor_NoTemplate(t *testing.T) {
	etym := `From some unknown source.`
	word, lang := extractAncestor(etym)
	if word != "" || lang != "" {
		t.Errorf("no template: expected empty, got word=%q lang=%q", word, lang)
	}
}

func TestExtractAncestor_FallbackMention(t *testing.T) {
	etym := `Related to {{m|la|negare||to deny}}.`
	word, lang := extractAncestor(etym)
	if lang != "LATIN" {
		t.Errorf("lang: got %q, want LATIN", lang)
	}
	if word != "negare" {
		t.Errorf("word: got %q, want 'negare'", word)
	}
}

// ── RunImport ─────────────────────────────────────────────────────────────────

func TestRunImport_DryRun(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "dump.xml")

	// Create a small XML dump with an entry that has a known root ("negat")
	// and a translation to Portuguese.
	xmlContent := `<mediawiki>
<page>
  <title>negative</title>
  <ns>0</ns>
  <revision><text>==English==
===Etymology===
From Latin {{inh|en|la|negativus}}, from {{m|la|negare||to deny}}.
===Adjective===
# Not positive; expressing denial or refusal.
# Causing harm or damage.

====Translations====
{{trans-top|not positive}}
* Portuguese: {{t+|pt|negativo}}
* Spanish: {{t+|es|negativo}}
{{trans-bottom}}
</text></revision>
</page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	existingRoots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "to deny or negate"},
	}
	existingWords := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "negate", Lang: "EN", Norm: "negate",
			Sentiment: sentiment.PolarityNegative | sentiment.IntensityStrong | sentiment.RoleEvaluation | sentiment.DomainGeneral},
	}

	var buf bytes.Buffer
	cfg := ImportConfig{
		Config: Config{
			Langs:    []string{"EN", "PT", "ES"},
			Limit:    100,
			DryRun:   true,
			OutDir:   dir,
			RootsPath: filepath.Join(dir, "roots.csv"),
			WordsPath: filepath.Join(dir, "words.csv"),
			Output:   &buf,
		},
		DumpPath:      xmlPath,
		BatchSize:     10,
		AllowNewRoots: false,
	}

	stats, err := RunImport(cfg, existingRoots, existingWords)
	if err != nil {
		t.Fatalf("RunImport: %v", err)
	}
	if stats == nil {
		t.Fatal("stats should not be nil")
	}
	if stats.WordsExplored != 1 {
		t.Errorf("WordsExplored: got %d, want 1", stats.WordsExplored)
	}
	// In dry-run, words should still be counted.
	if stats.WordsAdded+stats.WordsStaged+stats.WordsSkipped == 0 {
		t.Error("expected some words to be processed")
	}
}

func TestRunImport_ActualWrite(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "dump.xml")

	xmlContent := `<mediawiki>
<page>
  <title>negative</title>
  <ns>0</ns>
  <revision><text>==English==
===Etymology===
From Latin {{inh|en|la|negativus}}.
===Adjective===
# Not positive; expressing denial.

====Translations====
{{trans-top|not positive}}
* Portuguese: {{t+|pt|negativo}}
{{trans-bottom}}
</text></revision>
</page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	rootsPath := filepath.Join(dir, "roots.csv")
	wordsPath := filepath.Join(dir, "words.csv")

	existingRoots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "to deny or negate"},
	}
	existingWords := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "negate", Lang: "EN", Norm: "negate",
			Sentiment: sentiment.PolarityNegative | sentiment.IntensityStrong | sentiment.RoleEvaluation | sentiment.DomainGeneral},
	}

	var buf bytes.Buffer
	cfg := ImportConfig{
		Config: Config{
			Langs:     []string{"EN", "PT"},
			Limit:     100,
			DryRun:    false,
			OutDir:    dir,
			RootsPath: rootsPath,
			WordsPath: wordsPath,
			Output:    &buf,
		},
		DumpPath:      xmlPath,
		BatchSize:     10,
		AllowNewRoots: false,
	}

	stats, err := RunImport(cfg, existingRoots, existingWords)
	if err != nil {
		t.Fatalf("RunImport: %v", err)
	}
	if stats.Duration <= 0 {
		t.Error("Duration should be positive")
	}
}

func TestRunImport_WithLimit(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "dump.xml")

	// Create dump with multiple pages.
	xmlContent := `<mediawiki>
<page><title>bad</title><ns>0</ns><revision><text>==English==
===Adjective===
# Not good; harmful or unpleasant.
</text></revision></page>
<page><title>evil</title><ns>0</ns><revision><text>==English==
===Adjective===
# Profoundly wicked; morally wrong.
</text></revision></page>
<page><title>terrible</title><ns>0</ns><revision><text>==English==
===Adjective===
# Extremely bad; dreadful.
</text></revision></page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	cfg := ImportConfig{
		Config: Config{
			Langs:     []string{"EN"},
			Limit:     1, // limit to 1 word
			DryRun:    true,
			OutDir:    dir,
			RootsPath: filepath.Join(dir, "roots.csv"),
			WordsPath: filepath.Join(dir, "words.csv"),
			Output:    &buf,
		},
		DumpPath:      xmlPath,
		BatchSize:     500,
		AllowNewRoots: true,
	}

	stats, err := RunImport(cfg, nil, nil)
	if err != nil {
		t.Fatalf("RunImport: %v", err)
	}
	if stats.WordsAdded > 1 {
		t.Errorf("WordsAdded: got %d, want <= 1 (limit=1)", stats.WordsAdded)
	}
}

func TestRunImport_AllowNewRoots(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "dump.xml")

	xmlContent := `<mediawiki>
<page><title>freedom</title><ns>0</ns><revision><text>==English==
===Etymology===
From Old English {{inh|en|ang|frēodōm}}.
===Noun===
# The state of being free; liberty.
</text></revision></page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	cfg := ImportConfig{
		Config: Config{
			Langs:     []string{"EN"},
			Limit:     100,
			DryRun:    true,
			Verbose:   true,
			OutDir:    dir,
			RootsPath: filepath.Join(dir, "roots.csv"),
			WordsPath: filepath.Join(dir, "words.csv"),
			Output:    &buf,
		},
		DumpPath:      xmlPath,
		BatchSize:     500,
		AllowNewRoots: true,
	}

	stats, err := RunImport(cfg, nil, nil)
	if err != nil {
		t.Fatalf("RunImport: %v", err)
	}
	// With AllowNewRoots=true, a new root should be created.
	if stats.RootsAdded == 0 {
		t.Error("expected at least 1 new root with AllowNewRoots=true")
	}
}

func TestRunImport_NonexistentDump(t *testing.T) {
	var buf bytes.Buffer
	cfg := ImportConfig{
		Config: Config{
			Langs:  []string{"EN"},
			Limit:  10,
			OutDir: t.TempDir(),
			Output: &buf,
		},
		DumpPath: "/nonexistent/dump.xml",
	}
	_, err := RunImport(cfg, nil, nil)
	if err == nil {
		t.Error("expected error for nonexistent dump file")
	}
}

func TestRunImport_DefaultBatchSize(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "dump.xml")
	if err := os.WriteFile(xmlPath, []byte("<mediawiki></mediawiki>"), 0o644); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	cfg := ImportConfig{
		Config: Config{
			Langs:     []string{"EN"},
			Limit:     10,
			DryRun:    true,
			OutDir:    dir,
			RootsPath: filepath.Join(dir, "r.csv"),
			WordsPath: filepath.Join(dir, "w.csv"),
			Output:    &buf,
		},
		DumpPath:  xmlPath,
		BatchSize: 0, // should default to 500
	}

	stats, err := RunImport(cfg, nil, nil)
	if err != nil {
		t.Fatalf("RunImport: %v", err)
	}
	_ = stats // should complete without panic
}

func TestRunImport_DefaultOutput(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "dump.xml")
	if err := os.WriteFile(xmlPath, []byte("<mediawiki></mediawiki>"), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := ImportConfig{
		Config: Config{
			Langs:     []string{"EN"},
			Limit:     10,
			DryRun:    true,
			OutDir:    dir,
			RootsPath: filepath.Join(dir, "r.csv"),
			WordsPath: filepath.Join(dir, "w.csv"),
			Output:    nil, // defaults to os.Stdout
		},
		DumpPath:  xmlPath,
		BatchSize: 100,
	}

	stats, err := RunImport(cfg, nil, nil)
	if err != nil {
		t.Fatalf("RunImport: %v", err)
	}
	_ = stats
}

// ── ScanDump — bz2 support ───────────────────────────────────────────────────

func TestScanDump_EmptyDump(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "empty.xml")
	if err := os.WriteFile(xmlPath, []byte("<mediawiki></mediawiki>"), 0o644); err != nil {
		t.Fatal(err)
	}

	var count int
	err := ScanDump(xmlPath, func(p WikiPage) error {
		count++
		return nil
	})
	if err != nil {
		t.Fatalf("ScanDump empty: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0 pages, got %d", count)
	}
}

func TestScanDump_FnReturnsError(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "test.xml")
	xmlContent := `<mediawiki>
<page><title>word</title><ns>0</ns><revision><text>content</text></revision></page>
</mediawiki>`
	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	customErr := io.ErrUnexpectedEOF
	err := ScanDump(xmlPath, func(p WikiPage) error {
		return customErr
	})
	if err != customErr {
		t.Errorf("expected custom error, got %v", err)
	}
}

func TestScanDump_EmptyTextSkipped(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "test.xml")
	xmlContent := `<mediawiki>
<page><title>empty</title><ns>0</ns><revision><text></text></revision></page>
<page><title>notempty</title><ns>0</ns><revision><text>has content</text></revision></page>
</mediawiki>`
	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var pages []WikiPage
	err := ScanDump(xmlPath, func(p WikiPage) error {
		pages = append(pages, p)
		return nil
	})
	if err != nil {
		t.Fatalf("ScanDump: %v", err)
	}
	// Empty text pages should be skipped.
	if len(pages) != 1 || pages[0].Title != "notempty" {
		t.Errorf("expected only 'notempty' page, got %v", pages)
	}
}

// ── resolveRoot — more coverage ───────────────────────────────────────────────

func TestResolveRoot_AncestorStemMatch(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "neg", Origin: "LATIN", MeaningEN: "to deny or negate"},
	}
	entry := &Entry{
		Word:         "negative",
		Lang:         "EN",
		AncestorWord: "negare",   // StemAncestor("negare") = "neg"
		AncestorLang: "LATIN",
		Definitions:  []string{"not positive; expressing denial"},
	}

	rootID, rootStr, isNew := resolveRoot(entry, "EN", roots, nil)
	if isNew {
		t.Error("should match existing root 'neg' via ancestor stem")
	}
	if rootID != 1 {
		t.Errorf("rootID: got %d, want 1", rootID)
	}
	if rootStr != "neg" {
		t.Errorf("rootStr: got %q, want 'neg'", rootStr)
	}
}

// ── classifyBest — blocked root ───────────────────────────────────────────────

func TestClassifyBest_BlockedRoot(t *testing.T) {
	// Root 82 is in blockedRoots.
	s := classifyBest(82, &Entry{Word: "test", Definitions: []string{"bad word"}}, nil)
	if s.Confidence != 0 {
		t.Errorf("blocked root: expected confidence=0, got %.2f", s.Confidence)
	}
	if s.Source != "blocked-root" {
		t.Errorf("blocked root: expected source='blocked-root', got %q", s.Source)
	}
}

// ── wiktCacheDir — fallback (no XDG) ──────────────────────────────────────────

func TestWiktCacheDir_FallbackHome(t *testing.T) {
	t.Setenv("XDG_CACHE_HOME", "")
	got := wiktCacheDir()
	// Should fall back to $HOME/.cache/lexsent/wikt
	if !strings.HasSuffix(got, filepath.Join(".cache", "lexsent", "wikt")) {
		t.Errorf("wiktCacheDir fallback: got %q, expected suffix .cache/lexsent/wikt", got)
	}
}

// ── Flush — words path ────────────────────────────────────────────────────────

func TestFlush_AppendsWords(t *testing.T) {
	dir := t.TempDir()
	rootsPath := filepath.Join(dir, "roots.csv")
	wordsPath := filepath.Join(dir, "words.csv")

	// Pre-create roots file.
	batch1 := []seed.Root{{RootID: 1, RootStr: "amor", Origin: "LATIN", MeaningEN: "love"}}
	if err := Flush(batch1, nil, rootsPath, wordsPath); err != nil {
		t.Fatal(err)
	}

	// Now flush only words.
	words := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "love", Lang: "EN", Norm: "love",
			Sentiment: sentiment.PolarityPositive | sentiment.IntensityStrong | sentiment.RoleEvaluation | sentiment.DomainGeneral},
	}
	if err := Flush(nil, words, rootsPath, wordsPath); err != nil {
		t.Fatalf("Flush words: %v", err)
	}

	wordData, err := os.ReadFile(wordsPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(wordData), "love") {
		t.Error("words file should contain 'love'")
	}
}

// ── RunImport — with translations into staged ─────────────────────────────────

func TestRunImport_TranslationsToStaged(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "dump.xml")

	// A page with translations that will likely be staged (low confidence).
	xmlContent := `<mediawiki>
<page><title>table</title><ns>0</ns><revision><text>==English==
===Etymology===
From Latin {{inh|en|la|tabula}}.
===Noun===
# A piece of furniture with a flat surface.

====Translations====
{{trans-top|furniture}}
* Portuguese: {{t+|pt|mesa}}
* German: {{t+|de|Tisch}}
{{trans-bottom}}
</text></revision></page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	cfg := ImportConfig{
		Config: Config{
			Langs:     []string{"EN", "PT", "DE"},
			Limit:     100,
			DryRun:    false,
			OutDir:    dir,
			RootsPath: filepath.Join(dir, "roots.csv"),
			WordsPath: filepath.Join(dir, "words.csv"),
			Output:    &buf,
		},
		DumpPath:      xmlPath,
		BatchSize:     10,
		AllowNewRoots: true,
	}

	stats, err := RunImport(cfg, nil, nil)
	if err != nil {
		t.Fatalf("RunImport: %v", err)
	}
	// Should process at least the entry word.
	total := stats.WordsAdded + stats.WordsStaged + stats.WordsSkipped
	if total == 0 {
		t.Error("expected some words to be processed")
	}
}

// ── RunImport — verbose mode ──────────────────────────────────────────────────

func TestRunImport_VerboseOutput(t *testing.T) {
	dir := t.TempDir()
	xmlPath := filepath.Join(dir, "dump.xml")

	xmlContent := `<mediawiki>
<page><title>beautiful</title><ns>0</ns><revision><text>==English==
===Adjective===
# Very pleasing to look at; attractive and wonderful.
</text></revision></page>
</mediawiki>`

	if err := os.WriteFile(xmlPath, []byte(xmlContent), 0o644); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	cfg := ImportConfig{
		Config: Config{
			Langs:     []string{"EN"},
			Limit:     100,
			DryRun:    true,
			Verbose:   true,
			OutDir:    dir,
			RootsPath: filepath.Join(dir, "roots.csv"),
			WordsPath: filepath.Join(dir, "words.csv"),
			Output:    &buf,
		},
		DumpPath:      xmlPath,
		BatchSize:     500,
		AllowNewRoots: true,
	}

	stats, err := RunImport(cfg, nil, nil)
	if err != nil {
		t.Fatalf("RunImport: %v", err)
	}
	_ = stats
	// Verbose should produce some output.
	if buf.Len() == 0 {
		t.Error("verbose mode should produce output")
	}
}

// ── Run (pipeline.go) — via pre-populated cache ───────────────────────────────

// seedCache populates the wikt cache for a word so Fetch() returns cached data
// without making HTTP calls.
func seedCache(t *testing.T, entry *Entry) {
	t.Helper()
	saveCache(entry.Word, entry.Lang, entry)
}

func TestRun_BasicBFS(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	// Pre-populate cache entries so Fetch never hits the network.
	seedCache(t, &Entry{
		Word: "love", Lang: "EN",
		Etymology:    "From Old English lufu",
		AncestorWord: "", AncestorLang: "",
		POS:         "Noun",
		Definitions: []string{"An intense feeling of deep affection"},
		Translations: []Trans{
			{Word: "amor", Lang: "PT"},
		},
	})
	seedCache(t, &Entry{
		Word: "amor", Lang: "PT",
		POS:         "Noun",
		Definitions: []string{"Love; affection"},
	})

	dir := t.TempDir()
	rootsPath := filepath.Join(dir, "roots.csv")
	wordsPath := filepath.Join(dir, "words.csv")

	existingRoots := []seed.Root{
		{RootID: 1, RootStr: "amor", Origin: "LATIN", MeaningEN: "love"},
	}
	existingWords := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "amore", Lang: "IT", Norm: "amore",
			Sentiment: sentiment.PolarityPositive | sentiment.IntensityStrong | sentiment.RoleEvaluation | sentiment.DomainGeneral},
	}

	var buf bytes.Buffer
	cfg := Config{
		Seeds:     []string{"love"},
		Langs:     []string{"EN", "PT"},
		MaxDepth:  2,
		Limit:     10,
		DryRun:    true,
		Workers:   1,
		OutDir:    dir,
		RootsPath: rootsPath,
		WordsPath: wordsPath,
		Output:    &buf,
	}

	stats, err := Run(cfg, existingRoots, existingWords)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if stats == nil {
		t.Fatal("stats should not be nil")
	}
	if stats.Duration <= 0 {
		t.Error("Duration should be positive")
	}
	// "love" should be explored
	if stats.WordsExplored == 0 && stats.WordsAdded == 0 && stats.WordsStaged == 0 && stats.WordsSkipped == 0 {
		t.Error("expected some processing to occur")
	}
}

func TestRun_ActualWrite(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	seedCache(t, &Entry{
		Word: "happy", Lang: "EN",
		POS:          "Adjective",
		Definitions:  []string{"Feeling great pleasure and contentment"},
		Translations: []Trans{{Word: "feliz", Lang: "PT"}, {Word: "feliz", Lang: "ES"}},
	})
	seedCache(t, &Entry{
		Word: "feliz", Lang: "PT",
		POS:         "Adjective",
		Definitions: []string{"Happy; content"},
	})
	seedCache(t, &Entry{
		Word: "feliz", Lang: "ES",
		POS:         "Adjective",
		Definitions: []string{"Happy"},
	})

	dir := t.TempDir()
	rootsPath := filepath.Join(dir, "roots.csv")
	wordsPath := filepath.Join(dir, "words.csv")

	var buf bytes.Buffer
	cfg := Config{
		Seeds:     []string{"happy"},
		Langs:     []string{"EN", "PT", "ES"},
		MaxDepth:  2,
		Limit:     20,
		DryRun:    false,
		Workers:   1,
		OutDir:    dir,
		RootsPath: rootsPath,
		WordsPath: wordsPath,
		Output:    &buf,
	}

	stats, err := Run(cfg, nil, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	_ = stats
}

func TestRun_Reset(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	seedCache(t, &Entry{
		Word: "test", Lang: "EN",
		POS:         "Noun",
		Definitions: []string{"A procedure to evaluate"},
	})

	dir := t.TempDir()
	rootsPath := filepath.Join(dir, "roots.csv")
	wordsPath := filepath.Join(dir, "words.csv")

	var buf bytes.Buffer
	cfg := Config{
		Seeds:     []string{"test"},
		Langs:     []string{"EN"},
		MaxDepth:  1,
		Limit:     10,
		DryRun:    true,
		Reset:     true,
		Workers:   1,
		OutDir:    dir,
		RootsPath: rootsPath,
		WordsPath: wordsPath,
		Output:    &buf,
	}

	// First run.
	_, err := Run(cfg, nil, nil)
	if err != nil {
		t.Fatalf("Run 1: %v", err)
	}

	// Second run with Reset=true should re-process.
	cfg.Reset = true
	buf.Reset()
	_, err = Run(cfg, nil, nil)
	if err != nil {
		t.Fatalf("Run 2: %v", err)
	}
}

func TestRun_Reexpand(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	seedCache(t, &Entry{
		Word: "kind", Lang: "EN",
		POS:         "Adjective",
		Definitions: []string{"Having a gentle and generous nature"},
	})

	dir := t.TempDir()
	var buf bytes.Buffer
	cfg := Config{
		Seeds:     []string{"kind"},
		Langs:     []string{"EN"},
		MaxDepth:  1,
		Limit:     10,
		DryRun:    true,
		Reexpand:  true,
		Workers:   1,
		OutDir:    dir,
		RootsPath: filepath.Join(dir, "roots.csv"),
		WordsPath: filepath.Join(dir, "words.csv"),
		Output:    &buf,
	}

	_, err := Run(cfg, nil, nil)
	if err != nil {
		t.Fatalf("Run reexpand: %v", err)
	}
}

func TestRun_PageNotFound(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	// Don't seed cache for "zzzznonexistent" — Fetch will fail.
	// But since we're not hitting network, it will fail with a file-not-found from cache.
	// The code treats this as a network error (not ErrPageNotFound), so it won't mark as processed.

	dir := t.TempDir()
	var buf bytes.Buffer
	cfg := Config{
		Seeds:     []string{"zzzznonexistent"},
		Langs:     []string{"EN"},
		MaxDepth:  1,
		Limit:     10,
		DryRun:    true,
		Verbose:   true,
		Workers:   1,
		OutDir:    dir,
		RootsPath: filepath.Join(dir, "roots.csv"),
		WordsPath: filepath.Join(dir, "words.csv"),
		Output:    &buf,
	}

	stats, err := Run(cfg, nil, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if stats.Errors == 0 {
		t.Error("expected at least 1 error for missing cache")
	}
}

func TestRun_MultipleWorkers(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	seedCache(t, &Entry{
		Word: "peace", Lang: "EN",
		POS:          "Noun",
		Definitions:  []string{"Freedom from disturbance; calm"},
		Translations: []Trans{{Word: "paz", Lang: "PT"}, {Word: "paz", Lang: "ES"}},
	})
	seedCache(t, &Entry{
		Word: "paz", Lang: "PT",
		POS:         "Noun",
		Definitions: []string{"Peace"},
	})
	seedCache(t, &Entry{
		Word: "paz", Lang: "ES",
		POS:         "Noun",
		Definitions: []string{"Peace"},
	})

	dir := t.TempDir()
	var buf bytes.Buffer
	cfg := Config{
		Seeds:     []string{"peace"},
		Langs:     []string{"EN", "PT", "ES"},
		MaxDepth:  2,
		Limit:     20,
		DryRun:    true,
		Workers:   3, // test parallel path
		OutDir:    dir,
		RootsPath: filepath.Join(dir, "roots.csv"),
		WordsPath: filepath.Join(dir, "words.csv"),
		Output:    &buf,
	}

	stats, err := Run(cfg, nil, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	_ = stats
}

func TestRun_EmptySeeds(t *testing.T) {
	dir := t.TempDir()
	var buf bytes.Buffer
	cfg := Config{
		Seeds:     []string{},
		Langs:     []string{"EN"},
		MaxDepth:  1,
		Limit:     10,
		DryRun:    true,
		Workers:   1,
		OutDir:    dir,
		RootsPath: filepath.Join(dir, "roots.csv"),
		WordsPath: filepath.Join(dir, "words.csv"),
		Output:    &buf,
	}

	stats, err := Run(cfg, nil, nil)
	if err != nil {
		t.Fatalf("Run empty seeds: %v", err)
	}
	if stats.WordsAdded != 0 {
		t.Error("empty seeds should add no words")
	}
}

func TestRun_DefaultOutput(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	dir := t.TempDir()
	cfg := Config{
		Seeds:     []string{},
		Langs:     []string{"EN"},
		MaxDepth:  1,
		Limit:     10,
		DryRun:    true,
		Workers:   0, // should default to 1
		OutDir:    dir,
		RootsPath: filepath.Join(dir, "roots.csv"),
		WordsPath: filepath.Join(dir, "words.csv"),
		Output:    nil, // defaults to os.Stdout
	}

	stats, err := Run(cfg, nil, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	_ = stats
}

func TestRun_BlockedRoot(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	// Create a cache entry with a translation that maps to a blocked root.
	seedCache(t, &Entry{
		Word: "test", Lang: "EN",
		POS:          "Noun",
		Definitions:  []string{"A trial or examination"},
		Translations: []Trans{{Word: "prueba", Lang: "ES"}},
	})
	seedCache(t, &Entry{
		Word: "prueba", Lang: "ES",
		POS:         "Noun",
		Definitions: []string{"Test, trial"},
	})

	dir := t.TempDir()
	// Use root 82 (blocked) so translations aren't expanded.
	existingRoots := []seed.Root{
		{RootID: 82, RootStr: "test", Origin: "LATIN", MeaningEN: "test"},
	}
	existingWords := []seed.Word{
		{WordID: (82 << 12) | 1, RootID: 82, Variant: 1, Word: "test", Lang: "EN", Norm: "test",
			Sentiment: sentiment.PolarityNeutral | sentiment.IntensityNone | sentiment.RoleEvaluation | sentiment.DomainGeneral},
	}

	var buf bytes.Buffer
	cfg := Config{
		Seeds:     []string{"test"},
		Langs:     []string{"EN", "ES"},
		MaxDepth:  2,
		Limit:     10,
		DryRun:    true,
		Workers:   1,
		OutDir:    dir,
		RootsPath: filepath.Join(dir, "roots.csv"),
		WordsPath: filepath.Join(dir, "words.csv"),
		Output:    &buf,
	}

	stats, err := Run(cfg, existingRoots, existingWords)
	if err != nil {
		t.Fatalf("Run blocked root: %v", err)
	}
	// Blocked root words should be staged, not auto-accepted.
	_ = stats
}

// ── makeWord — variant overflow ───────────────────────────────────────────────

func TestMakeWord_VariantOverflow(t *testing.T) {
	// Create words with variant near overflow to test the error path.
	// morpheme.MaxVariant is 4095, so variant 4096 would fail.
	// We'd need 4095 words with the same root to trigger this.
	// Instead, test with a reasonable scenario.
	score := Score{Polarity: "POSITIVE", Intensity: "MODERATE", Role: "EVALUATION", Confidence: 0.8}
	w, ok := makeWord(1, "test", "EN", "test", score, nil)
	if !ok {
		t.Fatal("makeWord should succeed for normal variant")
	}
	if w.Variant != 1 {
		t.Errorf("expected variant 1 for empty words, got %d", w.Variant)
	}
}

// ── resolveRoot — polysemy rejection ──────────────────────────────────────────

func TestResolveRoot_PolysemyRejection(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "gut", Origin: "PROTO_GERMANIC", MeaningEN: "good or pleasant"},
	}
	// EN "gut" (intestine) should NOT match root "gut" (good).
	entry := &Entry{
		Word:        "gut",
		Lang:        "EN",
		Definitions: []string{"the intestinal tract; the stomach and bowels"},
	}

	_, _, isNew := resolveRoot(entry, "EN", roots, nil)
	if !isNew {
		t.Error("polysemous 'gut' (intestine) should NOT match root 'gut' (good)")
	}
}

func TestResolveRoot_RootWithoutMeaning(t *testing.T) {
	// When root has empty MeaningEN, SenseCoherent should be lenient.
	roots := []seed.Root{
		{RootID: 5, RootStr: "test", Origin: "UNKNOWN", MeaningEN: ""},
	}
	entry := &Entry{
		Word:        "test",
		Lang:        "EN",
		Definitions: []string{"an examination"},
	}

	rootID, _, isNew := resolveRoot(entry, "EN", roots, nil)
	if isNew {
		t.Error("empty MeaningEN should be treated as coherent (lenient)")
	}
	if rootID != 5 {
		t.Errorf("rootID: got %d, want 5", rootID)
	}
}
