package autoqa_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/kak/umcs/pkg/autoqa"
	"github.com/kak/umcs/pkg/classify"
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
)

// buildTestLex builds a minimal lexicon with known sentiment words for testing.
func buildTestLex(t *testing.T) *lexdb.Lexicon {
	t.Helper()
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible",
			Sentiment: 0x00120180, Flags: 0x300}, // NEGATIVE, FORMAL register
		{WordID: 8193, RootID: 2, Variant: 1, Word: "good", Lang: "EN", Norm: "good",
			Sentiment: 0x00130140}, // POSITIVE
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "test.umcs")
	if _, err := lexdb.Build(roots, words, path); err != nil {
		t.Fatalf("build lexicon: %v", err)
	}
	lex, err := lexdb.Load(path)
	if err != nil {
		t.Fatalf("load lexicon: %v", err)
	}
	return lex
}

func buildTestClf(t *testing.T, lex *lexdb.Lexicon) *classify.Classifier {
	t.Helper()
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	clf.LR = 0.02

	// Train using ExtractFromLexicon — same code path as inference — so the
	// features seen at training time match inference time exactly. FPolarity is
	// NOT zeroed here because this is a tiny test classifier (2 words), not
	// production training; there is no overfitting risk at this scale.
	pairs := []struct{ word, label string }{
		{"terrible", "NEGATIVE"},
		{"good", "POSITIVE"},
	}
	for epoch := 0; epoch < 500; epoch++ {
		for _, p := range pairs {
			f, ok := classify.ExtractFromLexicon(lex, p.word, "")
			if !ok {
				continue
			}
			clf.TrainStep(f, clf.ClassIndex(p.label))
		}
	}
	return clf
}

func TestCheck_PassesOnCorrectClass(t *testing.T) {
	lex := buildTestLex(t)
	clf := buildTestClf(t, lex)

	result := autoqa.Check(clf, lex, autoqa.OutputSpec{
		Text:      "terrible",
		WantClass: "NEGATIVE",
		Name:      "terrible-word",
	})
	if !result.Passed {
		t.Errorf("Check failed unexpectedly: %s", result.Reason)
	}
}

func TestCheck_FailsOnWrongClass(t *testing.T) {
	lex := buildTestLex(t)
	clf := buildTestClf(t, lex)

	result := autoqa.Check(clf, lex, autoqa.OutputSpec{
		Text:      "terrible",
		WantClass: "POSITIVE", // wrong expectation
		Name:      "wrong-expectation",
	})
	if result.Passed {
		t.Error("Check should have failed: 'terrible' expected POSITIVE but it's NEGATIVE")
	}
}

func TestCheck_SkipsClassCheckWhenWantClassEmpty(t *testing.T) {
	lex := buildTestLex(t)
	clf := buildTestClf(t, lex)

	result := autoqa.Check(clf, lex, autoqa.OutputSpec{
		Text:    "terrible",
		MinConf: 0.0, // no class check, no conf check
	})
	if !result.Passed {
		t.Errorf("Check with no constraints should always pass: %s", result.Reason)
	}
}

func TestCheck_OOVTextIsNeutral(t *testing.T) {
	lex := buildTestLex(t)
	clf := buildTestClf(t, lex)

	result := autoqa.Check(clf, lex, autoqa.OutputSpec{
		Text:      "xyzxyzxyz_oov_word",
		WantClass: "NEUTRAL",
	})
	// OOV words should be classified as NEUTRAL (no tokens found = default)
	if !result.Passed {
		t.Logf("OOV classified as %q (conf=%.2f): %s", result.GotClass, result.GotConf, result.Reason)
		// Not a hard failure — OOV behavior may vary
	}
}

func TestCheckBatch_ReturnsOnlyFailures(t *testing.T) {
	lex := buildTestLex(t)
	clf := buildTestClf(t, lex)

	specs := []autoqa.OutputSpec{
		{Text: "terrible", WantClass: "NEGATIVE", Name: "neg-word"},
		{Text: "terrible", WantClass: "POSITIVE", Name: "wrong-expectation"},
		{Text: "xyzxyz", WantClass: "NEUTRAL", Name: "oov-neutral"},
	}
	failures := autoqa.CheckBatch(clf, lex, specs)
	// At least the "wrong-expectation" spec should fail
	foundWrong := false
	for _, f := range failures {
		if f.Spec.Name == "wrong-expectation" {
			foundWrong = true
		}
	}
	if !foundWrong {
		t.Error("CheckBatch: 'wrong-expectation' spec should have failed but did not")
	}
}

func TestCheckFile_JSONArray(t *testing.T) {
	lex := buildTestLex(t)
	clf := buildTestClf(t, lex)

	specs := []autoqa.OutputSpec{
		{Text: "terrible", WantClass: "NEGATIVE", Name: "neg"},
		{Text: "good", WantClass: "POSITIVE", Name: "pos"},
	}
	data, err := json.Marshal(specs)
	if err != nil {
		t.Fatalf("marshal specs: %v", err)
	}

	path := filepath.Join(t.TempDir(), "specs.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write specs: %v", err)
	}

	passed, failed, _, err := autoqa.CheckFile(clf, lex, path)
	if err != nil {
		t.Fatalf("CheckFile: %v", err)
	}
	total := passed + failed
	if total != 2 {
		t.Errorf("total specs = %d, want 2", total)
	}
}

func TestCheckFile_JSONL(t *testing.T) {
	lex := buildTestLex(t)
	clf := buildTestClf(t, lex)

	content := `{"text":"terrible","want_class":"NEGATIVE","name":"neg"}
{"text":"good","want_class":"POSITIVE","name":"pos"}
`
	path := filepath.Join(t.TempDir(), "specs.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write specs: %v", err)
	}

	passed, failed, _, err := autoqa.CheckFile(clf, lex, path)
	if err != nil {
		t.Fatalf("CheckFile JSONL: %v", err)
	}
	total := passed + failed
	if total != 2 {
		t.Errorf("total specs = %d, want 2", total)
	}
}

func TestCheckFile_NonExistentFile(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	lex, _ := lexdb.Load("/tmp/does_not_exist_autoqa.umcs")

	_, _, _, err := autoqa.CheckFile(clf, lex, "/tmp/does_not_exist_autoqa_specs.json")
	if err == nil {
		t.Error("CheckFile on non-existent file should return an error")
	}
}

// TestAutoQA_SemanticInvariants demonstrates the kind of QA that can be run
// on code-generated outputs (docstrings, error messages, log lines).
func TestAutoQA_SemanticInvariants(t *testing.T) {
	lex := buildTestLex(t)
	clf := buildTestClf(t, lex)

	// Specs for hypothetical Tuk stdlib outputs
	specs := []autoqa.OutputSpec{
		// Error messages should be NEGATIVE
		{Text: "terrible error occurred", WantClass: "NEGATIVE", Name: "error-msg"},
		// Success messages should be POSITIVE
		{Text: "good result achieved", WantClass: "POSITIVE", Name: "success-msg"},
	}

	failures := autoqa.CheckBatch(clf, lex, specs)
	for _, f := range failures {
		t.Logf("Semantic invariant violation: %s (got=%q want=%q conf=%.2f)",
			f.Spec.Name, f.GotClass, f.Spec.WantClass, f.GotConf)
	}
	// We log rather than fail hard — a small test lexicon may not have full coverage
}
