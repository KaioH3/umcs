// Package autoqa provides automated quality assurance for code-generated text.
//
// It uses the UMCS classifier and lexicon to verify that generated outputs —
// docstrings, error messages, variable names, log lines — carry the intended
// sentiment, register, and semantic profile. This allows any language whose
// toolchain links libumcs.so to embed semantic QA directly in its CI pipeline.
//
// # Workflow
//
//	specs := []autoqa.OutputSpec{
//	    {Text: "frees allocated memory", WantClass: "NEUTRAL", WantRegister: "FORMAL"},
//	    {Text: "fatal error: out of memory", WantClass: "NEGATIVE", MinConf: 0.7},
//	    {Text: "successfully loaded lexicon", WantClass: "POSITIVE", MinConf: 0.6},
//	}
//	failures := autoqa.CheckBatch(clf, lex, specs)
//	for _, f := range failures { log.Println(f) }
//
// # Integration
//
// From the CLI:
//
//	lexsent autoqa --specs tests/autoqa_specs.json --model models/classifier.bin
package autoqa

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/kak/umcs/pkg/classify"
	"github.com/kak/umcs/pkg/lexdb"
)

// OutputSpec defines the expected semantic properties of a generated output.
// Fields with zero values are not checked (e.g. MinConf=0 → any confidence passes).
type OutputSpec struct {
	// Text is the generated string to analyze (required).
	Text string `json:"text"`

	// WantClass is the expected sentiment class: "POSITIVE", "NEGATIVE", "NEUTRAL".
	// If empty, the sentiment check is skipped.
	WantClass string `json:"want_class,omitempty"`

	// MinConf is the minimum required classifier confidence [0, 1].
	// Default 0 means any confidence is accepted.
	MinConf float64 `json:"min_conf,omitempty"`

	// WantRegister is the expected register: "FORMAL", "NEUTRAL", "INFORMAL".
	// If empty, the register check is skipped. Register is determined by the
	// majority register of content words in the text.
	WantRegister string `json:"want_register,omitempty"`

	// Name is an optional label for display in error messages.
	Name string `json:"name,omitempty"`
}

// Result describes the outcome of one OutputSpec check.
type Result struct {
	Spec       OutputSpec
	GotClass   string
	GotConf    float64
	GotRegister string
	Passed     bool
	Reason     string // empty when Passed=true
}

// Check analyzes generated text against its spec using the provided classifier
// and lexicon. Returns nil on pass, a descriptive error on violation.
//
// The text is tokenized on whitespace; each token is looked up in the lexicon.
// The final classification is the consensus of per-token predictions.
func Check(clf *classify.Classifier, lex *lexdb.Lexicon, spec OutputSpec) *Result {
	class, conf, register := analyzeText(clf, lex, spec.Text)
	result := &Result{
		Spec:        spec,
		GotClass:    class,
		GotConf:     conf,
		GotRegister: register,
	}

	label := spec.Name
	if label == "" {
		label = fmt.Sprintf("%q", spec.Text)
	}

	var violations []string

	if spec.WantClass != "" && class != spec.WantClass {
		violations = append(violations, fmt.Sprintf("class: got %q, want %q (conf=%.2f)",
			class, spec.WantClass, conf))
	}
	if spec.MinConf > 0 && conf < spec.MinConf {
		violations = append(violations, fmt.Sprintf("confidence: got %.2f, want ≥ %.2f",
			conf, spec.MinConf))
	}
	if spec.WantRegister != "" && !strings.EqualFold(register, spec.WantRegister) {
		violations = append(violations, fmt.Sprintf("register: got %q, want %q",
			register, spec.WantRegister))
	}

	if len(violations) > 0 {
		result.Reason = fmt.Sprintf("%s: %s", label, strings.Join(violations, "; "))
		return result
	}
	result.Passed = true
	return result
}

// CheckBatch runs specs concurrently and returns all failed results.
// Specs are checked in parallel (one goroutine each). Thread-safe.
func CheckBatch(clf *classify.Classifier, lex *lexdb.Lexicon, specs []OutputSpec) []Result {
	results := make([]Result, len(specs))
	var wg sync.WaitGroup
	for i, spec := range specs {
		wg.Add(1)
		go func(i int, spec OutputSpec) {
			defer wg.Done()
			results[i] = *Check(clf, lex, spec)
		}(i, spec)
	}
	wg.Wait()

	var failures []Result
	for _, r := range results {
		if !r.Passed {
			failures = append(failures, r)
		}
	}
	return failures
}

// CheckFile reads a JSONL or JSON-array file of OutputSpec definitions and
// checks each one. Returns counts of passed and failed specs plus all failures.
//
// File format: either a JSON array of OutputSpec objects, or one JSON object
// per line (JSONL). The format is auto-detected.
func CheckFile(clf *classify.Classifier, lex *lexdb.Lexicon, path string) (passed, failed int, failures []Result, err error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, nil, fmt.Errorf("read %s: %w", path, err)
	}

	var specs []OutputSpec
	trimmed := strings.TrimSpace(string(data))
	if strings.HasPrefix(trimmed, "[") {
		// JSON array
		if err := json.Unmarshal(data, &specs); err != nil {
			return 0, 0, nil, fmt.Errorf("parse JSON array in %s: %w", path, err)
		}
	} else {
		// JSONL: one object per line
		for _, line := range strings.Split(trimmed, "\n") {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "//") {
				continue
			}
			var spec OutputSpec
			if err := json.Unmarshal([]byte(line), &spec); err != nil {
				return 0, 0, nil, fmt.Errorf("parse JSONL line in %s: %w", path, err)
			}
			specs = append(specs, spec)
		}
	}

	failList := CheckBatch(clf, lex, specs)
	return len(specs) - len(failList), len(failList), failList, nil
}

// analyzeText tokenizes the text on whitespace, looks up each token in the
// lexicon, and returns the consensus classification, confidence, and register.
//
// Strategy:
//   - Each content word (found in lexicon) contributes one vote via clf.Predict.
//   - The winning class is the most frequent prediction weighted by confidence.
//   - Register is the most common register among content words.
//   - Falls back to "NEUTRAL" with conf=0 when no words are found.
func analyzeText(clf *classify.Classifier, lex *lexdb.Lexicon, text string) (class string, conf float64, register string) {
	tokens := strings.Fields(strings.ToLower(text))

	type vote struct {
		class string
		conf  float64
	}
	var votes []vote
	var registers []string

	for _, tok := range tokens {
		// Strip common punctuation
		tok = strings.Trim(tok, ".,;:!?\"'()")
		if tok == "" {
			continue
		}
		f, ok := classify.ExtractFromLexicon(lex, tok, "")
		if !ok {
			continue
		}
		c, probs := clf.Predict(f)
		idx := clf.ClassIndex(c)
		var p float64
		if idx >= 0 && idx < len(probs) {
			p = probs[idx]
		}
		votes = append(votes, vote{c, p})

		// Infer register from word record
		w := lex.LookupWord(tok)
		if w != nil {
			reg := (w.Flags >> 8) & 0xF
			registers = append(registers, registerName(reg))
		}
	}

	if len(votes) == 0 {
		return "NEUTRAL", 0, "NEUTRAL"
	}

	// Weighted vote: accumulate confidence per class
	classTotals := make(map[string]float64)
	for _, v := range votes {
		classTotals[v.class] += v.conf
	}
	var bestClass string
	var bestTotal float64
	for c, total := range classTotals {
		if total > bestTotal {
			bestTotal = total
			bestClass = c
		}
	}
	avgConf := bestTotal / float64(len(votes))

	// Register majority
	regCounts := make(map[string]int)
	for _, r := range registers {
		regCounts[r]++
	}
	var bestReg string
	var bestRegCount int
	for r, c := range regCounts {
		if c > bestRegCount {
			bestRegCount = c
			bestReg = r
		}
	}
	if bestReg == "" {
		bestReg = "NEUTRAL"
	}

	return bestClass, avgConf, bestReg
}

// registerName maps a register enum value to a human-readable name.
func registerName(reg uint32) string {
	switch reg {
	case 1, 7, 8: // Formal, Technical, Scientific
		return "FORMAL"
	case 2, 3, 4: // Informal, Slang, Vulgar
		return "INFORMAL"
	default:
		return "NEUTRAL"
	}
}
