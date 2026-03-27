package classify

import (
	"math/rand"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/sentiment"
)

// DefaultClasses is the standard three-way sentiment classification target.
var DefaultClasses = []string{"NEGATIVE", "NEUTRAL", "POSITIVE"}

// Example is one labeled training sample.
type Example struct {
	Features FeatureVec
	Label    string // "NEGATIVE", "NEUTRAL", "POSITIVE"
	LabelIdx int    // index into Classifier.Classes
	RootID   uint32 // root_id for root-stratified split (prevents cognate leakage)
	Word     string // surface form (for debugging / error analysis)
	Lang     string // ISO 639-1 language code
}

// polarityLabel maps a packed sentiment uint32 to a class label.
func polarityLabel(s uint32) string {
	switch sentiment.Polarity(s) {
	case sentiment.PolarityPositive:
		return "POSITIVE"
	case sentiment.PolarityNegative:
		return "NEGATIVE"
	default:
		return "NEUTRAL"
	}
}

// GenerateFromLexicon creates labeled examples from the compiled lexicon.
// Every word with a non-ambiguous polarity becomes one example.
// The label is derived directly from the stored polarity annotation —
// no human labeling required for the initial training set.
func GenerateFromLexicon(lex *lexdb.Lexicon) []Example {
	var examples []Example
	seen := make(map[uint32]bool, len(lex.Words))

	for i := range lex.Words {
		w := &lex.Words[i]
		if seen[w.WordID] {
			continue
		}
		seen[w.WordID] = true

		// Skip ambiguous polarity — not a reliable training signal
		if sentiment.Polarity(w.Sentiment) == sentiment.PolarityAmbiguous {
			continue
		}

		root := lex.LookupRoot(w.RootID)
		f := Extract(lex, w, root, Context{})
		// Zero the polarity feature to prevent data leakage: FPolarity is
		// derived from the same Sentiment field used to generate the label.
		// Leaving it in causes the classifier to learn the identity function (F1=1.000).
		f.ZeroLeakyFeatures()
		label := polarityLabel(w.Sentiment)

		idx := -1
		for i, cl := range DefaultClasses {
			if cl == label {
				idx = i
				break
			}
		}
		if idx < 0 {
			continue
		}
		examples = append(examples, Example{
			Features: f,
			Label:    label,
			LabelIdx: idx,
			RootID:   w.RootID,
			Word:     lex.WordStr(w),
			Lang:     lexdb.LangName(w.Lang),
		})
	}
	return examples
}

// Split partitions examples into train and validation sets deterministically.
// valFrac is the fraction reserved for validation (e.g. 0.2 = 20%).
// Uses every Nth element for validation so the split is stable across runs.
func Split(examples []Example, valFrac float64) (train, val []Example) {
	n := len(examples)
	if n == 0 {
		return nil, nil
	}
	valN := int(float64(n) * valFrac)
	if valN < 1 {
		valN = 1
	}
	stride := n / valN
	if stride < 1 {
		stride = 1
	}
	for i, ex := range examples {
		if i%stride == 0 && len(val) < valN {
			val = append(val, ex)
		} else {
			train = append(train, ex)
		}
	}
	return train, val
}

// SplitByRoot partitions examples so that all words sharing a root_id go
// entirely to train or entirely to validation — preventing cognate leakage.
//
// Cognate leakage occurs when, for example, "terrible" (EN) is in train and
// "terrível" (PT) is in validation: they share a root and the same polarity,
// so the classifier can trivially generalise without learning real features.
//
// Uses a Fisher-Yates shuffle with seed=42 for deterministic, reproducible
// splits. The first valFrac × nRoots root groups go to validation; the rest
// to train.
func SplitByRoot(examples []Example, valFrac float64) (train, val []Example) {
	if len(examples) == 0 {
		return nil, nil
	}

	// Group examples by root_id
	groups := make(map[uint32][]Example)
	var order []uint32 // preserve insertion order for determinism
	for _, ex := range examples {
		if _, exists := groups[ex.RootID]; !exists {
			order = append(order, ex.RootID)
		}
		groups[ex.RootID] = append(groups[ex.RootID], ex)
	}

	// Fisher-Yates shuffle with fixed seed for reproducibility
	rng := rand.New(rand.NewSource(42))
	for i := len(order) - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		order[i], order[j] = order[j], order[i]
	}

	valN := int(float64(len(order)) * valFrac)
	// Only force at least 1 val root when there are multiple roots.
	// With a single root, all examples go to train (cannot split by root).
	if valN < 1 && len(order) > 1 {
		valN = 1
	}

	for i, rootID := range order {
		if i < valN {
			val = append(val, groups[rootID]...)
		} else {
			train = append(train, groups[rootID]...)
		}
	}
	return train, val
}

// F1Macro computes the macro-averaged F1 score of clf on examples.
// Macro averaging treats each class equally regardless of size,
// which is appropriate for imbalanced sentiment datasets.
func F1Macro(clf *Classifier, examples []Example) float64 {
	nc := len(clf.Classes)
	tp := make([]float64, nc)
	fp := make([]float64, nc)
	fn := make([]float64, nc)

	for _, ex := range examples {
		pred, _ := clf.Predict(ex.Features)
		predIdx := clf.ClassIndex(pred)
		if predIdx == ex.LabelIdx {
			tp[ex.LabelIdx]++
		} else {
			if predIdx >= 0 {
				fp[predIdx]++
			}
			fn[ex.LabelIdx]++
		}
	}

	f1sum := 0.0
	for c := 0; c < nc; c++ {
		prec := 0.0
		if tp[c]+fp[c] > 0 {
			prec = tp[c] / (tp[c] + fp[c])
		}
		rec := 0.0
		if tp[c]+fn[c] > 0 {
			rec = tp[c] / (tp[c] + fn[c])
		}
		if prec+rec > 0 {
			f1sum += 2 * prec * rec / (prec + rec)
		}
	}
	return f1sum / float64(nc)
}

// Accuracy computes plain accuracy (fraction of correct predictions).
func Accuracy(clf *Classifier, examples []Example) float64 {
	if len(examples) == 0 {
		return 0
	}
	correct := 0
	for _, ex := range examples {
		pred, _ := clf.Predict(ex.Features)
		if pred == ex.Label {
			correct++
		}
	}
	return float64(correct) / float64(len(examples))
}

// MajorityClassF1 returns the macro-F1 of the trivial baseline that always
// predicts the most frequent class. Use this as a lower-bound sanity check:
// any trained model must score strictly above this to be considered non-trivial.
func MajorityClassF1(examples []Example) float64 {
	if len(examples) == 0 {
		return 0
	}
	counts := make(map[string]int)
	for _, ex := range examples {
		counts[ex.Label]++
	}
	var majority string
	var maxCount int
	for label, c := range counts {
		if c > maxCount {
			maxCount = c
			majority = label
		}
	}
	// Compute macro-F1 where every example is predicted as majority class.
	classes := make(map[string]bool)
	for _, ex := range examples {
		classes[ex.Label] = true
	}
	f1sum := 0.0
	for cls := range classes {
		var tp, fp, fn float64
		for _, ex := range examples {
			pred := majority
			if pred == cls && ex.Label == cls {
				tp++
			} else if pred == cls && ex.Label != cls {
				fp++
			} else if pred != cls && ex.Label == cls {
				fn++
			}
		}
		prec := 0.0
		if tp+fp > 0 {
			prec = tp / (tp + fp)
		}
		rec := 0.0
		if tp+fn > 0 {
			rec = tp / (tp + fn)
		}
		if prec+rec > 0 {
			f1sum += 2 * prec * rec / (prec + rec)
		}
	}
	return f1sum / float64(len(classes))
}

// F1PerClass returns a map from class name to its F1 score.
func F1PerClass(clf *Classifier, examples []Example) map[string]float64 {
	nc := len(clf.Classes)
	tp := make([]float64, nc)
	fp := make([]float64, nc)
	fn := make([]float64, nc)

	for _, ex := range examples {
		pred, _ := clf.Predict(ex.Features)
		predIdx := clf.ClassIndex(pred)
		if predIdx == ex.LabelIdx {
			tp[ex.LabelIdx]++
		} else {
			if predIdx >= 0 {
				fp[predIdx]++
			}
			fn[ex.LabelIdx]++
		}
	}

	result := make(map[string]float64, nc)
	for c, name := range clf.Classes {
		prec := 0.0
		if tp[c]+fp[c] > 0 {
			prec = tp[c] / (tp[c] + fp[c])
		}
		rec := 0.0
		if tp[c]+fn[c] > 0 {
			rec = tp[c] / (tp[c] + fn[c])
		}
		if prec+rec > 0 {
			result[name] = 2 * prec * rec / (prec + rec)
		} else {
			result[name] = 0
		}
	}
	return result
}
