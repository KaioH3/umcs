package classify

import (
	"math/rand"
	"testing"

	"github.com/kak/umcs/pkg/lexdb"
)

func BenchmarkSoftmax_3Class(b *testing.B) {
	logits := []float64{1.2, -0.5, 0.8}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Softmax(logits)
	}
}

func BenchmarkExtract(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	w := lexdb.WordRecord{
		Sentiment: 0x00C30040, // POSITIVE, STRONG, AROUSAL_MED
		Flags:     0x1C400200, // 7 syllables, stress penult, ADJ, formal register
		Lang:      1,          // EN
		FreqRank:  1000,
	}
	root := lexdb.RootRecord{
		RootID:         rng.Uint32(),
		LangCoverage:   0x3F, // 6 languages
		AntonymRootID:  1,
		HypernymRootID: 2,
	}
	ctx := Context{IntInWindow: true}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Extract(nil, &w, &root, ctx)
	}
}

func BenchmarkClassifier_Predict(b *testing.B) {
	clf := New(NFeatures, DefaultClasses)
	var f FeatureVec
	f[FPolarity] = 1.0
	f[FIntensity] = 0.75
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = clf.Predict(f)
	}
}

func BenchmarkClassifier_TrainStep(b *testing.B) {
	clf := New(NFeatures, DefaultClasses)
	posIdx := clf.ClassIndex("POSITIVE")
	var f FeatureVec
	f[FPolarity] = 1.0
	f[FIntensity] = 0.75
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		clf.TrainStep(f, posIdx)
	}
}

func BenchmarkSplitByRoot(b *testing.B) {
	// Build 2440 examples across 364 roots (matches production data size)
	examples := make([]Example, 0, 2440)
	for rootID := uint32(1); rootID <= 364; rootID++ {
		for v := 0; v < 6; v++ {
			examples = append(examples, Example{
				RootID:   rootID,
				Label:    "NEUTRAL",
				LabelIdx: 1,
			})
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = SplitByRoot(examples, 0.2)
	}
}
