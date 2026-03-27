package classify

import (
	"math"
	"math/rand"
	"testing"

	"github.com/kak/umcs/pkg/lexdb"
)

// randomWordRecord creates a random WordRecord for property testing.
func randomWordRecord(rng *rand.Rand) lexdb.WordRecord {
	return lexdb.WordRecord{
		WordID:    rng.Uint32(),
		RootID:    rng.Uint32(),
		Lang:      rng.Uint32() % 24,
		Sentiment: rng.Uint32(),
		FreqRank:  rng.Uint32() % 100000,
		Flags:     rng.Uint32(),
	}
}

func randomRootRecord(rng *rand.Rand) lexdb.RootRecord {
	return lexdb.RootRecord{
		RootID:       rng.Uint32(),
		WordCount:    rng.Uint32() % 100,
		LangCoverage: rng.Uint32(),
		AntonymRootID: rng.Uint32() % 2,  // 50% chance of having antonym
		HypernymRootID: rng.Uint32() % 2,
		SynonymRootID:  rng.Uint32() % 2,
	}
}

func randomContext(rng *rand.Rand) Context {
	return Context{
		NegInWindow: rng.Intn(2) == 0,
		IntInWindow: rng.Intn(2) == 0,
		DwnInWindow: rng.Intn(2) == 0,
	}
}

// Property: Extract never panics on any random WordRecord input (nil lex).
func TestProperty_ExtractNeverPanics(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 10_000; i++ {
		w := randomWordRecord(rng)
		root := randomRootRecord(rng)
		ctx := randomContext(rng)
		// nil lex — semantic relation features (ant/hyp/IPA/etym) will be 0
		f := Extract(nil, &w, &root, ctx)
		_ = f
	}
}

// Property: All features in [-1.0, 1.0] for any random input.
func TestProperty_AllFeaturesInRange(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 5_000; i++ {
		w := randomWordRecord(rng)
		root := randomRootRecord(rng)
		ctx := randomContext(rng)
		f := Extract(nil, &w, &root, ctx)
		for j, v := range f {
			if v < -1.0 || v > 1.0 {
				t.Errorf("iteration %d: f[%d] = %v out of [-1,1]", i, j, v)
			}
		}
	}
}

// Property: ZeroLeakyFeatures idempotent for any input.
func TestProperty_ZeroLeakyFeaturesIdempotent(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1_000; i++ {
		var f FeatureVec
		for j := range f {
			f[j] = rng.Float64()*2 - 1 // [-1, 1]
		}
		f.ZeroLeakyFeatures()
		saved := f
		f.ZeroLeakyFeatures() // second call must not change anything
		if f != saved {
			t.Errorf("ZeroLeakyFeatures not idempotent at iteration %d", i)
		}
	}
}

// Property: Softmax output always sums to 1.0 for any float64 slice.
func TestProperty_SoftmaxAlwaysSumsToOne(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1_000; i++ {
		n := rng.Intn(10) + 1
		logits := make([]float64, n)
		for j := range logits {
			// Include edge cases: very large, very small, zero
			logits[j] = (rng.Float64() - 0.5) * 2000
		}
		probs := Softmax(logits)
		sum := 0.0
		for _, p := range probs {
			sum += p
		}
		if math.IsNaN(sum) || math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("iteration %d: softmax sum = %v (logits=%v)", i, sum, logits)
		}
	}
}

// Property: Classifier.Predict never panics on any FeatureVec.
func TestProperty_PredictNeverPanics(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1_000; i++ {
		var f FeatureVec
		for j := range f {
			f[j] = (rng.Float64() - 0.5) * 4 // outside [-1,1] on purpose
		}
		pred, probs := clf.Predict(f)
		if pred == "" {
			t.Errorf("iteration %d: Predict returned empty string", i)
		}
		for j, p := range probs {
			if math.IsNaN(p) || math.IsInf(p, 0) {
				t.Errorf("iteration %d: probs[%d] = %v", i, j, p)
			}
		}
	}
}
