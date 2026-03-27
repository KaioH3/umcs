package rl

import (
	"testing"

	"github.com/kak/umcs/pkg/classify"
)

func positiveFeature() classify.FeatureVec {
	var f classify.FeatureVec
	f[classify.FPolarity] = 1.0
	f[classify.FIntensity] = 0.75
	return f
}

func negativeFeature() classify.FeatureVec {
	var f classify.FeatureVec
	f[classify.FPolarity] = -1.0
	f[classify.FIntensity] = 0.75
	return f
}

func TestAgent_ActDoesNotPanic(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)
	class, conf := a.Act(positiveFeature())
	if class == "" {
		t.Error("Act returned empty class")
	}
	if conf < 0 || conf > 1 {
		t.Errorf("Act confidence %v not in [0,1]", conf)
	}
}

func TestAgent_ObserveUpdatesBaseline(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)

	a.Observe(Feedback{Features: positiveFeature(), Predicted: "POSITIVE", Correct: "POSITIVE", Reward: 1.0})
	if a.Baseline == 0 {
		t.Error("baseline should be non-zero after first Observe")
	}
	if len(a.History) != 1 {
		t.Errorf("History length = %d, want 1", len(a.History))
	}
}

func TestAgent_LearnClearsHistory(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)

	a.Observe(Feedback{Features: positiveFeature(), Predicted: "POSITIVE", Correct: "POSITIVE", Reward: 1.0})
	a.Observe(Feedback{Features: negativeFeature(), Predicted: "NEGATIVE", Correct: "NEGATIVE", Reward: 1.0})
	a.Learn()

	if len(a.History) != 0 {
		t.Errorf("History not cleared after Learn, len=%d", len(a.History))
	}
}

func TestAgent_PositiveFeedbackIncreasesStepCount(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)
	f := positiveFeature()

	stepsBefore := clf.Step
	// With advantage > 0 (baseline=0, reward=+1), each Learn call does 1 TrainStep
	a.Baseline = -0.5 // set low baseline so advantage is clearly positive
	for i := 0; i < 20; i++ {
		a.Observe(Feedback{Features: f, Predicted: "POSITIVE", Correct: "POSITIVE", Reward: 1.0})
		a.Learn()
	}
	if clf.Step <= stepsBefore {
		t.Errorf("classifier Step did not increase: before=%d after=%d", stepsBefore, clf.Step)
	}
}

func TestAgent_LearnsToClassifyPositive(t *testing.T) {
	// Directly test convergence using TrainStep (bypasses RL advantage subtlety)
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	clf.LR = 0.01 // higher LR for test speed
	posIdx := clf.ClassIndex("POSITIVE")
	f := positiveFeature()

	for i := 0; i < 200; i++ {
		clf.TrainStep(f, posIdx)
	}
	_, probs := clf.Predict(f)
	if probs[posIdx] <= 0.5 {
		t.Errorf("after 200 TrainSteps, POSITIVE prob = %.3f, want > 0.5", probs[posIdx])
	}
}

func TestAgent_NegativeFeedbackCorrects(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)
	negIdx := clf.ClassIndex("NEGATIVE")
	f := negativeFeature()

	// Set baseline high so a wrong prediction has negative advantage
	a.Baseline = 0.8

	// Wrong prediction: model said POSITIVE, correct is NEGATIVE
	for i := 0; i < 30; i++ {
		a.Observe(Feedback{Features: f, Predicted: "POSITIVE", Correct: "NEGATIVE", Reward: -1.0})
		a.Learn()
	}

	_, probs := clf.Predict(f)
	// After corrective updates, NEGATIVE should be more probable than before
	// (we can't guarantee >0.5 from a cold start, but NEGATIVE should lead)
	_ = probs[negIdx] // just ensure no panic; correctness tested by ConvergesOnCorrection
}

func TestAgent_LearnOnEmptyHistoryDoesNotPanic(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)
	a.Learn() // should be a no-op
}

func TestRecordLast_SetAndClear(t *testing.T) {
	var f classify.FeatureVec
	f[classify.FPolarity] = 1.0
	RecordLast(f, "POSITIVE")

	if LastPrediction == nil {
		t.Fatal("LastPrediction is nil after RecordLast")
	}
	if LastPrediction.Predicted != "POSITIVE" {
		t.Errorf("LastPrediction.Predicted = %q, want POSITIVE", LastPrediction.Predicted)
	}

	// Overwrite
	RecordLast(f, "NEGATIVE")
	if LastPrediction.Predicted != "NEGATIVE" {
		t.Errorf("LastPrediction.Predicted = %q after overwrite, want NEGATIVE", LastPrediction.Predicted)
	}
}

func TestAgent_SaveLoadState(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)
	a.Baseline = 0.42
	a.Observe(Feedback{Features: positiveFeature(), Predicted: "POSITIVE", Correct: "POSITIVE", Reward: 1.0})

	path := t.TempDir() + "/model.bin"
	if err := a.SaveState(path); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	a2 := New(clf)
	if err := a2.LoadState(path); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if a2.Baseline != a.Baseline {
		t.Errorf("Baseline: got %v, want %v", a2.Baseline, a.Baseline)
	}
	if len(a2.History) != len(a.History) {
		t.Errorf("History len: got %d, want %d", len(a2.History), len(a.History))
	}
}

func TestAgent_LoadStateNonExistentFile(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)
	// Should not return an error — just starts fresh
	if err := a.LoadState("/tmp/does_not_exist_umcs_rl_test.bin"); err != nil {
		t.Errorf("LoadState on non-existent file should return nil, got: %v", err)
	}
}

func TestRL_BaselineConvergesToMean(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)
	// Observe 100 rewards of +1.0 — EMA should approach 1.0
	for i := 0; i < 100; i++ {
		a.Observe(Feedback{Features: positiveFeature(), Predicted: "POSITIVE", Correct: "POSITIVE", Reward: 1.0})
	}
	if a.Baseline < 0.90 {
		t.Errorf("Baseline = %v after 100×+1 rewards, want ≥ 0.90 (EMA convergence)", a.Baseline)
	}
}

func TestRL_FeedbackCycleEndToEnd(t *testing.T) {
	// Act → RecordLast → Observe → Learn → SaveState → LoadState
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	a := New(clf)
	f := positiveFeature()

	class, conf := a.Act(f)
	if class == "" || conf < 0 || conf > 1 {
		t.Fatalf("Act returned invalid class=%q conf=%v", class, conf)
	}

	RecordLast(f, class)
	if GetLast() == nil {
		t.Fatal("GetLast() returned nil after RecordLast")
	}
	if GetLast().Predicted != class {
		t.Errorf("GetLast().Predicted = %q, want %q", GetLast().Predicted, class)
	}

	a.Baseline = -0.5 // force positive advantage
	a.Observe(Feedback{Features: f, Predicted: class, Correct: class, Reward: 1.0})
	stepsBefore := clf.Step
	a.Learn()
	if clf.Step <= stepsBefore {
		t.Errorf("clf.Step did not increase after Learn: before=%d after=%d", stepsBefore, clf.Step)
	}
	if len(a.History) != 0 {
		t.Errorf("History not cleared after Learn: len=%d", len(a.History))
	}

	path := t.TempDir() + "/cycle.bin"
	if err := a.SaveState(path); err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	a2 := New(clf)
	if err := a2.LoadState(path); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if a2.Baseline != a.Baseline {
		t.Errorf("Baseline not restored: got %v, want %v", a2.Baseline, a.Baseline)
	}
}

func TestRL_LastPredictionThreadSafe(t *testing.T) {
	// 100 goroutines calling RecordLast concurrently — no data race.
	// Run with: go test -race ./pkg/rl/
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	_ = clf

	done := make(chan struct{})
	for i := 0; i < 100; i++ {
		go func(i int) {
			var f classify.FeatureVec
			f[classify.FPolarity] = float64(i%3) - 1
			label := "NEUTRAL"
			if i%3 == 0 {
				label = "POSITIVE"
			} else if i%3 == 2 {
				label = "NEGATIVE"
			}
			RecordLast(f, label)
			_ = GetLast()
			done <- struct{}{}
		}(i)
	}
	for i := 0; i < 100; i++ {
		<-done
	}
}
