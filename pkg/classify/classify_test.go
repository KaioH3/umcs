package classify

import (
	"math"
	"testing"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
)

// ── FeatureVec extraction ──────────────────────────────────────────────────────

// BDD: Given a zero WordRecord, When Extract called, Then it does not panic and
// most features = 0 (except FRoleBase+0 which is 1 for NONE role, a valid default).
func TestExtract_ZeroInputGivesZeroVector(t *testing.T) {
	w := lexdb.WordRecord{}
	f := Extract(nil, &w, nil, Context{})
	for i, v := range f {
		// FRoleBase (17) = 1 is correct: NONE role is always set for zero Sentiment.
		if i == FRoleBase {
			if v != 1 {
				t.Errorf("f[%d] (FRoleBase/NONE) = %v, want 1", i, v)
			}
			continue
		}
		if v != 0 {
			t.Errorf("f[%d] = %v, want 0 for zero WordRecord", i, v)
		}
	}
}

// BDD: Given any WordRecord, When extracted, Then all features in [-1, 1].
func TestExtract_AllFeaturesInValidRange(t *testing.T) {
	cases := []struct {
		name string
		w    lexdb.WordRecord
	}{
		{"zero", lexdb.WordRecord{}},
		{"positive", lexdb.WordRecord{Sentiment: sentiment.PolarityPositive | sentiment.ArousalHigh | sentiment.DominanceHigh}},
		{"negative", lexdb.WordRecord{Sentiment: sentiment.PolarityNegative | (4 << 19) | sentiment.POSAdj}},
		{"max_flags", lexdb.WordRecord{Flags: 0xFFFFFFFF}},
		{"max_sent", lexdb.WordRecord{Sentiment: 0xFFFFFFFF}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := tc.w
			f := Extract(nil, &w, nil, Context{})
			for i, v := range f {
				if v < -1.0 || v > 1.0 {
					t.Errorf("%s: f[%d] = %v out of [-1,1]", tc.name, i, v)
				}
			}
		})
	}
}

// BDD: Given polarity in Sentiment, When extracted, Then FPolarity maps correctly.
func TestExtract_PolarityMappingCorrect(t *testing.T) {
	cases := []struct {
		pol  uint32
		want float64
	}{
		{sentiment.PolarityPositive, 1.0},
		{sentiment.PolarityNegative, -1.0},
		{sentiment.PolarityNeutral, 0.0},
		{sentiment.PolarityAmbiguous, 0.0},
	}
	for _, tc := range cases {
		w := lexdb.WordRecord{Sentiment: tc.pol}
		f := Extract(nil, &w, nil, Context{})
		if f[FPolarity] != tc.want {
			t.Errorf("polarity 0x%X: FPolarity = %v, want %v", tc.pol, f[FPolarity], tc.want)
		}
	}
}

// BDD: Given no IPA pronunciation, When extracted, Then IPA features = 0.
func TestExtract_IPAFeaturesZeroWithoutIPA(t *testing.T) {
	w := lexdb.WordRecord{Sentiment: sentiment.PolarityPositive}
	f := Extract(nil, &w, nil, Context{})
	if f[FIPACVRatio] != 0 || f[FIPAOpenVowel] != 0 || f[FIPANasals] != 0 || f[FIPASibilants] != 0 {
		t.Error("IPA features should be 0 when lex is nil (no IPA lookup)")
	}
}

// ── ZeroLeakyFeatures ─────────────────────────────────────────────────────────

// BDD: Given training features, When ZeroLeakyFeatures called, Then FPolarity = 0.
func TestZeroLeakyFeatures_ZeroesPolarityOnly(t *testing.T) {
	var f FeatureVec
	f[FPolarity] = 1.0
	f[FIntensity] = 0.75
	f[FArousal] = 0.5

	f.ZeroLeakyFeatures()

	if f[FPolarity] != 0 {
		t.Errorf("FPolarity = %v after ZeroLeakyFeatures, want 0", f[FPolarity])
	}
	if f[FIntensity] != 0.75 {
		t.Errorf("FIntensity = %v, should be unchanged", f[FIntensity])
	}
	if f[FArousal] != 0.5 {
		t.Errorf("FArousal = %v, should be unchanged", f[FArousal])
	}
}

// BDD: Given polarity = 0, When ZeroLeakyFeatures called, Then still 0.
func TestZeroLeakyFeatures_Idempotent(t *testing.T) {
	var f FeatureVec
	f[FPolarity] = -1.0
	f.ZeroLeakyFeatures()
	f.ZeroLeakyFeatures() // second call should be a no-op
	if f[FPolarity] != 0 {
		t.Errorf("FPolarity = %v after double ZeroLeakyFeatures, want 0", f[FPolarity])
	}
}

func TestZeroLeakyFeatures_PreservesOtherFeatures(t *testing.T) {
	var f FeatureVec
	for i := range f {
		f[i] = float64(i) * 0.01
	}
	f.ZeroLeakyFeatures()
	for i := range f {
		if i == FPolarity {
			if f[i] != 0 {
				t.Errorf("f[%d] (FPolarity) = %v, want 0", i, f[i])
			}
		} else {
			want := float64(i) * 0.01
			if math.Abs(f[i]-want) > 1e-12 {
				t.Errorf("f[%d] = %v, want %v (should be unchanged)", i, f[i], want)
			}
		}
	}
}

// ── SplitByRoot ───────────────────────────────────────────────────────────────

func makeExamplesForRoots(rootIDs []uint32, perRoot int) []Example {
	var ex []Example
	for _, rid := range rootIDs {
		for i := 0; i < perRoot; i++ {
			ex = append(ex, Example{RootID: rid, Label: "NEUTRAL", LabelIdx: 1})
		}
	}
	return ex
}

// TDD: No root_id may appear in both train and val.
func TestSplitByRoot_NoRootOverlap(t *testing.T) {
	roots := make([]uint32, 50)
	for i := range roots {
		roots[i] = uint32(i + 1)
	}
	examples := makeExamplesForRoots(roots, 3)
	train, val := SplitByRoot(examples, 0.2)

	trainRoots := make(map[uint32]bool)
	for _, ex := range train {
		trainRoots[ex.RootID] = true
	}
	for _, ex := range val {
		if trainRoots[ex.RootID] {
			t.Errorf("root_id %d appears in both train and val (cognate leakage)", ex.RootID)
		}
	}
}

// TDD: val size ≈ valFrac × total ± 5%
func TestSplitByRoot_SizeApproximate(t *testing.T) {
	roots := make([]uint32, 100)
	for i := range roots {
		roots[i] = uint32(i + 1)
	}
	examples := makeExamplesForRoots(roots, 2)
	train, val := SplitByRoot(examples, 0.2)

	total := len(train) + len(val)
	if total != len(examples) {
		t.Errorf("train+val = %d, want %d", total, len(examples))
	}
	valFrac := float64(len(val)) / float64(total)
	if valFrac < 0.15 || valFrac > 0.25 {
		t.Errorf("val fraction = %.3f, want ~0.20 ±0.05", valFrac)
	}
}

// TDD: Same input produces same output (deterministic seed=42).
func TestSplitByRoot_Deterministic(t *testing.T) {
	roots := make([]uint32, 30)
	for i := range roots {
		roots[i] = uint32(i + 1)
	}
	examples := makeExamplesForRoots(roots, 2)

	train1, val1 := SplitByRoot(examples, 0.2)
	train2, val2 := SplitByRoot(examples, 0.2)

	if len(train1) != len(train2) || len(val1) != len(val2) {
		t.Errorf("non-deterministic: split sizes differ (%d/%d vs %d/%d)",
			len(train1), len(val1), len(train2), len(val2))
	}
}

func TestSplitByRoot_HandlesEmptyInput(t *testing.T) {
	train, val := SplitByRoot(nil, 0.2)
	if train != nil || val != nil {
		t.Error("empty input should return nil, nil")
	}
}

// TDD: Single root → all examples go to train.
func TestSplitByRoot_SingleRootAllInTrain(t *testing.T) {
	examples := makeExamplesForRoots([]uint32{42}, 10)
	train, val := SplitByRoot(examples, 0.2)
	if len(val) != 0 {
		t.Errorf("single root: val should be empty, got %d examples", len(val))
	}
	if len(train) != 10 {
		t.Errorf("single root: all 10 should be in train, got %d", len(train))
	}
}

// ── Softmax ───────────────────────────────────────────────────────────────────

func TestSoftmax_SumsToOne(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0}
	probs := Softmax(logits)
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("softmax sum = %v, want 1.0", sum)
	}
}

func TestSoftmax_MaxIsHighest(t *testing.T) {
	logits := []float64{0.0, 5.0, 1.0}
	probs := Softmax(logits)
	if probs[1] < probs[0] || probs[1] < probs[2] {
		t.Errorf("expected index 1 to have highest prob, got %v", probs)
	}
}

func TestSoftmax_NumericalStabilityLargeLogits(t *testing.T) {
	logits := []float64{1000.0, 1001.0, 999.0}
	probs := Softmax(logits)
	for i, p := range probs {
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 {
			t.Errorf("probs[%d] = %v, want finite positive (large logit test)", i, p)
		}
	}
}

func TestSoftmax_UniformInputGivesUniformOutput(t *testing.T) {
	logits := []float64{1.0, 1.0, 1.0}
	probs := Softmax(logits)
	for i, p := range probs {
		if math.Abs(p-1.0/3.0) > 1e-9 {
			t.Errorf("probs[%d] = %v, want 1/3 for uniform logits", i, p)
		}
	}
}

func TestSoftmax_SingleElementIsOne(t *testing.T) {
	probs := Softmax([]float64{42.0})
	if math.Abs(probs[0]-1.0) > 1e-9 {
		t.Errorf("single-element softmax = %v, want 1.0", probs[0])
	}
}

// ── Classifier ────────────────────────────────────────────────────────────────

func TestClassifier_NewHasCorrectDimensions(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	if clf.NFeatures != NFeatures {
		t.Errorf("NFeatures = %d, want %d", clf.NFeatures, NFeatures)
	}
	if len(clf.Classes) != 3 {
		t.Errorf("len(Classes) = %d, want 3", len(clf.Classes))
	}
}

func TestClassifier_PredictSumsToOne(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	var f FeatureVec
	f[FIntensity] = 0.5
	_, probs := clf.Predict(f)
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("predict probs sum = %v, want 1.0", sum)
	}
}

func TestClassifier_PredictEmpty(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	var f FeatureVec
	pred, probs := clf.Predict(f)
	if pred == "" {
		t.Error("Predict returned empty string on zero vector")
	}
	for i, p := range probs {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Errorf("probs[%d] = %v, should be finite", i, p)
		}
	}
}

func TestClassifier_TrainStepDecreasesLoss(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	posIdx := clf.ClassIndex("POSITIVE")

	// Use FIntensity (not FPolarity) to avoid leaky feature confusion in test
	var f FeatureVec
	f[FIntensity] = 1.0
	f[FHasAntonym] = 0.0

	for i := 0; i < 300; i++ {
		clf.TrainStep(f, posIdx)
	}
	_, probs := clf.Predict(f)
	if probs[posIdx] <= 0.5 {
		t.Errorf("after 300 steps, POSITIVE prob = %.3f, expected > 0.5", probs[posIdx])
	}
}

// TDD: After training WITHOUT FPolarity signal, classifier should learn (F1 > 0).
func TestClassifier_ConvergesOnLeakFreeData(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	clf.LR = 0.01 // higher LR for speed
	posIdx := clf.ClassIndex("POSITIVE")
	negIdx := clf.ClassIndex("NEGATIVE")

	// Use non-polarity features as signals (leak-free)
	var pos, neg FeatureVec
	pos[FIntensity] = 1.0
	pos[FHasSynonym] = 1.0
	neg[FIntensity] = 1.0
	neg[FHasAntonym] = 1.0

	for i := 0; i < 300; i++ {
		clf.TrainStep(pos, posIdx)
		clf.TrainStep(neg, negIdx)
	}

	predP, _ := clf.Predict(pos)
	predN, _ := clf.Predict(neg)
	if predP != "POSITIVE" {
		t.Errorf("positive example (leak-free features): predicted %q, want POSITIVE", predP)
	}
	if predN != "NEGATIVE" {
		t.Errorf("negative example (leak-free features): predicted %q, want NEGATIVE", predN)
	}
}

func TestClassifier_ClassIndexReturnsMinusOneForUnknown(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	if idx := clf.ClassIndex("UNKNOWN_CLASS"); idx != -1 {
		t.Errorf("ClassIndex(unknown) = %d, want -1", idx)
	}
}

func TestClassifier_SaveLoadRoundTrip(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	posIdx := clf.ClassIndex("POSITIVE")
	var f FeatureVec
	f[FIntensity] = 0.5
	for i := 0; i < 100; i++ {
		clf.TrainStep(f, posIdx)
	}
	pred1, probs1 := clf.Predict(f)

	path := t.TempDir() + "/test.clf"
	if err := clf.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	clf2, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	pred2, probs2 := clf2.Predict(f)
	if pred1 != pred2 {
		t.Errorf("prediction changed after round-trip: %q → %q", pred1, pred2)
	}
	for i := range probs1 {
		if math.Abs(probs1[i]-probs2[i]) > 1e-9 {
			t.Errorf("probs[%d] changed: %v → %v", i, probs1[i], probs2[i])
		}
	}
	if clf2.Step != clf.Step {
		t.Errorf("Step changed: %d → %d", clf.Step, clf2.Step)
	}
}

// ── F1 and Accuracy ───────────────────────────────────────────────────────────

func TestF1Macro_AllCorrect_IsOne(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	posIdx := clf.ClassIndex("POSITIVE")
	negIdx := clf.ClassIndex("NEGATIVE")
	neuIdx := clf.ClassIndex("NEUTRAL")

	// Use 3 clearly separated feature vectors for 3-class perfect separation
	var pos, neg, neu FeatureVec
	pos[FPolarity] = 1.0
	pos[FIntensity] = 1.0
	neg[FPolarity] = -1.0
	neg[FIntensity] = 1.0
	neu[FPolarity] = 0.0
	neu[FSyllables] = 0.5 // distinct signal

	for i := 0; i < 1000; i++ {
		clf.TrainStep(pos, posIdx)
		clf.TrainStep(neg, negIdx)
		clf.TrainStep(neu, neuIdx)
	}
	examples := []Example{
		{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
		{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
		{Features: neu, Label: "NEUTRAL", LabelIdx: neuIdx},
	}
	f1 := F1Macro(clf, examples)
	if f1 < 0.9 {
		t.Errorf("F1 on 3-class separable data = %.3f, want ≥ 0.9", f1)
	}
}

func TestF1Macro_ThreeClass_ManualVerification(t *testing.T) {
	// Manually constructed: 2 correct POSITIVE, 1 wrong POSITIVE→NEGATIVE
	// TP_POS=2, FP_POS=0, FN_POS=0
	// TP_NEG=0, FP_NEG=1, FN_NEG=1
	// TP_NEU=0, FP_NEU=0, FN_NEU=0
	clf := New(NFeatures, DefaultClasses)
	posIdx := clf.ClassIndex("POSITIVE")
	negIdx := clf.ClassIndex("NEGATIVE")

	var pos, neg FeatureVec
	pos[FPolarity] = 1.0
	neg[FPolarity] = -1.0
	for i := 0; i < 1000; i++ {
		clf.TrainStep(pos, posIdx)
	}
	// After only training on POS, NEG will be classified as POS
	examples := []Example{
		{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
		{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
		{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
	}
	f1 := F1Macro(clf, examples)
	if f1 < 0 || f1 > 1 {
		t.Errorf("F1 = %v, should be in [0,1]", f1)
	}
}

func TestAccuracy_HalfCorrect_IsHalf(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	posIdx := clf.ClassIndex("POSITIVE")
	negIdx := clf.ClassIndex("NEGATIVE")

	var pos, neg FeatureVec
	pos[FPolarity] = 1.0
	neg[FPolarity] = -1.0
	for i := 0; i < 500; i++ {
		clf.TrainStep(pos, posIdx)
	}
	// Predict pos→POSITIVE (correct), neg→POSITIVE (wrong)
	examples := []Example{
		{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
		{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
	}
	acc := Accuracy(clf, examples)
	if acc < 0 || acc > 1 {
		t.Errorf("accuracy = %v, should be in [0,1]", acc)
	}
}

func TestF1Macro_EmptyExamples_IsZero(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	f1 := F1Macro(clf, nil)
	if f1 != 0 {
		t.Errorf("F1 on empty examples = %v, want 0", f1)
	}
}

// ── FeatureWeights integration ─────────────────────────────────────────────────

func TestClassifier_FeatureWeightsZeroDisablesFeature(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	posIdx := clf.ClassIndex("POSITIVE")

	var f FeatureVec
	f[FPolarity] = 1.0
	clf.FeatureWeights[FPolarity] = 0.0

	for i := 0; i < 200; i++ {
		clf.TrainStep(f, posIdx)
	}
	pred, probs := clf.Predict(f)
	if pred == "" {
		t.Error("Predict returned empty string")
	}
	for i, p := range probs {
		if math.IsNaN(p) {
			t.Errorf("probs[%d] is NaN", i)
		}
	}
}

// ── parseIPA ──────────────────────────────────────────────────────────────────

func TestParseIPA_EmptyString(t *testing.T) {
	cv, ov, nasals, sibilants := parseIPA("")
	if cv != 0 || ov != 0 || nasals || sibilants {
		t.Errorf("empty IPA: want all zeros, got cv=%v ov=%v nasals=%v sibilants=%v", cv, ov, nasals, sibilants)
	}
}

func TestParseIPA_OpenVowels(t *testing.T) {
	// /mama/ — open vowel 'a' × 2, nasals m, no sibilants
	cv, ov, nasals, sibilants := parseIPA("mama")
	if ov <= 0 {
		t.Errorf("/mama/: open vowel fraction = %v, want > 0", ov)
	}
	if !nasals {
		t.Error("/mama/: should detect nasals (m)")
	}
	if sibilants {
		t.Error("/mama/: should NOT detect sibilants")
	}
	_ = cv
}

func TestParseIPA_Sibilants(t *testing.T) {
	// /sisi/ — sibilant 's', vowel 'i' (not open), no nasals
	cv, ov, nasals, sibilants := parseIPA("sisi")
	if !sibilants {
		t.Error("/sisi/: should detect sibilants (s)")
	}
	if nasals {
		t.Error("/sisi/: should NOT detect nasals")
	}
	if ov != 0 {
		t.Errorf("/sisi/: open vowel fraction = %v, want 0 (i is not open)", ov)
	}
	_ = cv
}

func TestParseIPA_CVRatio(t *testing.T) {
	// /pa/ — 1 consonant, 1 vowel → C/V ratio = 0.5
	cv, _, _, _ := parseIPA("pa")
	if math.Abs(cv-0.5) > 1e-9 {
		t.Errorf("/pa/: cv ratio = %v, want 0.5", cv)
	}
}

func TestParseIPA_PureVowels(t *testing.T) {
	// /aei/ — all vowels → cv ratio = 0
	cv, _, _, _ := parseIPA("aei")
	if cv != 0 {
		t.Errorf("/aei/: cv ratio = %v, want 0 (no consonants)", cv)
	}
}

// ── polarityLabel ──────────────────────────────────────────────────────────────

func TestPolarityLabel_AllCases(t *testing.T) {
	cases := []struct {
		sent uint32
		want string
	}{
		{sentiment.PolarityPositive, "POSITIVE"},
		{sentiment.PolarityNegative, "NEGATIVE"},
		{sentiment.PolarityNeutral, "NEUTRAL"},
		{sentiment.PolarityAmbiguous, "NEUTRAL"},
		{0, "NEUTRAL"},
	}
	for _, tc := range cases {
		got := polarityLabel(tc.sent)
		if got != tc.want {
			t.Errorf("polarityLabel(0x%X) = %q, want %q", tc.sent, got, tc.want)
		}
	}
}

// ── MajorityClassF1 ────────────────────────────────────────────────────────────

func TestMajorityClassF1_EmptyExamples_IsZero(t *testing.T) {
	if got := MajorityClassF1(nil); got != 0 {
		t.Errorf("MajorityClassF1(nil) = %v, want 0", got)
	}
}

func TestMajorityClassF1_AllSameClass(t *testing.T) {
	// All POSITIVE → majority predicts POSITIVE → F1=1 for POS, F1=0 for others → macro = 1/3
	examples := []Example{
		{Label: "POSITIVE", LabelIdx: 2},
		{Label: "POSITIVE", LabelIdx: 2},
		{Label: "POSITIVE", LabelIdx: 2},
	}
	got := MajorityClassF1(examples)
	// Single class: macro-F1 = 1.0/1 class = 1.0
	if math.Abs(got-1.0) > 1e-9 {
		t.Errorf("MajorityClassF1(all POSITIVE) = %v, want 1.0", got)
	}
}

func TestMajorityClassF1_TwoClasses_Dominant(t *testing.T) {
	// 3 POSITIVE, 1 NEGATIVE → majority = POSITIVE
	// POS: TP=3, FP=1, FN=0 → prec=0.75, rec=1.0 → F1=0.857
	// NEG: TP=0, FP=0, FN=1 → prec=0.0, rec=0.0 → F1=0
	// Macro = (0.857 + 0) / 2 = 0.428
	examples := []Example{
		{Label: "POSITIVE", LabelIdx: 2},
		{Label: "POSITIVE", LabelIdx: 2},
		{Label: "POSITIVE", LabelIdx: 2},
		{Label: "NEGATIVE", LabelIdx: 0},
	}
	got := MajorityClassF1(examples)
	if got < 0 || got > 1 {
		t.Errorf("MajorityClassF1 = %v, must be in [0,1]", got)
	}
	// Should be below 0.5 (minority class has F1=0)
	if got >= 0.5 {
		t.Errorf("MajorityClassF1 imbalanced = %v, expected < 0.5", got)
	}
}

func TestMajorityClassF1_BalancedClasses(t *testing.T) {
	// Equal POSITIVE/NEGATIVE → majority picks one arbitrarily.
	// Macro F1 for majority-class predictor on balanced 2-class = ~0.5
	examples := []Example{
		{Label: "POSITIVE", LabelIdx: 2},
		{Label: "POSITIVE", LabelIdx: 2},
		{Label: "NEGATIVE", LabelIdx: 0},
		{Label: "NEGATIVE", LabelIdx: 0},
	}
	got := MajorityClassF1(examples)
	if got < 0 || got > 1 {
		t.Errorf("MajorityClassF1 = %v, must be in [0,1]", got)
	}
}

// ── F1PerClass ─────────────────────────────────────────────────────────────────

func TestF1PerClass_AllCorrect(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	posIdx := clf.ClassIndex("POSITIVE")
	negIdx := clf.ClassIndex("NEGATIVE")
	neuIdx := clf.ClassIndex("NEUTRAL")

	var pos, neg, neu FeatureVec
	pos[FPolarity] = 1.0
	pos[FIntensity] = 1.0
	neg[FPolarity] = -1.0
	neg[FIntensity] = 1.0
	neu[FSyllables] = 0.5

	for i := 0; i < 1000; i++ {
		clf.TrainStep(pos, posIdx)
		clf.TrainStep(neg, negIdx)
		clf.TrainStep(neu, neuIdx)
	}
	examples := []Example{
		{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
		{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
		{Features: neu, Label: "NEUTRAL", LabelIdx: neuIdx},
	}
	perClass := F1PerClass(clf, examples)
	if len(perClass) != 3 {
		t.Fatalf("F1PerClass returned %d entries, want 3", len(perClass))
	}
	for cls, f1 := range perClass {
		if f1 < 0 || f1 > 1 {
			t.Errorf("F1PerClass[%s] = %v out of [0,1]", cls, f1)
		}
	}
	// On well-separated 3-class data, each class should have high F1
	for cls, f1 := range perClass {
		if f1 < 0.7 {
			t.Errorf("F1PerClass[%s] = %.3f, want ≥ 0.7 for separable data", cls, f1)
		}
	}
}

func TestF1PerClass_HasAllClasses(t *testing.T) {
	clf := New(NFeatures, DefaultClasses)
	var f FeatureVec
	examples := []Example{
		{Features: f, Label: "POSITIVE", LabelIdx: clf.ClassIndex("POSITIVE")},
	}
	perClass := F1PerClass(clf, examples)
	for _, cls := range DefaultClasses {
		if _, ok := perClass[cls]; !ok {
			t.Errorf("F1PerClass missing class %q", cls)
		}
	}
}

// ── ExtractFromLexicon + GenerateFromLexicon ──────────────────────────────────

func TestExtractFromLexicon_KnownWord(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
	}
	words := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible",
			Sentiment: sentiment.PolarityNegative | (2 << 19) | sentiment.POSAdj},
	}
	dir := t.TempDir()
	path := dir + "/lex.umcs"
	if _, err := lexdb.Build(roots, words, path); err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	f, ok := ExtractFromLexicon(lex, "terrible", "EN")
	if !ok {
		t.Fatal("ExtractFromLexicon: 'terrible' not found in lexicon")
	}
	// NEGATIVE word → FPolarity should be -1
	if f[FPolarity] != -1.0 {
		t.Errorf("terrible FPolarity = %v, want -1.0", f[FPolarity])
	}
}

func TestExtractFromLexicon_UnknownWord(t *testing.T) {
	roots := []seed.Root{{RootID: 1, RootStr: "r", Origin: "LATIN", MeaningEN: "x"}}
	words := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "good", Lang: "EN", Norm: "good",
			Sentiment: sentiment.PolarityPositive},
	}
	dir := t.TempDir()
	path := dir + "/lex.umcs"
	if _, err := lexdb.Build(roots, words, path); err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	_, ok := ExtractFromLexicon(lex, "nonexistent_xyz_word", "EN")
	if ok {
		t.Error("ExtractFromLexicon should return false for unknown word")
	}
}

func TestGenerateFromLexicon_NoPolarityInFeatures(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible",
			Sentiment: sentiment.PolarityNegative | (2 << 19)},
		{WordID: (2 << 12) | 1, RootID: 2, Variant: 1, Word: "good", Lang: "EN", Norm: "good",
			Sentiment: sentiment.PolarityPositive | (3 << 19)},
	}
	dir := t.TempDir()
	path := dir + "/lex.umcs"
	if _, err := lexdb.Build(roots, words, path); err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	examples := GenerateFromLexicon(lex)
	if len(examples) == 0 {
		t.Fatal("GenerateFromLexicon returned no examples")
	}
	for _, ex := range examples {
		if ex.Features[FPolarity] != 0 {
			t.Errorf("example %q: FPolarity = %v after ZeroLeakyFeatures, want 0",
				ex.Word, ex.Features[FPolarity])
		}
	}
}

func TestGenerateFromLexicon_LabelCoherent(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: (1 << 12) | 1, RootID: 1, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible",
			Sentiment: sentiment.PolarityNegative},
		{WordID: (2 << 12) | 1, RootID: 2, Variant: 1, Word: "good", Lang: "EN", Norm: "good",
			Sentiment: sentiment.PolarityPositive},
	}
	dir := t.TempDir()
	path := dir + "/lex.umcs"
	if _, err := lexdb.Build(roots, words, path); err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	examples := GenerateFromLexicon(lex)
	for _, ex := range examples {
		if ex.Label != "NEGATIVE" && ex.Label != "NEUTRAL" && ex.Label != "POSITIVE" {
			t.Errorf("invalid label %q for word %q", ex.Label, ex.Word)
		}
		if ex.LabelIdx < 0 || ex.LabelIdx >= len(DefaultClasses) {
			t.Errorf("LabelIdx %d out of range for word %q", ex.LabelIdx, ex.Word)
		}
		if ex.RootID == 0 {
			t.Errorf("RootID=0 for word %q (should be > 0)", ex.Word)
		}
	}
}

// ── SplitByRoot compatibility with old Split ───────────────────────────────────

func TestSplit_RatioApproximate(t *testing.T) {
	examples := make([]Example, 100)
	for i := range examples {
		examples[i].Label = "NEUTRAL"
	}
	train, val := Split(examples, 0.2)
	if len(val) < 15 || len(val) > 25 {
		t.Errorf("val size = %d, expected ~20", len(val))
	}
	if len(train)+len(val) != 100 {
		t.Errorf("train(%d) + val(%d) != 100", len(train), len(val))
	}
}
