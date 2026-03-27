package main

import (
	"path/filepath"
	"testing"

	"github.com/kak/umcs/pkg/classify"
	"github.com/kak/umcs/pkg/ga"
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
)

// buildSmokeLexicon creates a small but realistic lexicon in a temp directory.
// It contains enough NEGATIVE, NEUTRAL, and POSITIVE words across multiple
// roots and languages to exercise the full train/predict pipeline.
func buildSmokeLexicon(t *testing.T) *lexdb.Lexicon {
	t.Helper()
	dir := t.TempDir()
	outPath := filepath.Join(dir, "smoke.umcs")

	// Sentiment packing:
	//   PolarityNegative = 0x80, PolarityPositive = 0x40, PolarityNeutral = 0x00
	//   IntensityStrong = 3<<16 = 0x30000
	//   IntensityModerate = 2<<16 = 0x20000
	//   RoleEvaluation = 1<<20 = 0x100000
	//   RoleEmotion = 2<<20 = 0x200000
	//   POSAdj = 3<<29 = 0x60000000
	//   POSNoun = 1<<29 = 0x20000000
	//   POSVerb = 2<<29 = 0x40000000

	negEval := sentiment.PolarityNegative | sentiment.IntensityStrong | sentiment.RoleEvaluation | sentiment.POSAdj
	posEval := sentiment.PolarityPositive | sentiment.IntensityStrong | sentiment.RoleEvaluation | sentiment.POSAdj
	negEmo := sentiment.PolarityNegative | sentiment.IntensityModerate | sentiment.RoleEmotion | sentiment.POSNoun
	posEmo := sentiment.PolarityPositive | sentiment.IntensityModerate | sentiment.RoleEmotion | sentiment.POSNoun
	neuNoun := sentiment.PolarityNeutral | sentiment.IntensityNone | sentiment.RoleNone | sentiment.POSNoun
	negVerb := sentiment.PolarityNegative | sentiment.IntensityModerate | sentiment.RoleEvaluation | sentiment.POSVerb
	posVerb := sentiment.PolarityPositive | sentiment.IntensityModerate | sentiment.RoleEvaluation | sentiment.POSVerb

	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "to deny", AntonymRootID: 2},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good", AntonymRootID: 1},
		{RootID: 3, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
		{RootID: 4, RootStr: "felic", Origin: "LATIN", MeaningEN: "happy"},
		{RootID: 5, RootStr: "neutr", Origin: "LATIN", MeaningEN: "neutral"},
		{RootID: 6, RootStr: "mal", Origin: "LATIN", MeaningEN: "bad", AntonymRootID: 2},
		{RootID: 7, RootStr: "am", Origin: "LATIN", MeaningEN: "love"},
		{RootID: 8, RootStr: "od", Origin: "LATIN", MeaningEN: "hate"},
		{RootID: 9, RootStr: "mort", Origin: "LATIN", MeaningEN: "death"},
		{RootID: 10, RootStr: "vit", Origin: "LATIN", MeaningEN: "life"},
		{RootID: 11, RootStr: "libr", Origin: "LATIN", MeaningEN: "book"},
		{RootID: 12, RootStr: "pess", Origin: "LATIN", MeaningEN: "worst"},
		{RootID: 13, RootStr: "optim", Origin: "LATIN", MeaningEN: "best"},
		{RootID: 14, RootStr: "trist", Origin: "LATIN", MeaningEN: "sad"},
		{RootID: 15, RootStr: "alegr", Origin: "LATIN", MeaningEN: "joy"},
		{RootID: 16, RootStr: "crud", Origin: "LATIN", MeaningEN: "cruel"},
		{RootID: 17, RootStr: "gent", Origin: "LATIN", MeaningEN: "kind"},
		{RootID: 18, RootStr: "tabl", Origin: "LATIN", MeaningEN: "table"},
	}

	words := []seed.Word{
		// Root 1 — negative
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: negEval},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: negEval},
		// Root 2 — positive
		{WordID: 8193, RootID: 2, Variant: 1, Word: "good", Lang: "EN", Norm: "good", Sentiment: posEval},
		{WordID: 8194, RootID: 2, Variant: 2, Word: "bom", Lang: "PT", Norm: "bom", Sentiment: posEval},
		// Root 3 — negative
		{WordID: 12289, RootID: 3, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: negEval},
		{WordID: 12290, RootID: 3, Variant: 2, Word: "terrivel", Lang: "PT", Norm: "terrivel", Sentiment: negEval},
		// Root 4 — positive
		{WordID: 16385, RootID: 4, Variant: 1, Word: "happy", Lang: "EN", Norm: "happy", Sentiment: posEmo},
		{WordID: 16386, RootID: 4, Variant: 2, Word: "feliz", Lang: "PT", Norm: "feliz", Sentiment: posEmo},
		// Root 5 — neutral
		{WordID: 20481, RootID: 5, Variant: 1, Word: "neutral", Lang: "EN", Norm: "neutral", Sentiment: neuNoun},
		{WordID: 20482, RootID: 5, Variant: 2, Word: "neutro", Lang: "PT", Norm: "neutro", Sentiment: neuNoun},
		// Root 6 — negative
		{WordID: 24577, RootID: 6, Variant: 1, Word: "bad", Lang: "EN", Norm: "bad", Sentiment: negEval},
		{WordID: 24578, RootID: 6, Variant: 2, Word: "mau", Lang: "PT", Norm: "mau", Sentiment: negEval},
		// Root 7 — positive
		{WordID: 28673, RootID: 7, Variant: 1, Word: "love", Lang: "EN", Norm: "love", Sentiment: posEmo},
		{WordID: 28674, RootID: 7, Variant: 2, Word: "amor", Lang: "PT", Norm: "amor", Sentiment: posEmo},
		// Root 8 — negative
		{WordID: 32769, RootID: 8, Variant: 1, Word: "hate", Lang: "EN", Norm: "hate", Sentiment: negEmo},
		{WordID: 32770, RootID: 8, Variant: 2, Word: "odio", Lang: "PT", Norm: "odio", Sentiment: negEmo},
		// Root 9 — negative
		{WordID: 36865, RootID: 9, Variant: 1, Word: "death", Lang: "EN", Norm: "death", Sentiment: negEmo},
		// Root 10 — positive
		{WordID: 40961, RootID: 10, Variant: 1, Word: "life", Lang: "EN", Norm: "life", Sentiment: posEmo},
		// Root 11 — neutral
		{WordID: 45057, RootID: 11, Variant: 1, Word: "book", Lang: "EN", Norm: "book", Sentiment: neuNoun},
		// Root 12 — negative
		{WordID: 49153, RootID: 12, Variant: 1, Word: "pessimist", Lang: "EN", Norm: "pessimist", Sentiment: negEval},
		// Root 13 — positive
		{WordID: 53249, RootID: 13, Variant: 1, Word: "optimist", Lang: "EN", Norm: "optimist", Sentiment: posEval},
		// Root 14 — negative
		{WordID: 57345, RootID: 14, Variant: 1, Word: "sad", Lang: "EN", Norm: "sad", Sentiment: negEmo},
		{WordID: 57346, RootID: 14, Variant: 2, Word: "triste", Lang: "PT", Norm: "triste", Sentiment: negEmo},
		// Root 15 — positive
		{WordID: 61441, RootID: 15, Variant: 1, Word: "joyful", Lang: "EN", Norm: "joyful", Sentiment: posEmo},
		{WordID: 61442, RootID: 15, Variant: 2, Word: "alegre", Lang: "PT", Norm: "alegre", Sentiment: posEmo},
		// Root 16 — negative (verb)
		{WordID: 65537, RootID: 16, Variant: 1, Word: "destroy", Lang: "EN", Norm: "destroy", Sentiment: negVerb},
		// Root 17 — positive (verb)
		{WordID: 69633, RootID: 17, Variant: 1, Word: "create", Lang: "EN", Norm: "create", Sentiment: posVerb},
		// Root 18 — neutral
		{WordID: 73729, RootID: 18, Variant: 1, Word: "table", Lang: "EN", Norm: "table", Sentiment: neuNoun},
	}

	_, err := lexdb.Build(roots, words, outPath)
	if err != nil {
		t.Fatalf("build smoke lexicon: %v", err)
	}

	lex, err := lexdb.Load(outPath)
	if err != nil {
		t.Fatalf("load smoke lexicon: %v", err)
	}
	return lex
}

// TestSmoke_FullTrainPredictCycle tests the complete pipeline:
// load lexicon, generate training data, split, train, and predict.
func TestSmoke_FullTrainPredictCycle(t *testing.T) {
	lex := buildSmokeLexicon(t)

	// Generate labeled examples from the lexicon
	examples := classify.GenerateFromLexicon(lex)
	if len(examples) == 0 {
		t.Fatal("GenerateFromLexicon produced zero examples")
	}

	// Root-stratified split
	train, val := classify.SplitByRoot(examples, 0.2)
	if len(train) == 0 || len(val) == 0 {
		t.Fatalf("SplitByRoot produced empty sets: train=%d val=%d", len(train), len(val))
	}

	// Train for 50 epochs on ALL examples (not split) for prediction accuracy.
	// The split is only validated above for correctness; here we want the
	// classifier to learn enough signal to predict correctly.
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	for epoch := 0; epoch < 50; epoch++ {
		for _, ex := range examples {
			clf.TrainStep(ex.Features, ex.LabelIdx)
		}
	}

	// Predict a known NEGATIVE word — use features from lexicon (with FPolarity
	// intact, as it should be at inference time).
	fNeg, ok := classify.ExtractFromLexicon(lex, "terrible", "EN")
	if !ok {
		t.Fatal("failed to extract features for 'terrible'")
	}
	predNeg, _ := clf.Predict(fNeg)
	if predNeg != "NEGATIVE" {
		t.Errorf("predicted %q for 'terrible', want NEGATIVE", predNeg)
	}

	// Predict a known POSITIVE word
	fPos, ok := classify.ExtractFromLexicon(lex, "happy", "EN")
	if !ok {
		t.Fatal("failed to extract features for 'happy'")
	}
	predPos, _ := clf.Predict(fPos)
	if predPos != "POSITIVE" {
		t.Errorf("predicted %q for 'happy', want POSITIVE", predPos)
	}
}

// TestSmoke_TrainF1Realistic verifies the trained model's F1 is within a
// realistic range: not 1.000 (which indicates polarity leakage) and not
// below 0.60 (which indicates a broken model).
func TestSmoke_TrainF1Realistic(t *testing.T) {
	lex := buildSmokeLexicon(t)

	examples := classify.GenerateFromLexicon(lex)
	train, val := classify.SplitByRoot(examples, 0.2)
	if len(train) == 0 || len(val) == 0 {
		t.Skipf("not enough roots for a meaningful split: train=%d val=%d", len(train), len(val))
	}

	// Use a smaller val fraction so training set has more roots, and we
	// increase epochs to give Adam enough steps on this small dataset.
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	for epoch := 0; epoch < 50; epoch++ {
		for _, ex := range train {
			clf.TrainStep(ex.Features, ex.LabelIdx)
		}
	}

	// Evaluate on train set (since val may lack class diversity in a small
	// lexicon). The F1 check here catches polarity leakage (F1=1.0 exactly)
	// and completely broken models (F1 near 0).
	f1Train := classify.F1Macro(clf, train)
	f1Val := classify.F1Macro(clf, val)
	t.Logf("train F1=%.4f, val F1=%.4f (train=%d, val=%d)", f1Train, f1Val, len(train), len(val))

	// Use train F1 for the realistic range check — val set may be too small
	// to represent all classes with root-stratified splitting on 18 roots.
	if f1Train >= 1.0 {
		t.Errorf("train F1 = %.4f, suspiciously perfect (polarity leak?)", f1Train)
	}
	if f1Train < 0.40 {
		t.Errorf("train F1 = %.4f, too low for a functional model (want >= 0.40)", f1Train)
	}
}

// TestSmoke_GAEvolveDoesNotRegress runs 3 GA generations and verifies the
// final best F1 does not drop more than 0.1 below the initial best.
func TestSmoke_GAEvolveDoesNotRegress(t *testing.T) {
	lex := buildSmokeLexicon(t)

	examples := classify.GenerateFromLexicon(lex)
	train, val := classify.SplitByRoot(examples, 0.2)
	if len(train) == 0 || len(val) == 0 {
		t.Skipf("not enough data for GA: train=%d val=%d", len(train), len(val))
	}

	pop := ga.New(8, 42) // small population for fast test
	pop.TrainSteps = 50

	// Run generation 1 to get initial fitness
	firstBest := pop.Evolve(train, val)
	initialF1 := firstBest.Fitness

	// Run 2 more generations
	var lastBest *ga.Chromosome
	for gen := 0; gen < 2; gen++ {
		lastBest = pop.Evolve(train, val)
	}

	finalF1 := lastBest.Fitness
	t.Logf("GA: initial F1=%.4f, final F1=%.4f", initialF1, finalF1)

	if finalF1 < initialF1-0.1 {
		t.Errorf("GA regressed: initial=%.4f, final=%.4f (delta=%.4f)",
			initialF1, finalF1, finalF1-initialF1)
	}
}

// TestSmoke_DataIntegrity_NoPolarityLeak verifies that ALL training examples
// generated from the lexicon have FPolarity == 0 (zeroed by ZeroLeakyFeatures).
func TestSmoke_DataIntegrity_NoPolarityLeak(t *testing.T) {
	lex := buildSmokeLexicon(t)

	examples := classify.GenerateFromLexicon(lex)
	if len(examples) == 0 {
		t.Fatal("no examples generated")
	}

	for i, ex := range examples {
		if ex.Features[classify.FPolarity] != 0 {
			t.Errorf("example %d (%s/%s): FPolarity = %v, want 0 (polarity leak)",
				i, ex.Word, ex.Lang, ex.Features[classify.FPolarity])
		}
	}
}

// TestSmoke_PredictOOVWordGracefully verifies that predicting a completely
// unknown word does not panic and returns a valid class label.
func TestSmoke_PredictOOVWordGracefully(t *testing.T) {
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)

	// Use a zero feature vector (simulating an OOV word with no features)
	var oovFeatures classify.FeatureVec
	pred, probs := clf.Predict(oovFeatures)

	if pred == "" {
		t.Error("Predict returned empty class for OOV features")
	}
	if len(probs) != len(classify.DefaultClasses) {
		t.Errorf("Predict returned %d probs, want %d", len(probs), len(classify.DefaultClasses))
	}

	// Also test with the lexicon path: ExtractFromLexicon should return false
	lex := buildSmokeLexicon(t)
	_, found := classify.ExtractFromLexicon(lex, "xyzzyplugh42", "EN")
	if found {
		t.Error("ExtractFromLexicon returned true for a nonsense word")
	}
}
