package ga

import (
	"testing"

	"github.com/kak/umcs/pkg/classify"
)

// makeSeparableData builds a tiny perfectly-separable dataset:
// POSITIVE examples have FPolarity=+1, NEGATIVE have FPolarity=-1.
func makeSeparableData(n int) (train, val []classify.Example) {
	posIdx := 2 // "POSITIVE" in DefaultClasses
	negIdx := 0 // "NEGATIVE"
	for i := 0; i < n; i++ {
		var pos, neg classify.FeatureVec
		pos[classify.FPolarity] = 1.0
		pos[classify.FIntensity] = 0.75
		neg[classify.FPolarity] = -1.0
		neg[classify.FIntensity] = 0.75
		if i < n*4/5 {
			train = append(train,
				classify.Example{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
				classify.Example{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
			)
		} else {
			val = append(val,
				classify.Example{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
				classify.Example{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
			)
		}
	}
	return train, val
}

func TestRandomChromosome_WeightsInRange(t *testing.T) {
	p := New(32, 42)
	for _, chr := range p.Individuals {
		for j, w := range chr.Weights {
			if w < 0 || w > 3 {
				t.Errorf("weight[%d] = %v out of [0,3]", j, w)
			}
		}
	}
}

func TestEvolve_DoesNotPanic(t *testing.T) {
	train, val := makeSeparableData(20)
	p := New(8, 42)
	p.TrainSteps = 20
	_ = p.Evolve(train, val)
}

func TestRun_ReturnsBestChromosome(t *testing.T) {
	train, val := makeSeparableData(30)
	p := New(16, 42)
	p.TrainSteps = 50

	best := p.Run(train, val, 5, nil)
	if best == nil {
		t.Fatal("Run returned nil")
	}
	if best.Fitness < 0 || best.Fitness > 1.01 {
		t.Errorf("best fitness %v out of [0,1]", best.Fitness)
	}
}

func TestElitism_FitnessDoesNotRegress(t *testing.T) {
	train, val := makeSeparableData(20)
	p := New(8, 42)
	p.TrainSteps = 30
	p.Elite = 2

	b1 := p.Evolve(train, val).Fitness
	b2 := p.Evolve(train, val).Fitness
	// Elite individuals survive; fitness should not drop by more than a small margin
	if b2 < b1-0.2 {
		t.Errorf("fitness regressed: gen1=%v gen2=%v", b1, b2)
	}
}

func TestCrossover_WeightsComeFromParents(t *testing.T) {
	p := New(2, 0)
	a := p.Individuals[0]
	b := p.Individuals[1]
	child := crossover(a, b, p.rng)
	for i, w := range child.Weights {
		if w != a.Weights[i] && w != b.Weights[i] {
			t.Errorf("child.Weights[%d]=%v not from a(%v) or b(%v)", i, w, a.Weights[i], b.Weights[i])
		}
	}
}

func TestMutate_ClampsWeights(t *testing.T) {
	p := New(4, 0)
	p.MutRate = 1.0 // mutate every gene every call
	c := &Chromosome{}
	for i := range c.Weights {
		c.Weights[i] = 2.95
	}
	for i := 0; i < 50; i++ {
		p.mutate(c)
	}
	for i, w := range c.Weights {
		if w < 0 || w > 3 {
			t.Errorf("weight[%d] = %v out of [0,3] after mutation", i, w)
		}
	}
}

func TestGA_FitnessInValidRange(t *testing.T) {
	train, val := makeSeparableData(20)
	p := New(8, 7)
	p.TrainSteps = 30
	best := p.Evolve(train, val)
	if best.Fitness < 0 || best.Fitness > 1.001 {
		t.Errorf("fitness %v not in [0,1]", best.Fitness)
	}
}

func TestGA_EliteIndividualsSurvive(t *testing.T) {
	train, val := makeSeparableData(30)
	p := New(16, 42)
	p.TrainSteps = 40
	p.Elite = 2

	// Evaluate initial fitness so elites are defined
	best := p.Evolve(train, val)
	fitnessAfterGen1 := best.Fitness

	// Run another generation — elite should keep us at least close
	best2 := p.Evolve(train, val)
	if best2.Fitness < fitnessAfterGen1-0.15 {
		t.Errorf("elite survival: fitness dropped from %.3f to %.3f (too much)", fitnessAfterGen1, best2.Fitness)
	}
}

func TestGA_ConvergesOnLeakFreeData(t *testing.T) {
	// Construct data that is separable without FPolarity (simulating leak-free
	// training). FIntensity=1.0 for POSITIVE, FIntensity=0.0 for NEGATIVE,
	// FPolarity=0 for both. The GA must discover FIntensity as the key signal.
	posIdx := 2
	negIdx := 0
	var train, val []classify.Example
	for i := 0; i < 50; i++ {
		var pos, neg classify.FeatureVec
		pos[classify.FIntensity] = 1.0  // separating feature
		neg[classify.FIntensity] = 0.0  // FPolarity stays 0 for both
		pos[classify.FSyllables] = 0.3  // secondary signal
		neg[classify.FSyllables] = 0.7
		if i < 40 {
			train = append(train,
				classify.Example{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
				classify.Example{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
			)
		} else {
			val = append(val,
				classify.Example{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
				classify.Example{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
			)
		}
	}
	p := New(16, 42)
	p.TrainSteps = 100
	best := p.Run(train, val, 10, nil)
	if best == nil {
		t.Fatal("Run returned nil")
	}
	if best.Fitness < 0.50 {
		t.Errorf("leak-free GA F1 = %.3f, want ≥ 0.50 (non-trivial learning)", best.Fitness)
	}
}

func TestGA_WithSeed_Deterministic(t *testing.T) {
	train, val := makeSeparableData(20)
	p1 := New(8, 99)
	p1.TrainSteps = 20
	b1 := p1.Evolve(train, val)

	p2 := New(8, 99)
	p2.TrainSteps = 20
	b2 := p2.Evolve(train, val)

	if b1.Fitness != b2.Fitness {
		t.Errorf("same seed gave different fitness: %.4f vs %.4f", b1.Fitness, b2.Fitness)
	}
	for i, w := range b1.Weights {
		if w != b2.Weights[i] {
			t.Errorf("same seed gave different weight[%d]: %v vs %v", i, w, b2.Weights[i])
		}
	}
}

func TestGA_MutationRateZero_WeightsUnchanged(t *testing.T) {
	p := New(4, 42)
	p.MutRate = 0.0
	// Record original weights
	orig := make([]classify.FeatureVec, len(p.Individuals))
	for i, ind := range p.Individuals {
		orig[i] = ind.Weights
	}
	// Mutate 100 times — zero rate means no change
	for _, ind := range p.Individuals {
		for i := 0; i < 100; i++ {
			p.mutate(ind)
		}
	}
	for i, ind := range p.Individuals {
		for j, w := range ind.Weights {
			if w != orig[i][j] {
				t.Errorf("individual %d weight[%d] changed despite MutRate=0: %v → %v", i, j, orig[i][j], w)
				return
			}
		}
	}
}

func TestGA_EmptyData_DoesNotPanic(t *testing.T) {
	p := New(4, 0)
	p.TrainSteps = 5
	_ = p.Evolve(nil, nil)
}

func TestGA_SingleExampleEachClass(t *testing.T) {
	posIdx := 2
	negIdx := 0
	var pos, neg classify.FeatureVec
	pos[classify.FPolarity] = 1.0
	neg[classify.FPolarity] = -1.0
	train := []classify.Example{
		{Features: pos, Label: "POSITIVE", LabelIdx: posIdx},
		{Features: neg, Label: "NEGATIVE", LabelIdx: negIdx},
	}
	val := train // tiny test — same data, just checking no panic
	p := New(4, 0)
	p.TrainSteps = 10
	best := p.Evolve(train, val)
	if best == nil {
		t.Fatal("Evolve with 1 example per class returned nil")
	}
}
