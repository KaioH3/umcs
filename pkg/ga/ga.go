// Package ga implements a genetic algorithm that evolves the feature weights
// for the UMCS classify.Classifier.
//
// Each Chromosome encodes a 48-element weight vector (one per feature in
// FeatureVec). The fitness function trains a fresh Classifier for a fixed
// number of Adam steps and evaluates macro-averaged F1 on a held-out
// validation set. The GA then evolves weights toward higher F1 over
// successive generations.
//
// Algorithm summary:
//
//  1. Initialize population of random chromosomes
//  2. Evaluate fitness (F1) for every chromosome
//  3. Preserve top-E elites unchanged (elitism)
//  4. Fill remaining slots via tournament selection + uniform crossover + Gaussian mutation
//  5. Repeat until convergence or max generations
//
// Tournament selection (k=3) is O(k) per individual and robust to fitness
// scaling issues, unlike roulette wheel selection.
package ga

import (
	"math/rand"

	"github.com/kak/umcs/pkg/classify"
)

// Chromosome encodes feature weights for one population individual.
// Weights[i] multiplies feature i before the classifier forward pass.
// A weight of 0 effectively disables that feature.
type Chromosome struct {
	Weights classify.FeatureVec
	Fitness float64 // macro F1 on validation set (higher = better)
}

// Population manages the GA population and evolution parameters.
type Population struct {
	Individuals []*Chromosome
	Size        int     // number of individuals (default 64)
	Elite       int     // top N elites copied unchanged each generation (default 8)
	MutRate     float64 // per-gene mutation probability (default 0.03)
	CrossRate   float64 // crossover probability (default 0.70)
	TrainSteps  int     // Adam steps per fitness evaluation (default 200)
	rng         *rand.Rand
}

// New creates a randomly initialized population.
// seed controls reproducibility; use 0 for a random seed each run.
func New(size int, seed int64) *Population {
	rng := rand.New(rand.NewSource(seed))
	p := &Population{
		Individuals: make([]*Chromosome, size),
		Size:        size,
		Elite:       intMax(1, size/8),
		MutRate:     0.03,
		CrossRate:   0.70,
		TrainSteps:  200,
		rng:         rng,
	}
	for i := range p.Individuals {
		p.Individuals[i] = randomChromosome(rng)
	}
	return p
}

// randomChromosome generates a random chromosome.
// Most weights are drawn from Uniform(0.7, 1.5); ~15% are zeroed (disabled).
func randomChromosome(rng *rand.Rand) *Chromosome {
	c := &Chromosome{}
	for i := range c.Weights {
		if rng.Float64() < 0.15 {
			c.Weights[i] = 0.0 // feature disabled
		} else {
			c.Weights[i] = 0.7 + rng.Float64()*0.8
		}
	}
	return c
}

// evaluate assigns Fitness to every chromosome.
func (p *Population) evaluate(train, val []classify.Example) {
	for _, chr := range p.Individuals {
		chr.Fitness = p.fitnessOf(chr, train, val)
	}
}

// fitnessOf trains a fresh classifier using chr's weights and returns F1.
func (p *Population) fitnessOf(chr *Chromosome, train, val []classify.Example) float64 {
	if len(train) == 0 || len(val) == 0 {
		return 0
	}
	clf := classify.New(classify.NFeatures, classify.DefaultClasses)
	clf.FeatureWeights = chr.Weights
	for step := 0; step < p.TrainSteps; step++ {
		ex := &train[step%len(train)]
		clf.TrainStep(ex.Features, ex.LabelIdx)
	}
	return classify.F1Macro(clf, val)
}

// insertionSort sorts p.Individuals descending by Fitness.
// Population size is small (≤256) so O(n²) is fine.
func (p *Population) insertionSort() {
	ind := p.Individuals
	for i := 1; i < len(ind); i++ {
		key := ind[i]
		j := i - 1
		for j >= 0 && ind[j].Fitness < key.Fitness {
			ind[j+1] = ind[j]
			j--
		}
		ind[j+1] = key
	}
}

// tournamentSelect picks the best of k random individuals (k=3).
func (p *Population) tournamentSelect() *Chromosome {
	best := p.Individuals[p.rng.Intn(len(p.Individuals))]
	for i := 1; i < 3; i++ {
		c := p.Individuals[p.rng.Intn(len(p.Individuals))]
		if c.Fitness > best.Fitness {
			best = c
		}
	}
	return best
}

// crossover produces a child via uniform crossover of two parents.
func crossover(a, b *Chromosome, rng *rand.Rand) *Chromosome {
	child := &Chromosome{}
	for i := range child.Weights {
		if rng.Float64() < 0.5 {
			child.Weights[i] = a.Weights[i]
		} else {
			child.Weights[i] = b.Weights[i]
		}
	}
	return child
}

// mutate applies Gaussian perturbation to each gene with probability MutRate.
// Weights are clamped to [0, 3] to keep the scale interpretable.
func (p *Population) mutate(c *Chromosome) {
	for i := range c.Weights {
		if p.rng.Float64() < p.MutRate {
			c.Weights[i] += p.rng.NormFloat64() * 0.1
			if c.Weights[i] < 0 {
				c.Weights[i] = 0
			} else if c.Weights[i] > 3 {
				c.Weights[i] = 3
			}
		}
	}
}

// Evolve runs one complete generation (evaluate → sort → select → reproduce).
// Returns the best individual in the new population.
func (p *Population) Evolve(train, val []classify.Example) *Chromosome {
	p.evaluate(train, val)
	p.insertionSort()

	next := make([]*Chromosome, p.Size)

	// Elitism: copy top-E individuals unchanged
	for i := 0; i < p.Elite && i < len(p.Individuals); i++ {
		cp := *p.Individuals[i]
		next[i] = &cp
	}

	// Fill the rest via selection + optional crossover + mutation
	for i := p.Elite; i < p.Size; i++ {
		parent1 := p.tournamentSelect()
		var child *Chromosome
		if p.rng.Float64() < p.CrossRate {
			parent2 := p.tournamentSelect()
			child = crossover(parent1, parent2, p.rng)
		} else {
			cp := *parent1
			child = &cp
		}
		p.mutate(child)
		next[i] = child
	}

	p.Individuals = next
	// Return the elite best (it was the best before reproduction too)
	return p.Individuals[0]
}

// Run executes up to `generations` generations and returns the best
// Chromosome ever seen. logFn (may be nil) is called after each generation
// with the generation number and current best F1.
func (p *Population) Run(train, val []classify.Example, generations int, logFn func(gen int, bestF1 float64)) *Chromosome {
	var best *Chromosome
	for gen := 1; gen <= generations; gen++ {
		b := p.Evolve(train, val)
		if best == nil || b.Fitness > best.Fitness {
			cp := *b
			best = &cp
		}
		if logFn != nil {
			logFn(gen, best.Fitness)
		}
	}
	return best
}

func intMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}
