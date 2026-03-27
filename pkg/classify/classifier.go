package classify

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const (
	classifierMagic   = uint32(0x434C5346) // "CLSF"
	classifierVersion = uint32(1)
)

// Classifier is a softmax logistic regression model trained with the Adam
// optimizer. It maps FeatureVec → one of several sentiment classes.
//
// Adam (Kingma & Ba, 2014) maintains per-parameter 1st and 2nd moment
// estimates for adaptive learning rates, making it robust to sparse features
// (which Token64 features often are — e.g. role one-hot dims).
type Classifier struct {
	W []float64 // weights, row-major [NClasses × NFeatures]
	B []float64 // per-class biases [NClasses]

	Classes   []string
	NFeatures int

	// Adam optimizer state
	Step int
	M    []float64 // 1st moment (mean) for W
	V    []float64 // 2nd moment (variance) for W
	MB   []float64 // 1st moment for B
	VB   []float64 // 2nd moment for B

	// Hyperparameters (sane defaults; override after New() if desired)
	LR    float64 // learning rate (default 0.001)
	L2    float64 // L2 weight decay (default 1e-4)
	Beta1 float64 // Adam β₁ — 1st moment decay (default 0.9)
	Beta2 float64 // Adam β₂ — 2nd moment decay (default 0.999)
	Eps   float64 // Adam ε — numerical stability (default 1e-8)

	// FeatureWeights scales each input feature before the forward pass.
	// Set by the Genetic Algorithm to select/weight features.
	// Default: all 1.0 (no scaling).
	FeatureWeights FeatureVec
}

// New creates a Classifier with default Adam hyperparameters.
func New(nFeatures int, classes []string) *Classifier {
	nc := len(classes)
	c := &Classifier{
		W:         make([]float64, nc*nFeatures),
		B:         make([]float64, nc),
		Classes:   classes,
		NFeatures: nFeatures,
		M:         make([]float64, nc*nFeatures),
		V:         make([]float64, nc*nFeatures),
		MB:        make([]float64, nc),
		VB:        make([]float64, nc),
		LR:        0.001,
		L2:        1e-4,
		Beta1:     0.9,
		Beta2:     0.999,
		Eps:       1e-8,
	}
	for i := range c.FeatureWeights {
		c.FeatureWeights[i] = 1.0
	}
	return c
}

// ClassIndex returns the index of a class label, or -1 if not found.
func (c *Classifier) ClassIndex(label string) int {
	for i, cl := range c.Classes {
		if cl == label {
			return i
		}
	}
	return -1
}

// applyWeights scales features by FeatureWeights (GA-set multipliers).
func (c *Classifier) applyWeights(f FeatureVec) FeatureVec {
	for i := range f {
		f[i] *= c.FeatureWeights[i]
	}
	return f
}

// logits computes the raw (pre-softmax) scores for each class.
func (c *Classifier) logits(f FeatureVec) []float64 {
	fw := c.applyWeights(f)
	nc := len(c.Classes)
	out := make([]float64, nc)
	for cl := 0; cl < nc; cl++ {
		sum := c.B[cl]
		base := cl * c.NFeatures
		for fi := 0; fi < c.NFeatures; fi++ {
			sum += c.W[base+fi] * fw[fi]
		}
		out[cl] = sum
	}
	return out
}

// Softmax converts a logit vector to a probability distribution.
// Numerically stable: subtracts max before exp.
func Softmax(logits []float64) []float64 {
	maxV := logits[0]
	for _, v := range logits[1:] {
		if v > maxV {
			maxV = v
		}
	}
	probs := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		probs[i] = math.Exp(v - maxV)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

// Predict returns the predicted class and per-class probabilities.
func (c *Classifier) Predict(f FeatureVec) (class string, probs []float64) {
	probs = Softmax(c.logits(f))
	best, bestP := 0, probs[0]
	for i, p := range probs[1:] {
		if p > bestP {
			best, bestP = i+1, p
		}
	}
	return c.Classes[best], probs
}

// TrainStep performs one Adam gradient descent step on (f, labelIdx).
// Returns the cross-entropy loss before the update.
//
// Cross-entropy gradient w.r.t. logits:
//
//	dL/dz_c = p_c - 1{c==label}   (softmax + CE combined gradient)
func (c *Classifier) TrainStep(f FeatureVec, labelIdx int) float64 {
	c.Step++
	probs := Softmax(c.logits(f))

	loss := -math.Log(probs[labelIdx] + 1e-12)

	dz := make([]float64, len(c.Classes))
	copy(dz, probs)
	dz[labelIdx] -= 1.0

	fw := c.applyWeights(f)

	// Bias-corrected moment denominators for this step
	b1t := 1.0 - math.Pow(c.Beta1, float64(c.Step))
	b2t := 1.0 - math.Pow(c.Beta2, float64(c.Step))

	for cl, gz := range dz {
		// ── Bias Adam update ──────────────────────────────────────────────────
		c.MB[cl] = c.Beta1*c.MB[cl] + (1-c.Beta1)*gz
		c.VB[cl] = c.Beta2*c.VB[cl] + (1-c.Beta2)*gz*gz
		mhat := c.MB[cl] / b1t
		vhat := c.VB[cl] / b2t
		c.B[cl] -= c.LR * mhat / (math.Sqrt(vhat) + c.Eps)

		// ── Weight Adam update ────────────────────────────────────────────────
		base := cl * c.NFeatures
		for fi := 0; fi < c.NFeatures; fi++ {
			idx := base + fi
			gw := gz*fw[fi] + c.L2*c.W[idx]
			c.M[idx] = c.Beta1*c.M[idx] + (1-c.Beta1)*gw
			c.V[idx] = c.Beta2*c.V[idx] + (1-c.Beta2)*gw*gw
			mhat = c.M[idx] / b1t
			vhat = c.V[idx] / b2t
			c.W[idx] -= c.LR * mhat / (math.Sqrt(vhat) + c.Eps)
		}
	}
	return loss
}

// Save writes the classifier weights and state to a binary file.
//
// Format: magic(4) version(4) NClasses(4) NFeatures(4)
//
//	[NClasses × (len(4) + class_bytes)]
//	W[NClasses×NFeatures × float64]
//	B[NClasses × float64]
//	FeatureWeights[NFeatures × float64]
//	Step(int64)
func (c *Classifier) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w32 := func(v uint32) error { return binary.Write(f, binary.LittleEndian, v) }
	wf64 := func(v []float64) error { return binary.Write(f, binary.LittleEndian, v) }

	if err := w32(classifierMagic); err != nil {
		return err
	}
	if err := w32(classifierVersion); err != nil {
		return err
	}
	if err := w32(uint32(len(c.Classes))); err != nil {
		return err
	}
	if err := w32(uint32(c.NFeatures)); err != nil {
		return err
	}
	for _, cl := range c.Classes {
		b := []byte(cl)
		if err := w32(uint32(len(b))); err != nil {
			return err
		}
		if _, err := f.Write(b); err != nil {
			return err
		}
	}
	if err := wf64(c.W); err != nil {
		return err
	}
	if err := wf64(c.B); err != nil {
		return err
	}
	if err := wf64(c.FeatureWeights[:]); err != nil {
		return err
	}
	return binary.Write(f, binary.LittleEndian, int64(c.Step))
}

// Load reads a Classifier from a binary file written by Save.
func Load(path string) (*Classifier, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 16 {
		return nil, fmt.Errorf("classify: file too short")
	}
	if binary.LittleEndian.Uint32(data[0:]) != classifierMagic {
		return nil, fmt.Errorf("classify: not a classifier file")
	}
	nc := int(binary.LittleEndian.Uint32(data[8:]))
	nf := int(binary.LittleEndian.Uint32(data[12:]))
	off := 16

	classes := make([]string, nc)
	for i := range classes {
		if off+4 > len(data) {
			return nil, fmt.Errorf("classify: truncated")
		}
		cl := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		if off+cl > len(data) {
			return nil, fmt.Errorf("classify: truncated class string")
		}
		classes[i] = string(data[off : off+cl])
		off += cl
	}

	c := New(nf, classes)

	readF64s := func(dst []float64) error {
		need := len(dst) * 8
		if off+need > len(data) {
			return fmt.Errorf("classify: truncated float64 block")
		}
		for i := range dst {
			dst[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[off:]))
			off += 8
		}
		return nil
	}

	if err := readF64s(c.W); err != nil {
		return nil, err
	}
	if err := readF64s(c.B); err != nil {
		return nil, err
	}
	if err := readF64s(c.FeatureWeights[:]); err != nil {
		return nil, err
	}
	if off+8 <= len(data) {
		c.Step = int(int64(binary.LittleEndian.Uint64(data[off:])))
	}
	return c, nil
}
