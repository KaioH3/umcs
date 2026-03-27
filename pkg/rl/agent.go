// Package rl implements a REINFORCE policy-gradient feedback loop on top of
// the UMCS classify.Classifier.
//
// # Algorithm
//
// REINFORCE (Williams, 1992) is a Monte-Carlo policy gradient method.
// Given a trajectory of (state, action, reward) tuples, it updates the policy
// parameters θ to maximize expected return:
//
//	∇_θ J(θ) ≈ (1/N) Σ_i ∇_θ log π(a_i | s_i) · (R_i − b)
//
// where b is a baseline (running mean reward) that reduces variance without
// introducing bias. For logistic regression, the gradient is:
//
//	∇_θ log π(a|s) ≈ (1 − p(a|s)) · x
//
// We approximate this by running a TrainStep toward the correct label,
// scaled by the advantage (R − b).  Negative advantage (wrong prediction)
// uses a corrective step at a reduced LR.
//
// # Usage
//
//	agent := rl.New(clf)
//	agent.LoadState("models/classifier.bin")
//
//	// Each prediction cycle:
//	f, _ := classify.ExtractFromLexicon(lex, "terrible", "EN")
//	class, conf := agent.Act(f)
//	rl.RecordLast(f, class)
//
//	// User approves or corrects:
//	agent.Observe(rl.Feedback{Features: f, Predicted: class, Correct: class, Reward: +1})
//	agent.Learn()   // REINFORCE update
//	agent.SaveState("models/classifier.bin")
package rl

import (
	"encoding/json"
	"math"
	"os"
	"sync"
	"time"

	"github.com/kak/umcs/pkg/classify"
)

// Feedback records one classification event and its human-judged outcome.
type Feedback struct {
	Features  classify.FeatureVec `json:"features"`
	Predicted string              `json:"predicted"` // what the model said
	Correct   string              `json:"correct"`   // what it should have said
	Reward    float64             `json:"reward"`    // +1=correct, -1=wrong
	Timestamp int64               `json:"ts"`        // Unix nanoseconds
}

// Agent wraps a Classifier with a REINFORCE feedback loop.
// Call Act to make a prediction, Observe to record the reward,
// and Learn to apply the accumulated REINFORCE update.
type Agent struct {
	Clf      *classify.Classifier
	History  []Feedback
	Baseline float64 // exponential moving average of rewards (variance reduction)
	Gamma    float64 // discount factor for future rewards (default 0.95)
}

// New creates an Agent wrapping clf with default hyperparameters.
func New(clf *classify.Classifier) *Agent {
	return &Agent{
		Clf:   clf,
		Gamma: 0.95,
	}
}

// Act returns the predicted class and confidence (softmax probability) for f.
// Call RecordLast(f, class) immediately after if using the single-shot
// predict → feedback CLI workflow.
func (a *Agent) Act(f classify.FeatureVec) (class string, conf float64) {
	class, probs := a.Clf.Predict(f)
	idx := a.Clf.ClassIndex(class)
	if idx >= 0 && idx < len(probs) {
		conf = probs[idx]
	}
	return class, conf
}

// Observe records one feedback event. The baseline is updated immediately.
func (a *Agent) Observe(fb Feedback) {
	if fb.Timestamp == 0 {
		fb.Timestamp = time.Now().UnixNano()
	}
	a.History = append(a.History, fb)
	// Exponential moving average baseline (α=0.1)
	if a.Baseline == 0 {
		a.Baseline = fb.Reward
	} else {
		a.Baseline = 0.9*a.Baseline + 0.1*fb.Reward
	}
}

// Learn applies REINFORCE updates for all accumulated feedback, then clears
// the history. Call after a batch of Observe calls.
//
// For positive advantage (R > baseline): take a full TrainStep toward the
// correct label — reinforce the good decision.
// For negative advantage (R < baseline): take a scaled TrainStep toward the
// correct label with LR proportional to |advantage| — corrective update.
func (a *Agent) Learn() {
	if len(a.History) == 0 {
		return
	}
	origLR := a.Clf.LR

	for _, fb := range a.History {
		advantage := fb.Reward - a.Baseline
		if math.Abs(advantage) < 1e-6 {
			continue // no update for near-zero advantage
		}
		correctIdx := a.Clf.ClassIndex(fb.Correct)
		if correctIdx < 0 {
			continue
		}

		if advantage > 0 {
			// Positive: reinforce correct class at full LR
			a.Clf.LR = origLR
		} else {
			// Negative: corrective update at scaled LR
			a.Clf.LR = origLR * math.Min(math.Abs(advantage), 1.0)
		}
		a.Clf.TrainStep(fb.Features, correctIdx)
	}

	a.Clf.LR = origLR
	a.History = a.History[:0]
}

// ── Single-shot workflow state ─────────────────────────────────────────────────

// lastMu protects LastPrediction for concurrent access (e.g. HTTP server usage).
var lastMu sync.RWMutex

// LastPrediction stores the most recent prediction for the predict → feedback
// CLI workflow, where predict and feedback are separate invocations.
// Always access through RecordLast / GetLast — never read directly in concurrent code.
var LastPrediction *LastState

// LastState is the minimal state needed to apply feedback to the last prediction.
type LastState struct {
	Features  classify.FeatureVec `json:"f"`
	Predicted string              `json:"p"`
}

// RecordLast stores the last prediction so a subsequent `lexsent feedback`
// call can retrieve it. Safe for concurrent use.
func RecordLast(f classify.FeatureVec, predicted string) {
	lastMu.Lock()
	LastPrediction = &LastState{Features: f, Predicted: predicted}
	lastMu.Unlock()
}

// GetLast returns the last recorded prediction. Safe for concurrent use.
func GetLast() *LastState {
	lastMu.RLock()
	defer lastMu.RUnlock()
	return LastPrediction
}

// ── Persistence ───────────────────────────────────────────────────────────────

// agentState is the JSON-serializable sidecar for the RL agent.
type agentState struct {
	Baseline float64    `json:"baseline"`
	Gamma    float64    `json:"gamma"`
	History  []Feedback `json:"history,omitempty"`
	Last     *LastState `json:"last,omitempty"`
}

// SaveState writes the agent state to a JSON sidecar alongside the model file.
// The sidecar path is modelPath + ".rl".
func (a *Agent) SaveState(modelPath string) error {
	st := agentState{
		Baseline: a.Baseline,
		Gamma:    a.Gamma,
		History:  a.History,
		Last:     GetLast(),
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(modelPath+".rl", data, 0o644)
}

// LoadState restores agent state from the JSON sidecar.
// If the file does not exist, the agent starts fresh (no error).
func (a *Agent) LoadState(modelPath string) error {
	data, err := os.ReadFile(modelPath + ".rl")
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	var st agentState
	if err := json.Unmarshal(data, &st); err != nil {
		return err
	}
	a.Baseline = st.Baseline
	a.Gamma = st.Gamma
	a.History = st.History
	lastMu.Lock()
	LastPrediction = st.Last
	lastMu.Unlock()
	return nil
}
