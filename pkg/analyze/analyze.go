// Package analyze implements the sentiment analysis pipeline:
// tokenize → lookup → scope resolution → aggregate.
//
// Scope resolution handles:
//   - Negation markers: invert polarity of next N tokens
//   - Intensifiers: amplify weight of next token by 2×
//   - Downtoners: halve weight of next token
package analyze

import (
	"strings"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/sentiment"
)

const negationWindow = 3 // how many tokens are affected by a negation marker

// Token is the result of analyzing a single word.
type Token struct {
	Surface   string
	Found     bool
	WordID    uint32
	RootID    uint32
	RootStr   string
	Lang      string
	Polarity  string
	Intensity string
	Role      string
	Weight    int // signed, after scope application
	Negated   bool
	Amplified bool
}

// Result is the aggregate result of analyzing a full text.
type Result struct {
	Tokens     []Token
	TotalScore int
	Matched    int
	Total      int
	Verdict    string // POSITIVE, NEGATIVE, NEUTRAL
}

// Analyze tokenizes text and scores each token against the lexicon.
func Analyze(lex *lexdb.Lexicon, text string) Result {
	words := tokenize(text)
	tokens := make([]Token, 0, len(words))

	negationScope := 0
	intensifyNext := false
	downtonNext := false

	for _, surface := range words {
		t := Token{Surface: surface}
		w := lex.LookupWord(surface)

		if w == nil {
			// Still decrement scopes for unfound tokens
			if negationScope > 0 {
				negationScope--
			}
			intensifyNext = false
			downtonNext = false
			tokens = append(tokens, t)
			continue
		}

		t.Found = true
		t.WordID = w.WordID
		t.RootID = w.RootID
		t.Lang = lexdb.LangName(w.Lang)

		root := lex.LookupRoot(w.RootID)
		if root != nil {
			t.RootStr = lex.RootStr(root)
		}

		sent := sentiment.Decode(w.Sentiment)
		t.Polarity = sent["polarity"]
		t.Intensity = sent["intensity"]
		t.Role = sent["role"]

		// Scope: negation marker — double negation cancels the first
		if sentiment.IsNegationMarker(w.Sentiment) {
			if negationScope > 0 {
				negationScope = 0 // cancel existing negation (double negation = affirmation)
			} else {
				negationScope = negationWindow
			}
			intensifyNext = false
			downtonNext = false
			tokens = append(tokens, t)
			continue
		}

		// Compute base weight
		weight := sentiment.Weight(w.Sentiment)

		// Apply negation scope
		if negationScope > 0 {
			weight = -weight
			t.Negated = true
			negationScope--
		}

		// Apply intensifier
		if intensifyNext {
			weight *= 2
			t.Amplified = true
			intensifyNext = false
		}
		if downtonNext {
			if weight > 0 {
				weight = max(1, weight/2)
			} else if weight < 0 {
				weight = min(-1, weight/2)
			}
			downtonNext = false
		}

		// Check if THIS token is a scope modifier for NEXT token
		if sentiment.IsIntensifier(w.Sentiment) {
			intensifyNext = true
		}
		if sentiment.IsDowntoner(w.Sentiment) {
			downtonNext = true
		}

		t.Weight = weight
		tokens = append(tokens, t)
	}

	// Aggregate
	totalScore := 0
	matched := 0
	for _, t := range tokens {
		if t.Found {
			matched++
			totalScore += t.Weight
		}
	}

	verdict := "NEUTRAL"
	switch {
	case totalScore >= 2:
		verdict = "POSITIVE"
	case totalScore <= -2:
		verdict = "NEGATIVE"
	}

	return Result{
		Tokens:     tokens,
		TotalScore: totalScore,
		Matched:    matched,
		Total:      len(tokens),
		Verdict:    verdict,
	}
}

// tokenize splits and normalizes a text into individual word tokens.
func tokenize(text string) []string {
	raw := strings.Fields(strings.ToLower(text))
	out := make([]string, 0, len(raw))
	for _, w := range raw {
		// Strip punctuation
		w = strings.Trim(w, ".,!?;:\"'()[]{}…")
		if w != "" {
			out = append(out, w)
		}
	}
	return out
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ── Emotion Decomposition ───────────────────────────────────────────────────

// EmotionProfile holds fine-grained emotion scores derived from
// the morpheme-level sentiment dimensions. All values in [0.0, 1.0].
type EmotionProfile struct {
	Joy      float64 // high valence + high arousal + high dominance
	Trust    float64 // positive polarity + low arousal + high dominance
	Fear     float64 // negative polarity + high arousal + low dominance
	Anger    float64 // negative polarity + high arousal + high dominance
	Sadness  float64 // negative polarity + low arousal + low dominance
	Surprise float64 // high arousal + neutral polarity
	Disgust  float64 // strongly negative + high arousal
	Serenity float64 // positive + low arousal
	Dominant string  // the strongest emotion label
}

// EmotionDecompose extracts an emotion profile from the analysis result.
// Uses Plutchik's wheel mapped to UMCS dimensions:
//
// EmotionDecompose maps sentiment to Plutchik's 8 primary emotions.
// Uses VAD (Valence, Arousal, Dominance) from lexicon, with fallback heuristics
// when data is unavailable.
func EmotionDecompose(r Result, lex *lexdb.Lexicon) EmotionProfile {
	var ep EmotionProfile
	var total float64

	for _, t := range r.Tokens {
		if !t.Found {
			continue
		}
		w := lex.LookupWord(t.Surface)
		if w == nil {
			continue
		}

		s := sentiment.Decode(w.Sentiment)
		polStr := s["polarity"]
		pol := polVal(polStr)

		// Use VAD from lexicon if available, otherwise infer from intensity/polarity
		aroStr := s["arousal"]
		aro := aroVal(aroStr)
		if aroStr == "NONE" || aroStr == "" {
			aro = inferArousal(s["intensity"])
		}

		domStr := s["dominance"]
		dom := domVal(domStr)
		if domStr == "NONE" || domStr == "" {
			dom = inferDominance(polStr)
		}

		intenStr := s["intensity"]
		inten := intenVal(intenStr)

		weight := inten
		if weight == 0 {
			weight = 0.5
		}

		// Plutchik mapping: polarity × arousal × dominance → emotion
		// Joy: positive + high arousal + high dominance
		if pol > 0 && aro > 0.5 && dom > 0.5 {
			ep.Joy += weight
		}
		// Trust: positive + low arousal + high dominance
		if pol > 0 && aro <= 0.5 && dom > 0.5 {
			ep.Trust += weight
		}
		// Serenity: positive + low arousal
		if pol > 0 && aro <= 0.5 {
			ep.Serenity += weight
		}
		// Fear: negative + high arousal + low dominance
		if pol < 0 && aro > 0.5 && dom <= 0.35 {
			ep.Fear += weight
		}
		// Anger: negative + high arousal + high dominance
		if pol < 0 && aro > 0.5 && dom > 0.35 {
			ep.Anger += weight
		}
		// Sadness: negative + low arousal (not high)
		if pol < 0 && aro < 0.5 {
			ep.Sadness += weight
		}
		// Surprise: high arousal + neutral polarity
		if aro > 0.7 && pol == 0 {
			ep.Surprise += weight
		}
		// Disgust: negative + high arousal + high intensity
		if pol < 0 && aro > 0.5 && inten > 0.5 {
			ep.Disgust += weight
		}

		total += weight
	}

	// Normalize to [0, 1]
	if total > 0 {
		ep.Joy /= total
		ep.Trust /= total
		ep.Fear /= total
		ep.Anger /= total
		ep.Sadness /= total
		ep.Surprise /= total
		ep.Disgust /= total
		ep.Serenity /= total
	}

	// Find dominant
	best := 0.0
	emotions := map[string]float64{
		"JOY": ep.Joy, "TRUST": ep.Trust, "FEAR": ep.Fear,
		"ANGER": ep.Anger, "SADNESS": ep.Sadness, "SURPRISE": ep.Surprise,
		"DISGUST": ep.Disgust, "SERENITY": ep.Serenity,
	}
	ep.Dominant = "NEUTRAL"
	for name, val := range emotions {
		if val > best {
			best = val
			ep.Dominant = name
		}
	}

	return ep
}

// polVal converts polarity string to numeric value.
func polVal(s string) float64 {
	switch s {
	case "POSITIVE":
		return 1.0
	case "NEGATIVE":
		return -1.0
	default:
		return 0.0
	}
}

// aroVal converts arousal string to numeric value.
func aroVal(s string) float64 {
	switch s {
	case "HIGH":
		return 1.0
	case "MED":
		return 0.5
	case "LOW":
		return 0.25
	default:
		return 0.0
	}
}

// inferArousal estimates arousal from intensity when VAD data is unavailable.
// High intensity words tend to have high arousal.
func inferArousal(intensity string) float64 {
	switch intensity {
	case "STRONG", "EXTREME":
		return 0.8 // HIGH
	case "MODERATE":
		return 0.5 // MED
	case "WEAK":
		return 0.25 // LOW
	default:
		return 0.3 // Default to low-medium for unknown
	}
}

// domVal converts dominance string to numeric value.
func domVal(s string) float64 {
	switch s {
	case "HIGH":
		return 1.0
	case "MED":
		return 0.5
	case "LOW":
		return 0.25
	default:
		return 0.5 // default to mid
	}
}

// inferDominance estimates dominance from polarity when VAD data is unavailable.
// Negative emotions often have lower dominance (feeling overwhelmed).
func inferDominance(polarity string) float64 {
	switch polarity {
	case "NEGATIVE":
		return 0.25 // LOW - negative emotions often feel overwhelming
	case "POSITIVE":
		return 0.7 // HIGH - positive emotions feel empowering
	default:
		return 0.5 // MED - neutral
	}
}

// intenVal converts intensity string to numeric value.
func intenVal(s string) float64 {
	switch s {
	case "EXTREME":
		return 1.0
	case "STRONG":
		return 0.75
	case "MODERATE":
		return 0.5
	case "WEAK":
		return 0.25
	default:
		return 0.0
	}
}

// ── Sentiment Drift Detection ────────────────────────────────────────────────

// DriftPoint records a sentiment measurement at a position in text.
type DriftPoint struct {
	Position int     // token index
	Score    float64 // cumulative sentiment at this point
	Delta    float64 // change from previous point
}

// DetectDrift analyzes sentiment trajectory across a text,
// identifying shifts in emotional tone. Useful for:
//   - Product review analysis (starts positive, turns negative)
//   - Customer support conversations (detecting escalation)
//   - Literary analysis (emotional arcs in narrative)
func DetectDrift(r Result) []DriftPoint {
	if len(r.Tokens) == 0 {
		return nil
	}

	points := make([]DriftPoint, 0, len(r.Tokens))
	cumulative := 0.0
	prev := 0.0

	for i, t := range r.Tokens {
		if t.Found {
			cumulative += float64(t.Weight)
		}
		delta := cumulative - prev
		points = append(points, DriftPoint{
			Position: i,
			Score:    cumulative,
			Delta:    delta,
		})
		prev = cumulative
	}

	return points
}

// DriftSummary provides high-level drift analysis.
type DriftSummary struct {
	Pattern     string  // "STABLE", "ASCENDING", "DESCENDING", "V-SHAPE", "INV-V", "VOLATILE"
	MaxPositive float64 // highest cumulative score
	MaxNegative float64 // lowest cumulative score
	Volatility  float64 // average absolute delta
	Shifts      int     // number of polarity sign changes
}

// SummarizeDrift analyzes the drift trajectory.
func SummarizeDrift(points []DriftPoint) DriftSummary {
	if len(points) == 0 {
		return DriftSummary{Pattern: "STABLE"}
	}

	var s DriftSummary
	var sumAbsDelta float64
	lastSign := 0

	for _, p := range points {
		if p.Score > s.MaxPositive {
			s.MaxPositive = p.Score
		}
		if p.Score < s.MaxNegative {
			s.MaxNegative = p.Score
		}
		sumAbsDelta += abs(p.Delta)

		sign := 0
		if p.Score > 0 {
			sign = 1
		} else if p.Score < 0 {
			sign = -1
		}
		if lastSign != 0 && sign != 0 && sign != lastSign {
			s.Shifts++
		}
		if sign != 0 {
			lastSign = sign
		}
	}

	s.Volatility = sumAbsDelta / float64(len(points))

	// Classify pattern - improved detection
	first := points[0].Score
	last := points[len(points)-1].Score

	// Find the peak (max) and valley (min) for better pattern detection
	var peakScore, valleyScore float64
	peakIdx, valleyIdx := 0, 0
	for i, p := range points {
		if p.Score > peakScore {
			peakScore = p.Score
			peakIdx = i
		}
		if p.Score < valleyScore {
			valleyScore = p.Score
			valleyIdx = i
		}
	}

	// Analyze trajectory
	hasPositive := peakScore > 0.5
	hasNegative := valleyScore < -0.5
	hasSignificantShift := s.Shifts >= 1 || s.Volatility > 0.5

	switch {
	case s.Volatility < 0.3 && s.Shifts == 0:
		s.Pattern = "STABLE"
	case s.Shifts >= 3 || s.Volatility > 2.0:
		s.Pattern = "VOLATILE"
	case hasPositive && hasNegative && peakIdx < valleyIdx:
		// Positive → Negative (DESCENDING or V-SHAPE)
		if valleyIdx > len(points)/2 && first >= 0 {
			s.Pattern = "V-SHAPE"
		} else {
			s.Pattern = "DESCENDING"
		}
	case hasPositive && hasNegative && peakIdx > valleyIdx:
		// Negative → Positive (ASCENDING or INV-V)
		if peakIdx > len(points)/2 && last >= 0 {
			s.Pattern = "INV-V"
		} else {
			s.Pattern = "ASCENDING"
		}
	case last > first+1 && !hasNegative:
		s.Pattern = "ASCENDING"
	case last < first-1 && !hasPositive:
		s.Pattern = "DESCENDING"
	default:
		if hasSignificantShift {
			s.Pattern = "VOLATILE"
		} else {
			s.Pattern = "STABLE"
		}
	}

	return s
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// ── Cross-lingual Sentiment Transfer ─────────────────────────────────────────

// CrossLingualScore returns the sentiment consensus for a word across all
// languages in the lexicon. This is the UMCS "universal coordinate":
// a word's meaning transcends its surface form.
//
// Algorithm:
//  1. Look up the word in the lexicon
//  2. Find its root
//  3. Enumerate all cognates (words sharing the same root)
//  4. Aggregate sentiment across all cognates
//
// Returns: consensus polarity, confidence [0,1], and number of languages.
func CrossLingualScore(lex *lexdb.Lexicon, word string) (polarity string, confidence float64, nLangs int) {
	w := lex.LookupWord(word)
	if w == nil {
		return "UNKNOWN", 0, 0
	}

	root := lex.LookupRoot(w.RootID)
	if root == nil {
		s := sentiment.Decode(w.Sentiment)
		return s["polarity"], 0.5, 1
	}

	// Enumerate cognates (all words sharing the same root)
	cognates := lex.Cognates(w.WordID)
	posVotes := 0
	negVotes := 0
	neuVotes := 0
	langSet := make(map[uint32]bool)
	totalWeight := 0

	// Include the word itself
	allWords := append([]lexdb.WordRecord{*w}, cognates...)
	for _, ww := range allWords {
		langSet[ww.Lang] = true
		s := sentiment.Decode(ww.Sentiment)

		intensity := 1
		switch s["intensity"] {
		case "EXTREME":
			intensity = 4
		case "STRONG":
			intensity = 3
		case "MODERATE":
			intensity = 2
		case "WEAK":
			intensity = 1
		}

		switch s["polarity"] {
		case "POSITIVE":
			posVotes += intensity
		case "NEGATIVE":
			negVotes += intensity
		default:
			neuVotes += intensity
		}
		totalWeight += intensity
	}

	nLangs = len(langSet)

	if totalWeight == 0 {
		return "NEUTRAL", 0, nLangs
	}

	posRatio := float64(posVotes) / float64(totalWeight)
	negRatio := float64(negVotes) / float64(totalWeight)

	switch {
	case posRatio > negRatio && posRatio > 0.4:
		polarity = "POSITIVE"
		confidence = posRatio
	case negRatio > posRatio && negRatio > 0.4:
		polarity = "NEGATIVE"
		confidence = negRatio
	default:
		polarity = "NEUTRAL"
		confidence = 1.0 - posRatio - negRatio
		if confidence < 0 {
			confidence = 0
		}
	}

	// Confidence boost for multi-language consensus
	if nLangs > 1 {
		confidence = confidence * (1 + 0.1*float64(nLangs-1))
		if confidence > 1 {
			confidence = 1
		}
	}

	return polarity, confidence, nLangs
}
