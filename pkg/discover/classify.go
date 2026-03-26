package discover

import (
	"strings"

	"github.com/kak/lex-sentiment/pkg/seed"
	"github.com/kak/lex-sentiment/pkg/sentiment"
)

// Score holds the sentiment classification result for a word.
type Score struct {
	Polarity   string  // POSITIVE, NEGATIVE, NEUTRAL
	Intensity  string  // WEAK, MODERATE, STRONG, EXTREME, NONE
	Role       string  // EVALUATION, EMOTION, COGNITION, CONNECTOR, etc.
	Confidence float64 // 0.0–1.0
	Source     string  // "propagation", "definition", "morphology"
}

// ScoreViaPropagation uses majority vote among annotated cognates of the same root.
// Returns confidence proportional to unanimity of the annotated cognates.
func ScoreViaPropagation(rootID uint32, allWords []seed.Word) Score {
	var annotated []seed.Word
	for _, w := range allWords {
		if w.RootID == rootID && w.Sentiment != 0 {
			annotated = append(annotated, w)
		}
	}
	if len(annotated) == 0 {
		return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
			Confidence: 0, Source: "propagation"}
	}

	// Count polarity votes.
	polarityCount := map[string]int{}
	for _, w := range annotated {
		d := sentiment.Decode(w.Sentiment)
		polarityCount[d["polarity"]]++
	}

	bestPol := "NEUTRAL"
	bestCount := 0
	for pol, count := range polarityCount {
		if count > bestCount {
			bestCount = count
			bestPol = pol
		}
	}

	confidence := float64(bestCount) / float64(len(annotated))

	intensityCount := map[string]int{}
	roleCount := map[string]int{}
	for _, w := range annotated {
		d := sentiment.Decode(w.Sentiment)
		if d["polarity"] == bestPol {
			intensityCount[d["intensity"]]++
			roleCount[d["role"]]++
		}
	}

	return Score{
		Polarity:   bestPol,
		Intensity:  bestKey(intensityCount),
		Role:       bestKey(roleCount),
		Confidence: confidence,
		Source:     "propagation",
	}
}

// ScoreViaDefinition scans Wiktionary definition text for sentiment indicator words.
// Returns low confidence (0.4 max) since keyword matching is approximate.
func ScoreViaDefinition(defs []string) Score {
	text := strings.ToLower(strings.Join(defs, " "))

	posScore := 0
	negScore := 0

	for _, w := range positiveIndicators {
		if strings.Contains(text, w) {
			posScore++
		}
	}
	for _, w := range negativeIndicators {
		if strings.Contains(text, w) {
			negScore++
		}
	}

	total := posScore + negScore
	if total == 0 {
		return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
			Confidence: 0.1, Source: "definition"}
	}

	if posScore > negScore {
		conf := 0.2 + float64(posScore-negScore)/float64(total)*0.2
		intensity := "MODERATE"
		if posScore >= 3 {
			intensity = "STRONG"
		}
		return Score{Polarity: "POSITIVE", Intensity: intensity, Role: "EVALUATION",
			Confidence: conf, Source: "definition"}
	}
	if negScore > posScore {
		conf := 0.2 + float64(negScore-posScore)/float64(total)*0.2
		intensity := "MODERATE"
		if negScore >= 3 {
			intensity = "STRONG"
		}
		return Score{Polarity: "NEGATIVE", Intensity: intensity, Role: "EVALUATION",
			Confidence: conf, Source: "definition"}
	}
	return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
		Confidence: 0.1, Source: "definition"}
}

// ScoreViaMorphology classifies based on known prefix/suffix patterns.
// Returns low confidence (0.3–0.45) since morphology alone is unreliable.
func ScoreViaMorphology(word string) Score {
	w := strings.ToLower(word)

	// Negative prefixes → likely negates/reverses a concept
	for _, pfx := range []string{"un", "dis", "in", "im", "il", "ir", "mis", "non", "des", "anti", "mal"} {
		if strings.HasPrefix(w, pfx) && len(w) > len(pfx)+2 {
			return Score{Polarity: "NEGATIVE", Intensity: "WEAK", Role: "EVALUATION",
				Confidence: 0.35, Source: "morphology"}
		}
	}
	// Diminutive suffixes → positive or neutral with WEAK intensity
	for _, sfx := range []string{"inho", "inha", "chen", "lein", "ito", "ita", "ette", "let", "cule"} {
		if strings.HasSuffix(w, sfx) && len(w) > len(sfx)+2 {
			return Score{Polarity: "NEUTRAL", Intensity: "WEAK", Role: "EVALUATION",
				Confidence: 0.30, Source: "morphology"}
		}
	}
	// Superlative suffixes → amplifies existing polarity (we don't know direction)
	for _, sfx := range []string{"issimo", "issima", "ísimo", "érrimo"} {
		if strings.HasSuffix(w, sfx) {
			return Score{Polarity: "NEUTRAL", Intensity: "EXTREME", Role: "EVALUATION",
				Confidence: 0.25, Source: "morphology"}
		}
	}

	return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
		Confidence: 0.10, Source: "morphology"}
}

// BestScore picks the highest-confidence score among candidates.
func BestScore(scores ...Score) Score {
	best := Score{Confidence: -1}
	for _, s := range scores {
		if s.Confidence > best.Confidence {
			best = s
		}
	}
	if best.Confidence < 0 {
		return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
			Confidence: 0, Source: "fallback"}
	}
	return best
}

func bestKey(counts map[string]int) string {
	best := ""
	bestCount := 0
	for k, v := range counts {
		if v > bestCount {
			bestCount = v
			best = k
		}
	}
	return best
}

// positiveIndicators are English words that strongly suggest positive sentiment
// when found in a word's definition.
var positiveIndicators = []string{
	"good", "excellent", "great", "beautiful", "happy", "positive",
	"beneficial", "pleasant", "wonderful", "nice", "best", "perfect",
	"kind", "joyful", "love", "lovely", "bright", "clean", "pure",
	"healthy", "strong", "powerful", "skilled", "wise", "calm",
	"peaceful", "friendly", "generous", "brave", "honest", "fair",
}

// negativeIndicators are English words that strongly suggest negative sentiment.
var negativeIndicators = []string{
	"bad", "terrible", "awful", "poor", "negative", "harmful", "unpleasant",
	"worst", "evil", "wrong", "sad", "dark", "fear", "danger", "pain",
	"ugly", "dirty", "sick", "weak", "foolish", "cruel", "hate", "violence",
	"angry", "bitter", "corrupt", "false", "guilty", "hostile", "jealous",
	"lazy", "mad", "nervous", "offensive", "rude", "selfish", "stupid",
}
