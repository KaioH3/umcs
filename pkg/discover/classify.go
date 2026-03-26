package discover

import (
	"strings"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
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

// ScoreViaDefinition is the public entry point for definition-based scoring.
// It uses weighted keyword matching with negation/intensifier context windows.
// Max confidence is 0.75; above 0.60 threshold only when multiple strong
// unambiguous indicators agree.
func ScoreViaDefinition(defs []string) Score {
	return scoreDefinition(defs, nil)
}

// scoreDefinition is the full scorer used internally in classifyBest.
// When allWords is non-nil it also applies cross-lexicon evidence: if a
// definition word matches a known lexicon entry the polarity of that entry
// serves as additional high-quality evidence.
//
// Design rationale (VADER-inspired):
//  1. Weighted indicator lexicon — strong(1.0), moderate(0.6), weak(0.3)
//  2. Negation window — any negation token within 4 positions before an
//     indicator flips its sign ("not good" → negative contribution)
//  3. Intensifier window — amplifies indicator weight by 1.4×
//  4. Cross-lexicon evidence — definition words present in the UMCS lexicon
//     with known polarity contribute ±0.4 each (capped at ±0.6 total)
//  5. Confidence formula: floor=0.35, scales with signal strength and clarity;
//     reaches threshold (0.60) only with ≥2 moderate or ≥1 strong indicators
func scoreDefinition(defs []string, allWords []seed.Word) Score {
	if len(defs) == 0 {
		return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
			Confidence: 0.05, Source: "definition"}
	}

	text := strings.ToLower(strings.Join(defs, " "))
	tokens := strings.Fields(text)

	posScore := scoreTokens(tokens, posIndicators)
	negScore := scoreTokens(tokens, negIndicators)

	// Cross-lexicon evidence: definition tokens that match known lexicon norms.
	if allWords != nil && len(allWords) > 0 {
		normIndex := buildNormPolarityIndex(allWords)
		var lexPos, lexNeg float64
		for _, tok := range tokens {
			norm := lexdb.Normalize(tok)
			if len(norm) < 3 {
				continue
			}
			switch normIndex[norm] {
			case "POSITIVE":
				lexPos += 0.4
			case "NEGATIVE":
				lexNeg += 0.4
			}
		}
		// Cap cross-lexicon contribution to avoid dominating the signal.
		if lexPos > 0.6 {
			lexPos = 0.6
		}
		if lexNeg > 0.6 {
			lexNeg = 0.6
		}
		posScore += lexPos
		negScore += lexNeg
	}

	total := posScore + negScore
	if total < 0.1 {
		return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
			Confidence: 0.10, Source: "definition"}
	}

	// Clarity ratio: 1.0 = completely one-sided, 0.0 = perfectly balanced.
	net := posScore - negScore
	if net < 0 {
		net = -net
	}
	clarity := net / total

	// Confidence: floor 0.35, grows with signal strength and clarity, cap 0.75.
	conf := 0.35 + net*0.12 + clarity*0.15
	if conf > 0.75 {
		conf = 0.75
	}

	if posScore > negScore {
		intensity := intensityLabel(posScore)
		return Score{
			Polarity: "POSITIVE", Intensity: intensity, Role: "EVALUATION",
			Confidence: conf, Source: "definition",
		}
	}
	if negScore > posScore {
		intensity := intensityLabel(negScore)
		return Score{
			Polarity: "NEGATIVE", Intensity: intensity, Role: "EVALUATION",
			Confidence: conf, Source: "definition",
		}
	}
	return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
		Confidence: 0.10, Source: "definition"}
}

// scoreTokens sums weighted indicator contributions across a token list,
// applying negation-flip and intensifier-amplify context windows.
func scoreTokens(tokens []string, indicators []sentIndicator) float64 {
	total := 0.0
	for i, tok := range tokens {
		for _, ind := range indicators {
			if !strings.Contains(tok, ind.token) {
				continue
			}
			w := ind.weight
			negated := false
			for j := i - 4; j < i; j++ {
				if j < 0 {
					continue
				}
				t := tokens[j]
				if negationSet[t] {
					negated = !negated
				}
				if intensifierSet[t] {
					w *= 1.4
				}
			}
			if negated {
				total -= w
			} else {
				total += w
			}
		}
	}
	if total < 0 {
		total = -total
	}
	return total
}

// intensityLabel maps a raw weighted score to an intensity label.
func intensityLabel(score float64) string {
	switch {
	case score >= 2.5:
		return "EXTREME"
	case score >= 1.5:
		return "STRONG"
	case score >= 0.7:
		return "MODERATE"
	default:
		return "WEAK"
	}
}

// buildNormPolarityIndex returns a map from phonetic norm → "POSITIVE"/"NEGATIVE"
// for all words with non-neutral, non-zero sentiment.
func buildNormPolarityIndex(allWords []seed.Word) map[string]string {
	idx := make(map[string]string, len(allWords))
	for _, w := range allWords {
		if w.Sentiment == 0 {
			continue
		}
		d := sentiment.Decode(w.Sentiment)
		pol := d["polarity"]
		if pol == "POSITIVE" || pol == "NEGATIVE" {
			idx[w.Norm] = pol
		}
	}
	return idx
}

// SenseCoherent returns false when entry definitions are semantically incompatible
// with an existing root's meaning. This guards against phonetic false-positive root
// assignments caused by polysemous words (e.g. EN "gut"=intestine matching root
// "gut"=good, or "but"=however matching root "et"=and).
//
// It works by checking whether any significant keyword from rootMeaning appears in
// the definition text. Short or empty definitions get the benefit of the doubt.
func SenseCoherent(entryDefs []string, rootMeaning string) bool {
	if rootMeaning == "" || len(entryDefs) == 0 {
		return true
	}
	rootTokens := significantTokens(rootMeaning)
	if len(rootTokens) == 0 {
		return true
	}
	defText := strings.ToLower(strings.Join(entryDefs, " "))
	// Short definitions lack enough signal — be lenient.
	if len(strings.Fields(defText)) < 6 {
		return true
	}
	for _, rt := range rootTokens {
		if strings.Contains(defText, rt) {
			return true
		}
	}
	return false
}

// significantTokens lowercases and strips punctuation, discarding stopwords and
// tokens shorter than 3 runes.
func significantTokens(s string) []string {
	var result []string
	for _, w := range strings.Fields(strings.ToLower(s)) {
		w = strings.Trim(w, ".,;:\"'()[]{}–—/\\")
		if len([]rune(w)) >= 3 && !stopwordSet[w] {
			result = append(result, w)
		}
	}
	return result
}

// ScoreViaMorphology classifies based on known prefix/suffix patterns.
// Returns low confidence (0.30–0.45); used only as a tiebreaker below threshold.
func ScoreViaMorphology(word string) Score {
	w := strings.ToLower(word)
	for _, pfx := range negPrefixes {
		if strings.HasPrefix(w, pfx) && len(w) > len(pfx)+2 {
			return Score{Polarity: "NEGATIVE", Intensity: "WEAK", Role: "EVALUATION",
				Confidence: 0.35, Source: "morphology"}
		}
	}
	for _, sfx := range diminutiveSuffixes {
		if strings.HasSuffix(w, sfx) && len(w) > len(sfx)+2 {
			return Score{Polarity: "NEUTRAL", Intensity: "WEAK", Role: "EVALUATION",
				Confidence: 0.30, Source: "morphology"}
		}
	}
	for _, sfx := range superlativeSuffixes {
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

// ── Lexicons ──────────────────────────────────────────────────────────────────

// sentIndicator pairs a lowercase token with a sentiment weight.
// Matching uses strings.Contains so "love" matches "loving", "loved", etc.
// Use complete words for high-precision indicators to avoid false matches.
type sentIndicator struct {
	token  string
	weight float64 // 1.0=strong, 0.6=moderate, 0.3=weak
}

var posIndicators = []sentIndicator{
	// Strong (1.0) — unambiguous positive emotional or evaluative concepts
	{"love", 1.0}, {"excellent", 1.0}, {"wonderful", 1.0}, {"amazing", 1.0},
	{"outstanding", 1.0}, {"superb", 1.0}, {"magnificent", 1.0}, {"perfect", 1.0},
	{"joyful", 1.0}, {"elated", 1.0}, {"ecstatic", 1.0}, {"blissful", 1.0},
	{"glorious", 1.0}, {"splendid", 1.0}, {"brilliant", 1.0}, {"delightful", 1.0},
	{"admirable", 1.0}, {"noble", 1.0}, {"honorable", 1.0}, {"grateful", 1.0},
	{"sublime", 1.0}, {"exquisite", 1.0}, {"triumphant", 1.0}, {"radiant", 1.0},
	// Moderate (0.6) — clearly positive but more common/polysemous
	{"good", 0.6}, {"happy", 0.6}, {"positive", 0.6}, {"kind", 0.6},
	{"pleasant", 0.6}, {"beautiful", 0.6}, {"bright", 0.6}, {"clean", 0.6},
	{"pure", 0.6}, {"healthy", 0.6}, {"strong", 0.6}, {"wise", 0.6},
	{"calm", 0.6}, {"peaceful", 0.6}, {"friendly", 0.6}, {"generous", 0.6},
	{"brave", 0.6}, {"honest", 0.6}, {"fair", 0.6}, {"virtuous", 0.6},
	{"loyal", 0.6}, {"faithful", 0.6}, {"trustworthy", 0.6}, {"helpful", 0.6},
	{"compassionate", 0.6}, {"sincere", 0.6}, {"gentle", 0.6}, {"sweet", 0.6},
	{"charming", 0.6}, {"graceful", 0.6}, {"creative", 0.6}, {"inspired", 0.6},
	{"enthusiastic", 0.6}, {"courageous", 0.6}, {"resilient", 0.6}, {"humble", 0.6},
	{"cherish", 0.6}, {"treasure", 0.6}, {"admire", 0.6}, {"respect", 0.6},
	{"prosper", 0.6}, {"thrive", 0.6}, {"flourish", 0.6}, {"succeed", 0.6},
	{"comfort", 0.6}, {"pleasure", 0.6}, {"delight", 0.6}, {"warmth", 0.6},
	// Weak (0.3) — positive but often context-dependent
	{"nice", 0.3}, {"decent", 0.3}, {"satisfactory", 0.3}, {"acceptable", 0.3},
	{"benefit", 0.3}, {"improve", 0.3}, {"progress", 0.3}, {"success", 0.3},
	{"achieve", 0.3}, {"gain", 0.3}, {"enjoy", 0.3}, {"appreciate", 0.3},
	{"thank", 0.3}, {"praise", 0.3}, {"reward", 0.3}, {"honor", 0.3},
	{"protect", 0.3}, {"support", 0.3}, {"unite", 0.3}, {"cooperate", 0.3},
}

var negIndicators = []sentIndicator{
	// Strong (1.0) — unambiguous negative concepts
	{"terrible", 1.0}, {"awful", 1.0}, {"horrible", 1.0}, {"atrocious", 1.0},
	{"devastating", 1.0}, {"catastrophic", 1.0}, {"vicious", 1.0}, {"wicked", 1.0},
	{"cruel", 1.0}, {"brutal", 1.0}, {"torment", 1.0}, {"agony", 1.0},
	{"miserable", 1.0}, {"despair", 1.0}, {"malevolent", 1.0}, {"abhorrent", 1.0},
	{"monstrous", 1.0}, {"heinous", 1.0}, {"reprehensible", 1.0}, {"loathe", 1.0},
	{"hatred", 1.0}, {"dread", 1.0}, {"nightmare", 1.0}, {"horrific", 1.0},
	// Moderate (0.6)
	{"bad", 0.6}, {"negative", 0.6}, {"harmful", 0.6}, {"unpleasant", 0.6},
	{"pain", 0.6}, {"sad", 0.6}, {"fear", 0.6}, {"angry", 0.6},
	{"wrong", 0.6}, {"corrupt", 0.6}, {"false", 0.6}, {"guilty", 0.6},
	{"hostile", 0.6}, {"jealous", 0.6}, {"selfish", 0.6}, {"coward", 0.6},
	{"shame", 0.6}, {"humiliate", 0.6}, {"betray", 0.6}, {"deceit", 0.6},
	{"suffer", 0.6}, {"grief", 0.6}, {"regret", 0.6}, {"anxious", 0.6},
	{"depress", 0.6}, {"lonely", 0.6}, {"isolat", 0.6}, {"rage", 0.6},
	{"resentment", 0.6}, {"disappoint", 0.6}, {"frustrat", 0.6}, {"despise", 0.6},
	{"disgust", 0.6}, {"contempt", 0.6}, {"distress", 0.6}, {"panic", 0.6},
	{"menace", 0.6}, {"threaten", 0.6}, {"oppress", 0.6}, {"exploit", 0.6},
	// Weak (0.3)
	{"poor", 0.3}, {"dark", 0.3}, {"danger", 0.3}, {"ugly", 0.3},
	{"dirty", 0.3}, {"sick", 0.3}, {"foolish", 0.3}, {"lazy", 0.3},
	{"nervous", 0.3}, {"offensive", 0.3}, {"rude", 0.3}, {"fail", 0.3},
	{"loss", 0.3}, {"absent", 0.3}, {"empty", 0.3}, {"void", 0.3},
	{"problem", 0.3}, {"difficult", 0.3}, {"obstacle", 0.3}, {"conflict", 0.3},
	{"violence", 0.3}, {"fight", 0.3}, {"attack", 0.3}, {"destroy", 0.3},
}

// negationSet contains tokens that flip the polarity of nearby indicators.
var negationSet = map[string]bool{
	"not": true, "no": true, "never": true, "without": true,
	"lack": true, "lacking": true, "absence": true, "absent": true,
	"opposite": true, "unlike": true, "against": true,
	"deny": true, "denying": true, "denied": true, "refuses": true,
}

// intensifierSet contains tokens that amplify nearby indicator weights by 1.4×.
var intensifierSet = map[string]bool{
	"very": true, "extremely": true, "deeply": true, "highly": true,
	"utterly": true, "absolutely": true, "completely": true, "totally": true,
	"profoundly": true, "intensely": true, "strongly": true, "greatly": true,
	"particularly": true, "especially": true, "remarkably": true, "exceptionally": true,
}

var negPrefixes = []string{
	"un", "dis", "in", "im", "il", "ir", "mis", "non", "des", "anti", "mal",
}

var diminutiveSuffixes = []string{
	"inho", "inha", "chen", "lein", "ito", "ita", "ette", "let", "cule",
}

var superlativeSuffixes = []string{
	"issimo", "issima", "ísimo", "érrimo",
}

// stopwordSet is used by significantTokens to discard grammatical function words.
var stopwordSet = map[string]bool{
	"a": true, "an": true, "the": true, "of": true, "or": true, "and": true,
	"to": true, "is": true, "are": true, "in": true, "it": true, "that": true,
	"for": true, "as": true, "by": true, "from": true, "on": true, "at": true,
	"be": true, "was": true, "has": true, "have": true, "with": true,
	"its": true, "his": true, "her": true, "their": true, "this": true,
	"which": true, "who": true, "when": true, "where": true, "how": true,
}
