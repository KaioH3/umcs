package analyze

import (
	"regexp"
	"strings"
)

type SarcasmResult struct {
	IsSarcastic bool     `json:"is_sarcastic"`
	Confidence  float64  `json:"confidence"`
	Patterns    []string `json:"patterns"`
	Text        string   `json:"text"`
}

var (
	laughingPatterns  = regexp.MustCompile(`(?i)(haha+|kk+|rs+|hehe+|huh+|lol+)`)
	quotePattern      = regexp.MustCompile(`["']`)
	questionPattern   = regexp.MustCompile(`\?{2,}`)
	contrastPatterns  = regexp.MustCompile(`(?i)(oh +wonderful|oh +great|oh +fantastic|what +a +great|wonderful|great +job|thanks +a +lot|wow|as +if|sure|obviously|clearly)`)
	sarcasticQuestion = regexp.MustCompile(`(?i)^(yeah +right|really|you +think|i +don'?t +think +so|me +too|obviously|seriously)`)
)

func DetectSarcasm(text string) *SarcasmResult {
	patterns := []string{}
	score := 0.0
	lower := strings.ToLower(text)

	if laughingPatterns.MatchString(text) {
		patterns = append(patterns, "laughing")
		score += 0.3
	}

	if strings.Contains(text, "oo") || strings.Contains(text, "aa") || strings.Contains(text, "ee") || strings.Contains(text, "!!") {
		patterns = append(patterns, "elongation")
		score += 0.2
	}

	if quotePattern.MatchString(text) {
		count := len(quotePattern.FindAllString(text, -1))
		if count >= 2 {
			patterns = append(patterns, "quotation_marks")
			score += 0.15
		}
	}

	if questionPattern.MatchString(text) {
		patterns = append(patterns, "rhetorical_question")
		score += 0.15
	}

	if contrastPatterns.MatchString(text) {
		patterns = append(patterns, "ironic_expression")
		score += 0.25
	}

	words := strings.Fields(lower)
	if len(words) > 0 {
		firstWord := words[0]
		if strings.HasSuffix(firstWord, "?") || sarcasticQuestion.MatchString(text) {
			patterns = append(patterns, "sarcastic_question")
			score += 0.2
		}
	}

	if strings.Contains(lower, "not ") && strings.Contains(lower, " but ") {
		patterns = append(patterns, "contrast")
		score += 0.2
	}

	if strings.Contains(lower, "love") && (strings.Contains(lower, "hate") || strings.Contains(lower, "terrible") || strings.Contains(lower, "worst")) {
		patterns = append(patterns, "contrast")
		score += 0.25
	}

	confidence := score
	if confidence > 1.0 {
		confidence = 1.0
	}

	isSarcastic := confidence >= 0.4

	return &SarcasmResult{
		IsSarcastic: isSarcastic,
		Confidence:  confidence,
		Patterns:    patterns,
		Text:        text,
	}
}
