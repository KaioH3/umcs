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

	"github.com/kak/lex-sentiment/pkg/lexdb"
	"github.com/kak/lex-sentiment/pkg/sentiment"
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
	case totalScore > 2:
		verdict = "POSITIVE"
	case totalScore < -2:
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
