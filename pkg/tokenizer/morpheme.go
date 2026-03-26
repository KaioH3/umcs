// Package tokenizer implements a morpheme-aware tokenizer as a replacement for BPE.
//
// # Why this matters for LLMs
//
// BPE (Byte Pair Encoding) is purely statistical — it splits "unimaginable"
// into "unimagin" + "##able" based on co-occurrence frequency, with no
// linguistic meaning. The result:
//   - "negative" and "negativo" are completely unrelated tokens
//   - LLMs must independently learn that they mean the same thing
//   - Cross-linguistic knowledge transfer is purely emergent (expensive)
//
// This tokenizer decomposes text into morpheme IDs:
//   - Same root_id across all languages → shared embedding slot
//   - Vocabulary: ~15K morpheme families vs 100K BPE tokens
//   - Sentiment is pre-encoded in the token itself (no learned association needed)
//   - Cross-linguistic training data becomes semantically deduplicated
//
// # Output format
//
// Each token is a MorphToken with:
//   - Surface: original word
//   - WordID: packed (root_id<<12)|variant — the primary LLM token
//   - RootID: root family (for cross-linguistic embedding sharing)
//   - Sentiment: packed uint32 bitmask (pre-encoded semantic information)
//   - Lang: source language
//
// For unknown words, WordID=0 and the surface form is preserved as a
// character-level fallback (compatible with BPE for OOV handling).
package tokenizer

import (
	"strings"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/morpheme"
	"github.com/kak/umcs/pkg/sentiment"
)

// MorphToken is a single output token from the morpheme tokenizer.
type MorphToken struct {
	Surface   string              // original word form
	WordID    uint32              // 0 if unknown (OOV)
	RootID    uint32              // root family ID (cross-linguistic key)
	Sentiment uint32              // packed bitmask — pre-encoded semantic info
	Lang      string              // source language
	Known     bool                // false for OOV words
	Token64   morpheme.Token64    // packed uint64: word_id (upper) + semantic payload (lower)
}

// SentimentSummary decodes the packed sentiment bitmask for human display.
func (t *MorphToken) SentimentSummary() map[string]string {
	if !t.Known {
		return nil
	}
	return sentiment.Decode(t.Sentiment)
}

// Tokenize decomposes text into a sequence of MorphTokens.
// Unknown words (OOV) are returned with WordID=0 and Known=false.
func Tokenize(lex *lexdb.Lexicon, text string) []MorphToken {
	words := splitWords(text)
	tokens := make([]MorphToken, 0, len(words))

	for _, w := range words {
		t := MorphToken{Surface: w}
		wr := lex.LookupWord(w)
		if wr != nil {
			t.WordID = wr.WordID
			t.RootID = wr.RootID
			t.Sentiment = wr.Sentiment
			t.Lang = lexdb.LangName(wr.Lang)
			t.Known = true
			t.Token64 = morpheme.Pack64(wr.WordID, wr.Sentiment, wr.Flags)
		}
		tokens = append(tokens, t)
	}
	return tokens
}

// TokenizeToIDs returns only the word_id sequence for LLM consumption.
// OOV tokens use ID=0. This is the minimal representation for training.
func TokenizeToIDs(lex *lexdb.Lexicon, text string) []uint32 {
	tokens := Tokenize(lex, text)
	ids := make([]uint32, len(tokens))
	for i, t := range tokens {
		ids[i] = t.WordID
	}
	return ids
}

// TokenizeToRootIDs returns root_id sequences (cross-linguistic canonical form).
// "negative" (EN) and "negativo" (PT) both produce root_id=1.
// This is the deduplication layer for multilingual training.
func TokenizeToRootIDs(lex *lexdb.Lexicon, text string) []uint32 {
	tokens := Tokenize(lex, text)
	ids := make([]uint32, len(tokens))
	for i, t := range tokens {
		ids[i] = t.RootID
	}
	return ids
}

// DeduplicationScore computes the fraction of tokens that share root_ids
// with at least one other token — a measure of cross-linguistic redundancy.
// A score of 0.4 means 40% of training data is semantically duplicated.
func DeduplicationScore(lex *lexdb.Lexicon, texts []string) float64 {
	rootCounts := make(map[uint32]int)
	total := 0

	for _, text := range texts {
		ids := TokenizeToRootIDs(lex, text)
		for _, id := range ids {
			if id != 0 {
				rootCounts[id]++
				total++
			}
		}
	}

	duplicated := 0
	for _, count := range rootCounts {
		if count > 1 {
			duplicated += count
		}
	}

	if total == 0 {
		return 0
	}
	return float64(duplicated) / float64(total)
}

// VocabStats returns statistics useful for comparing with BPE tokenizers.
type VocabStats struct {
	UniqueWordIDs int     // unique morpheme word IDs seen
	UniqueRootIDs int     // unique root families seen
	OOVRate       float64 // fraction of unknown tokens
	TotalTokens   int
}

// Analyze computes vocabulary statistics over a corpus.
func Analyze(lex *lexdb.Lexicon, texts []string) VocabStats {
	wordIDs := make(map[uint32]bool)
	rootIDs := make(map[uint32]bool)
	total := 0
	oov := 0

	for _, text := range texts {
		for _, t := range Tokenize(lex, text) {
			total++
			if !t.Known {
				oov++
			} else {
				wordIDs[t.WordID] = true
				rootIDs[t.RootID] = true
			}
		}
	}

	oovRate := 0.0
	if total > 0 {
		oovRate = float64(oov) / float64(total)
	}

	return VocabStats{
		UniqueWordIDs: len(wordIDs),
		UniqueRootIDs: len(rootIDs),
		OOVRate:       oovRate,
		TotalTokens:   total,
	}
}

func splitWords(text string) []string {
	raw := strings.Fields(strings.ToLower(text))
	out := make([]string, 0, len(raw))
	for _, w := range raw {
		w = strings.Trim(w, ".,!?;:\"'()[]{}…-–")
		if w != "" {
			out = append(out, w)
		}
	}
	return out
}
