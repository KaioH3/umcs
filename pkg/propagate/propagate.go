// Package propagate implements cross-linguistic sentiment inference.
//
// If "terrible" (EN) is annotated as NEGATIVE/STRONG, then its cognates
// terrível (PT), terrible (ES/FR), terribile (IT) can inherit the same
// sentiment automatically — annotate once, propagate to all languages.
package propagate

import (
	"github.com/kak/lex-sentiment/pkg/lexdb"
	"github.com/kak/lex-sentiment/pkg/sentiment"
)

// Result describes one sentiment propagation from source to target.
type Result struct {
	TargetWord string
	TargetLang string
	WordID     uint32
	OldSent    uint32
	NewSent    uint32
}

// Run copies sentiment from annotated words to unannotated cognates
// within the same root family.
//
// Strategy: majority vote among annotated cognates' polarities.
// This avoids false propagation when cognates diverge (false friends).
func Run(lex *lexdb.Lexicon) []Result {
	var results []Result

	for _, root := range lex.Roots {
		// Need at least one word_id in this family to call Cognates
		if root.WordCount == 0 {
			continue
		}
		sampleID := root.RootID<<12 | 1
		cognates := lex.Cognates(sampleID)
		if len(cognates) == 0 {
			// Try with FirstWordIdx
			if int(root.FirstWordIdx) < len(lex.Words) {
				sampleID = lex.Words[root.FirstWordIdx].WordID
				cognates = lex.Cognates(sampleID)
			}
		}
		if len(cognates) == 0 {
			continue
		}

		// Collect annotated sentiment values
		var annotated []uint32
		for _, w := range cognates {
			if sentiment.Polarity(w.Sentiment) != sentiment.PolarityNeutral || sentiment.Intensity(w.Sentiment) != 0 {
				annotated = append(annotated, w.Sentiment)
			}
		}
		if len(annotated) == 0 {
			continue
		}

		consensusSent := majorityVote(annotated)

		// Apply to unannotated cognates in the word table
		for i := int(root.FirstWordIdx); i < int(root.FirstWordIdx)+int(root.WordCount); i++ {
			if i >= len(lex.Words) {
				break
			}
			w := &lex.Words[i]
			if w.Sentiment != 0 {
				continue
			}
			old := w.Sentiment
			w.Sentiment = consensusSent
			results = append(results, Result{
				TargetWord: lex.WordStr(w),
				TargetLang: lexdb.LangName(w.Lang),
				WordID:     w.WordID,
				OldSent:    old,
				NewSent:    consensusSent,
			})
		}
	}
	return results
}

func majorityVote(sentiments []uint32) uint32 {
	counts := make(map[uint32]int)
	for _, s := range sentiments {
		key := (s & sentiment.PolarityMask) | (s & sentiment.IntensityMask) | (s & sentiment.RoleMask)
		counts[key]++
	}
	var best uint32
	var bestCount int
	for k, v := range counts {
		if v > bestCount {
			bestCount = v
			best = k
		}
	}
	return best | sentiment.DomainGeneral
}
