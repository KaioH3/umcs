// Package propagate implements cross-linguistic sentiment inference.
//
// If "terrible" (EN) is annotated as NEGATIVE/STRONG, then its cognates
// terrível (PT), terrible (ES/FR), terribile (IT) can inherit the same
// sentiment automatically — annotate once, propagate to all languages.
package propagate

import (
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/morpheme"
	"github.com/kak/umcs/pkg/sentiment"
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
		sampleID, _ := morpheme.MakeWordID(root.RootID, 1)
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

// PropagateExtended extends majority-vote propagation to VAD fields and flag
// fields that can be safely inferred from cognates.
//
// Fields propagated (consensus among annotated cognates):
//   - Arousal, Dominance, AoA, Concreteness (from Sentiment bits 5..0 and 28)
//   - POS (from Sentiment bits 31..29) — only when all annotated cognates agree
//   - Register, Ontological, Polysemy (from Flags bits 15..8 and 19..16)
//   - Syllable count, Stress (from Flags bits 31..26) — copied from first
//     IPA-annotated cognate, since these are phonological universals of the root
//
// Fields NOT propagated:
//   - IPA pronunciation (language-specific phonology; cannot transfer across languages)
//   - FreqRank (corpus-specific per language; use FreqRankProxy instead)
//   - Valency (language-specific argument structure)
//   - Polarity, Intensity, Role, Domain — handled by the existing Run() function
//
// Returns number of word records updated.
func PropagateExtended(lex *lexdb.Lexicon) int {
	updated := 0

	for ri := range lex.Roots {
		root := &lex.Roots[ri]
		if root.WordCount == 0 {
			continue
		}

		start := int(root.FirstWordIdx)
		end := start + int(root.WordCount)
		if start >= len(lex.Words) || end > len(lex.Words) {
			continue
		}
		cognates := lex.Words[start:end]

		// Collect annotated VAD values from cognates that have non-zero sentiment
		var (
			arousals    []uint32
			dominances  []uint32
			aoaVals     []uint32
			posVals     []uint32
			concrete    []bool
			registers   []uint32
			onto        []uint32
			polysemy    []uint32
			syllables   uint32
			stress      uint32
			hasPhon     bool
		)
		for _, w := range cognates {
			s := w.Sentiment
			if s == 0 {
				continue // skip unannotated
			}
			if ar := (s >> 4) & 0x3; ar != 0 {
				arousals = append(arousals, ar)
			}
			if dom := (s >> 2) & 0x3; dom != 0 {
				dominances = append(dominances, dom)
			}
			if aoa := s & 0x3; aoa != 0 {
				aoaVals = append(aoaVals, aoa)
			}
			posVals = append(posVals, (s>>29)&0x7)
			concrete = append(concrete, (s>>28)&1 == 1)

			flags := w.Flags
			if reg := (flags >> 8) & 0xF; reg != 0 {
				registers = append(registers, reg)
			}
			if o := (flags >> 12) & 0xF; o != 0 {
				onto = append(onto, o)
			}
			if poly := (flags >> 16) & 0xF; poly != 0 {
				polysemy = append(polysemy, poly)
			}
			if !hasPhon {
				if syl := (flags >> 28) & 0xF; syl != 0 {
					syllables = syl
					stress = (flags >> 26) & 0x3
					hasPhon = true
				}
			}
		}

		if len(arousals) == 0 && len(registers) == 0 && !hasPhon {
			continue
		}

		// Compute majority values
		arousalConsensus := majorityUint32(arousals)
		dominanceConsensus := majorityUint32(dominances)
		aoaConsensus := majorityUint32(aoaVals)
		registerConsensus := majorityUint32(registers)
		ontoConsensus := majorityUint32(onto)
		polysemyConsensus := majorityUint32(polysemy)

		// POS consensus: only propagate when all annotated cognates agree
		var posConsensus uint32
		if len(posVals) > 0 {
			allSame := true
			for _, p := range posVals[1:] {
				if p != posVals[0] {
					allSame = false
					break
				}
			}
			if allSame {
				posConsensus = posVals[0]
			}
		}

		// Majority concrete: true if majority
		var concreteMajority bool
		if len(concrete) > 0 {
			trueCount := 0
			for _, c := range concrete {
				if c {
					trueCount++
				}
			}
			concreteMajority = trueCount*2 > len(concrete)
		}

		// Apply to words missing these fields
		for i := start; i < end; i++ {
			w := &lex.Words[i]
			if w.Sentiment == 0 {
				continue // skip completely unannotated words
			}
			changed := false

			s := w.Sentiment
			flags := w.Flags

			// Arousal
			if (s>>4)&0x3 == 0 && arousalConsensus != 0 {
				s = (s &^ (0x3 << 4)) | (arousalConsensus << 4)
				changed = true
			}
			// Dominance
			if (s>>2)&0x3 == 0 && dominanceConsensus != 0 {
				s = (s &^ (0x3 << 2)) | (dominanceConsensus << 2)
				changed = true
			}
			// AoA
			if s&0x3 == 0 && aoaConsensus != 0 {
				s = (s &^ 0x3) | aoaConsensus
				changed = true
			}
			// Concreteness
			if (s>>28)&1 == 0 && len(concrete) > 0 && concreteMajority {
				s |= 1 << 28
				changed = true
			}
			// POS (only when unannotated and consensus exists)
			if (s>>29)&0x7 == 0 && posConsensus != 0 {
				s = (s &^ (0x7 << 29)) | (posConsensus << 29)
				changed = true
			}
			// Register
			if (flags>>8)&0xF == 0 && registerConsensus != 0 {
				flags = (flags &^ (0xF << 8)) | (registerConsensus << 8)
				changed = true
			}
			// Ontological
			if (flags>>12)&0xF == 0 && ontoConsensus != 0 {
				flags = (flags &^ (0xF << 12)) | (ontoConsensus << 12)
				changed = true
			}
			// Polysemy
			if (flags>>16)&0xF == 0 && polysemyConsensus != 0 {
				flags = (flags &^ (0xF << 16)) | (polysemyConsensus << 16)
				changed = true
			}
			// Syllables + Stress (from IPA-annotated cognate)
			if hasPhon && (flags>>28)&0xF == 0 {
				flags = (flags &^ (0xF << 28)) | (syllables << 28)
				flags = (flags &^ (0x3 << 26)) | (stress << 26)
				changed = true
			}

			if changed {
				w.Sentiment = s
				w.Flags = flags
				updated++
			}
		}
	}
	return updated
}

// majorityUint32 returns the most frequent value in vals, or 0 if vals is empty.
func majorityUint32(vals []uint32) uint32 {
	if len(vals) == 0 {
		return 0
	}
	counts := make(map[uint32]int, len(vals))
	for _, v := range vals {
		counts[v]++
	}
	var best uint32
	bestCount := 0
	for v, c := range counts {
		if c > bestCount {
			bestCount = c
			best = v
		}
	}
	return best
}

func majorityVote(sentiments []uint32) uint32 {
	counts := make(map[uint32]int)
	for _, s := range sentiments {
		key := (s & sentiment.PolarityMask) | (s & sentiment.IntensityMask) | (s & sentiment.RoleMask) | (s & sentiment.DomainMask)
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
	return best
}
