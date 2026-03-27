// Package classify provides a logistic regression classifier trained on
// features extracted from UMCS Token64 word embeddings.
//
// Feature extraction turns a 64-bit Token64 (polarity, intensity, role, POS,
// arousal, dominance, phonology, semantic relations) into a fixed-size float64
// vector. The classifier then maps this vector to a sentiment class using
// softmax logistic regression trained with the Adam optimizer.
//
// This is the "intelligence layer" on top of the deterministic UMCS lexicon:
// the lexicon provides a linguistic prior equivalent to weeks of LLM training,
// and the classifier learns which combinations of features predict sentiment
// best given user data.
package classify

import (
	"math/bits"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/phon"
	"github.com/kak/umcs/pkg/sentiment"
)

// NFeatures is the dimensionality of the feature vector.
const NFeatures = 48

// Feature indices — each constant is an index into a FeatureVec.
// All features are normalized to [0,1] or {-1,0,+1}.
const (
	// ── Sentiment dimensions (from WordRecord.Sentiment uint32) ──────────────
	FPolarity  = 0 // -1=NEGATIVE, 0=NEUTRAL/AMBIGUOUS, +1=POSITIVE
	FIntensity = 1 // 0.0=NONE .. 1.0=EXTREME (÷4)
	FArousal   = 2 // 0.0=NONE .. 1.0=HIGH (÷3)
	FDominance = 3 // 0.0=NONE .. 1.0=HIGH (÷3)
	FAoA       = 4 // 0.0=EARLY .. 1.0=TECHNICAL (÷3)
	FConcrete  = 5 // 1=concrete, 0=abstract

	// ── Phonology (from WordRecord.Flags upper bits via pkg/phon) ────────────
	FIrony     = 6  // 1 if IronyCapable flag set
	FNeologism = 7  // 1 if Neologism flag set
	FSyllables = 8  // syllable count ÷ 15 (0=unknown)
	FStress    = 9  // stress: 0=unknown, 0.33=final, 0.67=penult, 1.0=antepenult
	FValency   = 10 // Tesnière valency class ÷ 5 (0=N/A .. 1=modal)

	// ── Semantic relations (from RootRecord) ──────────────────────────────────
	FHasAntonym  = 11 // 1 if AntonymRootID != 0
	FHasHypernym = 12 // 1 if HypernymRootID != 0
	FHasSynonym  = 13 // 1 if SynonymRootID != 0

	// ── Context window (sentence-level scope) ─────────────────────────────────
	FNegCtx = 14 // 1 if negation marker seen in last 3 tokens
	FIntCtx = 15 // 1 if intensifier seen in last 2 tokens
	FDwnCtx = 16 // 1 if downtoner seen in last 2 tokens

	// ── Semantic role one-hot (11 values, indices 17..27) ────────────────────
	// Index = FRoleBase + role_enum_value
	// NONE=17, EVALUATION=18, EMOTION=19, COGNITION=20, VOLITION=21,
	// CAUSATION=22, TEMPORAL=23, QUANTIFIER=24, CONNECTOR=25,
	// NEGATION_MARKER=26, INTENSIFIER=27
	FRoleBase = 17

	// ── Lexical type (single normalized scalars) ──────────────────────────────
	FPOS         = 28 // POS enum ÷ 7 (OTHER=0 .. CONJ=1)
	FRegister    = 29 // register tier ÷ 10
	FOntological = 30 // ontological category ÷ 13
	FLang        = 31 // language ID ÷ 23 (PT=0 .. HE=1)

	// ── Lexical salience (indices 32..36) ────────────────────────────────────
	// Scientific basis: Zipf's law + Warriner VAD norms
	FFreqRank     = 32 // FreqRank÷50000: 0=very common, 1=very rare (hapax)
	FPolysemyF    = 33 // polysemy tier ÷ 15: high → semantically ambiguous → NEUTRAL
	FCultural     = 34 // 1 if CulturalSpecific flag set (lexdb.CulturalSpecific)
	FCognateCount = 35 // popcount(LangCoverage)÷24: cross-ling spread of root family
	FWordLen      = 36 // surface byte length ÷ 20: phonological complexity proxy

	// ── Language family (index 37) ────────────────────────────────────────────
	// Romance=0.0, Germanic=0.2, Slavic=0.4, Semitic=0.6, CJK=0.8, Other=1.0
	FLangFamily = 37

	// ── Semantic polarity inheritance (indices 38..39) ────────────────────────
	// Scientific basis: WordNet polarity inversion (Fellbaum 1998);
	// sentiment inheritance (Kim & Hovy 2006).
	FAntPolarity = 38 // antonym root canonical polarity: -1/0/+1
	FHypPolarity = 39 // hypernym root canonical polarity: -1/0/+1

	// ── IPA-derived phonological affect (indices 40..43) ─────────────────────
	// Based on Bouba/Kiki effect (Ramachandran 2001) and cross-linguistic
	// phonaesthetics (Magnus 2001). All 0.0 when no IPA annotation exists.
	FIPACVRatio   = 40 // consonant/(consonant+vowel) ratio from IPA
	FIPAOpenVowel = 41 // fraction of open vowels (a, æ, ɛ, ɑ, ɐ) among vowels
	FIPANasals    = 42 // 1 if IPA contains nasals (m, n, ŋ, ɲ, ɱ, ɴ)
	FIPASibilants = 43 // 1 if IPA contains sibilants (s, z, ʃ, ʒ)

	// ── Morphological structure (indices 44..45) ──────────────────────────────
	FEtymDepth = 44 // etymology chain depth ÷ 5: deep → technical register → NEG/NEU
	FWordCount = 45 // RootRecord.WordCount ÷ 20: morpheme family richness

	// 46..47 reserved for future IPA features (front/back vowel contrast)
)

// FeatureVec is the fixed-size input vector for the classifier.
type FeatureVec [NFeatures]float64

// ZeroLeakyFeatures zeroes the FPolarity dimension for training use.
//
// FPolarity is derived from WordRecord.Sentiment — the same field used to
// generate the training label. Including it causes the classifier to learn
// the identity function (F1=1.000 trivially). Always call this on training
// examples; NEVER call it at inference time (FPolarity is a valid signal
// for context-based predictions like negation flipping).
func (f *FeatureVec) ZeroLeakyFeatures() {
	f[FPolarity] = 0
}

// Context holds the scope resolution state at a token position.
type Context struct {
	NegInWindow bool // a negation marker was seen in the last 3 tokens
	IntInWindow bool // an intensifier was seen in the last 2 tokens
	DwnInWindow bool // a downtoner was seen in the last 2 tokens
}

// Extract builds a 48-dimensional FeatureVec from a word record, its root,
// and sentence context. root may be nil (semantic-relation features = 0).
// lex may be nil (features requiring full lexicon = 0). ctx may be zero-value
// for isolated-word lookups.
func Extract(lex *lexdb.Lexicon, w *lexdb.WordRecord, root *lexdb.RootRecord, ctx Context) FeatureVec {
	var f FeatureVec
	s := w.Sentiment

	// ── Polarity ───────────────────────────────────────────────────────────────
	switch sentiment.Polarity(s) {
	case sentiment.PolarityPositive:
		f[FPolarity] = 1.0
	case sentiment.PolarityNegative:
		f[FPolarity] = -1.0
	// NEUTRAL and AMBIGUOUS → 0.0
	}

	// ── Intensity (field is 4-bit 0..15, semantic max = 4/EXTREME; cap) ───────
	intensity := sentiment.Intensity(s)
	if intensity > 4 {
		intensity = 4
	}
	f[FIntensity] = float64(intensity) / 4.0

	// ── VAD axes ───────────────────────────────────────────────────────────────
	f[FArousal] = float64(sentiment.Arousal(s)) / 3.0
	f[FDominance] = float64(sentiment.Dominance(s)) / 3.0
	f[FAoA] = float64(sentiment.AOA(s)) / 3.0

	// ── Concreteness ───────────────────────────────────────────────────────────
	if sentiment.IsConcrete(s) {
		f[FConcrete] = 1.0
	}

	// ── Phonology (packed in Flags upper bits) ─────────────────────────────────
	flags := w.Flags
	if flags&(1<<22) != 0 {
		f[FIrony] = 1.0
	}
	if flags&(1<<21) != 0 {
		f[FNeologism] = 1.0
	}
	f[FSyllables] = float64(phon.Syllables(flags)) / 15.0

	switch phon.Stress(flags) {
	case phon.StressFinal:
		f[FStress] = 1.0 / 3.0
	case phon.StressPenultimate:
		f[FStress] = 2.0 / 3.0
	case phon.StressAntepenultimate:
		f[FStress] = 1.0
	}

	// phon.Valency(flags) returns the masked bits (e.g. ValencyModal = 5<<23).
	// >>23 extracts the enum value (0..5) before dividing by 5.
	// Field is 3-bit (0..7); cap at 5 to guard against invalid values.
	valencyEnum := phon.Valency(flags) >> 23
	if valencyEnum > 5 {
		valencyEnum = 5
	}
	f[FValency] = float64(valencyEnum) / 5.0

	// ── Semantic relations (root) ───────────────────────────────────────────────
	if root != nil {
		if root.AntonymRootID != 0 {
			f[FHasAntonym] = 1.0
		}
		if root.HypernymRootID != 0 {
			f[FHasHypernym] = 1.0
		}
		if root.SynonymRootID != 0 {
			f[FHasSynonym] = 1.0
		}
	}

	// ── Context window ─────────────────────────────────────────────────────────
	if ctx.NegInWindow {
		f[FNegCtx] = 1.0
	}
	if ctx.IntInWindow {
		f[FIntCtx] = 1.0
	}
	if ctx.DwnInWindow {
		f[FDwnCtx] = 1.0
	}

	// ── Semantic role one-hot ──────────────────────────────────────────────────
	role := sentiment.Role(s)
	if role < 11 {
		f[FRoleBase+int(role)] = 1.0
	}

	// ── Lexical type scalars ───────────────────────────────────────────────────
	f[FPOS] = float64(sentiment.POS(s)) / 7.0
	// Register: 4-bit field (0..15), semantic max = 10 (RegisterRegional); cap.
	reg := (flags >> 8) & 0xF
	if reg > 10 {
		reg = 10
	}
	f[FRegister] = float64(reg) / 10.0
	// Ontological: 4-bit field (0..15), semantic max = 13 (OntoAbstract); cap.
	onto := (flags >> 12) & 0xF
	if onto > 13 {
		onto = 13
	}
	f[FOntological] = float64(onto) / 13.0
	f[FLang] = float64(w.Lang) / 23.0

	// ── Lexical salience (32..36) ──────────────────────────────────────────────
	if w.FreqRank > 0 {
		freq := float64(w.FreqRank) / 50000.0
		if freq > 1.0 {
			freq = 1.0
		}
		f[FFreqRank] = freq
	}
	f[FPolysemyF] = float64((flags>>16)&0xF) / 15.0
	if flags&(1<<20) != 0 { // lexdb.CulturalSpecific
		f[FCultural] = 1.0
	}
	if root != nil {
		cog := float64(bits.OnesCount32(root.LangCoverage)) / 24.0
		if cog > 1.0 {
			cog = 1.0
		}
		f[FCognateCount] = cog
	}
	if lex != nil {
		wlen := float64(len(lex.WordStr(w))) / 20.0
		if wlen > 1.0 {
			wlen = 1.0
		}
		f[FWordLen] = wlen
	}

	// ── Language family (37) ───────────────────────────────────────────────────
	f[FLangFamily] = langFamily(w.Lang)

	// ── Semantic polarity inheritance (38..39) ────────────────────────────────
	// FAntPolarity: if the antonym root is POSITIVE, this word is likely NEGATIVE.
	// FHypPolarity: inherits polarity tendency from the hypernym (is-a parent).
	if lex != nil && root != nil {
		if ant := lex.Antonym(root); ant != nil {
			if cw := lex.RootCanonicalWord(ant.RootID); cw != nil {
				switch sentiment.Polarity(cw.Sentiment) {
				case sentiment.PolarityPositive:
					f[FAntPolarity] = 1.0
				case sentiment.PolarityNegative:
					f[FAntPolarity] = -1.0
				}
			}
		}
		if hyp := lex.Hypernym(root); hyp != nil {
			if cw := lex.RootCanonicalWord(hyp.RootID); cw != nil {
				switch sentiment.Polarity(cw.Sentiment) {
				case sentiment.PolarityPositive:
					f[FHypPolarity] = 1.0
				case sentiment.PolarityNegative:
					f[FHypPolarity] = -1.0
				}
			}
		}
	}

	// ── IPA phonological affect (40..43) ──────────────────────────────────────
	if lex != nil {
		pron := lex.WordPron(w)
		cvRatio, openVowelFrac, hasNasals, hasSibilants := parseIPA(pron)
		f[FIPACVRatio] = cvRatio
		f[FIPAOpenVowel] = openVowelFrac
		if hasNasals {
			f[FIPANasals] = 1.0
		}
		if hasSibilants {
			f[FIPASibilants] = 1.0
		}
	}

	// ── Morphological structure (44..45) ──────────────────────────────────────
	if lex != nil {
		chain := lex.EtymologyChain(w.RootID)
		depth := float64(len(chain)-1) / 5.0
		if depth < 0 {
			depth = 0
		}
		if depth > 1.0 {
			depth = 1.0
		}
		f[FEtymDepth] = depth
	}
	if root != nil {
		wc := float64(root.WordCount) / 20.0
		if wc > 1.0 {
			wc = 1.0
		}
		f[FWordCount] = wc
	}

	return f
}

// parseIPA extracts phonological affect features from an IPA pronunciation string.
// Returns zero values if pron is empty. Pure function, no external dependencies.
//
// Scientific basis:
//   - C/V ratio: consonant-heavy words tend toward perceived harshness (Bouba/Kiki)
//   - Open vowels (a, æ, ɛ, ɑ, ɐ): correlate with positivity/warmth (Ramachandran 2001)
//   - Nasals (m, n, ŋ, ɲ): correlate with softness/approachability
//   - Sibilants (s, z, ʃ, ʒ): correlate with negativity/sharpness (Magnus 2001)
func parseIPA(pron string) (cvRatio, openVowelFrac float64, hasNasals, hasSibilants bool) {
	consonants, vowels, openVowels := 0, 0, 0
	for _, r := range pron {
		switch r {
		// IPA vowels
		case 'a', 'e', 'i', 'o', 'u',
			'æ', 'ɛ', 'ɑ', 'ɐ', 'ə', 'ø', 'œ', 'y',
			'ʊ', 'ɪ', 'ɔ', 'ɯ', 'ʏ', 'ɶ', 'ɒ', 'ɜ',
			'ɘ', 'ɵ', 'ɤ', 'ʌ', 'ɞ':
			vowels++
			switch r {
			case 'a', 'æ', 'ɛ', 'ɑ', 'ɐ': // open vowels → positive/warm
				openVowels++
			}
		// Nasals → soft/approachable
		case 'm', 'n', 'ŋ', 'ɲ', 'ɱ', 'ɴ':
			consonants++
			hasNasals = true
		// Sibilants → sharp/negative
		case 's', 'z', 'ʃ', 'ʒ':
			consonants++
			hasSibilants = true
		// Other consonants (stops, fricatives, approximants)
		case 'p', 'b', 't', 'd', 'k', 'g', 'ʔ',
			'f', 'v', 'θ', 'ð', 'x', 'ɣ', 'χ', 'h', 'ħ', 'ʕ',
			'l', 'r', 'ɾ', 'ɹ', 'ɻ', 'j', 'w', 'ʋ', 'ɰ',
			'c', 'ɟ', 'ɡ', 'ʑ', 'ɕ', 'ʐ', 'ʂ':
			consonants++
		// Skip non-phonological: /, ˈ, ˌ, ., -, spaces, brackets
		}
	}
	total := consonants + vowels
	if total == 0 {
		return
	}
	cvRatio = float64(consonants) / float64(total)
	if vowels > 0 {
		openVowelFrac = float64(openVowels) / float64(vowels)
	}
	return
}

// langFamily maps a WordLang ID to a normalized language-family score.
// Romance=0.0, Germanic=0.2, Slavic=0.4, Semitic=0.6, CJK=0.8, Other=1.0
func langFamily(langID uint32) float64 {
	switch langID {
	case 0, 2, 3, 5: // PT, ES, IT, FR — Romance
		return 0.0
	case 1, 4, 6: // EN, DE, NL — Germanic
		return 0.2
	case 10, 19, 20: // RU, UK, PL — Slavic
		return 0.4
	case 7, 23: // AR, HE — Semitic
		return 0.6
	case 8, 9, 11: // ZH, JA, KO — CJK
		return 0.8
	default: // TG, HI, BN, ID, TR, FA, SW, SA, TA — Other
		return 1.0
	}
}

// ExtractFromLexicon looks up word+lang and extracts features.
// Returns a zero FeatureVec and false if the word is not found.
func ExtractFromLexicon(lex *lexdb.Lexicon, word, lang string) (FeatureVec, bool) {
	var w *lexdb.WordRecord
	if lang != "" {
		w = lex.LookupWordInLang(word, lang)
	}
	if w == nil {
		w = lex.LookupWord(word)
	}
	if w == nil {
		return FeatureVec{}, false
	}
	root := lex.LookupRoot(w.RootID)
	return Extract(lex, w, root, Context{}), true
}
