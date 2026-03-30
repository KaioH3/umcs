// Package fusedingest provides scientific data fusion for UMCS lexicon construction.
// Instead of simple deduplication, this module aggregates ALL evidence from ALL sources
// and uses statistical methods (median, mean, voting) to produce the best possible
// sentiment annotations.
//
// Key Features:
// - Accumulate all scores from all sources for each word
// - Use median aggregation (robust to outliers)
// - Handle NaN via median imputation
// - Track source agreement for confidence scoring
// - Use weighted voting based on source reliability
package fusedingest

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// Polarity score constants for aggregation
const (
	PolarityPositive  = 1.0
	PolarityNegative  = -1.0
	PolarityNeutral   = 0.0
	PolarityAmbiguous = 0.5 // when sources conflict

	IntensityNone     = 0.0
	IntensityWeak     = 0.25
	IntensityModerate = 0.5
	IntensityStrong   = 0.75
	IntensityExtreme  = 1.0
)

// Source reliability weights (higher = more reliable)
var sourceWeights = map[string]float64{
	"SentiLex":       0.95, // Expert annotated Portuguese
	"OpLexicon":      0.90, // Expert annotated Portuguese
	"NRC-Emotion":    0.85, // Crowd-sourced but validated
	"SentiWordNet":   0.80, // Algorithm + WordNet
	"AFINN":          0.75, // Crowd-sourced
	"VADER":          0.85, // Validated lexicon
	"MPQA":           0.80, // Academic lexicon
	"Warriner":       0.85, // Normed ratings
	"NRC-VAD":        0.80, // VAD ratings
	"Sentiment81":    0.60, // Simple binary lists
	"SimpleWordList": 0.50, // Simple lists
	"CogNet":         0.40, // Low confidence cognate mapping
	"EtymWn":         0.30, // Etymology only
	"UniMorph":       0.20, // Morphology only, no sentiment
	"Lexique383":     0.30, // Frequency data only
	"MorphoLex":      0.25, // Morphology only
}

// UnifiedEntry accumulates all evidence for a single word across all sources
type UnifiedEntry struct {
	Word string
	Lang string
	Norm string

	// Continuous scores (aggregated via median)
	ValenceScores   []float64 // VAD valence [-1, 1]
	ArousalScores   []float64 // VAD arousal [-1, 1]
	DominanceScores []float64 // VAD dominance [-1, 1]

	// Discrete polarity votes (for voting)
	polarityVotes map[string]float64 // source -> vote (-1, 0, 1)

	// Intensity votes
	intensityVotes map[string]float64 // source -> intensity (0-1)

	// POS votes
	posVotes map[string]string

	// IPA pronunciations
	ipas map[string]bool

	// Source tracking
	sources map[string]bool

	// Computed fields (set by Finalize)
	FusedPolarity  string
	FusedIntensity string
	FusedArousal   string
	FusedDominance string
	FusedPOS       string
	FusedIPA       string
	Confidence     float64
	IsFused        bool
}

// NewUnifiedEntry creates a new accumulator for a word
func NewUnifiedEntry(word, lang string) *UnifiedEntry {
	return &UnifiedEntry{
		Word:           word,
		Lang:           lang,
		Norm:           normalizeWord(word),
		polarityVotes:  make(map[string]float64),
		intensityVotes: make(map[string]float64),
		posVotes:       make(map[string]string),
		ipas:           make(map[string]bool),
		sources:        make(map[string]bool),
	}
}

// AddPolarity adds a polarity vote from a source
// polarity: "POSITIVE", "NEGATIVE", "NEUTRAL", "AMBIGUOUS"
func (ue *UnifiedEntry) AddPolarity(source, polarity string, weight float64) {
	if weight <= 0 {
		weight = sourceWeights[source]
		if weight <= 0 {
			weight = 0.5
		}
	}

	var score float64
	switch polarity {
	case "POSITIVE":
		score = PolarityPositive
	case "NEGATIVE":
		score = PolarityNegative
	case "NEUTRAL":
		score = PolarityNeutral
	case "AMBIGUOUS", "MIXED":
		score = PolarityAmbiguous
	default:
		return
	}

	ue.polarityVotes[source] = score * weight
	ue.sources[source] = true
}

// AddValence adds a valence score [-1, 1]
func (ue *UnifiedEntry) AddValence(source string, valence float64) {
	if valence < -1 || valence > 1 {
		return
	}
	ue.ValenceScores = append(ue.ValenceScores, valence)
	ue.sources[source] = true
}

// AddArousal adds an arousal score [-1, 1]
func (ue *UnifiedEntry) AddArousal(source string, arousal float64) {
	if arousal < -1 || arousal > 1 {
		return
	}
	ue.ArousalScores = append(ue.ArousalScores, arousal)
}

// AddDominance adds a dominance score [-1, 1]
func (ue *UnifiedEntry) AddDominance(source string, dominance float64) {
	if dominance < -1 || dominance > 1 {
		return
	}
	ue.DominanceScores = append(ue.DominanceScores, dominance)
}

// AddIntensity adds an intensity vote [0, 1]
func (ue *UnifiedEntry) AddIntensity(source, intensity string) {
	var score float64
	switch intensity {
	case "EXTREME":
		score = IntensityExtreme
	case "STRONG":
		score = IntensityStrong
	case "MODERATE":
		score = IntensityModerate
	case "WEAK":
		score = IntensityWeak
	default:
		score = IntensityNone
	}
	ue.intensityVotes[source] = score
}

// AddPOS adds a POS vote
func (ue *UnifiedEntry) AddPOS(source, pos string) {
	if pos == "" {
		return
	}
	ue.posVotes[source] = pos
}

// AddIPA adds an IPA pronunciation
func (ue *UnifiedEntry) AddIPA(ipa string) {
	if ipa != "" {
		ue.ipas[ipa] = true
	}
}

// Finalize computes the fused polarity, intensity, and other fields
// using median aggregation (robust to outliers)
func (ue *UnifiedEntry) Finalize() {
	ue.computePolarity()
	ue.computeIntensity()
	ue.computeVAD()
	ue.computePOS()
	ue.computeIPA()
	ue.computeConfidence()
	ue.IsFused = true
}

// computePolarity uses weighted median voting
func (ue *UnifiedEntry) computePolarity() {
	if len(ue.polarityVotes) == 0 {
		// Fall back to valence scores if available
		if len(ue.ValenceScores) > 0 {
			median := medianFloat64(ue.ValenceScores)
			if median > 0.1 {
				ue.FusedPolarity = "POSITIVE"
			} else if median < -0.1 {
				ue.FusedPolarity = "NEGATIVE"
			} else {
				ue.FusedPolarity = "NEUTRAL"
			}
		} else {
			ue.FusedPolarity = "NEUTRAL"
		}
		return
	}

	// Weighted voting
	var weightedSum, weightSum float64
	for _, vote := range ue.polarityVotes {
		// Extract weight from sign (vote already includes weight)
		absVote := math.Abs(vote)
		if absVote > 0 {
			weight := absVote
			weightedSum += vote
			weightSum += weight
		}
	}

	if weightSum == 0 {
		ue.FusedPolarity = "NEUTRAL"
		return
	}

	normalizedScore := weightedSum / weightSum

	// Check for conflict (sources strongly disagree)
	conflictScore := 0.0
	for _, vote := range ue.polarityVotes {
		// Count votes that disagree with the majority
		if math.Abs(vote) > 0.3 && math.Abs(vote-normalizedScore) > 0.5 {
			conflictScore++
		}
	}

	if conflictScore > 2 && len(ue.polarityVotes) > 3 {
		ue.FusedPolarity = "AMBIGUOUS"
		return
	}

	if normalizedScore > 0.1 {
		ue.FusedPolarity = "POSITIVE"
	} else if normalizedScore < -0.1 {
		ue.FusedPolarity = "NEGATIVE"
	} else {
		ue.FusedPolarity = "NEUTRAL"
	}
}

// computeIntensity uses median of intensity votes
func (ue *UnifiedEntry) computeIntensity() {
	if len(ue.intensityVotes) == 0 {
		// Infer from valence magnitude
		if len(ue.ValenceScores) > 0 {
			median := medianFloat64(ue.ValenceScores)
			absMedian := math.Abs(median)
			if absMedian > 0.6 {
				ue.FusedIntensity = "STRONG"
			} else if absMedian > 0.3 {
				ue.FusedIntensity = "MODERATE"
			} else if absMedian > 0.1 {
				ue.FusedIntensity = "WEAK"
			} else {
				ue.FusedIntensity = "NONE"
			}
		} else {
			ue.FusedIntensity = "MODERATE" // Default
		}
		return
	}

	var scores []float64
	for _, score := range ue.intensityVotes {
		scores = append(scores, score)
	}

	median := medianFloat64(scores)

	switch {
	case median >= IntensityExtreme:
		ue.FusedIntensity = "EXTREME"
	case median >= IntensityStrong:
		ue.FusedIntensity = "STRONG"
	case median >= IntensityModerate:
		ue.FusedIntensity = "MODERATE"
	case median >= IntensityWeak:
		ue.FusedIntensity = "WEAK"
	default:
		ue.FusedIntensity = "NONE"
	}
}

// computeVAD uses median aggregation (robust to outliers)
func (ue *UnifiedEntry) computeVAD() {
	if len(ue.ValenceScores) > 0 {
		median := medianFloat64(ue.ValenceScores)
		ue.FusedArousal = vadToLevel(median, -1.0, 1.0)
	}

	if len(ue.ArousalScores) > 0 {
		median := medianFloat64(ue.ArousalScores)
		ue.FusedArousal = vadToLevel(median, -1.0, 1.0)
	}

	if len(ue.DominanceScores) > 0 {
		median := medianFloat64(ue.DominanceScores)
		ue.FusedDominance = vadToLevel(median, -1.0, 1.0)
	}
}

// computePOS uses majority voting
func (ue *UnifiedEntry) computePOS() {
	if len(ue.posVotes) == 0 {
		return
	}

	posCounts := make(map[string]int)
	for _, pos := range ue.posVotes {
		posCounts[pos]++
	}

	var maxCount int
	var bestPOS string
	for pos, count := range posCounts {
		if count > maxCount {
			maxCount = count
			bestPOS = pos
		}
	}

	ue.FusedPOS = bestPOS
}

// computeIPA selects the most common IPA
func (ue *UnifiedEntry) computeIPA() {
	if len(ue.ipas) == 0 {
		return
	}

	// Just take the first one for now (could improve with frequency data)
	for ipa := range ue.ipas {
		ue.FusedIPA = ipa
		break
	}
}

// computeConfidence based on source agreement
func (ue *UnifiedEntry) computeConfidence() {
	numSources := len(ue.sources)
	if numSources == 0 {
		ue.Confidence = 0.0
		return
	}

	// Base confidence on number of sources
	baseConf := math.Min(float64(numSources)/5.0, 1.0)

	// Reduce confidence if polarity votes conflict
	if len(ue.polarityVotes) >= 3 {
		var posVotes, negVotes, neutralVotes int
		for _, vote := range ue.polarityVotes {
			if vote > 0.3 {
				posVotes++
			} else if vote < -0.3 {
				negVotes++
			} else {
				neutralVotes++
			}
		}

		// High conflict reduces confidence
		if (posVotes > 0 && negVotes > 0) || (posVotes > 0 && neutralVotes > 0 && negVotes > 0) {
			baseConf *= 0.5
		} else if posVotes > 1 && negVotes > 0 || negVotes > 1 && posVotes > 0 {
			baseConf *= 0.7
		}
	}

	ue.Confidence = math.Round(baseConf*100) / 100
}

// Sources returns the list of sources that contributed to this entry
func (ue *UnifiedEntry) Sources() []string {
	var sources []string
	for s := range ue.sources {
		sources = append(sources, s)
	}
	return sources
}

// Entry converts the unified entry to a standard Entry for CSV output
func (ue *UnifiedEntry) Entry(source string) Entry {
	return Entry{
		Word:         ue.Word,
		Lang:         ue.Lang,
		Norm:         ue.Norm,
		Polarity:     ue.FusedPolarity,
		Intensity:    ue.FusedIntensity,
		Arousal:      ue.FusedArousal,
		Dominance:    ue.FusedDominance,
		POS:          ue.FusedPOS,
		IPA:          ue.FusedIPA,
		Concreteness: "",
		Source:       fmt.Sprintf("FUSED[%s]", strings.Join(ue.Sources(), ",")),
	}
}

// Entry represents a word with sentiment annotations for CSV output
type Entry struct {
	Word         string
	Lang         string
	Norm         string
	Polarity     string
	Intensity    string
	Arousal      string
	Dominance    string
	AoA          string
	Concreteness string
	POS          string
	IPA          string
	Syllables    int
	FreqRank     int
	Source       string
}

// Result holds statistics from a fused import operation
type Result struct {
	TotalEntries int
	UniqueWords  int
	ByLanguage   map[string]int
	ByPolarity   map[string]int
	BySource     map[string]int
	HighConf     int // confidence >= 0.8
	LowConf      int // confidence < 0.5
}

// medianFloat64 computes the median of a float64 slice
// Uses sort and handles empty slice
func medianFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Filter out NaN and Inf
	var clean []float64
	for _, v := range values {
		if !math.IsNaN(v) && !math.IsInf(v, 0) {
			clean = append(clean, v)
		}
	}

	if len(clean) == 0 {
		return 0
	}

	sort.Float64s(clean)

	n := len(clean)
	if n%2 == 0 {
		return (clean[n/2-1] + clean[n/2]) / 2
	}
	return clean[n/2]
}

// vadToLevel converts a VAD float [-1,1] to LOW/MED/HIGH
func vadToLevel(v, lo, hi float64) string {
	if v < lo+0.33*(hi-lo) {
		return "LOW"
	}
	if v < lo+0.67*(hi-lo) {
		return "MED"
	}
	return "HIGH"
}

// normalizeWord lowercases and removes diacritics for matching
func normalizeWord(s string) string {
	var b strings.Builder
	for _, r := range s {
		// Strip diacritics
		r = unicode.ToLower(r)
		switch r {
		case 'á', 'à', 'ã', 'â', 'ä', 'å', 'ā', 'ą', 'ă':
			r = 'a'
		case 'é', 'è', 'ê', 'ë', 'ē', 'ę', 'ė', 'ě':
			r = 'e'
		case 'í', 'ì', 'î', 'ï', 'ī', 'į', 'ı':
			r = 'i'
		case 'ó', 'ò', 'õ', 'ô', 'ö', 'ø', 'ō', 'ő', 'ǫ':
			r = 'o'
		case 'ú', 'ù', 'û', 'ü', 'ū', 'ů', 'ű', 'ų':
			r = 'u'
		case 'ý', 'ỳ', 'ŷ', 'ÿ':
			r = 'y'
		case 'ñ', 'ń':
			r = 'n'
		case 'ç', 'ć', 'č':
			r = 'c'
		case 'ş', 'ś', 'š':
			r = 's'
		case 'ž', 'ź', 'ż':
			r = 'z'
		}
		b.WriteRune(r)
	}
	return b.String()
}

// isAlpha returns true if the word contains at least one letter
func isAlpha(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

// FusedImporter manages the unified import process
type FusedImporter struct {
	entries  map[string]*UnifiedEntry // key: lang|norm
	langMap  map[string]string
	ipaLangs map[string]string
	res      Result
}

// NewFusedImporter creates a new fused importer
func NewFusedImporter() *FusedImporter {
	return &FusedImporter{
		entries: make(map[string]*UnifiedEntry),
		langMap: loadLangMap(),
		ipaLangs: map[string]string{
			"pt": "PT", "en": "EN", "es": "ES", "fr": "FR", "de": "DE",
			"it": "IT", "nl": "NL", "ru": "RU", "zh": "ZH", "ja": "JA",
			"ko": "KO", "ar": "AR", "hi": "HI",
		},
		res: Result{
			ByLanguage: make(map[string]int),
			ByPolarity: make(map[string]int),
			BySource:   make(map[string]int),
		},
	}
}

// getOrCreateEntry gets or creates a unified entry for a word
func (fi *FusedImporter) getOrCreateEntry(word, lang, norm string) *UnifiedEntry {
	key := lang + "|" + norm
	if entry, ok := fi.entries[key]; ok {
		return entry
	}

	entry := NewUnifiedEntry(word, lang)
	entry.Norm = norm
	fi.entries[key] = entry
	return entry
}

// ImportSentiment81Langs imports all word lists from sentiment-81langs dataset
func (fi *FusedImporter) ImportSentiment81Langs(dir string) error {
	files, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	langMap := map[string]string{
		"pt": "PT", "br": "PT", "es": "ES", "en": "EN", "fr": "FR",
		"de": "DE", "it": "IT", "nl": "NL", "ru": "RU", "zh": "ZH",
		"ja": "JA", "ko": "KO", "ar": "AR", "hi": "HI", "tr": "TR",
		"pl": "PL", "sv": "SV", "da": "DA", "fi": "FI", "no": "NO",
		"cs": "CS", "hu": "HU", "ro": "RO", "uk": "UK", "bg": "BG",
		"el": "EL", "he": "HE", "th": "TH", "vi": "VI", "id": "ID",
		"ms": "MS", "ta": "TA", "bn": "BN", "ur": "UR", "fa": "FA",
	}

	for _, f := range files {
		if f.IsDir() {
			continue
		}
		name := f.Name()
		if name == "correctedMetadata.csv" || strings.HasPrefix(name, ".") {
			continue
		}

		parts := strings.Split(name, "_")
		if len(parts) != 3 || parts[2] != "words.txt" {
			continue
		}

		pol := parts[0]
		langCode := parts[1]
		lang, ok := langMap[langCode]
		if !ok {
			continue
		}

		polarity := "NEUTRAL"
		if pol == "positive" {
			polarity = "POSITIVE"
		} else if pol == "negative" {
			polarity = "NEGATIVE"
		}

		source := fmt.Sprintf("Sentiment81-%s", strings.ToUpper(langCode))
		filePath := filepath.Join(dir, name)

		if err := fi.importSimpleWordList(filePath, lang, polarity, source); err != nil {
			continue
		}
	}

	return nil
}

// ImportAFINN imports AFINN-165 lexicon with numerical scores [-5, +5]
func (fi *FusedImporter) ImportAFINN(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			continue
		}

		word := strings.TrimSpace(parts[0])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		score, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err != nil {
			continue
		}

		norm := normalizeWord(word)
		entry := fi.getOrCreateEntry(word, "EN", norm)

		// Add valence score (normalized to [-1, 1])
		entry.AddValence("AFINN", score/5.0)

		// Add polarity
		polarity := "NEUTRAL"
		intensity := "MODERATE"
		if score > 2 {
			polarity = "POSITIVE"
			intensity = "STRONG"
		} else if score > 0 {
			polarity = "POSITIVE"
			intensity = "WEAK"
		} else if score < -2 {
			polarity = "NEGATIVE"
			intensity = "STRONG"
		} else if score < 0 {
			polarity = "NEGATIVE"
			intensity = "WEAK"
		}
		entry.AddPolarity("AFINN", polarity, sourceWeights["AFINN"])
		entry.AddIntensity("AFINN", intensity)
		fi.res.BySource["AFINN"]++
	}

	return scanner.Err()
}

// ImportSentiWordNet imports SentiWordNet 3.0
func (fi *FusedImporter) ImportSentiWordNet(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	posMap := map[string]string{
		"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV", "s": "ADJ",
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) < 6 {
			continue
		}

		posTag := parts[0]
		pos, ok := posMap[posTag]
		if !ok {
			continue
		}

		posScore, err1 := strconv.ParseFloat(parts[2], 64)
		negScore, err2 := strconv.ParseFloat(parts[3], 64)
		if err1 != nil || err2 != nil {
			continue
		}

		// Extract terms
		terms := strings.Fields(parts[4])
		for _, term := range terms {
			wp := strings.Split(term, "#")
			if len(wp) < 1 {
				continue
			}
			word := wp[0]
			if !isAlpha(word) || strings.Contains(word, "_") {
				continue
			}

			norm := normalizeWord(word)
			entry := fi.getOrCreateEntry(word, "EN", norm)

			// Valence from scores
			valence := posScore - negScore
			entry.AddValence("SentiWordNet", valence)
			entry.AddPOS("SentiWordNet", pos)

			// Polarity
			polarity := "NEUTRAL"
			intensity := "MODERATE"
			netScore := posScore - negScore
			if netScore > 0.5 {
				polarity = "POSITIVE"
				intensity = "STRONG"
			} else if netScore > 0.1 {
				polarity = "POSITIVE"
				intensity = "MODERATE"
			} else if netScore > 0 {
				polarity = "POSITIVE"
				intensity = "WEAK"
			} else if netScore < -0.5 {
				polarity = "NEGATIVE"
				intensity = "STRONG"
			} else if netScore < -0.1 {
				polarity = "NEGATIVE"
				intensity = "MODERATE"
			} else if netScore < 0 {
				polarity = "NEGATIVE"
				intensity = "WEAK"
			}

			entry.AddPolarity("SentiWordNet", polarity, sourceWeights["SentiWordNet"])
			entry.AddIntensity("SentiWordNet", intensity)
			fi.res.BySource["SentiWordNet"]++
		}
	}

	return scanner.Err()
}

// ImportSimpleWordList imports a one-word-per-line file
func (fi *FusedImporter) importSimpleWordList(path, lang, polarity, source string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	for scanner.Scan() {
		word := strings.TrimSpace(scanner.Text())
		if word == "" || strings.Contains(word, " ") || !isAlpha(word) {
			continue
		}

		norm := normalizeWord(word)
		entry := fi.getOrCreateEntry(word, lang, norm)
		entry.AddPolarity(source, polarity, sourceWeights[source])
		entry.AddIntensity(source, "MODERATE")
		fi.res.BySource[source]++
	}

	return scanner.Err()
}

// ImportPositiveNegative imports positive-words.txt and negative-words.txt
func (fi *FusedImporter) ImportPositiveNegative(posPath, negPath string) error {
	if err := fi.importSimpleWordList(posPath, "EN", "POSITIVE", "SimpleWordList"); err != nil {
		return err
	}
	return fi.importSimpleWordList(negPath, "EN", "NEGATIVE", "SimpleWordList")
}

// ImportNRCVAD imports NRC VAD Lexicon v2.1
func (fi *FusedImporter) ImportNRCVAD(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	if scanner.Scan() {
		// Skip header
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 4 {
			continue
		}

		term := strings.TrimSpace(parts[0])
		if strings.Contains(term, " ") || !isAlpha(term) {
			continue
		}

		valence, err1 := strconv.ParseFloat(parts[1], 64)
		arousal, err2 := strconv.ParseFloat(parts[2], 64)
		dominance, err3 := strconv.ParseFloat(parts[3], 64)
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}

		norm := normalizeWord(term)
		entry := fi.getOrCreateEntry(term, "EN", norm)

		entry.AddValence("NRC-VAD", valence)
		entry.AddArousal("NRC-VAD", arousal)
		entry.AddDominance("NRC-VAD", dominance)

		polarity := "NEUTRAL"
		if valence > 0.1 {
			polarity = "POSITIVE"
		} else if valence < -0.1 {
			polarity = "NEGATIVE"
		}
		entry.AddPolarity("NRC-VAD", polarity, sourceWeights["NRC-VAD"])
		fi.res.BySource["NRC-VAD"]++
	}

	return scanner.Err()
}

// ImportMPQA imports MPQA Subjectivity Lexicon
func (fi *FusedImporter) ImportMPQA(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
			continue
		}

		// MPQA format: type=wordlist word=xxx polarity=xxx
		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}

		var word, polarity string
		for _, p := range parts {
			if strings.HasPrefix(p, "word=") {
				word = strings.TrimPrefix(p, "word=")
			} else if strings.HasPrefix(p, "polarity=") {
				polarity = strings.TrimPrefix(p, "polarity=")
			}
		}

		if word == "" || !isAlpha(word) {
			continue
		}

		norm := normalizeWord(word)
		entry := fi.getOrCreateEntry(word, "EN", norm)

		pol := "NEUTRAL"
		switch polarity {
		case "positive":
			pol = "POSITIVE"
		case "negative":
			pol = "NEGATIVE"
		case "both":
			pol = "AMBIGUOUS"
		}

		entry.AddPolarity("MPQA", pol, sourceWeights["MPQA"])
		entry.AddIntensity("MPQA", "MODERATE")
		fi.res.BySource["MPQA"]++
	}

	return scanner.Err()
}

// ImportWarrinerVAD imports Warriner VAD norms
func (fi *FusedImporter) ImportWarrinerVAD(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Warriner format: Word, V.Mean.Sum, A.Mean.Sum, D.Mean.Sum
	scanner := bufio.NewScanner(f)
	if scanner.Scan() {
		// Skip header
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ",")
		if len(parts) < 4 {
			continue
		}

		word := strings.TrimSpace(parts[0])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		valence, err1 := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		arousal, err2 := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
		dominance, err3 := strconv.ParseFloat(strings.TrimSpace(parts[3]), 64)
		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}

		// Normalize to [-1, 1] (Warriner is [1, 9])
		valence = (valence - 5) / 4
		arousal = (arousal - 5) / 4
		dominance = (dominance - 5) / 4

		norm := normalizeWord(word)
		entry := fi.getOrCreateEntry(word, "EN", norm)

		entry.AddValence("Warriner", valence)
		entry.AddArousal("Warriner", arousal)
		entry.AddDominance("Warriner", dominance)

		pol := "NEUTRAL"
		if valence > 0.1 {
			pol = "POSITIVE"
		} else if valence < -0.1 {
			pol = "NEGATIVE"
		}
		entry.AddPolarity("Warriner", pol, sourceWeights["Warriner"])
		fi.res.BySource["Warriner"]++
	}

	return scanner.Err()
}

// ImportOpLexicon imports OpLexicon v3.0 (Portuguese)
func (fi *FusedImporter) ImportOpLexicon(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
			continue
		}

		// OpLexicon format: word | polarity | intensity | POS | ...
		parts := strings.Split(line, "|")
		if len(parts) < 4 {
			continue
		}

		word := strings.TrimSpace(parts[0])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		polarity := strings.TrimSpace(parts[1])
		intensity := strings.TrimSpace(parts[2])

		norm := normalizeWord(word)
		entry := fi.getOrCreateEntry(word, "PT", norm)

		pol := "NEUTRAL"
		switch polarity {
		case "positive", "POS":
			pol = "POSITIVE"
		case "negative", "NEG":
			pol = "NEGATIVE"
		}

		entry.AddPolarity("OpLexicon", pol, sourceWeights["OpLexicon"])

		intens := "MODERATE"
		switch intensity {
		case "strong", "STR":
			intens = "STRONG"
		case "weak", "WEAK":
			intens = "WEAK"
		}
		entry.AddIntensity("OpLexicon", intens)
		fi.res.BySource["OpLexicon"]++
	}

	return scanner.Err()
}

// ImportSentiLex imports SentiLex-PT02 (Portuguese)
func (fi *FusedImporter) ImportSentiLex(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	posMap := map[string]string{
		"Adj": "ADJ", "V": "VERB", "N": "NOUN", "ADV": "ADV",
		"Adv": "ADV", "IDIOM": "",
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		dotIdx := strings.Index(line, ".PoS=")
		if dotIdx < 0 {
			continue
		}

		word := line[:dotIdx]
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		meta := line[dotIdx+5:]
		semiIdx := strings.Index(meta, ";")
		posTag := meta
		if semiIdx >= 0 {
			posTag = meta[:semiIdx]
		}

		polVal := 0
		if polIdx := strings.Index(line, "POL:N0="); polIdx >= 0 {
			valPart := line[polIdx+7:]
			if semi := strings.Index(valPart, ";"); semi >= 0 {
				valPart = valPart[:semi]
			}
			if v, err := strconv.Atoi(valPart); err == nil {
				polVal = v
			}
		}

		norm := normalizeWord(word)
		entry := fi.getOrCreateEntry(word, "PT", norm)

		pol := "NEUTRAL"
		if polVal > 0 {
			pol = "POSITIVE"
		} else if polVal < 0 {
			pol = "NEGATIVE"
		}
		entry.AddPolarity("SentiLex", pol, sourceWeights["SentiLex"])

		if umcsPOS := posMap[posTag]; umcsPOS != "" {
			entry.AddPOS("SentiLex", umcsPOS)
		}
		entry.AddIntensity("SentiLex", "MODERATE")
		fi.res.BySource["SentiLex"]++
	}

	return scanner.Err()
}

// ImportCogNet imports CogNet v2.0 (multilingual cognate data with sentiment)
func (fi *FusedImporter) ImportCogNet(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// CogNet format: lang:word  rel:type  target_lang:target_word
		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}

		// Parse source
		src := parts[0]
		colonIdx := strings.Index(src, ":")
		if colonIdx < 0 {
			continue
		}
		srcLang := src[:colonIdx]
		srcWord := src[colonIdx+1:]

		lang, ok := fi.langMap[srcLang]
		if !ok {
			continue
		}

		if !isAlpha(srcWord) || strings.Contains(srcWord, "_") {
			continue
		}

		norm := normalizeWord(srcWord)
		entry := fi.getOrCreateEntry(srcWord, lang, norm)
		entry.AddPolarity("CogNet", "NEUTRAL", sourceWeights["CogNet"])
		fi.res.BySource["CogNet"]++
	}

	return scanner.Err()
}

// ImportNRCEmotion imports NRC Emotion Lexicon
func (fi *FusedImporter) ImportNRCEmotion(dir string) error {
	wordLevelPath := filepath.Join(dir, "NRC-Emotion-Lexicon", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
	f, err := os.Open(wordLevelPath)
	if err != nil {
		// Try alternate location
		altPath := filepath.Join(dir, "NRC-Emotion-Lexicon", "OneFilePerLanguage", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
		f, err = os.Open(altPath)
		if err != nil {
			return err
		}
	}
	defer f.Close()

	emotionPolarity := map[string]string{
		"anger": "NEGATIVE", "anticipation": "POSITIVE", "disgust": "NEGATIVE",
		"fear": "NEGATIVE", "joy": "POSITIVE", "sadness": "NEGATIVE",
		"surprise": "NEUTRAL", "trust": "POSITIVE",
	}

	scanner := bufio.NewScanner(f)
	if !scanner.Scan() {
		return fmt.Errorf("empty NRC Emotion file")
	}

	header := scanner.Text()
	headers := strings.Split(header, "\t")
	emotionIdx := make(map[string]int)
	for i, h := range headers {
		h = strings.TrimSpace(h)
		if _, ok := emotionPolarity[h]; ok {
			emotionIdx[h] = i
		}
	}

	wordEmotions := make(map[string]map[string]bool)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			continue
		}

		word := strings.TrimSpace(parts[0])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		if _, ok := wordEmotions[word]; !ok {
			wordEmotions[word] = make(map[string]bool)
		}

		for emotion, idx := range emotionIdx {
			if idx < len(parts) && strings.TrimSpace(parts[idx]) == "1" {
				pol := emotionPolarity[emotion]
				if pol != "NEUTRAL" {
					wordEmotions[word][pol] = true
				}
			}
		}
	}

	for word, polMap := range wordEmotions {
		norm := normalizeWord(word)
		entry := fi.getOrCreateEntry(word, "EN", norm)

		posCount := 0
		negCount := 0
		for p := range polMap {
			if p == "POSITIVE" {
				posCount++
			} else if p == "NEGATIVE" {
				negCount++
			}
		}

		pol := "NEUTRAL"
		if posCount > negCount {
			pol = "POSITIVE"
		} else if negCount > posCount {
			pol = "NEGATIVE"
		}

		entry.AddPolarity("NRC-Emotion", pol, sourceWeights["NRC-Emotion"])
		entry.AddIntensity("NRC-Emotion", "MODERATE")
		fi.res.BySource["NRC-Emotion"]++
	}

	return scanner.Err()
}

// FinalizeAndExport finalizes all entries and exports to CSV
func (fi *FusedImporter) FinalizeAndExport(csvPath string, nextWordID int) error {
	// Finalize all entries
	for _, entry := range fi.entries {
		entry.Finalize()
		fi.res.TotalEntries++
		fi.res.ByLanguage[entry.Lang]++

		pol := entry.FusedPolarity
		if pol == "" {
			pol = "NEUTRAL"
		}
		fi.res.ByPolarity[pol]++

		if entry.Confidence >= 0.8 {
			fi.res.HighConf++
		} else if entry.Confidence < 0.5 {
			fi.res.LowConf++
		}
	}

	// Write to CSV
	file, err := os.OpenFile(csvPath, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	w := csv.NewWriter(file)
	defer w.Flush()

	// Check if file is empty to write header
	info, err := file.Stat()
	if err != nil || info.Size() == 0 {
		w.Write([]string{
			"word_id", "root_id", "variant", "word", "lang", "norm",
			"polarity", "intensity", "semantic_role", "domain", "freq_rank",
			"flags", "pos", "arousal", "dominance", "aoa", "concreteness",
			"register", "ontological", "polysemy", "pron", "syllables",
			"stress", "valency", "irony_capable", "neologism", "source",
		})
	}

	wordID := nextWordID
	for _, entry := range fi.entries {
		csvEntry := entry.Entry("FUSED")
		w.Write([]string{
			strconv.Itoa(wordID),
			"0", // root_id = 0 (will be assigned synthetic root)
			"1", // variant
			csvEntry.Word,
			csvEntry.Lang,
			csvEntry.Norm,
			csvEntry.Polarity,
			csvEntry.Intensity,
			"",  // semantic_role
			"",  // domain
			"0", // freq_rank
			"0", // flags
			csvEntry.POS,
			csvEntry.Arousal,
			csvEntry.Dominance,
			"",  // aoa
			"",  // concreteness
			"",  // register
			"",  // ontological
			"0", // polysemy
			csvEntry.IPA,
			"0", // syllables
			"",  // stress
			"",  // valency
			"0", // irony_capable
			"0", // neologism
			csvEntry.Source,
		})
		wordID++
		fi.res.UniqueWords++
	}

	return nil
}

// Result returns the import statistics
func (fi *FusedImporter) Result() Result {
	return fi.res
}

// loadLangMap loads the language code mapping
func loadLangMap() map[string]string {
	return map[string]string{
		"en": "EN", "pt": "PT", "br": "PT", "es": "ES", "fr": "FR",
		"de": "DE", "it": "IT", "nl": "NL", "ru": "RU", "zh": "ZH",
		"ja": "JA", "ko": "KO", "ar": "AR", "hi": "HI", "tr": "TR",
		"pl": "PL", "sv": "SV", "da": "DA", "fi": "FI", "no": "NO",
		"cs": "CS", "hu": "HU", "ro": "RO", "uk": "UK", "bg": "BG",
		"el": "EL", "he": "HE", "th": "TH", "vi": "VI", "id": "ID",
		"ms": "MS", "ta": "TA", "bn": "BN", "ur": "UR", "fa": "FA",
		"ca": "CA", "eu": "EU", "gl": "GL", "cy": "CY", "ga": "GA",
		"mt": "MT", "sk": "SK", "sl": "SI", "hr": "HR", "lt": "LT",
		"lv": "LV", "et": "ET", "mk": "MK", "sq": "SQ", "is": "IS",
		"ka": "KA", "hy": "HY", "ne": "NE", "mr": "MR", "te": "TE",
		"kn": "KN", "pa": "PA", "gu": "GU", "or": "OR", "ml": "ML",
		"tl": "TL", "af": "AF", "sw": "SW", "yo": "YO", "zu": "ZU",
		"ha": "HA", "ig": "IG", "am": "AM", "ti": "TG",
	}
}
