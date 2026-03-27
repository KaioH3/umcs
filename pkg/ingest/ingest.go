// Package ingest provides importers for external NLP datasets into UMCS
// CSV format (roots.csv / words.csv). Each importer reads a specific dataset
// format and produces normalized entries compatible with the UMCS lexicon.
//
// Supported datasets:
//   - NRC VAD Lexicon v2.1 (54k terms, valence/arousal/dominance)
//   - NRC Emotion Lexicon (14k words × 100+ languages, 8 emotions + polarity)
//   - AFINN-165 (3.3k words, integer valence [-5,+5])
//   - VADER (7.5k entries, mean sentiment [-4,+4])
//   - Bing Liu Opinion Lexicon (6.8k words, binary polarity)
//   - Warriner VAD norms (13.9k words, VAD means + SDs)
//   - IPA-dict (multiple languages, word → IPA pronunciation)
//   - SentiWordNet 3.0 (117k synsets with PosScore/NegScore)
//   - OpLexicon v3.0 (32k PT-BR words, POS + polarity [-1,0,1])
//   - SentiLex-PT02 (7k PT lemmas, fine-grained polarity + POS)
//   - Lexique383 (143k French words, frequency + phonology + POS)
//   - MPQA Subjectivity Lexicon (6.9k EN words, polarity [-1,0,1])
//   - Empath (194 semantic categories with keyword lists)
//   - 81-language sentiment (83 languages, binary polarity)
package ingest

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"unicode"
)

// Entry represents a word with sentiment annotations ready for CSV output.
type Entry struct {
	Word        string
	Lang        string
	Norm        string
	Polarity    string // POSITIVE, NEGATIVE, NEUTRAL, AMBIGUOUS
	Intensity   string // NONE, WEAK, MODERATE, STRONG, EXTREME
	Arousal     string // LOW, MED, HIGH or empty
	Dominance   string // LOW, MED, HIGH or empty
	AoA         string // EARLY, MID, LATE, TECHNICAL or empty
	Concreteness string // 1 (concrete) or empty
	POS         string
	IPA         string
	Syllables   int
	FreqRank    int
	Source      string // dataset that produced this entry
}

// Result holds statistics from an import operation.
type Result struct {
	Total     int // total entries processed
	New       int // entries not already in lexicon
	Skipped   int // entries already present
	Errors    int // parse errors
	ByPolarity map[string]int
}

// normalize lowercases and trims a word for matching.
func normalize(s string) string {
	s = strings.TrimSpace(s)
	var b strings.Builder
	for _, r := range s {
		b.WriteRune(unicode.ToLower(r))
	}
	return b.String()
}

// isAlpha returns true if the word contains at least one letter.
func isAlpha(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

// valenceToPolarity converts a continuous valence score to discrete polarity.
// Thresholds: > posThresh = POSITIVE, < negThresh = NEGATIVE, else NEUTRAL.
func valenceToPolarity(v, negThresh, posThresh float64) string {
	if v > posThresh {
		return "POSITIVE"
	}
	if v < negThresh {
		return "NEGATIVE"
	}
	return "NEUTRAL"
}

// valenceToIntensity converts absolute valence to intensity level.
func valenceToIntensity(absV float64, scale float64) string {
	norm := absV / scale
	switch {
	case norm >= 0.8:
		return "EXTREME"
	case norm >= 0.6:
		return "STRONG"
	case norm >= 0.4:
		return "MODERATE"
	case norm >= 0.2:
		return "WEAK"
	default:
		return "NONE"
	}
}

// vadToLevel converts a VAD float [-1,1] or [1,9] to LOW/MED/HIGH.
func vadToLevel(v float64, lo, hi float64) string {
	third := (hi - lo) / 3.0
	if v < lo+third {
		return "LOW"
	}
	if v < lo+2*third {
		return "MED"
	}
	return "HIGH"
}

// ImportNRCVAD reads the NRC VAD Lexicon v2.1 (TSV: term, valence, arousal, dominance).
// Valence/arousal/dominance are floats in [-1, 1] (centered around 0).
// Returns entries for English words only.
func ImportNRCVAD(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	scanner := bufio.NewScanner(f)

	// Skip header
	if scanner.Scan() {
		// header line
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 4 {
			res.Errors++
			continue
		}

		term := strings.TrimSpace(parts[0])
		// Skip multi-word terms (phrases) — only single words
		if strings.Contains(term, " ") {
			continue
		}
		if !isAlpha(term) {
			continue
		}

		valence, err1 := strconv.ParseFloat(parts[1], 64)
		arousal, err2 := strconv.ParseFloat(parts[2], 64)
		dominance, err3 := strconv.ParseFloat(parts[3], 64)
		if err1 != nil || err2 != nil || err3 != nil {
			res.Errors++
			continue
		}

		polarity := valenceToPolarity(valence, -0.1, 0.1)
		intensity := valenceToIntensity(math.Abs(valence), 1.0)
		arousalLvl := vadToLevel(arousal, -1.0, 1.0)
		dominanceLvl := vadToLevel(dominance, -1.0, 1.0)

		e := Entry{
			Word:      term,
			Lang:      "EN",
			Norm:      normalize(term),
			Polarity:  polarity,
			Intensity: intensity,
			Arousal:   arousalLvl,
			Dominance: dominanceLvl,
			Source:    "NRC-VAD",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity[polarity]++
	}

	return entries, res, scanner.Err()
}

// ImportAFINN reads AFINN-165 (TSV: word, valence [-5,+5]).
func ImportAFINN(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			res.Errors++
			continue
		}

		word := strings.TrimSpace(parts[0])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		val, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			res.Errors++
			continue
		}

		polarity := valenceToPolarity(val, -0.5, 0.5)
		intensity := valenceToIntensity(math.Abs(val), 5.0)

		e := Entry{
			Word:      word,
			Lang:      "EN",
			Norm:      normalize(word),
			Polarity:  polarity,
			Intensity: intensity,
			Source:    "AFINN",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity[polarity]++
	}

	return entries, res, scanner.Err()
}

// ImportVADER reads the VADER lexicon (TSV: token, mean_sentiment, std, raw_ratings).
func ImportVADER(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			res.Errors++
			continue
		}

		word := strings.TrimSpace(parts[0])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		val, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			res.Errors++
			continue
		}

		polarity := valenceToPolarity(val, -0.5, 0.5)
		intensity := valenceToIntensity(math.Abs(val), 4.0)

		e := Entry{
			Word:      word,
			Lang:      "EN",
			Norm:      normalize(word),
			Polarity:  polarity,
			Intensity: intensity,
			Source:    "VADER",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity[polarity]++
	}

	return entries, res, scanner.Err()
}

// ImportBingLiu reads positive-words.txt and negative-words.txt (one word per line).
func ImportBingLiu(posPath, negPath string) ([]Entry, Result, error) {
	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}

	readFile := func(path, polarity string) error {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			word := strings.TrimSpace(scanner.Text())
			if word == "" || word[0] == ';' {
				continue
			}
			if !isAlpha(word) {
				continue
			}
			e := Entry{
				Word:      word,
				Lang:      "EN",
				Norm:      normalize(word),
				Polarity:  polarity,
				Intensity: "MODERATE",
				Source:    "BingLiu",
			}
			entries = append(entries, e)
			res.Total++
			res.ByPolarity[polarity]++
		}
		return scanner.Err()
	}

	if err := readFile(posPath, "POSITIVE"); err != nil {
		return nil, res, err
	}
	if err := readFile(negPath, "NEGATIVE"); err != nil {
		return nil, res, err
	}

	return entries, res, nil
}

// ImportWarrinerVAD reads Warriner et al. (2013) VAD norms CSV.
// Columns: Word, V.Mean.Sum, A.Mean.Sum, D.Mean.Sum (1-9 scale).
func ImportWarrinerVAD(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}

	r := csv.NewReader(f)
	header, err := r.Read()
	if err != nil {
		return nil, res, err
	}

	// Find column indices
	idx := make(map[string]int)
	for i, h := range header {
		idx[h] = i
	}

	wordCol, ok1 := idx["Word"]
	vCol, ok2 := idx["V.Mean.Sum"]
	aCol, ok3 := idx["A.Mean.Sum"]
	dCol, ok4 := idx["D.Mean.Sum"]
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return nil, res, fmt.Errorf("missing required columns in Warriner CSV")
	}

	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			res.Errors++
			continue
		}

		if len(row) <= wordCol || len(row) <= vCol || len(row) <= aCol || len(row) <= dCol {
			res.Errors++
			continue
		}

		word := strings.TrimSpace(row[wordCol])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		v, err1 := strconv.ParseFloat(row[vCol], 64)
		a, err2 := strconv.ParseFloat(row[aCol], 64)
		d, err3 := strconv.ParseFloat(row[dCol], 64)
		if err1 != nil || err2 != nil || err3 != nil {
			res.Errors++
			continue
		}

		// Warriner scale is 1-9, center at 5
		polarity := valenceToPolarity(v-5.0, -0.5, 0.5)
		intensity := valenceToIntensity(math.Abs(v-5.0), 4.0)
		arousalLvl := vadToLevel(a, 1.0, 9.0)
		dominanceLvl := vadToLevel(d, 1.0, 9.0)

		e := Entry{
			Word:      word,
			Lang:      "EN",
			Norm:      normalize(word),
			Polarity:  polarity,
			Intensity: intensity,
			Arousal:   arousalLvl,
			Dominance: dominanceLvl,
			Source:    "Warriner",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity[polarity]++
	}

	return entries, res, nil
}

// ImportNRCEmoLex reads the NRC Emotion Lexicon multilingual file.
// Format: English_Word \t anger \t ... \t trust \t Lang1 \t Lang2 ...
// Returns entries for ALL languages found in the file.
func ImportNRCEmoLex(path string, targetLangs map[string]bool) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // Large buffer for wide lines

	// Read header to find language columns
	if !scanner.Scan() {
		return nil, res, fmt.Errorf("empty file")
	}
	header := strings.Split(scanner.Text(), "\t")

	// Emotion columns: anger(1), anticipation(2), disgust(3), fear(4),
	//   joy(5), negative(6), positive(7), sadness(8), surprise(9), trust(10)
	// Language columns start at index 11
	nrcToUMCS := map[string]string{
		"Afrikaans": "AF", "Albanian": "SQ", "Arabic": "AR",
		"Bengali": "BN", "Bosnian": "BS", "Bulgarian": "BG",
		"Catalan": "CA", "Chinese-Simplified": "ZH", "Chinese-Traditional": "ZH",
		"Croatian": "HR", "Czech": "CS", "Danish": "DA",
		"Dutch": "NL", "Estonian": "ET", "Filipino": "TL",
		"Finnish": "FI", "French": "FR", "Galician": "GL",
		"Georgian": "KA", "German": "DE", "Greek": "EL",
		"Gujarati": "GU", "Hebrew": "HE", "Hindi": "HI",
		"Hungarian": "HU", "Icelandic": "IS", "Indonesian": "ID",
		"Irish": "GA", "Italian": "IT", "Japanese": "JA",
		"Kannada": "KN", "Kazakh": "KK", "Korean": "KO",
		"Latin": "LA", "Latvian": "LV", "Lithuanian": "LT",
		"Macedonian": "MK", "Malay": "MS", "Malayalam": "ML",
		"Maltese": "MT", "Marathi": "MR", "Mongolian": "MN",
		"Nepali": "NE", "Norwegian": "NO", "Persian": "FA",
		"Polish": "PL", "Portuguese": "PT", "Punjabi": "PA",
		"Romanian": "RO", "Russian": "RU", "Serbian": "SR",
		"Slovak": "SK", "Slovenian": "SL", "Somali": "SO",
		"Spanish": "ES", "Swahili": "SW", "Swedish": "SV",
		"Tamil": "TA", "Telugu": "TE", "Thai": "TH",
		"Turkish": "TR", "Ukrainian": "UK", "Urdu": "UR",
		"Vietnamese": "VI", "Welsh": "CY", "Yiddish": "YI",
		"Zulu": "ZU",
	}

	langCols := make(map[int]string) // column index → UMCS lang code
	for i := 11; i < len(header); i++ {
		name := strings.TrimSpace(header[i])
		if code, ok := nrcToUMCS[name]; ok {
			if targetLangs == nil || targetLangs[code] {
				langCols[i] = code
			}
		}
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 11 {
			res.Errors++
			continue
		}

		enWord := strings.TrimSpace(parts[0])
		if !isAlpha(enWord) {
			continue
		}

		// Parse emotion columns
		negative := parts[6] == "1"
		positive := parts[7] == "1"

		var polarity string
		switch {
		case positive && !negative:
			polarity = "POSITIVE"
		case negative && !positive:
			polarity = "NEGATIVE"
		case positive && negative:
			polarity = "AMBIGUOUS"
		default:
			polarity = "NEUTRAL"
		}

		// Determine arousal from emotion type
		anger := parts[1] == "1"
		fear := parts[4] == "1"
		joy := parts[5] == "1"
		sadness := parts[8] == "1"
		surprise := parts[9] == "1"

		arousal := ""
		if anger || fear || surprise {
			arousal = "HIGH"
		} else if joy {
			arousal = "MED"
		} else if sadness {
			arousal = "LOW"
		}

		// Add English entry
		if targetLangs == nil || targetLangs["EN"] {
			e := Entry{
				Word:     enWord,
				Lang:     "EN",
				Norm:     normalize(enWord),
				Polarity: polarity,
				Intensity: "MODERATE",
				Arousal:  arousal,
				Source:   "NRC-EmoLex",
			}
			entries = append(entries, e)
			res.Total++
			res.ByPolarity[polarity]++
		}

		// Add translations
		for colIdx, langCode := range langCols {
			if colIdx >= len(parts) {
				continue
			}
			translated := strings.TrimSpace(parts[colIdx])
			if translated == "" || !isAlpha(translated) {
				continue
			}
			// Skip if it's the same as the English word
			if normalize(translated) == normalize(enWord) {
				continue
			}

			e := Entry{
				Word:      translated,
				Lang:      langCode,
				Norm:      normalize(translated),
				Polarity:  polarity,
				Intensity: "MODERATE",
				Arousal:   arousal,
				Source:    "NRC-EmoLex",
			}
			entries = append(entries, e)
			res.Total++
			res.ByPolarity[polarity]++
		}
	}

	return entries, res, scanner.Err()
}

// ImportIPADict reads an IPA-dict file (TSV: word \t /IPA/).
// Returns map[normalized_word]IPA for merging into existing entries.
func ImportIPADict(path, lang string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	result := make(map[string]string)
	scanner := bufio.NewScanner(f)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) < 2 {
			continue
		}
		word := strings.TrimSpace(parts[0])
		ipa := strings.TrimSpace(parts[1])
		if word == "" || ipa == "" {
			continue
		}
		// Take only first pronunciation if multiple
		if idx := strings.Index(ipa, ", "); idx >= 0 {
			ipa = ipa[:idx]
		}
		result[normalize(word)] = ipa
	}

	return result, scanner.Err()
}

// ImportSentiWordNet reads SentiWordNet 3.0.
// Format: POS \t ID \t PosScore \t NegScore \t SynsetTerms \t Gloss
// SynsetTerms: "word#sense word#sense ..."
func ImportSentiWordNet(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	seen := make(map[string]bool) // dedup by normalized word

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}

		parts := strings.Split(line, "\t")
		if len(parts) < 5 {
			res.Errors++
			continue
		}

		pos := parts[0]
		posScore, err1 := strconv.ParseFloat(parts[2], 64)
		negScore, err2 := strconv.ParseFloat(parts[3], 64)
		if err1 != nil || err2 != nil {
			res.Errors++
			continue
		}

		// Only process entries with clear sentiment
		netScore := posScore - negScore
		if math.Abs(netScore) < 0.25 {
			continue // too ambiguous
		}

		// Map POS
		var umcsPOS string
		switch pos {
		case "n":
			umcsPOS = "NOUN"
		case "v":
			umcsPOS = "VERB"
		case "a", "s":
			umcsPOS = "ADJ"
		case "r":
			umcsPOS = "ADV"
		}

		synsetTerms := parts[4]
		for _, term := range strings.Fields(synsetTerms) {
			// Remove sense number: word#1 → word
			if idx := strings.Index(term, "#"); idx >= 0 {
				term = term[:idx]
			}
			term = strings.ReplaceAll(term, "_", " ")
			// Skip multi-word
			if strings.Contains(term, " ") {
				continue
			}
			if !isAlpha(term) {
				continue
			}

			norm := normalize(term)
			if seen[norm] {
				continue
			}
			seen[norm] = true

			polarity := valenceToPolarity(netScore, -0.1, 0.1)
			intensity := valenceToIntensity(math.Abs(netScore), 1.0)

			e := Entry{
				Word:      term,
				Lang:      "EN",
				Norm:      norm,
				Polarity:  polarity,
				Intensity: intensity,
				POS:       umcsPOS,
				Source:    "SentiWordNet",
			}

			entries = append(entries, e)
			res.Total++
			res.ByPolarity[polarity]++
		}
	}

	return entries, res, scanner.Err()
}

// Import81LangSentiment reads the Kaggle 81-language sentiment lexicons.
// Each language has positive_words_XX.txt and negative_words_XX.txt.
func Import81LangSentiment(dir string, targetLangs map[string]bool) ([]Entry, Result, error) {
	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}

	// Map 2-letter file suffixes to UMCS lang codes
	codeMap := map[string]string{
		"en": "EN", "pt": "PT", "es": "ES", "fr": "FR", "de": "DE",
		"it": "IT", "nl": "NL", "ar": "AR", "ja": "JA", "ko": "KO",
		"zh": "ZH", "ru": "RU", "hi": "HI", "he": "HE", "tr": "TR",
		"pl": "PL", "sv": "SV", "da": "DA", "fi": "FI", "no": "NO",
		"cs": "CS", "hu": "HU", "ro": "RO", "bg": "BG", "hr": "HR",
		"sk": "SK", "sl": "SL", "et": "ET", "lv": "LV", "lt": "LT",
		"el": "EL", "th": "TH", "vi": "VI", "id": "ID", "ms": "MS",
		"tl": "TL", "uk": "UK", "fa": "FA", "ur": "UR", "bn": "BN",
		"ta": "TA", "te": "TE", "ml": "ML", "kn": "KN", "mr": "MR",
		"gu": "GU", "pa": "PA", "sw": "SW", "af": "AF", "sq": "SQ",
		"ka": "KA", "hy": "HY", "az": "AZ", "eu": "EU", "gl": "GL",
		"ca": "CA", "cy": "CY", "ga": "GA", "is": "IS", "mt": "MT",
		"mk": "MK", "sr": "SR", "bs": "BS", "lb": "LB",
	}

	readWords := func(path, polarity, langCode string) error {
		f, err := os.Open(path)
		if err != nil {
			return nil // Skip missing files
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			word := strings.TrimSpace(scanner.Text())
			if word == "" || !isAlpha(word) {
				continue
			}
			e := Entry{
				Word:      word,
				Lang:      langCode,
				Norm:      normalize(word),
				Polarity:  polarity,
				Intensity: "MODERATE",
				Source:    "81Lang",
			}
			entries = append(entries, e)
			res.Total++
			res.ByPolarity[polarity]++
		}
		return scanner.Err()
	}

	for suffix, langCode := range codeMap {
		if targetLangs != nil && !targetLangs[langCode] {
			continue
		}
		posFile := fmt.Sprintf("%s/positive_words_%s.txt", dir, suffix)
		negFile := fmt.Sprintf("%s/negative_words_%s.txt", dir, suffix)
		_ = readWords(posFile, "POSITIVE", langCode)
		_ = readWords(negFile, "NEGATIVE", langCode)
	}

	return entries, res, nil
}

// ImportOpLexicon reads OpLexicon v3.0 (PT-BR sentiment lexicon).
// Format: word,POS,polarity(-1/0/1),annotator
// Skips emoticons, hashtags, and non-alphabetic entries.
func ImportOpLexicon(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	seen := make(map[string]bool)
	scanner := bufio.NewScanner(f)

	posMap := map[string]string{
		"adj": "ADJ", "vb": "VERB", "n": "NOUN", "adv": "ADV",
		"emot": "", "htag": "",
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, ",")
		if len(parts) < 3 {
			res.Errors++
			continue
		}

		word := strings.TrimSpace(parts[0])
		posTag := strings.TrimSpace(parts[1])
		valStr := strings.TrimSpace(parts[2])

		// Skip emoticons, hashtags, non-words
		if posTag == "emot" || posTag == "htag" {
			continue
		}
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}
		// Skip reflexive forms (ababelar-se → already have ababelar)
		if strings.Contains(word, "-se") || strings.Contains(word, "-lhe") {
			continue
		}

		norm := normalize(word)
		if seen[norm] {
			continue
		}
		seen[norm] = true

		val, err := strconv.Atoi(valStr)
		if err != nil {
			res.Errors++
			continue
		}

		var polarity string
		switch val {
		case 1:
			polarity = "POSITIVE"
		case -1:
			polarity = "NEGATIVE"
		default:
			polarity = "NEUTRAL"
		}

		umcsPOS := posMap[posTag]

		e := Entry{
			Word:      word,
			Lang:      "PT",
			Norm:      norm,
			Polarity:  polarity,
			Intensity: "MODERATE",
			POS:       umcsPOS,
			Source:    "OpLexicon",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity[polarity]++
	}

	return entries, res, scanner.Err()
}

// ImportSentiLex reads SentiLex-PT02 (Portuguese sentiment lexicon).
// Format: lemma.PoS=TAG;TG=target;POL:N0=val;ANOT=annotator
// Example: abafado.PoS=Adj;TG=HUM:N0;POL:N0=-1;ANOT=JALC
func ImportSentiLex(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	scanner := bufio.NewScanner(f)

	posMap := map[string]string{
		"Adj": "ADJ", "V": "VERB", "N": "NOUN", "ADV": "ADV",
		"IDIOM": "", "Adv": "ADV",
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Split lemma from metadata
		dotIdx := strings.Index(line, ".PoS=")
		if dotIdx < 0 {
			res.Errors++
			continue
		}

		word := line[:dotIdx]
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		meta := line[dotIdx+5:] // after ".PoS="

		// Extract POS
		semiIdx := strings.Index(meta, ";")
		posTag := meta
		if semiIdx >= 0 {
			posTag = meta[:semiIdx]
		}

		// Extract polarity from POL:N0=val
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

		var polarity string
		switch {
		case polVal > 0:
			polarity = "POSITIVE"
		case polVal < 0:
			polarity = "NEGATIVE"
		default:
			polarity = "NEUTRAL"
		}

		umcsPOS := posMap[posTag]

		e := Entry{
			Word:      word,
			Lang:      "PT",
			Norm:      normalize(word),
			Polarity:  polarity,
			Intensity: "MODERATE",
			POS:       umcsPOS,
			Source:    "SentiLex",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity[polarity]++
	}

	return entries, res, scanner.Err()
}

// ImportLexique383 reads Lexique 3.83 (French lexicon with frequency + phonology).
// TSV with header: ortho, phon, lemme, cgram, genre, nombre, freqfilms2, ...
// Extracts: word, phonology (SAMPA), POS, frequency rank, syllable count.
func ImportLexique383(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	seen := make(map[string]bool)

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// Read header
	if !scanner.Scan() {
		return nil, res, fmt.Errorf("empty file")
	}
	header := strings.Split(scanner.Text(), "\t")
	idx := make(map[string]int)
	for i, h := range header {
		idx[h] = i
	}

	orthoCol := idx["ortho"]
	phonCol := idx["phon"]
	cgramCol := idx["cgram"]
	freqCol := idx["freqfilms2"]
	syllCol := idx["nbsyll"]

	cgramMap := map[string]string{
		"NOM": "NOUN", "VER": "VERB", "ADJ": "ADJ", "ADV": "ADV",
		"AUX": "VERB", "PRE": "PREP", "CON": "CONJ", "PRO": "PRON",
		"ART": "DET", "ONO": "INTJ",
	}

	for scanner.Scan() {
		parts := strings.Split(scanner.Text(), "\t")
		if len(parts) <= syllCol {
			res.Errors++
			continue
		}

		word := strings.TrimSpace(parts[orthoCol])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		norm := normalize(word)
		if seen[norm] {
			continue
		}
		seen[norm] = true

		phon := strings.TrimSpace(parts[phonCol])
		cgram := strings.TrimSpace(parts[cgramCol])
		umcsPOS := cgramMap[cgram]

		freq, _ := strconv.ParseFloat(parts[freqCol], 64)
		nsyll, _ := strconv.Atoi(parts[syllCol])

		// Convert frequency to rank (higher freq = lower rank)
		freqRank := 0
		if freq > 0 {
			freqRank = int(50000.0 / (freq + 1.0))
			if freqRank < 1 {
				freqRank = 1
			}
			if freqRank > 50000 {
				freqRank = 50000
			}
		}

		e := Entry{
			Word:      word,
			Lang:      "FR",
			Norm:      norm,
			Polarity:  "NEUTRAL", // Lexique has no sentiment, but provides phonology+freq
			Intensity: "NONE",
			POS:       umcsPOS,
			IPA:       phon,
			Syllables: nsyll,
			FreqRank:  freqRank,
			Source:    "Lexique383",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity["NEUTRAL"]++
	}

	return entries, res, scanner.Err()
}

// ImportMPQA reads the MPQA Subjectivity Lexicon (JSON format: word → polarity).
// Polarity values: 1 (positive), -1 (negative), 0 (neutral).
func ImportMPQA(path string) ([]Entry, Result, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, Result{}, err
	}

	// Parse JSON manually since it's a simple {word: int} map
	// The file is one big JSON object
	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}

	content := strings.TrimSpace(string(data))
	if len(content) < 2 || content[0] != '{' {
		return nil, res, fmt.Errorf("invalid MPQA JSON format")
	}
	content = content[1 : len(content)-1] // strip {}

	// Split by comma, parse key-value pairs
	for _, pair := range splitJSONPairs(content) {
		pair = strings.TrimSpace(pair)
		colonIdx := strings.LastIndex(pair, ":")
		if colonIdx < 0 {
			continue
		}

		key := strings.TrimSpace(pair[:colonIdx])
		valStr := strings.TrimSpace(pair[colonIdx+1:])

		// Remove quotes from key
		if len(key) >= 2 && key[0] == '"' {
			key = key[1 : len(key)-1]
		}
		if !isAlpha(key) || strings.Contains(key, " ") {
			continue
		}

		val, err := strconv.Atoi(valStr)
		if err != nil {
			res.Errors++
			continue
		}

		var polarity string
		switch val {
		case 1:
			polarity = "POSITIVE"
		case -1:
			polarity = "NEGATIVE"
		default:
			polarity = "NEUTRAL"
		}

		e := Entry{
			Word:      key,
			Lang:      "EN",
			Norm:      normalize(key),
			Polarity:  polarity,
			Intensity: "MODERATE",
			Source:    "MPQA",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity[polarity]++
	}

	return entries, res, nil
}

// splitJSONPairs splits a JSON object body by top-level commas.
func splitJSONPairs(s string) []string {
	var pairs []string
	depth := 0
	start := 0
	inStr := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"':
			if i == 0 || s[i-1] != '\\' {
				inStr = !inStr
			}
		case '{', '[':
			if !inStr {
				depth++
			}
		case '}', ']':
			if !inStr {
				depth--
			}
		case ',':
			if !inStr && depth == 0 {
				pairs = append(pairs, s[start:i])
				start = i + 1
			}
		}
	}
	if start < len(s) {
		pairs = append(pairs, s[start:])
	}
	return pairs
}

// ImportEmpath reads Empath semantic categories (TSV: category \t word1 word2 ...).
// Returns a map of category → word list for semantic domain enrichment.
func ImportEmpath(path string) (map[string][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	result := make(map[string][]string)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			continue
		}
		category := strings.TrimSpace(parts[0])
		if category == "" {
			continue
		}
		var words []string
		for _, w := range parts[1:] {
			w = strings.TrimSpace(w)
			if w != "" && isAlpha(w) {
				words = append(words, w)
			}
		}
		if len(words) > 0 {
			result[category] = words
		}
	}

	return result, scanner.Err()
}

// Merge deduplicates entries by (norm, lang), preferring entries with more data.
// When duplicates exist, it merges fields: VAD from one, polarity from another.
func Merge(entries []Entry) []Entry {
	type key struct {
		norm, lang string
	}
	best := make(map[key]*Entry)

	for i := range entries {
		e := &entries[i]
		k := key{e.Norm, e.Lang}
		if existing, ok := best[k]; ok {
			// Merge: prefer non-empty fields from new entry
			if existing.Arousal == "" && e.Arousal != "" {
				existing.Arousal = e.Arousal
			}
			if existing.Dominance == "" && e.Dominance != "" {
				existing.Dominance = e.Dominance
			}
			if existing.IPA == "" && e.IPA != "" {
				existing.IPA = e.IPA
			}
			if existing.POS == "" && e.POS != "" {
				existing.POS = e.POS
			}
			if existing.Concreteness == "" && e.Concreteness != "" {
				existing.Concreteness = e.Concreteness
			}
			// If existing is NEUTRAL but new has sentiment, prefer new
			if existing.Polarity == "NEUTRAL" && e.Polarity != "NEUTRAL" {
				existing.Polarity = e.Polarity
				existing.Intensity = e.Intensity
			}
		} else {
			clone := *e
			best[k] = &clone
		}
	}

	result := make([]Entry, 0, len(best))
	for _, e := range best {
		result = append(result, *e)
	}
	return result
}

// ExcludeExisting removes entries that already exist in the lexicon CSV.
func ExcludeExisting(entries []Entry, wordsCSVPath string) ([]Entry, int, error) {
	f, err := os.Open(wordsCSVPath)
	if err != nil {
		return entries, 0, nil // File doesn't exist = no exclusions
	}
	defer f.Close()

	existing := make(map[string]bool)
	r := csv.NewReader(f)
	header, _ := r.Read() // skip header

	// Find norm and lang columns
	normIdx, langIdx := -1, -1
	for i, h := range header {
		switch h {
		case "norm":
			normIdx = i
		case "lang":
			langIdx = i
		}
	}
	if normIdx < 0 || langIdx < 0 {
		return entries, 0, fmt.Errorf("missing norm/lang columns in words CSV")
	}

	for {
		row, err := r.Read()
		if err != nil {
			break
		}
		if len(row) > normIdx && len(row) > langIdx {
			key := row[normIdx] + "|" + row[langIdx]
			existing[key] = true
		}
	}

	var filtered []Entry
	skipped := 0
	for _, e := range entries {
		key := e.Norm + "|" + e.Lang
		if existing[key] {
			skipped++
			continue
		}
		filtered = append(filtered, e)
	}

	return filtered, skipped, nil
}

// WriteCSV appends entries to a words.csv file. If the file doesn't exist,
// it creates it with a header. Uses sequential word_id starting from nextID.
func WriteCSV(path string, entries []Entry, nextWordID, defaultRootID int) error {
	// Check if file exists
	_, err := os.Stat(path)
	isNew := os.IsNotExist(err)

	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	if isNew {
		w.Write([]string{
			"word_id", "root_id", "variant", "word", "lang", "norm",
			"polarity", "intensity", "semantic_role", "domain", "freq_rank",
			"flags", "pos", "arousal", "dominance", "aoa", "concreteness",
			"register", "ontological", "polysemy", "pron", "syllables",
			"stress", "valency", "irony_capable", "neologism",
		})
	}

	id := nextWordID
	for _, e := range entries {
		syllStr := ""
		if e.Syllables > 0 {
			syllStr = strconv.Itoa(e.Syllables)
		}
		freqStr := ""
		if e.FreqRank > 0 {
			freqStr = strconv.Itoa(e.FreqRank)
		}

		w.Write([]string{
			strconv.Itoa(id),
			strconv.Itoa(defaultRootID), // placeholder root
			"1",                          // variant
			e.Word, e.Lang, e.Norm,
			e.Polarity, e.Intensity,
			"", "",    // semantic_role, domain
			freqStr,   // freq_rank
			"0",       // flags
			e.POS,
			e.Arousal, e.Dominance,
			e.AoA, e.Concreteness,
			"", "", "", // register, ontological, polysemy
			e.IPA,
			syllStr,
			"", "", "", "", // stress, valency, irony_capable, neologism
		})
		id++
	}

	return nil
}
