// expand.go adds importers for large-scale lexicon expansion datasets:
//   - UniMorph (morphological inflections, 5 languages)
//   - CogNet v2.0 (cognate pairs, 338 languages)
//   - Brysbaert Concreteness Norms (40k EN)
//   - MorphoLex (31k EN morphological families)
//   - ML-Senticon (sentiment, 5 languages: EN/ES/CA/EU/GL)
//   - SO-CAL (sentiment dictionaries, EN + ES)
//   - EtymWn (etymological word network, 6M relations)
package ingest

import (
	"bufio"
	"encoding/xml"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

// ── UniMorph ─────────────────────────────────────────────────────────────────

// ImportUniMorph reads a UniMorph TSV file (lemma \t inflection \t tags).
// Extracts unique lemma forms with POS derived from morphological tags.
// UniMorph tags follow the UniMorph schema: N;PL, V;PST, ADJ, etc.
func ImportUniMorph(path, lang string) ([]Entry, Result, error) {
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

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 3 {
			continue
		}

		lemma := strings.TrimSpace(parts[0])
		inflection := strings.TrimSpace(parts[1])
		tags := strings.TrimSpace(parts[2])

		// Extract POS from tags (first tag before ;)
		pos := unimorphPOS(tags)

		// Add lemma
		normL := normalize(lemma)
		if normL != "" && isAlpha(lemma) && !strings.Contains(lemma, " ") && !seen[normL] {
			seen[normL] = true
			entries = append(entries, Entry{
				Word:     lemma,
				Lang:     lang,
				Norm:     normL,
				Polarity: "NEUTRAL",
				POS:      pos,
				Source:   "UniMorph",
			})
			res.Total++
			res.ByPolarity["NEUTRAL"]++
		}

		// Add inflection if different
		normI := normalize(inflection)
		if normI != "" && normI != normL && isAlpha(inflection) &&
			!strings.Contains(inflection, " ") && !seen[normI] {
			seen[normI] = true
			entries = append(entries, Entry{
				Word:     inflection,
				Lang:     lang,
				Norm:     normI,
				Polarity: "NEUTRAL",
				POS:      pos,
				Source:   "UniMorph",
			})
			res.Total++
			res.ByPolarity["NEUTRAL"]++
		}
	}

	return entries, res, scanner.Err()
}

// unimorphPOS maps UniMorph tags to UMCS POS.
func unimorphPOS(tags string) string {
	first := tags
	if idx := strings.Index(tags, ";"); idx >= 0 {
		first = tags[:idx]
	}
	switch first {
	case "N":
		return "NOUN"
	case "V":
		return "VERB"
	case "ADJ":
		return "ADJ"
	case "ADV":
		return "ADV"
	default:
		return ""
	}
}

// ── CogNet ───────────────────────────────────────────────────────────────────

// ImportCogNet reads CogNet v2.0 TSV (concept_id, lang1, word1, lang2, word2, translit1, translit2).
// Extracts unique (word, lang) pairs across all languages. Maps ISO 639-3 codes to 2-letter.
func ImportCogNet(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	seen := make(map[string]bool) // norm|lang → seen

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// Skip header
	if scanner.Scan() {
		// header: concept id, lang 1, word 1, lang 2, word 2, translit 1, translit 2
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 5 {
			continue
		}

		lang1ISO := strings.TrimSpace(parts[1])
		word1 := strings.TrimSpace(parts[2])
		lang2ISO := strings.TrimSpace(parts[3])
		word2 := strings.TrimSpace(parts[4])

		for _, pair := range [][2]string{{word1, lang1ISO}, {word2, lang2ISO}} {
			word, langISO := pair[0], pair[1]
			lang := cognetLang(langISO)
			if lang == "" {
				continue
			}
			if !isAlpha(word) || strings.Contains(word, " ") || len(word) > 40 {
				continue
			}
			norm := normalize(word)
			key := norm + "|" + lang
			if seen[key] {
				continue
			}
			seen[key] = true

			entries = append(entries, Entry{
				Word:     word,
				Lang:     lang,
				Norm:     norm,
				Polarity: "NEUTRAL",
				Source:   "CogNet",
			})
			res.Total++
			res.ByPolarity["NEUTRAL"]++
		}
	}

	return entries, res, scanner.Err()
}

// cognetLang maps ISO 639-3 to 2-letter UMCS codes.
// Returns empty string for unsupported languages.
func cognetLang(iso3 string) string {
	m := map[string]string{
		"eng": "EN", "por": "PT", "spa": "ES", "fra": "FR", "deu": "DE",
		"ita": "IT", "nld": "NL", "cat": "CA", "glg": "GL", "ron": "RO",
		"pol": "PL", "ces": "CS", "slk": "SK", "hrv": "HR", "slv": "SL",
		"bul": "BG", "ukr": "UK", "rus": "RU", "srp": "SR",
		"swe": "SV", "nor": "NO", "dan": "DA", "fin": "FI",
		"tur": "TR", "hun": "HU", "ell": "EL", "lat": "LA",
		"arb": "AR", "heb": "HE", "fas": "FA", "hin": "HI", "ben": "BN",
		"urd": "UR", "tam": "TA", "tel": "TE", "mal": "ML", "kan": "KN",
		"mar": "MR", "guj": "GU", "pan": "PA", "nep": "NE", "sin": "SI",
		"zho": "ZH", "jpn": "JA", "kor": "KO", "vie": "VI",
		"tha": "TH", "msa": "MS", "zsm": "MS", "ind": "ID",
		"eus": "EU", "cym": "CY", "gle": "GA", "kat": "KA",
		"hye": "HY", "mkd": "MK", "lit": "LT", "lav": "LV", "est": "ET",
		"sqi": "SQ", "afr": "AF", "isl": "IS", "mlt": "MT",
		"cmn": "ZH", "yue": "ZH", "hak": "ZH",
		"bos": "BS", "tgl": "TL", "swh": "SW", "amh": "AM",
	}
	return m[iso3]
}

// ── Brysbaert Concreteness ───────────────────────────────────────────────────

// ImportBrysbaertConcreteness reads Brysbaert et al. (2014) concreteness ratings.
// TSV with header: Word, Bigram, Conc.M, Conc.SD, Unknown, Total, Percent_known, SUBTLEX
// Conc.M range: 1 (abstract) to 5 (concrete). Threshold: ≥ 3.5 = concrete.
func ImportBrysbaertConcreteness(path string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}

	scanner := bufio.NewScanner(f)
	// Skip header
	if !scanner.Scan() {
		return nil, res, fmt.Errorf("empty file")
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 8 {
			res.Errors++
			continue
		}

		word := strings.TrimSpace(parts[0])
		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		concM, err := strconv.ParseFloat(parts[2], 64)
		if err != nil {
			res.Errors++
			continue
		}

		concrete := ""
		if concM >= 3.5 {
			concrete = "1"
		}

		// SUBTLEX frequency
		subtlex, _ := strconv.ParseFloat(parts[7], 64)
		freqRank := 0
		if subtlex > 0 {
			freqRank = int(50000.0 / (math.Log10(subtlex+1)*100 + 1))
			if freqRank < 1 {
				freqRank = 1
			}
			if freqRank > 50000 {
				freqRank = 50000
			}
		}

		e := Entry{
			Word:         word,
			Lang:         "EN",
			Norm:         normalize(word),
			Polarity:     "NEUTRAL",
			Concreteness: concrete,
			FreqRank:     freqRank,
			Source:       "Brysbaert",
		}

		entries = append(entries, e)
		res.Total++
		res.ByPolarity["NEUTRAL"]++
	}

	return entries, res, scanner.Err()
}

// ── MorphoLex ────────────────────────────────────────────────────────────────

// ImportMorphoLex reads MorphoLex-EN morphological family data.
// TSV with header: (blank), (blank), (blank), ELP_ItemID, Word, POS, Nmorph, ...
// Extracts words with POS and morpheme segmentation.
func ImportMorphoLex(path string) ([]Entry, Result, error) {
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

	// Skip header
	if !scanner.Scan() {
		return nil, res, fmt.Errorf("empty file")
	}

	morphoPOS := map[string]string{
		"NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN",
		"VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",
		"JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
		"RB": "ADV", "RBR": "ADV", "RBS": "ADV",
	}

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 7 {
			continue
		}

		// Columns start with 3 blank tabs, then ELP_ItemID, Word, POS, Nmorph...
		wordIdx := 4
		posIdx := 5
		if len(parts) <= posIdx {
			continue
		}

		word := strings.TrimSpace(parts[wordIdx])
		posTag := strings.TrimSpace(parts[posIdx])

		if !isAlpha(word) || strings.Contains(word, " ") {
			continue
		}

		norm := normalize(word)
		if seen[norm] {
			continue
		}
		seen[norm] = true

		// Map POS — posTag can be "JJ|RB" (multiple)
		umcsPOS := ""
		for _, p := range strings.Split(posTag, "|") {
			if mapped, ok := morphoPOS[p]; ok {
				umcsPOS = mapped
				break
			}
		}

		entries = append(entries, Entry{
			Word:     word,
			Lang:     "EN",
			Norm:     norm,
			Polarity: "NEUTRAL",
			POS:      umcsPOS,
			Source:   "MorphoLex",
		})
		res.Total++
		res.ByPolarity["NEUTRAL"]++
	}

	return entries, res, scanner.Err()
}

// ── ML-Senticon ──────────────────────────────────────────────────────────────

// senticonDoc represents the XML structure of ML-Senticon files.
type senticonDoc struct {
	Lang   string          `xml:"lang,attr"`
	Layers []senticonLayer `xml:"layer"`
}

type senticonLayer struct {
	Level    int             `xml:"level,attr"`
	Positive []senticonLemma `xml:"positive>lemma"`
	Negative []senticonLemma `xml:"negative>lemma"`
}

type senticonLemma struct {
	POS string  `xml:"pos,attr"`
	Pol float64 `xml:"pol,attr"`
	Std float64 `xml:"std,attr"`
	Word string `xml:",chardata"`
}

// ImportMLSenticon reads an ML-Senticon XML file (senticon.{lang}.xml).
// Returns sentiment-annotated entries with continuous polarity mapped to intensity.
func ImportMLSenticon(path, lang string) ([]Entry, Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, Result{}, err
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, Result{}, err
	}

	var doc senticonDoc
	if err := xml.Unmarshal(data, &doc); err != nil {
		return nil, Result{}, fmt.Errorf("XML parse: %w", err)
	}

	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	seen := make(map[string]bool)

	sentiPOS := map[string]string{
		"a": "ADJ", "n": "NOUN", "v": "VERB", "r": "ADV",
	}

	for _, layer := range doc.Layers {
		for _, lemma := range layer.Positive {
			word := strings.TrimSpace(lemma.Word)
			if !isAlpha(word) {
				continue
			}
			// Handle multi-word (e.g. "cheer_up") — replace _ with nothing, skip spaces
			if strings.Contains(word, " ") {
				continue
			}
			word = strings.ReplaceAll(word, "_", "")
			norm := normalize(word)
			if seen[norm] {
				continue
			}
			seen[norm] = true

			entries = append(entries, Entry{
				Word:      word,
				Lang:      lang,
				Norm:      norm,
				Polarity:  "POSITIVE",
				Intensity: polToIntensity(lemma.Pol),
				POS:       sentiPOS[lemma.POS],
				Source:    "ML-Senticon",
			})
			res.Total++
			res.ByPolarity["POSITIVE"]++
		}

		for _, lemma := range layer.Negative {
			word := strings.TrimSpace(lemma.Word)
			if !isAlpha(word) {
				continue
			}
			if strings.Contains(word, " ") {
				continue
			}
			word = strings.ReplaceAll(word, "_", "")
			norm := normalize(word)
			if seen[norm] {
				continue
			}
			seen[norm] = true

			entries = append(entries, Entry{
				Word:      word,
				Lang:      lang,
				Norm:      norm,
				Polarity:  "NEGATIVE",
				Intensity: polToIntensity(lemma.Pol),
				POS:       sentiPOS[lemma.POS],
				Source:    "ML-Senticon",
			})
			res.Total++
			res.ByPolarity["NEGATIVE"]++
		}
	}

	return entries, res, nil
}

// polToIntensity maps a continuous polarity score [0,1] to UMCS intensity.
func polToIntensity(pol float64) string {
	absPol := pol
	if absPol < 0 {
		absPol = -absPol
	}
	switch {
	case absPol >= 0.85:
		return "EXTREME"
	case absPol >= 0.65:
		return "STRONG"
	case absPol >= 0.35:
		return "MODERATE"
	case absPol > 0.05:
		return "WEAK"
	default:
		return "NONE"
	}
}

// ── SO-CAL ───────────────────────────────────────────────────────────────────

// ImportSOCAL reads SO-CAL sentiment dictionary files (word \t score format).
// Score range: [-5, +5]. Maps to UMCS polarity and intensity.
// Pass the directory containing the dictionary files.
func ImportSOCAL(dir, lang string) ([]Entry, Result, error) {
	var entries []Entry
	res := Result{ByPolarity: make(map[string]int)}
	seen := make(map[string]bool)

	// Read all dictionary files in the directory
	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		return nil, res, err
	}

	posFromFile := map[string]string{
		"adj":  "ADJ",
		"adv":  "ADV",
		"noun": "NOUN",
		"verb": "VERB",
		"int":  "INTJ",
	}

	for _, de := range dirEntries {
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".txt") {
			continue
		}
		// Skip google_dict (different format)
		if strings.Contains(de.Name(), "google") {
			continue
		}

		// Determine POS from filename
		filePOS := ""
		nameLower := strings.ToLower(de.Name())
		for prefix, pos := range posFromFile {
			if strings.Contains(nameLower, prefix) {
				filePOS = pos
				break
			}
		}

		fpath := dir + "/" + de.Name()
		f, err := os.Open(fpath)
		if err != nil {
			continue
		}

		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}

			parts := strings.Fields(line)
			if len(parts) < 2 {
				continue
			}

			word := parts[0]
			scoreStr := parts[len(parts)-1]

			if !isAlpha(word) || strings.Contains(word, " ") {
				continue
			}

			score, err := strconv.ParseFloat(scoreStr, 64)
			if err != nil {
				res.Errors++
				continue
			}

			norm := normalize(word)
			if seen[norm] {
				continue
			}
			seen[norm] = true

			var polarity string
			switch {
			case score > 0:
				polarity = "POSITIVE"
			case score < 0:
				polarity = "NEGATIVE"
			default:
				polarity = "NEUTRAL"
			}

			absScore := score
			if absScore < 0 {
				absScore = -absScore
			}
			var intensity string
			switch {
			case absScore >= 4:
				intensity = "EXTREME"
			case absScore >= 3:
				intensity = "STRONG"
			case absScore >= 2:
				intensity = "MODERATE"
			case absScore >= 1:
				intensity = "WEAK"
			default:
				intensity = "NONE"
			}

			entries = append(entries, Entry{
				Word:      word,
				Lang:      lang,
				Norm:      norm,
				Polarity:  polarity,
				Intensity: intensity,
				POS:       filePOS,
				Source:    "SO-CAL",
			})
			res.Total++
			res.ByPolarity[polarity]++
		}
		f.Close()
	}

	return entries, res, nil
}

// ── EtymWn ───────────────────────────────────────────────────────────────────

// ImportEtymWn reads the Etymological Wordnet TSV.
// Format: lang:word \t rel:relation \t lang:word
// Extracts unique (word, lang) pairs from both source and target.
// Only keeps words from languages we support.
func ImportEtymWn(path string) ([]Entry, Result, error) {
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

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) < 3 {
			continue
		}

		// Parse "lang: word" format
		for _, side := range []string{parts[0], parts[2]} {
			colonIdx := strings.Index(side, ": ")
			if colonIdx < 0 {
				continue
			}
			langISO := strings.TrimSpace(side[:colonIdx])
			word := strings.TrimSpace(side[colonIdx+2:])

			lang := etymwnLang(langISO)
			if lang == "" || !isAlpha(word) || strings.Contains(word, " ") || len(word) > 40 {
				continue
			}

			norm := normalize(word)
			key := norm + "|" + lang
			if seen[key] {
				continue
			}
			seen[key] = true

			entries = append(entries, Entry{
				Word:     word,
				Lang:     lang,
				Norm:     norm,
				Polarity: "NEUTRAL",
				Source:   "EtymWn",
			})
			res.Total++
			res.ByPolarity["NEUTRAL"]++
		}
	}

	return entries, res, scanner.Err()
}

// etymwnLang maps ISO 639-3 codes used in EtymWn to 2-letter UMCS codes.
func etymwnLang(iso3 string) string {
	m := map[string]string{
		"eng": "EN", "por": "PT", "spa": "ES", "fra": "FR", "deu": "DE",
		"ita": "IT", "nld": "NL", "cat": "CA", "glg": "GL", "ron": "RO",
		"pol": "PL", "ces": "CS", "slk": "SK", "hrv": "HR", "slv": "SL",
		"bul": "BG", "ukr": "UK", "rus": "RU", "srp": "SR",
		"swe": "SV", "nor": "NO", "dan": "DA", "fin": "FI",
		"tur": "TR", "hun": "HU", "ell": "EL", "lat": "LA",
		"arb": "AR", "heb": "HE", "fas": "FA", "hin": "HI",
		"zho": "ZH", "jpn": "JA", "kor": "KO", "vie": "VI",
		"tha": "TH", "msa": "MS", "ind": "ID",
		"eus": "EU", "cym": "CY", "gle": "GA",
		"sqi": "SQ", "afr": "AF", "isl": "IS",
	}
	return m[iso3]
}

// ── CMUDict IPA ──────────────────────────────────────────────────────────────

// ImportCMUDictIPA reads CMUDict IPA pronunciation file (word \t IPA format).
// Returns a map of normalized word → IPA string for enrichment.
func ImportCMUDictIPA(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	result := make(map[string]string)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, ";") || line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) < 2 {
			// Try space separator
			parts = strings.SplitN(line, "  ", 2)
			if len(parts) < 2 {
				continue
			}
		}

		word := strings.TrimSpace(parts[0])
		ipa := strings.TrimSpace(parts[1])

		// Skip entries with variant markers like WORD(2)
		if strings.Contains(word, "(") {
			continue
		}

		norm := normalize(word)
		if norm != "" && ipa != "" {
			result[norm] = ipa
		}
	}

	return result, scanner.Err()
}
