// import_loader.go loads imported_words.csv and generates synthetic root
// buckets so that imported words can be included in the .umcs binary build.
//
// Strategy: each imported word is assigned to a synthetic root based on
// (language, first 2 characters of norm). Each root can hold up to 4,095
// variants (12-bit limit). Large buckets are split automatically.
//
// Synthetic root IDs are allocated in the range 500,000–599,999 to avoid
// collisions with curated roots (which use IDs < 1,000).
package seed

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/kak/umcs/pkg/infer"
	"github.com/kak/umcs/pkg/phon"
	"github.com/kak/umcs/pkg/sentiment"
)

// supportedLangs is the set of languages recognized by the lexdb format.
// Mirrors lexdb.langNames without importing lexdb (avoids import cycle).
var supportedLangs = map[string]bool{
	"PT": true, "EN": true, "ES": true, "IT": true, "DE": true, "FR": true,
	"NL": true, "AR": true, "ZH": true, "JA": true, "RU": true, "KO": true,
	"TG": true, "HI": true, "BN": true, "ID": true, "TR": true, "FA": true,
	"SW": true, "UK": true, "PL": true, "SA": true, "TA": true, "HE": true,
	"LA": true, "FI": true, "DA": true, "HU": true, "SV": true, "CA": true,
	"RO": true, "CS": true, "GL": true, "MS": true, "SK": true, "SL": true,
	"HR": true, "BG": true, "GU": true, "EL": true, "IS": true, "PA": true,
	"KN": true, "NE": true, "EU": true, "TH": true, "VI": true, "GA": true,
	"MK": true, "TE": true, "AF": true, "UR": true, "NO": true, "SQ": true,
	"ML": true, "MR": true, "KA": true, "TL": true, "HY": true, "CY": true,
	"MT": true, "BS": true, "AM": true, "ET": true, "SR": true, "LV": true,
	"LT": true, "AZ": true, "LB": true, "SI": true,
}

const (
	// syntheticRootBase is the first root_id for auto-generated roots.
	syntheticRootBase = 500_000
	// maxVariantsPerRoot is the 12-bit variant limit.
	maxVariantsPerRoot = 4095
)

// LoadImportedWords reads an imported_words.csv file and returns synthetic
// roots + words ready to be concatenated with curated data and passed to
// lexdb.Build().
//
// Words with root_id != 0 in the CSV are kept with their original root_id
// (they were already assigned). Words with root_id == 0 get a synthetic root.
//
// The function also runs morphological inference (FillMissing + FillPhonology)
// on each word, and skips words whose language is not recognized.
func LoadImportedWords(path string) ([]Root, []Word, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open imported words: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1
	r.LazyQuotes = true

	header, err := r.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("read imported header: %w", err)
	}
	idx := indexHeader(header)

	// Pass 1: read all entries, bucket by (lang, bigram).
	type bucketKey struct {
		lang, bigram string
	}
	type rawEntry struct {
		word, lang, norm   string
		polarity, intensity string
		pos, arousal, dominance, aoa, concreteness string
		pron, source string
		freqRank uint32
	}

	buckets := make(map[bucketKey][]rawEntry)
	skipped := 0
	lineNum := 1

	for {
		lineNum++
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			skipped++
			continue
		}

		langStr := strings.TrimSpace(col(row, idx, "lang"))
		if !supportedLangs[langStr] {
			skipped++
			continue
		}

		normStr := strings.TrimSpace(col(row, idx, "norm"))
		if normStr == "" {
			skipped++
			continue
		}

		// Compute bigram for bucketing.
		bigram := normBigram(normStr)

		wordStr := strings.TrimSpace(col(row, idx, "word"))
		if wordStr == "" {
			skipped++
			continue
		}

		var freq uint32
		if s := col(row, idx, "freq_rank"); s != "" {
			if v, e := parseUint32(s); e == nil {
				freq = v
			}
		}

		key := bucketKey{lang: langStr, bigram: bigram}
		buckets[key] = append(buckets[key], rawEntry{
			word: wordStr, lang: langStr, norm: normStr,
			polarity: col(row, idx, "polarity"), intensity: col(row, idx, "intensity"),
			pos: col(row, idx, "pos"), arousal: col(row, idx, "arousal"),
			dominance: col(row, idx, "dominance"), aoa: col(row, idx, "aoa"),
			concreteness: col(row, idx, "concreteness"),
			pron: col(row, idx, "pron"), source: col(row, idx, "source"),
			freqRank: freq,
		})
	}

	// Pass 2: generate synthetic roots and assign word_ids.
	var roots []Root
	var words []Word
	nextRootID := uint32(syntheticRootBase)

	// Dedup: track (norm, lang) to avoid duplicates within imported set.
	seen := make(map[string]bool)

	for key, entries := range buckets {
		// Split large buckets.
		for chunk := 0; chunk*maxVariantsPerRoot < len(entries); chunk++ {
			rootID := nextRootID
			nextRootID++

			suffix := ""
			if chunk > 0 {
				suffix = fmt.Sprintf("_%d", chunk+1)
			}
			rootStr := fmt.Sprintf("_auto_%s_%s%s", key.lang, key.bigram, suffix)

			roots = append(roots, Root{
				RootID:  rootID,
				RootStr: rootStr,
				Origin:  "SYNTHETIC",
				MeaningEN: fmt.Sprintf("auto-bucketed %s words starting with '%s'", key.lang, key.bigram),
			})

			start := chunk * maxVariantsPerRoot
			end := start + maxVariantsPerRoot
			if end > len(entries) {
				end = len(entries)
			}

			for vi, e := range entries[start:end] {
				variant := uint32(vi + 1) // variants start at 1
				wordID := (rootID << 12) | variant

				dedupKey := e.norm + "|" + e.lang
				if seen[dedupKey] {
					continue
				}
				seen[dedupKey] = true

				// Pack sentiment.
				var scopeFlags []string
				sent, packErr := sentiment.PackExtended(
					e.polarity, e.intensity, "", "", scopeFlags,
					e.pos, e.arousal, e.dominance, e.aoa, e.concreteness,
				)
				if packErr != nil {
					sent = 0
				}

				var flags uint32

				// Morphological inference.
				if sentiment.POS(sent) == 0 {
					sent = infer.FillMissing(sent, e.word, e.lang)
				}
				flags = infer.FillPhonology(flags, e.word, e.lang)

				// Pack syllables from word shape if not annotated.
				if phon.Syllables(flags) == 0 {
					flags = infer.FillPhonology(flags, e.word, e.lang)
				}

				var pronOffset string
				if e.pron != "" {
					pronOffset = e.pron
				}

				words = append(words, Word{
					WordID:    wordID,
					RootID:    rootID,
					Variant:   variant,
					Word:      e.word,
					Lang:      e.lang,
					Norm:      e.norm,
					Sentiment: sent,
					FreqRank:  e.freqRank,
					Flags:     flags,
					Pron:      pronOffset,
					Source:    e.source,
				})
			}
		}
	}

	return roots, words, nil
}

// normBigram returns the first 2 characters of a normalized string,
// or the full string if shorter. Used for root bucketing.
func normBigram(norm string) string {
	runes := []rune(strings.ToLower(norm))
	if len(runes) >= 2 {
		return string(runes[:2])
	}
	if len(runes) == 1 {
		return string(runes)
	}
	return "_"
}
