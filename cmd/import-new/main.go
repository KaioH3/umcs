package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/kak/umcs/pkg/ingest"
)

func main() {
	base := filepath.Dir(filepath.Dir(os.Args[0]))
	if env := os.Getenv("UMCS_ROOT"); env != "" {
		base = env
	} else {
		wd, err := os.Getwd()
		if err == nil {
			if _, serr := os.Stat(filepath.Join(wd, "data", "imported_words.csv")); serr == nil {
				base = wd
			}
		}
	}

	dataDir := filepath.Join(base, "data")
	extDir := filepath.Join(dataDir, "external")
	csvPath := filepath.Join(dataDir, "imported_words.csv")

	nextID, err := findNextWordID(csvPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading existing CSV: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Existing CSV: next word_id = %d\n\n", nextID)

	var allEntries []ingest.Entry

	// ═══ Phase 1: Sentiment-bearing datasets ═══════════════════════════════

	// OpLexicon v3.0 (PT)
	importDataset(&allEntries, "OpLexicon", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportOpLexicon(filepath.Join(extDir, "OpLexicon_v3.0.txt"))
	})

	// SentiLex-PT02 (PT)
	importDataset(&allEntries, "SentiLex", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportSentiLex(filepath.Join(extDir, "SentiLex-PT02.txt"))
	})

	// MPQA (EN)
	importDataset(&allEntries, "MPQA", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportMPQA(filepath.Join(extDir, "mpqa-subj.tff"))
	})

	// ML-Senticon (EN, ES, CA, EU, GL)
	senticonLangs := map[string]string{
		"en": "EN", "es": "ES", "ca": "CA", "eu": "EU", "gl": "GL",
	}
	for code, lang := range senticonLangs {
		name := fmt.Sprintf("ML-Senticon-%s", strings.ToUpper(code))
		path := filepath.Join(extDir, "ML-Senticon", fmt.Sprintf("senticon.%s.xml", code))
		importDataset(&allEntries, name, func() ([]ingest.Entry, ingest.Result, error) {
			return ingest.ImportMLSenticon(path, lang)
		})
	}

	// SO-CAL English
	importDataset(&allEntries, "SO-CAL-EN", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportSOCAL(filepath.Join(extDir, "SO-CAL", "Resources", "dictionaries", "English"), "EN")
	})

	// SO-CAL Spanish
	importDataset(&allEntries, "SO-CAL-ES", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportSOCAL(filepath.Join(extDir, "SO-CAL", "Resources", "dictionaries", "Spanish"), "ES")
	})

	// ═══ Phase 2: Morphological / phonological datasets ════════════════════

	// Lexique383 (FR)
	importDataset(&allEntries, "Lexique383", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportLexique383(filepath.Join(extDir, "Lexique383.tsv"))
	})

	// UniMorph (5 languages)
	umLangs := map[string]string{
		"en": "EN", "pt": "PT", "es": "ES", "de": "DE", "fr": "FR",
	}
	for code, lang := range umLangs {
		name := fmt.Sprintf("UniMorph-%s", strings.ToUpper(code))
		path := filepath.Join(extDir, fmt.Sprintf("unimorph-%s.tsv", code))
		importDataset(&allEntries, name, func() ([]ingest.Entry, ingest.Result, error) {
			return ingest.ImportUniMorph(path, lang)
		})
	}

	// MorphoLex (EN)
	importDataset(&allEntries, "MorphoLex", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportMorphoLex(filepath.Join(extDir, "morpholex-en.tsv"))
	})

	// Brysbaert Concreteness (EN)
	importDataset(&allEntries, "Brysbaert", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportBrysbaertConcreteness(filepath.Join(extDir, "brysbaert-concreteness.tsv"))
	})

	// ═══ Phase 3: Cross-lingual / etymological datasets ════════════════════

	// CogNet v2.0 (338 languages)
	importDataset(&allEntries, "CogNet-v2", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportCogNet(filepath.Join(extDir, "CogNet", "CogNet-v2.0.tsv"))
	})

	// EtymWn (etymological word network)
	importDataset(&allEntries, "EtymWn", func() ([]ingest.Entry, ingest.Result, error) {
		return ingest.ImportEtymWn(filepath.Join(extDir, "etymwn", "etymwn.tsv"))
	})

	fmt.Printf("\n═══════════════════════════════════════════════════\n")
	fmt.Printf("Total raw entries: %d\n", len(allEntries))

	// ═══ Enrichment: IPA from IPA-dict + CMUDict ═══════════════════════════

	ipaLangs := map[string]string{
		"PT": "pt", "FR": "fr", "EN": "en", "ES": "es", "DE": "de",
		"AR": "ar", "JA": "ja", "KO": "ko", "ZH": "zh",
	}
	ipaMaps := make(map[string]map[string]string)
	for lang, code := range ipaLangs {
		ipaPath := filepath.Join(extDir, fmt.Sprintf("IPA-dict-%s.txt", code))
		m, err := ingest.ImportIPADict(ipaPath, lang)
		if err != nil {
			continue
		}
		ipaMaps[lang] = m
		fmt.Printf("IPA-dict-%s:  %d pronunciations loaded\n", code, len(m))
	}

	// Also load CMUDict IPA
	cmuPath := filepath.Join(extDir, "cmudict-ipa.txt")
	cmuIPA, err := ingest.ImportCMUDictIPA(cmuPath)
	if err == nil && len(cmuIPA) > 0 {
		fmt.Printf("CMUDict-IPA: %d pronunciations loaded\n", len(cmuIPA))
		// Merge into EN IPA map (CMUDict as fallback)
		if ipaMaps["EN"] == nil {
			ipaMaps["EN"] = cmuIPA
		} else {
			for k, v := range cmuIPA {
				if _, exists := ipaMaps["EN"][k]; !exists {
					ipaMaps["EN"][k] = v
				}
			}
		}
	}

	enriched := 0
	for i := range allEntries {
		e := &allEntries[i]
		if e.IPA != "" {
			continue
		}
		if m, ok := ipaMaps[e.Lang]; ok {
			norm := strings.ToLower(strings.TrimSpace(e.Norm))
			if ipa, found := m[norm]; found {
				e.IPA = ipa
				enriched++
			}
		}
	}
	fmt.Printf("IPA enriched: %d entries\n", enriched)

	// ═══ Merge, dedup, exclude ═════════════════════════════════════════════

	merged := ingest.Merge(allEntries)
	fmt.Printf("After merge:  %d unique entries\n", len(merged))

	filtered, skipped, err := ingest.ExcludeExisting(merged, csvPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ExcludeExisting: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Excluded:     %d already in CSV\n", skipped)
	fmt.Printf("New entries:  %d\n", len(filtered))

	if len(filtered) == 0 {
		fmt.Println("\nNothing new to import.")
		return
	}

	// ═══ Write to CSV ═════════════════════════════════════════════════════

	if err := ingest.WriteCSV(csvPath, filtered, nextID, 0); err != nil {
		fmt.Fprintf(os.Stderr, "WriteCSV: %v\n", err)
		os.Exit(1)
	}

	// ═══ Summary ══════════════════════════════════════════════════════════

	polarityCounts := make(map[string]int)
	langCounts := make(map[string]int)
	sourceCounts := make(map[string]int)
	for _, e := range filtered {
		polarityCounts[e.Polarity]++
		langCounts[e.Lang]++
		sourceCounts[e.Source]++
	}

	fmt.Printf("\n══════════════════════════════════════════════════════════\n")
	fmt.Printf("  IMPORT SUMMARY\n")
	fmt.Printf("══════════════════════════════════════════════════════════\n")
	fmt.Printf("Appended %d new entries (word_id %d..%d)\n", len(filtered), nextID, nextID+len(filtered)-1)
	fmt.Printf("\nBy language:\n")
	for lang, count := range langCounts {
		fmt.Printf("  %s: %d\n", lang, count)
	}
	fmt.Printf("\nBy source:\n")
	for src, count := range sourceCounts {
		fmt.Printf("  %s: %d\n", src, count)
	}
	fmt.Printf("\nBy polarity:\n")
	for pol, count := range polarityCounts {
		fmt.Printf("  %s: %d\n", pol, count)
	}
	fmt.Printf("══════════════════════════════════════════════════════════\n")
}

func importDataset(all *[]ingest.Entry, name string, fn func() ([]ingest.Entry, ingest.Result, error)) {
	entries, res, err := fn()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%-16s SKIP: %v\n", name+":", err)
		return
	}
	fmt.Printf("%-16s %d entries (polarity: %v)\n", name+":", res.Total, res.ByPolarity)
	*all = append(*all, entries...)
}

func findNextWordID(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 100000, nil
		}
		return 0, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.LazyQuotes = true
	r.FieldsPerRecord = -1

	header, err := r.Read()
	if err != nil {
		return 100000, nil
	}

	idIdx := -1
	for i, h := range header {
		if h == "word_id" {
			idIdx = i
			break
		}
	}
	if idIdx < 0 {
		return 0, fmt.Errorf("no word_id column found")
	}

	maxID := 0
	for {
		row, err := r.Read()
		if err != nil {
			break
		}
		if len(row) <= idIdx {
			continue
		}
		id, err := strconv.Atoi(row[idIdx])
		if err != nil {
			continue
		}
		if id > maxID {
			maxID = id
		}
	}

	return maxID + 1, nil
}
