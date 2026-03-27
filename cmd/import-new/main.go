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
	// Allow override via env or fallback to working directory layout
	if env := os.Getenv("UMCS_ROOT"); env != "" {
		base = env
	} else {
		// Detect project root by looking for data/ relative to cwd
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

	// 1. Read existing CSV to find max word_id
	nextID, err := findNextWordID(csvPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error reading existing CSV: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Existing CSV: next word_id = %d\n\n", nextID)

	var allEntries []ingest.Entry

	// 2. Import OpLexicon v3.0 (PT)
	{
		path := filepath.Join(extDir, "OpLexicon_v3.0.txt")
		entries, res, err := ingest.ImportOpLexicon(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "OpLexicon: %v\n", err)
		} else {
			fmt.Printf("OpLexicon:   %d entries (polarity: %v)\n", res.Total, res.ByPolarity)
			allEntries = append(allEntries, entries...)
		}
	}

	// 3. Import SentiLex-PT02 (PT)
	{
		path := filepath.Join(extDir, "SentiLex-PT02.txt")
		entries, res, err := ingest.ImportSentiLex(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "SentiLex: %v\n", err)
		} else {
			fmt.Printf("SentiLex:    %d entries (polarity: %v)\n", res.Total, res.ByPolarity)
			allEntries = append(allEntries, entries...)
		}
	}

	// 4. Import Lexique383 (FR)
	{
		path := filepath.Join(extDir, "Lexique383.tsv")
		entries, res, err := ingest.ImportLexique383(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Lexique383: %v\n", err)
		} else {
			fmt.Printf("Lexique383:  %d entries (polarity: %v)\n", res.Total, res.ByPolarity)
			allEntries = append(allEntries, entries...)
		}
	}

	// 5. Import MPQA (EN)
	{
		path := filepath.Join(extDir, "mpqa-subj.tff")
		entries, res, err := ingest.ImportMPQA(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "MPQA: %v\n", err)
		} else {
			fmt.Printf("MPQA:        %d entries (polarity: %v)\n", res.Total, res.ByPolarity)
			allEntries = append(allEntries, entries...)
		}
	}

	fmt.Printf("\nTotal raw entries: %d\n", len(allEntries))

	// 6. Enrich with IPA from IPA-dict files
	ipaLangs := map[string]string{
		"PT": "pt",
		"FR": "fr",
		"EN": "en",
	}
	ipaMaps := make(map[string]map[string]string)
	for lang, code := range ipaLangs {
		ipaPath := filepath.Join(extDir, fmt.Sprintf("IPA-dict-%s.txt", code))
		m, err := ingest.ImportIPADict(ipaPath, lang)
		if err != nil {
			fmt.Fprintf(os.Stderr, "IPA-dict-%s: %v\n", code, err)
			continue
		}
		ipaMaps[lang] = m
		fmt.Printf("IPA-dict-%s:  %d pronunciations loaded\n", code, len(m))
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

	// 7. Merge duplicates
	merged := ingest.Merge(allEntries)
	fmt.Printf("After merge:  %d unique entries\n", len(merged))

	// 8. Exclude existing
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

	// 9. Append to CSV
	if err := ingest.WriteCSV(csvPath, filtered, nextID, 0); err != nil {
		fmt.Fprintf(os.Stderr, "WriteCSV: %v\n", err)
		os.Exit(1)
	}

	// 10. Summary
	polarityCounts := make(map[string]int)
	langCounts := make(map[string]int)
	for _, e := range filtered {
		polarityCounts[e.Polarity]++
		langCounts[e.Lang]++
	}

	fmt.Printf("\n=== Import Summary ===\n")
	fmt.Printf("Appended %d new entries (word_id %d..%d)\n", len(filtered), nextID, nextID+len(filtered)-1)
	fmt.Printf("By language:\n")
	for lang, count := range langCounts {
		fmt.Printf("  %s: %d\n", lang, count)
	}
	fmt.Printf("By polarity:\n")
	for pol, count := range polarityCounts {
		fmt.Printf("  %s: %d\n", pol, count)
	}
}

// findNextWordID reads the CSV and returns max(word_id) + 1.
func findNextWordID(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 100000, nil // default starting ID
		}
		return 0, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.LazyQuotes = true
	r.FieldsPerRecord = -1 // variable fields

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
