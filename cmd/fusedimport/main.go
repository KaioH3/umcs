package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/kak/umcs/pkg/fusedingest"
)

func main() {
	base := filepath.Dir(filepath.Dir(os.Args[0]))
	if env := os.Getenv("UMCS_ROOT"); env != "" {
		base = env
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

	fi := fusedingest.NewFusedImporter()

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║           FUSED IMPORT - Scientific Data Fusion              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// ═══ Phase 1: High-quality sentiment lexicons ════════════════════════

	fmt.Println("[1/8] Importing NRC VAD Lexicon (54k EN words)...")
	if err := fi.ImportNRCVAD(filepath.Join(extDir, "NRC-VAD", "NRC-VAD-Lexicon-v2.1", "NRC-VAD-Lexicon-v2.1.txt")); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ Done")
	}

	fmt.Println("[2/8] Importing AFINN-165 (3.3k EN words)...")
	if err := fi.ImportAFINN(filepath.Join(extDir, "AFINN-en-165.txt")); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ Done")
	}

	fmt.Println("[3/8] Importing SentiWordNet 3.0 (117k synsets)...")
	if err := fi.ImportSentiWordNet(filepath.Join(extDir, "SentiWordNet_3.0.0.txt")); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ Done")
	}

	fmt.Println("[4/8] Importing Warriner VAD norms (13.9k EN words)...")
	if err := fi.ImportWarrinerVAD(filepath.Join(extDir, "Warriner_VAD.csv")); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ Done")
	}

	fmt.Println("[5/8] Importing MPQA Subjectivity (6.9k EN words)...")
	if err := fi.ImportMPQA(filepath.Join(extDir, "mpqa-subj.tff")); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ Done")
	}

	fmt.Println("[6/8] Importing positive/negative word lists...")
	if err := fi.ImportPositiveNegative(
		filepath.Join(extDir, "positive-words.txt"),
		filepath.Join(extDir, "negative-words.txt"),
	); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ Done")
	}

	// ═══ Phase 2: Portuguese specific ═══════════════════════════════════

	fmt.Println("[7/8] Importing Portuguese sentiment lexicons...")
	if err := fi.ImportOpLexicon(filepath.Join(extDir, "OpLexicon_v3.0.txt")); err != nil {
		fmt.Printf("   ⚠ Warning (OpLexicon): %v\n", err)
	} else {
		fmt.Println("   ✓ OpLexicon v3.0 imported")
	}
	if err := fi.ImportSentiLex(filepath.Join(extDir, "SentiLex-PT02.txt")); err != nil {
		fmt.Printf("   ⚠ Warning (SentiLex): %v\n", err)
	} else {
		fmt.Println("   ✓ SentiLex-PT02 imported")
	}

	// ═══ Phase 3: Multilingual sentiment ════════════════════════════════

	fmt.Println("[8/8] Importing 81-language sentiment (167k+ words)...")
	if err := fi.ImportSentiment81Langs(filepath.Join(extDir, "sentiment-81langs", "sentiment-lexicons")); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ Sentiment-81langs imported")
	}

	// ═══ Phase 4: Additional sources ════════════════════════════════════

	fmt.Println()
	fmt.Println("[+] Importing NRC Emotion Lexicon...")
	if err := fi.ImportNRCEmotion(extDir); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ NRC Emotion imported")
	}

	fmt.Println()
	fmt.Println("[+] Importing CogNet (multilingual cognates)...")
	if err := fi.ImportCogNet(filepath.Join(extDir, "CogNet", "CogNet-v2.0.tsv")); err != nil {
		fmt.Printf("   ⚠ Warning: %v\n", err)
	} else {
		fmt.Println("   ✓ CogNet imported")
	}

	// ═══ Finalize and Export ════════════════════════════════════════════

	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                  FINALIZING FUSION                          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	fmt.Println()
	fmt.Println("Writing fused entries to CSV...")
	if err := fi.FinalizeAndExport(csvPath, nextID); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing CSV: %v\n", err)
		os.Exit(1)
	}

	res := fi.Result()
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                    IMPORT SUMMARY                            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Printf("  Total raw entries processed: %d\n", res.TotalEntries)
	fmt.Printf("  Unique fused words:          %d\n", res.UniqueWords)
	fmt.Printf("  High confidence (≥0.8):      %d\n", res.HighConf)
	fmt.Printf("  Low confidence (<0.5):      %d\n", res.LowConf)
	fmt.Println()
	fmt.Println("  By Language:")
	for lang, count := range res.ByLanguage {
		fmt.Printf("    %s: %d\n", lang, count)
	}
	fmt.Println()
	fmt.Println("  By Polarity:")
	for pol, count := range res.ByPolarity {
		fmt.Printf("    %s: %d\n", pol, count)
	}
	fmt.Println()
	fmt.Println("  By Source:")
	for src, count := range res.BySource {
		fmt.Printf("    %s: %d\n", src, count)
	}
	fmt.Println()
	fmt.Printf("  Word IDs assigned: %d..%d\n", nextID, nextID+res.UniqueWords-1)
	fmt.Println()
	fmt.Println("✓ Fused import complete!")
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
