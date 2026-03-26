package discover

import (
	"fmt"
	"io"
	"os"
	"time"

	"github.com/kak/lex-sentiment/pkg/seed"
	"github.com/kak/lex-sentiment/pkg/sentiment"
)

// ImportConfig extends Config with dump-specific settings.
type ImportConfig struct {
	Config
	DumpPath  string // path to .xml or .xml.bz2 Wiktionary dump
	BatchSize int    // flush to CSV every N new words (default: 500)
}

// RunImport processes a local Wiktionary XML dump using the same classify /
// reconcile / write pipeline as Run, but without any network calls or rate limits.
//
// Progress is printed every BatchSize new words.
func RunImport(cfg ImportConfig, existingRoots []seed.Root, existingWords []seed.Word) (*Stats, error) {
	out := cfg.Output
	if out == nil {
		out = os.Stdout
	}
	logf := func(format string, args ...any) {
		fmt.Fprintf(out, format+"\n", args...)
	}

	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 500
	}

	start := time.Now()
	stats := &Stats{}

	allRoots := append([]seed.Root(nil), existingRoots...)
	allWords := append([]seed.Word(nil), existingWords...)

	known := map[string]bool{}
	for _, w := range existingWords {
		known[w.Norm+"_"+w.Lang] = true
	}

	var pendingRoots []seed.Root
	var pendingWords []seed.Word
	var pendingStaged []StagedWord
	var pagesScanned int

	flush := func() {
		if cfg.DryRun {
			for _, r := range pendingRoots {
				logf("  [DRY] ROOT  id=%-4d str=%-12q origin=%s", r.RootID, r.RootStr, r.Origin)
			}
			for _, w := range pendingWords {
				d := sentiment.Decode(w.Sentiment)
				logf("  [DRY] WORD  %-20q [%s] root=%-4d polarity=%s intensity=%s",
					w.Word, w.Lang, w.RootID, d["polarity"], d["intensity"])
			}
		} else {
			if err := Flush(pendingRoots, pendingWords, cfg.RootsPath, cfg.WordsPath); err != nil {
				logf("  warning: flush: %v", err)
			}
			if err := WriteStagedCSV(pendingStaged, StagedPath(cfg.OutDir)); err != nil {
				logf("  warning: staged: %v", err)
			}
		}
		pendingRoots = nil
		pendingWords = nil
		pendingStaged = nil
	}

	scanErr := ScanDump(cfg.DumpPath, func(page WikiPage) error {
		if cfg.Limit > 0 && stats.WordsAdded >= cfg.Limit {
			return io.EOF // stop scanning
		}

		pagesScanned++

		entries := ParseDumpPage(page, cfg.Langs)

		for i := range entries {
			entry := &entries[i]

			rootID, rootStr, isNewRoot := resolveRoot(entry, entry.Lang, allRoots, allWords)

			if isNewRoot {
				origin := entry.AncestorLang
				if origin == "" {
					origin = "UNKNOWN"
				}
				newRoot := seed.Root{
					RootID:       rootID,
					RootStr:      rootStr,
					Origin:       origin,
					MeaningEN:    firstDef(entry.Definitions, entry.Word),
					Notes:        "auto-imported from Wiktionary dump",
					ParentRootID: 0,
				}
				allRoots = append(allRoots, newRoot)
				pendingRoots = append(pendingRoots, newRoot)
				stats.RootsAdded++
				if cfg.Verbose {
					logf("  + ROOT  id=%-4d str=%-12q origin=%s", rootID, rootStr, origin)
				}
			}

			entryNorm := PhoneticNorm(entry.Word)
			if !known[entryNorm+"_"+entry.Lang] && isValidWord(entry.Word) {
				score := classifyBest(rootID, entry, allWords)
				if score.Confidence >= confidenceThreshold {
					if w, ok := makeWord(rootID, entry.Word, entry.Lang, entryNorm, score, allWords); ok {
						allWords = append(allWords, w)
						known[entryNorm+"_"+entry.Lang] = true
						pendingWords = append(pendingWords, w)
						stats.WordsAdded++
						if cfg.Verbose {
							d := sentiment.Decode(w.Sentiment)
							logf("  + WORD  %-20q [%s] root=%-4d pol=%-8s conf=%.0f%% [%s]",
								entry.Word, entry.Lang, rootID, d["polarity"], score.Confidence*100, score.Source)
						}
					}
				} else {
					pendingStaged = append(pendingStaged, StagedWord{
						Word:           entry.Word,
						Lang:           entry.Lang,
						RootStr:        rootStr,
						ProposedRootID: rootID,
						Score:          score,
						Definition:     firstDef(entry.Definitions, ""),
					})
					stats.WordsStaged++
				}
			} else {
				stats.WordsSkipped++
			}

			// Process translations from this entry.
			for _, trans := range entry.Translations {
				if !isTargetLang(trans.Lang, cfg.Langs) {
					continue
				}
				if !isValidWord(trans.Word) {
					continue
				}
				transNorm := PhoneticNorm(trans.Word)
				transKey := transNorm + "_" + trans.Lang
				if known[transKey] {
					continue
				}
				score := classifyBest(rootID, nil, allWords)
				if score.Confidence >= confidenceThreshold {
					if w, ok := makeWord(rootID, trans.Word, trans.Lang, transNorm, score, allWords); ok {
						allWords = append(allWords, w)
						known[transKey] = true
						pendingWords = append(pendingWords, w)
						stats.WordsAdded++
						if cfg.Verbose {
							d := sentiment.Decode(w.Sentiment)
							logf("  + WORD  %-20q [%s] root=%-4d pol=%-8s conf=%.0f%% [propagation]",
								trans.Word, trans.Lang, rootID, d["polarity"], score.Confidence*100)
						}
					}
				} else {
					pendingStaged = append(pendingStaged, StagedWord{
						Word:           trans.Word,
						Lang:           trans.Lang,
						RootStr:        rootStr,
						ProposedRootID: rootID,
						Score:          score,
					})
					stats.WordsStaged++
				}
			}
		}

		if len(pendingWords) >= cfg.BatchSize {
			flush()
			elapsed := time.Since(start).Seconds()
			logf("  [progress] pages=%d added=%d staged=%d skipped=%d elapsed=%.0fs",
				pagesScanned, stats.WordsAdded, stats.WordsStaged, stats.WordsSkipped, elapsed)
		}

		return nil
	})

	if scanErr != nil {
		return nil, fmt.Errorf("scan dump: %w", scanErr)
	}

	flush()
	stats.WordsExplored = pagesScanned
	stats.Duration = time.Since(start)
	return stats, nil
}
