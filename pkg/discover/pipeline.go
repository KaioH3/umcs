package discover

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/kak/umcs/pkg/morpheme"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
)

// Config controls a discovery run.
type Config struct {
	Seeds     []string // initial words to expand from
	Langs     []string // target languages to collect (e.g. ["EN","PT","ES"])
	MaxDepth  int      // BFS depth (1 = seeds only, 2 = seeds + their translations)
	Limit     int      // max new words to add in this run
	DryRun    bool     // if true, print results but do not modify CSVs
	OutDir    string   // directory containing roots.csv, words.csv, checkpoints
	RootsPath string   // full path to roots.csv
	WordsPath string   // full path to words.csv
	Verbose   bool
	Output    io.Writer // progress sink (defaults to os.Stdout)
}

// Stats summarises a completed discovery run.
type Stats struct {
	WordsExplored int
	WordsAdded    int
	RootsAdded    int
	WordsSkipped  int
	WordsStaged   int // low-confidence, written to staged.csv
	Errors        int
	Duration      time.Duration
}

// confidenceThreshold is the minimum confidence to auto-accept a word.
const confidenceThreshold = 0.60

// blockedRoots are root IDs whose translations must not be auto-accepted.
// These are taboo/vulgar/profanity roots: their Wiktionary translations
// often include polysemous words that have primary meanings in other domains
// (e.g. PT "comer"=eat has a vulgar sense, but is not a vulgar word).
var blockedRoots = map[uint32]bool{
	84: true, // fod  — sexual vulgar
	85: true, // merd — excrement vulgar
	86: true, // fod intensifier
}

// bfsItem is one element of the BFS work queue.
type bfsItem struct {
	word  string
	lang  string
	depth int
}

// Run executes the automated discovery pipeline.
//
// Phase 1 — BFS: fetch Wiktionary entries for each seed, extract translations.
// Phase 2 — Classify: score new words via propagation → definition keywords → morphology.
// Phase 3 — Write: append high-confidence words to CSVs; low-confidence → staged.csv.
func Run(cfg Config, existingRoots []seed.Root, existingWords []seed.Word) (*Stats, error) {
	out := cfg.Output
	if out == nil {
		out = os.Stdout
	}
	logf := func(format string, args ...any) {
		fmt.Fprintf(out, format+"\n", args...)
	}

	start := time.Now()
	stats := &Stats{}

	// Working copies that grow as new entries are accepted.
	allRoots := append([]seed.Root(nil), existingRoots...)
	allWords := append([]seed.Word(nil), existingWords...)

	// Lookup index: "norm_LANG" → true (known words).
	known := map[string]bool{}
	for _, w := range existingWords {
		known[w.Norm+"_"+w.Lang] = true
	}

	// Pending writes (flushed every 50 new words).
	var pendingRoots []seed.Root
	var pendingWords []seed.Word
	var pendingStaged []StagedWord

	// Checkpoint for resumable runs.
	cpPath := CheckpointPath(cfg.OutDir)
	cp, err := LoadCheckpoint(cpPath)
	if err != nil {
		return nil, fmt.Errorf("checkpoint: %w", err)
	}

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
			if err := cp.Save(cpPath); err != nil {
				logf("  warning: checkpoint save: %v", err)
			}
		}
		pendingRoots = nil
		pendingWords = nil
		pendingStaged = nil
	}

	// Build initial BFS queue from seeds × langs.
	queue := make([]bfsItem, 0, len(cfg.Seeds)*len(cfg.Langs))
	visited := map[string]bool{}
	for _, seedWord := range cfg.Seeds {
		// For each seed, start with EN if available (best Wiktionary coverage).
		// Other langs queued at depth 1 to benefit from translations extracted at depth 0.
		seedNorm := strings.ToLower(strings.TrimSpace(seedWord))
		if seedNorm == "" {
			continue
		}
		enKey := seedNorm + "_EN"
		if !visited[enKey] {
			visited[enKey] = true
			queue = append(queue, bfsItem{word: seedNorm, lang: "EN", depth: 0})
		}
	}

	logf("Starting BFS: %d seeds, langs=[%s], depth=%d, limit=%d",
		len(cfg.Seeds), strings.Join(cfg.Langs, ","), cfg.MaxDepth, cfg.Limit)

	for len(queue) > 0 && stats.WordsAdded < cfg.Limit {
		item := queue[0]
		queue = queue[1:]

		fetchKey := item.word + "_" + item.lang

		if cp.IsProcessed(fetchKey) {
			stats.WordsSkipped++
			continue
		}

		stats.WordsExplored++

		// Fetch Wiktionary.
		entry, err := Fetch(item.word, item.lang)
		if err != nil {
			if cfg.Verbose {
				logf("  skip %q [%s]: %v", item.word, item.lang, err)
			}
			stats.Errors++
			// Only mark as processed for permanent errors (missing page).
			// Network/transient errors are not marked — next run will retry.
			if errors.Is(err, ErrPageNotFound) {
				cp.Mark(fetchKey)
			}
			continue
		}

		// Resolve root: if the seed word is already in the lexicon, use its root_id directly.
		// This avoids false root creation when expanding from existing seeds.
		rootID, rootStr, isNewRoot := resolveRoot(entry, item.lang, allRoots, allWords)

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
				Notes:        "auto-discovered via Wiktionary",
				ParentRootID: 0,
			}
			allRoots = append(allRoots, newRoot)
			pendingRoots = append(pendingRoots, newRoot)
			stats.RootsAdded++
			logf("  + ROOT  id=%-4d str=%-12q origin=%s", rootID, rootStr, origin)
		}

		// Try to classify and add the entry word itself.
		entryNorm := PhoneticNorm(entry.Word)
		if !known[entryNorm+"_"+item.lang] && isValidWord(entry.Word) {
			score := classifyBest(rootID, entry, allWords)
			if score.Confidence >= confidenceThreshold {
				if w, ok := makeWord(rootID, entry.Word, item.lang, entryNorm, score, allWords); ok {
					allWords = append(allWords, w)
					known[entryNorm+"_"+item.lang] = true
					pendingWords = append(pendingWords, w)
					stats.WordsAdded++
					d := sentiment.Decode(w.Sentiment)
					logf("  + WORD  %-20q [%s] root=%-4d pol=%-8s conf=%.0f%% [%s]",
						entry.Word, item.lang, rootID, d["polarity"], score.Confidence*100, score.Source)
				}
			} else {
				pendingStaged = append(pendingStaged, StagedWord{
					Word:           entry.Word,
					Lang:           item.lang,
					RootStr:        rootStr,
					ProposedRootID: rootID,
					Score:          score,
					Definition:     firstDef(entry.Definitions, ""),
				})
				stats.WordsStaged++
				if cfg.Verbose {
					logf("  ~ STAGED %-20q [%s] conf=%.0f%% [%s]",
						entry.Word, item.lang, score.Confidence*100, score.Source)
				}
			}
		} else {
			stats.WordsSkipped++
		}

		// Process translations → add cognates and queue for BFS expansion.
		for _, trans := range entry.Translations {
			if !isTargetLang(trans.Lang, cfg.Langs) {
				continue
			}
			if !isValidWord(trans.Word) {
				stats.WordsSkipped++
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
					d := sentiment.Decode(w.Sentiment)
					logf("  + WORD  %-20q [%s] root=%-4d pol=%-8s conf=%.0f%% [%s]",
						trans.Word, trans.Lang, rootID, d["polarity"], score.Confidence*100, score.Source)
				}
			} else {
				pendingStaged = append(pendingStaged, StagedWord{
					Word:           trans.Word,
					Lang:           trans.Lang,
					RootStr:        rootStr,
					ProposedRootID: rootID,
					Score:          score,
					Definition:     "",
				})
				stats.WordsStaged++
			}

			// Queue this translation for BFS expansion at next depth.
			if item.depth+1 < cfg.MaxDepth && stats.WordsAdded < cfg.Limit {
				nextKey := trans.Word + "_" + trans.Lang
				if !visited[nextKey] {
					visited[nextKey] = true
					queue = append(queue, bfsItem{
						word:  strings.ToLower(trans.Word),
						lang:  trans.Lang,
						depth: item.depth + 1,
					})
				}
			}
		}

		cp.Mark(fetchKey)

		if len(pendingWords) >= 50 {
			flush()
		}
	}

	flush() // final flush

	stats.Duration = time.Since(start)
	return stats, nil
}

// isValidWord returns false for entries that should not be stored as lexicon words:
//   - wikitext markup artifacts ([[...]] or {{...}})
//   - multi-word phrases (contain a space) — these are idioms, not morphemes
//   - single-character entries that are not CJK — too short to be meaningful
//     (single CJK ideograms are valid: 愛, 悲, 喜 each represent a complete morpheme)
func isValidWord(word string) bool {
	runes := []rune(word)
	if len(runes) == 0 {
		return false
	}
	if len(runes) == 1 && !IsCJK(runes[0]) {
		return false
	}
	if strings.ContainsAny(word, " \t") {
		return false
	}
	return !strings.Contains(word, "[[") &&
		!strings.Contains(word, "{{") &&
		!strings.Contains(word, "}}")
}

// classifyBest picks the highest-confidence score from all available classifiers.
// entry may be nil when classifying a translation (only propagation is used then).
// Words for blocked (taboo/vulgar) roots always return zero confidence so they
// land in staged.csv for manual review.
func classifyBest(rootID uint32, entry *Entry, allWords []seed.Word) Score {
	if blockedRoots[rootID] {
		return Score{Polarity: "NEUTRAL", Intensity: "NONE", Role: "EVALUATION",
			Confidence: 0, Source: "blocked-root"}
	}

	propScore := ScoreViaPropagation(rootID, allWords)

	if entry == nil {
		return propScore
	}

	defScore := ScoreViaDefinition(entry.Definitions)
	morphScore := ScoreViaMorphology(entry.Word)

	return BestScore(propScore, defScore, morphScore)
}

// makeWord constructs a seed.Word with the packed sentiment bitmask.
// Returns false if the word_id cannot be computed (variant overflow).
func makeWord(rootID uint32, word, lang, norm string, score Score, allWords []seed.Word) (seed.Word, bool) {
	variant := NextVariant(rootID, allWords)
	wordID, err := morpheme.MakeWordID(rootID, variant)
	if err != nil {
		// variant > 4095 (MaxVariant) — root has too many words. Log and skip.
		fmt.Fprintf(os.Stderr, "warning: variant overflow root_id=%d variant=%d word=%q dropped\n",
			rootID, variant, word)
		return seed.Word{}, false
	}
	pol := score.Polarity
	if pol == "" {
		pol = "NEUTRAL"
	}
	intensity := score.Intensity
	if intensity == "" {
		intensity = "NONE"
	}
	role := score.Role
	if role == "" {
		role = "EVALUATION"
	}
	packed, err := sentiment.Pack(pol, intensity, role, "GENERAL", nil)
	if err != nil {
		return seed.Word{}, false
	}
	return seed.Word{
		WordID:    wordID,
		RootID:    rootID,
		Variant:   variant,
		Word:      word,
		Lang:      lang,
		Norm:      norm,
		Sentiment: packed,
		FreqRank:  0,
		Flags:     0,
	}, true
}

// resolveRoot finds the root_id for entry.Word and also returns the root string.
//
// Priority:
//  1. Exact match against existing words (by phonetic norm + lang) → use their root_id.
//  2. Etymology ancestor from Wiktionary → stem → Assign (may create new root).
//  3. Fallback: Assign(phonetic norm of the word) → may create new root.
func resolveRoot(entry *Entry, lang string, allRoots []seed.Root, allWords []seed.Word) (rootID uint32, rootStr string, isNew bool) {
	// 1. Look up existing word in lexicon.
	entryNorm := PhoneticNorm(entry.Word)
	for _, w := range allWords {
		if PhoneticNorm(w.Word) == entryNorm && w.Lang == lang {
			// Find the root string from existing roots.
			for _, r := range allRoots {
				if r.RootID == w.RootID {
					return w.RootID, r.RootStr, false
				}
			}
			return w.RootID, entry.Word, false
		}
	}

	// 2. Use etymology ancestor.
	if entry.AncestorWord != "" {
		rs := StemAncestor(entry.AncestorWord)
		if rs != "" {
			id, isNew := Assign(rs, allRoots)
			return id, rs, isNew
		}
	}

	// 3. Fallback: assign by word norm (may create a new root).
	id, isNew := Assign(entryNorm, allRoots)
	return id, entryNorm, isNew
}

func isTargetLang(lang string, targets []string) bool {
	for _, t := range targets {
		if t == lang {
			return true
		}
	}
	return false
}

func firstDef(defs []string, fallback string) string {
	if len(defs) > 0 {
		return defs[0]
	}
	return fallback
}
