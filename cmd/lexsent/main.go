// lexsent — Universal Morpheme Coordinate System CLI
//
// Usage:
//
//	lexsent build    [--roots PATH] [--words PATH] [--out PATH]
//	lexsent lookup   <word>
//	lexsent cognates <word>
//	lexsent etymo    <word>
//	lexsent analyze  <text>
//	lexsent tokenize <text>
//	lexsent stats    [--productive]
//	lexsent serve    [--port PORT]
//	lexsent discover [--expand | --seed words] [--lang CODES] [--depth N] [--limit N] [--dry-run]
//	lexsent import   --dump PATH [--lang CODES] [--limit N] [--dry-run] [--verbose]
package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/kak/umcs/pkg/analyze"
	"github.com/kak/umcs/pkg/api"
	"github.com/kak/umcs/pkg/discover"
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
	"github.com/kak/umcs/pkg/tokenizer"
)

const defaultLexicon = "lexicon.lsdb"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "build":
		cmdBuild(os.Args[2:])
	case "lookup":
		cmdLookup(os.Args[2:])
	case "cognates":
		cmdCognates(os.Args[2:])
	case "etymo":
		cmdEtymo(os.Args[2:])
	case "stats":
		cmdStats(os.Args[2:])
	case "analyze":
		cmdAnalyze(os.Args[2:])
	case "tokenize":
		cmdTokenize(os.Args[2:])
	case "serve":
		cmdServe(os.Args[2:])
	case "discover":
		cmdDiscover(os.Args[2:])
	case "import":
		cmdImport(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		usage()
		os.Exit(1)
	}
}

// --- build ---

func cmdBuild(args []string) {
	rootsPath := "data/roots.csv"
	wordsPath := "data/words.csv"
	outPath := defaultLexicon

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--roots":
			rootsPath = args[i+1]; i++
		case "--words":
			wordsPath = args[i+1]; i++
		case "--out":
			outPath = args[i+1]; i++
		}
	}

	roots, err := seed.LoadRoots(rootsPath)
	die(err)
	words, err := seed.LoadWords(wordsPath)
	die(err)

	stats, err := lexdb.Build(roots, words, outPath)
	die(err)

	fmt.Printf("Built %s\n", outPath)
	fmt.Printf("  Roots:  %d\n", stats.RootCount)
	fmt.Printf("  Words:  %d\n", stats.WordCount)
	fmt.Printf("  Langs:  %s\n", stats.Langs())
	fmt.Printf("  Heap:   %d bytes\n", stats.HeapSize)
	fmt.Printf("  Size:   %d bytes (%.1f KB)\n", stats.FileSize, float64(stats.FileSize)/1024)
}

// --- lookup ---

func cmdLookup(args []string) {
	if len(args) == 0 {
		fatalf("usage: lexsent lookup <word>")
	}
	word := args[0]
	lexPath := defaultLexicon
	for i := 1; i < len(args); i++ {
		if args[i] == "--lexicon" {
			lexPath = args[i+1]; i++
		}
	}

	lex := loadLex(lexPath)
	w := lex.LookupWord(word)
	if w == nil {
		fmt.Printf("not found: %q\n", word)
		os.Exit(1)
	}

	root := lex.LookupRoot(w.RootID)
	sent := sentiment.Decode(w.Sentiment)

	fmt.Printf("Word:      %s\n", lex.WordStr(w))
	fmt.Printf("Word ID:   %d (0x%08X)\n", w.WordID, w.WordID)
	fmt.Printf("Root ID:   %d\n", w.RootID)
	if root != nil {
		fmt.Printf("Root:      %s (%s — %s)\n", lex.RootStr(root), lex.RootOrigin(root), lex.RootMeaning(root))
		fmt.Printf("Langs:     %s\n", strings.Join(lex.LangCoverage(root.LangCoverage), ", "))
	}
	fmt.Printf("Lang:      %s\n", lexdb.LangName(w.Lang))
	fmt.Printf("Polarity:  %s\n", sent["polarity"])
	fmt.Printf("Intensity: %s\n", sent["intensity"])
	fmt.Printf("Role:      %s\n", sent["role"])
	fmt.Printf("Domain:    %s\n", sent["domain"])
	if sent["flags"] != "" {
		fmt.Printf("Flags:     %s\n", sent["flags"])
	}
	fmt.Printf("FreqRank:  %d\n", w.FreqRank)

	cognates := lex.Cognates(w.WordID)
	if len(cognates) > 1 {
		fmt.Printf("Cognates:\n")
		for _, c := range cognates {
			if c.WordID == w.WordID {
				continue
			}
			cs := sentiment.Decode(c.Sentiment)
			fmt.Printf("  %-20s [%s] polarity=%-9s intensity=%s\n",
				lex.WordStr(&c), lexdb.LangName(c.Lang), cs["polarity"], cs["intensity"])
		}
	}
}

// --- cognates ---

func cmdCognates(args []string) {
	if len(args) == 0 {
		fatalf("usage: lexsent cognates <word>")
	}
	lex := loadLex(defaultLexicon)
	w := lex.LookupWord(args[0])
	if w == nil {
		fatalf("not found: %q", args[0])
	}
	root := lex.LookupRoot(w.RootID)
	if root != nil {
		fmt.Printf("Root: %s (ID=%d, %s — %s)\n",
			lex.RootStr(root), root.RootID, lex.RootOrigin(root), lex.RootMeaning(root))
	}
	for _, c := range lex.Cognates(w.WordID) {
		fmt.Printf("  %-20s [%s]  word_id=%d\n", lex.WordStr(&c), lexdb.LangName(c.Lang), c.WordID)
	}
}

// --- etymo ---

func cmdEtymo(args []string) {
	if len(args) == 0 {
		fatalf("usage: lexsent etymo <word>")
	}
	lex := loadLex(defaultLexicon)
	w := lex.LookupWord(args[0])
	if w == nil {
		fatalf("not found: %q", args[0])
	}
	chain := lex.EtymologyChain(w.RootID)
	fmt.Printf("Etymology chain for %q:\n", args[0])
	for i, r := range chain {
		indent := strings.Repeat("  ", i)
		arrow := ""
		if i > 0 {
			arrow = "→ "
		}
		fmt.Printf("%s%s%s (ID=%d, %s: %s)\n",
			indent, arrow, lex.RootStr(&r), r.RootID, lex.RootOrigin(&r), lex.RootMeaning(&r))
	}
}

// --- stats ---

func cmdStats(args []string) {
	lexPath := defaultLexicon
	productive := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--lexicon":
			lexPath = args[i+1]; i++
		case "--productive":
			productive = true
		}
	}
	lex := loadLex(lexPath)
	s := lex.Stats
	fmt.Printf("Lexicon:   %s\n", lexPath)
	fmt.Printf("Roots:     %d\n", s.RootCount)
	fmt.Printf("Words:     %d\n", s.WordCount)
	fmt.Printf("Langs:     %s\n", strings.Join(lex.LangCoverage(s.LangFlags), " "))
	fmt.Printf("Heap:      %d bytes\n", s.HeapSize)
	fmt.Printf("File:      %d bytes (%.1f KB)\n", s.FileSize, float64(s.FileSize)/1024)
	fmt.Printf("Checksum:  0x%08X\n", s.Checksum)

	if productive {
		type score struct {
			root  string
			id    uint32
			words int
			langs int
		}
		var scores []score
		for _, r := range lex.Roots {
			langCount := 0
			for i := uint32(0); i < 11; i++ {
				if r.LangCoverage&(1<<i) != 0 {
					langCount++
				}
			}
			scores = append(scores, score{
				root:  lex.RootStr(&r),
				id:    r.RootID,
				words: int(r.WordCount),
				langs: langCount,
			})
		}
		// Sort by words×langs descending
		for i := range scores {
			for j := i + 1; j < len(scores); j++ {
				si := scores[i].words * scores[i].langs
				sj := scores[j].words * scores[j].langs
				if sj > si {
					scores[i], scores[j] = scores[j], scores[i]
				}
			}
		}
		fmt.Printf("\nMost productive roots (words × languages):\n")
		for i, s := range scores {
			if i >= 15 {
				break
			}
			fmt.Printf("  %-12s  words=%d  langs=%d  score=%d\n",
				s.root, s.words, s.langs, s.words*s.langs)
		}
	}
}

// --- analyze ---

func cmdAnalyze(args []string) {
	if len(args) == 0 {
		fatalf("usage: lexsent analyze <text>")
	}
	text := strings.Join(args, " ")
	lexPath := defaultLexicon
	for i := 0; i < len(args)-1; i++ {
		if args[i] == "--lexicon" {
			lexPath = args[i+1]
		}
	}

	lex := loadLex(lexPath)
	r := analyze.Analyze(lex, text)

	fmt.Printf("Analyzing: %q\n\n", text)
	for _, t := range r.Tokens {
		if !t.Found {
			fmt.Printf("  %-22s  [not found]\n", t.Surface)
			continue
		}
		modifier := ""
		if t.Negated {
			modifier = " [negated]"
		}
		if t.Amplified {
			modifier = " [×2 amplified]"
		}
		if t.Role == "NEGATION_MARKER" || t.Role == "CONNECTOR" {
			fmt.Printf("  %-22s  [%s, root=%s]\n", t.Surface, t.Role, t.RootStr)
			continue
		}
		fmt.Printf("  %-22s  polarity=%-9s intensity=%-8s root=%-10s weight=%+d%s\n",
			t.Surface, t.Polarity, t.Intensity, t.RootStr, t.Weight, modifier)
	}

	fmt.Printf("\nResult:  %d/%d tokens matched\n", r.Matched, r.Total)
	fmt.Printf("Score:   %+d\n", r.TotalScore)
	fmt.Printf("Verdict: %s\n", r.Verdict)
}

// --- tokenize ---

func cmdTokenize(args []string) {
	if len(args) == 0 {
		fatalf("usage: lexsent tokenize <text>")
	}
	text := strings.Join(args, " ")
	lex := loadLex(defaultLexicon)

	tokens := tokenizer.Tokenize(lex, text)
	fmt.Printf("Morpheme token sequence for: %q\n\n", text)
	fmt.Printf("  %-22s  %-10s  %-10s  %s\n", "surface", "root_id", "word_id", "sentiment_bits")
	fmt.Printf("  %s\n", strings.Repeat("-", 65))
	for _, t := range tokens {
		if !t.Known {
			fmt.Printf("  %-22s  %-10s  %-10s  %s\n", t.Surface, "?", "?", "?")
			continue
		}
		fmt.Printf("  %-22s  %-10d  %-10d  0x%08X\n", t.Surface, t.RootID, t.WordID, t.Sentiment)
	}

	// Show root_id sequence (cross-linguistic canonical form)
	rootIDs := make([]string, len(tokens))
	for i, t := range tokens {
		if t.Known {
			rootIDs[i] = fmt.Sprintf("%d", t.RootID)
		} else {
			rootIDs[i] = "?"
		}
	}
	fmt.Printf("\nRoot ID sequence: [%s]\n", strings.Join(rootIDs, ", "))
}

// --- serve ---

func cmdServe(args []string) {
	port := "8080"
	lexPath := defaultLexicon
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--port":
			port = args[i+1]; i++
		case "--lexicon":
			lexPath = args[i+1]; i++
		}
	}
	lex := loadLex(lexPath)
	srv := api.New(lex)
	die(srv.Listen(":" + port))
}

// --- helpers ---

func loadLex(path string) *lexdb.Lexicon {
	lex, err := lexdb.Load(path)
	if err != nil {
		fatalf("load lexicon: %v", err)
	}
	return lex
}

func die(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

// --- discover ---

func cmdDiscover(args []string) {
	expand := false
	seedWords := ""
	langs := "EN,PT,ES,IT,DE,FR,NL"
	depth := 2
	limit := 200
	dryRun := false
	outDir := "data"
	rootsPath := "data/roots.csv"
	wordsPath := "data/words.csv"
	verbose := false

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--expand":
			expand = true
		case "--seed":
			seedWords = args[i+1]; i++
		case "--lang":
			langs = args[i+1]; i++
		case "--depth":
			if n, err := strconv.Atoi(args[i+1]); err == nil {
				depth = n
			}
			i++
		case "--limit":
			if n, err := strconv.Atoi(args[i+1]); err == nil {
				limit = n
			}
			i++
		case "--dry-run":
			dryRun = true
		case "--out":
			outDir = args[i+1]; i++
		case "--roots":
			rootsPath = args[i+1]; i++
		case "--words":
			wordsPath = args[i+1]; i++
		case "--verbose":
			verbose = true
		}
	}

	if !expand && seedWords == "" {
		fatalf("discover: specify --expand or --seed word1,word2,...")
	}

	roots, err := seed.LoadRoots(rootsPath)
	die(err)
	words, err := seed.LoadWords(wordsPath)
	die(err)

	// Build seed list.
	var seeds []string
	if expand {
		// Use all English words from the existing lexicon as seeds.
		for _, w := range words {
			if w.Lang == "EN" {
				seeds = append(seeds, strings.ToLower(w.Word))
			}
		}
	} else {
		for _, s := range strings.Split(seedWords, ",") {
			if t := strings.TrimSpace(s); t != "" {
				seeds = append(seeds, strings.ToLower(t))
			}
		}
	}

	if len(seeds) == 0 {
		fatalf("discover: no seeds found")
	}

	langList := strings.Split(langs, ",")

	cfg := discover.Config{
		Seeds:     seeds,
		Langs:     langList,
		MaxDepth:  depth,
		Limit:     limit,
		DryRun:    dryRun,
		OutDir:    outDir,
		RootsPath: rootsPath,
		WordsPath: wordsPath,
		Verbose:   verbose,
	}

	fmt.Printf("lexsent discover\n")
	fmt.Printf("  Seeds:  %d words\n", len(seeds))
	fmt.Printf("  Langs:  %s\n", langs)
	fmt.Printf("  Depth:  %d\n", depth)
	fmt.Printf("  Limit:  %d new words\n", limit)
	if dryRun {
		fmt.Println("  Mode:   DRY RUN (no files modified)")
	} else {
		fmt.Println("  Mode:   LIVE (will append to CSVs)")
	}
	fmt.Println()

	stats, err := discover.Run(cfg, roots, words)
	die(err)

	fmt.Printf("\nCompleted in %.1fs\n", stats.Duration.Seconds())
	fmt.Printf("  Explored:  %d\n", stats.WordsExplored)
	fmt.Printf("  Added:     %d words, %d roots\n", stats.WordsAdded, stats.RootsAdded)
	fmt.Printf("  Staged:    %d (low-confidence → data/staged.csv)\n", stats.WordsStaged)
	fmt.Printf("  Skipped:   %d\n", stats.WordsSkipped)
	fmt.Printf("  Errors:    %d\n", stats.Errors)

	if !dryRun && stats.WordsAdded > 0 {
		fmt.Println("\nRebuilding lexicon...")
		newRoots, err := seed.LoadRoots(rootsPath)
		die(err)
		newWords, err := seed.LoadWords(wordsPath)
		die(err)
		bs, err := lexdb.Build(newRoots, newWords, defaultLexicon)
		die(err)
		fmt.Printf("  Lexicon: %d roots, %d words (%.1f KB)\n",
			bs.RootCount, bs.WordCount, float64(bs.FileSize)/1024)
	}
}

// --- import ---

func cmdImport(args []string) {
	dumpPath := ""
	langs := "EN,PT,ES,IT,DE,FR,NL"
	limit := 0
	dryRun := false
	outDir := "data"
	rootsPath := "data/roots.csv"
	wordsPath := "data/words.csv"
	verbose := false
	batchSize := 500
	allowNewRoots := false

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--dump":
			dumpPath = args[i+1]; i++
		case "--lang":
			langs = args[i+1]; i++
		case "--limit":
			if n, err := strconv.Atoi(args[i+1]); err == nil {
				limit = n
			}
			i++
		case "--dry-run":
			dryRun = true
		case "--out":
			outDir = args[i+1]; i++
		case "--roots":
			rootsPath = args[i+1]; i++
		case "--words":
			wordsPath = args[i+1]; i++
		case "--verbose":
			verbose = true
		case "--batch":
			if n, err := strconv.Atoi(args[i+1]); err == nil {
				batchSize = n
			}
			i++
		case "--allow-new-roots":
			allowNewRoots = true
		}
	}

	if dumpPath == "" {
		fatalf("import: --dump PATH is required")
	}

	roots, err := seed.LoadRoots(rootsPath)
	die(err)
	words, err := seed.LoadWords(wordsPath)
	die(err)

	langList := strings.Split(langs, ",")

	cfg := discover.ImportConfig{
		Config: discover.Config{
			Langs:     langList,
			Limit:     limit,
			DryRun:    dryRun,
			OutDir:    outDir,
			RootsPath: rootsPath,
			WordsPath: wordsPath,
			Verbose:   verbose,
		},
		DumpPath:      dumpPath,
		BatchSize:     batchSize,
		AllowNewRoots: allowNewRoots,
	}

	fmt.Printf("lexsent import\n")
	fmt.Printf("  Dump:   %s\n", dumpPath)
	fmt.Printf("  Langs:  %s\n", langs)
	if limit > 0 {
		fmt.Printf("  Limit:  %d new words\n", limit)
	} else {
		fmt.Printf("  Limit:  unlimited\n")
	}
	if dryRun {
		fmt.Println("  Mode:   DRY RUN (no files modified)")
	} else {
		fmt.Println("  Mode:   LIVE (will append to CSVs)")
	}
	fmt.Println()

	stats, err := discover.RunImport(cfg, roots, words)
	die(err)

	fmt.Printf("\nCompleted in %.1fs\n", stats.Duration.Seconds())
	fmt.Printf("  Pages scanned: %d\n", stats.WordsExplored)
	fmt.Printf("  Added:         %d words, %d roots\n", stats.WordsAdded, stats.RootsAdded)
	fmt.Printf("  Staged:        %d (low-confidence → data/staged.csv)\n", stats.WordsStaged)
	fmt.Printf("  Skipped:       %d\n", stats.WordsSkipped)

	if !dryRun && stats.WordsAdded > 0 {
		fmt.Println("\nRebuilding lexicon...")
		newRoots, err := seed.LoadRoots(rootsPath)
		die(err)
		newWords, err := seed.LoadWords(wordsPath)
		die(err)
		bs, err := lexdb.Build(newRoots, newWords, defaultLexicon)
		die(err)
		fmt.Printf("  Lexicon: %d roots, %d words (%.1f KB)\n",
			bs.RootCount, bs.WordCount, float64(bs.FileSize)/1024)
	}
}

func usage() {
	fmt.Println(`lexsent — Universal Morpheme Coordinate System

Commands:
  build    [--roots PATH] [--words PATH] [--out PATH]
  lookup   <word>
  cognates <word>
  etymo    <word>
  analyze  <text>
  tokenize <text>
  stats    [--productive]
  serve    [--port PORT]
  discover [--expand | --seed word1,word2] [--lang PT,EN,...] [--depth N] [--limit N] [--dry-run] [--verbose]
  import   --dump PATH.xml.bz2 [--lang PT,EN,...] [--limit N] [--dry-run] [--verbose] [--batch N]`)
}
