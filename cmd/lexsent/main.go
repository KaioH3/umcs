// lexsent — Universal Morpheme Coordinate System CLI
//
// Usage:
//
//	lexsent build    [--roots PATH] [--words PATH] [--out PATH]
//	lexsent demo     [--lexicon PATH]
//	lexsent lookup   <word>
//	lexsent cognates <word>
//	lexsent etymo    <word>
//	lexsent analyze  <text>
//	lexsent tokenize <text>
//	lexsent stats    [--productive]
//	lexsent serve    [--port PORT]
//	lexsent discover [--expand | --seed words] [--lang CODES] [--depth N] [--limit N] [--dry-run] [--reset]
//	lexsent import   --dump PATH [--lang CODES] [--limit N] [--dry-run] [--verbose]
package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"os/exec"

	"github.com/kak/umcs/pkg/analyze"
	"github.com/kak/umcs/pkg/api"
	"github.com/kak/umcs/pkg/classify"
	"github.com/kak/umcs/pkg/discover"
	"github.com/kak/umcs/pkg/ga"
	"github.com/kak/umcs/pkg/infer"
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/morpheme"
	"github.com/kak/umcs/pkg/phon"
	"github.com/kak/umcs/pkg/rl"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
	"github.com/kak/umcs/pkg/tokenizer"
)

const defaultLexicon = "lexicon.umcs"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "build":
		cmdBuild(os.Args[2:])
	case "demo":
		cmdDemo(os.Args[2:])
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
	case "train":
		cmdTrain(os.Args[2:])
	case "predict":
		cmdPredict(os.Args[2:])
	case "feedback":
		cmdFeedback(os.Args[2:])
	case "evolve":
		cmdEvolve(os.Args[2:])
	case "export-c":
		cmdExportC(os.Args[2:])
	case "stage-review":
		cmdStageReview(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		usage()
		os.Exit(1)
	}
}

// --- demo ---

func cmdDemo(args []string) {
	lexPath := defaultLexicon
	for i := 0; i < len(args); i++ {
		if args[i] == "--lexicon" && i+1 < len(args) {
			lexPath = args[i+1]
			i++
		}
	}

	sep := strings.Repeat("═", 56)
	fmt.Println(sep)
	fmt.Println(" UMCS Demo — Universal Morpheme Coordinate System")
	fmt.Println(sep)
	fmt.Println()

	lex, err := lexdb.Load(lexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load lexicon: %v\n  → run 'lexsent build' first\n", err)
		os.Exit(1)
	}

	// ── [1] Lookup ───────────────────────────────────────────────────────────
	demoWord := "negative"
	w := lex.LookupWord(demoWord)
	if w == nil {
		// Try a fallback if negative is not in lexicon.
		for _, wr := range lex.Words {
			if wr.WordID != 0 {
				demoWord = lex.WordStr(&wr)
				w = &wr
				break
			}
		}
	}

	if w != nil {
		fmt.Printf("[1] LOOKUP: %q (%s)\n", lex.WordStr(w), lexdb.LangName(w.Lang))
		fmt.Printf("  word_id   = %d  (root_id=%d, variant=%d)\n",
			w.WordID, morpheme.RootOf(w.WordID), morpheme.VariantOf(w.WordID))

		root := lex.LookupRoot(w.RootID)
		if root != nil {
			fmt.Printf("  root      = %-10s (%s: %q)\n",
				lex.RootStr(root), lex.RootOrigin(root), lex.RootMeaning(root))
		}

		dec := sentiment.Decode(w.Sentiment)
		fmt.Printf("  polarity  = %-10s intensity=%-9s role=%s\n",
			dec["polarity"], dec["intensity"], dec["role"])
		fmt.Printf("  POS       = %-10s arousal=%-10s dominance=%s\n",
			dec["pos"], dec["arousal"], dec["dominance"])
		fmt.Printf("  AoA       = %-10s concrete=%-10s register=%s\n",
			dec["aoa"], dec["concreteness"], dec["flags"])

		tok := morpheme.Pack64(w.WordID, w.Sentiment, w.Flags)
		fmt.Printf("  Token64   = 0x%016X\n", uint64(tok))
		fmt.Println()

		// ── [2] Token64 decoded ──────────────────────────────────────────────
		fmt.Printf("[2] Token64 DECODED: 0x%016X\n", uint64(tok))
		_, pay := morpheme.Unpack64(tok)
		fmt.Printf("  root_id=%-6d variant=%-4d pos=%s  concrete=%d\n",
			morpheme.RootOf64(tok), morpheme.VariantOf64(tok),
			dec["pos"], boolBit(sentiment.IsConcrete(pay)))
		fmt.Printf("  polarity=%-10s intensity=%-9s role=%s\n",
			dec["polarity"], dec["intensity"], dec["role"])
		fmt.Printf("  arousal=%-11s dominance=%-9s aoa=%s\n",
			dec["arousal"], dec["dominance"], dec["aoa"])
		fmt.Println()

		// ── [3] Cognates ─────────────────────────────────────────────────────
		cognates := lex.Cognates(w.WordID)
		if len(cognates) > 1 {
			fmt.Printf("[3] COGNATES of %q (root_id=%d):\n", lex.WordStr(w), w.RootID)
			for _, c := range cognates {
				cs := sentiment.Decode(c.Sentiment)
				ctok := morpheme.Pack64(c.WordID, c.Sentiment, c.Flags)
				fmt.Printf("  %-6s %-20s (%d) → %-9s %-9s Token64=0x%016X\n",
					lexdb.LangName(c.Lang), lex.WordStr(&c), c.WordID,
					cs["polarity"], cs["intensity"], uint64(ctok))
			}
			fmt.Println()
		}
	}

	// ── [4] Sentiment analysis ───────────────────────────────────────────────
	analysisText := "this product is not terrible at all"
	fmt.Printf("[4] SENTIMENT ANALYSIS: %q\n", analysisText)
	result := analyze.Analyze(lex, analysisText)
	for _, t := range result.Tokens {
		if !t.Found {
			fmt.Printf("  %-22s [OOV]\n", t.Surface)
			continue
		}
		mod := ""
		if t.Negated {
			mod = " [negated]"
		} else if t.Amplified {
			mod = " [amplified]"
		}
		if t.Role == "NEGATION_MARKER" {
			fmt.Printf("  %-22s [%s, root=%s]\n", t.Surface, t.Role, t.RootStr)
			continue
		}
		fmt.Printf("  %-22s polarity=%-9s weight=%+d%s\n",
			t.Surface, t.Polarity, t.Weight, mod)
	}
	fmt.Printf("  Score: %+d  Verdict: %s\n\n", result.TotalScore, result.Verdict)

	// ── [5] IPA & Phonology ──────────────────────────────────────────────────
	fmt.Println("[5] IPA PRONUNCIATION & PHONOLOGY:")
	phonWords := []string{"negative", "can", "must", "liberdade", "terrible", "será"}
	for _, pw := range phonWords {
		wr := lex.LookupWord(pw)
		if wr == nil {
			continue
		}
		pron := lex.WordPron(wr)
		syl := phon.Syllables(wr.Flags)
		stress := phon.StressName(wr.Flags)
		val := phon.ValencyName(wr.Flags)
		fmt.Printf("  %-20s [%s]  IPA:%-20s  syl=%-2d stress=%-14s valency=%s\n",
			lex.WordStr(wr), lexdb.LangName(wr.Lang), pron, syl, stress, val)
	}
	fmt.Println()

	// ── [6] Semantic relations ───────────────────────────────────────────────
	fmt.Println("[6] SEMANTIC RELATIONS (antonym / hypernym / synonym):")
	relRoots := []string{"negative", "good", "terrible", "can", "all"}
	for _, rw := range relRoots {
		wr := lex.LookupWord(rw)
		if wr == nil {
			continue
		}
		root := lex.LookupRoot(wr.RootID)
		if root == nil {
			continue
		}
		antStr, hypStr, synStr := "—", "—", "—"
		if r := lex.Antonym(root); r != nil {
			antStr = fmt.Sprintf("%s(%d)", lex.RootStr(r), r.RootID)
		}
		if r := lex.Hypernym(root); r != nil {
			hypStr = fmt.Sprintf("%s(%d)", lex.RootStr(r), r.RootID)
		}
		if r := lex.Synonym(root); r != nil {
			synStr = fmt.Sprintf("%s(%d)", lex.RootStr(r), r.RootID)
		}
		fmt.Printf("  %-12s → ant:%-14s hyp:%-14s syn:%s\n",
			lex.RootStr(root), antStr, hypStr, synStr)
	}
	fmt.Println()

	// ── [7] Morphological inference ──────────────────────────────────────────
	fmt.Println("[7] MORPHOLOGICAL INFERENCE (pkg/infer):")
	inferCases := []struct{ word, lang string }{
		{"liberdade", "PT"},
		{"rapidamente", "PT"},
		{"happiness", "EN"},
		{"beautiful", "EN"},
		{"liberation", "EN"},
		{"Freiheit", "DE"},
	}
	for _, c := range inferCases {
		pos := infer.POSFromShape(c.word, c.lang)
		abstract := infer.IsAbstractFromShape(c.word, c.lang)
		posName := sentiment.Decode(pos)["pos"]
		abstract_ := ""
		if abstract {
			abstract_ = " + ABSTRACT"
		}
		fmt.Printf("  %-20s (%s) → POS=%-8s%s\n", c.word, c.lang, posName, abstract_)
	}
	fmt.Println()

	// ── [8] Multilingual token stream ────────────────────────────────────────
	multiText := "not terrible very good"
	fmt.Printf("[8] MULTILINGUAL TOKEN STREAM: %q\n", multiText)
	tokens := tokenizer.Tokenize(lex, multiText)
	fmt.Printf("  %-20s  %-10s  %-10s  %s\n", "surface", "root_id", "word_id", "Token64")
	fmt.Printf("  %s\n", strings.Repeat("-", 70))
	for _, t := range tokens {
		if !t.Known {
			fmt.Printf("  %-20s  %-10s  %-10s  %s\n", t.Surface, "?", "?", "?")
			continue
		}
		fmt.Printf("  %-20s  %-10d  %-10d  0x%016X\n",
			t.Surface, t.RootID, t.WordID, uint64(t.Token64))
	}
	fmt.Println()

	// ── [9] Etymology ────────────────────────────────────────────────────────
	if w != nil {
		fmt.Printf("[9] ETYMOLOGY CHAIN: %q\n", lex.WordStr(w))
		chain := lex.EtymologyChain(w.RootID)
		for i, r := range chain {
			indent := strings.Repeat("  ", i+1)
			arrow := "→ "
			if i == 0 {
				arrow = "  "
			}
			fmt.Printf("%s%s%s (ID=%d, %s: %s)\n",
				indent, arrow, lex.RootStr(&r), r.RootID, lex.RootOrigin(&r), lex.RootMeaning(&r))
		}
		fmt.Println()
	}

	// ── [10] Lexicon stats ───────────────────────────────────────────────────
	s := lex.Stats
	fmt.Println("[10] LEXICON STATS:")
	fmt.Printf("  %d roots | %d words | %d languages\n",
		s.RootCount, s.WordCount, countLangs(s.LangFlags))
	fmt.Printf("  Token64 capacity: ~%d million word_ids × 32 semantic bits\n",
		(1<<20)/1000*1000)
	fmt.Printf("  Binary size: %.1f KB (entire cross-lingual lexicon in memory)\n",
		float64(s.FileSize)/1024)
	fmt.Println()
}

func boolBit(b bool) int {
	if b {
		return 1
	}
	return 0
}

func countLangs(flags uint32) int {
	n := 0
	for flags != 0 {
		n += int(flags & 1)
		flags >>= 1
	}
	return n
}

// --- build ---

func cmdBuild(args []string) {
	rootsPath := "data/roots.csv"
	wordsPath := "data/words.csv"
	importedPath := ""
	outPath := defaultLexicon

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--roots":
			rootsPath = args[i+1]; i++
		case "--words":
			wordsPath = args[i+1]; i++
		case "--imported":
			importedPath = args[i+1]; i++
		case "--out":
			outPath = args[i+1]; i++
		}
	}

	roots, err := seed.LoadRoots(rootsPath)
	die(err)
	words, err := seed.LoadWords(wordsPath)
	die(err)

	if importedPath != "" {
		fmt.Printf("Loading imported words from %s...\n", importedPath)
		impRoots, impWords, err := seed.LoadImportedWords(importedPath)
		die(err)
		fmt.Printf("  Imported: %d synthetic roots, %d words\n", len(impRoots), len(impWords))

		// Dedup imported against curated: curated words take priority.
		curated := make(map[string]bool, len(words))
		for _, w := range words {
			key := strings.ToLower(strings.TrimSpace(w.Norm)) + "|" + w.Lang
			curated[key] = true
		}
		dedupWords := make([]seed.Word, 0, len(impWords))
		usedRoots := make(map[uint32]bool)
		for _, w := range impWords {
			key := strings.ToLower(strings.TrimSpace(w.Norm)) + "|" + w.Lang
			if curated[key] {
				continue
			}
			dedupWords = append(dedupWords, w)
			usedRoots[w.RootID] = true
		}
		// Only include roots that have at least one word.
		dedupRoots := make([]seed.Root, 0, len(impRoots))
		for _, r := range impRoots {
			if usedRoots[r.RootID] {
				dedupRoots = append(dedupRoots, r)
			}
		}
		excluded := len(impWords) - len(dedupWords)
		if excluded > 0 {
			fmt.Printf("  Excluded: %d duplicates already in curated words\n", excluded)
		}
		fmt.Printf("  Final:    %d roots, %d words\n", len(dedupRoots), len(dedupWords))
		roots = append(roots, dedupRoots...)
		words = append(words, dedupWords...)
	}

	fmt.Printf("Building lexicon (%d roots, %d words)...\n", len(roots), len(words))
	stats, err := lexdb.Build(roots, words, outPath)
	die(err)

	fmt.Printf("Built %s\n", outPath)
	fmt.Printf("  Roots:  %d\n", stats.RootCount)
	fmt.Printf("  Words:  %d\n", stats.WordCount)
	fmt.Printf("  Langs:  %s\n", stats.Langs())
	fmt.Printf("  Heap:   %d bytes\n", stats.HeapSize)
	fmt.Printf("  Size:   %d bytes (%.1f MB)\n", stats.FileSize, float64(stats.FileSize)/1024/1024)
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
	fmt.Printf("POS:       %s\n", sent["pos"])
	fmt.Printf("Arousal:   %s  Dominance: %s  AoA: %s\n",
		sent["arousal"], sent["dominance"], sent["aoa"])
	if sent["flags"] != "" {
		fmt.Printf("Flags:     %s\n", sent["flags"])
	}
	fmt.Printf("FreqRank:  %d\n", w.FreqRank)

	// IPA pronunciation
	if pron := lex.WordPron(w); pron != "" {
		fmt.Printf("IPA:       %s\n", pron)
	}

	// Phonology from flags
	syl := phon.Syllables(w.Flags)
	if syl > 0 {
		fmt.Printf("Syllables: %d  Stress: %s  Valency: %s\n",
			syl, phon.StressName(w.Flags), phon.ValencyName(w.Flags))
	}
	if w.Flags&phon.IronyCapable != 0 {
		fmt.Printf("Note:      irony-capable\n")
	}
	if w.Flags&phon.Neologism != 0 {
		fmt.Printf("Note:      neologism\n")
	}

	// Semantic relations
	if root != nil {
		if ant := lex.Antonym(root); ant != nil {
			fmt.Printf("Antonym:   %s (ID=%d — %s)\n",
				lex.RootStr(ant), ant.RootID, lex.RootMeaning(ant))
		}
		if hyp := lex.Hypernym(root); hyp != nil {
			fmt.Printf("Hypernym:  %s (ID=%d — %s)\n",
				lex.RootStr(hyp), hyp.RootID, lex.RootMeaning(hyp))
		}
		if syn := lex.Synonym(root); syn != nil {
			fmt.Printf("Synonym:   %s (ID=%d — %s)\n",
				lex.RootStr(syn), syn.RootID, lex.RootMeaning(syn))
		}
	}

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
	reset := false
	reexpand := false
	workers := 1
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
		case "--reset":
			reset = true
		case "--reexpand":
			reexpand = true
		case "--workers":
			if i+1 < len(args) {
				if n, err := strconv.Atoi(args[i+1]); err == nil {
					workers = n
				}
				i++
			}
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

	langList := strings.Split(langs, ",")

	// Build seed list.
	var seeds []string
	if expand {
		// Use all words in the target language set as seeds.
		// This maximises coverage: PT/ES/IT/FR/DE words seed their own
		// Wiktionary pages, which often contain translations not reachable
		// from the EN entry alone.
		targetSet := map[string]bool{}
		for _, l := range langList {
			targetSet[l] = true
		}
		seen := map[string]bool{}
		for _, w := range words {
			if !targetSet[w.Lang] {
				continue
			}
			// Dedup by phonetic norm to avoid fetching the same Wiktionary page
			// twice for surface variants (e.g. "café" and "cafe").
			norm := discover.PhoneticNorm(w.Word)
			if !seen[norm] {
				seen[norm] = true
				seeds = append(seeds, strings.ToLower(w.Word))
			}
		}
	} else {
		seen := map[string]bool{}
		for _, s := range strings.Split(seedWords, ",") {
			if t := strings.TrimSpace(s); t != "" {
				norm := discover.PhoneticNorm(t)
				if !seen[norm] {
					seen[norm] = true
					seeds = append(seeds, strings.ToLower(t))
				}
			}
		}
	}

	if len(seeds) == 0 {
		fatalf("discover: no seeds found")
	}

	cfg := discover.Config{
		Seeds:     seeds,
		Langs:     langList,
		MaxDepth:  depth,
		Limit:     limit,
		DryRun:    dryRun,
		Reset:     reset,
		Reexpand:  reexpand,
		Workers:   workers,
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
	mode := "LIVE (will append to CSVs)"
	if dryRun {
		mode = "DRY RUN (no files modified)"
	}
	if reset {
		mode += " [checkpoint reset]"
	}
	fmt.Printf("  Mode:   %s\n", mode)
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

// ── cmdTrain ──────────────────────────────────────────────────────────────────

func cmdTrain(args []string) {
	lexPath := defaultLexicon
	dataPath := ""
	outPath := "models/classifier.bin"
	valSplit := 0.2
	epochs := 50
	autoGen := false

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--lexicon":
			if i+1 < len(args) {
				lexPath = args[i+1]
				i++
			}
		case "--data":
			if i+1 < len(args) {
				dataPath = args[i+1]
				i++
			}
		case "--out":
			if i+1 < len(args) {
				outPath = args[i+1]
				i++
			}
		case "--val-split":
			if i+1 < len(args) {
				v, err := strconv.ParseFloat(args[i+1], 64)
				if err == nil {
					valSplit = v
				}
				i++
			}
		case "--epochs":
			if i+1 < len(args) {
				v, err := strconv.Atoi(args[i+1])
				if err == nil {
					epochs = v
				}
				i++
			}
		case "--auto":
			autoGen = true
		}
	}

	_ = dataPath // future: load from JSONL

	lex, err := lexdb.Load(lexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load lexicon: %v\n", err)
		os.Exit(1)
	}

	if !autoGen {
		autoGen = true // default to auto when no data path given
	}

	examples := classify.GenerateFromLexicon(lex)
	if len(examples) == 0 {
		fmt.Fprintln(os.Stderr, "no training examples generated")
		os.Exit(1)
	}

	// Root-stratified split — cognates (words sharing root_id) go entirely to
	// train or val, preventing trivial generalisation via cognate leakage.
	train, val := classify.SplitByRoot(examples, valSplit)
	fmt.Printf("Training: %d examples  Val: %d examples (root-stratified, seed=42)\n", len(train), len(val))

	// Majority-class baseline — lower bound any real model must beat.
	majorityF1 := classify.MajorityClassF1(val)
	fmt.Printf("Baseline (majority class):   F1=%.3f\n", majorityF1)
	fmt.Printf("Baseline (random uniform):   F1=0.333\n")

	clf := classify.New(classify.NFeatures, classify.DefaultClasses)

	for epoch := 1; epoch <= epochs; epoch++ {
		for _, ex := range train {
			clf.TrainStep(ex.Features, ex.LabelIdx)
		}
		if epoch%10 == 0 || epoch == epochs {
			f1 := classify.F1Macro(clf, val)
			acc := classify.Accuracy(clf, val)
			pct := 0.0
			if majorityF1 > 0 {
				pct = (f1/majorityF1 - 1) * 100
			}
			fmt.Printf("  epoch %3d/%d  F1=%.3f  acc=%.3f  steps=%d  (+%.1f%% vs majority)\n",
				epoch, epochs, f1, acc, clf.Step, pct)
		}
	}

	if err := os.MkdirAll("models", 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir models: %v\n", err)
		os.Exit(1)
	}
	if err := clf.Save(outPath); err != nil {
		fmt.Fprintf(os.Stderr, "save classifier: %v\n", err)
		os.Exit(1)
	}

	f1Final := classify.F1Macro(clf, val)
	perClass := classify.F1PerClass(clf, val)
	fmt.Printf("Saved → %s\n", outPath)
	fmt.Printf("  Root-stratified val F1 = %.3f  (+%.1f%% vs majority class baseline)\n",
		f1Final, func() float64 {
			if majorityF1 > 0 {
				return (f1Final/majorityF1 - 1) * 100
			}
			return 0
		}())
	fmt.Printf("  Per-class: NEG=%.3f  NEU=%.3f  POS=%.3f\n",
		perClass["NEGATIVE"], perClass["NEUTRAL"], perClass["POSITIVE"])
	fmt.Printf("  No polarity leak (FPolarity zeroed in training features)\n")
}

// ── cmdPredict ────────────────────────────────────────────────────────────────

func cmdPredict(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: lexsent predict <word> [<lang>] [--model PATH] [--lexicon PATH]")
		os.Exit(1)
	}
	word := args[0]
	lang := "EN"
	modelPath := "models/classifier.bin"
	lexPath := defaultLexicon

	for i := 1; i < len(args); i++ {
		switch args[i] {
		case "--model":
			if i+1 < len(args) {
				modelPath = args[i+1]
				i++
			}
		case "--lexicon":
			if i+1 < len(args) {
				lexPath = args[i+1]
				i++
			}
		default:
			if !strings.HasPrefix(args[i], "--") {
				lang = args[i]
			}
		}
	}

	lex, err := lexdb.Load(lexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load lexicon: %v\n", err)
		os.Exit(1)
	}

	clf, err := classify.Load(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n  → run 'lexsent train --auto' first\n", err)
		os.Exit(1)
	}

	agent := rl.New(clf)
	_ = agent.LoadState(modelPath) // ignore error: starts fresh if no sidecar

	f, ok := classify.ExtractFromLexicon(lex, word, lang)
	if !ok {
		fmt.Fprintf(os.Stderr, "word %q (%s) not found in lexicon\n", word, lang)
		os.Exit(1)
	}

	class, conf := agent.Act(f)
	rl.RecordLast(f, class)
	_ = agent.SaveState(modelPath) // persist LastPrediction for feedback cmd

	// Build rich output line
	wr := lex.LookupWordInLang(word, lang)
	extra := ""
	if wr != nil {
		root := lex.LookupRoot(wr.RootID)
		if root != nil {
			extra += fmt.Sprintf("  root=%-8s", lex.RootStr(root))
			if ant := lex.Antonym(root); ant != nil {
				extra += fmt.Sprintf("  ant=%-8s", lex.RootStr(ant))
			}
		}
		pron := lex.WordPron(wr)
		if pron != "" {
			extra += fmt.Sprintf("  IPA=%s", pron)
		}
	}

	fmt.Printf("%s (%.1f%%)%s\n", class, conf*100, extra)
}

// ── cmdFeedback ───────────────────────────────────────────────────────────────

func cmdFeedback(args []string) {
	modelPath := "models/classifier.bin"
	label := ""
	ok := false

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--ok":
			ok = true
		case "--label":
			if i+1 < len(args) {
				label = strings.ToUpper(args[i+1])
				i++
			}
		case "--model":
			if i+1 < len(args) {
				modelPath = args[i+1]
				i++
			}
		}
	}

	if !ok && label == "" {
		fmt.Fprintln(os.Stderr, "usage: lexsent feedback [--ok | --label LABEL] [--model PATH]")
		os.Exit(1)
	}

	clf, err := classify.Load(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load model: %v\n", err)
		os.Exit(1)
	}

	agent := rl.New(clf)
	if err := agent.LoadState(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "load agent state: %v\n", err)
		os.Exit(1)
	}

	if rl.LastPrediction == nil {
		fmt.Fprintln(os.Stderr, "no pending prediction — run 'lexsent predict <word>' first")
		os.Exit(1)
	}

	predicted := rl.LastPrediction.Predicted
	correct := predicted
	reward := 1.0

	if ok {
		correct = predicted
		reward = 1.0
	} else {
		correct = label
		if correct != predicted {
			reward = -1.0
		}
	}

	agent.Observe(rl.Feedback{
		Features:  rl.LastPrediction.Features,
		Predicted: predicted,
		Correct:   correct,
		Reward:    reward,
	})
	agent.Learn()

	if err := clf.Save(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "save model: %v\n", err)
		os.Exit(1)
	}
	if err := agent.SaveState(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "save agent state: %v\n", err)
		os.Exit(1)
	}

	if reward > 0 {
		fmt.Printf("Reward +1 applied  (predicted=%s ✓)\n", predicted)
	} else {
		fmt.Printf("Reward -1, corrective update  (predicted=%s → correct=%s)\n", predicted, correct)
	}
}

// ── cmdEvolve ─────────────────────────────────────────────────────────────────

func cmdEvolve(args []string) {
	modelPath := "models/classifier.bin"
	lexPath := defaultLexicon
	generations := 50
	popSize := 32
	valSplit := 0.2

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--model":
			if i+1 < len(args) {
				modelPath = args[i+1]
				i++
			}
		case "--lexicon":
			if i+1 < len(args) {
				lexPath = args[i+1]
				i++
			}
		case "--generations":
			if i+1 < len(args) {
				v, err := strconv.Atoi(args[i+1])
				if err == nil {
					generations = v
				}
				i++
			}
		case "--pop":
			if i+1 < len(args) {
				v, err := strconv.Atoi(args[i+1])
				if err == nil {
					popSize = v
				}
				i++
			}
		}
	}

	lex, err := lexdb.Load(lexPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load lexicon: %v\n", err)
		os.Exit(1)
	}

	examples := classify.GenerateFromLexicon(lex)
	train, val := classify.SplitByRoot(examples, valSplit)
	fmt.Printf("GA evolution: pop=%d generations=%d train=%d val=%d\n",
		popSize, generations, len(train), len(val))

	pop := ga.New(popSize, 42)
	pop.TrainSteps = 150

	best := pop.Run(train, val, generations, func(gen int, bestF1 float64) {
		fmt.Printf("  gen %3d/%d  best F1=%.4f\n", gen, generations, bestF1)
	})

	// Apply best weights to the saved classifier
	clf, err := classify.Load(modelPath)
	if err != nil {
		// No model yet — create one
		clf = classify.New(classify.NFeatures, classify.DefaultClasses)
	}
	clf.FeatureWeights = best.Weights

	// Retrain with the evolved weights
	for _, ex := range train {
		clf.TrainStep(ex.Features, ex.LabelIdx)
	}

	if err := os.MkdirAll("models", 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir models: %v\n", err)
		os.Exit(1)
	}
	if err := clf.Save(modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "save model: %v\n", err)
		os.Exit(1)
	}

	finalF1 := classify.F1Macro(clf, val)
	fmt.Printf("Best chromosome F1=%.4f → saved to %s  (final F1=%.4f)\n",
		best.Fitness, modelPath, finalF1)
}

// ── cmdExportC ────────────────────────────────────────────────────────────────

func cmdExportC(args []string) {
	outDir := "."
	headerDir := "."

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--out":
			if i+1 < len(args) {
				outDir = args[i+1]
				i++
			}
		case "--header":
			if i+1 < len(args) {
				headerDir = args[i+1]
				i++
			}
		}
	}

	fmt.Println("Building libumcs.so ...")
	soPath := outDir + "/libumcs.so"
	hPath := headerDir + "/umcs.h"

	// Build the shared library via go build -buildmode=c-shared
	// We use os/exec here so the user sees the compiler output.
	buildArgs := []string{
		"build", "-buildmode=c-shared",
		"-o", soPath,
		"github.com/kak/umcs/pkg/capi",
	}
	cmd := exec.Command("go", buildArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "go build failed: %v\n", err)
		os.Exit(1)
	}

	// The c-shared build also auto-generates a .h — copy it if not already at hPath.
	autoH := soPath[:len(soPath)-3] + ".h"
	if autoH != hPath {
		data, err := os.ReadFile(autoH)
		if err == nil {
			_ = os.WriteFile(hPath, data, 0o644)
		}
	}

	fmt.Printf("OK  → %s\n    → %s\n", soPath, hPath)
}

// ── cmdStageReview ────────────────────────────────────────────────────────────
//
// Reads staged.csv (low-confidence candidates from discover/import runs) and
// either auto-accepts entries above a confidence threshold or prompts
// interactively for manual review. Accepted words are appended to words.csv
// and the lexicon is rebuilt.
//
// Usage:
//
//	lexsent stage-review [--staged PATH] [--words PATH] [--roots PATH]
//	                     [--min-conf 0.60] [--batch 50] [--auto] [--dry-run]
func cmdStageReview(args []string) {
	stagedPath := "data/staged.csv"
	wordsPath := "data/words.csv"
	rootsPath := "data/roots.csv"
	lexPath := defaultLexicon
	minConf := 0.60
	batch := 50
	auto := false
	dryRun := false

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--staged":
			if i+1 < len(args) {
				stagedPath = args[i+1]
				i++
			}
		case "--words":
			if i+1 < len(args) {
				wordsPath = args[i+1]
				i++
			}
		case "--roots":
			if i+1 < len(args) {
				rootsPath = args[i+1]
				i++
			}
		case "--lexicon":
			if i+1 < len(args) {
				lexPath = args[i+1]
				i++
			}
		case "--min-conf":
			if i+1 < len(args) {
				v, err := strconv.ParseFloat(args[i+1], 64)
				if err == nil {
					minConf = v
				}
				i++
			}
		case "--batch":
			if i+1 < len(args) {
				v, err := strconv.Atoi(args[i+1])
				if err == nil {
					batch = v
				}
				i++
			}
		case "--auto":
			auto = true
		case "--dry-run":
			dryRun = true
		}
	}

	// Load existing words to compute proper variant numbers.
	allWords, err := seed.LoadWords(wordsPath)
	if err != nil {
		fatalf("load words: %v", err)
	}
	allRoots, err := seed.LoadRoots(rootsPath)
	if err != nil {
		fatalf("load roots: %v", err)
	}
	_ = allRoots

	// Read staged.csv.
	f, err := os.Open(stagedPath)
	if err != nil {
		fatalf("open staged: %v", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.TrimLeadingSpace = true
	header, err := r.Read()
	if err != nil {
		fatalf("read staged header: %v", err)
	}
	// Expected: word,lang,root_str,proposed_root_id,polarity,intensity,role,confidence,source,definition
	colIdx := make(map[string]int)
	for i, h := range header {
		colIdx[strings.TrimSpace(h)] = i
	}
	col := func(row []string, name string) string {
		idx, ok := colIdx[name]
		if !ok || idx >= len(row) {
			return ""
		}
		return strings.TrimSpace(row[idx])
	}

	type candidate struct {
		word, lang, polarity, intensity, role, source, definition string
		rootID                                                     uint32
		confidence                                                 float64
	}
	var candidates []candidate
	for {
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue
		}
		conf, _ := strconv.ParseFloat(col(row, "confidence"), 64)
		if conf < minConf {
			continue
		}
		rid, _ := strconv.ParseUint(col(row, "proposed_root_id"), 10, 32)
		candidates = append(candidates, candidate{
			word:       col(row, "word"),
			lang:       col(row, "lang"),
			polarity:   col(row, "polarity"),
			intensity:  col(row, "intensity"),
			role:       col(row, "role"),
			source:     col(row, "source"),
			definition: col(row, "definition"),
			rootID:     uint32(rid),
			confidence: conf,
		})
	}

	fmt.Printf("stage-review\n")
	fmt.Printf("  Staged file: %s\n", stagedPath)
	fmt.Printf("  Candidates ≥ %.2f conf: %d\n", minConf, len(candidates))
	fmt.Printf("  Mode: %s\n\n", func() string {
		if auto {
			return "AUTO (accept all above threshold)"
		}
		return fmt.Sprintf("INTERACTIVE (batches of %d)", batch)
	}())

	if len(candidates) == 0 {
		fmt.Println("No candidates above confidence threshold.")
		return
	}

	// Build known-word set to skip duplicates.
	known := make(map[string]bool, len(allWords))
	for _, w := range allWords {
		known[w.Norm+"_"+w.Lang] = true
	}

	var accepted []seed.Word
	stdin := bufio.NewReader(os.Stdin)

	accept := func(c candidate) {
		norm := discover.PhoneticNorm(c.word)
		key := norm + "_" + c.lang
		if known[key] {
			return
		}
		pol := c.polarity
		if pol == "" {
			pol = "NEUTRAL"
		}
		intensity := c.intensity
		if intensity == "" {
			intensity = "NONE"
		}
		role := c.role
		if role == "" {
			role = "EVALUATION"
		}
		packed, err := sentiment.Pack(pol, intensity, role, "GENERAL", nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  pack error %q: %v\n", c.word, err)
			return
		}
		variant := discover.NextVariant(c.rootID, allWords)
		wordID, err := morpheme.MakeWordID(c.rootID, variant)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  variant overflow %q: %v\n", c.word, err)
			return
		}
		w := seed.Word{
			WordID:    wordID,
			RootID:    c.rootID,
			Variant:   variant,
			Word:      c.word,
			Lang:      c.lang,
			Norm:      norm,
			Sentiment: packed,
		}
		accepted = append(accepted, w)
		allWords = append(allWords, w)
		known[key] = true
		fmt.Printf("  + %-20s [%s] pol=%-8s conf=%.0f%% src=%s\n",
			c.word, c.lang, pol, c.confidence*100, c.source)
	}

	if auto {
		for _, c := range candidates {
			accept(c)
		}
	} else {
		for i := 0; i < len(candidates); i += batch {
			end := i + batch
			if end > len(candidates) {
				end = len(candidates)
			}
			slice := candidates[i:end]
			fmt.Printf("── Batch %d/%d ─────────────────────────\n",
				i/batch+1, (len(candidates)+batch-1)/batch)
			for j, c := range slice {
				fmt.Printf("  [%d] %-20s [%s] pol=%-8s conf=%.0f%%  %s\n",
					j+1, c.word, c.lang, c.polarity, c.confidence*100, c.definition)
			}
			fmt.Print("\n  [y]es-all  [n]o-all  [a]uto-rest  [q]uit  or number to toggle: ")
			line, _ := stdin.ReadString('\n')
			line = strings.TrimSpace(line)
			switch line {
			case "y", "yes":
				for _, c := range slice {
					accept(c)
				}
			case "n", "no":
				fmt.Printf("  Skipped %d entries.\n", len(slice))
			case "a", "auto":
				for _, c := range candidates[i:] {
					accept(c)
				}
				break
			case "q", "quit":
				fmt.Println("  Stopped.")
				goto done
			default:
				fmt.Printf("  Skipped batch.\n")
			}
			fmt.Println()
		}
	}
done:

	fmt.Printf("\nAccepted: %d new words\n", len(accepted))
	if len(accepted) == 0 || dryRun {
		if dryRun {
			fmt.Println("Dry run — no changes written.")
		}
		return
	}

	// Append accepted words to words.csv.
	wf, err := os.OpenFile(wordsPath, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		fatalf("open words for append: %v", err)
	}
	w := csv.NewWriter(wf)
	for _, word := range accepted {
		w.Write([]string{
			strconv.FormatUint(uint64(word.WordID), 10),
			strconv.FormatUint(uint64(word.RootID), 10),
			strconv.FormatUint(uint64(word.Variant), 10),
			word.Word,
			word.Lang,
			word.Norm,
			strconv.FormatUint(uint64(word.Sentiment), 10),
			"0", // FreqRank
			"0", // Flags
			"",  // Pron
		})
	}
	w.Flush()
	wf.Close()
	if err := w.Error(); err != nil {
		fatalf("write words csv: %v", err)
	}
	fmt.Printf("Written → %s\n", wordsPath)

	// Rebuild lexicon.
	fmt.Print("Rebuilding lexicon... ")
	roots, err := seed.LoadRoots(rootsPath)
	if err != nil {
		fatalf("load roots: %v", err)
	}
	words, err := seed.LoadWords(wordsPath)
	if err != nil {
		fatalf("reload words: %v", err)
	}
	if _, err := lexdb.Build(roots, words, lexPath); err != nil {
		fatalf("build lexicon: %v", err)
	}
	fmt.Printf("Lexicon: %d roots, %d words\n", len(roots), len(words))
}

func usage() {
	fmt.Println(`lexsent — Universal Morpheme Coordinate System

Commands:
  build    [--roots PATH] [--words PATH] [--out PATH]
  demo     [--lexicon PATH]
  lookup   <word>
  cognates <word>
  etymo    <word>
  analyze  <text>
  tokenize <text>
  stats    [--productive]
  serve    [--port PORT]
  discover [--expand | --seed word1,word2] [--lang PT,EN,...] [--depth N] [--limit N] [--dry-run] [--reset] [--verbose]
  import   --dump PATH.xml.bz2 [--lang PT,EN,...] [--limit N] [--dry-run] [--verbose] [--batch N]

ML Commands:
  train    [--auto] [--lexicon PATH] [--out models/classifier.bin] [--val-split 0.2] [--epochs 50]
  predict  <word> [<lang>] [--model models/classifier.bin] [--lexicon PATH]
  feedback [--ok | --label LABEL] [--model PATH]
  evolve   [--generations 50] [--pop 32] [--model PATH] [--lexicon PATH]
  export-c [--out DIR] [--header DIR]`)
}
