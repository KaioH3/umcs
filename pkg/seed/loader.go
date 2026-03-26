// Package seed reads the CSV seed files (roots.csv, words.csv) and returns
// structured records ready for the lexdb builder.
package seed

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/kak/umcs/pkg/morpheme"
	"github.com/kak/umcs/pkg/sentiment"
)

// Root holds a parsed root record from roots.csv.
type Root struct {
	RootID       uint32
	RootStr      string // e.g. "negat"
	Origin       string // e.g. "LATIN", "PROTO_GERMANIC"
	MeaningEN    string // e.g. "to deny or negate"
	Notes        string
	ParentRootID uint32 // 0 if no parent (etymological link)
}

// Word holds a parsed word record from words.csv.
type Word struct {
	WordID    uint32
	RootID    uint32
	Variant   uint32
	Word      string // actual surface form, may contain diacritics
	Lang      string // PT, EN, ES, IT, DE
	Norm      string // ASCII-normalized form for hashing
	Sentiment uint32 // packed bitmask
	FreqRank  uint32
	Flags     uint32
}

// LoadRoots reads roots.csv and returns all root records.
// Expected columns: root_id, root_str, origin, meaning_en, notes, parent_root_id (optional)
func LoadRoots(path string) ([]Root, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open roots csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.TrimLeadingSpace = true

	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read roots header: %w", err)
	}
	idx := indexHeader(header)

	var roots []Root
	lineNum := 1
	for {
		lineNum++
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("roots.csv line %d: %w", lineNum, err)
		}

		rootID, err := parseUint32(col(row, idx, "root_id"))
		if err != nil {
			return nil, fmt.Errorf("roots.csv line %d: root_id: %w", lineNum, err)
		}

		var parentID uint32
		if p := col(row, idx, "parent_root_id"); p != "" {
			parentID, err = parseUint32(p)
			if err != nil {
				return nil, fmt.Errorf("roots.csv line %d: parent_root_id: %w", lineNum, err)
			}
		}

		roots = append(roots, Root{
			RootID:       rootID,
			RootStr:      strings.TrimSpace(col(row, idx, "root_str")),
			Origin:       strings.TrimSpace(col(row, idx, "origin")),
			MeaningEN:    strings.TrimSpace(col(row, idx, "meaning_en")),
			Notes:        strings.TrimSpace(col(row, idx, "notes")),
			ParentRootID: parentID,
		})
	}
	return roots, nil
}

// LoadWords reads words.csv and returns all word records.
// Expected columns: word_id, root_id, variant, word, lang, norm,
//   polarity, intensity, semantic_role, domain, freq_rank, flags
func LoadWords(path string) ([]Word, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open words csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.TrimLeadingSpace = true

	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("read words header: %w", err)
	}
	idx := indexHeader(header)

	var words []Word
	lineNum := 1
	for {
		lineNum++
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("words.csv line %d: %w", lineNum, err)
		}

		wordID, err := parseUint32(col(row, idx, "word_id"))
		if err != nil {
			return nil, fmt.Errorf("words.csv line %d: word_id: %w", lineNum, err)
		}
		rootID, err := parseUint32(col(row, idx, "root_id"))
		if err != nil {
			return nil, fmt.Errorf("words.csv line %d: root_id: %w", lineNum, err)
		}
		variant, err := parseUint32(col(row, idx, "variant"))
		if err != nil {
			return nil, fmt.Errorf("words.csv line %d: variant: %w", lineNum, err)
		}
		var freqRank, flags uint32
		if s := col(row, idx, "freq_rank"); s != "" {
			if freqRank, err = parseUint32(s); err != nil {
				return nil, fmt.Errorf("words.csv line %d: freq_rank: %w", lineNum, err)
			}
		}
		if s := col(row, idx, "flags"); s != "" {
			if flags, err = parseUint32(s); err != nil {
				return nil, fmt.Errorf("words.csv line %d: flags: %w", lineNum, err)
			}
		}

		// Validate ID consistency
		if err := morpheme.Validate(wordID, rootID, variant); err != nil {
			return nil, fmt.Errorf("words.csv line %d: %w", lineNum, err)
		}

		// Pack sentiment from CSV string columns (core + extended dimensions).
		var scopeFlags []string
		role := col(row, idx, "semantic_role")
		switch role {
		case "NEGATION_MARKER":
			scopeFlags = append(scopeFlags, "NEGATION_MARKER")
		case "INTENSIFIER":
			scopeFlags = append(scopeFlags, "INTENSIFIER")
		case "DOWNTONER":
			scopeFlags = append(scopeFlags, "DOWNTONER")
		case "AFFIRMATION_MARKER":
			scopeFlags = append(scopeFlags, "AFFIRMATION_MARKER")
		}

		sent, err := sentiment.PackExtended(
			col(row, idx, "polarity"),
			col(row, idx, "intensity"),
			role,
			col(row, idx, "domain"),
			scopeFlags,
			col(row, idx, "pos"),
			col(row, idx, "arousal"),
			col(row, idx, "dominance"),
			col(row, idx, "aoa"),
			col(row, idx, "concreteness"),
		)
		if err != nil {
			return nil, fmt.Errorf("words.csv line %d: sentiment: %w", lineNum, err)
		}

		// Pack register, ontological category, polysemy, cultural flag into flags.
		if reg := col(row, idx, "register"); reg != "" {
			r, err := parseRegister(reg)
			if err != nil {
				return nil, fmt.Errorf("words.csv line %d: register: %w", lineNum, err)
			}
			flags |= r
		}
		if onto := col(row, idx, "ontological"); onto != "" {
			o, err := parseOntological(onto)
			if err != nil {
				return nil, fmt.Errorf("words.csv line %d: ontological: %w", lineNum, err)
			}
			flags |= o
		}
		if poly := col(row, idx, "polysemy"); poly != "" {
			p, err := parseUint32(poly)
			if err != nil {
				return nil, fmt.Errorf("words.csv line %d: polysemy: %w", lineNum, err)
			}
			if p > 15 {
				p = 15
			}
			flags |= (p & 0xF) << 16
		}
		if col(row, idx, "cultural_specific") == "1" || strings.ToUpper(col(row, idx, "cultural_specific")) == "TRUE" {
			flags |= 1 << 20
		}

		words = append(words, Word{
			WordID:    wordID,
			RootID:    rootID,
			Variant:   variant,
			Word:      strings.TrimSpace(col(row, idx, "word")),
			Lang:      strings.TrimSpace(col(row, idx, "lang")),
			Norm:      strings.TrimSpace(col(row, idx, "norm")),
			Sentiment: sent,
			FreqRank:  freqRank,
			Flags:     flags,
		})
	}
	return words, nil
}

func indexHeader(header []string) map[string]int {
	m := make(map[string]int, len(header))
	for i, h := range header {
		m[strings.TrimSpace(strings.ToLower(h))] = i
	}
	return m
}

func col(row []string, idx map[string]int, name string) string {
	i, ok := idx[name]
	if !ok || i >= len(row) {
		return ""
	}
	return strings.TrimSpace(row[i])
}

func parseRegister(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "", "NEUTRAL":
		return 0, nil
	case "FORMAL":
		return 1 << 8, nil
	case "INFORMAL":
		return 2 << 8, nil
	case "SLANG":
		return 3 << 8, nil
	case "VULGAR":
		return 4 << 8, nil
	case "ARCHAIC":
		return 5 << 8, nil
	case "POETIC":
		return 6 << 8, nil
	case "TECHNICAL":
		return 7 << 8, nil
	case "SCIENTIFIC":
		return 8 << 8, nil
	case "CHILD":
		return 9 << 8, nil
	case "REGIONAL":
		return 10 << 8, nil
	}
	return 0, fmt.Errorf("unknown register %q", s)
}

func parseOntological(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "", "NONE":
		return 0, nil
	case "PERSON":
		return 1 << 12, nil
	case "PLACE":
		return 2 << 12, nil
	case "ARTIFACT":
		return 3 << 12, nil
	case "NATURAL":
		return 4 << 12, nil
	case "EVENT":
		return 5 << 12, nil
	case "STATE":
		return 6 << 12, nil
	case "PROPERTY":
		return 7 << 12, nil
	case "QUANTITY":
		return 8 << 12, nil
	case "RELATION":
		return 9 << 12, nil
	case "TEMPORAL":
		return 10 << 12, nil
	case "BIOLOGICAL":
		return 11 << 12, nil
	case "SOCIAL":
		return 12 << 12, nil
	case "ABSTRACT":
		return 13 << 12, nil
	}
	return 0, fmt.Errorf("unknown ontological %q", s)
}

func parseUint32(s string) (uint32, error) {
	if s == "" {
		return 0, nil
	}
	v, err := strconv.ParseUint(s, 10, 32)
	if err != nil {
		return 0, err
	}
	return uint32(v), nil
}
