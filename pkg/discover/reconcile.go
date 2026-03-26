package discover

import (
	"strings"

	"github.com/kak/lex-sentiment/pkg/seed"
)

// rootStemSuffixes are common Latin/PIE endings removed to find the stem.
// Ordered from longest to shortest to avoid prefix clashes.
var rootStemSuffixes = []string{
	"ation", "ation", "ment", "ness", "ness",
	"are", "ere", "ire",  // Latin verb infinitives
	"alis", "ilis", "alis",
	"us", "um", "a", "is", "os",
}

// StemAncestor reduces a Latin/PIE ancestor word to its likely root morpheme.
// For CJK words the character itself is the morpheme — no stripping is done.
func StemAncestor(word string) string {
	w := strings.ToLower(strings.TrimSpace(word))
	if hasCJK(w) {
		return w // ideogram IS the morpheme
	}
	// Remove trailing non-alpha characters
	for len(w) > 0 && !isAlpha(rune(w[len(w)-1])) {
		w = w[:len(w)-1]
	}
	for _, sfx := range rootStemSuffixes {
		if strings.HasSuffix(w, sfx) && len(w) > len(sfx)+2 {
			return w[:len(w)-len(sfx)]
		}
	}
	return w
}

func isAlpha(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}

// Assign finds an existing root_id for rootStr, or proposes a new one.
// Uses exact match first, then phonetic fuzzy match (threshold: 0.82).
// Returns (root_id, isNewRoot).
func Assign(rootStr string, existing []seed.Root) (rootID uint32, isNew bool) {
	norm := PhoneticNorm(rootStr)

	// Exact phonetic match
	for _, r := range existing {
		if PhoneticNorm(r.RootStr) == norm {
			return r.RootID, false
		}
	}

	// Fuzzy match — require high similarity AND minimum length to prevent false assignments
	// like "color" → "dolor" (both len=5, dist=1, sim=0.80 which was accepted at 0.75).
	bestSim := 0.0
	var bestID uint32
	for _, r := range existing {
		rNorm := PhoneticNorm(r.RootStr)
		// Require both strings to be at least 4 chars to avoid short-string noise.
		if len([]rune(norm)) < 4 || len([]rune(rNorm)) < 4 {
			continue
		}
		sim := LevenshteinSim(norm, rNorm)
		if sim > bestSim && sim >= 0.85 {
			bestSim = sim
			bestID = r.RootID
		}
	}
	if bestID != 0 {
		return bestID, false
	}

	// Propose new root_id = current max + 1
	var maxID uint32
	for _, r := range existing {
		if r.RootID > maxID {
			maxID = r.RootID
		}
	}
	return maxID + 1, true
}

// NextVariant returns the next unused variant number for rootID.
func NextVariant(rootID uint32, existing []seed.Word) uint32 {
	var maxVariant uint32
	for _, w := range existing {
		if w.RootID == rootID && w.Variant > maxVariant {
			maxVariant = w.Variant
		}
	}
	return maxVariant + 1
}

// WordExists reports whether a normalized form in a given language is already in the lexicon.
func WordExists(norm, lang string, existing []seed.Word) bool {
	for _, w := range existing {
		if w.Norm == norm && w.Lang == lang {
			return true
		}
	}
	return false
}
