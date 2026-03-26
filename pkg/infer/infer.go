// Package infer provides morphological inference rules that auto-fill missing
// semantic annotation fields from suffix/prefix patterns.
//
// The core insight: word shape predicts meaning. "-ção"/"-tion" → NOUN+ABSTRACT.
// "-mente"/"-ly" → ADV. "ir-"/"des-"/"un-" → likely negation prefix.
//
// Infer is run during the build pipeline to fill in dimensions that the CSV
// annotator left empty (zero). It never overwrites an explicitly annotated field.
// This is the "standardize what repeats" layer of the UMCS pipeline.
//
// For a genetic algorithm or ML-based approach to weight tuning, see the
// propagate package (confidence-weighted majority vote among cognates).
package infer

import (
	"strings"
	"unicode"

	"github.com/kak/umcs/pkg/phon"
	"github.com/kak/umcs/pkg/sentiment"
)

// suffixRule maps a language-specific suffix to an inferred sentiment bit.
type suffixRule struct {
	suffix string
	lang   string // "" = any language
	bit    uint32
}

// posRules infers POS from suffix patterns.
var posRules = []suffixRule{
	// Portuguese / Spanish / Italian — nominalizers
	{"-ção", "PT", sentiment.POSNoun}, {"-cão", "PT", sentiment.POSNoun},
	{"-são", "PT", sentiment.POSNoun}, {"-dade", "PT", sentiment.POSNoun},
	{"-ismo", "PT", sentiment.POSNoun}, {"-ista", "PT", sentiment.POSNoun},
	{"-eza", "PT", sentiment.POSNoun}, {"-ura", "PT", sentiment.POSNoun},
	{"-ção", "ES", sentiment.POSNoun}, {"-ción", "ES", sentiment.POSNoun},
	{"-dad", "ES", sentiment.POSNoun}, {"-idad", "ES", sentiment.POSNoun},
	{"-ismo", "ES", sentiment.POSNoun}, {"-ista", "ES", sentiment.POSNoun},
	{"-zione", "IT", sentiment.POSNoun}, {"-tà", "IT", sentiment.POSNoun},
	{"-ismo", "IT", sentiment.POSNoun},
	// French
	{"-tion", "FR", sentiment.POSNoun}, {"-sion", "FR", sentiment.POSNoun},
	{"-té", "FR", sentiment.POSNoun}, {"-isme", "FR", sentiment.POSNoun},
	{"-eur", "FR", sentiment.POSNoun},
	// English — nominalizers
	{"-tion", "EN", sentiment.POSNoun}, {"-sion", "EN", sentiment.POSNoun},
	{"-ness", "EN", sentiment.POSNoun}, {"-ity", "EN", sentiment.POSNoun},
	{"-ism", "EN", sentiment.POSNoun}, {"-ist", "EN", sentiment.POSNoun},
	{"-ment", "EN", sentiment.POSNoun}, {"-er", "EN", sentiment.POSNoun},
	{"-or", "EN", sentiment.POSNoun}, {"-ance", "EN", sentiment.POSNoun},
	{"-ence", "EN", sentiment.POSNoun},
	// German
	{"-heit", "DE", sentiment.POSNoun}, {"-keit", "DE", sentiment.POSNoun},
	{"-ung", "DE", sentiment.POSNoun}, {"-ismus", "DE", sentiment.POSNoun},
	// Adverb suffixes
	{"-mente", "PT", sentiment.POSAdv}, {"-mente", "ES", sentiment.POSAdv},
	{"-mente", "IT", sentiment.POSAdv}, {"-ment", "FR", sentiment.POSAdv},
	{"-ly", "EN", sentiment.POSAdv},
	// Adjective suffixes
	{"-oso", "PT", sentiment.POSAdj}, {"-osa", "PT", sentiment.POSAdj},
	{"-oso", "ES", sentiment.POSAdj}, {"-osa", "ES", sentiment.POSAdj},
	{"-oso", "IT", sentiment.POSAdj},
	{"-eux", "FR", sentiment.POSAdj}, {"-euse", "FR", sentiment.POSAdj},
	{"-ful", "EN", sentiment.POSAdj}, {"-less", "EN", sentiment.POSAdj},
	{"-ive", "EN", sentiment.POSAdj}, {"-al", "EN", sentiment.POSAdj},
	{"-lich", "DE", sentiment.POSAdj}, {"-ig", "DE", sentiment.POSAdj},
	// Verb suffixes (less reliable — apply only when other rules don't match)
	{"-ar", "PT", sentiment.POSVerb}, {"-er", "PT", sentiment.POSVerb},
	{"-ir", "PT", sentiment.POSVerb},
	{"-ar", "ES", sentiment.POSVerb}, {"-er", "ES", sentiment.POSVerb},
	{"-ir", "ES", sentiment.POSVerb},
	{"-are", "IT", sentiment.POSVerb}, {"-ere", "IT", sentiment.POSVerb},
	{"-ire", "IT", sentiment.POSVerb},
	{"-er", "FR", sentiment.POSVerb}, {"-ir", "FR", sentiment.POSVerb},
	{"-re", "FR", sentiment.POSVerb},
}

// concreteRules: suffixes that strongly predict ABSTRACT (concreteness=0).
var abstractSuffixes = map[string][]string{
	"PT": {"-dade", "-ismo", "-eza", "-ura", "-ção", "-são", "-mento", "-ncia"},
	"ES": {"-dad", "-idad", "-ismo", "-eza", "-ción", "-ción", "-miento"},
	"IT": {"-tà", "-ismo", "-ezza", "-zione", "-mento"},
	"FR": {"-té", "-isme", "-tion", "-sion", "-ment"},
	"EN": {"-ness", "-ity", "-ism", "-tion", "-sion", "-ment", "-ance", "-ence"},
	"DE": {"-heit", "-keit", "-ismus", "-ung"},
}

// POSFromShape infers the most likely POS from the word's suffix in the given language.
// Returns 0 (POSOther) if no rule matches. Only call this when POS is unset (0).
func POSFromShape(word, lang string) uint32 {
	lower := strings.ToLower(word)
	// Longest-match wins: iterate rules and prefer longer suffixes.
	var best uint32
	var bestLen int
	for _, r := range posRules {
		if r.lang != "" && r.lang != lang {
			continue
		}
		suf := r.suffix
		if strings.HasPrefix(suf, "-") {
			suf = suf[1:]
		}
		if strings.HasSuffix(lower, suf) && len(suf) > bestLen {
			best = r.bit
			bestLen = len(suf)
		}
	}
	return best
}

// IsAbstractFromShape returns true if the word's suffix strongly predicts abstractness.
func IsAbstractFromShape(word, lang string) bool {
	lower := strings.ToLower(word)
	for _, suf := range abstractSuffixes[lang] {
		if strings.HasPrefix(suf, "-") {
			suf = suf[1:]
		}
		if strings.HasSuffix(lower, suf) {
			return true
		}
	}
	return false
}

// RegisterFromSentiment infers a register constant from existing sentiment bits.
// Useful when register is unset but other fields already encode the information.
func RegisterFromSentiment(sent uint32) uint32 {
	// Scope markers are grammatical function words — neutral register.
	if sentiment.IsNegationMarker(sent) || sentiment.IsIntensifier(sent) || sentiment.IsDowntoner(sent) {
		return 0 // NEUTRAL
	}
	// Technical AoA → technical register
	if sentiment.AOA(sent) == sentiment.AOATechnical {
		return 7 << 8 // RegisterTechnical
	}
	// Early AoA → child-accessible word
	if sentiment.AOA(sent) == sentiment.AOAEarly {
		return 9 << 8 // RegisterChild
	}
	return 0
}

// FillMissing applies morphological inference to fill in zero (unset) sentiment
// dimensions. It does NOT overwrite non-zero values. Returns the augmented value.
func FillMissing(sent uint32, word, lang string) uint32 {
	// Infer POS if unset
	if sentiment.POS(sent) == 0 {
		if pos := POSFromShape(word, lang); pos != 0 {
			sent |= pos
		}
	}
	// Infer concreteness if unset (bit 28 = 0 could mean abstract OR unset)
	// We infer ABSTRACT from suffix; concreteness defaults to 0 anyway.
	// Only set CONCRETE explicitly for known concrete patterns.
	// (Most words are abstract-unset; inference here would have too many false positives)

	return sent
}

// SyllablesFromShape counts approximate syllables by counting vowel groups.
// This is a coarse heuristic, not a proper syllabification algorithm.
// Returns 0 for empty or all-consonant words. Counts are clamped to [0, 15].
//
// Vowel nuclei are sequences of vowel letters (including diacritics mapped to
// their base). Final silent 'e' in English is not special-cased here because
// we operate on normalized (diacritic-stripped) forms.
func SyllablesFromShape(word, lang string) uint32 {
	if word == "" {
		return 0
	}
	lower := strings.ToLower(word)
	var count uint32
	inVowel := false
	for _, r := range lower {
		if isVowel(r) {
			if !inVowel {
				count++
				inVowel = true
			}
		} else {
			inVowel = false
		}
	}
	// English heuristic: subtract 1 for silent final 'e' (e.g. "love", "hate").
	// Only when the word ends in 'e' preceded by a consonant and count > 1.
	if lang == "EN" && count > 1 {
		runes := []rune(lower)
		n := len(runes)
		if n >= 3 && runes[n-1] == 'e' && !isVowel(runes[n-2]) && isVowel(runes[n-3]) {
			count--
		}
	}
	if count > 15 {
		count = 15
	}
	return count
}

// isVowel reports whether r is a vowel letter (basic Latin; accented forms are
// normalized before reaching here in most callers).
func isVowel(r rune) bool {
	if !unicode.IsLetter(r) {
		return false
	}
	switch r {
	case 'a', 'e', 'i', 'o', 'u',
		'á', 'à', 'â', 'ã', 'ä', 'å',
		'é', 'è', 'ê', 'ë',
		'í', 'ì', 'î', 'ï',
		'ó', 'ò', 'ô', 'õ', 'ö',
		'ú', 'ù', 'û', 'ü',
		'ý', 'ÿ':
		return true
	}
	return false
}

// stressSuffixes maps (lang, suffix) → stress pattern following documented phonology.
// Sources: Portuguese: Bisol (2013); Spanish: Harris (1983); French: Tranel (1987).
var stressSuffixRules = []struct {
	lang   string
	suffix string
	stress uint32
}{
	// Portuguese: oxytone suffixes (stressed on final syllable).
	{"PT", "ção", phon.StressFinal},
	{"PT", "são", phon.StressFinal},
	{"PT", "or", phon.StressFinal},
	{"PT", "ez", phon.StressFinal},
	{"PT", "il", phon.StressFinal},
	{"PT", "az", phon.StressFinal},
	// Spanish: paroxytone is default; mark exceptions.
	{"ES", "ción", phon.StressFinal},
	{"ES", "sión", phon.StressFinal},
	{"ES", "or", phon.StressFinal},
	// English: no fixed rule; primary patterns annotated in data.
	{"EN", "tion", phon.StressFinal},   // ac-TION
	{"EN", "sion", phon.StressFinal},   // ten-SION
	// French: always final (fixed stress).
	{"FR", "", phon.StressFinal}, // catch-all for FR
	// Turkish: always final (fixed stress).
	{"TR", "", phon.StressFinal},
	// German: initial stress for most native words.
	{"DE", "", phon.StressPenultimate}, // approximation; many exceptions
}

// defaultStress returns the most common stress pattern for a language.
// Based on typological frequency data (Gussenhoven & Jacobs, 2011).
var defaultStress = map[string]uint32{
	"PT": phon.StressPenultimate,  // paroxytone is most common
	"ES": phon.StressPenultimate,
	"IT": phon.StressPenultimate,
	"FR": phon.StressFinal,        // fixed final stress
	"TR": phon.StressFinal,        // fixed final stress
	"EN": phon.StressPenultimate,  // most common in English
	"DE": phon.StressPenultimate,  // most lexical words
	"NL": phon.StressPenultimate,
	"PL": phon.StressPenultimate,
}

// StressFromShape infers the stress pattern from language defaults and suffix patterns.
// This is a statistical heuristic, not a full phonological parser.
func StressFromShape(word, lang string) uint32 {
	lower := strings.ToLower(word)
	// Check suffix rules (longest match wins).
	var best uint32
	var bestLen int
	for _, rule := range stressSuffixRules {
		if rule.lang != lang {
			continue
		}
		if rule.suffix == "" {
			// Language-level default; only use if no suffix matched.
			if bestLen == 0 {
				best = rule.stress
			}
			continue
		}
		if strings.HasSuffix(lower, rule.suffix) && len(rule.suffix) > bestLen {
			best = rule.stress
			bestLen = len(rule.suffix)
		}
	}
	if best != 0 {
		return best
	}
	// Fall back to language default.
	if def, ok := defaultStress[lang]; ok {
		return def
	}
	return phon.StressUnknown
}

// FillPhonology fills missing phonology bits (syllables, stress) in flags using
// word-shape heuristics. It does NOT overwrite non-zero values.
// Called during seed loading after explicit CSV values are parsed.
func FillPhonology(flags uint32, word, lang string) uint32 {
	if phon.Syllables(flags) == 0 {
		if n := SyllablesFromShape(word, lang); n > 0 {
			flags = phon.SetSyllables(flags, n)
		}
	}
	if phon.Stress(flags) == 0 {
		if s := StressFromShape(word, lang); s != 0 {
			flags = phon.SetStress(flags, s)
		}
	}
	return flags
}
