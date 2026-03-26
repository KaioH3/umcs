// Package discover implements automated word/morpheme discovery from Wiktionary.
package discover

import "strings"

// IsCJK reports whether r is a CJK/Hangul/Kana ideographic character.
// These characters are morphemes themselves and must not be stripped.
func IsCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || // CJK Unified Ideographs
		(r >= 0x3400 && r <= 0x4DBF) || // CJK Extension A
		(r >= 0x20000 && r <= 0x2A6DF) || // CJK Extension B
		(r >= 0x3040 && r <= 0x30FF) || // Hiragana + Katakana
		(r >= 0xAC00 && r <= 0xD7A3) // Hangul syllables
}

// hasCJK reports whether word contains any CJK character.
func hasCJK(word string) bool {
	for _, r := range word {
		if IsCJK(r) {
			return true
		}
	}
	return false
}

// PhoneticNorm lowercases a word and removes common diacritics for comparison.
// For CJK words the characters are preserved unchanged (each is its own morpheme).
func PhoneticNorm(word string) string {
	if hasCJK(word) {
		// CJK: strip spaces and lowercase ASCII portions, preserve ideograms
		return strings.TrimSpace(strings.ToLower(word))
	}
	var b strings.Builder
	for _, r := range strings.ToLower(word) {
		switch r {
		// Acute, grave, circumflex, tilde, umlaut
		case 'á', 'à', 'â', 'ã', 'ä', 'ā', 'ă', 'ą':
			b.WriteRune('a')
		case 'é', 'è', 'ê', 'ë', 'ē', 'ě', 'ę':
			b.WriteRune('e')
		case 'í', 'ì', 'î', 'ï', 'ī', 'ĭ', 'į':
			b.WriteRune('i')
		case 'ó', 'ò', 'ô', 'õ', 'ö', 'ō', 'ő':
			b.WriteRune('o')
		case 'ú', 'ù', 'û', 'ü', 'ū', 'ű', 'ů':
			b.WriteRune('u')
		case 'ç', 'ć', 'č':
			b.WriteRune('c')
		case 'ñ', 'ń', 'ň':
			b.WriteRune('n')
		case 'ß':
			b.WriteString("ss")
		case 'ý', 'ÿ':
			b.WriteRune('y')
		case 'ž', 'ź', 'ż':
			b.WriteRune('z')
		case 'š', 'ś':
			b.WriteRune('s')
		case 'ř':
			b.WriteRune('r')
		case 'ğ':
			b.WriteRune('g')
		case 'ł':
			b.WriteRune('l')
		case 'đ':
			b.WriteRune('d')
		case 'æ':
			b.WriteString("ae")
		case 'œ':
			b.WriteString("oe")
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// LevenshteinSim returns a normalized similarity score in [0.0, 1.0].
// 1.0 means identical, 0.0 means completely different.
func LevenshteinSim(a, b string) float64 {
	ra, rb := []rune(a), []rune(b)
	la, lb := len(ra), len(rb)
	if la == 0 && lb == 0 {
		return 1.0
	}
	if la == 0 || lb == 0 {
		return 0.0
	}

	prev := make([]int, lb+1)
	curr := make([]int, lb+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			if ra[i-1] == rb[j-1] {
				curr[j] = prev[j-1]
			} else {
				curr[j] = 1 + min3(prev[j], curr[j-1], prev[j-1])
			}
		}
		prev, curr = curr, prev
	}
	dist := prev[lb]
	maxLen := la
	if lb > maxLen {
		maxLen = lb
	}
	return 1.0 - float64(dist)/float64(maxLen)
}

func min3(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}

// cognateThresholds defines minimum similarity for two words to be considered cognates.
// Values are per language pair (order-independent).
var cognateThresholds = map[[2]string]float64{
	{"PT", "ES"}: 0.70,
	{"PT", "IT"}: 0.65,
	{"PT", "FR"}: 0.60,
	{"ES", "IT"}: 0.65,
	{"ES", "FR"}: 0.60,
	{"IT", "FR"}: 0.60,
	{"EN", "DE"}: 0.55,
	{"EN", "NL"}: 0.55,
	{"DE", "NL"}: 0.65,
	{"EN", "FR"}: 0.55,
	{"PT", "EN"}: 0.55,
	{"PT", "DE"}: 0.50,
}

// IsCognate returns true if two words from different languages are likely cognates.
func IsCognate(a, langA, b, langB string) bool {
	na, nb := PhoneticNorm(a), PhoneticNorm(b)
	sim := LevenshteinSim(na, nb)
	for _, key := range [][2]string{{langA, langB}, {langB, langA}} {
		if t, ok := cognateThresholds[key]; ok {
			return sim >= t
		}
	}
	return sim >= 0.60
}
