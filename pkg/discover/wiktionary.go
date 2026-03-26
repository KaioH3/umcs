package discover

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// Entry holds data extracted from a Wiktionary page.
type Entry struct {
	Word         string
	Lang         string   // our code: "EN", "PT", etc.
	Etymology    string   // raw etymology wikitext section
	AncestorWord string   // extracted ancestor (e.g. "negativus")
	AncestorLang string   // e.g. "LATIN", "PROTO_INDO_EUROPEAN"
	Translations []Trans  // translations found in the Translations section
	POS          string   // Part of speech (Adjective, Verb, Noun, Adverb)
	Definitions  []string // cleaned English definitions
}

// Trans is a single translation entry (word + language code).
type Trans struct {
	Word string
	Lang string // our code: "PT", "DE", etc.
}

// wiktLangMap maps Wiktionary ISO 639 codes → our language codes.
var wiktLangMap = map[string]string{
	"pt": "PT", "en": "EN", "es": "ES", "it": "IT",
	"de": "DE", "fr": "FR", "nl": "NL",
	"ko": "KO", "ja": "JA", "zh": "ZH",
}

// wiktEtymoLangMap maps Wiktionary etymology language codes → our origin labels.
var wiktEtymoLangMap = map[string]string{
	"la": "LATIN", "lat": "LATIN",
	"ine-pro": "PROTO_INDO_EUROPEAN",
	"gem-pro": "PROTO_GERMANIC",
	"grc":     "GREEK",
	"ar":      "ARABIC",
	"zh":      "CHINESE",
	"ang":     "OLD_ENGLISH",
	"fro":     "OLD_FRENCH",
	"ja":      "JAPANESE",
	"ko":      "KOREAN",
	"ojp":     "OLD_JAPANESE",
	"ltc":     "MIDDLE_CHINESE",
	"och":     "OLD_CHINESE",
}

// wiktLangNames maps our lang codes → Wiktionary section headers.
var wiktLangNames = map[string]string{
	"EN": "English", "PT": "Portuguese", "ES": "Spanish",
	"IT": "Italian", "DE": "German", "FR": "French", "NL": "Dutch",
	"KO": "Korean", "JA": "Japanese", "ZH": "Chinese",
}

var (
	// Etymology template patterns (Wiktionary standard templates).
	reInh  = regexp.MustCompile(`\{\{(?:inh|der|bor)\|[^|]+\|([^|]+)\|([^|}\s]+)`)
	reMent = regexp.MustCompile(`\{\{m\|([^|]+)\|([^|}\s]+)`)
	// Translation: {{t|pt|negativo}} or {{t+|pt|...}} or {{t-|pt|...}}
	reTrans = regexp.MustCompile(`\{\{t[+\-]?\|([a-z]{2})\|([^|}\n]+)`)
	// Section headers: ==English==, ===Etymology===, etc.
	reSection    = regexp.MustCompile(`(?m)^={2,4}[^=\n]+=`)
	reLangHeader = regexp.MustCompile(`(?m)^==([^=\n]+)==`)
	rePosHeader  = regexp.MustCompile(`(?m)^===([^=\n]+)===`)
)

const (
	wiktAPI         = "https://en.wiktionary.org/w/api.php"
	wiktRatePerSec  = 3 // requests per second
)

// rateLimiter ensures we do not exceed Wiktionary's rate limits.
var wiktRateLimiter = time.NewTicker(time.Second / wiktRatePerSec)

var wiktClient = &http.Client{Timeout: 10 * time.Second}

// Fetch retrieves and parses a Wiktionary entry for word/lang.
// Results are cached to $XDG_CACHE_HOME/lexsent/wikt/.
func Fetch(word, lang string) (*Entry, error) {
	if e, ok := loadCache(word, lang); ok {
		return e, nil
	}

	<-wiktRateLimiter.C // rate limit

	params := url.Values{
		"action": {"parse"},
		"page":   {word},
		"prop":   {"wikitext"},
		"format": {"json"},
	}
	req, err := http.NewRequest("GET", wiktAPI+"?"+params.Encode(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "LexSent/1.0 (morpheme discovery; https://github.com/kak/lex-sentiment)")

	resp, err := wiktClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("wiktionary fetch %q: %w", word, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("wiktionary read %q: %w", word, err)
	}

	var parsed struct {
		Parse struct {
			Wikitext struct {
				Star string `json:"*"`
			} `json:"wikitext"`
		} `json:"parse"`
		Error struct {
			Code string `json:"code"`
			Info string `json:"info"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("wiktionary parse %q: %w", word, err)
	}
	if parsed.Error.Code != "" {
		return nil, fmt.Errorf("wiktionary: %s — %s", parsed.Error.Code, parsed.Error.Info)
	}
	wikitext := parsed.Parse.Wikitext.Star
	if wikitext == "" {
		return nil, fmt.Errorf("wiktionary: empty page %q", word)
	}

	e := parseWikitext(word, lang, wikitext)
	saveCache(word, lang, e)
	return e, nil
}

// parseWikitext extracts etymology, translations, and definitions from raw wikitext.
func parseWikitext(word, lang, wikitext string) *Entry {
	e := &Entry{Word: word, Lang: lang}

	// Use the English section for etymology and translations (most complete on en.wiktionary.org).
	section := extractLangSection(wikitext, "English")
	if section == "" {
		// Fallback: try the word's own language section.
		if name := wiktLangNames[lang]; name != "" {
			section = extractLangSection(wikitext, name)
		}
	}
	if section == "" {
		section = wikitext
	}

	e.Etymology = extractSubsection(section, "Etymology")
	e.AncestorWord, e.AncestorLang = extractAncestor(e.Etymology)
	e.Translations = extractTranslations(extractSubsection(section, "Translations"))

	for _, pos := range []string{"Adjective", "Verb", "Noun", "Adverb"} {
		if s := extractSubsection(section, pos); s != "" {
			e.POS = pos
			e.Definitions = extractDefinitions(s)
			break
		}
	}
	return e
}

func extractLangSection(wikitext, targetLang string) string {
	lines := strings.Split(wikitext, "\n")
	var buf strings.Builder
	inTarget := false
	for _, line := range lines {
		if m := reLangHeader.FindStringSubmatch(line); m != nil {
			if strings.TrimSpace(m[1]) == targetLang {
				inTarget = true
				continue
			} else if inTarget {
				break // entered next language section
			}
		}
		if inTarget {
			buf.WriteString(line)
			buf.WriteByte('\n')
		}
	}
	return buf.String()
}

func extractSubsection(section, name string) string {
	re := regexp.MustCompile(`(?m)^={2,4}` + regexp.QuoteMeta(name) + `={2,4}`)
	loc := re.FindStringIndex(section)
	if loc == nil {
		return ""
	}
	rest := section[loc[1]:]
	next := reSection.FindStringIndex(rest)
	if next == nil {
		return rest
	}
	return rest[:next[0]]
}

func extractAncestor(etym string) (word, lang string) {
	// Try {{inh|en|la|negativus}}, {{der|en|la|...}}, {{bor|en|la|...}}
	if m := reInh.FindStringSubmatch(etym); m != nil {
		lc := strings.TrimSpace(m[1])
		w := cleanWikitextArg(m[2])
		if mapped, ok := wiktEtymoLangMap[lc]; ok && w != "" {
			return w, mapped
		}
	}
	// Fallback: {{m|la|negare}}
	if m := reMent.FindStringSubmatch(etym); m != nil {
		lc := strings.TrimSpace(m[1])
		w := cleanWikitextArg(m[2])
		if mapped, ok := wiktEtymoLangMap[lc]; ok && w != "" {
			return w, mapped
		}
	}
	return "", ""
}

func extractTranslations(transSec string) []Trans {
	matches := reTrans.FindAllStringSubmatch(transSec, -1)
	var result []Trans
	seen := map[string]bool{}
	for _, m := range matches {
		lc := m[1]
		w := cleanWikitextArg(m[2])
		if w == "" {
			continue
		}
		mapped, ok := wiktLangMap[lc]
		if !ok {
			continue
		}
		key := mapped + "_" + w
		if seen[key] {
			continue
		}
		seen[key] = true
		result = append(result, Trans{Word: w, Lang: mapped})
	}
	return result
}

var (
	reWikiLink = regexp.MustCompile(`\[\[(?:[^\]|]+\|)?([^\]]+)\]\]`)
	reTemplate = regexp.MustCompile(`\{\{[^}]*\}\}`)
	reHTMLTag  = regexp.MustCompile(`<[^>]+>`)
)

func extractDefinitions(posSection string) []string {
	var defs []string
	for _, line := range strings.Split(posSection, "\n") {
		if !strings.HasPrefix(line, "# ") {
			continue
		}
		def := line[2:]
		def = reWikiLink.ReplaceAllString(def, "$1")
		def = reTemplate.ReplaceAllString(def, "")
		def = reHTMLTag.ReplaceAllString(def, "")
		def = strings.TrimSpace(def)
		if def != "" && !strings.HasPrefix(def, "#") {
			defs = append(defs, def)
		}
	}
	return defs
}

func cleanWikitextArg(s string) string {
	s = strings.TrimSpace(s)
	// Remove inline params (e.g. |alt=foo, |tr=...)
	if idx := strings.Index(s, "|"); idx >= 0 {
		s = s[:idx]
	}
	s = strings.Trim(s, "*- \t")
	return strings.TrimSpace(s)
}

// --- disk cache ---

func wiktCacheDir() string {
	base := os.Getenv("XDG_CACHE_HOME")
	if base == "" {
		home, _ := os.UserHomeDir()
		base = filepath.Join(home, ".cache")
	}
	return filepath.Join(base, "lexsent", "wikt")
}

func wiktCachePath(word, lang string) string {
	safe := url.QueryEscape(strings.ToLower(word))
	return filepath.Join(wiktCacheDir(), lang+"_"+safe+".json")
}

func loadCache(word, lang string) (*Entry, bool) {
	data, err := os.ReadFile(wiktCachePath(word, lang))
	if err != nil {
		return nil, false
	}
	var e Entry
	if json.Unmarshal(data, &e) != nil {
		return nil, false
	}
	return &e, true
}

func saveCache(word, lang string, e *Entry) {
	if err := os.MkdirAll(wiktCacheDir(), 0o755); err != nil {
		return
	}
	data, err := json.Marshal(e)
	if err != nil {
		return
	}
	_ = os.WriteFile(wiktCachePath(word, lang), data, 0o644)
}
