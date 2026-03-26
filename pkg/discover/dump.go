package discover

import (
	"compress/bzip2"
	"encoding/xml"
	"io"
	"os"
	"strings"
)

// WikiPage is a single page from a Wiktionary XML dump.
type WikiPage struct {
	Title string // page title, e.g. "love"
	Text  string // raw wikitext content
}

// errStopScan is a sentinel used to stop ScanDump early.
var errStopScan = io.EOF

// ScanDump streams a Wiktionary XML dump file (.xml or .xml.bz2), calling fn
// for each main-namespace article page (ns=0, no ":" in title).
// Returns nil on success. If fn returns io.EOF, scanning stops without error.
func ScanDump(path string, fn func(WikiPage) error) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var r io.Reader = f
	if strings.HasSuffix(strings.ToLower(path), ".bz2") {
		r = bzip2.NewReader(f)
	}

	dec := xml.NewDecoder(r)
	dec.Strict = false // tolerate minor wikitext encoding quirks

	// Streaming state: track which element we're inside.
	var (
		inPage  bool
		ns      string
		title   strings.Builder
		text    strings.Builder
		current string // "title" | "ns" | "text" | ""
	)

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			// Skip malformed XML tokens common in large wiki dumps.
			continue
		}

		switch t := tok.(type) {
		case xml.StartElement:
			switch t.Name.Local {
			case "page":
				inPage = true
				ns = ""
				title.Reset()
				text.Reset()
			case "title":
				if inPage {
					current = "title"
				}
			case "ns":
				if inPage {
					current = "ns"
				}
			case "text":
				if inPage {
					current = "text"
				}
			}

		case xml.EndElement:
			switch t.Name.Local {
			case "page":
				if inPage && ns == "0" {
					t := strings.TrimSpace(title.String())
					// Skip pages with ":" (templates, categories, etc.) or empty text.
					if t != "" && !strings.Contains(t, ":") {
						txt := text.String()
						if txt != "" {
							if err := fn(WikiPage{Title: t, Text: txt}); err != nil {
								if err == io.EOF {
									return nil // clean stop
								}
								return err
							}
						}
					}
				}
				inPage = false
				current = ""
			case "title", "ns", "text":
				current = ""
			}

		case xml.CharData:
			if !inPage {
				continue
			}
			switch current {
			case "title":
				title.Write(t)
			case "ns":
				ns += string(t)
			case "text":
				text.Write(t)
			}
		}
	}
	return nil
}

// ParseDumpPage converts a WikiPage into a slice of Entry values, one per
// relevant language section found in the wikitext.
//
// Strategy:
//  1. Always parse the English section (best translation coverage on en.wiktionary.org).
//  2. Also parse each non-English target-lang section to capture direct definitions
//     and etymology for words whose Wiktionary article is not English-centric.
func ParseDumpPage(page WikiPage, targetLangs []string) []Entry {
	var entries []Entry

	// 1. English section — most likely to have cross-lingual Translations table.
	e := parseWikitext(page.Title, "EN", page.Text)
	if e != nil && (len(e.Translations) > 0 || len(e.Definitions) > 0 || e.AncestorWord != "") {
		entries = append(entries, *e)
	}

	// 2. Non-English target sections — extract definitions/etymology when present.
	for _, lang := range targetLangs {
		if lang == "EN" {
			continue
		}
		langName, ok := wiktLangNames[lang]
		if !ok {
			continue
		}
		section := extractLangSection(page.Text, langName)
		if section == "" {
			continue
		}
		ne := &Entry{Word: page.Title, Lang: lang}
		ne.Etymology = extractSubsection(section, "Etymology")
		ne.AncestorWord, ne.AncestorLang = extractAncestor(ne.Etymology)
		for _, pos := range []string{"Adjective", "Verb", "Noun", "Adverb"} {
			if s := extractSubsection(section, pos); s != "" {
				ne.POS = pos
				ne.Definitions = extractDefinitions(s)
				break
			}
		}
		if len(ne.Definitions) > 0 || ne.AncestorWord != "" {
			entries = append(entries, *ne)
		}
	}

	return entries
}
