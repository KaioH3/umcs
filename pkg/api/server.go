// Package api implements the HTTP API for lexsent.
package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/kak/lex-sentiment/pkg/analyze"
	"github.com/kak/lex-sentiment/pkg/lexdb"
	"github.com/kak/lex-sentiment/pkg/sentiment"
	"github.com/kak/lex-sentiment/pkg/tokenizer"
)

// Server holds the loaded lexicon and serves HTTP requests.
type Server struct {
	lex *lexdb.Lexicon
}

// New creates a new API server with the given lexicon.
func New(lex *lexdb.Lexicon) *Server {
	return &Server{lex: lex}
}

// Handler returns the HTTP handler for the server. Used for testing.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/lookup", s.handleLookup)
	mux.HandleFunc("/cognates", s.handleCognates)
	mux.HandleFunc("/etymo", s.handleEtymo)
	mux.HandleFunc("/analyze", s.handleAnalyze)
	mux.HandleFunc("/tokenize", s.handleTokenize)
	mux.HandleFunc("/roots", s.handleRoots)
	mux.HandleFunc("/root/", s.handleRoot)
	mux.HandleFunc("/sentiment/decode", s.handleSentimentDecode)
	return mux
}

// Listen starts the HTTP server on the given address.
func (s *Server) Listen(addr string) error {
	fmt.Printf("lexsent API listening on %s\n", addr)
	return http.ListenAndServe(addr, s.Handler())
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	jsonOK(w, map[string]any{
		"status": "ok",
		"roots":  s.lex.Stats.RootCount,
		"words":  s.lex.Stats.WordCount,
		"langs":  strings.Join(s.lex.LangCoverage(s.lex.Stats.LangFlags), ","),
	})
}

func (s *Server) handleLookup(w http.ResponseWriter, r *http.Request) {
	word := r.URL.Query().Get("word")
	if word == "" {
		httpErr(w, "missing ?word=", 400)
		return
	}
	wr := s.lex.LookupWord(word)
	if wr == nil {
		httpErr(w, fmt.Sprintf("not found: %q", word), 404)
		return
	}
	root := s.lex.LookupRoot(wr.RootID)
	sent := sentiment.Decode(wr.Sentiment)

	cognates := s.lex.Cognates(wr.WordID)
	cogList := make([]map[string]any, 0, len(cognates))
	for _, c := range cognates {
		if c.WordID == wr.WordID {
			continue
		}
		cogList = append(cogList, map[string]any{
			"word":    s.lex.WordStr(&c),
			"lang":    lexdb.LangName(c.Lang),
			"word_id": c.WordID,
		})
	}

	resp := map[string]any{
		"word":      s.lex.WordStr(wr),
		"word_id":   wr.WordID,
		"root_id":   wr.RootID,
		"lang":      lexdb.LangName(wr.Lang),
		"sentiment": sent,
		"freq_rank": wr.FreqRank,
		"cognates":  cogList,
	}
	if root != nil {
		resp["root"] = s.lex.RootStr(root)
		resp["root_origin"] = s.lex.RootOrigin(root)
		resp["root_meaning"] = s.lex.RootMeaning(root)
	}
	jsonOK(w, resp)
}

func (s *Server) handleCognates(w http.ResponseWriter, r *http.Request) {
	word := r.URL.Query().Get("word")
	if word == "" {
		httpErr(w, "missing ?word=", 400)
		return
	}
	wr := s.lex.LookupWord(word)
	if wr == nil {
		httpErr(w, fmt.Sprintf("not found: %q", word), 404)
		return
	}
	root := s.lex.LookupRoot(wr.RootID)
	cognates := s.lex.Cognates(wr.WordID)

	list := make([]map[string]any, 0, len(cognates))
	for _, c := range cognates {
		list = append(list, map[string]any{
			"word":    s.lex.WordStr(&c),
			"lang":    lexdb.LangName(c.Lang),
			"word_id": c.WordID,
		})
	}

	resp := map[string]any{"cognates": list}
	if root != nil {
		resp["root"] = s.lex.RootStr(root)
		resp["root_id"] = root.RootID
		resp["origin"] = s.lex.RootOrigin(root)
		resp["meaning"] = s.lex.RootMeaning(root)
	}
	jsonOK(w, resp)
}

func (s *Server) handleEtymo(w http.ResponseWriter, r *http.Request) {
	word := r.URL.Query().Get("word")
	if word == "" {
		httpErr(w, "missing ?word=", 400)
		return
	}
	wr := s.lex.LookupWord(word)
	if wr == nil {
		httpErr(w, fmt.Sprintf("not found: %q", word), 404)
		return
	}
	chain := s.lex.EtymologyChain(wr.RootID)
	list := make([]map[string]any, len(chain))
	for i, r := range chain {
		list[i] = map[string]any{
			"root_id": r.RootID,
			"root":    s.lex.RootStr(&r),
			"origin":  s.lex.RootOrigin(&r),
			"meaning": s.lex.RootMeaning(&r),
		}
	}
	jsonOK(w, map[string]any{"word": word, "etymology_chain": list})
}

func (s *Server) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	text, err := readBody(r)
	if err != nil {
		httpErr(w, "read body: "+err.Error(), 400)
		return
	}
	if text == "" {
		text = r.URL.Query().Get("text")
	}
	if text == "" {
		httpErr(w, "send text in body or ?text=", 400)
		return
	}

	result := analyze.Analyze(s.lex, text)

	tokens := make([]map[string]any, 0, len(result.Tokens))
	for _, t := range result.Tokens {
		tok := map[string]any{"surface": t.Surface, "found": t.Found}
		if t.Found {
			tok["word_id"] = t.WordID
			tok["root_id"] = t.RootID
			tok["root"] = t.RootStr
			tok["lang"] = t.Lang
			tok["polarity"] = t.Polarity
			tok["intensity"] = t.Intensity
			tok["role"] = t.Role
			tok["weight"] = t.Weight
			tok["negated"] = t.Negated
			tok["amplified"] = t.Amplified
		}
		tokens = append(tokens, tok)
	}

	jsonOK(w, map[string]any{
		"text":    text,
		"verdict": result.Verdict,
		"score":   result.TotalScore,
		"matched": result.Matched,
		"total":   result.Total,
		"tokens":  tokens,
	})
}

func (s *Server) handleTokenize(w http.ResponseWriter, r *http.Request) {
	text, err := readBody(r)
	if err != nil {
		httpErr(w, "read body: "+err.Error(), 400)
		return
	}
	if text == "" {
		text = r.URL.Query().Get("text")
	}
	if text == "" {
		httpErr(w, "send text in body or ?text=", 400)
		return
	}

	tokens := tokenizer.Tokenize(s.lex, text)
	ids := make([]uint32, len(tokens))
	rootIDs := make([]uint32, len(tokens))
	list := make([]map[string]any, len(tokens))

	for i, t := range tokens {
		ids[i] = t.WordID
		rootIDs[i] = t.RootID
		list[i] = map[string]any{
			"surface":       t.Surface,
			"word_id":       t.WordID,
			"root_id":       t.RootID,
			"sentiment_hex": fmt.Sprintf("0x%08X", t.Sentiment),
			"known":         t.Known,
		}
	}

	jsonOK(w, map[string]any{
		"text":     text,
		"word_ids": ids,
		"root_ids": rootIDs,
		"tokens":   list,
	})
}

func (s *Server) handleRoots(w http.ResponseWriter, r *http.Request) {
	productive := r.URL.Query().Get("productive") == "true"

	roots := make([]map[string]any, 0, len(s.lex.Roots))
	for _, root := range s.lex.Roots {
		langCount := 0
		for i := uint32(0); i < 11; i++ {
			if root.LangCoverage&(1<<i) != 0 {
				langCount++
			}
		}
		roots = append(roots, map[string]any{
			"root_id":       root.RootID,
			"root":          s.lex.RootStr(&root),
			"origin":        s.lex.RootOrigin(&root),
			"meaning":       s.lex.RootMeaning(&root),
			"word_count":    root.WordCount,
			"lang_count":    langCount,
			"productivity":  int(root.WordCount) * langCount,
			"lang_coverage": strings.Join(s.lex.LangCoverage(root.LangCoverage), ","),
		})
	}

	if productive {
		// Sort by productivity descending (simple insertion sort for small N)
		for i := range roots {
			for j := i + 1; j < len(roots); j++ {
				if roots[j]["productivity"].(int) > roots[i]["productivity"].(int) {
					roots[i], roots[j] = roots[j], roots[i]
				}
			}
		}
	}

	jsonOK(w, map[string]any{"count": len(roots), "roots": roots})
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	idStr := strings.TrimPrefix(r.URL.Path, "/root/")
	id, err := strconv.ParseUint(idStr, 10, 32)
	if err != nil {
		httpErr(w, "invalid root id", 400)
		return
	}
	root := s.lex.LookupRoot(uint32(id))
	if root == nil {
		httpErr(w, fmt.Sprintf("root_id %d not found", id), 404)
		return
	}
	jsonOK(w, map[string]any{
		"root_id":       root.RootID,
		"root":          s.lex.RootStr(root),
		"origin":        s.lex.RootOrigin(root),
		"meaning":       s.lex.RootMeaning(root),
		"word_count":    root.WordCount,
		"lang_coverage": strings.Join(s.lex.LangCoverage(root.LangCoverage), ","),
		"parent_root_id": root.ParentRootID,
	})
}

func (s *Server) handleSentimentDecode(w http.ResponseWriter, r *http.Request) {
	sStr := r.URL.Query().Get("s")
	if sStr == "" {
		httpErr(w, "missing ?s=", 400)
		return
	}
	var val uint64
	var err error
	if strings.HasPrefix(sStr, "0x") || strings.HasPrefix(sStr, "0X") {
		val, err = strconv.ParseUint(sStr[2:], 16, 32)
	} else {
		val, err = strconv.ParseUint(sStr, 10, 32)
	}
	if err != nil {
		httpErr(w, "invalid sentiment value", 400)
		return
	}
	jsonOK(w, sentiment.Decode(uint32(val)))
}

func jsonOK(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func httpErr(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func readBody(r *http.Request) (string, error) {
	if r.Body == nil {
		return "", nil
	}
	b, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1MB limit
	return strings.TrimSpace(string(b)), err
}
