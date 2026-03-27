// Package api implements the HTTP API for lexsent.
package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/kak/umcs/pkg/analyze"
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/morpheme"
	"github.com/kak/umcs/pkg/phon"
	"github.com/kak/umcs/pkg/sentiment"
	"github.com/kak/umcs/pkg/tokenizer"
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
	mux.HandleFunc("/stats", s.handleStats)
	mux.HandleFunc("/lookup", s.handleLookup)
	mux.HandleFunc("/lookup/batch", s.handleLookupBatch)
	mux.HandleFunc("/cognates", s.handleCognates)
	mux.HandleFunc("/etymo", s.handleEtymo)
	mux.HandleFunc("/analyze", s.handleAnalyze)
	mux.HandleFunc("/analyze/batch", s.handleAnalyzeBatch)
	mux.HandleFunc("/tokenize", s.handleTokenize)
	mux.HandleFunc("/vocab", s.handleVocab)
	mux.HandleFunc("/roots", s.handleRoots)
	mux.HandleFunc("/root/", s.handleRoot)
	mux.HandleFunc("/sentiment/decode", s.handleSentimentDecode)
	mux.HandleFunc("/emotion", s.handleEmotion)
	mux.HandleFunc("/drift", s.handleDrift)
	mux.HandleFunc("/crosslingual", s.handleCrossLingual)
	mux.HandleFunc("/search", s.handleSearch)
	mux.HandleFunc("/phonology", s.handlePhonology)
	mux.HandleFunc("/embeddings", s.handleEmbeddings)
	mux.HandleFunc("/ground", s.handleGround)
	return mux
}

// Listen starts the HTTP server on the given address.
func (s *Server) Listen(addr string) error {
	fmt.Printf("lexsent API listening on %s\n", addr)
	srv := &http.Server{
		Addr:         addr,
		Handler:      s.Handler(),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
	return srv.ListenAndServe()
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	jsonOK(w, map[string]any{
		"status":   "ok",
		"version":  fmt.Sprintf("%d", lexdb.Version),
		"checksum": fmt.Sprintf("0x%08X", s.lex.Stats.Checksum),
		"roots":    s.lex.Stats.RootCount,
		"words":    s.lex.Stats.WordCount,
		"langs":    strings.Join(s.lex.LangCoverage(s.lex.Stats.LangFlags), ","),
	})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	byLang := make(map[string]int)
	for _, w := range s.lex.Words {
		lang := lexdb.LangName(w.Lang)
		byLang[lang]++
	}

	avgCognates := 0.0
	if s.lex.Stats.RootCount > 0 {
		avgCognates = float64(s.lex.Stats.WordCount) / float64(s.lex.Stats.RootCount)
	}

	jsonOK(w, map[string]any{
		"roots":                s.lex.Stats.RootCount,
		"words":                s.lex.Stats.WordCount,
		"heap_bytes":           s.lex.Stats.HeapSize,
		"file_bytes":           s.lex.Stats.FileSize,
		"by_lang":              byLang,
		"avg_cognates_per_root": avgCognates,
	})
}

func (s *Server) handleLookup(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	word := r.URL.Query().Get("word")
	if word == "" {
		httpErr(w, "missing ?word=", 400)
		return
	}
	lang := r.URL.Query().Get("lang")
	var wr *lexdb.WordRecord
	if lang != "" {
		wr = s.lex.LookupWordInLang(word, strings.ToUpper(lang))
	} else {
		wr = s.lex.LookupWord(word)
	}
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

// handleLookupBatch looks up up to 100 words in a single request.
// Request: [{"word": "negative", "lang": "EN"}, ...]
// Response: array at same index — null entry for words not found.
func (s *Server) handleLookupBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		httpErr(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	var items []struct {
		Word string `json:"word"`
		Lang string `json:"lang"`
	}
	if err := json.Unmarshal(body, &items); err != nil {
		httpErr(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(items) > 100 {
		httpErr(w, "batch limit is 100 items", http.StatusBadRequest)
		return
	}

	results := make([]any, len(items))
	for i, item := range items {
		var wr *lexdb.WordRecord
		if item.Lang != "" {
			wr = s.lex.LookupWordInLang(item.Word, strings.ToUpper(item.Lang))
		} else {
			wr = s.lex.LookupWord(item.Word)
		}
		if wr == nil {
			results[i] = nil
			continue
		}
		root := s.lex.LookupRoot(wr.RootID)
		entry := map[string]any{
			"word":    s.lex.WordStr(wr),
			"word_id": wr.WordID,
			"root_id": wr.RootID,
			"lang":    lexdb.LangName(wr.Lang),
		}
		if root != nil {
			entry["root"] = s.lex.RootStr(root)
			entry["root_origin"] = s.lex.RootOrigin(root)
		}
		results[i] = entry
	}
	jsonOK(w, results)
}

func (s *Server) handleCognates(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
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
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
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
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
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
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
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
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
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
		sort.Slice(roots, func(i, j int) bool {
			return roots[i]["productivity"].(int) > roots[j]["productivity"].(int)
		})
	}

	total := len(roots)

	// Pagination: ?offset=N&limit=N
	offset := 0
	if s := r.URL.Query().Get("offset"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v >= 0 {
			offset = v
		}
	}
	if offset > len(roots) {
		offset = len(roots)
	}
	roots = roots[offset:]
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if lim, err := strconv.Atoi(limitStr); err == nil && lim > 0 && lim < len(roots) {
			roots = roots[:lim]
		}
	}

	jsonOK(w, map[string]any{"total": total, "count": len(roots), "roots": roots})
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Route: /root/{id} or /root/{id}/words
	path := strings.TrimPrefix(r.URL.Path, "/root/")
	parts := strings.SplitN(path, "/", 2)
	id, err := strconv.ParseUint(parts[0], 10, 32)
	if err != nil {
		httpErr(w, "invalid root id", http.StatusBadRequest)
		return
	}
	root := s.lex.LookupRoot(uint32(id))
	if root == nil {
		httpErr(w, fmt.Sprintf("root_id %d not found", id), http.StatusNotFound)
		return
	}

	// /root/{id}/words — morphological family grouped by language
	if len(parts) == 2 && parts[1] == "words" {
		var cognates []lexdb.WordRecord
		if int(root.FirstWordIdx) < len(s.lex.Words) {
			cognates = s.lex.Cognates(s.lex.Words[root.FirstWordIdx].WordID)
		}
		byLang := make(map[string][]map[string]any)
		for _, c := range cognates {
			lang := lexdb.LangName(c.Lang)
			sent := sentiment.Decode(c.Sentiment)
			byLang[lang] = append(byLang[lang], map[string]any{
				"word":    s.lex.WordStr(&c),
				"word_id": c.WordID,
				"polarity": sent["polarity"],
				"intensity": sent["intensity"],
			})
		}
		jsonOK(w, map[string]any{
			"root_id": root.RootID,
			"root":    s.lex.RootStr(root),
			"origin":  s.lex.RootOrigin(root),
			"meaning": s.lex.RootMeaning(root),
			"words":   byLang,
		})
		return
	}

	jsonOK(w, map[string]any{
		"root_id":        root.RootID,
		"root":           s.lex.RootStr(root),
		"origin":         s.lex.RootOrigin(root),
		"meaning":        s.lex.RootMeaning(root),
		"word_count":     root.WordCount,
		"lang_coverage":  strings.Join(s.lex.LangCoverage(root.LangCoverage), ","),
		"parent_root_id": root.ParentRootID,
	})
}

// handleAnalyzeBatch processes up to 100 texts in a single request.
func (s *Server) handleAnalyzeBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		httpErr(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}

	var items []struct {
		Text string `json:"text"`
		Lang string `json:"lang"`
	}
	if err := json.Unmarshal(body, &items); err != nil {
		httpErr(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(items) > 100 {
		httpErr(w, "batch limit is 100 items", http.StatusBadRequest)
		return
	}

	results := make([]map[string]any, len(items))
	for i, item := range items {
		result := analyze.Analyze(s.lex, item.Text)
		results[i] = map[string]any{
			"text":    item.Text,
			"verdict": result.Verdict,
			"score":   result.TotalScore,
			"matched": result.Matched,
			"total":   result.Total,
		}
	}
	jsonOK(w, results)
}

// handleVocab exports the vocabulary in a HuggingFace-compatible tokenizer format.
func (s *Server) handleVocab(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	vocab := make(map[string]uint32, len(s.lex.Words))
	rootMap := make(map[uint32]uint32, len(s.lex.Words))
	var maxRootID uint32
	for _, wr := range s.lex.Words {
		word := s.lex.WordStr(&wr)
		if word != "" {
			vocab[word] = wr.WordID
			rootMap[wr.WordID] = morpheme.RootOf(wr.WordID)
		}
		if morpheme.RootOf(wr.WordID) > maxRootID {
			maxRootID = morpheme.RootOf(wr.WordID)
		}
	}

	jsonOK(w, map[string]any{
		"version": "1.0",
		"model": map[string]any{
			"type":        "morpheme",
			"max_root_id": maxRootID,
			"vocab_size":  len(vocab),
		},
		"vocab":    vocab,
		"root_map": rootMap,
	})
}

func (s *Server) handleSentimentDecode(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
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
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("jsonOK encode error: %v", err)
	}
}

func httpErr(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(map[string]string{"error": msg}); err != nil {
		log.Printf("httpErr encode error: %v", err)
	}
}

// handleEmotion decomposes text into Plutchik emotion profile.
// GET /emotion?text=...
func (s *Server) handleEmotion(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	text := r.URL.Query().Get("text")
	if text == "" && r.Method == http.MethodPost {
		var err error
		text, err = readBody(r)
		if err != nil {
			httpErr(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
	}
	if text == "" {
		httpErr(w, "missing text", 400)
		return
	}

	result := analyze.Analyze(s.lex, text)
	ep := analyze.EmotionDecompose(result, s.lex)

	jsonOK(w, map[string]any{
		"text":     text,
		"verdict":  result.Verdict,
		"score":    result.TotalScore,
		"dominant": ep.Dominant,
		"emotions": map[string]float64{
			"joy":      ep.Joy,
			"trust":    ep.Trust,
			"fear":     ep.Fear,
			"anger":    ep.Anger,
			"sadness":  ep.Sadness,
			"surprise": ep.Surprise,
			"disgust":  ep.Disgust,
			"serenity": ep.Serenity,
		},
	})
}

// handleDrift analyzes sentiment trajectory across text.
// GET /drift?text=...
func (s *Server) handleDrift(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	text := r.URL.Query().Get("text")
	if text == "" && r.Method == http.MethodPost {
		var err error
		text, err = readBody(r)
		if err != nil {
			httpErr(w, "read body: "+err.Error(), http.StatusBadRequest)
			return
		}
	}
	if text == "" {
		httpErr(w, "missing text", 400)
		return
	}

	result := analyze.Analyze(s.lex, text)
	points := analyze.DetectDrift(result)
	summary := analyze.SummarizeDrift(points)

	jsonOK(w, map[string]any{
		"text":    text,
		"verdict": result.Verdict,
		"pattern": summary.Pattern,
		"summary": map[string]any{
			"max_positive": summary.MaxPositive,
			"max_negative": summary.MaxNegative,
			"volatility":  summary.Volatility,
			"shifts":      summary.Shifts,
		},
		"points": points,
	})
}

// handleCrossLingual returns cross-lingual sentiment consensus for a word.
// GET /crosslingual?word=...
func (s *Server) handleCrossLingual(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	word := r.URL.Query().Get("word")
	if word == "" {
		httpErr(w, "missing ?word=", 400)
		return
	}

	polarity, confidence, nLangs := analyze.CrossLingualScore(s.lex, word)
	jsonOK(w, map[string]any{
		"word":       word,
		"polarity":   polarity,
		"confidence": confidence,
		"languages":  nLangs,
	})
}

// handleSearch performs prefix search across the lexicon.
// GET /search?q=terr&limit=20&lang=EN
func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query().Get("q")
	if q == "" {
		httpErr(w, "missing ?q=", 400)
		return
	}
	limit := 20
	if ls := r.URL.Query().Get("limit"); ls != "" {
		if v, err := strconv.Atoi(ls); err == nil && v > 0 && v <= 200 {
			limit = v
		}
	}
	lang := strings.ToUpper(r.URL.Query().Get("lang"))

	results := s.lex.PrefixSearch(q, limit*2) // over-collect to filter by lang
	list := make([]map[string]any, 0, limit)
	for _, wr := range results {
		if lang != "" && lexdb.LangName(wr.Lang) != lang {
			continue
		}
		decoded := sentiment.Decode(wr.Sentiment)
		entry := map[string]any{
			"word":      s.lex.WordStr(&wr),
			"word_id":   wr.WordID,
			"root_id":   morpheme.RootOf(wr.WordID),
			"lang":      lexdb.LangName(wr.Lang),
			"polarity":  decoded["polarity"],
			"intensity": decoded["intensity"],
		}
		pron := s.lex.WordPron(&wr)
		if pron != "" {
			entry["ipa"] = pron
		}
		list = append(list, entry)
		if len(list) >= limit {
			break
		}
	}
	jsonOK(w, map[string]any{"query": q, "count": len(list), "results": list})
}

// handlePhonology returns phonological analysis for a word.
// GET /phonology?word=terrible
func (s *Server) handlePhonology(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	word := r.URL.Query().Get("word")
	if word == "" {
		httpErr(w, "missing ?word=", 400)
		return
	}
	lang := r.URL.Query().Get("lang")
	var wr *lexdb.WordRecord
	if lang != "" {
		wr = s.lex.LookupWordInLang(word, strings.ToUpper(lang))
	} else {
		wr = s.lex.LookupWord(word)
	}
	if wr == nil {
		httpErr(w, fmt.Sprintf("not found: %q", word), 404)
		return
	}

	jsonOK(w, map[string]any{
		"word":      s.lex.WordStr(wr),
		"lang":      lexdb.LangName(wr.Lang),
		"ipa":       s.lex.WordPron(wr),
		"syllables": phon.Syllables(wr.Flags),
		"stress":    phon.StressName(phon.Stress(wr.Flags)),
		"valency":   phon.ValencyName(phon.Valency(wr.Flags)),
	})
}

// handleEmbeddings exports root-indexed semantic vectors for LLM integration.
// Each root maps to a fixed-size vector derived from its words' sentiment dimensions.
// GET /embeddings?format=json&limit=1000
//
// Use case: LLMs can use these vectors as pre-computed semantic anchors to
// ground their outputs. Since cognates share root_id, a single vector covers
// all languages for a morphological family — zero-shot cross-lingual transfer.
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	limit := 1000
	if ls := r.URL.Query().Get("limit"); ls != "" {
		if v, err := strconv.Atoi(ls); err == nil && v > 0 {
			limit = v
		}
	}

	type embedding struct {
		RootID     uint32    `json:"root_id"`
		Root       string    `json:"root"`
		Meaning    string    `json:"meaning"`
		Vector     []float64 `json:"vector"`
		LangCount  int       `json:"lang_count"`
		WordCount  int       `json:"word_count"`
	}

	embeddings := make([]embedding, 0, limit)
	for _, root := range s.lex.Roots {
		if len(embeddings) >= limit {
			break
		}
		// Skip synthetic auto-bucketed roots.
		rootStr := s.lex.RootStr(&root)
		if strings.HasPrefix(rootStr, "_auto_") {
			continue
		}
		if root.WordCount == 0 {
			continue
		}

		// Compute mean vector from all words in this root family.
		// Vector: [polarity, intensity, arousal, dominance, aoa, concreteness,
		//          pos, role, syllables]
		var vec [9]float64
		count := 0
		start := int(root.FirstWordIdx)
		end := start + int(root.WordCount)
		if end > len(s.lex.Words) {
			end = len(s.lex.Words)
		}
		langSet := make(map[uint32]bool)
		for _, wr := range s.lex.Words[start:end] {
			sent := wr.Sentiment
			pol := float64(sentiment.Polarity(sent))
			if pol == 2 { pol = -1 } // NEGATIVE → -1
			if pol == 3 { pol = 0 }  // AMBIGUOUS → 0
			vec[0] += pol
			vec[1] += float64(sentiment.Intensity(sent)) / 4.0
			vec[2] += float64(sentiment.Arousal(sent)) / 3.0
			vec[3] += float64(sentiment.Dominance(sent)) / 3.0
			vec[4] += float64(sentiment.AOA(sent)) / 3.0
			if sent&(1<<28) != 0 { vec[5] += 1.0 }
			vec[6] += float64(sentiment.POS(sent)) / 7.0
			vec[7] += float64(sentiment.Role(sent)) / 11.0
			vec[8] += float64(phon.Syllables(wr.Flags)) / 15.0
			langSet[wr.Lang] = true
			count++
		}
		if count > 0 {
			for i := range vec {
				vec[i] /= float64(count)
			}
		}

		embeddings = append(embeddings, embedding{
			RootID:    root.RootID,
			Root:      rootStr,
			Meaning:   s.lex.RootMeaning(&root),
			Vector:    vec[:],
			LangCount: len(langSet),
			WordCount: count,
		})
	}

	jsonOK(w, map[string]any{
		"version":    "1.0",
		"dimensions": 9,
		"labels":     []string{"polarity", "intensity", "arousal", "dominance", "aoa", "concreteness", "pos", "role", "syllables"},
		"count":      len(embeddings),
		"embeddings": embeddings,
	})
}

// handleGround validates LLM-generated text against UMCS ground truth.
// POST /ground — body: {"text": "...", "expected_sentiment": "POSITIVE"}
//
// Use case: after an LLM generates text, call /ground to verify that the
// sentiment of the generated text matches expectations. This catches
// hallucinated sentiment ("I love this disaster!") where surface words
// contradict the declared intent.
func (s *Server) handleGround(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		httpErr(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		httpErr(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	var req struct {
		Text              string `json:"text"`
		ExpectedSentiment string `json:"expected_sentiment"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		httpErr(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Text == "" {
		httpErr(w, "missing text field", 400)
		return
	}

	result := analyze.Analyze(s.lex, req.Text)
	ep := analyze.EmotionDecompose(result, s.lex)
	points := analyze.DetectDrift(result)
	summary := analyze.SummarizeDrift(points)

	// Compute grounding quality.
	matches := req.ExpectedSentiment == "" || strings.EqualFold(result.Verdict, req.ExpectedSentiment)
	confidence := 0.0
	if result.Total > 0 {
		confidence = float64(result.Matched) / float64(result.Total)
	}

	// Token-level breakdown for debugging LLM outputs.
	conflicts := make([]map[string]string, 0)
	for _, t := range result.Tokens {
		if !t.Found {
			continue
		}
		tokenPol := t.Polarity
		if req.ExpectedSentiment != "" && !strings.EqualFold(tokenPol, req.ExpectedSentiment) && tokenPol != "NEUTRAL" {
			conflicts = append(conflicts, map[string]string{
				"word":     t.Surface,
				"polarity": tokenPol,
				"expected": req.ExpectedSentiment,
			})
		}
	}

	jsonOK(w, map[string]any{
		"text":               req.Text,
		"expected":           req.ExpectedSentiment,
		"actual_verdict":     result.Verdict,
		"actual_score":       result.TotalScore,
		"matches":            matches,
		"coverage":           confidence,
		"dominant_emotion":   ep.Dominant,
		"drift_pattern":      summary.Pattern,
		"conflicts":          conflicts,
		"conflict_count":     len(conflicts),
		"recommendation":     groundRecommendation(matches, len(conflicts), confidence),
	})
}

// groundRecommendation provides actionable feedback for LLM text quality.
func groundRecommendation(matches bool, conflicts int, coverage float64) string {
	if matches && conflicts == 0 {
		return "PASS: text sentiment aligns with expected intent"
	}
	if !matches && conflicts > 3 {
		return "FAIL: multiple conflicting sentiment tokens — rewrite recommended"
	}
	if !matches {
		return "WARN: overall sentiment diverges from intent — review phrasing"
	}
	if coverage < 0.3 {
		return "LOW_COVERAGE: most words not in lexicon — confidence is low"
	}
	return "PARTIAL: some conflicting tokens but overall sentiment matches"
}

func readBody(r *http.Request) (string, error) {
	if r.Body == nil {
		return "", nil
	}
	b, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1MB limit
	return strings.TrimSpace(string(b)), err
}
