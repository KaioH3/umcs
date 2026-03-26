package api_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kak/lex-sentiment/pkg/api"
	"github.com/kak/lex-sentiment/pkg/lexdb"
	"github.com/kak/lex-sentiment/pkg/seed"
	"github.com/kak/lex-sentiment/pkg/sentiment"
)

func buildAPILex(t *testing.T) *lexdb.Lexicon {
	t.Helper()
	dir := t.TempDir()

	notSent, _ := sentiment.Pack("NEUTRAL", "NONE", "NEGATION_MARKER", "GENERAL", []string{"NEGATION_MARKER"})
	terrSent, _ := sentiment.Pack("NEGATIVE", "STRONG", "EVALUATION", "GENERAL", nil)

	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
		{RootID: 10, RootStr: "terr", Origin: "LATIN", MeaningEN: "fear"},
		{RootID: 61, RootStr: "ne", Origin: "PIE", MeaningEN: "negation"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: 0x00120180},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: 0x00120180},
		{WordID: 8193, RootID: 2, Variant: 1, Word: "good", Lang: "EN", Norm: "good", Sentiment: 0x00130140},
		{WordID: 40961, RootID: 10, Variant: 1, Word: "terrible", Lang: "EN", Norm: "terrible", Sentiment: terrSent},
		{WordID: 249857, RootID: 61, Variant: 1, Word: "not", Lang: "EN", Norm: "not", Sentiment: notSent},
	}

	_, err := lexdb.Build(roots, words, filepath.Join(dir, "api_test.lsdb"))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(filepath.Join(dir, "api_test.lsdb"))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	return lex
}

func doRequest(h http.Handler, method, url, body string) *httptest.ResponseRecorder {
	var r *http.Request
	if body != "" {
		r = httptest.NewRequest(method, url, strings.NewReader(body))
	} else {
		r = httptest.NewRequest(method, url, nil)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	return rec
}

func decodeJSON(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatalf("decode JSON: %v (body=%s)", err, rec.Body.String())
	}
	return m
}

func TestHealthEndpoint(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/health", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if body["status"] != "ok" {
		t.Fatalf("want status=ok, got %v", body["status"])
	}
	if body["roots"].(float64) == 0 {
		t.Fatal("roots count should be > 0")
	}
	if body["words"].(float64) == 0 {
		t.Fatal("words count should be > 0")
	}
}

func TestLookupKnownWord(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/lookup?word=negative", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d (body=%s)", rec.Code, rec.Body.String())
	}
	body := decodeJSON(t, rec)
	if body["word_id"].(float64) != 4097 {
		t.Fatalf("want word_id=4097, got %v", body["word_id"])
	}
	if body["root_id"].(float64) != 1 {
		t.Fatalf("want root_id=1, got %v", body["root_id"])
	}
	if body["root"] != "negat" {
		t.Fatalf("want root=negat, got %v", body["root"])
	}
}

func TestLookupUnknownWord(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/lookup?word=xyzzy_unknown", "")

	if rec.Code != 404 {
		t.Fatalf("want 404, got %d", rec.Code)
	}
}

func TestLookupMissingParam(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/lookup", "")

	if rec.Code != 400 {
		t.Fatalf("want 400 for missing ?word=, got %d", rec.Code)
	}
}

func TestLookupWithCognates(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/lookup?word=negative", "")

	body := decodeJSON(t, rec)
	cognates, ok := body["cognates"].([]any)
	if !ok {
		t.Fatal("cognates field should be an array")
	}
	// "negative" (EN) and "negativo" (PT) share root_id=1
	if len(cognates) == 0 {
		t.Fatal("negative should have at least one cognate (negativo)")
	}
}

func TestCognatesEndpoint(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/cognates?word=negative", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	cognates, ok := body["cognates"].([]any)
	if !ok || len(cognates) == 0 {
		t.Fatal("cognates should be a non-empty array")
	}
}

func TestCognatesUnknownWord(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/cognates?word=xyzzy_unknown", "")

	if rec.Code != 404 {
		t.Fatalf("want 404, got %d", rec.Code)
	}
}

func TestAnalyzeViaBody(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "POST", "/analyze", "not terrible")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d (body=%s)", rec.Code, rec.Body.String())
	}
	body := decodeJSON(t, rec)
	if _, ok := body["verdict"]; !ok {
		t.Fatal("response must include verdict field")
	}
	if _, ok := body["score"]; !ok {
		t.Fatal("response must include score field")
	}
}

func TestAnalyzeViaQuery(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/analyze?text=good", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if body["verdict"] == nil {
		t.Fatal("verdict field missing")
	}
}

func TestAnalyzeMissingText(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "POST", "/analyze", "")

	if rec.Code != 400 {
		t.Fatalf("want 400 for empty text, got %d", rec.Code)
	}
}

func TestTokenizeEndpoint(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "POST", "/tokenize", "negative negativo")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)

	rootIDs, ok := body["root_ids"].([]any)
	if !ok || len(rootIDs) != 2 {
		t.Fatalf("want root_ids=[1,1], got %v", body["root_ids"])
	}
	// Both "negative" (EN) and "negativo" (PT) share root_id=1
	if rootIDs[0].(float64) != 1 || rootIDs[1].(float64) != 1 {
		t.Fatalf("negative/negativo should both map to root_id=1, got %v", rootIDs)
	}
}

func TestSentimentDecodeDecimal(t *testing.T) {
	// PolarityPositive = 0b01<<6 = 64
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/sentiment/decode?s=64", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if body["polarity"] != "POSITIVE" {
		t.Fatalf("s=64 should decode to POSITIVE, got %v", body["polarity"])
	}
}

func TestSentimentDecodeHex(t *testing.T) {
	// 0x80 = PolarityNegative
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/sentiment/decode?s=0x80", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if body["polarity"] != "NEGATIVE" {
		t.Fatalf("0x80 should decode to NEGATIVE, got %v", body["polarity"])
	}
}

func TestSentimentDecodeMissing(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/sentiment/decode", "")

	if rec.Code != 400 {
		t.Fatalf("want 400 for missing ?s=, got %d", rec.Code)
	}
}

func TestRootsEndpoint(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/roots", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	count, ok := body["count"].(float64)
	if !ok || count == 0 {
		t.Fatalf("want non-zero count, got %v", body["count"])
	}
}

func TestRootsProductiveSorted(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/roots?productive=true", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	roots, ok := body["roots"].([]any)
	if !ok || len(roots) < 2 {
		return // not enough roots to verify ordering
	}
	// First root should have productivity >= last root
	first := roots[0].(map[string]any)["productivity"].(float64)
	last := roots[len(roots)-1].(map[string]any)["productivity"].(float64)
	if first < last {
		t.Fatalf("productive=true should sort descending: first=%v last=%v", first, last)
	}
}

func TestRootByIDFound(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/root/1", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if body["root_id"].(float64) != 1 {
		t.Fatalf("want root_id=1, got %v", body["root_id"])
	}
	if body["root"] != "negat" {
		t.Fatalf("want root=negat, got %v", body["root"])
	}
}

func TestRootByIDNotFound(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/root/9999", "")

	if rec.Code != 404 {
		t.Fatalf("want 404 for unknown root_id, got %d", rec.Code)
	}
}

func TestRootByIDInvalid(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/root/abc", "")

	if rec.Code != 400 {
		t.Fatalf("want 400 for non-numeric root id, got %d", rec.Code)
	}
}

func TestEtymoEndpoint(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/etymo?word=negative", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if _, ok := body["etymology_chain"]; !ok {
		t.Fatal("response must include etymology_chain field")
	}
}
