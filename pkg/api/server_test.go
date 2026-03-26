package api_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kak/umcs/pkg/api"
	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
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

	_, err := lexdb.Build(roots, words, filepath.Join(dir, "api_test.umcs"))
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	lex, err := lexdb.Load(filepath.Join(dir, "api_test.umcs"))
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

// TestMethodValidation verifies that GET-only endpoints reject non-GET methods.
func TestMethodValidation(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()

	cases := []struct {
		name   string
		method string
		url    string
	}{
		{"health POST", "POST", "/health"},
		{"health DELETE", "DELETE", "/health"},
		{"lookup POST", "POST", "/lookup?word=negative"},
		{"cognates DELETE", "DELETE", "/cognates?word=negative"},
		{"roots PUT", "PUT", "/roots"},
		{"root PUT", "PUT", "/root/1"},
		{"sentiment/decode POST", "POST", "/sentiment/decode?s=64"},
		{"stats DELETE", "DELETE", "/stats"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := doRequest(h, tc.method, tc.url, "")
			if rec.Code != http.StatusMethodNotAllowed {
				t.Fatalf("%s %s: want 405, got %d", tc.method, tc.url, rec.Code)
			}
		})
	}
}

// TestAnalyzeBatch verifies the batch endpoint accepts up to 100 items.
func TestAnalyzeBatch(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()

	body := `[{"text":"not terrible","lang":"EN"},{"text":"good","lang":"EN"}]`
	rec := doRequest(h, "POST", "/analyze/batch", body)

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d (body=%s)", rec.Code, rec.Body.String())
	}

	var results []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &results); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("want 2 results, got %d", len(results))
	}
	for _, r := range results {
		if _, ok := r["verdict"]; !ok {
			t.Fatal("each result must have a verdict field")
		}
	}
}

func TestAnalyzeBatchMethodNotAllowed(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/analyze/batch", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", rec.Code)
	}
}

func TestAnalyzeBatchLimitExceeded(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	// Build a JSON array with 101 items
	var sb strings.Builder
	sb.WriteString("[")
	for i := range 101 {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(`{"text":"test","lang":"EN"}`)
	}
	sb.WriteString("]")
	rec := doRequest(h, "POST", "/analyze/batch", sb.String())
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 for batch > 100, got %d", rec.Code)
	}
}

func TestVocabEndpoint(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/vocab?format=json", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d (body=%s)", rec.Code, rec.Body.String())
	}
	body := decodeJSON(t, rec)
	if _, ok := body["vocab"]; !ok {
		t.Fatal("response must include vocab field")
	}
	if _, ok := body["root_map"]; !ok {
		t.Fatal("response must include root_map field")
	}
	model, ok := body["model"].(map[string]any)
	if !ok {
		t.Fatal("response must include model field")
	}
	if model["type"] != "morpheme" {
		t.Fatalf("model.type should be 'morpheme', got %v", model["type"])
	}
}

func TestStatsEndpoint(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/stats", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d (body=%s)", rec.Code, rec.Body.String())
	}
	body := decodeJSON(t, rec)
	if _, ok := body["roots"]; !ok {
		t.Fatal("stats must include roots field")
	}
	if _, ok := body["by_lang"]; !ok {
		t.Fatal("stats must include by_lang field")
	}
}

func TestRootWords(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/root/1/words", "")

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d (body=%s)", rec.Code, rec.Body.String())
	}
	body := decodeJSON(t, rec)
	if _, ok := body["words"]; !ok {
		t.Fatal("response must include words field")
	}
	if body["root_id"].(float64) != 1 {
		t.Fatalf("want root_id=1, got %v", body["root_id"])
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

// ── New feature tests ─────────────────────────────────────────────────────────

func TestHealthHasVersion(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/health", "")

	body := decodeJSON(t, rec)
	if body["version"] == nil {
		t.Fatal("/health must include 'version' field")
	}
	if body["checksum"] == nil {
		t.Fatal("/health must include 'checksum' field")
	}
}

func TestLookupWithDiacritic(t *testing.T) {
	// Build lex with a word that has accent — "négativo" should find it via Normalize
	lex := buildAPILex(t)
	h := api.New(lex).Handler()

	// "negative" is in the lex — "NEGATIVE" uppercase should also work via Normalize
	rec := doRequest(h, "GET", "/lookup?word=NEGATIVE", "")
	if rec.Code != 200 {
		t.Fatalf("case-insensitive lookup: want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if body["word_id"].(float64) != 4097 {
		t.Fatalf("want word_id=4097, got %v", body["word_id"])
	}
}

func TestLookupWithLang(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()

	// "negativo" is PT in the test lex
	rec := doRequest(h, "GET", "/lookup?word=negativo&lang=PT", "")
	if rec.Code != 200 {
		t.Fatalf("lang-specific lookup: want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if body["lang"] != "PT" {
		t.Fatalf("want lang=PT, got %v", body["lang"])
	}
}

func TestRootsPaginationLimit(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()

	rec := doRequest(h, "GET", "/roots?limit=2", "")
	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	roots, ok := body["roots"].([]any)
	if !ok {
		t.Fatal("roots should be an array")
	}
	if len(roots) > 2 {
		t.Fatalf("limit=2: want at most 2 roots, got %d", len(roots))
	}
	// total should reflect full count, count is paginated
	if body["total"] == nil {
		t.Fatal("response must include 'total' field")
	}
}

func TestRootsPaginationOffset(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()

	all := doRequest(h, "GET", "/roots", "")
	paged := doRequest(h, "GET", "/roots?offset=1&limit=100", "")

	allBody := decodeJSON(t, all)
	pagedBody := decodeJSON(t, paged)

	allRoots := allBody["roots"].([]any)
	pagedRoots := pagedBody["roots"].([]any)

	// total must be same
	if allBody["total"].(float64) != pagedBody["total"].(float64) {
		t.Fatalf("total should be same regardless of pagination: %v vs %v",
			allBody["total"], pagedBody["total"])
	}
	// paginated slice should be one shorter
	if len(pagedRoots) != len(allRoots)-1 {
		t.Fatalf("offset=1 should skip first root: want %d, got %d", len(allRoots)-1, len(pagedRoots))
	}
}

func TestLookupBatch(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()

	body := `[{"word":"negative","lang":"EN"},{"word":"xyzzy_notfound"},{"word":"good"}]`
	rec := doRequest(h, "POST", "/lookup/batch", body)

	if rec.Code != 200 {
		t.Fatalf("want 200, got %d (body=%s)", rec.Code, rec.Body.String())
	}

	var results []any
	if err := json.Unmarshal(rec.Body.Bytes(), &results); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("want 3 results, got %d", len(results))
	}
	// first: negative found
	if results[0] == nil {
		t.Fatal("results[0] (negative) should not be null")
	}
	// second: not found → null
	if results[1] != nil {
		t.Fatalf("results[1] (xyzzy) should be null, got %v", results[1])
	}
	// third: good found
	if results[2] == nil {
		t.Fatal("results[2] (good) should not be null")
	}
}

func TestLookupBatchMethodNotAllowed(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "GET", "/lookup/batch", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("want 405, got %d", rec.Code)
	}
}

func TestLookupBatchLimitExceeded(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	var sb strings.Builder
	sb.WriteString("[")
	for i := range 101 {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(`{"word":"test"}`)
	}
	sb.WriteString("]")
	rec := doRequest(h, "POST", "/lookup/batch", sb.String())
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 for batch > 100, got %d", rec.Code)
	}
}

func TestBatchEmptyArray(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "POST", "/analyze/batch", "[]")
	if rec.Code != 200 {
		t.Fatalf("empty batch: want 200, got %d", rec.Code)
	}
	var results []any
	if err := json.Unmarshal(rec.Body.Bytes(), &results); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("empty batch: want 0 results, got %d", len(results))
	}
}

func TestBatchEmptyTextItem(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	rec := doRequest(h, "POST", "/analyze/batch", `[{"text":""}]`)
	if rec.Code != 200 {
		t.Fatalf("batch with empty text: want 200, got %d", rec.Code)
	}
	var results []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &results); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	if results[0]["verdict"] != "NEUTRAL" {
		t.Fatalf("empty text → NEUTRAL, got %v", results[0]["verdict"])
	}
}

func TestAnalyzeSentimentCorrect(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()
	// "not terrible" → negation inverts NEGATIVE → positive score → POSITIVE
	rec := doRequest(h, "POST", "/analyze", "not terrible")
	if rec.Code != 200 {
		t.Fatalf("want 200, got %d", rec.Code)
	}
	body := decodeJSON(t, rec)
	if body["verdict"] != "POSITIVE" {
		t.Fatalf("'not terrible' should be POSITIVE, got %v (score=%v)", body["verdict"], body["score"])
	}
}

// TestServerSmoke exercises every endpoint in sequence to catch integration failures.
func TestServerSmoke(t *testing.T) {
	h := api.New(buildAPILex(t)).Handler()

	endpoints := []struct {
		method string
		url    string
		body   string
		want   int
	}{
		{"GET", "/health", "", 200},
		{"GET", "/stats", "", 200},
		{"GET", "/lookup?word=negative", "", 200},
		{"GET", "/lookup?word=negative&lang=EN", "", 200},
		{"GET", "/cognates?word=negative", "", 200},
		{"GET", "/etymo?word=negative", "", 200},
		{"POST", "/analyze", "not terrible", 200},
		{"GET", "/analyze?text=good", "", 200},
		{"POST", "/analyze/batch", `[{"text":"good"},{"text":"terrible"}]`, 200},
		{"POST", "/tokenize", "negative negativo", 200},
		{"GET", "/vocab", "", 200},
		{"GET", "/roots", "", 200},
		{"GET", "/roots?productive=true", "", 200},
		{"GET", "/roots?limit=2&offset=0", "", 200},
		{"GET", "/root/1", "", 200},
		{"GET", "/root/1/words", "", 200},
		{"GET", "/sentiment/decode?s=0x80", "", 200},
		{"POST", "/lookup/batch", `[{"word":"negative"}]`, 200},
	}

	for _, tc := range endpoints {
		t.Run(tc.method+" "+tc.url, func(t *testing.T) {
			rec := doRequest(h, tc.method, tc.url, tc.body)
			if rec.Code != tc.want {
				t.Fatalf("want %d, got %d (body=%s)", tc.want, rec.Code, rec.Body.String())
			}
		})
	}
}
