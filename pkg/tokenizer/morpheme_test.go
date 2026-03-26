package tokenizer_test

import (
	"path/filepath"
	"testing"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/tokenizer"
)

func buildTokenLex(t *testing.T) *lexdb.Lexicon {
	t.Helper()
	dir := t.TempDir()

	roots := []seed.Root{
		{RootID: 1, RootStr: "negat", Origin: "LATIN", MeaningEN: "deny"},
		{RootID: 2, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "negative", Lang: "EN", Norm: "negative", Sentiment: 0x00120180},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "negativo", Lang: "PT", Norm: "negativo", Sentiment: 0x00120180},
		{WordID: 8193, RootID: 2, Variant: 1, Word: "bom", Lang: "PT", Norm: "bom", Sentiment: 0x00130140},
		{WordID: 8194, RootID: 2, Variant: 2, Word: "good", Lang: "EN", Norm: "good", Sentiment: 0x00130140},
	}

	_, err := lexdb.Build(roots, words, filepath.Join(dir, "tok.lsdb"))
	if err != nil {
		t.Fatal(err)
	}
	lex, err := lexdb.Load(filepath.Join(dir, "tok.lsdb"))
	if err != nil {
		t.Fatal(err)
	}
	return lex
}

func TestTokenizeKnownWords(t *testing.T) {
	lex := buildTokenLex(t)
	tokens := tokenizer.Tokenize(lex, "negative bom")

	if len(tokens) != 2 {
		t.Fatalf("want 2 tokens, got %d", len(tokens))
	}
	if !tokens[0].Known || tokens[0].WordID != 4097 {
		t.Fatalf("token[0]: expected negative(4097), got %+v", tokens[0])
	}
	if !tokens[1].Known || tokens[1].WordID != 8193 {
		t.Fatalf("token[1]: expected bom(8193), got %+v", tokens[1])
	}
}

func TestTokenizeOOV(t *testing.T) {
	lex := buildTokenLex(t)
	tokens := tokenizer.Tokenize(lex, "negative xyzzy good")

	if len(tokens) != 3 {
		t.Fatalf("want 3 tokens, got %d", len(tokens))
	}
	if tokens[1].Known {
		t.Fatal("xyzzy should be OOV")
	}
	if tokens[1].WordID != 0 {
		t.Fatal("OOV token should have WordID=0")
	}
}

func TestCrossLinguisticRootIDs(t *testing.T) {
	lex := buildTokenLex(t)
	// "negative" (EN) and "negativo" (PT) must share root_id=1
	enTokens := tokenizer.Tokenize(lex, "negative")
	ptTokens := tokenizer.Tokenize(lex, "negativo")

	if enTokens[0].RootID != ptTokens[0].RootID {
		t.Fatalf("negative and negativo must share root_id: EN=%d PT=%d",
			enTokens[0].RootID, ptTokens[0].RootID)
	}
	if enTokens[0].RootID != 1 {
		t.Fatalf("root_id should be 1, got %d", enTokens[0].RootID)
	}
}

func TestTokenizeToRootIDs(t *testing.T) {
	lex := buildTokenLex(t)
	// EN text
	enIDs := tokenizer.TokenizeToRootIDs(lex, "negative good")
	// PT text (different words, same roots)
	ptIDs := tokenizer.TokenizeToRootIDs(lex, "negativo bom")

	if len(enIDs) != 2 || len(ptIDs) != 2 {
		t.Fatalf("wrong token counts: en=%d pt=%d", len(enIDs), len(ptIDs))
	}
	if enIDs[0] != ptIDs[0] {
		t.Fatalf("negative/negativo must share root_id: %d vs %d", enIDs[0], ptIDs[0])
	}
	if enIDs[1] != ptIDs[1] {
		t.Fatalf("good/bom must share root_id: %d vs %d", enIDs[1], ptIDs[1])
	}
}

func TestDeduplicationScore(t *testing.T) {
	lex := buildTokenLex(t)
	// Same content in two languages → high dedup score
	texts := []string{
		"negative good",
		"negativo bom",
	}
	score := tokenizer.DeduplicationScore(lex, texts)
	// All tokens appear in both languages → 100% duplicated
	if score != 1.0 {
		t.Fatalf("want dedup score=1.0 for identical cross-lingual content, got %.2f", score)
	}
}

func TestVocabStats(t *testing.T) {
	lex := buildTokenLex(t)
	stats := tokenizer.Analyze(lex, []string{
		"negative good xyzzy unknown_word",
	})

	if stats.TotalTokens != 4 {
		t.Fatalf("want 4 total tokens, got %d", stats.TotalTokens)
	}
	if stats.UniqueWordIDs != 2 {
		t.Fatalf("want 2 unique word IDs, got %d", stats.UniqueWordIDs)
	}
	if stats.UniqueRootIDs != 2 {
		t.Fatalf("want 2 unique root IDs, got %d", stats.UniqueRootIDs)
	}
	if stats.OOVRate < 0.4 || stats.OOVRate > 0.6 {
		t.Fatalf("want ~50%% OOV rate, got %.2f", stats.OOVRate)
	}
}
