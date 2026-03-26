package tokenizer_test

// Stress tests: empty input, all-OOV, repeated words, deduplication edge cases,
// and corpus statistics. Uses buildTokenLex fixture from morpheme_test.go.

import (
	"testing"

	"github.com/kak/umcs/pkg/tokenizer"
)

func TestTokenizeEmpty(t *testing.T) {
	lex := buildTokenLex(t)
	tokens := tokenizer.Tokenize(lex, "")
	if len(tokens) != 0 {
		t.Fatalf("empty string: want 0 tokens, got %d", len(tokens))
	}
}

func TestTokenizeOnlySpaces(t *testing.T) {
	lex := buildTokenLex(t)
	tokens := tokenizer.Tokenize(lex, "   \t\n   ")
	if len(tokens) != 0 {
		t.Fatalf("whitespace only: want 0 tokens, got %d", len(tokens))
	}
}

func TestTokenizeAllOOV(t *testing.T) {
	lex := buildTokenLex(t)
	tokens := tokenizer.Tokenize(lex, "qwerty asdfg zxcvb")
	if len(tokens) != 3 {
		t.Fatalf("want 3 OOV tokens, got %d", len(tokens))
	}
	for _, tok := range tokens {
		if tok.Known {
			t.Errorf("token %q should be OOV", tok.Surface)
		}
		if tok.WordID != 0 {
			t.Errorf("OOV token %q: want WordID=0, got %d", tok.Surface, tok.WordID)
		}
		if tok.RootID != 0 {
			t.Errorf("OOV token %q: want RootID=0, got %d", tok.Surface, tok.RootID)
		}
	}
}

func TestTokenizeRepeatedWord(t *testing.T) {
	lex := buildTokenLex(t)
	tokens := tokenizer.Tokenize(lex, "good good good")
	if len(tokens) != 3 {
		t.Fatalf("want 3 tokens, got %d", len(tokens))
	}
	for i, tok := range tokens {
		if !tok.Known {
			t.Errorf("token[%d] (good) should be known", i)
		}
		if tok.RootID != 2 {
			t.Errorf("token[%d] good: want root_id=2, got %d", i, tok.RootID)
		}
	}
}

func TestTokenizeToRootIDsSingleWord(t *testing.T) {
	lex := buildTokenLex(t)
	ids := tokenizer.TokenizeToRootIDs(lex, "negative")
	if len(ids) != 1 {
		t.Fatalf("want 1 root_id, got %d", len(ids))
	}
	if ids[0] != 1 {
		t.Fatalf("want root_id=1 for negative, got %d", ids[0])
	}
}

func TestTokenizeToRootIDsOOV(t *testing.T) {
	lex := buildTokenLex(t)
	ids := tokenizer.TokenizeToRootIDs(lex, "xyzzy")
	if len(ids) != 1 {
		t.Fatalf("want 1 entry, got %d", len(ids))
	}
	if ids[0] != 0 {
		t.Fatalf("OOV root_id should be 0, got %d", ids[0])
	}
}

func TestDeduplicationScoreIdentical(t *testing.T) {
	lex := buildTokenLex(t)
	// Same text twice → all root_ids appear 2+ times → score=1.0
	texts := []string{"good", "good"}
	score := tokenizer.DeduplicationScore(lex, texts)
	if score != 1.0 {
		t.Fatalf("identical texts: want score=1.0, got %.4f", score)
	}
}

func TestDeduplicationScoreNoOverlap(t *testing.T) {
	lex := buildTokenLex(t)
	// "negative" (root_id=1) and "good" (root_id=2) — no shared root_ids → score=0.0
	texts := []string{"negative", "good"}
	score := tokenizer.DeduplicationScore(lex, texts)
	if score != 0.0 {
		t.Fatalf("no-overlap texts: want score=0.0, got %.4f", score)
	}
}

func TestDeduplicationScoreSingleText(t *testing.T) {
	lex := buildTokenLex(t)
	// Single text: no cross-text duplication → score=0.0
	texts := []string{"good"}
	score := tokenizer.DeduplicationScore(lex, texts)
	if score != 0.0 {
		t.Fatalf("single text: want score=0.0, got %.4f", score)
	}
}

func TestDeduplicationScoreEmptyTexts(t *testing.T) {
	lex := buildTokenLex(t)
	score := tokenizer.DeduplicationScore(lex, []string{})
	if score != 0.0 {
		t.Fatalf("empty texts: want score=0.0, got %.4f", score)
	}
}

func TestDeduplicationScoreAllOOV(t *testing.T) {
	lex := buildTokenLex(t)
	// All tokens OOV → root_ids=0 (excluded from counting) → score=0.0
	texts := []string{"xyzzy qwerty", "asdfg zxcvb"}
	score := tokenizer.DeduplicationScore(lex, texts)
	if score != 0.0 {
		t.Fatalf("all-OOV texts: want score=0.0, got %.4f", score)
	}
}

func TestVocabStatsEmpty(t *testing.T) {
	lex := buildTokenLex(t)
	stats := tokenizer.Analyze(lex, []string{})
	if stats.TotalTokens != 0 {
		t.Fatalf("empty corpus: want TotalTokens=0, got %d", stats.TotalTokens)
	}
	if stats.UniqueWordIDs != 0 {
		t.Fatalf("empty corpus: want UniqueWordIDs=0, got %d", stats.UniqueWordIDs)
	}
	if stats.OOVRate != 0.0 {
		t.Fatalf("empty corpus: want OOVRate=0.0, got %.4f", stats.OOVRate)
	}
}

func TestVocabStatsSingleKnownToken(t *testing.T) {
	lex := buildTokenLex(t)
	stats := tokenizer.Analyze(lex, []string{"good"})
	if stats.TotalTokens != 1 {
		t.Fatalf("want TotalTokens=1, got %d", stats.TotalTokens)
	}
	if stats.UniqueWordIDs != 1 {
		t.Fatalf("want UniqueWordIDs=1, got %d", stats.UniqueWordIDs)
	}
	if stats.UniqueRootIDs != 1 {
		t.Fatalf("want UniqueRootIDs=1, got %d", stats.UniqueRootIDs)
	}
	if stats.OOVRate != 0.0 {
		t.Fatalf("single known token: want OOVRate=0.0, got %.4f", stats.OOVRate)
	}
}

func TestVocabStatsSingleOOVToken(t *testing.T) {
	lex := buildTokenLex(t)
	stats := tokenizer.Analyze(lex, []string{"xyzzy"})
	if stats.TotalTokens != 1 {
		t.Fatalf("want TotalTokens=1, got %d", stats.TotalTokens)
	}
	if stats.UniqueWordIDs != 0 {
		t.Fatalf("OOV should not count in UniqueWordIDs, got %d", stats.UniqueWordIDs)
	}
	if stats.OOVRate != 1.0 {
		t.Fatalf("single OOV token: want OOVRate=1.0, got %.4f", stats.OOVRate)
	}
}

func TestVocabStatsCrossLinguistic(t *testing.T) {
	lex := buildTokenLex(t)
	// "negative" (EN root_id=1) and "negativo" (PT root_id=1) → same root_id
	stats := tokenizer.Analyze(lex, []string{"negative negativo"})
	if stats.TotalTokens != 2 {
		t.Fatalf("want 2 tokens, got %d", stats.TotalTokens)
	}
	if stats.UniqueWordIDs != 2 {
		t.Fatalf("different word_ids: want 2, got %d", stats.UniqueWordIDs)
	}
	if stats.UniqueRootIDs != 1 {
		t.Fatalf("same root_id=1: want UniqueRootIDs=1, got %d", stats.UniqueRootIDs)
	}
}

func TestTokenizeLongText(t *testing.T) {
	lex := buildTokenLex(t)
	// 10000 tokens — verify it doesn't hang and returns correct count
	words := make([]string, 1000)
	for i := range words {
		if i%2 == 0 {
			words[i] = "good"
		} else {
			words[i] = "negative"
		}
	}
	text := ""
	for _, w := range words {
		text += w + " "
	}
	tokens := tokenizer.Tokenize(lex, text)
	if len(tokens) != 1000 {
		t.Fatalf("want 1000 tokens, got %d", len(tokens))
	}
	for _, tok := range tokens {
		if !tok.Known {
			t.Errorf("token %q should be known", tok.Surface)
		}
	}
}
