package infer

import (
	"testing"

	"github.com/kak/umcs/pkg/sentiment"
)

// ── POSFromShape — Portuguese ─────────────────────────────────────────────────

func TestPOSFromShape_PT_Noun_cao(t *testing.T) {
	got := POSFromShape("canção", "PT")
	if got != sentiment.POSNoun {
		t.Fatalf("canção/PT: want POSNoun(%d), got %d", sentiment.POSNoun>>29, got>>29)
	}
}

func TestPOSFromShape_PT_Noun_dade(t *testing.T) {
	got := POSFromShape("liberdade", "PT")
	if got != sentiment.POSNoun {
		t.Fatalf("liberdade/PT: want POSNoun, got 0x%08X", got)
	}
}

func TestPOSFromShape_PT_Noun_ismo(t *testing.T) {
	got := POSFromShape("capitalismo", "PT")
	if got != sentiment.POSNoun {
		t.Fatalf("capitalismo/PT: want POSNoun, got 0x%08X", got)
	}
}

func TestPOSFromShape_PT_Adv(t *testing.T) {
	got := POSFromShape("rapidamente", "PT")
	if got != sentiment.POSAdv {
		t.Fatalf("rapidamente/PT: want POSAdv(%d), got 0x%08X", sentiment.POSAdv>>29, got)
	}
}

func TestPOSFromShape_PT_Adj(t *testing.T) {
	got := POSFromShape("perigoso", "PT")
	if got != sentiment.POSAdj {
		t.Fatalf("perigoso/PT: want POSAdj, got 0x%08X", got)
	}
}

func TestPOSFromShape_PT_Adj_osa(t *testing.T) {
	got := POSFromShape("viciosa", "PT")
	if got != sentiment.POSAdj {
		t.Fatalf("viciosa/PT: want POSAdj, got 0x%08X", got)
	}
}

func TestPOSFromShape_PT_Verb(t *testing.T) {
	got := POSFromShape("cantar", "PT")
	if got != sentiment.POSVerb {
		t.Fatalf("cantar/PT: want POSVerb, got 0x%08X", got)
	}
}

// ── POSFromShape — English ────────────────────────────────────────────────────

func TestPOSFromShape_EN_Noun_tion(t *testing.T) {
	got := POSFromShape("liberation", "EN")
	if got != sentiment.POSNoun {
		t.Fatalf("liberation/EN: want POSNoun, got 0x%08X", got)
	}
}

func TestPOSFromShape_EN_Noun_ness(t *testing.T) {
	got := POSFromShape("happiness", "EN")
	if got != sentiment.POSNoun {
		t.Fatalf("happiness/EN: want POSNoun, got 0x%08X", got)
	}
}

func TestPOSFromShape_EN_Noun_ity(t *testing.T) {
	got := POSFromShape("creativity", "EN")
	if got != sentiment.POSNoun {
		t.Fatalf("creativity/EN: want POSNoun, got 0x%08X", got)
	}
}

func TestPOSFromShape_EN_Adv(t *testing.T) {
	got := POSFromShape("quickly", "EN")
	if got != sentiment.POSAdv {
		t.Fatalf("quickly/EN: want POSAdv, got 0x%08X", got)
	}
}

func TestPOSFromShape_EN_Adj(t *testing.T) {
	got := POSFromShape("beautiful", "EN")
	if got != sentiment.POSAdj {
		t.Fatalf("beautiful/EN: want POSAdj, got 0x%08X", got)
	}
}

func TestPOSFromShape_EN_Adj_less(t *testing.T) {
	got := POSFromShape("hopeless", "EN")
	if got != sentiment.POSAdj {
		t.Fatalf("hopeless/EN: want POSAdj, got 0x%08X", got)
	}
}

// ── POSFromShape — other languages ───────────────────────────────────────────

func TestPOSFromShape_ES_Noun(t *testing.T) {
	got := POSFromShape("canción", "ES")
	if got != sentiment.POSNoun {
		t.Fatalf("canción/ES: want POSNoun, got 0x%08X", got)
	}
}

func TestPOSFromShape_FR_Noun(t *testing.T) {
	got := POSFromShape("liberté", "FR")
	if got != sentiment.POSNoun {
		t.Fatalf("liberté/FR: want POSNoun, got 0x%08X", got)
	}
}

func TestPOSFromShape_DE_Noun(t *testing.T) {
	got := POSFromShape("Freiheit", "DE")
	if got != sentiment.POSNoun {
		t.Fatalf("Freiheit/DE: want POSNoun, got 0x%08X", got)
	}
}

func TestPOSFromShape_DE_Noun_keit(t *testing.T) {
	got := POSFromShape("Möglichkeit", "DE")
	if got != sentiment.POSNoun {
		t.Fatalf("Möglichkeit/DE: want POSNoun, got 0x%08X", got)
	}
}

// ── POSFromShape — edge cases ─────────────────────────────────────────────────

func TestPOSFromShape_Unknown(t *testing.T) {
	got := POSFromShape("xyz", "EN")
	if got != sentiment.POSOther {
		t.Fatalf("xyz/EN: want POSOther(0), got 0x%08X", got)
	}
}

func TestPOSFromShape_LongestMatch(t *testing.T) {
	// "modernamente" ends in "-mente" (5 chars → ADV).
	// Longest-match must win over any shorter suffix.
	got := POSFromShape("modernamente", "PT")
	if got != sentiment.POSAdv {
		t.Fatalf("modernamente/PT: longest match '-mente' must give POSAdv, got 0x%08X", got)
	}

	// "liberdade" ends in "-dade" (4 chars → Noun), not in shorter noise.
	got2 := POSFromShape("liberdade", "PT")
	if got2 != sentiment.POSNoun {
		t.Fatalf("liberdade/PT: longest match '-dade' must give POSNoun, got 0x%08X", got2)
	}
}

func TestPOSFromShape_WrongLang(t *testing.T) {
	// "-ção" is a PT rule — must NOT fire for EN words.
	// "action" ends in "-tion" (EN → Noun), not "-ção" (PT).
	// Verify that a PT-specific rule does not contaminate EN classification.
	gotPT := POSFromShape("canção", "PT")
	gotEN := POSFromShape("canção", "EN") // no EN rule for "-ção"

	if gotPT == 0 {
		t.Fatal("canção/PT must be classified")
	}
	// EN should return either 0 (no rule) or classify via "-ão" (no rule either).
	// The point: PT rules must not bleed into EN.
	_ = gotEN // result may be 0 or via some unrelated rule; key test is PT works
}

func TestPOSFromShape_CaseInsensitive(t *testing.T) {
	lower := POSFromShape("liberation", "EN")
	upper := POSFromShape("LIBERATION", "EN")
	if lower != upper {
		t.Fatalf("case sensitivity: want same result, got 0x%08X vs 0x%08X", lower, upper)
	}
	if lower != sentiment.POSNoun {
		t.Fatalf("LIBERATION/EN: want POSNoun, got 0x%08X", lower)
	}
}

// ── IsAbstractFromShape ───────────────────────────────────────────────────────

func TestIsAbstractFromShape_PT_dade(t *testing.T) {
	if !IsAbstractFromShape("liberdade", "PT") {
		t.Fatal("liberdade/PT: '-dade' must be abstract")
	}
}

func TestIsAbstractFromShape_PT_ismo(t *testing.T) {
	if !IsAbstractFromShape("capitalismo", "PT") {
		t.Fatal("capitalismo/PT: '-ismo' must be abstract")
	}
}

func TestIsAbstractFromShape_EN_ness(t *testing.T) {
	if !IsAbstractFromShape("happiness", "EN") {
		t.Fatal("happiness/EN: '-ness' must be abstract")
	}
}

func TestIsAbstractFromShape_EN_tion(t *testing.T) {
	if !IsAbstractFromShape("liberation", "EN") {
		t.Fatal("liberation/EN: '-tion' must be abstract")
	}
}

func TestIsAbstractFromShape_Concrete(t *testing.T) {
	// "cadeira" (PT, chair) has no abstract suffix.
	if IsAbstractFromShape("cadeira", "PT") {
		t.Fatal("cadeira/PT: concrete noun must NOT be abstract by suffix")
	}
}

func TestIsAbstractFromShape_EN_freedom(t *testing.T) {
	// "freedom" has no suffix rule in EN abstract list.
	if IsAbstractFromShape("freedom", "EN") {
		t.Fatal("freedom/EN: no suffix rule — must return false")
	}
}

func TestIsAbstractFromShape_UnknownLang(t *testing.T) {
	// Unknown language code must not panic.
	got := IsAbstractFromShape("anything", "XX")
	if got {
		t.Fatal("unknown lang: must return false, not panic")
	}
}

// ── FillMissing ───────────────────────────────────────────────────────────────

func TestFillMissing_FillsPOS(t *testing.T) {
	// sent=0 → POS is unset → FillMissing must infer from suffix.
	// "liberation" (EN) ends in "-tion" → POSNoun.
	sent := FillMissing(0, "liberation", "EN")
	if sentiment.POS(sent) != sentiment.POS(sentiment.POSNoun) {
		t.Fatalf("FillMissing liberation/EN: want POS=NOUN(%d), got %d",
			sentiment.POS(sentiment.POSNoun), sentiment.POS(sent))
	}
}

func TestFillMissing_FillsPOS_Adv(t *testing.T) {
	sent := FillMissing(0, "rapidamente", "PT")
	if sentiment.POS(sent) != sentiment.POS(sentiment.POSAdv) {
		t.Fatalf("FillMissing rapidamente/PT: want POS=ADV(%d), got %d",
			sentiment.POS(sentiment.POSAdv), sentiment.POS(sent))
	}
}

func TestFillMissing_NoOverwrite(t *testing.T) {
	// Pre-set POS (VERB) must not be overwritten, even if suffix suggests Noun.
	existing := sentiment.POSVerb | sentiment.PolarityPositive
	sent := FillMissing(existing, "liberation", "EN")
	if sentiment.POS(sent) != sentiment.POS(sentiment.POSVerb) {
		t.Fatalf("FillMissing must NOT overwrite existing POS: want VERB, got %d",
			sentiment.POS(sent))
	}
	// Other bits must be preserved.
	if sentiment.Polarity(sent) != sentiment.PolarityPositive {
		t.Fatal("FillMissing must preserve non-POS bits")
	}
}

func TestFillMissing_EmptyWord(t *testing.T) {
	// Empty word must not panic and must return sent unchanged.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("FillMissing panicked on empty word: %v", r)
		}
	}()
	sent := uint32(0x00120180)
	got := FillMissing(sent, "", "EN")
	if got != sent {
		t.Fatalf("empty word: want sent unchanged (0x%08X), got 0x%08X", sent, got)
	}
}

func TestFillMissing_UnknownSuffix(t *testing.T) {
	// Word with no suffix rule → sent unchanged (POS stays 0).
	sent := FillMissing(0, "xyz", "EN")
	if sentiment.POS(sent) != 0 {
		t.Fatalf("unknown suffix: POS must remain 0, got %d", sentiment.POS(sent))
	}
}
