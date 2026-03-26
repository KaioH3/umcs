package morpheme

import (
	"math/rand"
	"testing"

	"github.com/kak/umcs/pkg/sentiment"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func mustWordID(t *testing.T, rootID, variant uint32) uint32 {
	t.Helper()
	id, err := MakeWordID(rootID, variant)
	if err != nil {
		t.Fatalf("MakeWordID(%d, %d): %v", rootID, variant, err)
	}
	return id
}

// ── Pack64 / Unpack64 basics ──────────────────────────────────────────────────

func TestPack64Basic(t *testing.T) {
	wordID := mustWordID(t, 5, 3)
	sent := sentiment.PolarityNegative | sentiment.IntensityStrong | sentiment.RoleEvaluation
	flags := uint32(0)

	tok := Pack64(wordID, sent, flags)
	if tok == 0 {
		t.Fatal("Pack64 must return a non-zero Token64")
	}

	gotWordID, gotPayload := Unpack64(tok)
	if gotWordID != wordID {
		t.Fatalf("Unpack64 wordID: want %d, got %d", wordID, gotWordID)
	}
	// Domain bits (15..8) are cleared in token; polarity should survive.
	if sentiment.Polarity(gotPayload) != sentiment.PolarityNegative {
		t.Fatalf("polarity not preserved in token: got 0x%08X", gotPayload)
	}
}

func TestUnpack64Roundtrip(t *testing.T) {
	wordID := mustWordID(t, 42, 7)
	sent := sentiment.POSAdj | sentiment.PolarityPositive | sentiment.ArousalHigh | sentiment.DominanceMed
	flags := uint32(0)

	tok := Pack64(wordID, sent, flags)
	gotID, gotPay := Unpack64(tok)

	if gotID != wordID {
		t.Fatalf("wordID roundtrip: want %d, got %d", wordID, gotID)
	}
	if sentiment.POS(gotPay) != sentiment.POS(sent) {
		t.Fatalf("POS roundtrip: want %d, got %d", sentiment.POS(sent), sentiment.POS(gotPay))
	}
	if sentiment.Polarity(gotPay) != sentiment.Polarity(sent) {
		t.Fatalf("polarity roundtrip: want %d, got %d", sentiment.Polarity(sent), sentiment.Polarity(gotPay))
	}
	if sentiment.Arousal(gotPay) != sentiment.Arousal(sent) {
		t.Fatalf("arousal roundtrip: want %d, got %d", sentiment.Arousal(sent), sentiment.Arousal(gotPay))
	}
}

// ── Root / Variant extraction ─────────────────────────────────────────────────

func TestPack64RootExtraction(t *testing.T) {
	for _, rootID := range []uint32{1, 10, 100, 1000, MaxRootID} {
		wordID := mustWordID(t, rootID, 1)
		tok := Pack64(wordID, 0, 0)
		if got := RootOf64(tok); got != rootID {
			t.Fatalf("root_id %d: RootOf64 returned %d", rootID, got)
		}
		if got := RootOf(wordID); got != rootID {
			t.Fatalf("root_id %d: RootOf returned %d (sanity check)", rootID, got)
		}
	}
}

func TestPack64VariantExtract(t *testing.T) {
	for _, variant := range []uint32{1, 5, 99, MaxVariant} {
		wordID := mustWordID(t, 1, variant)
		tok := Pack64(wordID, 0, 0)
		if got := VariantOf64(tok); got != variant {
			t.Fatalf("variant %d: VariantOf64 returned %d", variant, got)
		}
	}
}

// ── Cognates64 ────────────────────────────────────────────────────────────────

func TestCognates64(t *testing.T) {
	// Same root_id, different variants → cognates.
	a := Pack64(mustWordID(t, 10, 1), 0, 0)
	b := Pack64(mustWordID(t, 10, 2), 0, 0)
	if !Cognates64(a, b) {
		t.Fatal("same root must be cognates")
	}
}

func TestCognates64Different(t *testing.T) {
	a := Pack64(mustWordID(t, 10, 1), 0, 0)
	b := Pack64(mustWordID(t, 11, 1), 0, 0)
	if Cognates64(a, b) {
		t.Fatal("different roots must NOT be cognates")
	}
}

func TestCognates64Reflexive(t *testing.T) {
	tok := Pack64(mustWordID(t, 7, 3), 0, 0)
	if !Cognates64(tok, tok) {
		t.Fatal("a token must be a cognate of itself")
	}
}

// ── Specific bit fields ───────────────────────────────────────────────────────

func TestPack64PolarityBits(t *testing.T) {
	tests := []struct {
		sent uint32
		want uint32
	}{
		{sentiment.PolarityNeutral, sentiment.PolarityNeutral},
		{sentiment.PolarityPositive, sentiment.PolarityPositive},
		{sentiment.PolarityNegative, sentiment.PolarityNegative},
		{sentiment.PolarityAmbiguous, sentiment.PolarityAmbiguous},
	}
	for _, tt := range tests {
		tok := Pack64(mustWordID(t, 1, 1), tt.sent, 0)
		_, pay := Unpack64(tok)
		if sentiment.Polarity(pay) != tt.want {
			t.Fatalf("polarity 0x%X: want 0x%X, got 0x%X", tt.sent, tt.want, sentiment.Polarity(pay))
		}
	}
}

func TestPack64POSBits(t *testing.T) {
	tests := []struct {
		pos  uint32
		name string
	}{
		{sentiment.POSNoun, "NOUN"},
		{sentiment.POSVerb, "VERB"},
		{sentiment.POSAdj, "ADJ"},
		{sentiment.POSAdv, "ADV"},
		{sentiment.POSParticle, "PARTICLE"},
	}
	for _, tt := range tests {
		tok := Pack64(mustWordID(t, 1, 1), tt.pos, 0)
		_, pay := Unpack64(tok)
		if sentiment.POS(pay) != sentiment.POS(tt.pos) {
			t.Fatalf("POS %s: want %d, got %d", tt.name, sentiment.POS(tt.pos), sentiment.POS(pay))
		}
	}
}

func TestPack64ArousalBits(t *testing.T) {
	for _, ar := range []uint32{sentiment.ArousalNone, sentiment.ArousalLow, sentiment.ArousalMed, sentiment.ArousalHigh} {
		tok := Pack64(mustWordID(t, 1, 1), ar, 0)
		_, pay := Unpack64(tok)
		if sentiment.Arousal(pay) != sentiment.Arousal(ar) {
			t.Fatalf("arousal 0x%X: want %d, got %d", ar, sentiment.Arousal(ar), sentiment.Arousal(pay))
		}
	}
}

func TestPack64DominanceBits(t *testing.T) {
	for _, dom := range []uint32{sentiment.DominanceNone, sentiment.DominanceLow, sentiment.DominanceMed, sentiment.DominanceHigh} {
		tok := Pack64(mustWordID(t, 1, 1), dom, 0)
		_, pay := Unpack64(tok)
		if sentiment.Dominance(pay) != sentiment.Dominance(dom) {
			t.Fatalf("dominance 0x%X: want %d, got %d", dom, sentiment.Dominance(dom), sentiment.Dominance(pay))
		}
	}
}

func TestPack64AOABits(t *testing.T) {
	tests := []struct{ raw, want uint32 }{
		{sentiment.AOAEarly, sentiment.AOAEarly},
		{sentiment.AOAMid, sentiment.AOAMid},
		{sentiment.AOALate, sentiment.AOALate},
		{sentiment.AOATechnical, sentiment.AOATechnical},
	}
	for _, tt := range tests {
		tok := Pack64(mustWordID(t, 1, 1), tt.raw, 0)
		_, pay := Unpack64(tok)
		if sentiment.AOA(pay) != tt.want {
			t.Fatalf("AoA %d: want %d, got %d", tt.raw, tt.want, sentiment.AOA(pay))
		}
	}
}

func TestPack64ConcreteBit(t *testing.T) {
	// Concrete bit (28) must survive Pack64 → Unpack64.
	concreteSent := sentiment.Concrete | sentiment.PolarityPositive
	tok := Pack64(mustWordID(t, 1, 1), concreteSent, 0)
	_, pay := Unpack64(tok)
	if !sentiment.IsConcrete(pay) {
		t.Fatal("concrete bit must be preserved in Token64")
	}

	// Abstract (bit=0) must also survive.
	abstractSent := sentiment.PolarityNeutral // no concrete bit
	tok2 := Pack64(mustWordID(t, 1, 2), abstractSent, 0)
	_, pay2 := Unpack64(tok2)
	if sentiment.IsConcrete(pay2) {
		t.Fatal("abstract word must NOT have concrete bit set")
	}
}

// ── Ontological and Register from flags ───────────────────────────────────────

func TestPack64OntoBits(t *testing.T) {
	// OntoProperty = 7 << 12 in flags
	const OntoProperty = 7 << 12
	flags := uint32(OntoProperty)
	tok := Pack64(mustWordID(t, 1, 1), 0, flags)
	_, pay := Unpack64(tok)
	// bits 15..12 of pay should match bits 15..12 of flags
	want := flags & 0xF000
	got := pay & 0xF000
	if got != want {
		t.Fatalf("onto bits: want 0x%X, got 0x%X (full payload 0x%08X)", want, got, pay)
	}
}

func TestPack64RegisterBits(t *testing.T) {
	// RegisterFormal = 1 << 8 in flags
	const RegisterFormal = 1 << 8
	flags := uint32(RegisterFormal)
	tok := Pack64(mustWordID(t, 1, 1), 0, flags)
	_, pay := Unpack64(tok)
	// bits 11..8 of pay should match bits 11..8 of flags
	want := flags & 0x0F00
	got := pay & 0x0F00
	if got != want {
		t.Fatalf("register bits: want 0x%X, got 0x%X (full payload 0x%08X)", want, got, pay)
	}
}

// ── Zero-value word ───────────────────────────────────────────────────────────

func TestPack64ZeroWord(t *testing.T) {
	// wordID=0 is technically invalid per UMCS but Pack64 must not panic.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Pack64 panicked on wordID=0: %v", r)
		}
	}()
	tok := Pack64(0, 0, 0)
	gotID, _ := Unpack64(tok)
	if gotID != 0 {
		t.Fatalf("wordID=0 roundtrip: want 0, got %d", gotID)
	}
}

// ── Real-word test ────────────────────────────────────────────────────────────

func TestPack64RealWord(t *testing.T) {
	// "terrible" — canonical test word.
	// root_id=10, variant=1 → wordID=40961
	// sentiment: NEGATIVE, STRONG intensity, EVALUATION role, ADJ POS
	wordID := mustWordID(t, 10, 1)
	if wordID != 40961 {
		t.Fatalf("wordID sanity: want 40961, got %d", wordID)
	}

	sent := sentiment.POSAdj |
		sentiment.PolarityNegative |
		sentiment.IntensityStrong |
		sentiment.RoleEvaluation |
		sentiment.ArousalHigh |
		sentiment.DominanceLow |
		sentiment.Concrete

	tok := Pack64(wordID, sent, 0)
	if tok == 0 {
		t.Fatal("Token64 for 'terrible' must not be zero")
	}

	gotID, pay := Unpack64(tok)
	if gotID != wordID {
		t.Fatalf("word_id: want %d, got %d", wordID, gotID)
	}
	if RootOf64(tok) != 10 {
		t.Fatalf("root_id: want 10, got %d", RootOf64(tok))
	}
	if VariantOf64(tok) != 1 {
		t.Fatalf("variant: want 1, got %d", VariantOf64(tok))
	}
	if sentiment.Polarity(pay) != sentiment.PolarityNegative {
		t.Fatalf("polarity: want NEGATIVE, got 0x%X", sentiment.Polarity(pay))
	}
	if sentiment.Intensity(pay) != 3 { // STRONG=3
		t.Fatalf("intensity: want STRONG(3), got %d", sentiment.Intensity(pay))
	}
	if sentiment.POS(pay) != sentiment.POS(sentiment.POSAdj) {
		t.Fatalf("POS: want ADJ(%d), got %d", sentiment.POS(sentiment.POSAdj), sentiment.POS(pay))
	}
	if sentiment.Arousal(pay) != sentiment.Arousal(sentiment.ArousalHigh) {
		t.Fatalf("arousal: want HIGH, got %d", sentiment.Arousal(pay))
	}
	if !sentiment.IsConcrete(pay) {
		t.Fatal("concrete bit must be set for 'terrible'")
	}
}

// ── Stress test ───────────────────────────────────────────────────────────────

func TestCognates64Stress(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		rootID := uint32(r.Intn(int(MaxRootID)-1)) + 1
		v1 := uint32(r.Intn(int(MaxVariant)-1)) + 1
		v2 := uint32(r.Intn(int(MaxVariant)-1)) + 1

		id1 := mustWordID(t, rootID, v1)
		id2 := mustWordID(t, rootID, v2)

		tok1 := Pack64(id1, uint32(r.Uint32()), uint32(r.Uint32()))
		tok2 := Pack64(id2, uint32(r.Uint32()), uint32(r.Uint32()))

		if !Cognates64(tok1, tok2) {
			t.Fatalf("iter %d: same root_id=%d → must be cognates", i, rootID)
		}

		// Different root must not be cognate.
		otherRoot := rootID%uint32(MaxRootID) + 1
		if otherRoot == rootID {
			otherRoot = (rootID % uint32(MaxRootID-1)) + 2
		}
		if otherRoot > MaxRootID {
			otherRoot = 1
		}
		id3 := mustWordID(t, otherRoot, v1)
		tok3 := Pack64(id3, 0, 0)
		if Cognates64(tok1, tok3) {
			t.Fatalf("iter %d: root %d and %d must NOT be cognates", i, rootID, otherRoot)
		}
	}
}
