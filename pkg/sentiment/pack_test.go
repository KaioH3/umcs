package sentiment

import "testing"

func TestPackUnpack(t *testing.T) {
	s, err := Pack("NEGATIVE", "MODERATE", "EVALUATION", "GENERAL", nil)
	if err != nil {
		t.Fatal(err)
	}

	if Polarity(s) != PolarityNegative {
		t.Fatalf("want NEGATIVE polarity, got %d", Polarity(s))
	}
	if Intensity(s) != 2 {
		t.Fatalf("want intensity 2 (MODERATE), got %d", Intensity(s))
	}
	if Role(s) != 1 { // EVALUATION
		t.Fatalf("want role EVALUATION(1), got %d", Role(s))
	}
}

func TestNegationMarkerFlag(t *testing.T) {
	s, err := Pack("NEUTRAL", "NONE", "NEGATION_MARKER", "GENERAL", []string{"NEGATION_MARKER"})
	if err != nil {
		t.Fatal(err)
	}
	if !IsNegationMarker(s) {
		t.Fatal("must be negation marker")
	}
}

func TestIntensifierFlag(t *testing.T) {
	s, err := Pack("NEUTRAL", "NONE", "INTENSIFIER", "GENERAL", []string{"INTENSIFIER"})
	if err != nil {
		t.Fatal(err)
	}
	if !IsIntensifier(s) {
		t.Fatal("must be intensifier")
	}
}

func TestWeight(t *testing.T) {
	neg, _ := Pack("NEGATIVE", "STRONG", "EVALUATION", "GENERAL", nil)
	pos, _ := Pack("POSITIVE", "WEAK", "EVALUATION", "GENERAL", nil)
	neu, _ := Pack("NEUTRAL", "NONE", "NONE", "GENERAL", nil)

	if Weight(neg) != -3 {
		t.Fatalf("want -3, got %d", Weight(neg))
	}
	if Weight(pos) != 1 {
		t.Fatalf("want 1, got %d", Weight(pos))
	}
	if Weight(neu) != 0 {
		t.Fatalf("want 0, got %d", Weight(neu))
	}
}

func TestMultiDomain(t *testing.T) {
	s, err := Pack("NEGATIVE", "WEAK", "NONE", "FINANCIAL|LEGAL", nil)
	if err != nil {
		t.Fatal(err)
	}
	d := Domain(s)
	if d&(DomainFinancial>>8) == 0 {
		t.Fatal("must have FINANCIAL domain")
	}
	if d&(DomainLegal>>8) == 0 {
		t.Fatal("must have LEGAL domain")
	}
}

func TestDecode(t *testing.T) {
	s, _ := Pack("POSITIVE", "STRONG", "EMOTION", "SOCIAL", []string{"INTENSIFIER"})
	dec := Decode(s)
	if dec["polarity"] != "POSITIVE" {
		t.Fatalf("want POSITIVE, got %q", dec["polarity"])
	}
	if dec["intensity"] != "STRONG" {
		t.Fatalf("want STRONG, got %q", dec["intensity"])
	}
}

// ── PackExtended — new dimensions ─────────────────────────────────────────────

func TestPackExtendedPOS_Noun(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "NOUN", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if POS(s) != POS(POSNoun) {
		t.Fatalf("want NOUN(%d), got %d", POS(POSNoun), POS(s))
	}
}

func TestPackExtendedPOS_Verb(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "VERB", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if POS(s) != POS(POSVerb) {
		t.Fatalf("want VERB(%d), got %d", POS(POSVerb), POS(s))
	}
}

func TestPackExtendedPOS_Adj(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "ADJ", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if POS(s) != POS(POSAdj) {
		t.Fatalf("want ADJ(%d), got %d", POS(POSAdj), POS(s))
	}
}

func TestPackExtendedPOS_Adv(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "ADV", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if POS(s) != POS(POSAdv) {
		t.Fatalf("want ADV(%d), got %d", POS(POSAdv), POS(s))
	}
}

func TestPackExtendedConcrete(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "", "", "", "", "CONCRETE")
	if err != nil {
		t.Fatal(err)
	}
	if !IsConcrete(s) {
		t.Fatal("concreteness=CONCRETE must set concrete bit (28)")
	}
}

func TestPackExtendedAbstract(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if IsConcrete(s) {
		t.Fatal("empty concreteness must leave bit 28 unset (abstract)")
	}
}

func TestPackExtendedArousal_High(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "", "HIGH", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if Arousal(s) != Arousal(ArousalHigh) {
		t.Fatalf("want HIGH(%d), got %d", Arousal(ArousalHigh), Arousal(s))
	}
}

func TestPackExtendedArousal_AllLevels(t *testing.T) {
	cases := []struct {
		input string
		want  uint32
	}{
		{"NONE", ArousalNone},
		{"LOW", ArousalLow},
		{"MED", ArousalMed},
		{"MEDIUM", ArousalMed},
		{"HIGH", ArousalHigh},
		{"", ArousalNone},
	}
	for _, tc := range cases {
		s, err := PackExtended("", "", "", "", nil, "", tc.input, "", "", "")
		if err != nil {
			t.Fatalf("arousal=%q: %v", tc.input, err)
		}
		if Arousal(s) != Arousal(tc.want) {
			t.Fatalf("arousal=%q: want %d, got %d", tc.input, Arousal(tc.want), Arousal(s))
		}
	}
}

func TestPackExtendedDominance_Low(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "", "", "LOW", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if Dominance(s) != Dominance(DominanceLow) {
		t.Fatalf("want LOW(%d), got %d", Dominance(DominanceLow), Dominance(s))
	}
}

func TestPackExtendedDominance_AllLevels(t *testing.T) {
	cases := []struct {
		input string
		want  uint32
	}{
		{"NONE", DominanceNone},
		{"LOW", DominanceLow},
		{"MED", DominanceMed},
		{"HIGH", DominanceHigh},
		{"", DominanceNone},
	}
	for _, tc := range cases {
		s, err := PackExtended("", "", "", "", nil, "", "", tc.input, "", "")
		if err != nil {
			t.Fatalf("dominance=%q: %v", tc.input, err)
		}
		if Dominance(s) != Dominance(tc.want) {
			t.Fatalf("dominance=%q: want %d, got %d", tc.input, Dominance(tc.want), Dominance(s))
		}
	}
}

func TestPackExtendedAOA_Technical(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "", "", "", "TECHNICAL", "")
	if err != nil {
		t.Fatal(err)
	}
	if AOA(s) != AOATechnical {
		t.Fatalf("want TECHNICAL(%d), got %d", AOATechnical, AOA(s))
	}
}

func TestPackExtendedAOA_Early(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "", "", "", "EARLY", "")
	if err != nil {
		t.Fatal(err)
	}
	if AOA(s) != AOAEarly {
		t.Fatalf("want EARLY(%d), got %d", AOAEarly, AOA(s))
	}
}

func TestPackExtendedAOA_AllLevels(t *testing.T) {
	cases := []struct {
		input string
		want  uint32
	}{
		{"EARLY", AOAEarly},
		{"MID", AOAMid},
		{"LATE", AOALate},
		{"TECHNICAL", AOATechnical},
		{"", AOAEarly},
	}
	for _, tc := range cases {
		s, err := PackExtended("", "", "", "", nil, "", "", "", tc.input, "")
		if err != nil {
			t.Fatalf("aoa=%q: %v", tc.input, err)
		}
		if AOA(s) != tc.want {
			t.Fatalf("aoa=%q: want %d, got %d", tc.input, tc.want, AOA(s))
		}
	}
}

func TestPackExtendedAllDims(t *testing.T) {
	// All dimensions packed together and verified.
	s, err := PackExtended(
		"NEGATIVE", "STRONG", "EVALUATION", "GENERAL", nil,
		"ADJ", "HIGH", "LOW", "MID", "CONCRETE",
	)
	if err != nil {
		t.Fatalf("PackExtended: %v", err)
	}
	dec := Decode(s)
	if dec["polarity"] != "POSITIVE" || dec["polarity"] == "NEGATIVE" {
		// recalc: Polarity(s)
	}
	checks := map[string]string{
		"polarity":     "NEGATIVE",
		"intensity":    "STRONG",
		"role":         "EVALUATION",
		"pos":          "ADJ",
		"arousal":      "HIGH",
		"dominance":    "LOW",
		"aoa":          "MID",
		"concreteness": "CONCRETE",
	}
	for key, want := range checks {
		if got := dec[key]; got != want {
			t.Errorf("Decode[%q]: want %q, got %q (full sent=0x%08X)", key, want, got, s)
		}
	}
}

func TestPackExtendedInvalidPOS(t *testing.T) {
	_, err := PackExtended("", "", "", "", nil, "INVALID_POS", "", "", "", "")
	if err == nil {
		t.Fatal("invalid pos must return error")
	}
}

func TestPackExtendedInvalidArousal(t *testing.T) {
	_, err := PackExtended("", "", "", "", nil, "", "XXXX", "", "", "")
	if err == nil {
		t.Fatal("invalid arousal must return error")
	}
}

func TestPackExtendedInvalidDominance(t *testing.T) {
	_, err := PackExtended("", "", "", "", nil, "", "", "ZZZZZ", "", "")
	if err == nil {
		t.Fatal("invalid dominance must return error")
	}
}

func TestPackExtendedInvalidAOA(t *testing.T) {
	_, err := PackExtended("", "", "", "", nil, "", "", "", "BABY", "")
	if err == nil {
		t.Fatal("invalid aoa must return error")
	}
}

// ── Decode — new dimension names ──────────────────────────────────────────────

func TestDecodeAllNewDimensions(t *testing.T) {
	s := POSAdv | Concrete | ArousalMed | DominanceHigh | AOALate | PolarityPositive
	dec := Decode(s)

	if dec["pos"] != "ADV" {
		t.Errorf("pos: want ADV, got %q", dec["pos"])
	}
	if dec["concreteness"] != "CONCRETE" {
		t.Errorf("concreteness: want CONCRETE, got %q", dec["concreteness"])
	}
	if dec["arousal"] != "MED" {
		t.Errorf("arousal: want MED, got %q", dec["arousal"])
	}
	if dec["dominance"] != "HIGH" {
		t.Errorf("dominance: want HIGH, got %q", dec["dominance"])
	}
	if dec["aoa"] != "LATE" {
		t.Errorf("aoa: want LATE, got %q", dec["aoa"])
	}
}

func TestDecodeArousalNames(t *testing.T) {
	cases := []struct{ sent uint32; want string }{
		{ArousalNone, "NONE"},
		{ArousalLow, "LOW"},
		{ArousalMed, "MED"},
		{ArousalHigh, "HIGH"},
	}
	for _, tc := range cases {
		dec := Decode(tc.sent)
		if dec["arousal"] != tc.want {
			t.Errorf("arousal 0x%08X: want %q, got %q", tc.sent, tc.want, dec["arousal"])
		}
	}
}

func TestDecodeAOANames(t *testing.T) {
	cases := []struct{ sent uint32; want string }{
		{AOAEarly, "EARLY"},
		{AOAMid, "MID"},
		{AOALate, "LATE"},
		{AOATechnical, "TECHNICAL"},
	}
	for _, tc := range cases {
		dec := Decode(tc.sent)
		if dec["aoa"] != tc.want {
			t.Errorf("aoa 0x%X: want %q, got %q", tc.sent, tc.want, dec["aoa"])
		}
	}
}

func TestDecodePOSNames(t *testing.T) {
	cases := []struct{ sent uint32; want string }{
		{POSOther, "OTHER"},
		{POSNoun, "NOUN"},
		{POSVerb, "VERB"},
		{POSAdj, "ADJ"},
		{POSAdv, "ADV"},
		{POSParticle, "PARTICLE"},
		{POSPrep, "PREP"},
		{POSConj, "CONJ"},
	}
	for _, tc := range cases {
		dec := Decode(tc.sent)
		if dec["pos"] != tc.want {
			t.Errorf("POS 0x%08X: want %q, got %q", tc.sent, tc.want, dec["pos"])
		}
	}
}

func TestDecodeConcreteness(t *testing.T) {
	concrete := Decode(Concrete)
	if concrete["concreteness"] != "CONCRETE" {
		t.Errorf("want CONCRETE, got %q", concrete["concreteness"])
	}
	abstract := Decode(0)
	if abstract["concreteness"] != "ABSTRACT" {
		t.Errorf("want ABSTRACT, got %q", abstract["concreteness"])
	}
}
