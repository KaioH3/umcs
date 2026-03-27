package sentiment

import (
	"strings"
	"testing"
)

// ── IsDowntoner ─────────────────────────────────────────────────────────────

func TestIsDowntoner_Set(t *testing.T) {
	s := FlagDowntoner | PolarityNeutral
	if !IsDowntoner(s) {
		t.Fatal("IsDowntoner must return true when FlagDowntoner bit is set")
	}
}

func TestIsDowntoner_Unset(t *testing.T) {
	s := FlagIntensifier | PolarityPositive
	if IsDowntoner(s) {
		t.Fatal("IsDowntoner must return false when FlagDowntoner bit is not set")
	}
}

func TestIsDowntoner_Zero(t *testing.T) {
	if IsDowntoner(0) {
		t.Fatal("IsDowntoner(0) must be false")
	}
}

// ── parseRole — all role values ─────────────────────────────────────────────

func TestParseRole_AllValues(t *testing.T) {
	cases := []struct {
		input string
		want  uint32
	}{
		{"NONE", RoleNone},
		{"", RoleNone},
		{"EVALUATION", RoleEvaluation},
		{"EMOTION", RoleEmotion},
		{"COGNITION", RoleCognition},
		{"VOLITION", RoleVolition},
		{"CAUSATION", RoleCausation},
		{"TEMPORAL", RoleTemporal},
		{"QUANTIFIER", RoleQuantifier},
		{"CONNECTOR", RoleConnector},
		{"NEGATION_MARKER", RoleNegationMarker},
		{"INTENSIFIER", RoleIntensifier},
		{"DOWNTONER", RoleDowntoner},
		{"AFFIRMATION_MARKER", RoleNone}, // affirmation is a scope flag, mapped to NONE role
	}
	for _, tc := range cases {
		got, err := parseRole(tc.input)
		if err != nil {
			t.Fatalf("parseRole(%q): unexpected error: %v", tc.input, err)
		}
		if got != tc.want {
			t.Errorf("parseRole(%q) = 0x%X, want 0x%X", tc.input, got, tc.want)
		}
	}
}

func TestParseRole_CaseInsensitive(t *testing.T) {
	got, err := parseRole("  evaluation  ")
	if err != nil {
		t.Fatalf("parseRole with spaces/lowercase: %v", err)
	}
	if got != RoleEvaluation {
		t.Fatalf("want RoleEvaluation, got 0x%X", got)
	}
}

func TestParseRole_Invalid(t *testing.T) {
	_, err := parseRole("CAUSAL")
	if err == nil {
		t.Fatal("parseRole('CAUSAL') must return error (not a valid role)")
	}
	_, err = parseRole("EXPERIENTIAL")
	if err == nil {
		t.Fatal("parseRole('EXPERIENTIAL') must return error")
	}
	_, err = parseRole("EVALUATIVE")
	if err == nil {
		t.Fatal("parseRole('EVALUATIVE') must return error")
	}
}

// ── parsePOS — all POS values ───────────────────────────────────────────────

func TestParsePOS_AllValues(t *testing.T) {
	cases := []struct {
		input string
		want  uint32
	}{
		{"", POSOther},
		{"OTHER", POSOther},
		{"NOUN", POSNoun},
		{"VERB", POSVerb},
		{"ADJ", POSAdj},
		{"ADV", POSAdv},
		{"PARTICLE", POSParticle},
		{"PREP", POSPrep},
		{"CONJ", POSConj},
	}
	for _, tc := range cases {
		got, err := parsePOS(tc.input)
		if err != nil {
			t.Fatalf("parsePOS(%q): unexpected error: %v", tc.input, err)
		}
		if got != tc.want {
			t.Errorf("parsePOS(%q) = 0x%X, want 0x%X", tc.input, got, tc.want)
		}
	}
}

func TestParsePOS_CaseInsensitive(t *testing.T) {
	got, err := parsePOS(" noun ")
	if err != nil {
		t.Fatalf("parsePOS with spaces/lowercase: %v", err)
	}
	if got != POSNoun {
		t.Fatalf("want POSNoun, got 0x%X", got)
	}
}

func TestParsePOS_Invalid(t *testing.T) {
	_, err := parsePOS("ADJECTIVE")
	if err == nil {
		t.Fatal("parsePOS('ADJECTIVE') must return error")
	}
}

// ── polarityName — AMBIGUOUS case ───────────────────────────────────────────

func TestPolarityName_Ambiguous(t *testing.T) {
	got := polarityName(PolarityAmbiguous)
	if got != "AMBIGUOUS" {
		t.Fatalf("polarityName(PolarityAmbiguous) = %q, want %q", got, "AMBIGUOUS")
	}
}

func TestPolarityName_AllValues(t *testing.T) {
	cases := []struct {
		input uint32
		want  string
	}{
		{PolarityNeutral, "NEUTRAL"},
		{PolarityPositive, "POSITIVE"},
		{PolarityNegative, "NEGATIVE"},
		{PolarityAmbiguous, "AMBIGUOUS"},
	}
	for _, tc := range cases {
		got := polarityName(tc.input)
		if got != tc.want {
			t.Errorf("polarityName(0x%X) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

// ── intensityName — all intensity levels ────────────────────────────────────

func TestIntensityName_AllLevels(t *testing.T) {
	cases := []struct {
		input uint32
		want  string
	}{
		{0, "NONE"},
		{1, "WEAK"},
		{2, "MODERATE"},
		{3, "STRONG"},
		{4, "EXTREME"},
		{99, "NONE"}, // out of range defaults to NONE
	}
	for _, tc := range cases {
		got := intensityName(tc.input)
		if got != tc.want {
			t.Errorf("intensityName(%d) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

// ── flagNames — all flag combinations ───────────────────────────────────────

func TestFlagNames_NoFlags(t *testing.T) {
	got := flagNames(0)
	if got != "" {
		t.Fatalf("flagNames(0) = %q, want empty", got)
	}
}

func TestFlagNames_SingleFlags(t *testing.T) {
	cases := []struct {
		flag uint32
		want string
	}{
		{FlagNegationMarker, "NEGATION_MARKER"},
		{FlagAffirmationMarker, "AFFIRMATION_MARKER"},
		{FlagIntensifier, "INTENSIFIER"},
		{FlagDowntoner, "DOWNTONER"},
	}
	for _, tc := range cases {
		got := flagNames(tc.flag)
		if got != tc.want {
			t.Errorf("flagNames(0x%X) = %q, want %q", tc.flag, got, tc.want)
		}
	}
}

func TestFlagNames_CombinedFlags(t *testing.T) {
	// All flags set
	all := FlagNegationMarker | FlagAffirmationMarker | FlagIntensifier | FlagDowntoner
	got := flagNames(all)
	for _, want := range []string{"NEGATION_MARKER", "AFFIRMATION_MARKER", "INTENSIFIER", "DOWNTONER"} {
		if !strings.Contains(got, want) {
			t.Errorf("flagNames(all): want %q in %q", want, got)
		}
	}
	// Should have 3 pipe separators for 4 flags
	if strings.Count(got, "|") != 3 {
		t.Errorf("flagNames(all): want 3 pipes, got %d in %q", strings.Count(got, "|"), got)
	}
}

func TestFlagNames_TwoFlags(t *testing.T) {
	s := FlagIntensifier | FlagNegationMarker
	got := flagNames(s)
	if !strings.Contains(got, "INTENSIFIER") || !strings.Contains(got, "NEGATION_MARKER") {
		t.Errorf("flagNames(INTENSIFIER|NEGATION_MARKER) = %q", got)
	}
	if strings.Count(got, "|") != 1 {
		t.Errorf("want 1 pipe, got %d in %q", strings.Count(got, "|"), got)
	}
}

// ── Decode with AMBIGUOUS polarity (via flagNames and polarityName paths) ───

func TestDecode_AmbiguousPolarity(t *testing.T) {
	s := PolarityAmbiguous | IntensityExtreme | DomainGeneral
	dec := Decode(s)
	if dec["polarity"] != "AMBIGUOUS" {
		t.Errorf("Decode polarity: want AMBIGUOUS, got %q", dec["polarity"])
	}
	if dec["intensity"] != "EXTREME" {
		t.Errorf("Decode intensity: want EXTREME, got %q", dec["intensity"])
	}
}

func TestDecode_DowntonerFlag(t *testing.T) {
	s := FlagDowntoner
	dec := Decode(s)
	if !strings.Contains(dec["flags"], "DOWNTONER") {
		t.Errorf("Decode flags: want DOWNTONER in %q", dec["flags"])
	}
}

func TestDecode_IntensityWeak(t *testing.T) {
	s := IntensityWeak
	dec := Decode(s)
	if dec["intensity"] != "WEAK" {
		t.Errorf("Decode intensity: want WEAK, got %q", dec["intensity"])
	}
}

// ── Pack round-trip for Downtoner ───────────────────────────────────────────

func TestPackDowntoner(t *testing.T) {
	s, err := Pack("NEUTRAL", "NONE", "DOWNTONER", "GENERAL", []string{"DOWNTONER"})
	if err != nil {
		t.Fatal(err)
	}
	if !IsDowntoner(s) {
		t.Fatal("packed downtoner must report IsDowntoner=true")
	}
	if IsIntensifier(s) {
		t.Fatal("downtoner must not be intensifier")
	}
}

// ── PackExtended for remaining POS values ───────────────────────────────────

func TestPackExtendedPOS_Particle(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "PARTICLE", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if POS(s) != POS(POSParticle) {
		t.Fatalf("want PARTICLE(%d), got %d", POS(POSParticle), POS(s))
	}
}

func TestPackExtendedPOS_Prep(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "PREP", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if POS(s) != POS(POSPrep) {
		t.Fatalf("want PREP(%d), got %d", POS(POSPrep), POS(s))
	}
}

func TestPackExtendedPOS_Conj(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "CONJ", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if POS(s) != POS(POSConj) {
		t.Fatalf("want CONJ(%d), got %d", POS(POSConj), POS(s))
	}
}

func TestPackExtendedPOS_Other(t *testing.T) {
	s, err := PackExtended("", "", "", "", nil, "OTHER", "", "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	if POS(s) != POS(POSOther) {
		t.Fatalf("want OTHER(%d), got %d", POS(POSOther), POS(s))
	}
}

// ── Pack with AMBIGUOUS polarity ────────────────────────────────────────────

func TestPackAmbiguousPolarity(t *testing.T) {
	s, err := Pack("AMBIGUOUS", "MODERATE", "NONE", "GENERAL", nil)
	if err != nil {
		t.Fatal(err)
	}
	if Polarity(s) != PolarityAmbiguous {
		t.Fatalf("want PolarityAmbiguous, got 0x%X", Polarity(s))
	}
	// Weight for ambiguous should be 0
	if Weight(s) != 0 {
		t.Fatalf("Weight(AMBIGUOUS) = %d, want 0", Weight(s))
	}
}

// ── parseIntensity — all values ─────────────────────────────────────────────

func TestParseIntensity_AllValues(t *testing.T) {
	cases := []struct {
		input string
		want  uint32
	}{
		{"NONE", IntensityNone},
		{"", IntensityNone},
		{"WEAK", IntensityWeak},
		{"MODERATE", IntensityModerate},
		{"STRONG", IntensityStrong},
		{"EXTREME", IntensityExtreme},
	}
	for _, tc := range cases {
		s, err := Pack("NEUTRAL", tc.input, "NONE", "GENERAL", nil)
		if err != nil {
			t.Fatalf("parseIntensity(%q): %v", tc.input, err)
		}
		got := Intensity(s)
		want := Intensity(tc.want)
		if got != want {
			t.Errorf("parseIntensity(%q): got %d, want %d", tc.input, got, want)
		}
	}
}

func TestParseIntensity_Invalid(t *testing.T) {
	_, err := Pack("NEUTRAL", "SUPER", "NONE", "GENERAL", nil)
	if err == nil {
		t.Fatal("parseIntensity('SUPER') must return error")
	}
}

// ── parseDomain — all values ────────────────────────────────────────────────

func TestParseDomain_AllValues(t *testing.T) {
	domains := []string{"GENERAL", "FINANCIAL", "MEDICAL", "LEGAL", "TECHNICAL", "SOCIAL", "POLITICAL", "ACADEMIC"}
	for _, d := range domains {
		s, err := Pack("NEUTRAL", "NONE", "NONE", d, nil)
		if err != nil {
			t.Fatalf("parseDomain(%q): %v", d, err)
		}
		dom := Domain(s)
		if dom == 0 {
			t.Errorf("parseDomain(%q): domain bits are 0", d)
		}
	}
}

func TestParseDomain_Invalid(t *testing.T) {
	_, err := Pack("NEUTRAL", "NONE", "NONE", "CULINARY", nil)
	if err == nil {
		t.Fatal("parseDomain('CULINARY') must return error")
	}
}

// ── parseFlag — all values ──────────────────────────────────────────────────

func TestParseFlag_AllValues(t *testing.T) {
	flags := []string{"INTENSIFIER", "DOWNTONER", "NEGATION_MARKER", "AFFIRMATION_MARKER"}
	for _, f := range flags {
		s, err := Pack("NEUTRAL", "NONE", "NONE", "GENERAL", []string{f})
		if err != nil {
			t.Fatalf("parseFlag(%q): %v", f, err)
		}
		if s == 0 {
			t.Errorf("parseFlag(%q): result is 0", f)
		}
	}
}

func TestParseFlag_Empty(t *testing.T) {
	s, err := Pack("NEUTRAL", "NONE", "NONE", "GENERAL", []string{""})
	if err != nil {
		t.Fatalf("parseFlag(''): %v", err)
	}
	// Empty flag should not set any scope bits
	if s&FlagMask != 0 {
		t.Fatal("empty flag should not set scope bits")
	}
}

func TestParseFlag_Invalid(t *testing.T) {
	_, err := Pack("NEUTRAL", "NONE", "NONE", "GENERAL", []string{"BOOSTER"})
	if err == nil {
		t.Fatal("parseFlag('BOOSTER') must return error")
	}
}

// ── parsePolarity — all values ──────────────────────────────────────────────

func TestParsePolarity_AllValues(t *testing.T) {
	cases := []struct {
		input string
		want  uint32
	}{
		{"NEUTRAL", PolarityNeutral},
		{"", PolarityNeutral},
		{"POSITIVE", PolarityPositive},
		{"NEGATIVE", PolarityNegative},
		{"AMBIGUOUS", PolarityAmbiguous},
	}
	for _, tc := range cases {
		s, err := Pack(tc.input, "NONE", "NONE", "GENERAL", nil)
		if err != nil {
			t.Fatalf("parsePolarity(%q): %v", tc.input, err)
		}
		if Polarity(s) != tc.want {
			t.Errorf("parsePolarity(%q): got 0x%X, want 0x%X", tc.input, Polarity(s), tc.want)
		}
	}
}

func TestParsePolarity_Invalid(t *testing.T) {
	_, err := Pack("MIXED", "NONE", "NONE", "GENERAL", nil)
	if err == nil {
		t.Fatal("parsePolarity('MIXED') must return error")
	}
}

// ── roleName out of range ───────────────────────────────────────────────────

func TestRoleName_OutOfRange(t *testing.T) {
	// roleName is called via Decode; we need a value with role > 11
	// Role bits 23..20 with value 15 (0xF) → out of range
	s := uint32(0xF) << 20
	dec := Decode(s)
	if dec["role"] != "UNKNOWN" {
		t.Errorf("roleName(15) via Decode: want UNKNOWN, got %q", dec["role"])
	}
}

// ── posName out of range ────────────────────────────────────────────────────

func TestPosName_OutOfRange(t *testing.T) {
	// posName is called via Decode; POS bits 31..29 can hold max 7 (CONJ).
	// All 8 values (0-7) are valid POS, so posName never goes out of range
	// in practice, but we verify the boundary.
	dec := Decode(POSConj)
	if dec["pos"] != "CONJ" {
		t.Errorf("posName(CONJ) via Decode: want CONJ, got %q", dec["pos"])
	}
}

// ── Pack error propagation ──────────────────────────────────────────────────

func TestPackInvalidPolarity(t *testing.T) {
	_, err := Pack("INVALID", "NONE", "NONE", "GENERAL", nil)
	if err == nil {
		t.Fatal("Pack with invalid polarity must error")
	}
}

func TestPackInvalidIntensity(t *testing.T) {
	_, err := Pack("NEUTRAL", "INVALID", "NONE", "GENERAL", nil)
	if err == nil {
		t.Fatal("Pack with invalid intensity must error")
	}
}

func TestPackInvalidRole(t *testing.T) {
	_, err := Pack("NEUTRAL", "NONE", "INVALID", "GENERAL", nil)
	if err == nil {
		t.Fatal("Pack with invalid role must error")
	}
}

func TestPackInvalidDomain(t *testing.T) {
	_, err := Pack("NEUTRAL", "NONE", "NONE", "INVALID", nil)
	if err == nil {
		t.Fatal("Pack with invalid domain must error")
	}
}

func TestPackInvalidFlag(t *testing.T) {
	_, err := Pack("NEUTRAL", "NONE", "NONE", "GENERAL", []string{"INVALID"})
	if err == nil {
		t.Fatal("Pack with invalid flag must error")
	}
}

// ── Decode with all domain flags ────────────────────────────────────────────

func TestDecodeAllDomains(t *testing.T) {
	s, err := Pack("NEUTRAL", "NONE", "NONE", "GENERAL|FINANCIAL|MEDICAL|LEGAL|TECHNICAL|SOCIAL|POLITICAL|ACADEMIC", nil)
	if err != nil {
		t.Fatal(err)
	}
	dec := Decode(s)
	for _, want := range []string{"GENERAL", "FINANCIAL", "MEDICAL", "LEGAL", "TECHNICAL", "SOCIAL", "POLITICAL", "ACADEMIC"} {
		if !strings.Contains(dec["domain"], want) {
			t.Errorf("Decode domain: want %q in %q", want, dec["domain"])
		}
	}
}

// ── Decode empty domain ─────────────────────────────────────────────────────

func TestDecodeDomainNone(t *testing.T) {
	// Domain bits all zero → "NONE"
	dec := Decode(0)
	if dec["domain"] != "NONE" {
		t.Errorf("Decode domain(0): want NONE, got %q", dec["domain"])
	}
}
