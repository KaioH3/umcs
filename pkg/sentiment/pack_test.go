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
