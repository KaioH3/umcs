package phon_test

import (
	"testing"

	"github.com/kak/umcs/pkg/phon"
)

func TestSyllablesRoundtrip(t *testing.T) {
	for _, n := range []uint32{0, 1, 2, 3, 5, 10, 15} {
		flags := phon.SetSyllables(0, n)
		got := phon.Syllables(flags)
		if got != n {
			t.Errorf("syllables %d: roundtrip got %d", n, got)
		}
	}
}

func TestSyllablesClamp(t *testing.T) {
	flags := phon.SetSyllables(0, 99)
	if phon.Syllables(flags) != 15 {
		t.Error("syllables >15 must clamp to 15")
	}
}

func TestSyllablesPreservesLowBits(t *testing.T) {
	// Existing flags in low bits must not be disturbed
	const existingLow uint32 = 0x001FFFFF // all low 21 bits set
	flags := phon.SetSyllables(existingLow, 7)
	if phon.Syllables(flags) != 7 {
		t.Error("syllable set failed with existing low bits")
	}
	if flags&existingLow != existingLow {
		t.Error("SetSyllables must not modify bits 20..0")
	}
}

func TestStressRoundtrip(t *testing.T) {
	cases := []uint32{phon.StressFinal, phon.StressPenultimate, phon.StressAntepenultimate, phon.StressUnknown}
	for _, s := range cases {
		flags := phon.SetStress(0, s)
		got := phon.Stress(flags)
		if got != s {
			t.Errorf("stress 0x%X: roundtrip got 0x%X", s, got)
		}
	}
}

func TestValencyRoundtrip(t *testing.T) {
	cases := []uint32{
		phon.ValencyNA, phon.ValencyIntrans, phon.ValencyTrans,
		phon.ValencyDitrans, phon.ValencyCopular, phon.ValencyModal,
	}
	for _, v := range cases {
		flags := phon.SetValency(0, v)
		got := phon.Valency(flags)
		if got != v {
			t.Errorf("valency 0x%X: roundtrip got 0x%X", v, got)
		}
	}
}

func TestCombinedFieldsDoNotOverlap(t *testing.T) {
	// Set all phon fields simultaneously; verify each reads back correctly.
	var flags uint32
	flags = phon.SetSyllables(flags, 3)
	flags = phon.SetStress(flags, phon.StressPenultimate)
	flags = phon.SetValency(flags, phon.ValencyTrans)
	flags |= phon.IronyCapable
	flags |= phon.Neologism

	if phon.Syllables(flags) != 3 {
		t.Errorf("syllables: want 3, got %d", phon.Syllables(flags))
	}
	if phon.Stress(flags) != phon.StressPenultimate {
		t.Errorf("stress: want PAROXYTONE, got 0x%X", phon.Stress(flags))
	}
	if phon.Valency(flags) != phon.ValencyTrans {
		t.Errorf("valency: want TRANS, got 0x%X", phon.Valency(flags))
	}
	if flags&phon.IronyCapable == 0 {
		t.Error("irony_capable bit lost")
	}
	if flags&phon.Neologism == 0 {
		t.Error("neologism bit lost")
	}
}

func TestPhonBitsDoNotTouchLow21(t *testing.T) {
	// All phonology bits must live in 31..21; low 20..0 must be untouched.
	phonMask := phon.SyllableMask | phon.StressMask | phon.ValencyMask | phon.IronyCapable | phon.Neologism
	if phonMask&0x001FFFFF != 0 {
		t.Errorf("phonology mask overlaps with low 21 bits: 0x%08X", phonMask&0x001FFFFF)
	}
}

func TestStressName(t *testing.T) {
	cases := map[uint32]string{
		phon.StressFinal:           "OXYTONE",
		phon.StressPenultimate:     "PAROXYTONE",
		phon.StressAntepenultimate: "PROPAROXYTONE",
		phon.StressUnknown:         "UNKNOWN",
	}
	for stress, want := range cases {
		flags := phon.SetStress(0, stress)
		if got := phon.StressName(flags); got != want {
			t.Errorf("StressName(0x%X): want %q, got %q", stress, want, got)
		}
	}
}

func TestValencyName(t *testing.T) {
	cases := map[uint32]string{
		phon.ValencyIntrans: "INTRANSITIVE",
		phon.ValencyTrans:   "TRANSITIVE",
		phon.ValencyDitrans: "DITRANSITIVE",
		phon.ValencyCopular: "COPULAR",
		phon.ValencyModal:   "MODAL",
		phon.ValencyNA:      "N/A",
	}
	for v, want := range cases {
		flags := phon.SetValency(0, v)
		if got := phon.ValencyName(flags); got != want {
			t.Errorf("ValencyName(0x%X): want %q, got %q", v, want, got)
		}
	}
}
