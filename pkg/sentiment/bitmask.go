// Package sentiment implements a compact uint32 bitmask encoding the full
// semantic payload of a word: polarity, intensity, role, domain, scope,
// part-of-speech, concreteness, arousal, dominance, and age-of-acquisition.
//
// Bitmask layout (all 32 bits used):
//
//	bits 31..29  POS (3 bits)             — NOUN/VERB/ADJ/ADV/PARTICLE/PREP/CONJ
//	bit  28      CONCRETENESS (1 bit)     — 1=concrete, 0=abstract
//	bit  27      INTENSIFIER              — amplifies adjacent sentiment
//	bit  26      DOWNTONER                — reduces adjacent sentiment
//	bit  25      NEGATION_MARKER          — inverts scope of next N tokens
//	bit  24      AFFIRMATION_MARKER       — confirms scope
//	bits 23..20  SEMANTIC_ROLE (4-bit)    — EVALUATION/EMOTION/COGNITION/…
//	bits 19..16  INTENSITY (0–4)          — NONE/WEAK/MODERATE/STRONG/EXTREME
//	bits 15..8   DOMAIN (8 flags)         — GENERAL/FINANCIAL/MEDICAL/…
//	bits 7..6    POLARITY (2 bits)        — NEUTRAL/POSITIVE/NEGATIVE/AMBIGUOUS
//	bits 5..4    AROUSAL (2 bits)         — activation: NONE/LOW/MED/HIGH
//	bits 3..2    DOMINANCE (2 bits)       — power/control: NONE/LOW/MED/HIGH
//	bits 1..0    AOA (2 bits)             — age of acquisition: EARLY/MID/LATE/TECHNICAL
//
// Combined with the word_id coordinate (root_id<<12|variant) this sentiment
// uint32 forms the lower half of a Token64 — a single uint64 that encodes
// the complete semantic position of a word without any external lookup.
package sentiment

// ── Part of speech (bits 31..29) ─────────────────────────────────────────────

const (
	POSOther    uint32 = 0 << 29
	POSNoun     uint32 = 1 << 29
	POSVerb     uint32 = 2 << 29
	POSAdj      uint32 = 3 << 29
	POSAdv      uint32 = 4 << 29
	POSParticle uint32 = 5 << 29
	POSPrep     uint32 = 6 << 29
	POSConj     uint32 = 7 << 29
	POSMask     uint32 = 0b111 << 29
)

// Concreteness (bit 28): 1 = concrete (chair, fire), 0 = abstract (freedom, time).

const (
	Concrete uint32 = 1 << 28
	Abstract uint32 = 0
)

// ── Polarity (bits 7..6) ──────────────────────────────────────────────────────

// Polarity values (bits 7..6).
const (
	PolarityNeutral   uint32 = 0b00 << 6
	PolarityPositive  uint32 = 0b01 << 6
	PolarityNegative  uint32 = 0b10 << 6
	PolarityAmbiguous uint32 = 0b11 << 6
	PolarityMask      uint32 = 0b11 << 6
)

// ── Arousal (bits 5..4) ───────────────────────────────────────────────────────
// Psycholinguistic activation level, orthogonal to valence.
// HIGH arousal: "rage", "ecstasy", "panic". LOW: "boredom", "calm", "content".

const (
	ArousalNone uint32 = 0 << 4
	ArousalLow  uint32 = 1 << 4
	ArousalMed  uint32 = 2 << 4
	ArousalHigh uint32 = 3 << 4
	ArousalMask uint32 = 0b11 << 4
)

// ── Dominance (bits 3..2) ─────────────────────────────────────────────────────
// Sense of power/control the concept evokes (VAD model, third axis).
// HIGH: "command", "authority". LOW: "trapped", "helpless", "submissive".

const (
	DominanceNone uint32 = 0 << 2
	DominanceLow  uint32 = 1 << 2
	DominanceMed  uint32 = 2 << 2
	DominanceHigh uint32 = 3 << 2
	DominanceMask uint32 = 0b11 << 2
)

// ── Age of Acquisition (bits 1..0) ────────────────────────────────────────────
// When a native speaker typically learns this word.
// EARLY = first 1000 words a child learns (mama, water, no, good).
// TECHNICAL = learned in professional/academic contexts only.

const (
	AOAEarly     uint32 = 0 // first ~1000 words (child ≤3y)
	AOAMid       uint32 = 1 // school age (4–12y)
	AOALate      uint32 = 2 // adult vocabulary
	AOATechnical uint32 = 3 // domain-specific / academic
	AOAMask      uint32 = 0b11
)

// Intensity values (bits 19..16). Multiply raw × 65536.
const (
	IntensityNone     uint32 = 0 << 16
	IntensityWeak     uint32 = 1 << 16
	IntensityModerate uint32 = 2 << 16
	IntensityStrong   uint32 = 3 << 16
	IntensityExtreme  uint32 = 4 << 16
	IntensityMask     uint32 = 0xF << 16
)

// Domain flags (bits 15..8).
const (
	DomainGeneral    uint32 = 1 << 8
	DomainFinancial  uint32 = 1 << 9
	DomainMedical    uint32 = 1 << 10
	DomainLegal      uint32 = 1 << 11
	DomainTechnical  uint32 = 1 << 12
	DomainSocial     uint32 = 1 << 13
	DomainPolitical  uint32 = 1 << 14
	DomainAcademic   uint32 = 1 << 15
	DomainMask       uint32 = 0xFF << 8
)

// Semantic roles (bits 23..20).
const (
	RoleNone            uint32 = 0 << 20
	RoleEvaluation      uint32 = 1 << 20
	RoleEmotion         uint32 = 2 << 20
	RoleCognition       uint32 = 3 << 20
	RoleVolition        uint32 = 4 << 20
	RoleCausation       uint32 = 5 << 20
	RoleTemporal        uint32 = 6 << 20
	RoleQuantifier      uint32 = 7 << 20
	RoleConnector       uint32 = 8 << 20
	RoleNegationMarker  uint32 = 9 << 20
	RoleIntensifier     uint32 = 10 << 20
	RoleDowntoner       uint32 = 11 << 20
	RoleMask            uint32 = 0xF << 20
)

// Scope flags (bits 27..24).
const (
	FlagIntensifier       uint32 = 1 << 27
	FlagDowntoner         uint32 = 1 << 26
	FlagNegationMarker    uint32 = 1 << 25
	FlagAffirmationMarker uint32 = 1 << 24
	FlagMask              uint32 = 0xF << 24
)

// ── Accessors ─────────────────────────────────────────────────────────────────

// Polarity returns the polarity bits of a sentiment value.
func Polarity(s uint32) uint32 { return s & PolarityMask }

// Intensity returns the intensity level (0–4) from a sentiment value.
func Intensity(s uint32) uint32 { return (s & IntensityMask) >> 16 }

// Role returns the semantic role from a sentiment value.
func Role(s uint32) uint32 { return (s & RoleMask) >> 20 }

// Domain returns the domain bitmask from a sentiment value.
func Domain(s uint32) uint32 { return (s & DomainMask) >> 8 }

// POS returns the part-of-speech from a sentiment value.
func POS(s uint32) uint32 { return (s & POSMask) >> 29 }

// IsConcrete reports whether the word refers to a concrete concept.
func IsConcrete(s uint32) bool { return s&Concrete != 0 }

// Arousal returns the arousal tier (0=NONE, 1=LOW, 2=MED, 3=HIGH).
func Arousal(s uint32) uint32 { return (s & ArousalMask) >> 4 }

// Dominance returns the dominance tier (0=NONE, 1=LOW, 2=MED, 3=HIGH).
func Dominance(s uint32) uint32 { return (s & DominanceMask) >> 2 }

// AOA returns the age-of-acquisition tier (AOAEarly/AOAMid/AOALate/AOATechnical).
func AOA(s uint32) uint32 { return s & AOAMask }

// IsNegationMarker reports whether the word inverts sentiment scope.
func IsNegationMarker(s uint32) bool { return s&FlagNegationMarker != 0 }

// IsIntensifier reports whether the word amplifies adjacent sentiment.
func IsIntensifier(s uint32) bool { return s&FlagIntensifier != 0 }

// IsDowntoner reports whether the word reduces adjacent sentiment.
func IsDowntoner(s uint32) bool { return s&FlagDowntoner != 0 }

// PolaritySign returns +1 for positive, -1 for negative, 0 for neutral/ambiguous.
func PolaritySign(s uint32) int {
	switch Polarity(s) {
	case PolarityPositive:
		return 1
	case PolarityNegative:
		return -1
	default:
		return 0
	}
}

// Weight returns the sentiment weight: polarity × intensity (signed).
func Weight(s uint32) int {
	return PolaritySign(s) * int(Intensity(s))
}
