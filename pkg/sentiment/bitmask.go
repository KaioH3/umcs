// Package sentiment implements a compact uint32 bitmask encoding sentiment,
// semantic role, domain, polarity, and intensity for any word.
//
// Bitmask layout:
//   bits 31..28  reserved (0)
//   bit  27      INTENSIFIER        — amplifies adjacent sentiment
//   bit  26      DOWNTONER          — reduces adjacent sentiment
//   bit  25      NEGATION_MARKER    — inverts scope of next N tokens
//   bit  24      AFFIRMATION_MARKER — confirms scope
//   bits 23..20  SEMANTIC_ROLE (4-bit enum)
//   bits 19..16  INTENSITY (0–4)
//   bits 15..8   DOMAIN (8 flags)
//   bits 7..6    POLARITY (2 bits)
//   bits 5..0    reserved (0)
package sentiment

// Polarity values (bits 7..6).
const (
	PolarityNeutral   uint32 = 0b00 << 6
	PolarityPositive  uint32 = 0b01 << 6
	PolarityNegative  uint32 = 0b10 << 6
	PolarityAmbiguous uint32 = 0b11 << 6
	PolarityMask      uint32 = 0b11 << 6
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

// Polarity returns the polarity bits of a sentiment value.
func Polarity(s uint32) uint32 { return s & PolarityMask }

// Intensity returns the intensity level (0–4) from a sentiment value.
func Intensity(s uint32) uint32 { return (s & IntensityMask) >> 16 }

// Role returns the semantic role from a sentiment value.
func Role(s uint32) uint32 { return (s & RoleMask) >> 20 }

// Domain returns the domain bitmask from a sentiment value.
func Domain(s uint32) uint32 { return (s & DomainMask) >> 8 }

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
