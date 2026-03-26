// Package phon defines phonological constants packed into the Flags uint32 field
// of a WordRecord. These bits occupy the upper 11 bits (31..21) of Flags, which
// are free in the v1 format (v1 used only bits 20..0).
//
// # Phonological dimensions
//
// All values follow established linguistic standards:
//
//   - Syllable count: 0..15 (upper bound of 15 covers the vast majority of words)
//   - Stress: ISO 80000-style placement categories used in cross-linguistic phonology
//     (Gussenhoven & Jacobs "Understanding Phonology")
//   - Valency: Lucien Tesnière's dependency grammar valency (0..5)
//   - IronyCapable: marks lexical items that routinely participate in ironic usage
//     (Attardo, "Ironic Negation")
//   - Neologism: word coined post-1990 or not yet in major reference dictionaries
//
// # Bit layout within Flags uint32
//
//	bits 31..28  syllable count (4 bits, 0=unknown, 1-15)
//	bits 27..26  stress pattern  (2 bits)
//	bits 25..23  valency         (3 bits)
//	bit  22      irony_capable
//	bit  21      neologism
//	bits 20..0   (used by lexdb: cultural_specific, polysemy, ontological, register, flags)
package phon

const (
	// SyllableMask extracts the syllable-count field (bits 31..28).
	SyllableMask uint32 = 0xF << 28
	// SyllableShift is the bit offset of the syllable count field.
	SyllableShift = 28

	// Stress pattern (bits 27..26) — following IPA/phonological convention.
	// Languages with fixed stress: Turkish (final), French (final), Czech (initial).
	// Languages with default penultimate: Spanish, Italian, Polish.
	// Portuguese: penultimate by default, oxytone (final) when marked.
	StressMask           uint32 = 0x3 << 26
	StressUnknown        uint32 = 0 << 26 // not annotated
	StressFinal          uint32 = 1 << 26 // oxytone   (e.g. PT: negação, FR: toujours)
	StressPenultimate    uint32 = 2 << 26 // paroxytone (default in ES/IT/PT most words)
	StressAntepenultimate uint32 = 3 << 26 // proparoxytone (PT: lâmpada, ES: sílaba)

	// Valency (bits 25..23) — Tesnière dependency grammar (1959).
	// Valency counts the number of syntactic arguments a verb requires.
	// Non-verbal parts of speech use ValencyNA.
	ValencyMask     uint32 = 0x7 << 23
	ValencyNA       uint32 = 0 << 23 // not a predicate (nouns, adj, adv, particles)
	ValencyIntrans  uint32 = 1 << 23 // 0 arguments  (e.g. "snow", "arrive")
	ValencyTrans    uint32 = 2 << 23 // 1 argument   (e.g. "eat", "see")
	ValencyDitrans  uint32 = 3 << 23 // 2 arguments  (e.g. "give", "show")
	ValencyCopular  uint32 = 4 << 23 // copula/linking verb (e.g. "be", "seem", "become")
	ValencyModal    uint32 = 5 << 23 // modal/auxiliary (e.g. "can", "must", "poder")

	// IronyCapable (bit 22) — the word regularly participates in ironic inversion.
	// E.g. "wonderful" used sarcastically, "brilliant" (EN slang), "ótimo" (PT ironic).
	// Based on the Irony Flagging corpus (Attardo et al.) and lexical pragmatics.
	IronyCapable uint32 = 1 << 22

	// Neologism (bit 21) — word coined or lexicalized post-1990, or absent from
	// major reference dictionaries (Merriam-Webster, DPLP, DRAE, Larousse).
	// Useful for AoA (age-of-acquisition) and register inference.
	Neologism uint32 = 1 << 21
)

// Syllables extracts the syllable count from a packed Flags value.
// Returns 0 if unknown (not annotated).
func Syllables(flags uint32) uint32 {
	return (flags & SyllableMask) >> SyllableShift
}

// SetSyllables encodes syllable count n into flags. n is clamped to [0, 15].
func SetSyllables(flags, n uint32) uint32 {
	if n > 15 {
		n = 15
	}
	return (flags &^ SyllableMask) | (n << SyllableShift)
}

// Stress extracts the stress pattern from a packed Flags value.
func Stress(flags uint32) uint32 {
	return flags & StressMask
}

// SetStress encodes a stress constant (StressFinal, StressPenultimate, etc.) into flags.
func SetStress(flags, s uint32) uint32 {
	return (flags &^ StressMask) | (s & StressMask)
}

// Valency extracts the valency class from a packed Flags value.
func Valency(flags uint32) uint32 {
	return flags & ValencyMask
}

// SetValency encodes a valency constant (ValencyIntrans, ValencyTrans, etc.) into flags.
func SetValency(flags, v uint32) uint32 {
	return (flags &^ ValencyMask) | (v & ValencyMask)
}

// StressName returns the standard phonological name for a stress value.
func StressName(flags uint32) string {
	switch Stress(flags) {
	case StressFinal:
		return "OXYTONE"
	case StressPenultimate:
		return "PAROXYTONE"
	case StressAntepenultimate:
		return "PROPAROXYTONE"
	default:
		return "UNKNOWN"
	}
}

// ValencyName returns the Tesnière valency class name.
func ValencyName(flags uint32) string {
	switch Valency(flags) {
	case ValencyIntrans:
		return "INTRANSITIVE"
	case ValencyTrans:
		return "TRANSITIVE"
	case ValencyDitrans:
		return "DITRANSITIVE"
	case ValencyCopular:
		return "COPULAR"
	case ValencyModal:
		return "MODAL"
	default:
		return "N/A"
	}
}
