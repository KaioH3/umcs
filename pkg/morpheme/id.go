// Package morpheme implements the Universal Morpheme Coordinate System (UMCS).
// Every word gets a deterministic uint32 ID encoding its root and variant.
//
// ID layout:
//   bits 31..12 = root_id  (20 bits, up to 1,048,575 roots)
//   bits 11..0  = variant  (12 bits, up to 4,095 variants per root)
//
// Example:
//   root "negat" → root_id=1
//   "negative" (EN, variant 1) → WordID = (1<<12)|1 = 4097
//   "negativo" (PT, variant 2) → WordID = (1<<12)|2 = 4098
//   "negativ"  (DE, variant 3) → WordID = (1<<12)|3 = 4099
package morpheme

import "fmt"

const (
	RootShift   = 12
	VariantMask = 0xFFF // 12 bits
	MaxRootID   = (1 << 20) - 1
	MaxVariant  = VariantMask
)

// MakeWordID packs root_id and variant into a word ID.
func MakeWordID(rootID, variant uint32) (uint32, error) {
	if rootID == 0 || rootID > MaxRootID {
		return 0, fmt.Errorf("root_id %d out of range [1, %d]", rootID, MaxRootID)
	}
	if variant == 0 || variant > MaxVariant {
		return 0, fmt.Errorf("variant %d out of range [1, %d]", variant, MaxVariant)
	}
	return (rootID << RootShift) | variant, nil
}

// RootOf extracts the root_id from a word ID — one bit shift, zero lookup.
func RootOf(wordID uint32) uint32 {
	return wordID >> RootShift
}

// VariantOf extracts the variant from a word ID.
func VariantOf(wordID uint32) uint32 {
	return wordID & VariantMask
}

// Cognates returns true if two word IDs share the same root.
func Cognates(a, b uint32) bool {
	return RootOf(a) == RootOf(b)
}

// Validate checks that a word_id is consistent with the given root_id and variant.
func Validate(wordID, rootID, variant uint32) error {
	got, err := MakeWordID(rootID, variant)
	if err != nil {
		return err
	}
	if got != wordID {
		return fmt.Errorf("word_id %d != expected %d for root_id=%d variant=%d", wordID, got, rootID, variant)
	}
	return nil
}
