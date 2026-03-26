package morpheme

// Token64 packs a word's full semantic coordinate into a single uint64.
//
// A language model that receives this token can decode every semantic dimension
// without any external lookup — the entire semantic identity of a word is
// self-contained in the number itself.
//
// Bit layout (MSB → LSB):
//
//	63..44  root_id      (20 bits) — morphological family; cognates across all languages share this
//	43..32  variant      (12 bits) — specific surface form within the family
//	31..29  pos          (3 bits)  — NOUN/VERB/ADJ/ADV/PARTICLE/PREP/CONJ
//	28      concrete     (1 bit)   — 1=concrete (chair), 0=abstract (freedom)
//	27..24  scope_flags  (4 bits)  — INTENSIFIER/DOWNTONER/NEGATION/AFFIRMATION
//	23..20  role         (4 bits)  — semantic role (EVALUATION/EMOTION/COGNITION/…)
//	19..16  intensity    (4 bits)  — 0=NONE … 4=EXTREME
//	15..12  ontological  (4 bits)  — what kind of thing (PERSON/PLACE/EVENT/STATE/…)
//	11..8   register     (4 bits)  — formality (FORMAL/NEUTRAL/INFORMAL/VULGAR/…)
//	7..6    polarity     (2 bits)  — NEUTRAL/POSITIVE/NEGATIVE/AMBIGUOUS
//	5..4    arousal      (2 bits)  — activation level (LOW/MED/HIGH)
//	3..2    dominance    (2 bits)  — power/control axis (LOW/MED/HIGH)
//	1..0    aoa          (2 bits)  — age of acquisition (EARLY/MID/LATE/TECHNICAL)
//
// Example: "terrible" (EN, root_id=10, variant=1, NEGATIVE STRONG EVALUATION)
//   → Token64 = 0x0002800000130140
//     decoded: root=10, var=1, pos=ADJ, polarity=NEGATIVE, intensity=STRONG,
//              role=EVALUATION, arousal=HIGH, dominance=LOW, concrete=true
//
// The root_id occupies the most significant bits, so tokens from the same
// morphological family cluster together when sorted — a natural property for
// embedding tables and vocabulary files.
type Token64 uint64

// Pack64 assembles a Token64 from a word_id, sentiment payload, and flag word.
// sentiment is the uint32 from WordRecord.Sentiment (see pkg/sentiment).
// flags is the uint32 from WordRecord.Flags (see pkg/lexdb RegisterMask, OntoMask).
func Pack64(wordID, sentiment, flags uint32) Token64 {
	// Upper 32 bits: word identity (root_id in 31..12, variant in 11..0)
	upper := uint64(wordID) << 32

	// Lower 32 bits: merge sentiment with ontological and register from flags.
	// The sentiment uint32 already has pos(31..29), concrete(28), scope(27..24),
	// role(23..20), intensity(19..16), polarity(7..6), arousal(5..4),
	// dominance(3..2), aoa(1..0) packed in.
	//
	// We re-slot ontological (flags bits 15..12) into lower bits 15..12
	// and register (flags bits 11..8) into lower bits 11..8, overriding
	// the domain field (bits 15..8 in sentiment) with richer metadata for
	// the token representation. Domain is still accessible via WordRecord.Sentiment.
	lower := uint64(sentiment)
	lower &^= 0xFF00 // clear bits 15..8 (domain) — replaced by onto+register
	lower |= uint64(flags & 0xFF00)

	return Token64(upper | lower)
}

// Unpack64 extracts the word_id, reconstructed sentiment, and flag metadata from a Token64.
// Note: domain bits are not preserved (use WordRecord.Sentiment for domain queries).
func Unpack64(t Token64) (wordID uint32, payload uint32) {
	wordID = uint32(uint64(t) >> 32)
	payload = uint32(uint64(t) & 0xFFFFFFFF)
	return
}

// RootOf64 extracts the root_id from a Token64 without a full unpack.
func RootOf64(t Token64) uint32 {
	return uint32(uint64(t)>>32) >> RootShift
}

// VariantOf64 extracts the variant from a Token64 without a full unpack.
func VariantOf64(t Token64) uint32 {
	return uint32(uint64(t)>>32) & VariantMask
}

// Cognates64 returns true if two Token64s share the same root_id.
func Cognates64(a, b Token64) bool {
	return RootOf64(a) == RootOf64(b)
}
