package sentiment

import (
	"fmt"
	"strings"
)

// Pack assembles a sentiment bitmask from the core sentiment fields.
// New dimensions (POS, arousal, dominance, AoA, concreteness) default to zero.
// Use PackExtended for full annotation.
func Pack(polarity, intensity, role, domain string, flags []string) (uint32, error) {
	var s uint32

	p, err := parsePolarity(polarity)
	if err != nil {
		return 0, err
	}
	s |= p

	i, err := parseIntensity(intensity)
	if err != nil {
		return 0, err
	}
	s |= i

	r, err := parseRole(role)
	if err != nil {
		return 0, err
	}
	s |= r

	d, err := parseDomain(domain)
	if err != nil {
		return 0, err
	}
	s |= d

	for _, f := range flags {
		fl, err := parseFlag(f)
		if err != nil {
			return 0, err
		}
		s |= fl
	}

	return s, nil
}

// PackExtended assembles the full sentiment bitmask including all semantic dimensions.
// Empty strings default to 0/NONE for the respective field.
func PackExtended(polarity, intensity, role, domain string, flags []string, pos, arousal, dominance, aoa, concreteness string) (uint32, error) {
	s, err := Pack(polarity, intensity, role, domain, flags)
	if err != nil {
		return 0, err
	}

	p, err := parsePOS(pos)
	if err != nil {
		return 0, err
	}
	s |= p

	ar, err := parseArousal(arousal)
	if err != nil {
		return 0, err
	}
	s |= ar

	dom, err := parseDominance(dominance)
	if err != nil {
		return 0, err
	}
	s |= dom

	a, err := parseAOA(aoa)
	if err != nil {
		return 0, err
	}
	s |= a

	if strings.ToUpper(strings.TrimSpace(concreteness)) == "CONCRETE" {
		s |= Concrete
	}

	return s, nil
}

// Decode returns a human-readable map of all sentiment fields.
func Decode(s uint32) map[string]string {
	return map[string]string{
		"polarity":    polarityName(Polarity(s)),
		"intensity":   intensityName(Intensity(s)),
		"role":        roleName(Role(s)),
		"domain":      domainNames(Domain(s)),
		"flags":       flagNames(s),
		"pos":         posName(POS(s)),
		"arousal":     levelName(Arousal(s)),
		"dominance":   levelName(Dominance(s)),
		"aoa":         aoaName(AOA(s)),
		"concreteness": concretenessName(s),
	}
}

func parsePolarity(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "NEUTRAL", "":
		return PolarityNeutral, nil
	case "POSITIVE":
		return PolarityPositive, nil
	case "NEGATIVE":
		return PolarityNegative, nil
	case "AMBIGUOUS":
		return PolarityAmbiguous, nil
	}
	return 0, fmt.Errorf("unknown polarity %q", s)
}

func parseIntensity(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "NONE", "":
		return IntensityNone, nil
	case "WEAK":
		return IntensityWeak, nil
	case "MODERATE":
		return IntensityModerate, nil
	case "STRONG":
		return IntensityStrong, nil
	case "EXTREME":
		return IntensityExtreme, nil
	}
	return 0, fmt.Errorf("unknown intensity %q", s)
}

func parseRole(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "NONE", "":
		return RoleNone, nil
	case "EVALUATION":
		return RoleEvaluation, nil
	case "EMOTION":
		return RoleEmotion, nil
	case "COGNITION":
		return RoleCognition, nil
	case "VOLITION":
		return RoleVolition, nil
	case "CAUSATION":
		return RoleCausation, nil
	case "TEMPORAL":
		return RoleTemporal, nil
	case "QUANTIFIER":
		return RoleQuantifier, nil
	case "CONNECTOR":
		return RoleConnector, nil
	case "NEGATION_MARKER":
		return RoleNegationMarker, nil
	case "INTENSIFIER":
		return RoleIntensifier, nil
	case "DOWNTONER":
		return RoleDowntoner, nil
	case "AFFIRMATION_MARKER":
		return RoleNone, nil // affirmation is a scope flag, not a role; treat as NONE role
	}
	return 0, fmt.Errorf("unknown semantic role %q", s)
}

func parseDomain(s string) (uint32, error) {
	var d uint32
	for _, part := range strings.Split(s, "|") {
		part = strings.ToUpper(strings.TrimSpace(part))
		switch part {
		case "GENERAL", "":
			d |= DomainGeneral
		case "FINANCIAL":
			d |= DomainFinancial
		case "MEDICAL":
			d |= DomainMedical
		case "LEGAL":
			d |= DomainLegal
		case "TECHNICAL":
			d |= DomainTechnical
		case "SOCIAL":
			d |= DomainSocial
		case "POLITICAL":
			d |= DomainPolitical
		case "ACADEMIC":
			d |= DomainAcademic
		default:
			return 0, fmt.Errorf("unknown domain %q", part)
		}
	}
	if d == 0 {
		d = DomainGeneral
	}
	return d, nil
}

func parseFlag(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "INTENSIFIER":
		return FlagIntensifier, nil
	case "DOWNTONER":
		return FlagDowntoner, nil
	case "NEGATION_MARKER":
		return FlagNegationMarker, nil
	case "AFFIRMATION_MARKER":
		return FlagAffirmationMarker, nil
	case "":
		return 0, nil
	}
	return 0, fmt.Errorf("unknown flag %q", s)
}

func parsePOS(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "", "OTHER":
		return POSOther, nil
	case "NOUN":
		return POSNoun, nil
	case "VERB":
		return POSVerb, nil
	case "ADJ":
		return POSAdj, nil
	case "ADV":
		return POSAdv, nil
	case "PARTICLE":
		return POSParticle, nil
	case "PREP":
		return POSPrep, nil
	case "CONJ":
		return POSConj, nil
	}
	return 0, fmt.Errorf("unknown pos %q", s)
}

func parseArousal(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "", "NONE":
		return ArousalNone, nil
	case "LOW":
		return ArousalLow, nil
	case "MED", "MEDIUM":
		return ArousalMed, nil
	case "HIGH":
		return ArousalHigh, nil
	}
	return 0, fmt.Errorf("unknown arousal %q", s)
}

func parseDominance(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "", "NONE":
		return DominanceNone, nil
	case "LOW":
		return DominanceLow, nil
	case "MED", "MEDIUM":
		return DominanceMed, nil
	case "HIGH":
		return DominanceHigh, nil
	}
	return 0, fmt.Errorf("unknown dominance %q", s)
}

func parseAOA(s string) (uint32, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "", "EARLY":
		return AOAEarly, nil
	case "MID":
		return AOAMid, nil
	case "LATE":
		return AOALate, nil
	case "TECHNICAL":
		return AOATechnical, nil
	}
	return 0, fmt.Errorf("unknown aoa %q", s)
}

func posName(p uint32) string {
	names := []string{"OTHER", "NOUN", "VERB", "ADJ", "ADV", "PARTICLE", "PREP", "CONJ"}
	if int(p) < len(names) {
		return names[p]
	}
	return "OTHER"
}

func levelName(v uint32) string {
	switch v {
	case 1:
		return "LOW"
	case 2:
		return "MED"
	case 3:
		return "HIGH"
	default:
		return "NONE"
	}
}

func aoaName(v uint32) string {
	switch v {
	case AOAMid:
		return "MID"
	case AOALate:
		return "LATE"
	case AOATechnical:
		return "TECHNICAL"
	default:
		return "EARLY"
	}
}

func concretenessName(s uint32) string {
	if IsConcrete(s) {
		return "CONCRETE"
	}
	return "ABSTRACT"
}

func polarityName(p uint32) string {
	switch p {
	case PolarityPositive:
		return "POSITIVE"
	case PolarityNegative:
		return "NEGATIVE"
	case PolarityAmbiguous:
		return "AMBIGUOUS"
	default:
		return "NEUTRAL"
	}
}

func intensityName(i uint32) string {
	switch i {
	case 1:
		return "WEAK"
	case 2:
		return "MODERATE"
	case 3:
		return "STRONG"
	case 4:
		return "EXTREME"
	default:
		return "NONE"
	}
}

func roleName(r uint32) string {
	names := []string{"NONE", "EVALUATION", "EMOTION", "COGNITION", "VOLITION",
		"CAUSATION", "TEMPORAL", "QUANTIFIER", "CONNECTOR", "NEGATION_MARKER",
		"INTENSIFIER", "DOWNTONER"}
	if int(r) < len(names) {
		return names[r]
	}
	return "UNKNOWN"
}

func domainNames(d uint32) string {
	var parts []string
	flags := []struct {
		bit  uint32
		name string
	}{
		{DomainGeneral >> 8, "GENERAL"},
		{DomainFinancial >> 8, "FINANCIAL"},
		{DomainMedical >> 8, "MEDICAL"},
		{DomainLegal >> 8, "LEGAL"},
		{DomainTechnical >> 8, "TECHNICAL"},
		{DomainSocial >> 8, "SOCIAL"},
		{DomainPolitical >> 8, "POLITICAL"},
		{DomainAcademic >> 8, "ACADEMIC"},
	}
	for _, f := range flags {
		if d&f.bit != 0 {
			parts = append(parts, f.name)
		}
	}
	if len(parts) == 0 {
		return "NONE"
	}
	return strings.Join(parts, "|")
}

func flagNames(s uint32) string {
	var parts []string
	if s&FlagNegationMarker != 0 {
		parts = append(parts, "NEGATION_MARKER")
	}
	if s&FlagAffirmationMarker != 0 {
		parts = append(parts, "AFFIRMATION_MARKER")
	}
	if s&FlagIntensifier != 0 {
		parts = append(parts, "INTENSIFIER")
	}
	if s&FlagDowntoner != 0 {
		parts = append(parts, "DOWNTONER")
	}
	return strings.Join(parts, "|")
}
