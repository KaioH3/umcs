package analyze

import (
	"regexp"
	"strings"
)

type BiasLabel string

const (
	BiasLeft   BiasLabel = "LEFT"
	BiasRight  BiasLabel = "RIGHT"
	BiasCenter BiasLabel = "CENTER"
)

type PoliticalBiasResult struct {
	Bias       BiasLabel `json:"bias"`
	Confidence float64   `json:"confidence"`
	Indicators []string  `json:"indicators"`
	Text       string    `json:"text"`
}

var (
	leftIndicators = regexp.MustCompile(`(?i)(\bworkers[' ]?(rights?|unions?)\b|\bsocial (justice|democracy)\b|\b(progressive|liberal)\b.*(agenda|media|elite)|inequality\b|\b(wealth|income)\b.*(gap|redistribution|tax)|universal (healthcare|basic income)|\benvironment(al)?\b.*(protection|climate|green)|\bhuman rights\b|\bimmigration\b.*(reform|rights|refugee)|\bcollective\b.*(bargaining|ownership)|\blabor\b.*(rights|movement)|anti-?corporate|\bpublic\b.*(option|school|transport)|\bdemocrat(ic)?\b.*(reform|vote|access)|\b(equality|equity)\b|\bbipoc\b|\bmarxist|\bsocialist\b|\banarchist\b)`)

	rightIndicators = regexp.MustCompile(`(?i)(\bconservative\b.*(values|family|tradition)|traditional (family|values)|\bcapitalism\b.*(free|market|enterprise)|\bfree market\b|\bfiscal responsibility\b|\bsmall government\b|\bconstitution(alism)?\b|\bbill of rights\b|\bsecond amendment\b|\bgun rights\b|\bborder (security|control)\b|\bimmigration\b.*(enforcement|illegal|wall)|\btraditional marriage\b|\bfamily values\b|\bchristian\b.*(values|nation|heritage)|\bnational(ist)?\b|\bpatriot(ic|ism)?\b|\blaw and order\b|\bcapitalis|\bpro-?life\b|\bpro-?business\b|\breligious freedom\b|\beconomic freedom\b|\bpersonal responsibility\b|\bself-?made\b)`)

	centerIndicators = regexp.MustCompile(`(?i)(\bbipartisan\b|\bmoderate\b|\bcentrist\b|\bcommonsense\b|\bpragmatic\b|\bbalanced\b.*(approach|budget)|\bmainstream\b|\bfiscally\b.*(conservative|liberal)|\bsocially\b.*(liberal|conservative)|\bmoderate\b.*(views|position)|)\b`)
)

func DetectPoliticalBias(text string) *PoliticalBiasResult {
	lower := strings.ToLower(text)
	indicators := []string{}
	leftScore := 0.0
	rightScore := 0.0
	centerScore := 0.0

	leftMatches := leftIndicators.FindAllString(lower, -1)
	if len(leftMatches) > 0 {
		leftScore = float64(len(leftMatches)) * 0.25
		if leftScore > 1.0 {
			leftScore = 1.0
		}
		for _, m := range leftMatches {
			indicators = append(indicators, "LEFT:"+m)
		}
	}

	rightMatches := rightIndicators.FindAllString(lower, -1)
	if len(rightMatches) > 0 {
		rightScore = float64(len(rightMatches)) * 0.25
		if rightScore > 1.0 {
			rightScore = 1.0
		}
		for _, m := range rightMatches {
			indicators = append(indicators, "RIGHT:"+m)
		}
	}

	centerMatches := centerIndicators.FindAllString(lower, -1)
	if len(centerMatches) > 0 {
		centerScore = float64(len(centerMatches)) * 0.2
		if centerScore > 0.5 {
			centerScore = 0.5
		}
		for _, m := range centerMatches {
			indicators = append(indicators, "CENTER:"+m)
		}
	}

	var bias BiasLabel
	var confidence float64

	if leftScore > rightScore && leftScore > centerScore {
		bias = BiasLeft
		confidence = leftScore
	} else if rightScore > leftScore && rightScore > centerScore {
		bias = BiasRight
		confidence = rightScore
	} else if centerScore > 0 {
		bias = BiasCenter
		confidence = centerScore
	} else {
		bias = BiasCenter
		confidence = 0.3
	}

	if confidence < 0.3 {
		bias = BiasCenter
		confidence = 0.3
	}

	return &PoliticalBiasResult{
		Bias:       bias,
		Confidence: confidence,
		Indicators: indicators,
		Text:       text,
	}
}
