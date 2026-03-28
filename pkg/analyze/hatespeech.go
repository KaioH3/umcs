package analyze

import (
	"regexp"
	"strings"
)

type HateCategory string

const (
	HateRacism          HateCategory = "RACISM"
	HateSexism          HateCategory = "SEXISM"
	HateHomophobia      HateCategory = "HOMOPHOBIA"
	HateReligiousHatred HateCategory = "RELIGIOUS_HATRED"
	HateViolence        HateCategory = "VIOLENCE"
	HateAbleism         HateCategory = "ABLEISM"
	HateGordophobia     HateCategory = "GORDOPHOBIA"
	HateCapacitism      HateCategory = "CAPACITISM"
)

type HateResult struct {
	IsHate     bool           `json:"is_hate"`
	Categories []HateCategory `json:"categories"`
	Confidence float64        `json:"confidence"`
	Text       string         `json:"text"`
}

var (
	racismPatterns = regexp.MustCompile(`(?i)(\bnazi\b|\bfascist\b|\bwhite power\b|\brace\b.*inferior|\bethnic\b.*cleansing|\bapartheid\b|\bsupremac(y|ist)\b|\banti-.*race\b|\bracial\b.*hate|\bimmigrant\b.*(bad|wrong|invad|steal)|\brefugee\b.*(bad|wrong|invad)|\basylum\b.*(seek|migr))`)

	sexismPatterns = regexp.MustCompile(`(?i)(\bwoman\b.*(should|need|can'?t).*(cook|clean|submit|obey|man)|\bbitch\b|\bwhore\b|\bslut\b|\bharass\b|\b(man|men)\b.*superior|\bfeminist\b.*(hate|kill|die)|\bwomen\b.*(belong|home|submit)|\bsexist\b|\bpatriarchy\b.*(bad|hate))`)

	homophobiaPatterns = regexp.MustCompile(`(?i)(\bgay\b.*(bad|wrong|disease|sin|hell)|\bhomosexual\b.*(bad|sin|abnormal)|\bqueer\b.*(bad|hate)|\b(lgbt|bisexual|transgender).* (bad|wrong|sin)|\bpride\b.*(shame|disgust)|\bsodomite\b|\bfaggot\b|\bdyke\b)`)

	religiousHatredPatterns = regexp.MustCompile(`(?i)(\bmuslim\b.*(terror|bomb|kill|invad|radical)|\bislam\b.*(terror|radical|extrem|bad)|\bjew\b.*(control|money|bank|media)|\bchristian\b.*(hypocrite|fake)|\bhindu\b.*(caste|lower)|\bbuddhis\b.*(cult|brainwash)|\bterrorist\b.*(islam|muslim)|\binfidel\b)`)

	violencePatterns = regexp.MustCompile(`(?i)(\bkill\b.*(them|all|you|it)|\b(murder|assassin|execut)\b.*(them|all|you)|death.*(to|for)|beat.*(up|them)|attack.*(them|it)|rape\b|\bstab\b|\bshoot\b|\bbomb\b|\bexplod\b|\bterror\b|\bmassacre\b|\bgenocide\b)`)

	ableismPatterns = regexp.MustCompile(`(?i)(\bretard\b|\b(spastic|crippled)\b.*(brain|mind|idiot)|\b(blind|deaf)\b.*(can'?t|shouldn)?t|\bdisabled\b.*(lazy|stupid)|special.*(needs|olympi|olympia)|normal.*(people|person).*(vs|versus|not)|(\bsane\b|\bable-bodied\b).* (vs|versus|not)|handicap.*(joke|stupid))`)

	gordophobiaPatterns = regexp.MustCompile(`(?i)(\bfat\b.*(ugly|lazy|stupid|cow|pig|whale)|\bobese\b.*(bad|wrong|sick)|\boverweight\b.*(should|lose|eat)|(you|people).*(fat|fatty)|weight.*(shame|joke|insult)|(no|not).*(date|marry|love).*fat|\bbulim\b|\banorex\b)`)

	capacitismPatterns = regexp.MustCompile(`(?i)(\bstupid\b|\bidiot\b|\bmoron\b|\bimbecil(e|)\b|\bdumb\b|\b(mentally?\s*)?disabled\b.*(person|people)|(\b retardation\b)|\b(autistic|asperger)\b.*(bad|weird|special)|\basperger\b.*(smart|rain|geek)|neurodivergent.*(bad|wrong|disorder)|\bADHD\b.*(disorder|problem|bad)|\bdisorder\b.*(mental|psych))`)
)

func DetectHateSpeech(text string) *HateResult {
	lower := strings.ToLower(text)
	categories := []HateCategory{}
	score := 0.0

	if racismPatterns.MatchString(lower) {
		categories = append(categories, HateRacism)
		score += 0.8
	}

	if sexismPatterns.MatchString(lower) {
		categories = append(categories, HateSexism)
		score += 0.75
	}

	if homophobiaPatterns.MatchString(lower) {
		categories = append(categories, HateHomophobia)
		score += 0.8
	}

	if religiousHatredPatterns.MatchString(lower) {
		categories = append(categories, HateReligiousHatred)
		score += 0.75
	}

	if violencePatterns.MatchString(lower) {
		categories = append(categories, HateViolence)
		score += 0.7
	}

	if ableismPatterns.MatchString(lower) {
		categories = append(categories, HateAbleism)
		score += 0.65
	}

	if gordophobiaPatterns.MatchString(lower) {
		categories = append(categories, HateGordophobia)
		score += 0.6
	}

	if capacitismPatterns.MatchString(lower) {
		categories = append(categories, HateCapacitism)
		score += 0.6
	}

	confidence := score
	if confidence > 1.0 {
		confidence = 1.0
	}

	isHate := len(categories) > 0 && confidence >= 0.3

	return &HateResult{
		IsHate:     isHate,
		Categories: categories,
		Confidence: confidence,
		Text:       text,
	}
}
