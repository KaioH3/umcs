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

	portugueseRacismPatterns = regexp.MustCompile(`(?i)(\bpreto\b.*(burro|estupido|lixo|macaco|idiota|vagabundo|feio|desgraçado|merda|odiar|nojo)|macaco\b.*(preto|negro|branco)|\bnigga\b|\bnegreiro\b|\bnegão\b|\bnegrinho\b|racismo\b|racista\b|preto.*macaco|macaco.*preto|preto.*burro|preto.*idiota|preto.*desgraçado)`)

	portugueseSexismPatterns = regexp.MustCompile(`(?i)(\bviado\b|\bbicha\b|\bveado\b|\bbuneca\b|\bsapata\b|\bsapatão\b|\bhomosexual\b|\btravesti\b|\btrans\b.*(doente|errado|surgery)|sapatão\b.*(doente|pecado)|viado\b.*(nojo|doente|pecado)|bicha\b.*(nojo|doente)|bunda\b.*(sua|dar)|lolita\b|\bpiroca\b|\brola\b|\bporra\b.*(mulher|gay)|mulher\b.*(programa|meretriz|puta|vagabunda|entregue)|puta\b|\bputo\b|\bvadia\b|\bpiranha\b|\bgarota\b.*(programa|pega)|fdp\b|\bfoder\b|\bcaralho\b)`)

	portugueseAbleismPatterns = regexp.MustCompile(`(?i)(\bretardado\b|\bidiota\b|\bburro\b|\bestupido\b|\bpalhaço\b|\bdesgraçado\b|\btrouxa\b|\botario\b|\botário\b|\bmedíocre\b|\bfraco\b.*(mental|inteligente)|doente\b.*(mental|psico)|louco\b|\bcrazy\b|\btarado\b|\bpervertido\b|\bordinário\b|\bvagabundo\b|\bnojento\b|\bimundo\b|\bfedido\b|\bfdp\b|\bfilha da puta\b|\bfilho da puta\b|\bcu\b.*(de|mao)|viado\b|\bveado\b|\bbicha\b)`)

	spanishRacismPatterns = regexp.MustCompile(`(?i)(\bnegro\b.*(estúpido|idiota|burro|loco)|maricón\b|\bmono\b.*(negro|persona)|indio\b.*(sucio|primitivo)|gordo\b.*(estúpido|idiota)|racismo\b|racista\b|\bcholo\b)`)

	spanishSexismPatterns = regexp.MustCompile(`(?i)(\bmarica\b|\bmaricón\b|\bbimbo\b|\bzorra\b|\bputa\b|\bcochina\b|\bfulana\b|\btonta\b|\bempleada\b.*(sexual|obediente)|mujer\b.*(casa|cocina|obedecer)|homosexual\b|\b lésbic[oa]\b)`)

	spanishAbleismPatterns = regexp.MustCompile(`(?i)(\btonto\b|\bidiota\b|\bestúpido\b|\bburro\b|\bloco\b|\bloco\b|\bchiflado\b|\bdemente\b|\bretrasado\b|\bfrustrado\b|\bsubnormal\b)`)

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

	if racismPatterns.MatchString(lower) || portugueseRacismPatterns.MatchString(lower) || spanishRacismPatterns.MatchString(lower) {
		categories = append(categories, HateRacism)
		score += 0.8
	}

	if sexismPatterns.MatchString(lower) || portugueseSexismPatterns.MatchString(lower) || spanishSexismPatterns.MatchString(lower) {
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

	if ableismPatterns.MatchString(lower) || portugueseAbleismPatterns.MatchString(lower) || spanishAbleismPatterns.MatchString(lower) {
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
