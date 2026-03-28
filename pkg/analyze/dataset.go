package analyze

import (
	"encoding/binary"
	"fmt"
	"sort"
	"strings"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/morpheme"
	"github.com/kak/umcs/pkg/sentiment"
)

var langNames = []string{
	"PT", "EN", "ES", "IT", "DE", "FR", "NL", "AR",
	"ZH", "JA", "RU", "KO", "TG", "HI", "BN", "ID",
	"TR", "FA", "SW", "UK", "PL", "SA", "TA", "HE",
	"LA", "FI", "DA", "HU", "SV", "CA", "RO", "CS",
	"GL", "MS", "SK", "SL", "HR", "BG", "GU", "EL",
	"IS", "PA", "KN", "NE", "EU", "TH", "VI", "GA",
	"MK", "TE", "AF", "UR", "NO", "SQ", "ML", "MR",
	"KA", "TL", "HY", "CY", "MT", "BS", "AM", "ET",
	"SR", "LV", "LT", "AZ", "LB", "SI",
}

func langFromID(id uint32) string {
	if id < uint32(len(langNames)) {
		return langNames[id]
	}
	return "XX"
}

type DatasetEntry struct {
	RootID    uint32   `json:"root_id"`
	RootStr   string   `json:"root_str"`
	Languages []string `json:"languages"`
	Words     []string `json:"words"`
	Polarity  string   `json:"polarity"`
	Intensity float64  `json:"intensity"`
	Emotions  []string `json:"emotions"`
	Arousal   float64  `json:"arousal"`
	Valence   float64  `json:"valence"`
	Dominance float64  `json:"dominance"`
	Concrete  bool     `json:"concrete"`
	POS       string   `json:"pos"`
	Register  string   `json:"register"`
}

type Dataset struct {
	Version    string         `json:"version"`
	Language   string         `json:"language"`
	EntryCount int            `json:"entry_count"`
	TotalWords int            `json:"total_words"`
	Entries    []DatasetEntry `json:"entries"`
}

func GenerateDataset(lex *lexdb.Lexicon, langFilter string) *Dataset {
	entries := make(map[uint32]*DatasetEntry)
	seen := make(map[string]bool)

	for i := range lex.Words {
		wr := &lex.Words[i]
		word := lex.WordStr(wr)
		if word == "" || seen[word] {
			continue
		}
		seen[word] = true

		rootID := morpheme.RootOf(wr.WordID)
		rootStr := ""
		root := lex.LookupRoot(rootID)
		if root != nil {
			rootStr = lex.RootStr(root)
		}
		if rootStr == "" {
			rootStr = word
		}

		wordLang := langFromID(wr.Lang)
		if langFilter != "" && wordLang != langFilter {
			continue
		}

		if _, ok := entries[rootID]; !ok {
			entries[rootID] = &DatasetEntry{
				RootID:    rootID,
				RootStr:   rootStr,
				Languages: []string{},
				Words:     []string{},
			}
		}

		entry := entries[rootID]

		langFound := false
		for _, l := range entry.Languages {
			if l == wordLang {
				langFound = true
				break
			}
		}
		if !langFound {
			entry.Languages = append(entry.Languages, wordLang)
		}

		entry.Words = append(entry.Words, word)

		sent := sentiment.Decode(wr.Sentiment)
		if entry.Polarity == "" {
			entry.Polarity = sent["polarity"]
		}
		if entry.POS == "" {
			entry.POS = sent["pos"]
		}
	}

	result := &Dataset{
		Version:    "1.0",
		Language:   langFilter,
		EntryCount: len(entries),
		TotalWords: 0,
		Entries:    make([]DatasetEntry, 0, len(entries)),
	}

	for _, entry := range entries {
		sort.Strings(entry.Words)
		sort.Strings(entry.Languages)
		result.TotalWords += len(entry.Words)
		result.Entries = append(result.Entries, *entry)
	}

	sort.Slice(result.Entries, func(i, j int) bool {
		return result.Entries[i].RootID < result.Entries[j].RootID
	})

	return result
}

type CompactCorpus struct {
	Version      string            `json:"version"`
	RootCount    int               `json:"root_count"`
	TotalWords   int               `json:"total_words"`
	Correlations []RootCorrelation `json:"correlations"`
}

type RootCorrelation struct {
	RootID       uint32            `json:"root_id"`
	RootStr      string            `json:"root_str"`
	Concepts     []ConceptRelation `json:"concepts"`
	Polarity     string            `json:"polarity"`
	EmotionGroup string            `json:"emotion_group"`
	Intensity    float64           `json:"intensity"`
}

type ConceptRelation struct {
	Word       string  `json:"word"`
	Lang       string  `json:"lang"`
	Similarity float64 `json:"similarity"`
	POS        string  `json:"pos"`
}

func GenerateCorrelations(lex *lexdb.Lexicon) *CompactCorpus {
	rootMap := make(map[uint32]*RootCorrelation)
	wordSet := make(map[string]bool)

	for i := range lex.Words {
		wr := &lex.Words[i]
		word := lex.WordStr(wr)
		if word == "" || wordSet[word] {
			continue
		}
		wordSet[word] = true

		rootID := morpheme.RootOf(wr.WordID)
		rootStr := ""
		root := lex.LookupRoot(rootID)
		if root != nil {
			rootStr = lex.RootStr(root)
		}
		if rootStr == "" {
			rootStr = word
		}

		if _, ok := rootMap[rootID]; !ok {
			rootMap[rootID] = &RootCorrelation{
				RootID:    rootID,
				RootStr:   rootStr,
				Concepts:  []ConceptRelation{},
				Intensity: 0.5,
			}
		}

		entry := rootMap[rootID]
		wordLang := langFromID(wr.Lang)
		sent := sentiment.Decode(wr.Sentiment)

		entry.Concepts = append(entry.Concepts, ConceptRelation{
			Word:       word,
			Lang:       wordLang,
			Similarity: 1.0,
			POS:        sent["pos"],
		})

		if entry.Polarity == "" {
			entry.Polarity = sent["polarity"]
		}
	}

	correlations := make([]RootCorrelation, 0, len(rootMap))
	totalWords := 0

	for _, rc := range rootMap {
		correlations = append(correlations, *rc)
		totalWords += len(rc.Concepts)

		emotionGroup := "neutral"
		if rc.Intensity > 0.7 {
			emotionGroup = "high_intensity"
		} else if rc.Intensity > 0.4 {
			emotionGroup = "medium_intensity"
		}
		if rc.Polarity == "POSITIVE" {
			emotionGroup += "_positive"
		} else if rc.Polarity == "NEGATIVE" {
			emotionGroup += "_negative"
		}
		rc.EmotionGroup = emotionGroup
	}

	sort.Slice(correlations, func(i, j int) bool {
		return correlations[i].RootID < correlations[j].RootID
	})

	return &CompactCorpus{
		Version:      "1.0",
		RootCount:    len(correlations),
		TotalWords:   totalWords,
		Correlations: correlations,
	}
}

type CrossLingualEmbedding struct {
	RootID    uint32    `json:"root_id"`
	RootStr   string    `json:"root_str"`
	Vector    []float32 `json:"vector"`
	Polarity  float32   `json:"polarity"`
	Languages []string  `json:"languages"`
	WordCount int       `json:"word_count"`
}

func GenerateCrossLingualEmbeddings(lex *lexdb.Lexicon) []CrossLingualEmbedding {
	rootData := make(map[uint32]*CrossLingualEmbedding)

	for i := range lex.Words {
		wr := &lex.Words[i]
		word := lex.WordStr(wr)
		if word == "" {
			continue
		}

		rootID := morpheme.RootOf(wr.WordID)
		rootStr := ""
		root := lex.LookupRoot(rootID)
		if root != nil {
			rootStr = lex.RootStr(root)
		}
		if rootStr == "" {
			rootStr = word
		}

		if _, ok := rootData[rootID]; !ok {
			rootData[rootID] = &CrossLingualEmbedding{
				RootID:    rootID,
				RootStr:   rootStr,
				Vector:    make([]float32, 8),
				Languages: []string{},
			}
		}

		entry := rootData[rootID]
		wordLang := langFromID(wr.Lang)

		langFound := false
		for _, l := range entry.Languages {
			if l == wordLang {
				langFound = true
				break
			}
		}
		if !langFound {
			entry.Languages = append(entry.Languages, wordLang)
		}

		sent := sentiment.Decode(wr.Sentiment)

		if entry.Polarity == 0 {
			if sent["polarity"] == "POSITIVE" {
				entry.Polarity = 1.0
			} else if sent["polarity"] == "NEGATIVE" {
				entry.Polarity = -1.0
			}
		}

		entry.Vector[0] = entry.Polarity
		entry.WordCount++
	}

	result := make([]CrossLingualEmbedding, 0, len(rootData))
	for _, emb := range rootData {
		result = append(result, *emb)
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].RootID < result[j].RootID
	})

	return result
}

type ProgrammingCorpus struct {
	Version      string               `json:"version"`
	Constructs   []ProgramConstruct   `json:"constructs"`
	Correlations []ProgramCorrelation `json:"correlations"`
}

type ProgramConstruct struct {
	Name      string   `json:"name"`
	Category  string   `json:"category"`
	Languages []string `json:"languages"`
	Keywords  []string `json:"keywords"`
}

type ProgramCorrelation struct {
	Concept    string   `json:"concept"`
	Category   string   `json:"category"`
	Similarity float64  `json:"similarity"`
	Languages  []string `json:"languages"`
	Keywords   []string `json:"keywords"`
}

func GenerateProgrammingCorpus() *ProgrammingCorpus {
	constructs := []ProgramConstruct{
		{Name: "variable", Category: "declaration", Languages: []string{"go", "python", "javascript", "java", "rust", "cpp"}, Keywords: []string{"var", "let", "const", "val", "def", "int", "str", "bool"}},
		{Name: "function", Category: "declaration", Languages: []string{"go", "python", "javascript", "java", "rust", "cpp"}, Keywords: []string{"func", "def", "function", "fn", "void", "int"}},
		{Name: "loop", Category: "control", Languages: []string{"go", "python", "javascript", "java", "rust", "cpp"}, Keywords: []string{"for", "while", "loop", "iterate"}},
		{Name: "conditional", Category: "control", Languages: []string{"go", "python", "javascript", "java", "rust", "cpp"}, Keywords: []string{"if", "else", "switch", "match", "when"}},
		{Name: "class", Category: "oop", Languages: []string{"python", "javascript", "java", "cpp", "go"}, Keywords: []string{"class", "struct", "type"}},
		{Name: "import", Category: "module", Languages: []string{"python", "javascript", "java", "go", "rust"}, Keywords: []string{"import", "require", "include", "use"}},
		{Name: "error_handling", Category: "exception", Languages: []string{"python", "javascript", "java", "go", "rust"}, Keywords: []string{"try", "catch", "except", "panic", "error"}},
		{Name: "async", Category: "concurrency", Languages: []string{"javascript", "python", "go", "rust"}, Keywords: []string{"async", "await", "promise", "goroutine", "await"}},
		{Name: "type", Category: "typing", Languages: []string{"go", "python", "javascript", "java", "rust", "cpp"}, Keywords: []string{"type", "interface", "trait", "struct", "enum"}},
		{Name: "return", Category: "control", Languages: []string{"go", "python", "javascript", "java", "rust", "cpp"}, Keywords: []string{"return", "yield", "->", "=>"}},
	}

	correlations := []ProgramCorrelation{
		{Concept: "null_check", Category: "safety", Similarity: 0.9, Languages: []string{"go", "java", "kotlin", "rust"}, Keywords: []string{"nil", "null", "None", "Optional"}},
		{Concept: "immutable", Category: "design", Similarity: 0.85, Languages: []string{"rust", "kotlin", "python", "go"}, Keywords: []string{"const", "final", "val", "let"}},
		{Concept: "lambda", Category: "function", Similarity: 0.95, Languages: []string{"python", "javascript", "java", "go"}, Keywords: []string{"lambda", "arrow", "->", "=>"}},
		{Concept: "list_comprehension", Category: "iteration", Similarity: 0.8, Languages: []string{"python", "javascript", "rust"}, Keywords: []string{"list", "for", "in", "map"}},
		{Concept: "pattern_matching", Category: "control", Similarity: 0.9, Languages: []string{"rust", "go", "python", "scala"}, Keywords: []string{"match", "switch", "case", "when"}},
		{Concept: "ownership", Category: "memory", Similarity: 0.95, Languages: []string{"rust"}, Keywords: []string{"ownership", "borrow", "lifetime", "move"}},
		{Concept: "generics", Category: "typing", Similarity: 0.85, Languages: []string{"java", "go", "rust", "cpp"}, Keywords: []string{"generic", "type", "T", "generic"}},
		{Concept: "defer", Category: "control", Similarity: 0.9, Languages: []string{"go", "python"}, Keywords: []string{"defer", "finally", "context"}},
	}

	return &ProgrammingCorpus{
		Version:      "1.0",
		Constructs:   constructs,
		Correlations: correlations,
	}
}

type MergedCorpus struct {
	Version     string            `json:"version"`
	Linguistic  LinguisticCorpus  `json:"linguistic"`
	Programming ProgrammingCorpus `json:"programming"`
	Embeddings  EmbeddingCorpus   `json:"embeddings"`
	Metadata    Metadata          `json:"metadata"`
}

type LinguisticCorpus struct {
	RootCount  int               `json:"root_count"`
	TotalWords int               `json:"total_words"`
	Entries    []RootCorrelation `json:"entries"`
}

type EmbeddingCorpus struct {
	Dimension int                     `json:"dimension"`
	Count     int                     `json:"count"`
	Vectors   []CrossLingualEmbedding `json:"vectors"`
}

type Metadata struct {
	RootCount   int `json:"root_count"`
	WordCount   int `json:"word_count"`
	LangCount   int `json:"lang_count"`
	ProgLangCnt int `json:"prog_langs"`
}

func GenerateMergedCorpus(lex *lexdb.Lexicon) *MergedCorpus {
	linguistic := GenerateCorrelations(lex)
	programming := GenerateProgrammingCorpus()
	embeddings := GenerateCrossLingualEmbeddings(lex)

	langs := make(map[string]bool)
	for _, wr := range lex.Words {
		langs[langFromID(wr.Lang)] = true
	}

	return &MergedCorpus{
		Version: "1.0",
		Linguistic: LinguisticCorpus{
			RootCount:  linguistic.RootCount,
			TotalWords: linguistic.TotalWords,
			Entries:    linguistic.Correlations,
		},
		Programming: *programming,
		Embeddings: EmbeddingCorpus{
			Dimension: 8,
			Count:     len(embeddings),
			Vectors:   embeddings,
		},
		Metadata: Metadata{
			RootCount:   linguistic.RootCount,
			WordCount:   linguistic.TotalWords,
			LangCount:   len(langs),
			ProgLangCnt: len(programming.Correlations),
		},
	}
}

func (d *Dataset) Summary() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("UMCS Dataset v%s\n", d.Version))
	b.WriteString(fmt.Sprintf("Entries: %d, Total Words: %d\n", d.EntryCount, d.TotalWords))

	langCounts := make(map[string]int)
	polarityCounts := make(map[string]int)

	for _, e := range d.Entries {
		for _, lang := range e.Languages {
			langCounts[lang]++
		}
		if e.Polarity != "" {
			polarityCounts[e.Polarity]++
		}
	}

	b.WriteString("\nLanguages:\n")
	for lang, count := range langCounts {
		b.WriteString(fmt.Sprintf("  %s: %d roots\n", lang, count))
	}

	b.WriteString("\nPolarity:\n")
	for pol, count := range polarityCounts {
		b.WriteString(fmt.Sprintf("  %s: %d\n", pol, count))
	}

	return b.String()
}

type BinaryCorpus struct {
	Magic      [4]byte
	Version    uint16
	Flags      uint16
	EntryCount uint32
	Roots      []byte
	Words      []byte
	LangIndex  []byte
}

func (c *CompactCorpus) ToBinary() *BinaryCorpus {
	bc := &BinaryCorpus{
		Magic:      [4]byte{'U', 'M', 'C', 'B'},
		Version:    1,
		EntryCount: uint32(len(c.Correlations)),
	}

	var roots []byte
	var words []byte

	for _, corr := range c.Correlations {
		roots = append(roots, uint8(corr.RootID&0xFF))
		roots = append(roots, uint8((corr.RootID>>8)&0xFF))
		roots = append(roots, uint8((corr.RootID>>16)&0xFF))
		roots = append(roots, uint8((corr.RootID>>24)&0xFF))

		for _, word := range corr.Concepts {
			words = append(words, []byte(word.Word)...)
			words = append(words, 0)
		}
	}

	bc.Roots = roots
	bc.Words = words

	return bc
}

func (bc *BinaryCorpus) MarshalBinary() ([]byte, error) {
	buf := make([]byte, 12)
	buf[0] = bc.Magic[0]
	buf[1] = bc.Magic[1]
	buf[2] = bc.Magic[2]
	buf[3] = bc.Magic[3]
	binary.LittleEndian.PutUint16(buf[4:6], bc.Version)
	binary.LittleEndian.PutUint16(buf[6:8], bc.Flags)
	binary.LittleEndian.PutUint32(buf[8:12], bc.EntryCount)

	buf = append(buf, bc.Roots...)
	buf = append(buf, bc.Words...)

	return buf, nil
}
