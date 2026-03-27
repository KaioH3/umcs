package ingest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ── helpers ──────────────────────────────────────────────────────────────────

func writeTempFile(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

// ── normalize ────────────────────────────────────────────────────────────────

func TestNormalize_LowercaseTrim(t *testing.T) {
	if got := normalize("  Hello  "); got != "hello" {
		t.Errorf("got %q, want %q", got, "hello")
	}
}

func TestNormalize_Unicode(t *testing.T) {
	if got := normalize("Café"); got != "café" {
		t.Errorf("got %q, want %q", got, "café")
	}
}

func TestNormalize_Empty(t *testing.T) {
	if got := normalize(""); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

// ── isAlpha ──────────────────────────────────────────────────────────────────

func TestIsAlpha_Word(t *testing.T) {
	if !isAlpha("hello") {
		t.Error("expected true")
	}
}

func TestIsAlpha_Numbers(t *testing.T) {
	if isAlpha("123") {
		t.Error("expected false for pure numbers")
	}
}

func TestIsAlpha_Empty(t *testing.T) {
	if isAlpha("") {
		t.Error("expected false for empty")
	}
}

func TestIsAlpha_Mixed(t *testing.T) {
	if !isAlpha("h3llo") {
		t.Error("expected true for mixed")
	}
}

// ── valenceToPolarity ────────────────────────────────────────────────────────

func TestValenceToPolarity_Positive(t *testing.T) {
	if got := valenceToPolarity(0.5, -0.1, 0.1); got != "POSITIVE" {
		t.Errorf("got %q", got)
	}
}

func TestValenceToPolarity_Negative(t *testing.T) {
	if got := valenceToPolarity(-0.5, -0.1, 0.1); got != "NEGATIVE" {
		t.Errorf("got %q", got)
	}
}

func TestValenceToPolarity_Neutral(t *testing.T) {
	if got := valenceToPolarity(0.0, -0.1, 0.1); got != "NEUTRAL" {
		t.Errorf("got %q", got)
	}
}

func TestValenceToPolarity_Boundary(t *testing.T) {
	if got := valenceToPolarity(0.1, -0.1, 0.1); got != "NEUTRAL" {
		t.Errorf("got %q at exact threshold", got)
	}
}

// ── valenceToIntensity ───────────────────────────────────────────────────────

func TestValenceToIntensity_Extreme(t *testing.T) {
	if got := valenceToIntensity(4.5, 5.0); got != "EXTREME" {
		t.Errorf("got %q", got)
	}
}

func TestValenceToIntensity_Strong(t *testing.T) {
	if got := valenceToIntensity(3.5, 5.0); got != "STRONG" {
		t.Errorf("got %q", got)
	}
}

func TestValenceToIntensity_Moderate(t *testing.T) {
	if got := valenceToIntensity(2.5, 5.0); got != "MODERATE" {
		t.Errorf("got %q", got)
	}
}

func TestValenceToIntensity_Weak(t *testing.T) {
	if got := valenceToIntensity(1.5, 5.0); got != "WEAK" {
		t.Errorf("got %q", got)
	}
}

func TestValenceToIntensity_None(t *testing.T) {
	if got := valenceToIntensity(0.5, 5.0); got != "NONE" {
		t.Errorf("got %q", got)
	}
}

// ── vadToLevel ───────────────────────────────────────────────────────────────

func TestVadToLevel_Low(t *testing.T) {
	if got := vadToLevel(2.0, 1.0, 9.0); got != "LOW" {
		t.Errorf("got %q", got)
	}
}

func TestVadToLevel_Med(t *testing.T) {
	if got := vadToLevel(5.0, 1.0, 9.0); got != "MED" {
		t.Errorf("got %q", got)
	}
}

func TestVadToLevel_High(t *testing.T) {
	if got := vadToLevel(8.0, 1.0, 9.0); got != "HIGH" {
		t.Errorf("got %q", got)
	}
}

func TestVadToLevel_NegativeRange(t *testing.T) {
	if got := vadToLevel(-0.8, -1.0, 1.0); got != "LOW" {
		t.Errorf("got %q", got)
	}
	if got := vadToLevel(0.8, -1.0, 1.0); got != "HIGH" {
		t.Errorf("got %q", got)
	}
}

// ── ImportAFINN ──────────────────────────────────────────────────────────────

func TestImportAFINN_Basic(t *testing.T) {
	data := "abandon\t-2\nabandoned\t-2\nhappy\t3\njoy\t4\n"
	p := writeTempFile(t, "afinn.txt", data)

	entries, res, err := ImportAFINN(p)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 4 {
		t.Errorf("total=%d, want 4", res.Total)
	}
	if len(entries) != 4 {
		t.Fatalf("entries=%d, want 4", len(entries))
	}
	// abandon = -2 → NEGATIVE
	if entries[0].Polarity != "NEGATIVE" {
		t.Errorf("abandon polarity=%q", entries[0].Polarity)
	}
	// happy = 3 → POSITIVE
	if entries[2].Polarity != "POSITIVE" {
		t.Errorf("happy polarity=%q", entries[2].Polarity)
	}
	if entries[0].Source != "AFINN" {
		t.Errorf("source=%q", entries[0].Source)
	}
}

func TestImportAFINN_SkipsMultiWord(t *testing.T) {
	data := "can't stand\t-3\nhappy\t3\n"
	p := writeTempFile(t, "afinn.txt", data)

	entries, _, err := ImportAFINN(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("entries=%d, want 1 (should skip multi-word)", len(entries))
	}
}

func TestImportAFINN_EmptyFile(t *testing.T) {
	p := writeTempFile(t, "afinn.txt", "")
	entries, res, err := ImportAFINN(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 || res.Total != 0 {
		t.Errorf("expected empty results")
	}
}

func TestImportAFINN_InvalidScore(t *testing.T) {
	data := "word\tnot_a_number\n"
	p := writeTempFile(t, "afinn.txt", data)

	_, res, err := ImportAFINN(p)
	if err != nil {
		t.Fatal(err)
	}
	if res.Errors != 1 {
		t.Errorf("errors=%d, want 1", res.Errors)
	}
}

func TestImportAFINN_NonAlpha(t *testing.T) {
	data := "$$$\t-2\nhappy\t3\n"
	p := writeTempFile(t, "afinn.txt", data)

	entries, _, err := ImportAFINN(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("entries=%d, want 1 (skip non-alpha)", len(entries))
	}
}

// ── ImportVADER ──────────────────────────────────────────────────────────────

func TestImportVADER_Basic(t *testing.T) {
	data := "happy\t2.3\t0.9\t[1, 3, 3]\nsad\t-1.8\t0.7\t[-2, -1, -2]\n"
	p := writeTempFile(t, "vader.txt", data)

	entries, res, err := ImportVADER(p)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 2 {
		t.Errorf("total=%d, want 2", res.Total)
	}
	if entries[0].Polarity != "POSITIVE" {
		t.Errorf("happy=%q", entries[0].Polarity)
	}
	if entries[1].Polarity != "NEGATIVE" {
		t.Errorf("sad=%q", entries[1].Polarity)
	}
}

func TestImportVADER_SkipsEmojis(t *testing.T) {
	data := "$:\t-1.5\t0.8\t[-1]\nhappy\t2.3\t0.9\t[1]\n"
	p := writeTempFile(t, "vader.txt", data)

	entries, _, err := ImportVADER(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("entries=%d, want 1", len(entries))
	}
}

// ── ImportBingLiu ────────────────────────────────────────────────────────────

func TestImportBingLiu_Basic(t *testing.T) {
	posData := "; comment\nhappy\njoyful\n"
	negData := "; comment\nsad\nawful\nterrible\n"
	posP := writeTempFile(t, "positive.txt", posData)
	negP := writeTempFile(t, "negative.txt", negData)

	entries, res, err := ImportBingLiu(posP, negP)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 5 {
		t.Errorf("total=%d, want 5", res.Total)
	}
	if res.ByPolarity["POSITIVE"] != 2 {
		t.Errorf("positive=%d", res.ByPolarity["POSITIVE"])
	}
	if res.ByPolarity["NEGATIVE"] != 3 {
		t.Errorf("negative=%d", res.ByPolarity["NEGATIVE"])
	}
	// Check source
	for _, e := range entries {
		if e.Source != "BingLiu" {
			t.Errorf("source=%q", e.Source)
		}
	}
}

func TestImportBingLiu_SkipsComments(t *testing.T) {
	posData := "; this is a comment\n;another\nhappy\n"
	negData := ""
	posP := writeTempFile(t, "positive.txt", posData)
	negP := writeTempFile(t, "negative.txt", negData)

	entries, _, err := ImportBingLiu(posP, negP)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("entries=%d, want 1", len(entries))
	}
}

func TestImportBingLiu_MissingFile(t *testing.T) {
	posP := writeTempFile(t, "positive.txt", "happy\n")
	_, _, err := ImportBingLiu(posP, "/nonexistent")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

// ── ImportNRCVAD ─────────────────────────────────────────────────────────────

func TestImportNRCVAD_Basic(t *testing.T) {
	data := "term\tvalence\tarousal\tdominance\nhappy\t0.960\t0.735\t0.633\nsad\t-0.877\t-0.212\t-0.530\n"
	p := writeTempFile(t, "nrcvad.txt", data)

	entries, res, err := ImportNRCVAD(p)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 2 {
		t.Errorf("total=%d, want 2", res.Total)
	}
	if entries[0].Polarity != "POSITIVE" {
		t.Errorf("happy=%q", entries[0].Polarity)
	}
	if entries[0].Arousal == "" {
		t.Error("expected arousal for happy")
	}
	if entries[0].Dominance == "" {
		t.Error("expected dominance for happy")
	}
	if entries[1].Polarity != "NEGATIVE" {
		t.Errorf("sad=%q", entries[1].Polarity)
	}
}

func TestImportNRCVAD_SkipsMultiWord(t *testing.T) {
	data := "term\tvalence\tarousal\tdominance\na bit\t-0.096\t-0.264\t-0.214\nhappy\t0.5\t0.3\t0.2\n"
	p := writeTempFile(t, "nrcvad.txt", data)

	entries, _, err := ImportNRCVAD(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("entries=%d, want 1", len(entries))
	}
}

func TestImportNRCVAD_MissingFile(t *testing.T) {
	_, _, err := ImportNRCVAD("/nonexistent")
	if err == nil {
		t.Error("expected error")
	}
}

// ── ImportWarrinerVAD ────────────────────────────────────────────────────────

func TestImportWarrinerVAD_Basic(t *testing.T) {
	data := "Word,V.Mean.Sum,A.Mean.Sum,D.Mean.Sum\nhappy,7.47,5.57,6.49\nsad,2.38,4.13,3.49\n"
	p := writeTempFile(t, "warriner.csv", data)

	entries, res, err := ImportWarrinerVAD(p)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 2 {
		t.Errorf("total=%d, want 2", res.Total)
	}
	// happy: v=7.47, center=5 → +2.47 → POSITIVE
	if entries[0].Polarity != "POSITIVE" {
		t.Errorf("happy=%q", entries[0].Polarity)
	}
	// sad: v=2.38, center=5 → -2.62 → NEGATIVE
	if entries[1].Polarity != "NEGATIVE" {
		t.Errorf("sad=%q", entries[1].Polarity)
	}
}

func TestImportWarrinerVAD_MissingColumns(t *testing.T) {
	data := "Word,SomeOther\nhappy,7.47\n"
	p := writeTempFile(t, "warriner.csv", data)

	_, _, err := ImportWarrinerVAD(p)
	if err == nil {
		t.Error("expected error for missing columns")
	}
}

// ── ImportSentiWordNet ───────────────────────────────────────────────────────

func TestImportSentiWordNet_Basic(t *testing.T) {
	data := "# comment line\na\t00001740\t0.875\t0.000\tgood#1 nice#1\tgloss\nn\t00002000\t0.000\t0.750\tbad#1 evil#1\tanother gloss\n"
	p := writeTempFile(t, "swn.txt", data)

	entries, res, err := ImportSentiWordNet(p)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total < 2 {
		t.Errorf("total=%d, want >=2", res.Total)
	}
	// Check that we got positive and negative entries
	hasPos, hasNeg := false, false
	for _, e := range entries {
		if e.Polarity == "POSITIVE" {
			hasPos = true
		}
		if e.Polarity == "NEGATIVE" {
			hasNeg = true
		}
	}
	if !hasPos {
		t.Error("expected POSITIVE entries")
	}
	if !hasNeg {
		t.Error("expected NEGATIVE entries")
	}
}

func TestImportSentiWordNet_SkipsAmbiguous(t *testing.T) {
	// Net score = 0.1 - 0.1 = 0.0, below threshold of 0.25
	data := "a\t00001740\t0.100\t0.100\tambiguous#1\tgloss\n"
	p := writeTempFile(t, "swn.txt", data)

	entries, _, err := ImportSentiWordNet(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Errorf("entries=%d, want 0 (ambiguous should be skipped)", len(entries))
	}
}

func TestImportSentiWordNet_DeduplicatesWords(t *testing.T) {
	data := "a\t001\t0.800\t0.000\tgood#1\tgloss1\na\t002\t0.900\t0.000\tgood#2\tgloss2\n"
	p := writeTempFile(t, "swn.txt", data)

	entries, _, err := ImportSentiWordNet(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("entries=%d, want 1 (dedup same word)", len(entries))
	}
}

func TestImportSentiWordNet_POS(t *testing.T) {
	data := "n\t001\t0.800\t0.000\thappiness#1\tgloss\nv\t002\t0.000\t0.800\tkill#1\tgloss\na\t003\t0.800\t0.000\tbeautiful#1\tgloss\nr\t004\t0.000\t0.800\thorribly#1\tgloss\n"
	p := writeTempFile(t, "swn.txt", data)

	entries, _, err := ImportSentiWordNet(p)
	if err != nil {
		t.Fatal(err)
	}
	posMap := make(map[string]string)
	for _, e := range entries {
		posMap[e.Word] = e.POS
	}
	if posMap["happiness"] != "NOUN" {
		t.Errorf("happiness POS=%q", posMap["happiness"])
	}
	if posMap["kill"] != "VERB" {
		t.Errorf("kill POS=%q", posMap["kill"])
	}
	if posMap["beautiful"] != "ADJ" {
		t.Errorf("beautiful POS=%q", posMap["beautiful"])
	}
	if posMap["horribly"] != "ADV" {
		t.Errorf("horribly POS=%q", posMap["horribly"])
	}
}

// ── ImportIPADict ────────────────────────────────────────────────────────────

func TestImportIPADict_Basic(t *testing.T) {
	data := "hello\t/hɛˈloʊ/\nworld\t/wɜːld/\n"
	p := writeTempFile(t, "ipa.txt", data)

	m, err := ImportIPADict(p, "EN")
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 2 {
		t.Errorf("len=%d, want 2", len(m))
	}
	if m["hello"] != "/hɛˈloʊ/" {
		t.Errorf("hello=%q", m["hello"])
	}
}

func TestImportIPADict_MultiPron(t *testing.T) {
	data := "either\t/ˈiːðər/, /ˈaɪðər/\n"
	p := writeTempFile(t, "ipa.txt", data)

	m, err := ImportIPADict(p, "EN")
	if err != nil {
		t.Fatal(err)
	}
	// Should take first pronunciation only
	if m["either"] != "/ˈiːðər/" {
		t.Errorf("either=%q", m["either"])
	}
}

func TestImportIPADict_Empty(t *testing.T) {
	p := writeTempFile(t, "ipa.txt", "")
	m, err := ImportIPADict(p, "EN")
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 0 {
		t.Errorf("len=%d", len(m))
	}
}

// ── Import81LangSentiment ────────────────────────────────────────────────────

func TestImport81Lang_Basic(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "positive_words_pt.txt"), []byte("feliz\nalegre\n"), 0644)
	os.WriteFile(filepath.Join(dir, "negative_words_pt.txt"), []byte("triste\n"), 0644)

	langs := map[string]bool{"PT": true}
	entries, res, err := Import81LangSentiment(dir, langs)
	if err != nil {
		t.Fatal(err)
	}
	if res.Total != 3 {
		t.Errorf("total=%d, want 3", res.Total)
	}
	if res.ByPolarity["POSITIVE"] != 2 {
		t.Errorf("positive=%d", res.ByPolarity["POSITIVE"])
	}
	if res.ByPolarity["NEGATIVE"] != 1 {
		t.Errorf("negative=%d", res.ByPolarity["NEGATIVE"])
	}
	for _, e := range entries {
		if e.Lang != "PT" {
			t.Errorf("lang=%q, want PT", e.Lang)
		}
	}
}

func TestImport81Lang_MissingFiles(t *testing.T) {
	dir := t.TempDir()
	langs := map[string]bool{"XX": true}
	entries, _, err := Import81LangSentiment(dir, langs)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Errorf("entries=%d, want 0", len(entries))
	}
}

// ── ImportNRCEmoLex ──────────────────────────────────────────────────────────

func TestImportNRCEmoLex_Basic(t *testing.T) {
	header := "English Word\tanger\tanticipation\tdisgust\tfear\tjoy\tnegative\tpositive\tsadness\tsurprise\ttrust\tPortuguese\tSpanish\n"
	row1 := "happy\t0\t1\t0\t0\t1\t0\t1\t0\t0\t1\tfeliz\tfeliz\n"
	row2 := "sad\t0\t0\t0\t0\t0\t1\t0\t1\t0\t0\ttriste\ttriste\n"
	data := header + row1 + row2
	p := writeTempFile(t, "emolex.txt", data)

	langs := map[string]bool{"EN": true, "PT": true, "ES": true}
	entries, res, err := ImportNRCEmoLex(p, langs)
	if err != nil {
		t.Fatal(err)
	}
	// Each word → EN + PT + ES = 3 entries per word, 2 words = 6
	// But "feliz" PT appears twice (from happy and potentially from sad) so it depends
	if res.Total < 4 {
		t.Errorf("total=%d, want >=4", res.Total)
	}
	// Check that we have EN and PT entries
	hasEN, hasPT := false, false
	for _, e := range entries {
		if e.Lang == "EN" {
			hasEN = true
		}
		if e.Lang == "PT" {
			hasPT = true
		}
	}
	if !hasEN {
		t.Error("expected EN entries")
	}
	if !hasPT {
		t.Error("expected PT entries")
	}
}

func TestImportNRCEmoLex_EmptyFile(t *testing.T) {
	p := writeTempFile(t, "emolex.txt", "")
	_, _, err := ImportNRCEmoLex(p, nil)
	if err == nil {
		t.Error("expected error for empty file")
	}
}

// ── Merge ────────────────────────────────────────────────────────────────────

func TestMerge_DeduplicatesByNormLang(t *testing.T) {
	entries := []Entry{
		{Word: "happy", Lang: "EN", Norm: "happy", Polarity: "POSITIVE"},
		{Word: "HAPPY", Lang: "EN", Norm: "happy", Polarity: "POSITIVE"},
	}
	merged := Merge(entries)
	if len(merged) != 1 {
		t.Errorf("len=%d, want 1", len(merged))
	}
}

func TestMerge_MergesFields(t *testing.T) {
	entries := []Entry{
		{Word: "happy", Lang: "EN", Norm: "happy", Polarity: "NEUTRAL", Arousal: "HIGH"},
		{Word: "happy", Lang: "EN", Norm: "happy", Polarity: "POSITIVE", Dominance: "MED"},
	}
	merged := Merge(entries)
	if len(merged) != 1 {
		t.Fatalf("len=%d", len(merged))
	}
	e := merged[0]
	// Should prefer POSITIVE over NEUTRAL
	if e.Polarity != "POSITIVE" {
		t.Errorf("polarity=%q, want POSITIVE", e.Polarity)
	}
	// Should merge arousal from first
	if e.Arousal != "HIGH" {
		t.Errorf("arousal=%q, want HIGH", e.Arousal)
	}
	// Should merge dominance from second
	if e.Dominance != "MED" {
		t.Errorf("dominance=%q, want MED", e.Dominance)
	}
}

func TestMerge_DifferentLangs(t *testing.T) {
	entries := []Entry{
		{Word: "happy", Lang: "EN", Norm: "happy", Polarity: "POSITIVE"},
		{Word: "happy", Lang: "FR", Norm: "happy", Polarity: "POSITIVE"},
	}
	merged := Merge(entries)
	if len(merged) != 2 {
		t.Errorf("len=%d, want 2 (different langs)", len(merged))
	}
}

func TestMerge_MergesIPA(t *testing.T) {
	entries := []Entry{
		{Word: "hello", Lang: "EN", Norm: "hello", Polarity: "NEUTRAL"},
		{Word: "hello", Lang: "EN", Norm: "hello", Polarity: "NEUTRAL", IPA: "/hɛˈloʊ/"},
	}
	merged := Merge(entries)
	if len(merged) != 1 {
		t.Fatalf("len=%d", len(merged))
	}
	if merged[0].IPA != "/hɛˈloʊ/" {
		t.Errorf("IPA=%q", merged[0].IPA)
	}
}

// ── ExcludeExisting ──────────────────────────────────────────────────────────

func TestExcludeExisting_FiltersKnownWords(t *testing.T) {
	csvData := "word_id,root_id,variant,word,lang,norm\n4097,1,1,happy,EN,happy\n"
	csvPath := writeTempFile(t, "words.csv", csvData)

	entries := []Entry{
		{Word: "happy", Lang: "EN", Norm: "happy"},
		{Word: "sad", Lang: "EN", Norm: "sad"},
	}

	filtered, skipped, err := ExcludeExisting(entries, csvPath)
	if err != nil {
		t.Fatal(err)
	}
	if skipped != 1 {
		t.Errorf("skipped=%d, want 1", skipped)
	}
	if len(filtered) != 1 {
		t.Fatalf("filtered=%d, want 1", len(filtered))
	}
	if filtered[0].Word != "sad" {
		t.Errorf("remaining=%q, want sad", filtered[0].Word)
	}
}

func TestExcludeExisting_MissingFile(t *testing.T) {
	entries := []Entry{{Word: "happy", Lang: "EN", Norm: "happy"}}
	filtered, skipped, err := ExcludeExisting(entries, "/nonexistent")
	if err != nil {
		t.Fatal(err)
	}
	if skipped != 0 {
		t.Errorf("skipped=%d", skipped)
	}
	if len(filtered) != 1 {
		t.Errorf("filtered=%d", len(filtered))
	}
}

// ── WriteCSV ─────────────────────────────────────────────────────────────────

func TestWriteCSV_CreatesNewFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "out.csv")
	entries := []Entry{
		{Word: "happy", Lang: "EN", Norm: "happy", Polarity: "POSITIVE", Intensity: "STRONG"},
	}

	err := WriteCSV(p, entries, 10000, 999)
	if err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if !strings.Contains(content, "word_id") {
		t.Error("expected header")
	}
	if !strings.Contains(content, "happy") {
		t.Error("expected happy in output")
	}
	if !strings.Contains(content, "POSITIVE") {
		t.Error("expected POSITIVE in output")
	}
	if !strings.Contains(content, "10000") {
		t.Error("expected word_id 10000")
	}
}

func TestWriteCSV_AppendsToExisting(t *testing.T) {
	p := filepath.Join(t.TempDir(), "out.csv")
	e1 := []Entry{{Word: "happy", Lang: "EN", Norm: "happy", Polarity: "POSITIVE", Intensity: "STRONG"}}
	e2 := []Entry{{Word: "sad", Lang: "EN", Norm: "sad", Polarity: "NEGATIVE", Intensity: "MODERATE"}}

	WriteCSV(p, e1, 10000, 999)
	WriteCSV(p, e2, 10001, 999)

	data, _ := os.ReadFile(p)
	content := string(data)
	if !strings.Contains(content, "happy") || !strings.Contains(content, "sad") {
		t.Error("expected both words")
	}
	// Header should only appear once
	if strings.Count(content, "word_id") != 1 {
		t.Error("header should appear exactly once")
	}
}

// ── Property tests ───────────────────────────────────────────────────────────

func TestProperty_AllEntriesHaveNorm(t *testing.T) {
	data := "happy\t3\nsad\t-2\nangry\t-4\njoyful\t4\n"
	p := writeTempFile(t, "afinn.txt", data)

	entries, _, _ := ImportAFINN(p)
	for _, e := range entries {
		if e.Norm == "" {
			t.Errorf("entry %q has empty norm", e.Word)
		}
		if e.Norm != strings.ToLower(e.Norm) {
			t.Errorf("norm %q not lowercase", e.Norm)
		}
	}
}

func TestProperty_AllEntriesHaveLang(t *testing.T) {
	data := "happy\t3\n"
	p := writeTempFile(t, "afinn.txt", data)

	entries, _, _ := ImportAFINN(p)
	for _, e := range entries {
		if e.Lang == "" {
			t.Errorf("entry %q has empty lang", e.Word)
		}
	}
}

func TestProperty_AllEntriesHaveSource(t *testing.T) {
	data := "happy\t3\n"
	p := writeTempFile(t, "afinn.txt", data)
	entries, _, _ := ImportAFINN(p)
	for _, e := range entries {
		if e.Source == "" {
			t.Errorf("entry %q has empty source", e.Word)
		}
	}
}

func TestProperty_PolarityIsValid(t *testing.T) {
	valid := map[string]bool{
		"POSITIVE": true, "NEGATIVE": true, "NEUTRAL": true, "AMBIGUOUS": true,
	}
	data := "happy\t3\nsad\t-2\nneutral\t0\n"
	p := writeTempFile(t, "afinn.txt", data)

	entries, _, _ := ImportAFINN(p)
	for _, e := range entries {
		if !valid[e.Polarity] {
			t.Errorf("invalid polarity %q for %q", e.Polarity, e.Word)
		}
	}
}

func TestProperty_IntensityIsValid(t *testing.T) {
	valid := map[string]bool{
		"NONE": true, "WEAK": true, "MODERATE": true, "STRONG": true, "EXTREME": true,
	}
	data := "happy\t3\nsad\t-2\nneutral\t0\nextreme\t5\n"
	p := writeTempFile(t, "afinn.txt", data)

	entries, _, _ := ImportAFINN(p)
	for _, e := range entries {
		if !valid[e.Intensity] {
			t.Errorf("invalid intensity %q for %q", e.Intensity, e.Word)
		}
	}
}
