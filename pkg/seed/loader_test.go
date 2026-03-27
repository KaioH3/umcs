package seed_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/kak/umcs/pkg/seed"
)

func writeCSV(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

const validRootsCSV = `root_id,root_str,origin,meaning_en,notes,parent_root_id
1,negat,LATIN,to deny,,
2,bon,LATIN,good,,`

const validWordsCSV = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0
8193,2,1,good,EN,good,POSITIVE,STRONG,EVALUATION,GENERAL,50,0`

func TestLoadRootsBasic(t *testing.T) {
	path := writeCSV(t, "roots.csv", validRootsCSV)
	roots, err := seed.LoadRoots(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 2 {
		t.Fatalf("want 2 roots, got %d", len(roots))
	}
	if roots[0].RootID != 1 {
		t.Fatalf("want root_id=1, got %d", roots[0].RootID)
	}
	if roots[0].RootStr != "negat" {
		t.Fatalf("want RootStr=negat, got %q", roots[0].RootStr)
	}
	if roots[0].Origin != "LATIN" {
		t.Fatalf("want Origin=LATIN, got %q", roots[0].Origin)
	}
	if roots[1].RootID != 2 {
		t.Fatalf("want root_id=2, got %d", roots[1].RootID)
	}
}

func TestLoadWordsBasic(t *testing.T) {
	path := writeCSV(t, "words.csv", validWordsCSV)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(words) != 2 {
		t.Fatalf("want 2 words, got %d", len(words))
	}
	// word_id must equal (root_id<<12)|variant
	if words[0].WordID != (1<<12)|1 {
		t.Fatalf("want word_id=%d, got %d", (1<<12)|1, words[0].WordID)
	}
	if words[0].Word != "negative" {
		t.Fatalf("want Word=negative, got %q", words[0].Word)
	}
	if words[0].Lang != "EN" {
		t.Fatalf("want Lang=EN, got %q", words[0].Lang)
	}
	if words[0].Sentiment == 0 {
		t.Fatal("sentiment should be non-zero for annotated word")
	}
}

func TestLoadRootsEmpty(t *testing.T) {
	path := writeCSV(t, "roots.csv", "")
	_, err := seed.LoadRoots(path)
	if err == nil {
		t.Fatal("empty file should return error")
	}
}

func TestLoadWordsHeaderOnly(t *testing.T) {
	// CSV with only header, no data rows → 0 words, no error
	csv := "word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags\n"
	path := writeCSV(t, "words.csv", csv)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(words) != 0 {
		t.Fatalf("want 0 words from header-only CSV, got %d", len(words))
	}
}

func TestLoadRootsHeaderOnly(t *testing.T) {
	csv := "root_id,root_str,origin,meaning_en,notes,parent_root_id\n"
	path := writeCSV(t, "roots.csv", csv)
	roots, err := seed.LoadRoots(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 0 {
		t.Fatalf("want 0 roots from header-only CSV, got %d", len(roots))
	}
}

func TestLoadRootsNonexistent(t *testing.T) {
	_, err := seed.LoadRoots("/tmp/doesnotexist_lexsent_roots_test.csv")
	if err == nil {
		t.Fatal("nonexistent file should return error")
	}
}

func TestLoadWordsNonexistent(t *testing.T) {
	_, err := seed.LoadWords("/tmp/doesnotexist_lexsent_words_test.csv")
	if err == nil {
		t.Fatal("nonexistent file should return error")
	}
}

func TestLoadWordsWrongWordID(t *testing.T) {
	// word_id=999 but root_id=1, variant=1 → expected (1<<12)|1=4097
	csv := `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
999,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0`
	path := writeCSV(t, "words.csv", csv)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("word_id inconsistent with root_id/variant should return error")
	}
}

func TestLoadWordsInvalidPolarity(t *testing.T) {
	csv := `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,negative,EN,negative,INVALID_POLARITY,MODERATE,EVALUATION,GENERAL,1200,0`
	path := writeCSV(t, "words.csv", csv)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid polarity should return error from sentiment.Pack")
	}
}

func TestLoadRootsParentLink(t *testing.T) {
	csv := `root_id,root_str,origin,meaning_en,notes,parent_root_id
1,negat,LATIN,deny,,
2,ne,PIE,negation,,1`
	path := writeCSV(t, "roots.csv", csv)
	roots, err := seed.LoadRoots(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 2 {
		t.Fatalf("want 2 roots, got %d", len(roots))
	}
	if roots[1].ParentRootID != 1 {
		t.Fatalf("want ParentRootID=1, got %d", roots[1].ParentRootID)
	}
}

func TestLoadWordsNegationMarker(t *testing.T) {
	// Negation markers have NEGATION_MARKER as semantic role and scope flag
	csv := `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,not,EN,not,NEUTRAL,NONE,NEGATION_MARKER,GENERAL,10,0`
	path := writeCSV(t, "words.csv", csv)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(words) != 1 {
		t.Fatalf("want 1 word, got %d", len(words))
	}
	if words[0].Sentiment == 0 {
		t.Fatal("negation marker should have non-zero sentiment (flag bits set)")
	}
}

func TestLoadWordsIntensifier(t *testing.T) {
	csv := `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,very,EN,very,NEUTRAL,NONE,INTENSIFIER,GENERAL,10,0`
	path := writeCSV(t, "words.csv", csv)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatal(err)
	}
	if words[0].Sentiment == 0 {
		t.Fatal("intensifier should have non-zero sentiment (flag bits set)")
	}
}

func TestLoadRootsWithNotes(t *testing.T) {
	// Notes field with commas requires proper CSV quoting
	csv := `root_id,root_str,origin,meaning_en,notes,parent_root_id
1,negat,LATIN,deny,"from negare, related to ne",`
	path := writeCSV(t, "roots.csv", csv)
	roots, err := seed.LoadRoots(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 1 {
		t.Fatalf("want 1 root, got %d", len(roots))
	}
}

func TestLoadWordsInvalidFreqRank(t *testing.T) {
	csv := `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,not_a_number,0`
	path := writeCSV(t, "words.csv", csv)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid freq_rank should return error")
	}
}

func TestLoadWordsInvalidFlags(t *testing.T) {
	csv := `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,100,bad_flag`
	path := writeCSV(t, "words.csv", csv)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid flags should return error")
	}
}

func TestLoadWordsInvalidRootID(t *testing.T) {
	csv := `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,not_a_number,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,100,0`
	path := writeCSV(t, "words.csv", csv)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid root_id should return error")
	}
}

// ── parseRegister (via LoadWords with register column) ─────────────────────────

func TestLoadWordsRegister_Formal(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,register
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0,FORMAL`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with register=FORMAL: %v", err)
	}
	if len(words) != 1 {
		t.Fatalf("want 1 word, got %d", len(words))
	}
	// register=FORMAL packs to 1<<8 in Flags
	if words[0].Flags&(0xF<<8) != 1<<8 {
		t.Errorf("FORMAL register not set: Flags=0x%08X, want bit [11:8]=1", words[0].Flags)
	}
}

func TestLoadWordsRegister_Slang(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,register
4097,1,1,yo,EN,yo,NEUTRAL,NONE,EVALUATION,GENERAL,5000,0,SLANG`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with register=SLANG: %v", err)
	}
	if words[0].Flags&(0xF<<8) != 3<<8 {
		t.Errorf("SLANG register: Flags bits [11:8] = %d, want 3", (words[0].Flags>>8)&0xF)
	}
}

func TestLoadWordsRegister_Invalid(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,register
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0,NONSENSE_REGISTER`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid register value should return error")
	}
}

func TestLoadWordsRegister_Empty_IsNeutral(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,register
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0,`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("empty register: %v", err)
	}
	if words[0].Flags&(0xF<<8) != 0 {
		t.Errorf("empty register should be 0, got %d", (words[0].Flags>>8)&0xF)
	}
}

// ── parseOntological (via LoadWords with ontological column) ───────────────────

func TestLoadWordsOntological_Person(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,ontological
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0,PERSON`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with ontological=PERSON: %v", err)
	}
	if words[0].Flags&(0xF<<12) != 1<<12 {
		t.Errorf("PERSON ontological: bits [15:12] = %d, want 1", (words[0].Flags>>12)&0xF)
	}
}

func TestLoadWordsOntological_Abstract(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,ontological
4097,1,1,freedom,EN,freedom,POSITIVE,MODERATE,EVALUATION,GENERAL,500,0,ABSTRACT`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with ontological=ABSTRACT: %v", err)
	}
	if words[0].Flags&(0xF<<12) != 13<<12 {
		t.Errorf("ABSTRACT ontological: bits [15:12] = %d, want 13", (words[0].Flags>>12)&0xF)
	}
}

func TestLoadWordsOntological_Invalid(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,ontological
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0,ROBOT_ONTOLOGY`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid ontological value should return error")
	}
}

// ── parseStress (via LoadWords with stress column) ────────────────────────────

func TestLoadWordsStress_Final(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,syllables,stress
4097,1,1,liberation,EN,liberation,NEUTRAL,NONE,EVALUATION,GENERAL,2000,0,4,FINAL`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with stress=FINAL: %v", err)
	}
	_ = words // stress bits checked via phon package; existence of no error is the key test
}

func TestLoadWordsStress_Penultimate(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,syllables,stress
4097,1,1,terrible,EN,terrible,NEGATIVE,MODERATE,EVALUATION,GENERAL,3000,0,3,PENULTIMATE`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with stress=PENULTIMATE: %v", err)
	}
}

func TestLoadWordsStress_Oxytone(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,syllables,stress
4097,1,1,liberdade,PT,liberdade,POSITIVE,MODERATE,EVALUATION,GENERAL,1000,0,4,OXYTONE`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with stress=OXYTONE (alias for FINAL): %v", err)
	}
}

func TestLoadWordsStress_Invalid(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,stress
4097,1,1,terrible,EN,terrible,NEGATIVE,MODERATE,EVALUATION,GENERAL,3000,0,WEIRD_STRESS`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid stress value should return error")
	}
}

// ── parseValency (via LoadWords with valency column) ─────────────────────────

func TestLoadWordsValency_Intransitive(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,valency
4097,1,1,sleep,EN,sleep,NEUTRAL,NONE,EVALUATION,GENERAL,500,0,INTRANS`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with valency=INTRANS: %v", err)
	}
}

func TestLoadWordsValency_Transitive(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,valency
4097,1,1,love,EN,love,POSITIVE,STRONG,EVALUATION,GENERAL,300,0,TRANS`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with valency=TRANS: %v", err)
	}
}

func TestLoadWordsValency_Modal(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,valency
4097,1,1,can,EN,can,NEUTRAL,NONE,EVALUATION,GENERAL,5,0,MODAL`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("LoadWords with valency=MODAL: %v", err)
	}
}

func TestLoadWordsValency_Invalid(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,valency
4097,1,1,love,EN,love,POSITIVE,STRONG,EVALUATION,GENERAL,300,0,HEXAVALENT`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid valency should return error")
	}
}

func TestLoadWordsValency_NA_IsOK(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,valency
4097,1,1,love,EN,love,POSITIVE,STRONG,EVALUATION,GENERAL,300,0,N/A`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("valency=N/A should be valid: %v", err)
	}
}

// ── Additional LoadRoots tests ──────────────────────────────────────────────

func TestLoadRoots_ExtraColumns(t *testing.T) {
	// CSV with extra columns beyond known ones → should be ignored gracefully
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id,extra_col
1,negat,LATIN,deny,,, some_extra_value
2,bon,LATIN,good,,, another_value`
	path := writeCSV(t, "roots.csv", csvData)
	roots, err := seed.LoadRoots(path)
	if err != nil {
		t.Fatalf("extra columns should not cause error: %v", err)
	}
	if len(roots) != 2 {
		t.Fatalf("want 2 roots, got %d", len(roots))
	}
}

func TestLoadRoots_InvalidRootID(t *testing.T) {
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id
not_a_number,negat,LATIN,deny,,`
	path := writeCSV(t, "roots.csv", csvData)
	_, err := seed.LoadRoots(path)
	if err == nil {
		t.Fatal("invalid root_id should return error")
	}
}

func TestLoadRoots_InvalidParentRootID(t *testing.T) {
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id
1,negat,LATIN,deny,,not_a_number`
	path := writeCSV(t, "roots.csv", csvData)
	_, err := seed.LoadRoots(path)
	if err == nil {
		t.Fatal("invalid parent_root_id should return error")
	}
}

func TestLoadRoots_HypernymAntonymSynonymLinks(t *testing.T) {
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id,hypernym_root_id,antonym_root_id,synonym_root_id
1,negat,LATIN,deny,,,10,20,30
2,bon,LATIN,good,,,11,,`
	path := writeCSV(t, "roots.csv", csvData)
	roots, err := seed.LoadRoots(path)
	if err != nil {
		t.Fatalf("roots with hypernym/antonym/synonym: %v", err)
	}
	if roots[0].HypernymRootID != 10 {
		t.Errorf("want HypernymRootID=10, got %d", roots[0].HypernymRootID)
	}
	if roots[0].AntonymRootID != 20 {
		t.Errorf("want AntonymRootID=20, got %d", roots[0].AntonymRootID)
	}
	if roots[0].SynonymRootID != 30 {
		t.Errorf("want SynonymRootID=30, got %d", roots[0].SynonymRootID)
	}
	// Root 2 has hypernym but no antonym/synonym
	if roots[1].HypernymRootID != 11 {
		t.Errorf("want HypernymRootID=11, got %d", roots[1].HypernymRootID)
	}
	if roots[1].AntonymRootID != 0 {
		t.Errorf("want AntonymRootID=0 (empty), got %d", roots[1].AntonymRootID)
	}
}

func TestLoadRoots_InvalidHypernymRootID(t *testing.T) {
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id,hypernym_root_id
1,negat,LATIN,deny,,,xyz`
	path := writeCSV(t, "roots.csv", csvData)
	_, err := seed.LoadRoots(path)
	if err == nil {
		t.Fatal("invalid hypernym_root_id should return error")
	}
}

func TestLoadRoots_InvalidAntonymRootID(t *testing.T) {
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id,hypernym_root_id,antonym_root_id
1,negat,LATIN,deny,,,,abc`
	path := writeCSV(t, "roots.csv", csvData)
	_, err := seed.LoadRoots(path)
	if err == nil {
		t.Fatal("invalid antonym_root_id should return error")
	}
}

func TestLoadRoots_InvalidSynonymRootID(t *testing.T) {
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id,hypernym_root_id,antonym_root_id,synonym_root_id
1,negat,LATIN,deny,,,,,bad`
	path := writeCSV(t, "roots.csv", csvData)
	_, err := seed.LoadRoots(path)
	if err == nil {
		t.Fatal("invalid synonym_root_id should return error")
	}
}

func TestLoadRoots_WhitespaceInFields(t *testing.T) {
	const csvData = `root_id, root_str, origin, meaning_en, notes, parent_root_id
1, negat , LATIN , to deny or negate , some notes ,`
	path := writeCSV(t, "roots.csv", csvData)
	roots, err := seed.LoadRoots(path)
	if err != nil {
		t.Fatalf("whitespace in fields: %v", err)
	}
	if roots[0].RootStr != "negat" {
		t.Errorf("want trimmed RootStr='negat', got %q", roots[0].RootStr)
	}
	if roots[0].Origin != "LATIN" {
		t.Errorf("want trimmed Origin='LATIN', got %q", roots[0].Origin)
	}
}

func TestLoadRoots_MalformedCSV(t *testing.T) {
	// A line with wrong number of fields (fewer than header, not using FieldsPerRecord=-1)
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id
1,negat,LATIN`
	path := writeCSV(t, "roots.csv", csvData)
	_, err := seed.LoadRoots(path)
	if err == nil {
		t.Fatal("malformed CSV row (fewer fields than header) should return error")
	}
}

func TestLoadRoots_SingleRow(t *testing.T) {
	const csvData = `root_id,root_str,origin,meaning_en,notes,parent_root_id
1,sol,LATIN,alone,,`
	path := writeCSV(t, "roots.csv", csvData)
	roots, err := seed.LoadRoots(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 1 {
		t.Fatalf("want 1 root, got %d", len(roots))
	}
	if roots[0].MeaningEN != "alone" {
		t.Errorf("want MeaningEN='alone', got %q", roots[0].MeaningEN)
	}
}

// ── Additional LoadWords tests ──────────────────────────────────────────────

func TestLoadWords_EmptyFile(t *testing.T) {
	path := writeCSV(t, "words.csv", "")
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("empty words file should return error (no header)")
	}
}

func TestLoadWords_InvalidWordID(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
notanumber,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid word_id should return error")
	}
}

func TestLoadWords_InvalidVariant(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,xyz,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid variant should return error")
	}
}

func TestLoadWords_UnicodeWord(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,négation,FR,negation,NEGATIVE,MODERATE,EVALUATION,GENERAL,800,0`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("unicode word: %v", err)
	}
	if words[0].Word != "négation" {
		t.Errorf("want Word='négation', got %q", words[0].Word)
	}
	if words[0].Norm != "negation" {
		t.Errorf("want Norm='negation', got %q", words[0].Norm)
	}
}

func TestLoadWords_ExtraColumns(t *testing.T) {
	// Extra columns not known by the loader → ignored
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,unknown_col
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0,ignored`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("extra columns: %v", err)
	}
	if len(words) != 1 {
		t.Fatalf("want 1 word, got %d", len(words))
	}
}

func TestLoadWords_MissingOptionalColumns(t *testing.T) {
	// CSV without freq_rank or flags columns → defaults to 0
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("missing optional columns: %v", err)
	}
	if words[0].FreqRank != 0 {
		t.Errorf("want FreqRank=0 (missing column), got %d", words[0].FreqRank)
	}
	if words[0].Flags == 0 {
		// Flags may be non-zero due to phonological inference, so just check no error
	}
}

func TestLoadWords_DowntonerRole(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,slightly,EN,slightly,NEUTRAL,NONE,DOWNTONER,GENERAL,500,0`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("DOWNTONER role: %v", err)
	}
	if words[0].Sentiment == 0 {
		t.Fatal("DOWNTONER should produce non-zero sentiment (scope flag)")
	}
}

func TestLoadWords_AffirmationMarkerRole(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,indeed,EN,indeed,NEUTRAL,NONE,AFFIRMATION_MARKER,GENERAL,200,0`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("AFFIRMATION_MARKER role: %v", err)
	}
	if words[0].Sentiment == 0 {
		t.Fatal("AFFIRMATION_MARKER should produce non-zero sentiment")
	}
}

func TestLoadWords_PolysemyClamp(t *testing.T) {
	// Polysemy > 15 should be clamped to 15
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,polysemy
4097,1,1,set,EN,set,NEUTRAL,NONE,EVALUATION,GENERAL,100,0,99`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("polysemy clamp: %v", err)
	}
	poly := (words[0].Flags >> 16) & 0xF
	if poly != 15 {
		t.Errorf("polysemy = %d, want 15 (clamped from 99)", poly)
	}
}

func TestLoadWords_PolysemyInvalid(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,polysemy
4097,1,1,set,EN,set,NEUTRAL,NONE,EVALUATION,GENERAL,100,0,not_a_number`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid polysemy should return error")
	}
}

func TestLoadWords_CulturalSpecific_True(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,cultural_specific
4097,1,1,saudade,PT,saudade,NEGATIVE,MODERATE,EVALUATION,GENERAL,100,0,1`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("cultural_specific=1: %v", err)
	}
	if words[0].Flags&(1<<20) == 0 {
		t.Error("cultural_specific=1 should set bit 20 in Flags")
	}
}

func TestLoadWords_CulturalSpecific_TRUE_Upper(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,cultural_specific
4097,1,1,saudade,PT,saudade,NEGATIVE,MODERATE,EVALUATION,GENERAL,100,0,TRUE`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("cultural_specific=TRUE: %v", err)
	}
	if words[0].Flags&(1<<20) == 0 {
		t.Error("cultural_specific=TRUE should set bit 20 in Flags")
	}
}

func TestLoadWords_IronyCapable(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,irony_capable
4097,1,1,brilliant,EN,brilliant,POSITIVE,STRONG,EVALUATION,GENERAL,1500,0,1`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("irony_capable=1: %v", err)
	}
	if words[0].Flags&(1<<22) == 0 {
		t.Error("irony_capable=1 should set bit 22 in Flags")
	}
}

func TestLoadWords_IronyCapable_TRUE(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,irony_capable
4097,1,1,brilliant,EN,brilliant,POSITIVE,STRONG,EVALUATION,GENERAL,1500,0,TRUE`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("irony_capable=TRUE: %v", err)
	}
	if words[0].Flags&(1<<22) == 0 {
		t.Error("irony_capable=TRUE should set bit 22 in Flags")
	}
}

func TestLoadWords_Neologism(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,neologism
4097,1,1,hashtag,EN,hashtag,NEUTRAL,NONE,EVALUATION,GENERAL,900,0,1`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("neologism=1: %v", err)
	}
	if words[0].Flags&(1<<21) == 0 {
		t.Error("neologism=1 should set bit 21 in Flags")
	}
}

func TestLoadWords_Neologism_TRUE(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,neologism
4097,1,1,hashtag,EN,hashtag,NEUTRAL,NONE,EVALUATION,GENERAL,900,0,TRUE`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("neologism=TRUE: %v", err)
	}
	if words[0].Flags&(1<<21) == 0 {
		t.Error("neologism=TRUE should set bit 21 in Flags")
	}
}

func TestLoadWords_SyllablesInvalid(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,syllables
4097,1,1,terrible,EN,terrible,NEGATIVE,MODERATE,EVALUATION,GENERAL,3000,0,abc`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err == nil {
		t.Fatal("invalid syllables should return error")
	}
}

func TestLoadWords_PronField(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,pron
4097,1,1,terrible,EN,terrible,NEGATIVE,MODERATE,EVALUATION,GENERAL,3000,0,/ˈtɛr.ɪ.bəl/`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("pron field: %v", err)
	}
	if words[0].Pron != "/ˈtɛr.ɪ.bəl/" {
		t.Errorf("want Pron='/ˈtɛr.ɪ.bəl/', got %q", words[0].Pron)
	}
}

func TestLoadWords_PronFieldEmpty(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,pron
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0,`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("empty pron: %v", err)
	}
	if words[0].Pron != "" {
		t.Errorf("empty pron column should yield empty Pron, got %q", words[0].Pron)
	}
}

func TestLoadWords_AllExtendedSentimentFields(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,pos,arousal,dominance,aoa,concreteness
4097,1,1,terrible,EN,terrible,NEGATIVE,STRONG,EVALUATION,GENERAL,3000,0,ADJ,HIGH,LOW,EARLY,TRUE`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("all extended sentiment fields: %v", err)
	}
	if words[0].Sentiment == 0 {
		t.Error("full annotation should produce non-zero sentiment")
	}
}

func TestLoadWords_MultipleRows(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0
4098,1,2,negativo,PT,negativo,NEGATIVE,MODERATE,EVALUATION,GENERAL,800,0
8193,2,1,good,EN,good,POSITIVE,STRONG,EVALUATION,GENERAL,50,0`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("multiple rows: %v", err)
	}
	if len(words) != 3 {
		t.Fatalf("want 3 words, got %d", len(words))
	}
	if words[1].Lang != "PT" {
		t.Errorf("second word Lang: want PT, got %q", words[1].Lang)
	}
	if words[2].Word != "good" {
		t.Errorf("third word: want 'good', got %q", words[2].Word)
	}
}

// ── parseRegister additional coverage ──────────────────────────────────────

func TestLoadWordsRegister_AllValidValues(t *testing.T) {
	registers := []struct {
		name string
		want uint32
	}{
		{"NEUTRAL", 0},
		{"INFORMAL", 2 << 8},
		{"VULGAR", 4 << 8},
		{"ARCHAIC", 5 << 8},
		{"POETIC", 6 << 8},
		{"TECHNICAL", 7 << 8},
		{"SCIENTIFIC", 8 << 8},
		{"CHILD", 9 << 8},
		{"REGIONAL", 10 << 8},
	}
	for _, tc := range registers {
		t.Run(tc.name, func(t *testing.T) {
			csvData := "word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,register\n" +
				"4097,1,1,test,EN,test,NEUTRAL,NONE,EVALUATION,GENERAL,100,0," + tc.name
			path := writeCSV(t, "words.csv", csvData)
			words, err := seed.LoadWords(path)
			if err != nil {
				t.Fatalf("register=%s: %v", tc.name, err)
			}
			got := words[0].Flags & (0xF << 8)
			if got != tc.want {
				t.Errorf("register=%s: flags bits [11:8] = 0x%X, want 0x%X", tc.name, got, tc.want)
			}
		})
	}
}

// ── parseOntological additional coverage ─────────────────────────────────

func TestLoadWordsOntological_AllValidValues(t *testing.T) {
	ontos := []struct {
		name string
		want uint32
	}{
		{"NONE", 0},
		{"PLACE", 2 << 12},
		{"ARTIFACT", 3 << 12},
		{"NATURAL", 4 << 12},
		{"EVENT", 5 << 12},
		{"STATE", 6 << 12},
		{"PROPERTY", 7 << 12},
		{"QUANTITY", 8 << 12},
		{"RELATION", 9 << 12},
		{"TEMPORAL", 10 << 12},
		{"BIOLOGICAL", 11 << 12},
		{"SOCIAL", 12 << 12},
	}
	for _, tc := range ontos {
		t.Run(tc.name, func(t *testing.T) {
			csvData := "word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,ontological\n" +
				"4097,1,1,test,EN,test,NEUTRAL,NONE,EVALUATION,GENERAL,100,0," + tc.name
			path := writeCSV(t, "words.csv", csvData)
			words, err := seed.LoadWords(path)
			if err != nil {
				t.Fatalf("ontological=%s: %v", tc.name, err)
			}
			got := words[0].Flags & (0xF << 12)
			if got != tc.want {
				t.Errorf("ontological=%s: flags bits [15:12] = 0x%X, want 0x%X", tc.name, got, tc.want)
			}
		})
	}
}

// ── parseStress additional coverage ─────────────────────────────────────

func TestLoadWordsStress_Antepenultimate(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,syllables,stress
4097,1,1,lampada,PT,lampada,NEUTRAL,NONE,EVALUATION,GENERAL,2000,0,3,ANTEPENULTIMATE`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("stress=ANTEPENULTIMATE: %v", err)
	}
}

func TestLoadWordsStress_Proparoxytone(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,syllables,stress
4097,1,1,silaba,ES,silaba,NEUTRAL,NONE,EVALUATION,GENERAL,2000,0,3,PROPAROXYTONE`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("stress=PROPAROXYTONE (alias for ANTEPENULTIMATE): %v", err)
	}
}

func TestLoadWordsStress_Paroxytone(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,syllables,stress
4097,1,1,casa,PT,casa,NEUTRAL,NONE,EVALUATION,GENERAL,100,0,2,PAROXYTONE`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("stress=PAROXYTONE (alias for PENULTIMATE): %v", err)
	}
}

func TestLoadWordsStress_Unknown(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,stress
4097,1,1,test,EN,test,NEUTRAL,NONE,EVALUATION,GENERAL,100,0,UNKNOWN`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("stress=UNKNOWN: %v", err)
	}
}

func TestLoadWordsStress_None(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,stress
4097,1,1,test,EN,test,NEUTRAL,NONE,EVALUATION,GENERAL,100,0,NONE`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("stress=NONE: %v", err)
	}
}

// ── parseValency additional coverage ────────────────────────────────────

func TestLoadWordsValency_Ditransitive(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,valency
4097,1,1,give,EN,give,NEUTRAL,NONE,EVALUATION,GENERAL,50,0,DITRANS`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("valency=DITRANS: %v", err)
	}
}

func TestLoadWordsValency_Copular(t *testing.T) {
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,valency
4097,1,1,be,EN,be,NEUTRAL,NONE,EVALUATION,GENERAL,1,0,COPULAR`
	path := writeCSV(t, "words.csv", csvData)
	_, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("valency=COPULAR: %v", err)
	}
}

func TestLoadWordsValency_LongForms(t *testing.T) {
	longForms := []string{"INTRANSITIVE", "TRANSITIVE", "DITRANSITIVE", "COPULA", "NONE"}
	for _, v := range longForms {
		t.Run(v, func(t *testing.T) {
			csvData := "word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags,valency\n" +
				"4097,1,1,test,EN,test,NEUTRAL,NONE,EVALUATION,GENERAL,100,0," + v
			path := writeCSV(t, "words.csv", csvData)
			_, err := seed.LoadWords(path)
			if err != nil {
				t.Fatalf("valency=%s: %v", v, err)
			}
		})
	}
}

// ── col function edge cases (tested via missing column names) ───────────

func TestLoadWords_ColumnNotInHeader(t *testing.T) {
	// CSV without some optional columns → col() returns "" for missing names
	const csvData = `word_id,root_id,variant,word,lang,norm
4097,1,1,negative,EN,negative`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("minimal columns: %v", err)
	}
	if len(words) != 1 {
		t.Fatalf("want 1 word, got %d", len(words))
	}
	// All optional fields should default
	if words[0].FreqRank != 0 {
		t.Errorf("FreqRank should be 0, got %d", words[0].FreqRank)
	}
}

func TestLoadWords_FewerRowFieldsThanHeader(t *testing.T) {
	// FieldsPerRecord=-1 allows rows with fewer fields → col() handles out-of-bounds
	const csvData = `word_id,root_id,variant,word,lang,norm,polarity,intensity,semantic_role,domain,freq_rank,flags
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("fewer fields than header: %v", err)
	}
	if len(words) != 1 {
		t.Fatalf("want 1 word, got %d", len(words))
	}
	// freq_rank and flags columns exist in header but not in row → defaults
	if words[0].FreqRank != 0 {
		t.Errorf("FreqRank should default to 0, got %d", words[0].FreqRank)
	}
}

// ── indexHeader edge cases ──────────────────────────────────────────────

func TestLoadWords_DuplicateHeaderColumns(t *testing.T) {
	// Duplicate column name → last one wins in indexHeader
	const csvData = `word_id,root_id,variant,word,lang,norm,word
4097,1,1,first,EN,first,second`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("duplicate header: %v", err)
	}
	// The second "word" column (index 6) should win
	if words[0].Word != "second" {
		t.Errorf("duplicate 'word' header: want 'second' (last wins), got %q", words[0].Word)
	}
}

func TestLoadWords_CaseInsensitiveHeader(t *testing.T) {
	// Header with mixed case → indexHeader normalizes to lowercase
	const csvData = `Word_ID,Root_ID,Variant,WORD,Lang,Norm,Polarity,Intensity,Semantic_Role,Domain,Freq_Rank,Flags
4097,1,1,negative,EN,negative,NEGATIVE,MODERATE,EVALUATION,GENERAL,1200,0`
	path := writeCSV(t, "words.csv", csvData)
	words, err := seed.LoadWords(path)
	if err != nil {
		t.Fatalf("case-insensitive header: %v", err)
	}
	if len(words) != 1 {
		t.Fatalf("want 1 word, got %d", len(words))
	}
	if words[0].Word != "negative" {
		t.Errorf("want Word='negative', got %q", words[0].Word)
	}
}
