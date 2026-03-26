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
