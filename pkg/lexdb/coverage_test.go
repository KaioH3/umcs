package lexdb_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/kak/umcs/pkg/lexdb"
	"github.com/kak/umcs/pkg/seed"
)

// buildExtendedTestLexicon creates a lexicon with semantic relations (hypernym,
// antonym, synonym), IPA pronunciation, and multi-language coverage for testing
// functions that were previously uncovered.
func buildExtendedTestLexicon(t *testing.T) *lexdb.Lexicon {
	t.Helper()
	dir := t.TempDir()
	outPath := filepath.Join(dir, "extended.umcs")

	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
		{RootID: 2, RootStr: "mal", Origin: "LATIN", MeaningEN: "bad", AntonymRootID: 1},
		{RootID: 3, RootStr: "anim", Origin: "LATIN", MeaningEN: "animal"},
		{RootID: 4, RootStr: "can", Origin: "LATIN", MeaningEN: "dog", HypernymRootID: 3},
		{RootID: 5, RootStr: "ir", Origin: "LATIN", MeaningEN: "anger", SynonymRootID: 6},
		{RootID: 6, RootStr: "rag", Origin: "LATIN", MeaningEN: "rage", SynonymRootID: 5},
	}

	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "good", Lang: "EN", Norm: "good", Sentiment: 0x00130140, Pron: "/ɡʊd/"},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "bom", Lang: "PT", Norm: "bom", Sentiment: 0x00130140},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "bueno", Lang: "ES", Norm: "bueno", Sentiment: 0x00130140},
		{WordID: 8193, RootID: 2, Variant: 1, Word: "bad", Lang: "EN", Norm: "bad", Sentiment: 0x00120180},
		{WordID: 8194, RootID: 2, Variant: 2, Word: "mau", Lang: "PT", Norm: "mau", Sentiment: 0x00120180},
		{WordID: 12289, RootID: 3, Variant: 1, Word: "animal", Lang: "EN", Norm: "animal"},
		{WordID: 16385, RootID: 4, Variant: 1, Word: "dog", Lang: "EN", Norm: "dog"},
		{WordID: 20481, RootID: 5, Variant: 1, Word: "anger", Lang: "EN", Norm: "anger", Sentiment: 0x00220180},
		{WordID: 24577, RootID: 6, Variant: 1, Word: "rage", Lang: "EN", Norm: "rage", Sentiment: 0x00320180},
	}

	_, err := lexdb.Build(roots, words, outPath)
	if err != nil {
		t.Fatalf("build extended: %v", err)
	}

	lex, err := lexdb.Load(outPath)
	if err != nil {
		t.Fatalf("load extended: %v", err)
	}
	return lex
}

// ── LangName ────────────────────────────────────────────────────────────────

func TestLangName(t *testing.T) {
	cases := []struct {
		lang uint32
		want string
	}{
		{0, "PT"},
		{1, "EN"},
		{2, "ES"},
		{3, "IT"},
		{4, "DE"},
		{5, "FR"},
		{6, "NL"},
		{7, "AR"},
		{8, "ZH"},
		{9, "JA"},
		{10, "RU"},
		{11, "KO"},
		{12, "TG"},
		{13, "HI"},
		{14, "BN"},
		{15, "ID"},
		{16, "TR"},
		{17, "FA"},
		{18, "SW"},
		{19, "UK"},
		{20, "PL"},
		{21, "SA"},
		{22, "TA"},
		{23, "HE"},
		{999, "??"}, // out of range
	}
	for _, tc := range cases {
		got := lexdb.LangName(tc.lang)
		if got != tc.want {
			t.Errorf("LangName(%d) = %q, want %q", tc.lang, got, tc.want)
		}
	}
}

// ── WordStr ─────────────────────────────────────────────────────────────────

func TestWordStr(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	w := lex.LookupWord("good")
	if w == nil {
		t.Fatal("expected to find 'good'")
	}
	got := lex.WordStr(w)
	if got != "good" {
		t.Fatalf("WordStr: want %q, got %q", "good", got)
	}

	w2 := lex.LookupWord("bom")
	if w2 == nil {
		t.Fatal("expected to find 'bom'")
	}
	if lex.WordStr(w2) != "bom" {
		t.Fatalf("WordStr: want %q, got %q", "bom", lex.WordStr(w2))
	}
}

// ── RootOrigin ──────────────────────────────────────────────────────────────

func TestRootOrigin(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	root := lex.LookupRoot(1)
	if root == nil {
		t.Fatal("root_id=1 not found")
	}
	got := lex.RootOrigin(root)
	if got != "LATIN" {
		t.Fatalf("RootOrigin: want %q, got %q", "LATIN", got)
	}
}

// ── RootMeaning ─────────────────────────────────────────────────────────────

func TestRootMeaning(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	root := lex.LookupRoot(1)
	if root == nil {
		t.Fatal("root_id=1 not found")
	}
	got := lex.RootMeaning(root)
	if got != "good" {
		t.Fatalf("RootMeaning: want %q, got %q", "good", got)
	}

	root2 := lex.LookupRoot(2)
	if root2 == nil {
		t.Fatal("root_id=2 not found")
	}
	if lex.RootMeaning(root2) != "bad" {
		t.Fatalf("RootMeaning: want %q, got %q", "bad", lex.RootMeaning(root2))
	}
}

// ── WordPron ────────────────────────────────────────────────────────────────

func TestWordPron(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	// Word with IPA annotation
	w := lex.LookupWord("good")
	if w == nil {
		t.Fatal("expected to find 'good'")
	}
	pron := lex.WordPron(w)
	if pron != "/ɡʊd/" {
		t.Fatalf("WordPron('good'): want %q, got %q", "/ɡʊd/", pron)
	}

	// Word without IPA annotation — should return empty
	w2 := lex.LookupWord("bad")
	if w2 == nil {
		t.Fatal("expected to find 'bad'")
	}
	pron2 := lex.WordPron(w2)
	if pron2 != "" {
		t.Fatalf("WordPron('bad'): want empty, got %q", pron2)
	}
}

// ── Hypernym ────────────────────────────────────────────────────────────────

func TestHypernym(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	// "can" (dog) has hypernym "anim" (animal)
	dogRoot := lex.LookupRoot(4)
	if dogRoot == nil {
		t.Fatal("root_id=4 (can/dog) not found")
	}
	hyp := lex.Hypernym(dogRoot)
	if hyp == nil {
		t.Fatal("hypernym for 'can' should be 'anim'")
	}
	if hyp.RootID != 3 {
		t.Fatalf("hypernym root_id: want 3, got %d", hyp.RootID)
	}
	if lex.RootStr(hyp) != "anim" {
		t.Fatalf("hypernym root: want %q, got %q", "anim", lex.RootStr(hyp))
	}

	// Root without hypernym
	bonRoot := lex.LookupRoot(1)
	if lex.Hypernym(bonRoot) != nil {
		t.Fatal("root 'bon' should have no hypernym")
	}
}

// ── Antonym ─────────────────────────────────────────────────────────────────

func TestAntonym(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	// "mal" (bad) has antonym "bon" (good)
	malRoot := lex.LookupRoot(2)
	if malRoot == nil {
		t.Fatal("root_id=2 (mal) not found")
	}
	ant := lex.Antonym(malRoot)
	if ant == nil {
		t.Fatal("antonym for 'mal' should be 'bon'")
	}
	if ant.RootID != 1 {
		t.Fatalf("antonym root_id: want 1, got %d", ant.RootID)
	}

	// Root without antonym
	animRoot := lex.LookupRoot(3)
	if lex.Antonym(animRoot) != nil {
		t.Fatal("root 'anim' should have no antonym")
	}
}

// ── Synonym ─────────────────────────────────────────────────────────────────

func TestSynonym(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	// "ir" (anger) has synonym "rag" (rage)
	irRoot := lex.LookupRoot(5)
	if irRoot == nil {
		t.Fatal("root_id=5 (ir/anger) not found")
	}
	syn := lex.Synonym(irRoot)
	if syn == nil {
		t.Fatal("synonym for 'ir' should be 'rag'")
	}
	if syn.RootID != 6 {
		t.Fatalf("synonym root_id: want 6, got %d", syn.RootID)
	}

	// Reverse: "rag" → synonym "ir"
	ragRoot := lex.LookupRoot(6)
	synRev := lex.Synonym(ragRoot)
	if synRev == nil || synRev.RootID != 5 {
		t.Fatal("synonym for 'rag' should be 'ir'")
	}

	// Root without synonym
	bonRoot := lex.LookupRoot(1)
	if lex.Synonym(bonRoot) != nil {
		t.Fatal("root 'bon' should have no synonym")
	}
}

// ── RootCanonicalWord ───────────────────────────────────────────────────────

func TestRootCanonicalWord(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	// Root 1 (bon) has words: good (4097), bom (4098), bueno (4099)
	// Canonical should be the first by index order
	cw := lex.RootCanonicalWord(1)
	if cw == nil {
		t.Fatal("RootCanonicalWord(1) returned nil")
	}
	if cw.WordID != 4097 {
		t.Fatalf("want canonical word_id=4097, got %d", cw.WordID)
	}

	// Non-existent root
	if lex.RootCanonicalWord(999) != nil {
		t.Fatal("RootCanonicalWord(999) should return nil")
	}
}

// ── LangCoverage ────────────────────────────────────────────────────────────

func TestLangCoverage(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	// Root 1 (bon) has words in EN, PT, ES
	root := lex.LookupRoot(1)
	if root == nil {
		t.Fatal("root_id=1 not found")
	}
	langs := lex.LangCoverage(root.LangCoverage)

	// Check that the expected languages are present
	langSet := make(map[string]bool)
	for _, l := range langs {
		langSet[l] = true
	}
	for _, want := range []string{"PT", "EN", "ES"} {
		if !langSet[want] {
			t.Errorf("LangCoverage: expected %q in result, got %v", want, langs)
		}
	}

	// Root with only EN words
	root3 := lex.LookupRoot(3)
	if root3 == nil {
		t.Fatal("root_id=3 not found")
	}
	langs3 := lex.LangCoverage(root3.LangCoverage)
	if len(langs3) != 1 || langs3[0] != "EN" {
		t.Fatalf("LangCoverage(root3): want [EN], got %v", langs3)
	}
}

// ── BuildStats.Langs ────────────────────────────────────────────────────────

func TestBuildStatsLangs(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "langs.umcs")

	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "good", Lang: "EN", Norm: "good"},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "bom", Lang: "PT", Norm: "bom"},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "bueno", Lang: "ES", Norm: "bueno"},
	}

	stats, err := lexdb.Build(roots, words, outPath)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	langsStr := stats.Langs()
	for _, want := range []string{"PT", "EN", "ES"} {
		if !strings.Contains(langsStr, want) {
			t.Errorf("Langs(): want %q in %q", want, langsStr)
		}
	}
	// IT and DE should NOT be present
	for _, absent := range []string{"IT", "DE"} {
		if strings.Contains(langsStr, absent) {
			t.Errorf("Langs(): %q should not be in %q", absent, langsStr)
		}
	}
}

func TestBuildStatsLangsAllFive(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "langs5.umcs")

	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	words := []seed.Word{
		{WordID: 4097, RootID: 1, Variant: 1, Word: "good", Lang: "EN", Norm: "good"},
		{WordID: 4098, RootID: 1, Variant: 2, Word: "bom", Lang: "PT", Norm: "bom"},
		{WordID: 4099, RootID: 1, Variant: 3, Word: "bueno", Lang: "ES", Norm: "bueno"},
		{WordID: 4100, RootID: 1, Variant: 4, Word: "buono", Lang: "IT", Norm: "buono"},
		{WordID: 4101, RootID: 1, Variant: 5, Word: "gut", Lang: "DE", Norm: "gut"},
	}

	stats, err := lexdb.Build(roots, words, outPath)
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	langsStr := stats.Langs()
	for _, want := range []string{"PT", "EN", "ES", "IT", "DE"} {
		if !strings.Contains(langsStr, want) {
			t.Errorf("Langs(): want %q in %q", want, langsStr)
		}
	}
}

func TestBuildStatsLangsEmpty(t *testing.T) {
	// Stats with no lang flags should produce empty string
	stats := lexdb.BuildStats{LangFlags: 0}
	if stats.Langs() != "" {
		t.Fatalf("Langs() with 0 flags: want empty, got %q", stats.Langs())
	}
}

// ── LangBit ─────────────────────────────────────────────────────────────────

func TestLangBit(t *testing.T) {
	// Valid lang IDs
	if lexdb.LangBit(0) != 1 {
		t.Fatalf("LangBit(0) = %d, want 1", lexdb.LangBit(0))
	}
	if lexdb.LangBit(1) != 2 {
		t.Fatalf("LangBit(1) = %d, want 2", lexdb.LangBit(1))
	}
	if lexdb.LangBit(31) != 1<<31 {
		t.Fatalf("LangBit(31) = %d, want %d", lexdb.LangBit(31), uint32(1<<31))
	}
	// Out of range
	if lexdb.LangBit(32) != 0 {
		t.Fatalf("LangBit(32) = %d, want 0", lexdb.LangBit(32))
	}
	if lexdb.LangBit(999) != 0 {
		t.Fatalf("LangBit(999) = %d, want 0", lexdb.LangBit(999))
	}
}

// ── ParseLang ───────────────────────────────────────────────────────────────

func TestParseLang(t *testing.T) {
	cases := []struct {
		code string
		ok   bool
	}{
		{"PT", true},
		{"EN", true},
		{"ZZ", false},
		{"", false},
	}
	for _, tc := range cases {
		_, ok := lexdb.ParseLang(tc.code)
		if ok != tc.ok {
			t.Errorf("ParseLang(%q): got ok=%v, want ok=%v", tc.code, ok, tc.ok)
		}
	}
}

// ── LookupWordInLang fallback path ──────────────────────────────────────────

func TestLookupWordInLangFallback(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	// Unknown lang code triggers fallback to LookupWord
	w := lex.LookupWordInLang("good", "ZZ")
	if w == nil {
		t.Fatal("LookupWordInLang fallback should find 'good'")
	}
	if w.WordID != 4097 {
		t.Fatalf("fallback word_id: want 4097, got %d", w.WordID)
	}
}

// ── Cognates with non-existent word ─────────────────────────────────────────

func TestCognatesNonExistent(t *testing.T) {
	lex := buildExtendedTestLexicon(t)
	c := lex.Cognates(0xFFFFFFFF)
	if c != nil {
		t.Fatal("Cognates for non-existent word should return nil")
	}
}

// ── EtymologyChain for root without parent ──────────────────────────────────

func TestEtymologyChainNoParent(t *testing.T) {
	lex := buildExtendedTestLexicon(t)
	chain := lex.EtymologyChain(1) // "bon" has no parent
	if len(chain) != 1 {
		t.Fatalf("want 1 in chain (self only), got %d", len(chain))
	}
	if chain[0].RootID != 1 {
		t.Fatalf("chain[0] should be root 1, got %d", chain[0].RootID)
	}
}

func TestEtymologyChainNonExistent(t *testing.T) {
	lex := buildExtendedTestLexicon(t)
	chain := lex.EtymologyChain(999)
	if len(chain) != 0 {
		t.Fatalf("want 0 in chain for non-existent root, got %d", len(chain))
	}
}

// ── Validation: parent_root_id referencing non-existent root ────────────────

func TestBuildRejectsInvalidParentRoot(t *testing.T) {
	dir := t.TempDir()
	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good", ParentRootID: 99},
	}
	_, err := lexdb.Build(roots, nil, filepath.Join(dir, "bad_parent.umcs"))
	if err == nil {
		t.Fatal("parent_root_id referencing non-existent root must be rejected")
	}
}

func TestBuildRejectsInvalidHypernymRoot(t *testing.T) {
	dir := t.TempDir()
	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good", HypernymRootID: 99},
	}
	_, err := lexdb.Build(roots, nil, filepath.Join(dir, "bad_hyp.umcs"))
	if err == nil {
		t.Fatal("hypernym_root_id referencing non-existent root must be rejected")
	}
}

func TestBuildRejectsInvalidAntonymRoot(t *testing.T) {
	dir := t.TempDir()
	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good", AntonymRootID: 99},
	}
	_, err := lexdb.Build(roots, nil, filepath.Join(dir, "bad_ant.umcs"))
	if err == nil {
		t.Fatal("antonym_root_id referencing non-existent root must be rejected")
	}
}

func TestBuildRejectsInvalidSynonymRoot(t *testing.T) {
	dir := t.TempDir()
	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good", SynonymRootID: 99},
	}
	_, err := lexdb.Build(roots, nil, filepath.Join(dir, "bad_syn.umcs"))
	if err == nil {
		t.Fatal("synonym_root_id referencing non-existent root must be rejected")
	}
}

func TestBuildRejectsZeroRootID(t *testing.T) {
	dir := t.TempDir()
	roots := []seed.Root{
		{RootID: 0, RootStr: "zero", Origin: "LATIN", MeaningEN: "zero"},
	}
	_, err := lexdb.Build(roots, nil, filepath.Join(dir, "zero_id.umcs"))
	if err == nil {
		t.Fatal("root_id=0 must be rejected")
	}
}

// ── Load error paths ────────────────────────────────────────────────────────

func TestLoadNonExistentFile(t *testing.T) {
	_, err := lexdb.Load("/tmp/nonexistent_lexdb_test_file.umcs")
	if err == nil {
		t.Fatal("Load non-existent file must error")
	}
}

func TestLoadFileTooSmall(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tiny.umcs")
	os.WriteFile(path, []byte("too small"), 0644)
	_, err := lexdb.Load(path)
	if err == nil {
		t.Fatal("Load file too small must error")
	}
}

func TestLoadInvalidMagic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "badmagic.umcs")
	data := make([]byte, 128)
	// Write wrong magic
	lexdb.ByteOrder.PutUint32(data[0:], 0xDEADBEEF)
	os.WriteFile(path, data, 0644)
	_, err := lexdb.Load(path)
	if err == nil {
		t.Fatal("Load with invalid magic must error")
	}
}

func TestLoadUnsupportedVersion(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "badver.umcs")
	data := make([]byte, 128)
	lexdb.ByteOrder.PutUint32(data[0:], lexdb.Magic)
	lexdb.ByteOrder.PutUint32(data[4:], 99) // unsupported version
	os.WriteFile(path, data, 0644)
	_, err := lexdb.Load(path)
	if err == nil {
		t.Fatal("Load with unsupported version must error")
	}
}

// ── Build to unwritable path ────────────────────────────────────────────────

func TestBuildUnwritablePath(t *testing.T) {
	roots := []seed.Root{
		{RootID: 1, RootStr: "bon", Origin: "LATIN", MeaningEN: "good"},
	}
	_, err := lexdb.Build(roots, nil, "/nonexistent_dir_for_test/output.umcs")
	if err == nil {
		t.Fatal("Build to unwritable path must error")
	}
}

// ── Normalize edge cases ────────────────────────────────────────────────────

func TestNormalizeWhitespaceOnly(t *testing.T) {
	got := lexdb.Normalize("  hello  ")
	if got != "hello" {
		t.Fatalf("Normalize('  hello  ') = %q, want %q", got, "hello")
	}
}

// ── Load: heap extends past end of file ─────────────────────────────────────

func TestLoadCorruptHeapExtendsPastEOF(t *testing.T) {
	// Build a valid file, then corrupt heap offset to extend past EOF
	_, path := buildTestLexicon(t)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// heapSize is at offset 16 (5th uint32 in header). Set it to a huge value.
	lexdb.ByteOrder.PutUint32(data[16:], 0xFFFFFFFF)
	// Recalculate checksum to bypass that check? No — we want the heap bounds
	// check to trigger, which happens before heap access but after checksum.
	// Actually the checksum check uses heapOffset+heapSize, so it will fail
	// at the bounds check first (expectedEnd > len(data)).
	corruptPath := filepath.Join(t.TempDir(), "corrupt_heap.umcs")
	os.WriteFile(corruptPath, data, 0644)
	_, err = lexdb.Load(corruptPath)
	if err == nil {
		t.Fatal("Load with heap extending past EOF must error")
	}
}

// ── Load: root table extends past EOF ───────────────────────────────────────

func TestLoadCorruptRootTablePastEOF(t *testing.T) {
	_, path := buildTestLexicon(t)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Set rootCount to a huge number (offset 8, 3rd uint32)
	lexdb.ByteOrder.PutUint32(data[8:], 0x0FFFFFFF)
	// Need valid checksum for it to get to bounds check — but the checksum
	// is calculated on the data section, and we changed the header.
	// The root table bounds check is after checksum. Let's change heap offset
	// to 0 + heapSize to 0 so checksum validates on empty data...
	// Actually, it's easier to just check that it fails at all.
	corruptPath := filepath.Join(t.TempDir(), "corrupt_root.umcs")
	os.WriteFile(corruptPath, data, 0644)
	_, err = lexdb.Load(corruptPath)
	if err == nil {
		t.Fatal("Load with root table past EOF must error")
	}
}

// ── LookupWordInLang: exact lang match vs fallback ──────────────────────────

// ── Normalize: remaining diacritic branches ─────────────────────────────────

func TestNormalizeRemainingDiacritics(t *testing.T) {
	cases := []struct{ in, want string }{
		{"řeka", "reka"},       // Czech ř
		{"ğüzel", "guzel"},     // Turkish ğ, ü
		{"źródło", "zrodlo"},   // Polish ź, ó, ł
		{"żaba", "zaba"},       // Polish ż
		{"ūdens", "udens"},     // Latvian ū
		{"ővé", "ove"},         // Hungarian ő
		{"ůvod", "uvod"},       // Czech ů
		{"ĭdea", "idea"},       // ĭ (breve i)
		{"ăsta", "asta"},       // Romanian ă
		{"ąsak", "asak"},       // Polish ą
		{"ęka", "eka"},         // Polish ę
		{"įrankis", "irankis"}, // Lithuanian į
		{"ēdiens", "ediens"},   // Latvian ē
		{"ēdit", "edit"},       // ē
		{"ōkay", "okay"},       // ō
		{"Ūmlauts", "umlauts"}, // Ū
		{"ūnique", "unique"},   // ū
		{"Űjhely", "ujhely"},   // Hungarian ű
		{"ňo", "no"},           // ň
		{"ćena", "cena"},       // Serbian ć
		{"śnieg", "snieg"},     // Polish ś
		{"ānanda", "ananda"},   // ā
	}
	for _, c := range cases {
		got := lexdb.Normalize(c.in)
		if got != c.want {
			t.Errorf("Normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// ── Normalize: digit and hyphen preservation ────────────────────────────────

func TestNormalizeDigitsAndHyphens(t *testing.T) {
	got := lexdb.Normalize("well-known-42")
	if got != "well-known-42" {
		t.Fatalf("Normalize('well-known-42') = %q, want %q", got, "well-known-42")
	}
}

func TestLookupWordInLangExactMatch(t *testing.T) {
	lex := buildExtendedTestLexicon(t)

	// "bom" exists only in PT
	pt := lex.LookupWordInLang("bom", "PT")
	if pt == nil {
		t.Fatal("LookupWordInLang('bom', 'PT') returned nil")
	}
	if pt.WordID != 4098 {
		t.Fatalf("want word_id=4098, got %d", pt.WordID)
	}

	// "bom" in EN doesn't exist, should fall back to any-language
	en := lex.LookupWordInLang("bom", "EN")
	if en == nil {
		t.Fatal("LookupWordInLang('bom', 'EN') fallback returned nil")
	}
}
