package discover

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kak/umcs/pkg/infer"
	"github.com/kak/umcs/pkg/morpheme"
	"github.com/kak/umcs/pkg/seed"
	"github.com/kak/umcs/pkg/sentiment"
)

// Checkpoint tracks which words have already been processed so runs can be resumed.
// Processed is a set (map) for O(1) lookup; serialised as a JSON object.
type Checkpoint struct {
	Processed map[string]bool `json:"processed"`
	LastRun   string          `json:"last_run"`
}

// LoadCheckpoint reads the checkpoint from path or returns an empty one.
// Handles the old list format gracefully by starting fresh on parse error.
func LoadCheckpoint(path string) (*Checkpoint, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &Checkpoint{Processed: make(map[string]bool)}, nil
	}
	if err != nil {
		return nil, err
	}
	var cp Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		log.Printf("warning: checkpoint format changed, starting fresh: %v", err)
		return &Checkpoint{Processed: make(map[string]bool)}, nil
	}
	if cp.Processed == nil {
		cp.Processed = make(map[string]bool)
	}
	return &cp, nil
}

// Save writes the checkpoint to disk.
func (cp *Checkpoint) Save(path string) error {
	cp.LastRun = time.Now().UTC().Format(time.RFC3339)
	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// IsProcessed reports whether a fetch key was already processed.
func (cp *Checkpoint) IsProcessed(key string) bool {
	return cp.Processed[key]
}

// Mark records a key as processed.
func (cp *Checkpoint) Mark(key string) {
	cp.Processed[key] = true
}

// Flush appends new roots and words to the CSV files.
func Flush(newRoots []seed.Root, newWords []seed.Word, rootsPath, wordsPath string) error {
	if len(newRoots) > 0 {
		if err := appendRoots(newRoots, rootsPath); err != nil {
			return fmt.Errorf("append roots: %w", err)
		}
	}
	if len(newWords) > 0 {
		if err := appendWords(newWords, wordsPath); err != nil {
			return fmt.Errorf("append words: %w", err)
		}
	}
	return nil
}

func appendRoots(roots []seed.Root, path string) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	for _, r := range roots {
		if err := w.Write([]string{
			fmt.Sprintf("%d", r.RootID),
			r.RootStr,
			r.Origin,
			r.MeaningEN,
			r.Notes,
			fmt.Sprintf("%d", r.ParentRootID),
			"", // hypernym_root_id — set manually after review
			"", // antonym_root_id  — set manually after review
			"", // synonym_root_id  — set manually after review
		}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func appendWords(words []seed.Word, path string) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	for _, word := range words {
		if err := morpheme.Validate(word.WordID, word.RootID, word.Variant); err != nil {
			return fmt.Errorf("word %q: %w", word.Word, err)
		}
		d := sentiment.Decode(word.Sentiment)
		role := d["role"]
		if role == "" {
			role = "EVALUATION"
		}
		// Infer phonology into flags so the next build picks up syllables/stress.
		flags := infer.FillPhonology(word.Flags, word.Word, word.Lang)
		if err := w.Write([]string{
			fmt.Sprintf("%d", word.WordID),
			fmt.Sprintf("%d", word.RootID),
			fmt.Sprintf("%d", word.Variant),
			word.Word,
			word.Lang,
			word.Norm,
			d["polarity"],
			d["intensity"],
			role,
			d["domain"],
			fmt.Sprintf("%d", word.FreqRank),
			fmt.Sprintf("%d", flags),
			"", // pos          — inferred by infer.FillMissing on next build
			"", // arousal
			"", // dominance
			"", // aoa
			"", // concreteness
			"", // register
			"", // ontological
			"", // polysemy
			word.Pron,
			"", // syllables  — packed into flags above
			"", // stress     — packed into flags above
			"", // valency
			"", // irony_capable
			"", // neologism
		}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

// WriteStagedCSV appends low-confidence words to the staging file for manual review.
// Deduplicates by (word, lang) against existing entries to avoid accumulating duplicates
// across multiple discover runs.
func WriteStagedCSV(staged []StagedWord, path string) error {
	if len(staged) == 0 {
		return nil
	}

	// Load existing (word,lang) keys to skip duplicates.
	existing := map[string]bool{}
	if data, err := os.ReadFile(path); err == nil {
		r := csv.NewReader(strings.NewReader(string(data)))
		r.FieldsPerRecord = -1
		records, _ := r.ReadAll()
		for i, rec := range records {
			if i == 0 || len(rec) < 2 {
				continue
			}
			existing[rec[0]+"_"+rec[1]] = true
		}
	}

	writeHeader := len(existing) == 0
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if writeHeader {
		_ = w.Write([]string{
			"word", "lang", "root_str", "proposed_root_id",
			"polarity", "intensity", "role", "confidence", "source",
			"definition",
		})
	}
	for _, s := range staged {
		key := s.Word + "_" + s.Lang
		if existing[key] {
			continue
		}
		existing[key] = true
		_ = w.Write([]string{
			s.Word, s.Lang, s.RootStr,
			fmt.Sprintf("%d", s.ProposedRootID),
			s.Score.Polarity, s.Score.Intensity, s.Score.Role,
			fmt.Sprintf("%.2f", s.Score.Confidence),
			s.Score.Source,
			s.Definition,
		})
	}
	w.Flush()
	return w.Error()
}

// StagedWord is a candidate word that did not meet the confidence threshold.
type StagedWord struct {
	Word           string
	Lang           string
	RootStr        string
	ProposedRootID uint32
	Score          Score
	Definition     string
}

// CheckpointPath returns the standard checkpoint file path.
func CheckpointPath(outDir string) string {
	return filepath.Join(outDir, ".discover_checkpoint.json")
}

// StagedPath returns the standard staging file path.
func StagedPath(outDir string) string {
	return filepath.Join(outDir, "staged.csv")
}
