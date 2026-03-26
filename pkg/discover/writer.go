package discover

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kak/lex-sentiment/pkg/morpheme"
	"github.com/kak/lex-sentiment/pkg/seed"
	"github.com/kak/lex-sentiment/pkg/sentiment"
)

// Checkpoint tracks which words have already been processed so runs can be resumed.
type Checkpoint struct {
	Processed  []string `json:"processed"`
	LastRootID uint32   `json:"last_root_id"`
	LastRun    string   `json:"last_run"`
}

// LoadCheckpoint reads the checkpoint from path or returns an empty one.
func LoadCheckpoint(path string) (*Checkpoint, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &Checkpoint{}, nil
	}
	if err != nil {
		return nil, err
	}
	var cp Checkpoint
	if json.Unmarshal(data, &cp) != nil {
		return &Checkpoint{}, nil // corrupt checkpoint → start fresh
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
	for _, k := range cp.Processed {
		if k == key {
			return true
		}
	}
	return false
}

// Mark records a key as processed.
func (cp *Checkpoint) Mark(key string) {
	cp.Processed = append(cp.Processed, key)
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
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
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
		}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func appendWords(words []seed.Word, path string) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
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
			fmt.Sprintf("%d", word.Flags),
		}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

// WriteStagedCSV writes low-confidence words to a staging file for manual review.
// The file uses the same format as words.csv.
func WriteStagedCSV(staged []StagedWord, path string) error {
	if len(staged) == 0 {
		return nil
	}
	writeHeader := true
	if _, err := os.Stat(path); err == nil {
		writeHeader = false
	}

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
