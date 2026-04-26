package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBM25Basics(t *testing.T) {
	corpus := [][]string{
		Tokenize("the quick brown fox jumps over the lazy dog"),
		Tokenize("a fast brown fox is quick"),
		Tokenize("never gonna give you up"),
	}
	bm := NewBM25(corpus)
	scores := bm.Score(Tokenize("quick fox"))
	if scores[0] <= 0 || scores[1] <= 0 {
		t.Errorf("expected positive scores for matching docs, got %v", scores)
	}
	if scores[2] != 0 {
		t.Errorf("expected zero for non-matching doc, got %v", scores[2])
	}
	if scores[0] < scores[2] && scores[1] < scores[2] {
		t.Error("matching docs should outrank unrelated")
	}
}

func TestBM25ScorePairs(t *testing.T) {
	scores := BM25ScorePairs("learning rate", []string{
		"adam learning rate schedule",
		"random forest depth",
		"sgd lr 0.01 schedule",
	})
	if scores[0] <= scores[1] {
		t.Errorf("expected first to outrank second: %v", scores)
	}
}

func TestRecallChunkByHeading(t *testing.T) {
	dir := t.TempDir()
	SetRepoRootForTest(dir)
	body := strings.Join([]string{
		"intro line",
		"## Section A",
		"alpha beta gamma",
		"## Section B",
		"delta epsilon zeta",
	}, "\n")
	os.WriteFile(filepath.Join(dir, "knowledge.md"), []byte(body), 0o644)

	// Reset the global index so the test sees this temp root.
	globalIndex = &bm25Index{}

	r := Recall.Invoke(context.Background(), map[string]any{"query": "alpha"})
	if !r.OK || !strings.Contains(r.Text, "Section A") {
		t.Fatalf("recall: %#v", r)
	}
}
