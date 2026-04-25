// Package util has small filesystem / path helpers shared across the TUI.
package util

import (
	"os"
	"path/filepath"
)

// RepoRoot returns the repo root by walking upward from cwd looking for
// an `automedal/` directory. Falls back to cwd if not found.
func RepoRoot() string {
	cwd, err := os.Getwd()
	if err != nil {
		return "."
	}
	cur := cwd
	for i := 0; i < 10; i++ {
		if _, err := os.Stat(filepath.Join(cur, "automedal")); err == nil {
			return cur
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}
	return cwd
}

// JournalDir is where experiment markdown entries live (journal/*.md).
func JournalDir() string { return filepath.Join(RepoRoot(), "journal") }

// EventsPath is the bespoke kernel's JSONL event log.
func EventsPath() string { return filepath.Join(RepoRoot(), "agent_loop.events.jsonl") }

// ResultsPath points at the TSV leaderboard source.
func ResultsPath() string { return filepath.Join(RepoRoot(), "agent", "results.tsv") }

// KnowledgePath is the `knowledge.md` running corpus.
func KnowledgePath() string { return filepath.Join(RepoRoot(), "knowledge.md") }

// ModelsCachePath mirrors automedal/advisor/list_models.py.
func ModelsCachePath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".automedal", "models_cache.json")
}

// FileExists is a one-liner — good enough for "should we render this?".
func FileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
