// Package util has small filesystem / path helpers shared across the TUI.
package util

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// modulePath is the canonical Go module path of the AutoMedal repo.
// We anchor RepoRoot() on this so the resolver still works after the
// Phase-4 deletion of the Python `automedal/` directory.
const modulePath = "github.com/Flameingmoy/automedal"

// RepoRoot returns the resolved AutoMedal project root.  See
// RepoRootResolved for the heuristic order.  Falls back to cwd when
// nothing matches.
func RepoRoot() string {
	root, _ := RepoRootResolved()
	return root
}

// RepoRootResolved returns the project root and a short label naming
// the heuristic that matched.  Useful for surfacing "where am I
// looking?" in empty-state UI.
//
// Resolution order:
//  1. $AUTOMEDAL_CWD if it points to an existing directory.
//  2. Walk upward from cwd looking for a go.mod that declares the
//     AutoMedal module path.
//  3. Walk upward looking for `agent_loop.events.jsonl` or
//     `agent/results.tsv` (an active project even mid-port).
//  4. cwd.
func RepoRootResolved() (root, source string) {
	if env := strings.TrimSpace(os.Getenv("AUTOMEDAL_CWD")); env != "" {
		if st, err := os.Stat(env); err == nil && st.IsDir() {
			return env, "AUTOMEDAL_CWD"
		}
	}

	cwd, err := os.Getwd()
	if err != nil {
		return ".", "fallback"
	}

	if r := walkUp(cwd, hasModuleGoMod); r != "" {
		return r, "go.mod"
	}
	if r := walkUp(cwd, hasProjectArtefacts); r != "" {
		return r, "artefacts"
	}
	return cwd, "cwd"
}

// walkUp climbs at most 12 ancestors of `start`, returning the first
// directory for which `match` returns true.  Empty string if none.
func walkUp(start string, match func(dir string) bool) string {
	cur := start
	for i := 0; i < 12; i++ {
		if match(cur) {
			return cur
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return ""
		}
		cur = parent
	}
	return ""
}

func hasModuleGoMod(dir string) bool {
	f, err := os.Open(filepath.Join(dir, "go.mod"))
	if err != nil {
		return false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if strings.HasPrefix(line, "module ") {
			return strings.Contains(line, modulePath)
		}
	}
	return false
}

func hasProjectArtefacts(dir string) bool {
	candidates := []string{
		"agent_loop.events.jsonl",
		filepath.Join("agent", "results.tsv"),
		"competition.yaml",
	}
	for _, p := range candidates {
		if _, err := os.Stat(filepath.Join(dir, p)); err == nil {
			return true
		}
	}
	return false
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
