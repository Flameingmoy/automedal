// Repo root resolver for non-tool callers (harness verify, scout, ...).
//
// Tools have their own copy in internal/agent/tools/base.go; we duplicate
// the trivial wrapper here so packages that don't import tools can still
// resolve a stable root.
package paths

import (
	"os"
	"path/filepath"
	"sync"
)

var (
	repoRootOnce sync.Once
	repoRootVal  string
)

// RepoRoot returns the absolute path AutoMedal is anchored at.
// Set AUTOMEDAL_CWD to override; otherwise os.Getwd().
func RepoRoot() string {
	repoRootOnce.Do(func() {
		if v := os.Getenv("AUTOMEDAL_CWD"); v != "" {
			if abs, err := filepath.Abs(v); err == nil {
				repoRootVal = abs
				return
			}
		}
		if cwd, err := os.Getwd(); err == nil {
			repoRootVal = cwd
			return
		}
		repoRootVal = "."
	})
	return repoRootVal
}

// SetRepoRootForTest overrides the resolved root. Tests only.
func SetRepoRootForTest(p string) {
	abs, _ := filepath.Abs(p)
	repoRootOnce.Do(func() {})
	repoRootVal = abs
}
