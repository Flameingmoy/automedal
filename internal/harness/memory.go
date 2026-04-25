package harness

import (
	"fmt"
	"os"
	"path/filepath"
)

const (
	knowledgeHeader = `# AutoMedal Knowledge Base
_Last curated: (none)_

## Open questions
- (Strategist will populate this on the first planning pass.)
`

	queueHeader = `# Experiment Queue
_Empty — awaiting first Strategist run._
`

	researchHeader = `# Research Notes
`
)

// MemoryResult describes what happened to one artifact during init.
// One of "created" | "reset" | "kept".
type MemoryResult = string

// InitMemory creates (or, with force=true, resets) the file-based
// memory artifacts the agent loop relies on. Port of harness/init_memory.py.
// Returns a map[artifact_name]MemoryResult.
func InitMemory(projectRoot string, force bool) (map[string]MemoryResult, error) {
	out := map[string]MemoryResult{}

	for _, entry := range []struct {
		name, path, header string
	}{
		{"knowledge.md", filepath.Join(projectRoot, "knowledge.md"), knowledgeHeader},
		{"experiment_queue.md", filepath.Join(projectRoot, "experiment_queue.md"), queueHeader},
		{"research_notes.md", filepath.Join(projectRoot, "research_notes.md"), researchHeader},
	} {
		existed := exists(entry.path)
		wrote, err := writeFile(entry.path, entry.header, force)
		if err != nil {
			return out, fmt.Errorf("%s: %w", entry.name, err)
		}
		switch {
		case !wrote:
			out[entry.name] = "kept"
		case existed:
			out[entry.name] = "reset"
		default:
			out[entry.name] = "created"
		}
	}

	journal := filepath.Join(projectRoot, "journal")
	if err := os.MkdirAll(journal, 0o755); err != nil {
		return out, err
	}
	gitkeep := filepath.Join(journal, ".gitkeep")
	if !exists(gitkeep) {
		if err := os.WriteFile(gitkeep, nil, 0o644); err != nil {
			return out, err
		}
		out["journal/"] = "created"
	} else {
		out["journal/"] = "kept"
	}

	return out, nil
}

func writeFile(path, content string, force bool) (bool, error) {
	if exists(path) && !force {
		return false, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return false, err
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return false, err
	}
	return true, nil
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
