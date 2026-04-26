// Render the bootstrap files (AGENTS.md, agent/program.md,
// agent/prepare.py) from the embedded scout templates. Mirrors
// scout/render.py.
package scout

import (
	"os"
	"path/filepath"

	"github.com/Flameingmoy/automedal/internal/scout/templates"
)

// renderTargets maps template basename → output path (relative to root).
var renderTargets = map[string]string{
	"AGENTS":  "AGENTS.md",
	"program": filepath.Join("agent", "program.md"),
}

// RenderTemplates renders AGENTS.md + agent/program.md.
func RenderTemplates(root string, slots map[string]any) ([]string, error) {
	var written []string
	for name, rel := range renderTargets {
		out, err := templates.Render(name, slots)
		if err != nil {
			return written, err
		}
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			return written, err
		}
		if err := os.WriteFile(full, []byte(out), 0o644); err != nil {
			return written, err
		}
		written = append(written, full)
	}
	return written, nil
}

// RenderPrepareStarter writes agent/prepare.py only when it does not
// already exist (mirrors the Python skip semantics).
func RenderPrepareStarter(root string, slots map[string]any) (string, bool, error) {
	full := filepath.Join(root, "agent", "prepare.py")
	if _, err := os.Stat(full); err == nil {
		return full, false, nil
	}
	out, err := templates.Render("prepare_starter.py", slots)
	if err != nil {
		return "", false, err
	}
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return "", false, err
	}
	if err := os.WriteFile(full, []byte(out), 0o644); err != nil {
		return "", false, err
	}
	return full, true, nil
}
