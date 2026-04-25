package prompts

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var advisorFixtures = map[string]map[string]any{
	"stagnation": {
		"question": "Why is val_loss flat at 0.0501 for the last 4 iterations?",
		"context":  "exp 0040: lgbm bagging  -0.0001\nexp 0041: catboost depth -0.0000\nexp 0042: xgb dart    +0.0003",
	},
	"audit": {
		"question": "Are any KB bullets stale or contradicted?",
		"context":  "## Models\n- xgb beats lgbm on this task (exp 0010, 0024)\n- lgbm with dart is the new best (exp 0034)",
	},
	"tool": {
		"question": "Should I drop the 1k-row early-stop guard?",
		"context":  "I'm seeing tree counts plateau at 480 across 3 trials.",
	},
}

func TestAdvisorPromptParity(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not on PATH")
	}
	repoRoot := findRepoRoot(t)
	pyEntry := filepath.Join(repoRoot, "automedal", "advisor", "client.py")
	if _, err := os.Stat(pyEntry); err != nil {
		t.Skip("python advisor package gone (post-Phase-4)")
	}

	for junction, slots := range advisorFixtures {
		junction, slots := junction, slots
		t.Run(junction, func(t *testing.T) {
			pyOut, err := renderPython(repoRoot, junction, slots)
			if err != nil {
				t.Fatalf("python render: %v", err)
			}
			goOut, err := Render(junction, slots)
			if err != nil {
				t.Fatalf("go render: %v", err)
			}
			if pyOut != goOut {
				diffLine := firstDiff(pyOut, goOut)
				t.Errorf("byte mismatch at line %d\n--- python ---\n%q\n--- go ---\n%q",
					diffLine, snippet(pyOut, diffLine), snippet(goOut, diffLine))
			}
		})
	}
}

// renderPython mirrors the in-test helper used by the agent prompt parity
// suite. It loads jinja and renders the named advisor template.
func renderPython(repoRoot, junction string, slots map[string]any) (string, error) {
	slotsJSON, err := json.Marshal(slots)
	if err != nil {
		return "", err
	}
	script := `
import json, sys
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
env = Environment(
    loader=FileSystemLoader("automedal/advisor/prompts"),
    autoescape=select_autoescape(disabled_extensions=("j2","md"), default=False),
    undefined=StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=True,
)
slots = json.loads(sys.argv[1])
sys.stdout.write(env.get_template(sys.argv[2] + ".md.j2").render(**slots))
`
	cmd := exec.Command("python3", "-c", script, string(slotsJSON), junction)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return "", &renderErr{stderr: string(ee.Stderr), err: err}
		}
		return "", err
	}
	return string(out), nil
}

type renderErr struct {
	stderr string
	err    error
}

func (e *renderErr) Error() string { return e.err.Error() + "\n" + e.stderr }

func findRepoRoot(t *testing.T) string {
	t.Helper()
	wd, _ := os.Getwd()
	for d := wd; d != "/" && d != ""; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, "go.mod")); err == nil {
			return d
		}
	}
	t.Fatal("repo root not found")
	return ""
}

func firstDiff(a, b string) int {
	la := strings.Split(a, "\n")
	lb := strings.Split(b, "\n")
	n := len(la)
	if len(lb) < n {
		n = len(lb)
	}
	for i := 0; i < n; i++ {
		if la[i] != lb[i] {
			return i + 1
		}
	}
	return n + 1
}

func snippet(s string, line int) string {
	lines := strings.Split(s, "\n")
	start := line - 2
	if start < 0 {
		start = 0
	}
	end := line + 2
	if end > len(lines) {
		end = len(lines)
	}
	return strings.Join(lines[start:end], "\n")
}
