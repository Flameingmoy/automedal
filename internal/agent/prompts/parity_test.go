package prompts

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// fixtures define a canonical slot dict per phase. We render both Go and
// Python with these inputs and assert byte-equal output.
var phaseFixtures = map[string]map[string]any{
	"researcher": {
		"exp_id":             "0042",
		"trigger":            "stagnation",
		"stagnating":         true,
		"scheduled_research": false,
		"best_val_loss":      0.0501,
	},
	"strategist": {
		"exp_id":         "0042",
		"iteration":      7,
		"max_iters":      50,
		"stagnating":     true,
		"best_val_loss":  0.0501,
		"pending":        2,
		"reflective":     "(empty trace)",
		"ranked":         "(empty rank)",
		"advisor_note":   "Switch to dart-style boosting.",
	},
	"experimenter": {
		"exp_id":        "0042",
		"best_val_loss": 0.0501,
		"retry":         false,
	},
	"experimenter_eval": {
		"exp_id":        "0042",
		"best_val_loss": 0.0501,
		"train_rc":      0,
		"final_loss":    0.0498,
	},
	"analyzer": {
		"exp_id":         "0042",
		"slug":           "xgboost-deeper",
		"status":         "better",
		"final_loss":     0.0498,
		"best_val_loss":  0.0501,
		"val_loss_delta": -0.0003,
	},
}

// TestPhasePromptParity is a strict byte-equal parity gate against the
// existing Python jinja renderer. Skipped when python3 / the source
// templates are unavailable (CI minimal images, post-Phase-4 deletion).
func TestPhasePromptParity(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not on PATH")
	}
	repoRoot := findRepoRoot(t)
	pyEntry := filepath.Join(repoRoot, "automedal", "agent", "prompts", "__init__.py")
	if _, err := os.Stat(pyEntry); err != nil {
		t.Skip("python prompts package gone (post-Phase-4)")
	}

	for phase, slots := range phaseFixtures {
		phase, slots := phase, slots
		t.Run(phase, func(t *testing.T) {
			pyOut, err := renderPython(repoRoot, phase, slots)
			if err != nil {
				t.Fatalf("python render failed: %v", err)
			}
			goOut, err := Render(phase, slots)
			if err != nil {
				t.Fatalf("go render failed: %v", err)
			}
			if pyOut != goOut {
				diffLine := firstDiff(pyOut, goOut)
				t.Errorf("byte mismatch at line %d\n--- python ---\n%s\n--- go ---\n%s",
					diffLine, snippet(pyOut, diffLine), snippet(goOut, diffLine))
			}
		})
	}
}

// renderPython shells out to python and asks it to render a phase prompt
// with the given slots, returning the rendered text.
func renderPython(repoRoot, phase string, slots map[string]any) (string, error) {
	slotsJSON, err := json.Marshal(slots)
	if err != nil {
		return "", err
	}
	script := `
import json, sys
sys.path.insert(0, ".")
from automedal.agent.prompts import render_prompt
slots = json.loads(sys.argv[1])
sys.stdout.write(render_prompt(sys.argv[2], **slots))
`
	cmd := exec.Command("python3", "-c", script, string(slotsJSON), phase)
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
