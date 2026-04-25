package templates

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var scoutFixtures = map[string]map[string]any{
	"AGENTS": {
		"competition": map[string]any{
			"slug":     "playground-series-s6e4",
			"subtitle": "Multi-class shroom",
		},
		"task": map[string]any{
			"type":               "multiclass",
			"target_col":         "class",
			"id_col":             "id",
			"class_names":        []any{"edible", "poisonous"},
			"eval_metric_kaggle": "log_loss",
			"eval_metric_proxy":  "log_loss",
		},
		"dataset": map[string]any{
			"train_rows":           500,
			"test_rows":            120,
			"numeric_features":     []any{"f1", "f2", "f3"},
			"categorical_features": []any{"c1", "c2"},
		},
	},
	"program": {
		"competition": map[string]any{
			"slug":     "playground-series-s6e4",
			"subtitle": "Multi-class shroom",
		},
	},
	"prepare_starter.py": {
		"competition": map[string]any{
			"title":    "Playground Series S6E4",
			"subtitle": "Multi-class shroom",
		},
		"task": map[string]any{
			"type":       "multiclass",
			"target_col": "class",
		},
	},
}

func TestScoutTemplateParity(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not on PATH")
	}
	repoRoot := findRepoRoot(t)
	tmplRoot := filepath.Join(repoRoot, "templates")
	if _, err := os.Stat(tmplRoot); err != nil {
		t.Skip("python templates dir gone (post-Phase-4)")
	}

	for name, slots := range scoutFixtures {
		name, slots := name, slots
		t.Run(name, func(t *testing.T) {
			pyOut, err := renderPython(repoRoot, name, slots)
			if err != nil {
				t.Fatalf("python render: %v", err)
			}
			goOut, err := Render(name, slots)
			if err != nil {
				t.Fatalf("go render: %v", err)
			}
			if pyOut != goOut {
				diffLine := firstDiff(pyOut, goOut)
				t.Errorf("byte mismatch at line %d\n--- python ---\n%s\n--- go ---\n%s",
					diffLine, snippet(pyOut, diffLine), snippet(goOut, diffLine))
			}
		})
	}
}

func renderPython(repoRoot, name string, slots map[string]any) (string, error) {
	slotsJSON, err := json.Marshal(slots)
	if err != nil {
		return "", err
	}
	// All three scout templates use the same Jinja env shape as the agent.
	pyName := name + ".j2"
	if name == "AGENTS" || name == "program" {
		pyName = name + ".md.j2"
	}
	if name == "prepare_starter.py" {
		pyName = "prepare_starter.py.j2"
	}
	script := `
import json, sys
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(disabled_extensions=("j2","md","py"), default=False),
    undefined=StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=True,
)
slots = json.loads(sys.argv[1])
sys.stdout.write(env.get_template(sys.argv[2]).render(**slots))
`
	cmd := exec.Command("python3", "-c", script, string(slotsJSON), pyName)
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
