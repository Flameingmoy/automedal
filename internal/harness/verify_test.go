package harness

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Flameingmoy/automedal/internal/paths"
)

func setRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	paths.SetRepoRootForTest(dir)
	return dir
}

func write(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyResearcher_NoFile(t *testing.T) {
	setRoot(t)
	w := VerifyResearcher()
	if len(w) != 1 || !strings.Contains(w[0], "does not exist") {
		t.Fatalf("expected does-not-exist warning; got %v", w)
	}
}

func TestVerifyResearcher_HappyPath(t *testing.T) {
	root := setRoot(t)
	body := strings.Join([]string{
		"# Research notes",
		"",
		"## exp 0001",
		"query: tabular learning rate schedule",
		"",
		"- Paper: A learning rate schedule (arxiv:2401.0001)",
		"- Paper: B another (arxiv:2401.0002)",
	}, "\n")
	write(t, root, "research_notes.md", body)
	if w := VerifyResearcher(); len(w) != 0 {
		t.Errorf("expected no warnings; got %v", w)
	}
}

func TestVerifyStrategist_BulletCapAndCitations(t *testing.T) {
	root := setRoot(t)
	var b strings.Builder
	b.WriteString("## Open questions\n- still figuring out\n\n## Findings\n")
	for i := 0; i < 81; i++ {
		// 81 bullets in non-Open section without citations triggers both
		// the cap warning and a citation warning.
		b.WriteString("- a generic bullet without citation\n")
	}
	write(t, root, "knowledge.md", b.String())
	write(t, root, "experiment_queue.md", "")
	w := VerifyStrategist()
	hasCap, hasCite := false, false
	for _, s := range w {
		if strings.Contains(s, "cap is 80") {
			hasCap = true
		}
		if strings.Contains(s, "missing exp citation") {
			hasCite = true
		}
	}
	if !hasCap || !hasCite {
		t.Fatalf("expected cap + citation warnings; got %v", w)
	}
}

func TestVerifyStrategist_ValidQueue(t *testing.T) {
	root := setRoot(t)
	write(t, root, "knowledge.md", "## Open questions\n- something\n\n## Findings\n- exp 0001 helped\n")
	q := ""
	for i := 1; i <= 5; i++ {
		q += "## " + itoa(i) + ". slug-" + itoa(i) + " [axis: HPO] [STATUS: pending]\n" +
			"**Hypothesis:** x\n**Sketch:** y\n**Expected:** z\n\n"
	}
	write(t, root, "experiment_queue.md", q)
	w := VerifyStrategist()
	// Only allowed warning: max-per-axis (5 HPO > 2). Anything else fails.
	axisOnly := true
	for _, s := range w {
		if !strings.Contains(s, "max 2") {
			axisOnly = false
			break
		}
	}
	if !axisOnly {
		t.Fatalf("unexpected warnings: %v", w)
	}
}

func TestVerifyExperimenter_RequiresExpID(t *testing.T) {
	setRoot(t)
	w := VerifyExperimenter("")
	if len(w) == 0 || !strings.Contains(w[0], "requires --exp-id") {
		t.Fatalf("expected exp-id warning; got %v", w)
	}
}

func TestVerifyExperimenter_HappyPath(t *testing.T) {
	root := setRoot(t)
	journal := strings.Join([]string{
		"---",
		"id: 0007",
		"slug: lr-warmup",
		"timestamp: 2026-04-26T10:00:00Z",
		"git_tag: exp-0007",
		"queue_entry: 1",
		"status: improved",
		"val_loss: 0.21",
		"val_accuracy: 0.95",
		"best_so_far: 0.21",
		"---",
		"",
		"## Hypothesis",
		"warmup helps stability",
		"",
		"## What I changed",
		"added warmup_steps=500",
		"",
		"## Result",
		"val_loss 0.21",
		"",
		"## What I learned",
		"warmup converges faster",
	}, "\n")
	write(t, root, "journal/0007-lr-warmup.md", journal)
	// Non-empty knowledge → KB-consulted required.
	write(t, root, "knowledge.md", "## Findings\n- exp 0001 baseline\n")
	write(t, root, "agent/results.tsv", "exp_id\tval_loss\n0007\t0.21\n")
	w := VerifyExperimenter("0007")
	// Expect exactly the KB-consulted warning (we omitted that section).
	if len(w) != 1 || !strings.Contains(w[0], "KB entries consulted") {
		t.Fatalf("expected KB-consulted warning only; got %v", w)
	}
}

func TestCheckRegression(t *testing.T) {
	if w := CheckRegression(0.99, 1.0); len(w) != 0 {
		t.Errorf("0.99 vs 1.0 should pass; got %v", w)
	}
	if w := CheckRegression(1.02, 1.0); len(w) == 0 {
		t.Error("1.02 vs 1.0 should warn (>1%)")
	}
	if w := CheckRegression(1.005, 1.0); len(w) != 0 {
		t.Errorf("1.005 vs 1.0 should pass (≤1%%); got %v", w)
	}
}

func TestCheckSuccessCriteria(t *testing.T) {
	root := setRoot(t)
	q := strings.Join([]string{
		"## 1. lr-warmup [axis: HPO] [STATUS: running]",
		"**Hypothesis:** x",
		"**Sketch:** y",
		"**Expected:** z",
		"success_criteria: val_loss <= 0.5",
		"",
	}, "\n")
	write(t, root, "experiment_queue.md", q)
	passed, near, crit := CheckSuccessCriteria("0007", 0.4, 0.4)
	if !passed || near || crit == "" {
		t.Errorf("expected pass; got passed=%v near=%v crit=%q", passed, near, crit)
	}
	passed, near, _ = CheckSuccessCriteria("0007", 0.55, 0.55)
	if passed {
		t.Error("0.55 should fail criterion 0.5")
	}
	if !near {
		// 0.55 * 0.99 = 0.5445; that's still > 0.5. Not near-miss in this case.
		// But e.g. 0.504 * 0.99 ≈ 0.499 → near-miss.
	}
	passed, near, _ = CheckSuccessCriteria("0007", 0.504, 0.504)
	if passed || !near {
		t.Errorf("0.504 should be near-miss; got passed=%v near=%v", passed, near)
	}
}

// tiny strconv-free helper to avoid importing in tests
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	out := ""
	for i > 0 {
		out = string(rune('0'+i%10)) + out
		i /= 10
	}
	return out
}
