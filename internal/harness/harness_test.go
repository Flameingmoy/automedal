package harness

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ── ReadFrontmatter + ExtractSection ────────────────────────────────────

func TestReadFrontmatterBasic(t *testing.T) {
	body := `---
id: 0042
slug: xgboost-tune
status: better
val_loss: 0.0501
---

## What I learned
- tuned eta to 0.05
- bagging helps
`
	fm := ReadFrontmatter(body)
	if fm["id"] != "0042" || fm["status"] != "better" || fm["val_loss"] != "0.0501" {
		t.Errorf("bad frontmatter: %v", fm)
	}
}

func TestReadFrontmatterMissingFences(t *testing.T) {
	fm := ReadFrontmatter("no fences here\n")
	if len(fm) != 0 {
		t.Error("expected empty map")
	}
}

func TestExtractSectionStopsAtNextH2(t *testing.T) {
	body := `## What I learned
alpha
beta

## Next steps
gamma
`
	got := ExtractSection(body, "What I learned")
	if got != "alpha\nbeta" {
		t.Errorf("section body mismatch: %q", got)
	}
}

// ── NextExpID ───────────────────────────────────────────────────────────

func TestNextExpIDEmptyJournal(t *testing.T) {
	dir := t.TempDir()
	journal := filepath.Join(dir, "journal")
	id, err := NextExpID(journal)
	if err != nil {
		t.Fatal(err)
	}
	if id != "0001" {
		t.Errorf("want 0001, got %q", id)
	}
}

func TestNextExpIDScansDirectory(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "0007-alpha.md"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(dir, "0003-beta.md"), []byte("x"), 0o644)
	id, err := NextExpID(dir)
	if err != nil {
		t.Fatal(err)
	}
	if id != "0008" {
		t.Errorf("want 0008, got %q", id)
	}
}

func TestNextExpIDSentinelRoundtrip(t *testing.T) {
	dir := t.TempDir()
	if id, _ := NextExpID(dir); id != "0001" {
		t.Fatalf("first call: %q", id)
	}
	if id, _ := NextExpID(dir); id != "0002" {
		t.Fatalf("second call: %q", id)
	}
	// Sentinel file should exist with "2".
	b, _ := os.ReadFile(filepath.Join(dir, ".last_exp_id"))
	if strings.TrimSpace(string(b)) != "2" {
		t.Errorf("sentinel not updated: %q", string(b))
	}
}

func TestNextExpIDFallbackOnCorruptSentinel(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "0005-foo.md"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(dir, ".last_exp_id"), []byte("garbage"), 0o644)
	id, _ := NextExpID(dir)
	if id != "0006" {
		t.Errorf("want 0006, got %q", id)
	}
}

// ── Stagnation ──────────────────────────────────────────────────────────

func TestReadValLossesParsesTSV(t *testing.T) {
	dir := t.TempDir()
	tsv := filepath.Join(dir, "r.tsv")
	os.WriteFile(tsv, []byte("id\tval_loss\texp\n"+
		"1\t0.5\ta\n"+
		"2\t0.4\tb\n"+
		"3\t\tc\n"+ // missing — skipped
		"4\t0.3\td\n"), 0o644)
	xs, err := ReadValLosses(tsv)
	if err != nil {
		t.Fatal(err)
	}
	if len(xs) != 3 || xs[0] != 0.5 || xs[1] != 0.4 || xs[2] != 0.3 {
		t.Errorf("bad losses: %v", xs)
	}
}

func TestReadValLossesMissingFile(t *testing.T) {
	xs, err := ReadValLosses(filepath.Join(t.TempDir(), "gone.tsv"))
	if err != nil || xs != nil {
		t.Errorf("missing file: xs=%v err=%v", xs, err)
	}
}

func TestIsStagnatingFewerThanKPlus1IsFalse(t *testing.T) {
	if IsStagnating(3, []float64{0.5, 0.4, 0.3}) {
		t.Error("3 points + k=3 should not stagnate")
	}
	if !IsStagnating(2, []float64{0.5, 0.4, 0.4, 0.4}) {
		t.Error("k=2 with plateau should stagnate")
	}
}

func TestIsStagnatingStrictImprovement(t *testing.T) {
	// The only improvement is a tie — still stagnating.
	if !IsStagnating(3, []float64{0.5, 0.4, 0.4, 0.4, 0.4}) {
		t.Error("tie doesn't count as improvement")
	}
	// Strict improvement in the window.
	if IsStagnating(3, []float64{0.5, 0.5, 0.4, 0.3, 0.2}) {
		t.Error("improvement in window — should not stagnate")
	}
}

func TestBestValLossInfWhenEmpty(t *testing.T) {
	if !math.IsInf(BestValLoss(nil), 1) {
		t.Error("empty → +Inf")
	}
	if BestValLoss([]float64{0.3, 0.1, 0.5}) != 0.1 {
		t.Error("min mismatch")
	}
}

// ── RankJournals ────────────────────────────────────────────────────────

func TestRankJournalsEmpty(t *testing.T) {
	dir := t.TempDir()
	got, _ := RankJournals(filepath.Join(dir, "gone"), 10, 5)
	if !strings.HasPrefix(got, "(no journal") {
		t.Errorf("missing-dir placeholder not returned: %q", got)
	}
	// Empty but existing dir.
	empty := filepath.Join(dir, "journal")
	os.MkdirAll(empty, 0o755)
	got, _ = RankJournals(empty, 10, 5)
	if !strings.HasPrefix(got, "(no journal") {
		t.Errorf("empty dir placeholder not returned: %q", got)
	}
}

func TestRankJournalsStatusBonus(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "0001-better.md"),
		[]byte("---\nid: 0001\nslug: better\nstatus: better\nval_loss: 0.1\nval_loss_delta: -0.05\naxis: trees\n---\n## What I learned\n- foo\n"),
		0o644)
	os.WriteFile(filepath.Join(dir, "0002-worse.md"),
		[]byte("---\nid: 0002\nslug: worse\nstatus: worse\nval_loss: 0.2\nval_loss_delta: +0.01\naxis: trees\n---\n"),
		0o644)
	got, _ := RankJournals(dir, 10, 5)
	iBetter := strings.Index(got, "exp 0001")
	iWorse := strings.Index(got, "exp 0002")
	if iBetter < 0 || iWorse < 0 {
		t.Fatalf("missing experiments in output: %q", got)
	}
	if iBetter >= iWorse {
		t.Errorf("better-status entry should rank above worse-status: %q", got)
	}
}

// ── BuildTrace ──────────────────────────────────────────────────────────

func TestBuildTraceChronologicalOrder(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "0010-old.md"),
		[]byte("---\nid: 0010\nslug: old\nstatus: kept\nval_loss: 0.3\n---\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "0011-mid.md"),
		[]byte("---\nid: 0011\nslug: mid\nstatus: kept\nval_loss: 0.2\n---\n"), 0o644)
	os.WriteFile(filepath.Join(dir, "0012-new.md"),
		[]byte("---\nid: 0012\nslug: new\nstatus: kept\nval_loss: 0.1\n---\n"), 0o644)

	got, _ := BuildTrace(dir, 3)
	iOld := strings.Index(got, "exp 0010")
	iMid := strings.Index(got, "exp 0011")
	iNew := strings.Index(got, "exp 0012")
	if !(iOld < iMid && iMid < iNew) {
		t.Errorf("chronological order broken: got %q", got)
	}
}

func TestBuildTraceEmpty(t *testing.T) {
	got, _ := BuildTrace(filepath.Join(t.TempDir(), "gone"), 3)
	if got != "(no journal entries yet)" {
		t.Errorf("empty placeholder: %q", got)
	}
}
