package runloop

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTemp(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestApplyDedupe_MarksMatch(t *testing.T) {
	tmp := t.TempDir()
	journalDir := filepath.Join(tmp, "journal")
	if err := os.MkdirAll(journalDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeTemp(t, journalDir, "0001-foo.md", `id: 0001
diff_summary: catboost native categoricals tuned with grid search

# body
`)

	queue := `# Experiment Queue

## 1. catboost-native-cats [axis: HPO] [STATUS: pending]
**Hypothesis:** catboost native categoricals tuned with grid search
success_criteria: val_loss < 0.5

## 2. unrelated-thing [axis: new-model] [STATUS: pending]
**Hypothesis:** train a transformer on the tabular features
success_criteria: val_loss < 0.5
`
	qpath := writeTemp(t, tmp, "experiment_queue.md", queue)

	summary := ApplyDedupe(qpath, journalDir, 0.5)
	if summary.Scanned != 2 {
		t.Errorf("scanned want 2 got %d", summary.Scanned)
	}
	if summary.Marked != 1 {
		t.Errorf("marked want 1 got %d", summary.Marked)
	}
	updated, _ := os.ReadFile(qpath)
	if !strings.Contains(string(updated), "[STATUS: skipped-duplicate]") {
		t.Errorf("expected skipped-duplicate marker; got:\n%s", updated)
	}
	if !strings.Contains(string(updated), "_dedupe note:") {
		t.Errorf("expected dedupe note line")
	}
}

func TestApplyDedupe_ForceTokenBypass(t *testing.T) {
	tmp := t.TempDir()
	journalDir := filepath.Join(tmp, "journal")
	_ = os.MkdirAll(journalDir, 0o755)
	writeTemp(t, journalDir, "0001-foo.md", `diff_summary: train an xgboost
`)
	queue := `## 1. xgb [axis: HPO] [STATUS: pending]
**Hypothesis:** train an xgboost
[force]
`
	qpath := writeTemp(t, tmp, "experiment_queue.md", queue)
	summary := ApplyDedupe(qpath, journalDir, 0.1)
	if summary.Marked != 0 {
		t.Errorf("[force] should bypass dedupe; marked=%d", summary.Marked)
	}
}

func TestApplyDedupe_NoJournal(t *testing.T) {
	tmp := t.TempDir()
	queue := `## 1. xgb [axis: HPO] [STATUS: pending]
**Hypothesis:** train an xgboost
`
	qpath := writeTemp(t, tmp, "experiment_queue.md", queue)
	summary := ApplyDedupe(qpath, filepath.Join(tmp, "journal"), 0.1)
	if summary.Marked != 0 {
		t.Errorf("no journal → no marks; got %d", summary.Marked)
	}
	if summary.Scanned != 1 {
		t.Errorf("scanned should still count entry; got %d", summary.Scanned)
	}
}
