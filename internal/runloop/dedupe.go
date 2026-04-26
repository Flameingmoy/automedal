// Motivation-similarity dedupe for the experiment queue.
//
// After the Strategist writes a fresh queue, scan each pending entry's
// **Hypothesis** field against the journal's recent diff_summaries. If
// BM25 similarity exceeds a configurable threshold, mark the queue
// entry as `[STATUS: skipped-duplicate]` and append a one-line note
// citing the matched journal entry.
//
// Bypass: include the literal token `[force]` anywhere in a queue
// entry to skip dedupe for that entry.
//
// Mirrors automedal/dedupe.py.
package runloop

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

var (
	// "## 3. catboost-native-cats [axis: HPO] [STATUS: pending]"
	dedupeEntryRE = regexp.MustCompile(`(?m)^## (\d+)\. ([^\s\[]+)[^\n]*?\[STATUS:\s*([^\]]+)\]`)
	// "**Hypothesis:** ..." up to the next "**" block, "success_criteria:", or EOF.
	dedupeHypothesisRE = regexp.MustCompile(`(?s)\*\*Hypothesis:\*\*\s*(.+?)(?:\n\*\*|\nsuccess_criteria|\z)`)
	dedupeDiffRE       = regexp.MustCompile(`(?m)^diff_summary:\s*(.+)$`)
	dedupePendingRE    = regexp.MustCompile(`\[STATUS:\s*pending\]`)
)

// DedupeSummary captures one apply() call's bookkeeping for the harness log.
type DedupeSummary struct {
	Scanned   int
	Marked    int
	Threshold float64
	JournalN  int
}

// ApplyDedupe walks pending entries in queuePath and marks any whose
// hypothesis BM25-matches a recent journal diff_summary. Mutates the
// queue file in place when at least one entry is marked.
//
// threshold == 0 → read AUTOMEDAL_DEDUPE_THRESHOLD env var, default 5.0.
func ApplyDedupe(queuePath, journalDir string, threshold float64) DedupeSummary {
	if threshold == 0 {
		threshold = 5.0
		if raw := strings.TrimSpace(os.Getenv("AUTOMEDAL_DEDUPE_THRESHOLD")); raw != "" {
			if v, err := strconv.ParseFloat(raw, 64); err == nil {
				threshold = v
			}
		}
	}

	summary := DedupeSummary{Threshold: threshold}

	raw, err := os.ReadFile(queuePath)
	if err != nil {
		return summary
	}
	text := string(raw)

	entries := splitDedupeEntries(text)
	diffs := journalDiffs(journalDir, 30)
	summary.JournalN = len(diffs)

	diffBodies := make([]string, len(diffs))
	diffLabels := make([]string, len(diffs))
	for i, d := range diffs {
		diffBodies[i] = d.body
		diffLabels[i] = d.label
	}

	newText := text
	// Walk in reverse so absolute offsets remain valid as we mutate.
	for i := len(entries) - 1; i >= 0; i-- {
		e := entries[i]
		// Re-find the header inside the body so we know status / pending.
		m := dedupeEntryRE.FindStringSubmatch(e.body)
		if m == nil {
			continue
		}
		status := strings.TrimSpace(m[3])
		if !strings.EqualFold(status, "pending") {
			continue
		}
		if strings.Contains(e.body, "[force]") {
			continue
		}
		hyp := extractHypothesis(e.body)
		if hyp == "" {
			continue
		}
		summary.Scanned++
		if len(diffBodies) == 0 {
			continue
		}
		scores := tools.BM25ScorePairs(hyp, diffBodies)
		if len(scores) == 0 {
			continue
		}
		peakIdx, peak := argmax(scores)
		if peak < threshold {
			continue
		}
		reason := "matches journal entry " + strconv.Quote(diffLabels[peakIdx]) +
			" (BM25=" + strconv.FormatFloat(peak, 'f', 2, 64) +
			" ≥ " + strconv.FormatFloat(threshold, 'f', 2, 64) + ")"
		newBody := markSkipped(e.body, reason)
		if newBody != e.body {
			newText = newText[:e.start] + newBody + newText[e.end:]
			summary.Marked++
		}
	}

	if newText != text {
		_ = os.WriteFile(queuePath, []byte(newText), 0o644)
	}
	return summary
}

// ── helpers ──────────────────────────────────────────────────────────

type dedupeEntry struct {
	start, end int
	body       string
}

func splitDedupeEntries(text string) []dedupeEntry {
	idxs := dedupeEntryRE.FindAllStringIndex(text, -1)
	if len(idxs) == 0 {
		return nil
	}
	out := make([]dedupeEntry, 0, len(idxs))
	for i, h := range idxs {
		start := h[0]
		end := len(text)
		if i+1 < len(idxs) {
			end = idxs[i+1][0]
		}
		out = append(out, dedupeEntry{start: start, end: end, body: text[start:end]})
	}
	return out
}

func extractHypothesis(entry string) string {
	m := dedupeHypothesisRE.FindStringSubmatch(entry)
	if m == nil {
		return ""
	}
	first := strings.TrimSpace(m[1])
	if i := strings.IndexByte(first, '\n'); i >= 0 {
		first = first[:i]
	}
	return strings.TrimSpace(first)
}

type journalDiff struct {
	label, body string
}

func journalDiffs(journalDir string, n int) []journalDiff {
	info, err := os.Stat(journalDir)
	if err != nil || !info.IsDir() {
		return nil
	}
	entries, err := filepath.Glob(filepath.Join(journalDir, "*.md"))
	if err != nil {
		return nil
	}
	sort.Strings(entries)
	if len(entries) > n {
		entries = entries[len(entries)-n:]
	}
	out := make([]journalDiff, 0, len(entries))
	for _, p := range entries {
		raw, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		m := dedupeDiffRE.FindStringSubmatch(string(raw))
		if m == nil {
			continue
		}
		stem := strings.TrimSuffix(filepath.Base(p), ".md")
		out = append(out, journalDiff{label: stem, body: strings.TrimSpace(m[1])})
	}
	return out
}

func markSkipped(entry, reason string) string {
	replaced := dedupePendingRE.ReplaceAllStringFunc(entry, func(_ string) string {
		return "[STATUS: skipped-duplicate]"
	})
	if !strings.Contains(replaced, "skipped-duplicate") {
		return entry
	}
	return strings.TrimRight(replaced, "\n") + "\n_dedupe note: " + reason + "_\n"
}

func argmax(xs []float64) (int, float64) {
	if len(xs) == 0 {
		return -1, 0
	}
	best := 0
	for i := 1; i < len(xs); i++ {
		if xs[i] > xs[best] {
			best = i
		}
	}
	return best, xs[best]
}
