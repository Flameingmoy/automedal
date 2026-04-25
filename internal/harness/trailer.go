package harness

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// BuildTrace renders a markdown block summarizing the last n journal
// entries (oldest first so the Strategist reads chronologically). Port
// of harness/build_trace_trailer.py:build_trace.
func BuildTrace(journalDir string, n int) (string, error) {
	info, err := os.Stat(journalDir)
	if err != nil || !info.IsDir() {
		return "(no journal entries yet)", nil
	}
	entries, err := os.ReadDir(journalDir)
	if err != nil {
		return "", err
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".md") {
			files = append(files, e.Name())
		}
	}
	sort.Sort(sort.Reverse(sort.StringSlice(files)))
	if len(files) > n {
		files = files[:n]
	}
	if len(files) == 0 {
		return "(no journal entries yet)", nil
	}

	// Reverse back to chronological (oldest first).
	sort.Strings(files)

	var blocks []string
	for _, fname := range files {
		body, err := os.ReadFile(filepath.Join(journalDir, fname))
		if err != nil {
			continue
		}
		fm := ReadFrontmatter(string(body))
		learned := ExtractSection(string(body), "What I learned")

		expID := firstNonEmpty(fm["id"], "?")
		slug := firstNonEmpty(fm["slug"], strings.TrimSuffix(fname, ".md"))
		status := firstNonEmpty(fm["status"], "?")
		valLoss := firstNonEmpty(fm["val_loss"], "?")
		delta := fm["val_loss_delta"]
		diff := fm["diff_summary"]

		deltaStr := ""
		if delta != "" {
			deltaStr = "  delta=" + delta
		}
		diffStr := ""
		if diff != "" {
			diffStr = "\n  diff: " + diff
		}

		block := fmt.Sprintf(
			"### exp %s — %s\nstatus=%s  val_loss=%s%s%s\n",
			expID, slug, status, valLoss, deltaStr, diffStr,
		)
		if learned != "" {
			block += "\n**What I learned:**\n" + learned + "\n"
		}
		blocks = append(blocks, block)
	}
	return strings.Join(blocks, "\n---\n"), nil
}
