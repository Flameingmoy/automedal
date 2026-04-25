package harness

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// RankedEntry is one journal ranked by learning value.
type RankedEntry struct {
	Fname   string
	ID      string
	Slug    string
	Status  string
	ValLoss *float64
	Delta   *float64
	Axis    string
	Learned string
	Score   int
}

// RankJournals scans journalDir/*.md (most recent M by filename desc),
// computes a learning-value score per entry, and returns the top K as a
// markdown summary. Empty string (wrapped in a placeholder) when the
// journal is empty. No LLM calls — pure heuristic mirroring
// harness/rank_journals.py:rank_journals.
//
//	+2 if status ∈ {better, improved, kept}
//	-1 if status ∈ {worse, reverted}
//	+1 if |Δ| > 0.5 · stddev(|deltas|)
//	+1 if axis isn't among the last-5 axes (diversity bonus)
func RankJournals(journalDir string, m, k int) (string, error) {
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
	if len(files) > m {
		files = files[:m]
	}
	if len(files) == 0 {
		return "(no journal entries yet)", nil
	}

	items := make([]*RankedEntry, 0, len(files))
	for _, fname := range files {
		body, err := os.ReadFile(filepath.Join(journalDir, fname))
		if err != nil {
			continue
		}
		fm := ReadFrontmatter(string(body))
		learned := ExtractSection(string(body), "What I learned")

		entry := &RankedEntry{
			Fname:   fname,
			ID:      fm["id"],
			Slug:    firstNonEmpty(fm["slug"], strings.TrimSuffix(fname, ".md")),
			Status:  strings.ToLower(fm["status"]),
			ValLoss: safeFloat(fm["val_loss"]),
			Delta:   safeFloat(fm["val_loss_delta"]),
			Axis:    fm["axis"],
			Learned: learned,
		}
		if entry.ID == "" {
			entry.ID = "?"
		}
		items = append(items, entry)
	}

	// Rolling |delta| stddev for threshold.
	var deltas []float64
	for _, e := range items {
		if e.Delta != nil {
			deltas = append(deltas, math.Abs(*e.Delta))
		}
	}
	deltaStd := stddev(deltas)

	// Last-5 axes for diversity bonus.
	recent := map[string]bool{}
	for i, e := range items {
		if i >= 5 {
			break
		}
		if e.Axis != "" {
			recent[e.Axis] = true
		}
	}

	for _, e := range items {
		score := 0
		switch e.Status {
		case "better", "improved", "kept":
			score += 2
		case "worse", "reverted":
			score -= 1
		}
		if e.Delta != nil && deltaStd > 0 && math.Abs(*e.Delta) > 0.5*deltaStd {
			score++
		}
		if e.Axis != "" && !recent[e.Axis] {
			score++
		}
		e.Score = score
	}

	sort.SliceStable(items, func(i, j int) bool { return items[i].Score > items[j].Score })
	if len(items) > k {
		items = items[:k]
	}

	var out strings.Builder
	fmt.Fprintf(&out, "## Top-%d experiments by learning value (out of last %d)\n\n", k, len(files))
	for _, e := range items {
		lossStr := ""
		if e.ValLoss != nil {
			lossStr = fmt.Sprintf("val_loss=%.4f", *e.ValLoss)
		}
		deltaStr := ""
		if e.Delta != nil {
			deltaStr = fmt.Sprintf("  Δ%+.4f", *e.Delta)
		}
		fmt.Fprintf(&out, "### exp %s — %s  [score=%d  %s  %s%s]\n",
			e.ID, e.Slug, e.Score, e.Status, lossStr, deltaStr)
		if e.Learned != "" {
			count := 0
			for _, line := range strings.Split(e.Learned, "\n") {
				if count >= 3 {
					break
				}
				trim := strings.TrimSpace(line)
				if trim != "" {
					fmt.Fprintf(&out, "  %s\n", trim)
					count++
				}
			}
		}
		out.WriteString("\n")
	}
	return out.String(), nil
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if x != "" {
			return x
		}
	}
	return ""
}

func safeFloat(s string) *float64 {
	s = strings.TrimSpace(strings.TrimLeft(s, "+"))
	if s == "" {
		return nil
	}
	var v float64
	if _, err := fmt.Sscanf(s, "%f", &v); err != nil {
		return nil
	}
	return &v
}

func stddev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean := sum / float64(len(xs))
	var variance float64
	for _, x := range xs {
		d := x - mean
		variance += d * d
	}
	variance /= float64(len(xs))
	return math.Sqrt(variance)
}
