package components

import (
	"encoding/csv"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/Flameingmoy/automedal/internal/ui/util"
	"github.com/charmbracelet/lipgloss"
)

// LeaderRow is one row in the leaderboard. Parsed from agent/results.tsv.
type LeaderRow struct {
	Timestamp string
	Method    string
	Trials    int
	ValLoss   float64
	ValAcc    float64
	Notes     string
}

// ReadLeaderboard parses the TSV, drops the header, and returns rows
// sorted by val_loss ascending (best first). Returns an empty slice on
// any error — caller shows "no results yet".
func ReadLeaderboard() []LeaderRow {
	f, err := os.Open(util.ResultsPath())
	if err != nil {
		return nil
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = '\t'
	r.FieldsPerRecord = -1 // tolerate ragged rows

	recs, err := r.ReadAll()
	if err != nil || len(recs) < 2 {
		return nil
	}
	// Assume the first row is the header (timestamp|method|trials|val_loss|val_accuracy|submission|notes).
	var out []LeaderRow
	for _, row := range recs[1:] {
		if len(row) < 5 {
			continue
		}
		trials, _ := strconv.Atoi(row[2])
		loss, _ := strconv.ParseFloat(row[3], 64)
		acc, _ := strconv.ParseFloat(row[4], 64)
		notes := ""
		if len(row) >= 7 {
			notes = row[6]
		}
		out = append(out, LeaderRow{
			Timestamp: row[0],
			Method:    row[1],
			Trials:    trials,
			ValLoss:   loss,
			ValAcc:    acc,
			Notes:     notes,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ValLoss < out[j].ValLoss })
	return out
}

// Leaderboard renders a compact styled table of the top N rows.
func Leaderboard(rows []LeaderRow, n, width int) string {
	title := theme.Accent.Render("leaderboard")
	if len(rows) == 0 {
		return theme.Panel.Copy().Width(width - 2).Render(
			title + "\n" + theme.Muted.Render("(no results yet)"),
		)
	}
	if len(rows) > n {
		rows = rows[:n]
	}

	header := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorMuted)).Bold(true).
		Render("  rank  method          loss       acc    notes")
	var lines []string
	lines = append(lines, title, header)
	for i, r := range rows {
		rank := "  " + strconv.Itoa(i+1)
		method := r.Method
		if len(method) > 14 {
			method = method[:13] + "…"
		}
		notes := strings.ReplaceAll(r.Notes, "\n", " ")
		remain := width - 40
		if remain > 0 && len(notes) > remain {
			notes = notes[:remain-1] + "…"
		}
		lines = append(lines, lpad(rank, 6)+"  "+rpad(method, 14)+"  "+
			strconvF4(r.ValLoss)+"  "+strconvF4(r.ValAcc)+"  "+theme.Muted.Render(notes))
	}
	return theme.Panel.Copy().Width(width - 2).Render(strings.Join(lines, "\n"))
}

func lpad(s string, w int) string {
	if len(s) >= w {
		return s
	}
	return strings.Repeat(" ", w-len(s)) + s
}

func rpad(s string, w int) string {
	if len(s) >= w {
		return s
	}
	return s + strings.Repeat(" ", w-len(s))
}
