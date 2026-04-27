package models

import (
	"fmt"
	"sort"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/components"
	"github.com/Flameingmoy/automedal/internal/ui/theme"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// TimelineModel — full-page progression chart + ranked experiment table.
// Sourced from agent/results.tsv.  No live tail; refreshed each visit.
type TimelineModel struct {
	width, height int
	rows          []components.LeaderRow
}

func NewTimeline() TimelineModel {
	return TimelineModel{rows: components.ReadLeaderboard()}
}

func (m TimelineModel) Init() tea.Cmd { return nil }

func (m TimelineModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "q", "esc":
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenHome} }
		case "d":
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenDash} }
		}
	}
	return m, nil
}

func (m TimelineModel) View() string {
	if m.width <= 0 {
		m.width = 80
	}

	// Recompute on every render so the chart picks up new TSV rows
	// the next time the screen is visited.
	rows := m.rows
	if rows == nil {
		rows = components.ReadLeaderboard()
	}

	// chronological order for the chart
	chronological := append([]components.LeaderRow(nil), rows...)
	sort.SliceStable(chronological, func(i, j int) bool {
		return chronological[i].Timestamp < chronological[j].Timestamp
	})
	series := make([]float64, 0, len(chronological))
	best := 1e18
	for _, r := range chronological {
		if r.ValLoss < best {
			best = r.ValLoss
		}
		series = append(series, best)
	}

	header := lipgloss.NewStyle().
		Background(lipgloss.Color(theme.ColorSurf)).
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Padding(0, 2).
		Width(m.width).
		Render(fmt.Sprintf("AUTORESEARCH PROGRESS  ·  %d experiments  ·  %d kept",
			len(rows), countKept(rows)))

	var bestLoss, baseLoss float64
	bestLossSet := false
	if len(rows) > 0 {
		bestLoss = rows[0].ValLoss
		bestLossSet = true
	}
	if len(chronological) > 0 {
		baseLoss = chronological[0].ValLoss
	}

	cardW := (m.width - 8) / 4
	if cardW < 14 {
		cardW = 14
	}
	bestStr := "—"
	bestColor := theme.ColorTextDim
	if bestLossSet {
		bestStr = fmt.Sprintf("%.4f", bestLoss)
		bestColor = theme.ColorOK
	}
	baseStr := "—"
	if baseLoss > 0 {
		baseStr = fmt.Sprintf("%.4f", baseLoss)
	}
	deltaStr := "—"
	deltaColor := theme.ColorTextDim
	if bestLossSet && baseLoss > 0 {
		deltaStr = fmt.Sprintf("%+.4f", bestLoss-baseLoss)
		if bestLoss < baseLoss {
			deltaColor = theme.ColorOK
		} else {
			deltaColor = theme.ColorError
		}
	}
	successStr := "—"
	if len(rows) > 0 {
		successStr = fmt.Sprintf("%.1f%%",
			100*float64(countKept(rows))/float64(len(rows)))
	}
	cards := lipgloss.JoinHorizontal(lipgloss.Top,
		components.StatCard("Baseline", baseStr, "first run", theme.ColorTextDim, cardW),
		" ",
		components.StatCard("Current Best", bestStr, "running min", bestColor, cardW),
		" ",
		components.StatCard("Total Δ", deltaStr, "best vs baseline", deltaColor, cardW),
		" ",
		components.StatCard("Success Rate", successStr, "kept / total", theme.ColorJade, cardW),
	)

	chartTitle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Render("VAL_LOSS PROGRESSION  (running best)")
	chart := components.Sparkline(series, m.width-8)
	chartBox := theme.Panel.Copy().Width(m.width - 4).Render(chartTitle + "\n" + chart)

	tableTitle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Render("EXPERIMENT LOG")

	var tbl strings.Builder
	tbl.WriteString(tableTitle + "\n")
	tbl.WriteString(lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Bold(true).
		Render("  rank  method                loss        delta       status"))
	tbl.WriteByte('\n')

	prevBest := 1e18
	for i, r := range rows {
		method := r.Method
		if len(method) > 22 {
			method = method[:21] + "…"
		}
		delta := ""
		if i == len(rows)-1 {
			delta = "— baseline"
		} else if prevBest != 1e18 {
			delta = fmt.Sprintf("%+.4f", r.ValLoss-prevBest)
		}
		prevBest = r.ValLoss

		status := "✓ kept"
		statusColor := theme.ColorOK
		if i == 0 {
			status = "★ best"
			statusColor = theme.ColorWarn
		}

		row := fmt.Sprintf("  %4d  %-22s  %-10s  %-10s  ",
			i+1, method, fmt.Sprintf("%.4f", r.ValLoss), delta)
		statusStyled := lipgloss.NewStyle().
			Foreground(lipgloss.Color(statusColor)).
			Render(status)
		tbl.WriteString(row + statusStyled + "\n")
	}

	tableBox := theme.Panel.Copy().Width(m.width - 4).Render(tbl.String())

	footer := components.FooterHints([]components.HintPair{
		{Key: "d", Desc: "dashboard"},
		{Key: "q", Desc: "home"},
		{Key: "ctrl+c", Desc: "quit"},
	}, m.width)

	return lipgloss.JoinVertical(lipgloss.Left,
		header,
		"",
		cards,
		"",
		chartBox,
		"",
		tableBox,
		footer,
	)
}

func countKept(rows []components.LeaderRow) int {
	// We do not currently mark "reverted" in TSV — every recorded row is
	// considered kept for now.  Hook here for when that field exists.
	return len(rows)
}
