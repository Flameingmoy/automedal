package components

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// EventRow is the data shape rendered by EventItem.  Detail may be
// multi-line; it shows when Expanded is true.
type EventRow struct {
	Type     string // "phase" | "advisor" | "tool" | "training" | other
	Icon     string
	TS       string
	Label    string
	Summary  string
	Detail   string
	Expanded bool
}

func eventColor(t string) lipgloss.Color {
	switch t {
	case "phase":
		return lipgloss.Color(theme.ColorStrategist)
	case "advisor":
		return lipgloss.Color(theme.ColorAdvisor)
	case "tool":
		return lipgloss.Color(theme.ColorJade)
	case "training":
		return lipgloss.Color(theme.ColorExperimenter)
	}
	return lipgloss.Color(theme.ColorMuted)
}

// EventItem renders one row of the live event log, with an inset
// detail block when ev.Expanded.
func EventItem(ev EventRow, width int) string {
	col := eventColor(ev.Type)

	icon := lipgloss.NewStyle().Foreground(col).Render(ev.Icon)
	if icon == "" {
		icon = "·"
	}
	ts := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Render(ev.TS)
	label := lipgloss.NewStyle().
		Foreground(col).
		Bold(true).
		Render(ev.Label)
	summary := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorTextDim)).
		Render(ev.Summary)
	chev := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Render(map[bool]string{true: "▾", false: "▸"}[ev.Expanded])

	row := strings.Join([]string{icon, ts, label, summary, chev}, "  ")
	if !ev.Expanded || ev.Detail == "" {
		return row
	}
	detail := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorTextDim)).
		Background(lipgloss.Color(theme.ColorSurf)).
		BorderStyle(lipgloss.Border{Left: "▌"}).
		BorderForeground(col).
		BorderLeft(true).
		Padding(0, 1, 0, 2).
		Render(ev.Detail)
	return row + "\n" + detail
}
