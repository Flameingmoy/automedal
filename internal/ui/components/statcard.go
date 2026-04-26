package components

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// StatCard renders a small bordered box with a muted uppercase label,
// a coloured value, and an optional sub line.  Matches the v2 mockup.
func StatCard(label, value, sub, valueColor string, width int) string {
	if width < 14 {
		width = 14
	}
	labelStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Bold(true)
	col := lipgloss.Color(theme.ColorText)
	if valueColor != "" {
		col = lipgloss.Color(valueColor)
	}
	valueStyle := lipgloss.NewStyle().
		Foreground(col).
		Bold(true)
	subStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted))

	body := labelStyle.Render(strings.ToUpper(label)) + "\n" +
		valueStyle.Render(value)
	if sub != "" {
		body += "\n" + subStyle.Render(sub)
	}

	box := theme.Panel.Copy().
		Width(width).
		Render(body)
	return box
}

// MiniBar renders a one-row coloured progress bar of width `w` filled
// to `pct` ∈ [0,1].  Used inside StatCard sub-lines.
func MiniBar(pct float64, w int, color string) string {
	if w < 4 {
		w = 4
	}
	if pct < 0 {
		pct = 0
	}
	if pct > 1 {
		pct = 1
	}
	filled := int(pct * float64(w))
	if filled > w {
		filled = w
	}
	col := lipgloss.Color(color)
	if color == "" {
		col = lipgloss.Color(theme.ColorJade)
	}
	on := lipgloss.NewStyle().Foreground(col).Render(strings.Repeat("━", filled))
	off := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorBorder)).
		Render(strings.Repeat("━", w-filled))
	return on + off
}
