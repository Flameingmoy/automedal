package components

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// InfoRow is one (key, value, valueColor) tuple for the neofetch-style
// info table on the Home screen.
type InfoRow struct {
	Key   string
	Value string
	Color string // hex; empty = ColorText
}

// InfoTable renders a two-column key/value list.  Keys are jade,
// values use their per-row colour.  `keyWidth` controls left column.
func InfoTable(rows []InfoRow, keyWidth int) string {
	if keyWidth < 8 {
		keyWidth = 14
	}
	keyStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true)

	var b strings.Builder
	for i, r := range rows {
		key := keyStyle.Render(padR(r.Key, keyWidth))
		col := lipgloss.Color(theme.ColorText)
		if r.Color != "" {
			col = lipgloss.Color(r.Color)
		}
		val := lipgloss.NewStyle().Foreground(col).Render(r.Value)
		b.WriteString(key + val)
		if i < len(rows)-1 {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func padR(s string, w int) string {
	for len(s) < w {
		s += " "
	}
	return s
}

// Separator renders a jade horizontal rule of `width` chars.
func Separator(width int, color string) string {
	if width <= 0 {
		return ""
	}
	col := lipgloss.Color(theme.ColorMuted)
	if color != "" {
		col = lipgloss.Color(color)
	}
	return lipgloss.NewStyle().
		Foreground(col).
		Render(strings.Repeat("─", width))
}

// PhaseSwatches renders coloured boxes for each phase (Home footer
// equivalent).
func PhaseSwatches() string {
	phases := []struct {
		Name, Color string
	}{
		{"researcher", theme.ColorResearcher},
		{"strategist", theme.ColorStrategist},
		{"experimenter", theme.ColorExperimenter},
		{"analyzer", theme.ColorAnalyzer},
		{"advisor", theme.ColorAdvisor},
	}
	parts := make([]string, 0, len(phases))
	for _, p := range phases {
		swatch := lipgloss.NewStyle().
			Background(lipgloss.Color(p.Color)).
			Foreground(lipgloss.Color(theme.ColorBg)).
			Padding(0, 1).
			Render("  ")
		label := lipgloss.NewStyle().
			Foreground(lipgloss.Color(p.Color)).
			Render(p.Name)
		parts = append(parts, swatch+" "+label)
	}
	return strings.Join(parts, "   ")
}
