package components

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// HintPair is one (key, description) entry for the footer hint bar.
type HintPair struct{ Key, Desc string }

// FooterHints renders the bottom-of-screen hint strip from `hints`.
// Keys appear bold-dim, descriptions in muted.
func FooterHints(hints []HintPair, width int) string {
	keyStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorTextDim)).
		Bold(true)
	descStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorMuted))

	parts := make([]string, 0, len(hints))
	for _, h := range hints {
		parts = append(parts, keyStyle.Render(h.Key)+" "+descStyle.Render(h.Desc))
	}
	body := strings.Join(parts, descStyle.Render("  ·  "))
	if width <= 0 {
		return body
	}
	return lipgloss.NewStyle().
		Background(lipgloss.Color(theme.ColorSurf)).
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Padding(0, 2).
		Width(width).
		Render(body)
}
