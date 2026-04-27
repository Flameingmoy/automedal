// Package theme centralizes Lipgloss styles for the v2 TUI. Tokens come
// from AutoMedal TUI v2 — true black surface, jade as the only primary
// accent, per-phase palette. See py-mockup at design/AutoMedal TUI v2.html.
package theme

import (
	"fmt"

	"github.com/charmbracelet/lipgloss"
	"github.com/lucasb-eyer/go-colorful"
)

// Palette — keep in sync with the v2 mockup tokens (DARK).
const (
	ColorBg       = "#0e0e0e"
	ColorSurf     = "#141414"
	ColorPanel    = "#1a1a1a"
	ColorBorder   = "#252525"
	ColorBorderHi = "#363636"
	ColorText     = "#d8d8d8"
	ColorTextDim  = "#888888"
	ColorMuted    = "#444444"
	ColorDim      = "#1e1e1e"

	// Primary accent — used sparingly.
	ColorJade = "#00cfa8"

	// Phase palette — each agent role gets its own colour (crush style).
	ColorResearcher   = "#00b899"
	ColorStrategist   = "#9b72f5"
	ColorExperimenter = "#e09b2d"
	ColorAnalyzer     = "#2ec785"
	ColorAdvisor      = "#e05595"
	ColorIdle         = "#444444"

	ColorOK    = "#2ec785"
	ColorWarn  = "#e09b2d"
	ColorError = "#e05252"

	// Banner gradient stops — jade → cyan → neon-blue.  Top of the
	// AUTOMEDAL wordmark is jade (the brand accent), bottom transitions
	// through cyan into a saturated neon blue.  Inspired by Charm's
	// crush splash but rebalanced for AutoMedal's jade-forward identity.
	GradStop1 = "#00cfa8"
	GradStop2 = "#22d3ee"
	GradStop3 = "#1d6bff"

	// Legacy aliases kept so older code compiles during migration.
	ColorFg     = ColorText
	ColorAccent = ColorJade
	ColorPrompt = ColorJade
	ColorLogo   = ColorJade
)

var (
	Base = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorText))

	Muted    = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorMuted))
	Dim      = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorTextDim))
	Subtle   = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorTextDim))
	Accent   = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorJade))
	Brand    = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorJade)).Bold(true)
	Strong   = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorText)).Bold(true)
	Headline = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorJade)).Bold(true)

	OKStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorOK))
	WarnStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorWarn))
	ErrorStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorError)).Bold(true)

	// Panel — rounded jade-trimmed surface for boxed content.
	Panel = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(ColorBorder)).
		Padding(0, 1)

	PanelHi = Panel.Copy().
		BorderForeground(lipgloss.Color(ColorJade))

	// Pills — small badges.  Variants per phase share the same shape.
	PillIdle = lipgloss.NewStyle().
			Foreground(lipgloss.Color(ColorTextDim)).
			Background(lipgloss.Color(ColorDim)).
			Padding(0, 1)
	PillJade = lipgloss.NewStyle().
			Foreground(lipgloss.Color(ColorJade)).
			Background(lipgloss.Color(ColorDim)).
			Padding(0, 1)
	PillAdvisor = lipgloss.NewStyle().
			Foreground(lipgloss.Color(ColorAdvisor)).
			Background(lipgloss.Color(ColorDim)).
			Padding(0, 1)
	PillError = lipgloss.NewStyle().
			Foreground(lipgloss.Color(ColorBg)).
			Background(lipgloss.Color(ColorError)).
			Padding(0, 1)
	PillOK = lipgloss.NewStyle().
		Foreground(lipgloss.Color(ColorOK)).
		Background(lipgloss.Color(ColorDim)).
		Padding(0, 1)

	// PillActive — generic "active" used by old callers; kept jade.
	PillActive = PillJade

	Logo = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorJade))

	Prompt = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorJade)).Bold(true)
)

// PhaseColor maps a phase name (any case) to its accent hex.  Defaults to
// the muted idle colour for unknown phases.
func PhaseColor(phase string) lipgloss.Color {
	switch toLower(phase) {
	case "researcher":
		return lipgloss.Color(ColorResearcher)
	case "strategist":
		return lipgloss.Color(ColorStrategist)
	case "experimenter", "experimenter-edit", "experimenter-eval":
		return lipgloss.Color(ColorExperimenter)
	case "analyzer":
		return lipgloss.Color(ColorAnalyzer)
	case "advisor":
		return lipgloss.Color(ColorAdvisor)
	}
	return lipgloss.Color(ColorIdle)
}

// PhaseStyle returns a foreground style coloured for the given phase.
func PhaseStyle(phase string) lipgloss.Style {
	return lipgloss.NewStyle().Foreground(PhaseColor(phase))
}

// GradientRows is an alias of GradientColors meant for callers that
// step the gradient down rows of multi-line ASCII art (banner, splash).
// Same colour math, distinct name so future per-column callers don't
// confuse the two intents.
func GradientRows(n int) []lipgloss.Color { return GradientColors(n) }

// GradientColors returns `n` interpolated colours stepping
// jade → cyan → neon-blue through HSL space.  Used by the banner.
func GradientColors(n int) []lipgloss.Color {
	if n <= 0 {
		return nil
	}
	stops := []string{GradStop1, GradStop2, GradStop3}
	c0, _ := colorful.Hex(stops[0])
	c1, _ := colorful.Hex(stops[1])
	c2, _ := colorful.Hex(stops[2])

	out := make([]lipgloss.Color, n)
	for i := 0; i < n; i++ {
		var t float64
		if n == 1 {
			t = 0
		} else {
			t = float64(i) / float64(n-1)
		}
		var col colorful.Color
		if t < 0.5 {
			col = c0.BlendHcl(c1, t*2).Clamped()
		} else {
			col = c1.BlendHcl(c2, (t-0.5)*2).Clamped()
		}
		out[i] = lipgloss.Color(hex(col))
	}
	return out
}

func hex(c colorful.Color) string {
	r, g, b := c.RGB255()
	return fmt.Sprintf("#%02x%02x%02x", r, g, b)
}

func toLower(s string) string {
	out := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		out[i] = c
	}
	return string(out)
}
