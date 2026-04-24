// Package theme centralizes Lipgloss styles for the whole TUI so the palette
// lives in one place. Mirrors the Dracula-ish colors the Textual TUI uses
// (see tui/screens/home.py DEFAULT_CSS) so the two TUIs look visually
// similar during the migration period.
package theme

import "github.com/charmbracelet/lipgloss"

// Palette — keep in sync with tui/screens/home.py DEFAULT_CSS.
const (
	ColorBg      = "#0F111A"
	ColorFg      = "#F8F8F2"
	ColorMuted   = "#6272A4"
	ColorAccent  = "#8BE9FD"
	ColorPrompt  = "#50FA7B"
	ColorLogo    = "#FFD700"
	ColorOK      = "#50FA7B"
	ColorWarn    = "#F1FA8C"
	ColorError   = "#FF5555"
	ColorAdvisor = "#BD93F9"
)

var (
	// Base surface — most content sits on this.
	Base = lipgloss.NewStyle().
		Background(lipgloss.Color(ColorBg)).
		Foreground(lipgloss.Color(ColorFg))

	Muted = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorMuted))

	// Panel — rounded border, padded, for boxed widgets.
	Panel = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color(ColorMuted)).
		Padding(0, 1).
		Background(lipgloss.Color(ColorBg))

	Logo = lipgloss.NewStyle().
		Foreground(lipgloss.Color(ColorLogo)).
		Padding(1, 1, 0, 1)

	Prompt = lipgloss.NewStyle().
		Foreground(lipgloss.Color(ColorPrompt)).
		Bold(true)

	Accent = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorAccent))

	ErrorStyle = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorError)).Bold(true)
	OKStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorOK))
	WarnStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color(ColorWarn))

	// PillActive / PillIdle — short colored badges for the status bar.
	PillActive = lipgloss.NewStyle().
			Foreground(lipgloss.Color(ColorBg)).
			Background(lipgloss.Color(ColorAccent)).
			Padding(0, 1)
	PillIdle = lipgloss.NewStyle().
			Foreground(lipgloss.Color(ColorMuted)).
			Background(lipgloss.Color(ColorBg)).
			Padding(0, 1)
	PillAdvisor = lipgloss.NewStyle().
			Foreground(lipgloss.Color(ColorBg)).
			Background(lipgloss.Color(ColorAdvisor)).
			Padding(0, 1)
	PillError = lipgloss.NewStyle().
			Foreground(lipgloss.Color(ColorBg)).
			Background(lipgloss.Color(ColorError)).
			Padding(0, 1)
)
