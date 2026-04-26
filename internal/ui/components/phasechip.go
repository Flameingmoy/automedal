package components

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// SpinnerFrames mirrors the v2 mockup's braille spinner.
var SpinnerFrames = []string{"⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"}

// SpinnerFrame returns the frame for tick `i` (any int — wraps).
func SpinnerFrame(i int) string {
	if i < 0 {
		i = -i
	}
	return SpinnerFrames[i%len(SpinnerFrames)]
}

// PhaseChip renders an inline pill: phase name + ● when active, spinner
// when busy, ○ when idle.  Mirrors the v2 React PhaseChip component.
func PhaseChip(phase string, active, busy bool, spinIdx int) string {
	label := strings.ToUpper(phase)
	if label == "" {
		label = "IDLE"
	}
	color := theme.PhaseColor(phase)
	if !active {
		color = lipgloss.Color(theme.ColorMuted)
	}
	mark := "○"
	if active {
		mark = "●"
	}
	if busy && active {
		mark = SpinnerFrame(spinIdx)
	}
	st := lipgloss.NewStyle().
		Foreground(color).
		Bold(active).
		Padding(0, 1)
	return st.Render(mark + " " + label)
}
