package components

import (
	"fmt"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// StatusData is a pure struct the model passes into the status bar.
// Mirrors tui/widgets/status_strip.py + advisor state.
type StatusData struct {
	Brand        string
	Competition  string
	Phase        string
	Iteration    int
	TotalIters   int
	BestLoss     float64
	BestLossSet  bool
	AdvisorOn    bool
	AdvisorModel string
	AdvisorBusy  bool
}

// StatusBar renders a horizontal strip: brand · competition · right-aligned pills.
func StatusBar(d StatusData, width int) string {
	brand := theme.Accent.Render(d.Brand)
	comp := d.Competition
	if comp == "" {
		comp = "(no competition)"
	}
	left := fmt.Sprintf("%s  · %s", brand, theme.Muted.Render(comp))

	var pills []string
	if d.BestLossSet {
		pills = append(pills, theme.PillIdle.Render(fmt.Sprintf("best=%.4f", d.BestLoss)))
	}
	if d.Iteration > 0 {
		total := ""
		if d.TotalIters > 0 {
			total = fmt.Sprintf("/%d", d.TotalIters)
		}
		pills = append(pills, theme.PillIdle.Render(fmt.Sprintf("iter %d%s", d.Iteration, total)))
	}
	if d.Phase != "" && strings.ToUpper(d.Phase) != "IDLE" {
		pills = append(pills, theme.PillActive.Render(strings.ToUpper(d.Phase)))
	}
	if d.AdvisorOn {
		label := "advisor off"
		style := theme.PillIdle
		if d.AdvisorModel != "" {
			label = "advisor:" + d.AdvisorModel
			style = theme.PillAdvisor
		}
		if d.AdvisorBusy {
			label += " ●"
		}
		pills = append(pills, style.Render(label))
	}
	right := strings.Join(pills, " ")

	// Pad left/right to fill width.
	if width > 0 {
		gap := width - lipgloss.Width(left) - lipgloss.Width(right)
		if gap < 1 {
			gap = 1
		}
		return left + strings.Repeat(" ", gap) + right
	}
	return left + "   " + right
}
