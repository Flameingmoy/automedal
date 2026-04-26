package components

import (
	"fmt"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// StatusData drives StatusBar.  Mirrors the v2 React props.
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
	SpinIdx      int // for the phase chip's braille spinner
}

// StatusBar renders the v2 horizontal strip:
//
//	AUTOMEDAL · competition         best=0.4987  iter 15/50  ⣾ ANALYZER  ◆ kimi-k2.6
//
// Background is the surf colour so it visually anchors the top.
func StatusBar(d StatusData, width int) string {
	brand := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Render(strings.ToUpper(orDefault(d.Brand, "AutoMedal")))

	dot := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorMuted)).Render("·")

	comp := orDefault(d.Competition, "(no competition)")
	compStyled := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorTextDim)).
		Render(comp)

	left := brand + "  " + dot + "  " + compStyled

	var pills []string
	if d.BestLossSet {
		pills = append(pills,
			theme.PillOK.Render(fmt.Sprintf("best %.4f", d.BestLoss)))
	}
	if d.Iteration > 0 {
		txt := fmt.Sprintf("iter %d", d.Iteration)
		if d.TotalIters > 0 {
			txt += fmt.Sprintf("/%d", d.TotalIters)
		}
		pills = append(pills, theme.PillIdle.Render(txt))
	}
	if d.Phase != "" {
		active := strings.ToUpper(d.Phase) != "IDLE"
		pills = append(pills, PhaseChip(d.Phase, active, active && d.AdvisorBusy, d.SpinIdx))
	}
	if d.AdvisorOn && d.AdvisorModel != "" {
		busy := ""
		if d.AdvisorBusy {
			busy = " " + SpinnerFrame(d.SpinIdx)
		}
		pills = append(pills,
			theme.PillAdvisor.Render("◆ "+d.AdvisorModel+busy))
	}
	right := strings.Join(pills, " ")

	bar := left
	if width > 0 {
		gap := width - lipgloss.Width(left) - lipgloss.Width(right) - 4
		if gap < 1 {
			gap = 1
		}
		bar = left + strings.Repeat(" ", gap) + right
	}
	return lipgloss.NewStyle().
		Background(lipgloss.Color(theme.ColorSurf)).
		Padding(0, 2).
		Width(width).
		Render(bar)
}

func orDefault(v, def string) string {
	if strings.TrimSpace(v) == "" {
		return def
	}
	return v
}
