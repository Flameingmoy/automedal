package components

import (
	"strings"
	"time"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/harmonica"
	"github.com/charmbracelet/lipgloss"
)

// NavTab is one entry in the top tab strip.
type NavTab struct {
	ID    string
	Label string
}

// NavTickMsg drives the spring physics.  Use NavTickCmd to pump it.
type NavTickMsg struct{}

// NavTickCmd returns a tea.Cmd that emits NavTickMsg ~30 times/sec.
// Wire it into Init() and re-arm it from Update() while the spring is
// not at rest.
func NavTickCmd() tea.Cmd {
	return tea.Tick(33*time.Millisecond, func(time.Time) tea.Msg {
		return NavTickMsg{}
	})
}

// NavBar is a bubbletea-friendly spring-animated tab strip.  The
// underline glides between tabs using harmonica spring physics.
type NavBar struct {
	tabs   []NavTab
	active int
	width  int

	// IterCur / IterTotal show the right-aligned progress.
	IterCur, IterTotal int

	spring     harmonica.Spring
	pos, vel   float64
	target     float64
	underWidth float64
	wTarget    float64
	wVel       float64
	wSpring    harmonica.Spring

	// pre-rendered tab geometry
	starts []int
	widths []int
}

// NewNavBar wires the springs and the tab list.  Initial active tab is
// the first one.
func NewNavBar(tabs []NavTab) *NavBar {
	return &NavBar{
		tabs:    tabs,
		active:  0,
		spring:  harmonica.NewSpring(harmonica.FPS(30), 6.0, 0.7),
		wSpring: harmonica.NewSpring(harmonica.FPS(30), 6.0, 0.7),
	}
}

// SetActive snaps the spring target to tab `id`.  No-op for unknown id.
func (n *NavBar) SetActive(id string) {
	for i, t := range n.tabs {
		if t.ID == id {
			n.active = i
			n.recompute()
			return
		}
	}
}

// SetWidth recomputes geometry for a new terminal width.
func (n *NavBar) SetWidth(w int) {
	n.width = w
	n.recompute()
}

// Tick advances the spring by one frame; returns true while the spring
// is still moving so the caller can keep tea.Ticking.
func (n *NavBar) Tick() bool {
	n.pos, n.vel = n.spring.Update(n.pos, n.vel, n.target)
	n.underWidth, n.wVel = n.wSpring.Update(n.underWidth, n.wVel, n.wTarget)
	if absf(n.target-n.pos) < 0.4 && absf(n.vel) < 0.2 &&
		absf(n.wTarget-n.underWidth) < 0.4 && absf(n.wVel) < 0.2 {
		n.pos, n.vel = n.target, 0
		n.underWidth, n.wVel = n.wTarget, 0
		return false
	}
	return true
}

func (n *NavBar) recompute() {
	const brand = " AM "
	const sep = " "
	col := len(brand) + 1 // +1 for the leading separator after brand
	n.starts = make([]int, len(n.tabs))
	n.widths = make([]int, len(n.tabs))
	for i, t := range n.tabs {
		w := len(t.Label) + 2 // 1ch padding both sides
		n.starts[i] = col
		n.widths[i] = w
		col += w + len(sep)
	}
	if n.active < len(n.starts) {
		n.target = float64(n.starts[n.active])
		n.wTarget = float64(n.widths[n.active])
		// Snap on first compute so we don't animate from zero.
		if n.pos == 0 && n.underWidth == 0 {
			n.pos = n.target
			n.underWidth = n.wTarget
		}
	}
}

// Render returns the two-line strip: tabs row + jade underline row.
func (n *NavBar) Render() string {
	if n.width == 0 {
		return ""
	}
	muted := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorMuted))
	jade := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true)
	dim := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorTextDim))

	brand := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Render(" AM ")

	var row strings.Builder
	row.WriteString(brand)
	row.WriteString(muted.Render("│"))
	for i, t := range n.tabs {
		label := " " + t.Label + " "
		if i == n.active {
			row.WriteString(jade.Render(label))
		} else {
			row.WriteString(muted.Render(label))
		}
		row.WriteString(" ")
	}

	right := ""
	if n.IterTotal > 0 {
		right = dim.Render(itoa(n.IterCur) + "/" + itoa(n.IterTotal))
	}

	rowStr := row.String()
	gap := n.width - lipgloss.Width(rowStr) - lipgloss.Width(right) - 2
	if gap < 1 {
		gap = 1
	}
	tabsLine := rowStr + strings.Repeat(" ", gap) + right + " "

	// Underline row — spring position controls left edge, spring width
	// controls bar length.
	underline := make([]rune, n.width)
	for i := range underline {
		underline[i] = ' '
	}
	left := int(n.pos + 0.5)
	w := int(n.underWidth + 0.5)
	if w < 1 {
		w = 1
	}
	for i := 0; i < w && left+i < n.width; i++ {
		if left+i >= 0 {
			underline[left+i] = '━'
		}
	}
	underlineStr := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Render(string(underline))

	bg := lipgloss.NewStyle().Background(lipgloss.Color(theme.ColorSurf))
	tabsLine = bg.Render(tabsLine)

	return tabsLine + "\n" + underlineStr
}

func absf(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

