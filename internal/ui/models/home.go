package models

import (
	"fmt"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/components"
	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// HomeModel — landing screen.  Hermes-style gradient banner left, live
// info table right, command palette below.  Mirrors AutoMedal TUI v2.
type HomeModel struct {
	input  textinput.Model
	width  int
	height int

	competition string
	status      components.StatusData
}

// NewHome returns a fresh HomeModel ready to Init().
func NewHome() HomeModel {
	ti := textinput.New()
	ti.Placeholder = "type a command  ·  tab to autocomplete"
	ti.Prompt = ""
	ti.CharLimit = 512
	ti.Focus()

	return HomeModel{
		input: ti,
		status: components.StatusData{
			Brand: "AutoMedal",
		},
	}
}

func (m HomeModel) Init() tea.Cmd {
	return textinput.Blink
}

func (m HomeModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		return m, nil

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyCtrlQ:
			return m, tea.Quit

		case tea.KeyTab:
			completed, ok := components.Autocomplete(m.input.Value())
			if ok {
				m.input.SetValue(completed)
				m.input.CursorEnd()
			}
			return m, nil

		case tea.KeyEnter:
			return m.dispatch(m.input.Value())

		case tea.KeyEsc:
			m.input.SetValue("")
			return m, nil
		}
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

// dispatch maps typed text to a screen swap or subprocess spawn.
func (m HomeModel) dispatch(raw string) (tea.Model, tea.Cmd) {
	text := components.Normalize(raw)
	if text == "" {
		return m, nil
	}
	if components.IsQuit(text) {
		return m, tea.Quit
	}

	parts := strings.Fields(text)
	cmd := strings.ToLower(parts[0])
	args := parts[1:]

	m.input.SetValue("")

	switch cmd {
	case "help":
		return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenHelp} }
	case "dashboard", "dash", "watch":
		return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenDash} }
	case "knowledge", "k":
		return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenKnowledge} }
	case "timeline", "tl":
		return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenTimeline} }
	case "config", "cfg":
		return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenConfig} }
	default:
		return m, func() tea.Msg {
			return SwitchScreenMsg{To: ScreenRun, Verb: cmd, Args: args}
		}
	}
}

// UpdateStatus lets main.go push live data in (advisor state, phase
// from JSONL, etc.).
func (m *HomeModel) UpdateStatus(s components.StatusData) {
	m.status = s
}

func (m HomeModel) View() string {
	if m.width <= 0 {
		return "\n  loading…\n"
	}

	// Banner — chunky pixel font, blue→cyan→jade gradient.
	banner := components.Banner("AUTOMEDAL")
	bannerW := components.BannerWidth("AUTOMEDAL")

	tagline := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorTextDim)).
		Italic(true).
		Render("autonomous ML research loop  ·  jade always")

	infoRows := m.buildInfo()
	info := components.InfoTable(infoRows, 14)

	// Decide layout: side-by-side if there's room, else stack.
	innerW := m.width - 4
	infoW := lipgloss.Width(info)
	bannerBlock := lipgloss.JoinVertical(lipgloss.Left, banner, "", tagline)

	var hero string
	if innerW >= bannerW+infoW+6 {
		gap := strings.Repeat(" ", 6)
		infoBlock := lipgloss.JoinVertical(lipgloss.Left,
			components.Separator(infoW, theme.ColorJade),
			info,
			"",
			components.Separator(infoW, theme.ColorMuted),
			components.PhaseSwatches(),
		)
		hero = lipgloss.JoinHorizontal(lipgloss.Top, bannerBlock, gap, infoBlock)
	} else {
		hero = lipgloss.JoinVertical(lipgloss.Left,
			bannerBlock,
			"",
			components.Separator(min(innerW, 60), theme.ColorJade),
			info,
			"",
			components.PhaseSwatches(),
		)
	}

	// Command palette.
	prompt := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Render("▸ ")
	hints := components.HintLine(m.input.Value())
	paletteBody := prompt + m.input.View() + "\n" + hints
	palette := theme.PanelHi.Copy().
		Width(m.width - 2).
		Render(paletteBody)

	footer := components.FooterHints([]components.HintPair{
		{Key: "tab", Desc: "complete"},
		{Key: "enter", Desc: "run"},
		{Key: "d", Desc: "dashboard"},
		{Key: "t", Desc: "timeline"},
		{Key: "c", Desc: "config"},
		{Key: "k", Desc: "knowledge"},
		{Key: "ctrl+c", Desc: "quit"},
	}, m.width)

	heroBox := lipgloss.NewStyle().Padding(1, 2).Render(hero)

	available := m.height - lipgloss.Height(heroBox) - lipgloss.Height(palette) - lipgloss.Height(footer)
	pad := ""
	if available > 0 {
		pad = strings.Repeat("\n", available)
	}

	return lipgloss.JoinVertical(lipgloss.Left,
		heroBox,
		pad,
		palette,
		footer,
	)
}

func (m HomeModel) buildInfo() []components.InfoRow {
	s := m.status

	phaseVal := "⣾ " + strings.ToUpper(orStr(s.Phase, "idle"))
	phaseColor := theme.ColorTextDim
	if s.Phase != "" && strings.ToUpper(s.Phase) != "IDLE" {
		phaseColor = colorHex(theme.PhaseColor(s.Phase))
	}

	iter := "—"
	if s.Iteration > 0 {
		if s.TotalIters > 0 {
			iter = fmt.Sprintf("%d / %d", s.Iteration, s.TotalIters)
		} else {
			iter = fmt.Sprintf("%d", s.Iteration)
		}
	}

	best := "—"
	bestColor := theme.ColorTextDim
	if s.BestLossSet {
		best = fmt.Sprintf("%.4f", s.BestLoss)
		bestColor = theme.ColorOK
	}

	advisor := "off"
	advisorColor := theme.ColorTextDim
	if s.AdvisorOn {
		advisor = "◆ " + s.AdvisorModel + "  ON"
		advisorColor = theme.ColorAdvisor
	}

	comp := orStr(s.Competition, "(no competition)")
	compColor := theme.ColorJade
	if s.Competition == "" {
		compColor = theme.ColorMuted
	}

	rows := []components.InfoRow{
		{Key: "competition", Value: comp, Color: compColor},
		{Key: "phase", Value: phaseVal, Color: phaseColor},
		{Key: "iter", Value: iter, Color: theme.ColorTextDim},
		{Key: "best_loss", Value: best, Color: bestColor},
		{Key: "advisor", Value: advisor, Color: advisorColor},
	}

	// Recent activity tail.
	if recents := components.ReadRecent(2); len(recents) > 0 {
		var slugs []string
		for _, r := range recents {
			slug := r.Slug
			if len(slug) > 22 {
				slug = slug[:21] + "…"
			}
			slugs = append(slugs, "#"+r.ID+" "+slug)
		}
		rows = append(rows, components.InfoRow{
			Key:   "recent",
			Value: strings.Join(slugs, "  "),
			Color: theme.ColorTextDim,
		})
	}

	return rows
}

func orStr(v, def string) string {
	if strings.TrimSpace(v) == "" {
		return def
	}
	return v
}

func colorHex(c lipgloss.Color) string { return string(c) }

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
