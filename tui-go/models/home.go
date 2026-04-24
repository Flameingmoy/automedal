package models

import (
	"strings"

	"github.com/cdharmaraj/automedal-tui/components"
	"github.com/cdharmaraj/automedal-tui/theme"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// HomeModel is the landing screen — logo, status strip, recent activity,
// command palette.  Mirrors tui/screens/home.py visually.
type HomeModel struct {
	input  textinput.Model
	width  int
	height int

	brand       string
	competition string
	status      components.StatusData
}

// NewHome returns a fresh HomeModel ready to Init().
func NewHome() HomeModel {
	ti := textinput.New()
	ti.Placeholder = "type a command (e.g. run 30)"
	ti.Prompt = "" // we draw our own ">" prompt outside the input
	ti.CharLimit = 512
	ti.Focus()

	return HomeModel{
		input: ti,
		brand: "AutoMedal",
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
			// Autocomplete on Tab — fix lands behavior we just shipped in
			// command_input.py: single match appends trailing space.
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

// dispatch decides what to do with the typed text. Quit aliases MUST NOT
// spawn a subprocess (the old Python bug — see tui/screens/home.py:102
// and docs/plans/go-tui-migration.md).
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

	// Clear the input immediately for UX.
	m.input.SetValue("")

	switch cmd {
	case "help":
		return m, func() tea.Msg {
			return SwitchScreenMsg{To: ScreenHelp}
		}
	case "dashboard", "dash", "watch":
		return m, func() tea.Msg {
			return SwitchScreenMsg{To: ScreenDash}
		}
	case "knowledge", "k":
		return m, func() tea.Msg {
			return SwitchScreenMsg{To: ScreenKnowledge}
		}
	default:
		// Everything else shells out to `automedal <cmd> <args>` via a
		// Run screen.  The old Python TUI did the same, but we do it
		// without the 1.4-second Python import tax every time.
		return m, func() tea.Msg {
			return SwitchScreenMsg{To: ScreenRun, Verb: cmd, Args: args}
		}
	}
}

// UpdateStatus lets main.go push an external StatusData in (advisor
// state, phase from the JSONL tailer, etc.).
func (m *HomeModel) UpdateStatus(s components.StatusData) {
	m.status = s
}

func (m HomeModel) View() string {
	if m.width <= 0 {
		return "\n  loading…\n"
	}
	logoW := m.width - 2
	logo := components.Logo(logoW)

	statusLine := components.StatusBar(m.status, m.width-2)

	recent := components.RecentPanel(5, m.width)

	prompt := theme.Prompt.Render("> ") + m.input.View()
	hints := components.HintLine(m.input.Value())
	promptBox := theme.Panel.Copy().
		Width(m.width - 2).
		Render(prompt + "\n" + hints)

	footer := theme.Muted.Render("  tab: complete  ·  enter: run  ·  esc: clear  ·  ctrl+c: quit")

	content := lipgloss.JoinVertical(lipgloss.Left,
		logo,
		statusLine,
		recent,
		promptBox,
		footer,
	)
	return content
}
