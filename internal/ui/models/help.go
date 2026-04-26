package models

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/components"
	"github.com/Flameingmoy/automedal/internal/ui/theme"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// HelpModel — keybindings reference grouped by screen.  Any keypress
// returns to Home.
type HelpModel struct {
	width, height int
}

func NewHelp() HelpModel { return HelpModel{} }

func (m HelpModel) Init() tea.Cmd { return nil }

func (m HelpModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		default:
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenHome} }
		}
	}
	return m, nil
}

func (m HelpModel) View() string {
	type section struct {
		title string
		keys  [][2]string
	}
	sections := []section{
		{title: "Home", keys: [][2]string{
			{"tab", "autocomplete"},
			{"enter", "execute"},
			{"esc", "clear"},
			{"d", "dashboard"},
			{"t", "timeline"},
			{"c", "config"},
			{"k", "knowledge"},
			{"ctrl+c", "quit"},
		}},
		{title: "Dashboard", keys: [][2]string{
			{"k", "knowledge"},
			{"t", "timeline"},
			{"c", "config"},
			{"q / esc", "home"},
			{"↑↓ / PgUp/Dn", "scroll log"},
		}},
		{title: "Run", keys: [][2]string{
			{"q / esc", "interrupt & home"},
			{"↑↓", "scroll output"},
		}},
		{title: "Global", keys: [][2]string{
			{"ctrl+c", "quit"},
			{"?", "help"},
		}},
	}

	titleStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true)
	keyStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorTextDim)).
		Bold(true)
	descStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorMuted))

	var blocks []string
	for _, s := range sections {
		var b strings.Builder
		b.WriteString(titleStyle.Render(s.title) + "\n")
		for _, k := range s.keys {
			b.WriteString("  " +
				keyStyle.Render(padR(k[0], 16)) +
				descStyle.Render(k[1]) + "\n")
		}
		boxW := (m.width - 8) / 2
		if boxW < 24 {
			boxW = 24
		}
		blocks = append(blocks, theme.Panel.Copy().Width(boxW).Render(b.String()))
	}

	header := lipgloss.NewStyle().
		Background(lipgloss.Color(theme.ColorSurf)).
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Padding(0, 2).
		Width(m.width).
		Render("KEYBINDINGS")

	row1 := lipgloss.JoinHorizontal(lipgloss.Top, blocks[0], " ", blocks[1])
	row2 := lipgloss.JoinHorizontal(lipgloss.Top, blocks[2], " ", blocks[3])

	footer := components.FooterHints([]components.HintPair{
		{Key: "any key", Desc: "home"},
		{Key: "ctrl+c", Desc: "quit"},
	}, m.width)

	body := lipgloss.NewStyle().Padding(1, 2).Render(
		lipgloss.JoinVertical(lipgloss.Left, row1, "", row2),
	)
	return lipgloss.JoinVertical(lipgloss.Left, header, body, footer)
}

func padR(s string, w int) string {
	for len(s) < w {
		s += " "
	}
	return s
}
