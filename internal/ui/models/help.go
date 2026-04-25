package models

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// HelpModel — full-screen keybindings + commands reference. Any keypress
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
	title := theme.Accent.Render("help — AutoMedal TUI")
	rows := []struct{ k, v string }{
		{"run N [--advisor MODEL]", "run N iterations of the kernel"},
		{"dashboard / dash / watch", "open the live dashboard"},
		{"knowledge / k", "view knowledge.md in a pager"},
		{"init / discover / select", "scout + competition setup"},
		{"doctor", "environment health check"},
		{"status", "one-line status"},
		{"clean / prepare / render", "pipeline helpers"},
		{"models [refresh]", "advisor model cache"},
		{"setup", "interactive provider setup"},
		{"help", "this screen"},
		{"quit / q / :q", "exit the TUI"},
	}
	var b strings.Builder
	b.WriteString(title + "\n\n")
	for _, r := range rows {
		b.WriteString("  " + theme.Accent.Render(padR(r.k, 26)) + " " + theme.Muted.Render(r.v) + "\n")
	}
	b.WriteString("\n" + theme.Muted.Render("keys:  tab=complete  ·  enter=run  ·  esc/any=back  ·  ctrl+c=quit"))
	box := theme.Panel.Copy().Width(m.width - 4).Render(b.String())
	return lipgloss.NewStyle().Padding(1).Render(box)
}

func padR(s string, w int) string {
	for len(s) < w {
		s += " "
	}
	return s
}
