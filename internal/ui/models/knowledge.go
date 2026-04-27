package models

import (
	"os"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/Flameingmoy/automedal/internal/ui/util"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
)

// KnowledgeModel renders knowledge.md through glamour. Scroll with arrows,
// q/esc back to home.
type KnowledgeModel struct {
	width, height int
	vp            viewport.Model
	loaded        bool
}

func NewKnowledge() KnowledgeModel {
	return KnowledgeModel{vp: viewport.New(80, 20)}
}

func (m KnowledgeModel) Init() tea.Cmd {
	return nil
}

func (m KnowledgeModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.vp.Width = msg.Width - 4
		m.vp.Height = msg.Height - 4
		if !m.loaded {
			m.loadContent()
		}
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "esc":
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenHome} }
		case "ctrl+c":
			return m, tea.Quit
		}
		var cmd tea.Cmd
		m.vp, cmd = m.vp.Update(msg)
		return m, cmd
	}
	var cmd tea.Cmd
	m.vp, cmd = m.vp.Update(msg)
	return m, cmd
}

func (m *KnowledgeModel) loadContent() {
	b, err := os.ReadFile(util.KnowledgePath())
	if err != nil {
		m.vp.SetContent(theme.Muted.Render("(knowledge.md not found — run `automedal run` to populate)"))
		m.loaded = true
		return
	}
	width := m.vp.Width
	if width <= 0 {
		width = 80
	}
	rendered, err := glamour.Render(string(b), "dracula")
	if err != nil {
		rendered = string(b)
	}
	_ = width
	m.vp.SetContent(rendered)
	m.loaded = true
}

func (m KnowledgeModel) View() string {
	if m.width <= 0 {
		m.width = 80
	}
	if m.height <= 0 {
		m.height = 24
	}
	title := theme.Accent.Render("knowledge.md") + "  " +
		theme.Muted.Render("(q to return)")
	box := theme.Panel.Copy().
		Width(m.width - 2).
		Height(m.height - 2).
		Render(m.vp.View())
	return lipgloss.JoinVertical(lipgloss.Left, title, box)
}
