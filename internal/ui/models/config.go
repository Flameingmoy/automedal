package models

import (
	"os"
	"sort"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/components"
	"github.com/Flameingmoy/automedal/internal/ui/theme"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ConfigVar — one displayed row.
type ConfigVar struct {
	Key, Default, Value string
	Set                 bool
}

var configKeys = []ConfigVar{
	{Key: "AUTOMEDAL_PROVIDER", Default: "anthropic"},
	{Key: "AUTOMEDAL_MODEL", Default: ""},
	{Key: "AUTOMEDAL_ADVISOR", Default: "0"},
	{Key: "AUTOMEDAL_ADVISOR_MODEL", Default: "kimi-k2.6"},
	{Key: "AUTOMEDAL_ANALYZER", Default: "1"},
	{Key: "AUTOMEDAL_QUICK_REJECT", Default: "0"},
	{Key: "AUTOMEDAL_DEDUPE", Default: "1"},
	{Key: "AUTOMEDAL_DEDUPE_THRESHOLD", Default: "5.0"},
	{Key: "AUTOMEDAL_REGRESSION_GATE", Default: "warn"},
	{Key: "STAGNATION_K", Default: "3"},
	{Key: "TRAIN_BUDGET_MINUTES", Default: "10"},
	{Key: "RESEARCH_EVERY", Default: "10"},
	{Key: "COOLDOWN_SECS", Default: "1"},
	{Key: "ANTHROPIC_API_KEY", Default: ""},
	{Key: "OPENAI_API_KEY", Default: ""},
	{Key: "OPENCODE_API_KEY", Default: ""},
	{Key: "KAGGLE_USERNAME", Default: ""},
}

// ConfigModel — flat scrollable env-var status table.
type ConfigModel struct {
	width, height int
	vars          []ConfigVar
}

func NewConfig() ConfigModel {
	vars := make([]ConfigVar, len(configKeys))
	for i, c := range configKeys {
		v := os.Getenv(c.Key)
		c.Value = v
		c.Set = v != ""
		// Mask API keys.
		if c.Set && strings.HasSuffix(c.Key, "_API_KEY") {
			c.Value = maskKey(v)
		}
		vars[i] = c
	}
	sort.SliceStable(vars, func(i, j int) bool {
		// set & non-default first, then defaults, then unset
		ri := rank(vars[i])
		rj := rank(vars[j])
		if ri != rj {
			return ri < rj
		}
		return vars[i].Key < vars[j].Key
	})
	return ConfigModel{vars: vars}
}

func rank(c ConfigVar) int {
	switch {
	case c.Set && c.Value != c.Default:
		return 0
	case c.Set:
		return 1
	default:
		return 2
	}
}

func maskKey(v string) string {
	if len(v) <= 6 {
		return strings.Repeat("●", len(v))
	}
	return v[:3] + strings.Repeat("●", 12)
}

func (m ConfigModel) Init() tea.Cmd { return nil }

func (m ConfigModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			return m, tea.Quit
		case "q", "esc":
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenHome} }
		}
	}
	return m, nil
}

func (m ConfigModel) View() string {
	if m.width <= 0 {
		m.width = 80
	}

	header := lipgloss.NewStyle().
		Background(lipgloss.Color(theme.ColorSurf)).
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Padding(0, 2).
		Width(m.width).
		Render("CONFIG  ·  ~/.automedal/.env  +  environment")

	hdrStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Bold(true)
	cols := hdrStyle.Render(
		"   key                              value                          status")

	var b strings.Builder
	b.WriteString(cols + "\n")
	for _, v := range m.vars {
		dot, dotColor := "○", theme.ColorMuted
		statusLabel := "default"
		statusColor := theme.ColorMuted
		valColor := theme.ColorMuted
		val := "(default: " + v.Default + ")"
		switch {
		case v.Set && v.Value != v.Default:
			dot, dotColor = "●", theme.ColorOK
			statusLabel = "set"
			statusColor = theme.ColorOK
			valColor = theme.ColorJade
			val = v.Value
		case v.Set:
			dot, dotColor = "●", theme.ColorTextDim
			statusLabel = "default"
			statusColor = theme.ColorTextDim
			valColor = theme.ColorTextDim
			val = v.Value
		default:
			if v.Default == "" {
				dot, dotColor = "✗", theme.ColorError
				statusLabel = "unset"
				statusColor = theme.ColorError
				valColor = theme.ColorError
				val = "(unset)"
			}
		}

		row := lipgloss.NewStyle().Foreground(lipgloss.Color(dotColor)).Render(dot) +
			" " +
			lipgloss.NewStyle().
				Foreground(lipgloss.Color(theme.ColorText)).
				Render(padR(v.Key, 32)) +
			" " +
			lipgloss.NewStyle().
				Foreground(lipgloss.Color(valColor)).
				Render(padR(val, 30)) +
			" " +
			lipgloss.NewStyle().
				Foreground(lipgloss.Color(statusColor)).
				Render(statusLabel)
		b.WriteString(row + "\n")
	}

	body := theme.Panel.Copy().Width(m.width - 2).Render(b.String())

	legend := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Padding(1, 2).
		Render(
			lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorOK)).Render("●") +
				" override   " +
				lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorTextDim)).Render("●") +
				" default    " +
				lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorError)).Render("✗") +
				" not set",
		)

	footer := components.FooterHints([]components.HintPair{
		{Key: "q", Desc: "home"},
		{Key: "ctrl+c", Desc: "quit"},
	}, m.width)

	return lipgloss.JoinVertical(lipgloss.Left, header, "", body, legend, footer)
}
