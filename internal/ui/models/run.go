package models

import (
	"context"
	"fmt"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/components"
	"github.com/Flameingmoy/automedal/internal/ui/proc"
	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// RunModel — v2 layout: full-width streaming log on the left, narrow
// side metrics panel on the right (phase / iter / best / gpu / tokens).
// Pressing q / Esc cancels the child and returns to Home.
type RunModel struct {
	verb   string
	args   []string
	vp     viewport.Model
	width  int
	height int

	handle *proc.Handle
	lines  []string
	done   bool
	exit   *proc.ExitMsg

	gpu  components.GpuSample
	spin int
}

const sidePanelWidth = 30

// NewRun spawns the subprocess immediately (via Init).
func NewRun(verb string, args []string) RunModel {
	vp := viewport.New(80, 20)
	return RunModel{verb: verb, args: args, vp: vp}
}

type spawnedMsg struct {
	h   *proc.Handle
	err error
}

func (m RunModel) Init() tea.Cmd {
	return tea.Batch(
		func() tea.Msg {
			h, err := proc.Spawn(context.Background(), m.verb, m.args)
			if err != nil {
				return spawnedMsg{err: err}
			}
			return spawnedMsg{h: h}
		},
		tickGPU(),
		tickSpin(),
	)
}

func (m RunModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.vp.Width = m.logColumn() - 4
		m.vp.Height = m.height - 5
		if m.vp.Height < 6 {
			m.vp.Height = 6
		}
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "esc", "ctrl+c":
			if m.handle != nil {
				m.handle.Cancel()
			}
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenHome} }
		}
		var cmd tea.Cmd
		m.vp, cmd = m.vp.Update(msg)
		return m, cmd

	case spawnedMsg:
		if msg.err != nil {
			m.appendLine(theme.ErrorStyle.Render("error: " + msg.err.Error()))
			m.done = true
			return m, nil
		}
		m.handle = msg.h
		return m, readLine(m.handle)

	case RunLineMsg:
		m.appendLine(colorizeLine(msg.Line.Text, msg.Line.IsErr))
		return m, readLine(m.handle)

	case RunExitMsg:
		m.done = true
		m.exit = &msg.Exit
		tag := lipgloss.NewStyle().
			Foreground(lipgloss.Color(theme.ColorOK)).
			Bold(true).
			Render("── done ──")
		if msg.Exit.ExitCode != 0 {
			tag = theme.ErrorStyle.Render(
				fmt.Sprintf("── exit %d ──", msg.Exit.ExitCode),
			)
		}
		mutedStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorMuted))
		m.appendLine(tag + "  " + mutedStyle.Render("(press q to return)"))
		return m, nil

	case TickMsg:
		switch msg.Kind {
		case "gpu":
			m.gpu = components.Poll()
			return m, tickGPU()
		case "spin":
			m.spin++
			return m, tickSpin()
		}
	}

	var cmd tea.Cmd
	m.vp, cmd = m.vp.Update(msg)
	return m, cmd
}

func (m *RunModel) appendLine(s string) {
	m.lines = append(m.lines, s)
	if len(m.lines) > 4000 {
		m.lines = m.lines[len(m.lines)-4000:]
	}
	m.vp.SetContent(stringsJoin(m.lines, "\n"))
	m.vp.GotoBottom()
}

func (m RunModel) logColumn() int {
	if m.width <= sidePanelWidth+10 {
		return m.width
	}
	return m.width - sidePanelWidth
}

func (m RunModel) View() string {
	if m.width == 0 {
		return "spawning…"
	}

	// Header strip — phase-coloured, with spinner.
	phaseStr := strings.ToUpper(orStr(m.guessPhase(), "experimenter"))
	header := lipgloss.NewStyle().
		Background(lipgloss.Color(theme.ColorSurf)).
		Foreground(lipgloss.Color(theme.ColorTextDim)).
		Padding(0, 2).
		Width(m.width).
		Render(
			components.SpinnerFrame(m.spin) + " " +
				lipgloss.NewStyle().
					Foreground(theme.PhaseColor(phaseStr)).
					Bold(true).
					Render(fmt.Sprintf("automedal %s %s",
						m.verb, joinArgs(m.args))) +
				"   " +
				lipgloss.NewStyle().
					Foreground(lipgloss.Color(theme.ColorMuted)).
					Render("q to interrupt  ·  ↑↓ scroll"),
		)

	logBox := theme.Panel.Copy().
		Width(m.logColumn() - 2).
		Height(m.height - 4).
		Render(m.vp.View())

	side := m.renderSide(sidePanelWidth - 2)

	body := lipgloss.JoinHorizontal(lipgloss.Top, logBox, side)
	return lipgloss.JoinVertical(lipgloss.Left, header, body)
}

func (m RunModel) renderSide(w int) string {
	if m.width <= sidePanelWidth+10 {
		return ""
	}
	muted := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Bold(true)
	val := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorText)).Bold(true)

	gpuPct := 0
	gpuColor := theme.ColorTextDim
	if m.gpu.OK {
		gpuPct = m.gpu.Util
		if gpuPct > 85 {
			gpuColor = theme.ColorOK
		} else {
			gpuColor = theme.ColorWarn
		}
	}

	rows := []string{
		muted.Render("PHASE"),
		lipgloss.NewStyle().
			Foreground(theme.PhaseColor(m.guessPhase())).
			Bold(true).
			Render(components.SpinnerFrame(m.spin) + " " + strings.ToUpper(m.guessPhase())),
		"",
		muted.Render("EXPERIMENTS"),
		val.Render(fmt.Sprintf("%d", len(components.ReadLeaderboard()))),
		"",
		muted.Render("GPU"),
		lipgloss.NewStyle().
			Foreground(lipgloss.Color(gpuColor)).
			Bold(true).
			Render(fmt.Sprintf("%d%%", gpuPct)),
		components.MiniBar(float64(gpuPct)/100, w-2, theme.ColorJade),
		lipgloss.NewStyle().
			Foreground(lipgloss.Color(theme.ColorMuted)).
			Render(fmt.Sprintf("%d / %d MiB", m.gpu.MemUsed, m.gpu.MemTotal)),
		"",
		muted.Render("LINES"),
		val.Render(fmt.Sprintf("%d", len(m.lines))),
	}

	body := strings.Join(rows, "\n")
	return lipgloss.NewStyle().
		Width(w).
		Background(lipgloss.Color(theme.ColorSurf)).
		Padding(1, 2).
		Render(body)
}

// guessPhase scans the most recent log lines for a phase header so the
// side panel and header colour stay in sync.
func (m RunModel) guessPhase() string {
	for i := len(m.lines) - 1; i >= 0 && i >= len(m.lines)-30; i-- {
		l := m.lines[i]
		switch {
		case strings.Contains(l, "RESEARCHER"):
			return "researcher"
		case strings.Contains(l, "STRATEGIST"):
			return "strategist"
		case strings.Contains(l, "EXPERIMENTER"):
			return "experimenter"
		case strings.Contains(l, "ANALYZER"):
			return "analyzer"
		case strings.Contains(l, "ADVISOR"):
			return "advisor"
		}
	}
	return "experimenter"
}

// colorizeLine applies a phase tint when the line is a phase header,
// or a warn tint for stderr.  Pure cosmetic — does not parse meaning.
func colorizeLine(text string, isErr bool) string {
	if isErr {
		return theme.WarnStyle.Render(text)
	}
	upper := strings.ToUpper(text)
	for _, p := range []string{"RESEARCHER", "STRATEGIST", "EXPERIMENTER",
		"ANALYZER", "ADVISOR", "TRAINING"} {
		if strings.Contains(upper, p) {
			return lipgloss.NewStyle().
				Foreground(theme.PhaseColor(strings.ToLower(p))).
				Bold(true).
				Render(text)
		}
	}
	return text
}

// readLine returns a tea.Cmd that reads one message off the handle.
func readLine(h *proc.Handle) tea.Cmd {
	return func() tea.Msg {
		select {
		case l, ok := <-h.Lines():
			if !ok {
				return RunExitMsg{Exit: <-h.Exit()}
			}
			return RunLineMsg{Line: l}
		case e := <-h.Exit():
			return RunExitMsg{Exit: e}
		}
	}
}

func stringsJoin(xs []string, sep string) string {
	if len(xs) == 0 {
		return ""
	}
	var n int
	for _, x := range xs {
		n += len(x)
	}
	n += len(sep) * (len(xs) - 1)
	out := make([]byte, 0, n)
	out = append(out, xs[0]...)
	for _, x := range xs[1:] {
		out = append(out, sep...)
		out = append(out, x...)
	}
	return string(out)
}

func joinArgs(args []string) string { return stringsJoin(args, " ") }
