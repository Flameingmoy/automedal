package models

import (
	"context"
	"fmt"

	"github.com/Flameingmoy/automedal/internal/ui/components"
	"github.com/Flameingmoy/automedal/internal/ui/proc"
	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// RunModel streams `automedal <verb> <args>` into a viewport. Pressing
// `q` or `Esc` cancels the child and returns to Home.
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
}

// NewRun spawns the subprocess immediately (via Init). On spawn failure
// the first line of output is the error.
func NewRun(verb string, args []string) RunModel {
	vp := viewport.New(80, 20)
	return RunModel{verb: verb, args: args, vp: vp}
}

// spawnCmd kicks off the subprocess and returns a tea.Cmd that resolves
// to the first line (or exit) message.
type spawnedMsg struct {
	h   *proc.Handle
	err error
}

func (m RunModel) Init() tea.Cmd {
	return func() tea.Msg {
		// proc.Spawn derives its own cancellable context internally — we
		// just pass context.Background() here. Handle.Cancel() is how we
		// tear down from Update on "q"/"ctrl+c".
		h, err := proc.Spawn(context.Background(), m.verb, m.args)
		if err != nil {
			return spawnedMsg{err: err}
		}
		return spawnedMsg{h: h}
	}
}

func (m RunModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.vp.Width = msg.Width - 4
		m.vp.Height = msg.Height - 5
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "esc", "ctrl+c":
			if m.handle != nil {
				m.handle.Cancel()
			}
			return m, func() tea.Msg {
				return SwitchScreenMsg{To: ScreenHome}
			}
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
		// Start listening for lines.
		return m, readLine(m.handle)

	case RunLineMsg:
		text := msg.Line.Text
		if msg.Line.IsErr {
			text = theme.WarnStyle.Render(text)
		}
		m.appendLine(text)
		return m, readLine(m.handle)

	case RunExitMsg:
		m.done = true
		m.exit = &msg.Exit
		tag := theme.OKStyle.Render("── done ──")
		if msg.Exit.ExitCode != 0 {
			tag = theme.ErrorStyle.Render(
				fmt.Sprintf("── exit %d ──", msg.Exit.ExitCode),
			)
		}
		m.appendLine(tag + "  " + theme.Muted.Render("(press q to return)"))
		return m, nil
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

func (m RunModel) View() string {
	if m.width == 0 {
		return "spawning…"
	}
	title := theme.Accent.Render(
		fmt.Sprintf("automedal %s %s", m.verb, joinArgs(m.args)),
	) + "  " + theme.Muted.Render("(q to return)")

	body := theme.Panel.Copy().
		Width(m.width - 2).
		Height(m.height - 3).
		Render(m.vp.View())
	return lipgloss.JoinVertical(lipgloss.Left, title, body)
}

// readLine returns a tea.Cmd that reads one message off the handle.
// After the line is consumed the Update re-arms itself.
func readLine(h *proc.Handle) tea.Cmd {
	return func() tea.Msg {
		select {
		case l, ok := <-h.Lines():
			if !ok {
				// lines channel closed → exit coming imminently
				return RunExitMsg{Exit: <-h.Exit()}
			}
			return RunLineMsg{Line: l}
		case e := <-h.Exit():
			return RunExitMsg{Exit: e}
		}
	}
}

// (tiny helpers — avoid importing strings just for Join in this tight loop)
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

// Ensure viewport's placeholder isn't empty-unused.
var _ = components.ShortBanner
