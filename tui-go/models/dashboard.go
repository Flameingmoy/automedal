package models

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/cdharmaraj/automedal-tui/components"
	"github.com/cdharmaraj/automedal-tui/events"
	"github.com/cdharmaraj/automedal-tui/theme"
	"github.com/cdharmaraj/automedal-tui/util"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// DashModel — the live observation screen. Tails agent_loop.events.jsonl,
// re-reads the leaderboard periodically, and polls nvidia-smi for the
// GPU strip.  Arrow keys / Page{Up,Dn} scroll the log viewport.
type DashModel struct {
	width, height int

	state *events.State
	evCh  <-chan events.Event
	cancel context.CancelFunc

	rows []components.LeaderRow
	gpu  components.GpuSample

	logVP   viewport.Model
	logBuf  []string // all formatted log lines (ring-capped)
}

// NewDash opens the JSONL tailer and wires up periodic pollers.
func NewDash() DashModel {
	vp := viewport.New(80, 20)
	return DashModel{
		state:  events.NewState(),
		logVP:  vp,
	}
}

func (m DashModel) Init() tea.Cmd {
	ctx, cancel := context.WithCancel(context.Background())
	ch, err := events.Tail(ctx, util.EventsPath(), events.TailOpts{})
	if err != nil {
		cancel()
		// We still render; the dashboard just has no live events.
		return tea.Batch(
			tickLeaderboard(),
			tickGPU(),
		)
	}
	m.evCh = ch
	m.cancel = cancel
	return tea.Batch(
		waitForEvent(ch),
		tickLeaderboard(),
		tickGPU(),
	)
}

func waitForEvent(ch <-chan events.Event) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-ch
		if !ok {
			return nil
		}
		return EventMsg{Ev: ev}
	}
}

func tickLeaderboard() tea.Cmd {
	return tea.Tick(2*time.Second, func(time.Time) tea.Msg {
		return TickMsg{Kind: "leaderboard"}
	})
}

func tickGPU() tea.Cmd {
	return tea.Tick(1*time.Second, func(time.Time) tea.Msg {
		return TickMsg{Kind: "gpu"}
	})
}

func (m DashModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		// Left column width ~= 45% of screen; viewport gets the rest.
		m.logVP.Width = rightColumn(m.width) - 4
		m.logVP.Height = m.height - 10
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "esc":
			if m.cancel != nil {
				m.cancel()
			}
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenHome} }
		case "ctrl+c":
			if m.cancel != nil {
				m.cancel()
			}
			return m, tea.Quit
		case "k":
			return m, func() tea.Msg {
				return SwitchScreenMsg{To: ScreenKnowledge}
			}
		}
		var cmd tea.Cmd
		m.logVP, cmd = m.logVP.Update(msg)
		return m, cmd

	case EventMsg:
		m.state.Reduce(&msg.Ev)
		if line, ok := events.Format(&msg.Ev); ok {
			m.logBuf = append(m.logBuf, line)
			if len(m.logBuf) > 2000 {
				m.logBuf = m.logBuf[len(m.logBuf)-2000:]
			}
			m.logVP.SetContent(strings.Join(m.logBuf, "\n"))
			m.logVP.GotoBottom()
		}
		return m, waitForEvent(m.evCh)

	case TickMsg:
		switch msg.Kind {
		case "leaderboard":
			m.rows = components.ReadLeaderboard()
			// Push the best loss onto the sparkline series.
			if len(m.rows) > 0 {
				// We track the *best* progression: read the TSV in chronological
				// order and record the running min so the sparkline shows
				// improvement over time.
				raw := components.ReadLeaderboard()
				// ReadLeaderboard sorts ascending by loss; for the sparkline we
				// want chronological. Re-sort by timestamp.
				sortByTimestamp(raw)
				m.state.LossSeries = m.state.LossSeries[:0]
				best := 1e18
				for _, r := range raw {
					if r.ValLoss < best {
						best = r.ValLoss
					}
					m.state.PushLoss(best)
				}
			}
			return m, tickLeaderboard()
		case "gpu":
			m.gpu = components.Poll()
			return m, tickGPU()
		}
	}

	var cmd tea.Cmd
	m.logVP, cmd = m.logVP.Update(msg)
	return m, cmd
}

// sortByTimestamp bubbles rows into chronological order. Small n (≤ a few
// hundred) — linear sort is fine.
func sortByTimestamp(rows []components.LeaderRow) {
	for i := 1; i < len(rows); i++ {
		for j := i; j > 0 && rows[j-1].Timestamp > rows[j].Timestamp; j-- {
			rows[j-1], rows[j] = rows[j], rows[j-1]
		}
	}
}

// lossAdapter exposes the dashboard State in the tiny interface LossPanel
// wants. Keeps components/ independent of events/.
type lossAdapter struct{ s *events.State }

func (a lossAdapter) Series() []float64     { return a.s.LossSeries }
func (a lossAdapter) Last() (float64, bool) { return a.s.LastLoss, a.s.LastLossSet }

func (m DashModel) View() string {
	if m.width == 0 {
		return "\n  dashboard loading…\n"
	}
	leftW := leftColumn(m.width)
	rightW := rightColumn(m.width)

	status := components.StatusBar(components.StatusData{
		Brand:       "AutoMedal — dashboard",
		Competition: "",
		Phase:       m.state.Phase,
		AdvisorBusy: m.state.AdvisorBusy,
		AdvisorOn:   m.state.AdvisorModel != "",
		AdvisorModel: m.state.AdvisorModel,
	}, m.width-2)

	loss := components.LossPanel(lossAdapter{m.state}, leftW)
	gpu := components.GpuPanel(m.gpu, leftW)
	if gpu == "" {
		gpu = theme.Panel.Copy().Width(leftW - 2).Render(
			theme.Muted.Render("gpu: nvidia-smi not available"),
		)
	}
	left := lipgloss.JoinVertical(lipgloss.Left, loss, gpu)

	lb := components.Leaderboard(m.rows, 8, rightW)
	logTitle := theme.Accent.Render(
		fmt.Sprintf("live events (%s tokens in/out: %d/%d)",
			m.state.Phase, m.state.TotalInTok, m.state.TotalOutTok),
	)
	logBox := theme.Panel.Copy().Width(rightW - 2).Render(
		logTitle + "\n" + m.logVP.View(),
	)
	right := lipgloss.JoinVertical(lipgloss.Left, lb, logBox)

	body := lipgloss.JoinHorizontal(lipgloss.Top, left, right)
	footer := theme.Muted.Render("  k: knowledge  ·  q/esc: home  ·  ↑↓/PgUp/PgDn: scroll log")
	return lipgloss.JoinVertical(lipgloss.Left, status, body, footer)
}

func leftColumn(width int) int {
	w := width * 45 / 100
	if w < 30 {
		w = 30
	}
	return w
}

func rightColumn(width int) int {
	w := width - leftColumn(width)
	if w < 40 {
		w = 40
	}
	return w
}
