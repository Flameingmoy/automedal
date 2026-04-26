package models

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Flameingmoy/automedal/internal/ui/components"
	"github.com/Flameingmoy/automedal/internal/ui/events"
	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/Flameingmoy/automedal/internal/ui/util"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// DashModel — v2 layout: stat cards row → compact loss sparkline →
// leaderboard & live-events split → GPU bar.  All sourced from
// agent/results.tsv + agent_loop.events.jsonl + nvidia-smi.
type DashModel struct {
	width, height int

	state  *events.State
	evCh   <-chan events.Event
	cancel context.CancelFunc

	rows []components.LeaderRow
	gpu  components.GpuSample

	logVP  viewport.Model
	logBuf []string
	spin   int
}

// NewDash opens the JSONL tailer and wires periodic pollers.
func NewDash() DashModel {
	vp := viewport.New(80, 12)
	return DashModel{
		state: events.NewState(),
		logVP: vp,
	}
}

func (m DashModel) Init() tea.Cmd {
	ctx, cancel := context.WithCancel(context.Background())
	ch, err := events.Tail(ctx, util.EventsPath(), events.TailOpts{})
	if err != nil {
		cancel()
		return tea.Batch(tickLeaderboard(), tickGPU(), tickSpin())
	}
	m.evCh = ch
	m.cancel = cancel
	return tea.Batch(
		waitForEvent(ch),
		tickLeaderboard(),
		tickGPU(),
		tickSpin(),
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

func tickSpin() tea.Cmd {
	return tea.Tick(120*time.Millisecond, func(time.Time) tea.Msg {
		return TickMsg{Kind: "spin"}
	})
}

func (m DashModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.logVP.Width = rightColumn(m.width) - 4
		m.logVP.Height = m.height - 18
		if m.logVP.Height < 6 {
			m.logVP.Height = 6
		}
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
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenKnowledge} }
		case "t":
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenTimeline} }
		case "c":
			return m, func() tea.Msg { return SwitchScreenMsg{To: ScreenConfig} }
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
			if len(m.rows) > 0 {
				raw := components.ReadLeaderboard()
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
		case "spin":
			m.spin++
			return m, tickSpin()
		}
	}

	var cmd tea.Cmd
	m.logVP, cmd = m.logVP.Update(msg)
	return m, cmd
}

func sortByTimestamp(rows []components.LeaderRow) {
	for i := 1; i < len(rows); i++ {
		for j := i; j > 0 && rows[j-1].Timestamp > rows[j].Timestamp; j-- {
			rows[j-1], rows[j] = rows[j], rows[j-1]
		}
	}
}

type lossAdapter struct{ s *events.State }

func (a lossAdapter) Series() []float64     { return a.s.LossSeries }
func (a lossAdapter) Last() (float64, bool) { return a.s.LastLoss, a.s.LastLossSet }

func (m DashModel) View() string {
	if m.width == 0 {
		return "\n  dashboard loading…\n"
	}

	bestLoss := 0.0
	bestLossSet := false
	bestMethod := ""
	if len(m.rows) > 0 {
		bestLoss = m.rows[0].ValLoss
		bestLossSet = true
		bestMethod = m.rows[0].Method
	}
	totalDelta := ""
	if len(m.rows) >= 2 {
		base := m.rows[len(m.rows)-1].ValLoss
		if base > 0 {
			pct := (m.rows[0].ValLoss - base) / base * 100
			totalDelta = fmt.Sprintf("%+.1f%% from baseline", pct)
		}
	}

	status := components.StatusBar(components.StatusData{
		Brand:        "AutoMedal — dashboard",
		Phase:        m.state.Phase,
		BestLoss:     bestLoss,
		BestLossSet:  bestLossSet,
		AdvisorBusy:  m.state.AdvisorBusy,
		AdvisorOn:    m.state.AdvisorModel != "",
		AdvisorModel: m.state.AdvisorModel,
		SpinIdx:      m.spin,
	}, m.width)

	// Stat cards row.
	cardW := (m.width - 14) / 5
	if cardW < 14 {
		cardW = 14
	}
	phaseStr := strings.ToUpper(orStr(m.state.Phase, "idle"))
	phaseColor := string(theme.PhaseColor(m.state.Phase))
	if phaseStr == "IDLE" {
		phaseColor = theme.ColorTextDim
	}
	iterVal := fmt.Sprintf("%d", len(m.rows))
	bestStr := "—"
	bestColor := theme.ColorTextDim
	if bestLossSet {
		bestStr = fmt.Sprintf("%.4f", bestLoss)
		bestColor = theme.ColorOK
	}
	deltaStr := "—"
	deltaColor := theme.ColorTextDim
	if totalDelta != "" {
		deltaStr = totalDelta
		deltaColor = theme.ColorOK
	}
	gpuVal := "—"
	gpuColor := theme.ColorTextDim
	if m.gpu.OK {
		gpuVal = fmt.Sprintf("%d%%", m.gpu.Util)
		if m.gpu.Util > 85 {
			gpuColor = theme.ColorOK
		} else {
			gpuColor = theme.ColorWarn
		}
	}
	advisorVal := "off"
	advisorColor := theme.ColorTextDim
	if m.state.AdvisorModel != "" {
		advisorVal = m.state.AdvisorModel
		advisorColor = theme.ColorAdvisor
	}

	cards := lipgloss.JoinHorizontal(lipgloss.Top,
		components.StatCard("Phase", phaseStr, "● running", phaseColor, cardW),
		" ",
		components.StatCard("Experiments", iterVal, "tracked", theme.ColorText, cardW),
		" ",
		components.StatCard("Best Loss", bestStr, bestMethod, bestColor, cardW),
		" ",
		components.StatCard("Δ Total", deltaStr, "from baseline", deltaColor, cardW),
		" ",
		components.StatCard("GPU Util", gpuVal, gpuName(m.gpu), gpuColor, cardW),
		" ",
		components.StatCard("Advisor", advisorVal, advisorSub(m.state), advisorColor, cardW),
	)

	// Compact loss sparkline.
	lossPanel := components.LossPanel(lossAdapter{m.state}, m.width-4)

	// Leaderboard | events split.
	leftW := m.width * 40 / 100
	if leftW < 30 {
		leftW = 30
	}
	rightW := m.width - leftW - 4

	leader := components.Leaderboard(m.rows, 7, leftW)
	logTitle := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorJade)).
		Bold(true).
		Render("LIVE EVENTS")
	logSub := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Render(fmt.Sprintf("  %s · in %d / out %d tokens",
			strings.ToLower(orStr(m.state.Phase, "idle")),
			m.state.TotalInTok, m.state.TotalOutTok))
	logBox := theme.Panel.Copy().Width(rightW - 2).Render(
		logTitle + logSub + "\n" + m.logVP.View(),
	)
	body := lipgloss.JoinHorizontal(lipgloss.Top, leader, " ", logBox)

	// GPU strip.
	gpu := components.GpuPanel(m.gpu, m.width-4)
	if gpu == "" {
		gpu = theme.Panel.Copy().Width(m.width - 4).Render(
			lipgloss.NewStyle().
				Foreground(lipgloss.Color(theme.ColorMuted)).
				Render("gpu: nvidia-smi not available"),
		)
	}

	footer := components.FooterHints([]components.HintPair{
		{Key: "k", Desc: "knowledge"},
		{Key: "t", Desc: "timeline"},
		{Key: "c", Desc: "config"},
		{Key: "q", Desc: "home"},
		{Key: "↑↓", Desc: "scroll"},
		{Key: "ctrl+c", Desc: "quit"},
	}, m.width)

	return lipgloss.JoinVertical(lipgloss.Left,
		status,
		"",
		cards,
		"",
		lossPanel,
		"",
		body,
		"",
		gpu,
		footer,
	)
}

func gpuName(g components.GpuSample) string {
	if !g.OK {
		return "(unavailable)"
	}
	return g.Name
}

func advisorSub(s *events.State) string {
	if s.AdvisorModel == "" {
		return "no consults"
	}
	return fmt.Sprintf("%d in / %d out", s.TotalInTok, s.TotalOutTok)
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
