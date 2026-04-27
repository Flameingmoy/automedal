package models

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
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

	// tail metadata — surfaced in the empty-state.
	eventsPath   string
	eventsSource string // "go.mod" | "AUTOMEDAL_CWD" | "artefacts" | "cwd"
	tailOK       bool

	rows []components.LeaderRow
	gpu  components.GpuSample

	logVP    viewport.Model
	logRows  []components.EventRow
	focusIdx int // index into logRows; -1 = no focus
	follow   bool
	spin     int
}

const maxLogRows = 500

// NewDash opens the JSONL tailer and wires periodic pollers.  Tail
// setup runs here (not Init) so the resolved path + connection bool
// can live on the value receiver — Init() can't mutate the model.
func NewDash() DashModel {
	vp := viewport.New(80, 12)

	root, source := util.RepoRootResolved()
	path := root + string(os.PathSeparator) + "agent_loop.events.jsonl"

	ctx, cancel := context.WithCancel(context.Background())
	ch, err := events.Tail(ctx, path, events.TailOpts{})
	if err != nil {
		cancel()
		ch = nil
	}

	return DashModel{
		state:        events.NewState(),
		logVP:        vp,
		focusIdx:     -1,
		follow:       true,
		evCh:         ch,
		cancel:       cancel,
		eventsPath:   path,
		eventsSource: source,
		tailOK:       err == nil,
	}
}

func (m DashModel) Init() tea.Cmd {
	cmds := []tea.Cmd{tickLeaderboard(), tickGPU(), tickSpin()}
	if m.evCh != nil {
		cmds = append(cmds, waitForEvent(m.evCh))
	}
	return tea.Batch(cmds...)
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
		// Re-render — EventItem layout depends on width.
		m.refreshLog()
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
		case "[":
			m.moveFocus(-1)
			m.refreshLog()
			return m, nil
		case "]":
			m.moveFocus(+1)
			m.refreshLog()
			return m, nil
		case " ", "enter":
			if m.focusIdx >= 0 && m.focusIdx < len(m.logRows) {
				m.logRows[m.focusIdx].Expanded = !m.logRows[m.focusIdx].Expanded
				m.refreshLog()
			}
			return m, nil
		case "G", "end":
			m.focusIdx = len(m.logRows) - 1
			m.follow = true
			m.refreshLog()
			return m, nil
		}
		var cmd tea.Cmd
		m.logVP, cmd = m.logVP.Update(msg)
		// Any explicit scroll input breaks follow-mode so the user
		// can keep reading older events as new ones stream in.
		if !m.logVP.AtBottom() {
			m.follow = false
		}
		return m, cmd

	case EventMsg:
		m.state.Reduce(&msg.Ev)
		if row, ok := eventToRow(&msg.Ev); ok {
			m.logRows = append(m.logRows, row)
			if len(m.logRows) > maxLogRows {
				drop := len(m.logRows) - maxLogRows
				m.logRows = m.logRows[drop:]
				if m.focusIdx >= 0 {
					m.focusIdx -= drop
					if m.focusIdx < 0 {
						m.focusIdx = -1
					}
				}
			}
			if m.follow {
				m.focusIdx = len(m.logRows) - 1
			}
			m.refreshLog()
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

// moveFocus shifts focusIdx by `delta`, clamping to the log bounds.
// Stepping past the newest row re-enables follow-mode so the next
// arriving event keeps the focus pinned to the bottom.
func (m *DashModel) moveFocus(delta int) {
	if len(m.logRows) == 0 {
		m.focusIdx = -1
		return
	}
	if m.focusIdx < 0 {
		m.focusIdx = len(m.logRows) - 1
		return
	}
	next := m.focusIdx + delta
	if next < 0 {
		next = 0
	}
	if next >= len(m.logRows) {
		next = len(m.logRows) - 1
	}
	m.focusIdx = next
	m.follow = m.focusIdx == len(m.logRows)-1
}

// refreshLog rebuilds the viewport content from logRows + focus state.
// Cheap enough to call on every key/event — the cap is 500 rows.
func (m *DashModel) refreshLog() {
	if len(m.logRows) == 0 {
		m.logVP.SetContent("")
		return
	}
	width := m.logVP.Width
	if width <= 0 {
		width = 60
	}
	parts := make([]string, len(m.logRows))
	for i, r := range m.logRows {
		r.Focused = i == m.focusIdx
		parts[i] = components.EventItem(r, width)
	}
	m.logVP.SetContent(strings.Join(parts, "\n"))
	if m.follow {
		m.logVP.GotoBottom()
	}
}

// eventToRow converts a JSONL event into the EventRow shape the
// dashboard renders.  Returns ok=false for events the user shouldn't
// see (successful tool_end with no error, empty deltas, raw usage).
func eventToRow(e *events.Event) (components.EventRow, bool) {
	ts := shortTime(e.T)
	switch e.Kind {
	case "phase_start":
		return components.EventRow{
			Type:    "phase",
			Icon:    "◈",
			TS:      ts,
			Label:   "phase_start",
			Summary: "→ " + strings.ToUpper(e.BasePhase()),
		}, true
	case "phase_end":
		usage := ""
		if in := e.InTokens(); in > 0 || e.OutTokens() > 0 {
			usage = fmt.Sprintf("  ·  %d in / %d out", in, e.OutTokens())
		}
		return components.EventRow{
			Type:    "phase",
			Icon:    "◈",
			TS:      ts,
			Label:   "phase_end",
			Summary: strings.ToUpper(e.BasePhase()) + " (" + e.Stop + ")" + usage,
			Detail:  fmt.Sprintf("phase: %s\nstop:  %s\nstep:  %d\n%s", e.Phase, e.Stop, e.Step, usage),
		}, true
	case "tool_start":
		args := formatArgsCompact(e.Args)
		return components.EventRow{
			Type:    "tool",
			Icon:    "◇",
			TS:      ts,
			Label:   "tool_call: " + e.Name,
			Summary: args,
			Detail:  detailFromArgs(e.Name, e.Args),
		}, true
	case "tool_end":
		if e.OK {
			return components.EventRow{}, false
		}
		return components.EventRow{
			Type:    "tool",
			Icon:    "✗",
			TS:      ts,
			Label:   "tool_error: " + e.Name,
			Summary: oneLine(e.Preview),
			Detail:  e.Preview,
		}, true
	case "delta":
		txt := strings.TrimSpace(e.Text)
		if txt == "" {
			return components.EventRow{}, false
		}
		return components.EventRow{
			Type:    "tool",
			Icon:    "·",
			TS:      ts,
			Label:   "delta",
			Summary: oneLine(clipS(txt, 200)),
			Detail:  txt,
		}, true
	case "thinking":
		return components.EventRow{
			Type:    "tool",
			Icon:    "✻",
			TS:      ts,
			Label:   "thinking",
			Summary: fmt.Sprintf("%d chars", len(e.Text)),
			Detail:  e.Text,
		}, true
	case "subagent_start":
		return components.EventRow{
			Type:    "tool",
			Icon:    "▷",
			TS:      ts,
			Label:   "subagent: " + e.Label,
			Summary: oneLine(clipS(e.Prompt, 100)),
			Detail:  e.Prompt,
		}, true
	case "subagent_end":
		return components.EventRow{
			Type:    "tool",
			Icon:    "◁",
			TS:      ts,
			Label:   "subagent_end: " + e.Label,
			Summary: fmt.Sprintf("ok=%v", e.OK),
		}, true
	case "advisor_consult":
		if e.Skipped {
			r := e.Reason
			if r == "" {
				r = "no_reason"
			}
			return components.EventRow{
				Type:    "advisor",
				Icon:    "◆",
				TS:      ts,
				Label:   "advisor: " + e.Purpose,
				Summary: "skipped (" + r + ")",
			}, true
		}
		return components.EventRow{
			Type: "advisor",
			Icon: "◆",
			TS:   ts,
			Label: "advisor: " + e.Purpose,
			Summary: fmt.Sprintf("%s  ·  %d in / %d out",
				e.Model, e.InTokens(), e.OutTokens()),
			Detail: "model:    " + e.Model + "\n" +
				"purpose:  " + e.Purpose + "\n" +
				"tokens:   " + fmt.Sprintf("%d in / %d out", e.InTokens(), e.OutTokens()) +
				"\n\n" + e.Preview,
		}, true
	case "error":
		return components.EventRow{
			Type:    "tool",
			Icon:    "✗",
			TS:      ts,
			Label:   "error: " + e.Where,
			Summary: oneLine(clipS(e.Msg, 120)),
			Detail:  e.Type + "\n\n" + e.Msg,
		}, true
	case "notice":
		evType := "tool"
		if strings.HasPrefix(e.Tag, "training") {
			evType = "training"
		}
		return components.EventRow{
			Type:    evType,
			Icon:    "▸",
			TS:      ts,
			Label:   e.Tag,
			Summary: oneLine(clipS(e.Message, 200)),
			Detail:  e.Message,
		}, true
	case "usage":
		return components.EventRow{}, false
	}
	return components.EventRow{}, false
}

// formatArgsCompact renders {"path":"x","limit":80} → path=x, limit=80.
func formatArgsCompact(raw []byte) string {
	if len(raw) == 0 {
		return ""
	}
	// Reuse the events package's plain renderer via a tiny re-impl —
	// avoids a dependency loop.
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return clipS(string(raw), 80)
	}
	parts := make([]string, 0, len(m))
	for k, v := range m {
		sv := strings.ReplaceAll(fmt.Sprintf("%v", v), "\n", " ")
		parts = append(parts, k+"="+clipS(sv, 60))
	}
	return strings.Join(parts, ", ")
}

// detailFromArgs pretty-prints args one-per-line for the expanded view.
func detailFromArgs(name string, raw []byte) string {
	if len(raw) == 0 {
		return name + "()"
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return name + "(" + string(raw) + ")"
	}
	var b strings.Builder
	b.WriteString(name + "(\n")
	for k, v := range m {
		sv := fmt.Sprintf("%v", v)
		b.WriteString("  " + k + " = " + sv + "\n")
	}
	b.WriteString(")")
	return b.String()
}

func oneLine(s string) string { return strings.ReplaceAll(s, "\n", " ") }

func clipS(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

func shortTime(iso string) string {
	if iso == "" {
		return "--:--:--"
	}
	t, err := time.Parse("2006-01-02T15:04:05Z", iso)
	if err != nil {
		return "--:--:--"
	}
	return t.Local().Format("15:04:05")
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
	if m.width <= 0 {
		m.width = 100
	}
	if m.height <= 0 {
		m.height = 30
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

	// Stat cards row — 4 cards keeps each one ≥ 22 cols on a 100-col
	// terminal.  Δ Total + Advisor were dropped because both already
	// surface in the status bar above.
	cardW := (m.width - 10) / 4
	if cardW < 18 {
		cardW = 18
	}
	phaseStr := strings.ToUpper(orStr(m.state.Phase, "idle"))
	phaseColor := string(theme.PhaseColor(m.state.Phase))
	phaseSub := "● running"
	if phaseStr == "IDLE" {
		phaseColor = theme.ColorTextDim
		phaseSub = "○ waiting"
	}
	iterVal := fmt.Sprintf("%d", len(m.rows))
	iterSub := "tracked"
	if totalDelta != "" {
		iterSub = totalDelta
	}
	bestStr := "—"
	bestColor := theme.ColorTextDim
	if bestLossSet {
		bestStr = fmt.Sprintf("%.4f", bestLoss)
		bestColor = theme.ColorOK
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

	cards := lipgloss.JoinHorizontal(lipgloss.Top,
		components.StatCard("Phase", phaseStr, phaseSub, phaseColor, cardW),
		" ",
		components.StatCard("Experiments", iterVal, iterSub, theme.ColorText, cardW),
		" ",
		components.StatCard("Best Loss", bestStr, bestMethod, bestColor, cardW),
		" ",
		components.StatCard("GPU Util", gpuVal, gpuName(m.gpu), gpuColor, cardW),
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

	// "tail ●" chip — jade dot when fsnotify is connected, dim ring
	// when we couldn't open the watcher.
	chipDot := "○"
	chipColor := theme.ColorTextDim
	if m.tailOK {
		chipDot = "●"
		chipColor = theme.ColorJade
	}
	chip := lipgloss.NewStyle().
		Foreground(lipgloss.Color(chipColor)).
		Render("  " + chipDot + " tail")

	focusInfo := ""
	if m.focusIdx >= 0 && m.focusIdx < len(m.logRows) {
		focusInfo = fmt.Sprintf("  ·  %d/%d", m.focusIdx+1, len(m.logRows))
	}
	logSub := lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorMuted)).
		Render(fmt.Sprintf("  %s · %d in / %d out%s",
			strings.ToLower(orStr(m.state.Phase, "idle")),
			m.state.TotalInTok, m.state.TotalOutTok, focusInfo))

	var logBody string
	if len(m.logRows) == 0 {
		logBody = m.emptyEventsState(rightW - 4)
	} else {
		logBody = m.logVP.View()
	}
	logBox := theme.Panel.Copy().Width(rightW - 2).Render(
		logTitle + chip + logSub + "\n" + logBody,
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
		{Key: "[ ]", Desc: "focus"},
		{Key: "space", Desc: "expand"},
		{Key: "G", Desc: "follow"},
		{Key: "k", Desc: "knowledge"},
		{Key: "t", Desc: "timeline"},
		{Key: "c", Desc: "config"},
		{Key: "q", Desc: "home"},
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

// emptyEventsState renders the LIVE EVENTS panel body when nothing has
// streamed yet — explains where we're looking and what we're waiting
// for, so a blank panel never looks broken.
func (m DashModel) emptyEventsState(width int) string {
	if width < 20 {
		width = 20
	}
	mutedDim := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorTextDim))
	muted := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorMuted))
	jade := lipgloss.NewStyle().Foreground(lipgloss.Color(theme.ColorJade))

	icon := "○"
	statusLine := "waiting for events…"
	hint := "no agent_loop.events.jsonl yet"
	if m.tailOK {
		icon = "●"
		statusLine = "tail connected — no events emitted yet"
		hint = "run `automedal run N` to populate this stream"
	}

	pathDisplay := m.eventsPath
	if len(pathDisplay) > width-4 {
		pathDisplay = "…" + pathDisplay[len(pathDisplay)-(width-5):]
	}

	rows := []string{
		"",
		"  " + jade.Render(icon) + " " + mutedDim.Render(statusLine),
		"",
		"  " + muted.Render("path   ") + mutedDim.Render(pathDisplay),
		"  " + muted.Render("source ") + mutedDim.Render(m.eventsSource),
		"",
		"  " + muted.Render(hint),
	}
	return strings.Join(rows, "\n")
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
