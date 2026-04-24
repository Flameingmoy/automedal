package events

import "strings"

// State is a minimal reducer over the JSONL stream. Mirrors the shape of
// tui/state.py:AppState but only the fields the Go TUI actually reads.
type State struct {
	Phase         string // researcher|strategist|experimenter_edit|experimenter_eval|analyzer|""
	Step          int
	Iteration     int
	TotalIters    int
	LastLoss      float64
	LastLossSet   bool
	LossSeries    []float64 // one per iter_end / phase_end-with-loss; capped
	AdvisorUses   int       // successful consults (not skipped)
	AdvisorBusy   bool      // in-flight consult_advisor tool call
	AdvisorModel  string
	TotalInTok    int
	TotalOutTok   int
	PendingTool   string // name of the last tool_start (shown in status)
}

// NewState starts empty.
func NewState() *State { return &State{LossSeries: make([]float64, 0, 256)} }

// Reduce applies one event to the state. Pure: returns a *new* state but
// reuses the underlying LossSeries slice (append in place — caller owns it).
func (s *State) Reduce(e *Event) {
	switch e.Kind {
	case "phase_start":
		s.Phase = e.BasePhase()
	case "phase_end":
		s.TotalInTok += e.InTokens()
		s.TotalOutTok += e.OutTokens()
		if strings.HasPrefix(e.Phase, "experimenter_eval") || e.Phase == "experimenter_eval" {
			// A clean eval phase end is a good iteration boundary proxy
			// until iter_end lands in the schema.
		}
	case "tool_start":
		s.PendingTool = e.Name
		if e.Name == "consult_advisor" {
			s.AdvisorBusy = true
		}
	case "tool_end":
		if e.Name == "consult_advisor" {
			s.AdvisorBusy = false
		}
	case "advisor_consult":
		s.AdvisorBusy = false
		if !e.Skipped {
			s.AdvisorUses++
			if e.Model != "" {
				s.AdvisorModel = e.Model
			}
		}
	case "usage":
		s.TotalInTok += e.InTokens()
		s.TotalOutTok += e.OutTokens()
	case "step_advance":
		s.Step++
	}
}

// PushLoss records a loss point on the running series and updates LastLoss.
// Called by the dashboard when it re-reads results.tsv.
func (s *State) PushLoss(v float64) {
	s.LastLoss = v
	s.LastLossSet = true
	s.LossSeries = append(s.LossSeries, v)
	// Cap at ~1024 points; sparkline never needs more.
	if len(s.LossSeries) > 1024 {
		s.LossSeries = s.LossSeries[len(s.LossSeries)-1024:]
	}
}
