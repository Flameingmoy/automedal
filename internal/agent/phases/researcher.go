// Researcher phase — arxiv-driven idea injection into research_notes.md.
package phases

import (
	"context"

	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

// ResearcherArgs is the slot dict for the researcher phase.
type ResearcherArgs struct {
	ExpID              string
	Trigger            string
	Stagnating         any // bool or string
	ScheduledResearch  any // bool/int or string
	BestValLoss        any
	MaxSteps           int
}

// RunResearcher runs the researcher phase.
func RunResearcher(ctx context.Context, chat agent.ChatStreamFunc, events *agent.EventSink, a ResearcherArgs) (agent.RunReport, error) {
	base := []tools.Tool{
		tools.ReadFile, tools.WriteFile, tools.EditFile, tools.ListDir, tools.Grep,
		tools.Recall, tools.ArxivSearch,
	}
	// Subagent inherits the base toolset.
	sub := agent.MakeSubagentTool(agent.SubagentConfig{
		Chat:        chat,
		ParentTools: base,
		Events:      events,
	})
	all := append([]tools.Tool{}, base...)
	all = append(all, sub)

	scheduled := 0
	if v, ok := a.ScheduledResearch.(bool); ok && v {
		scheduled = 1
	} else if v, ok := a.ScheduledResearch.(int); ok && v != 0 {
		scheduled = 1
	}

	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 30
	}

	return RunPhase(ctx, RunPhaseConfig{
		Phase:    "researcher",
		Chat:     chat,
		Tools:    all,
		Events:   events,
		MaxSteps: maxSteps,
		Slots: map[string]any{
			"exp_id":             a.ExpID,
			"trigger":            a.Trigger,
			"stagnating":         a.Stagnating,
			"scheduled_research": scheduled,
			"best_val_loss":      a.BestValLoss,
		},
	})
}
