// Experimenter-edit phase — pop top queue entry and edit agent/train.py.
package phases

import (
	"context"

	"github.com/Flameingmoy/automedal/internal/advisor"
	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

// ExperimenterEditArgs is the slot dict for the experimenter-edit phase.
type ExperimenterEditArgs struct {
	ExpID       string
	BestValLoss any
	Retry       bool
	PrevLoss    any
	MaxSteps    int
	ConsultFunc tools.ConsultFunc
}

// RunExperimenterEdit runs the experimenter-edit phase.
func RunExperimenterEdit(ctx context.Context, chat agent.ChatStreamFunc, events *agent.EventSink, a ExperimenterEditArgs) (agent.RunReport, error) {
	base := []tools.Tool{
		tools.ReadFile, tools.WriteFile, tools.EditFile, tools.ListDir, tools.Grep,
		tools.Recall, tools.RunShell,
	}

	extraFactory := func(sink *agent.EventSink) []tools.Tool {
		if advisor.IsEnabled("tool") && a.ConsultFunc != nil {
			return []tools.Tool{tools.MakeAdvisorTool(a.ConsultFunc, 1)}
		}
		return nil
	}

	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 50
	}

	return RunPhase(ctx, RunPhaseConfig{
		Phase:    "experimenter",
		Chat:     chat,
		Tools:    base,
		Events:   events,
		MaxSteps: maxSteps,
		Slots: map[string]any{
			"exp_id":        a.ExpID,
			"best_val_loss": a.BestValLoss,
			"retry":         a.Retry,
			"prev_loss":     a.PrevLoss,
		},
		ExtraToolsFactory: extraFactory,
	})
}
