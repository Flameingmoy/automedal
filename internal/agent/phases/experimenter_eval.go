// Experimenter-eval phase — parse training result, write journal, commit/revert.
package phases

import (
	"context"

	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

// ExperimenterEvalArgs is the slot dict for the experimenter-eval phase.
type ExperimenterEvalArgs struct {
	ExpID       string
	BestValLoss any
	TrainRC     int
	FinalLoss   any
	MaxSteps    int
}

// RunExperimenterEval runs the experimenter-eval phase.
func RunExperimenterEval(ctx context.Context, chat agent.ChatStreamFunc, events *agent.EventSink, a ExperimenterEvalArgs) (agent.RunReport, error) {
	base := []tools.Tool{
		tools.ReadFile, tools.WriteFile, tools.EditFile, tools.ListDir, tools.Grep, tools.RunShell,
	}
	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 30
	}
	return RunPhase(ctx, RunPhaseConfig{
		Phase:    "experimenter_eval",
		Chat:     chat,
		Tools:    base,
		Events:   events,
		MaxSteps: maxSteps,
		Slots: map[string]any{
			"exp_id":        a.ExpID,
			"best_val_loss": a.BestValLoss,
			"train_rc":      a.TrainRC,
			"final_loss":    a.FinalLoss,
		},
	})
}
