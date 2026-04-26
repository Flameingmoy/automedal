// Analyzer phase — compress one iteration's signal into knowledge.md.
package phases

import (
	"context"

	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

// AnalyzerArgs is the slot dict for the analyzer phase.
type AnalyzerArgs struct {
	ExpID         string
	Slug          string
	Status        string
	FinalLoss     any
	BestValLoss   any
	ValLossDelta  any
	MaxSteps      int
}

// RunAnalyzer runs the analyzer phase. No write_file — analyzer must
// edit knowledge.md in place.
func RunAnalyzer(ctx context.Context, chat agent.ChatStreamFunc, events *agent.EventSink, a AnalyzerArgs) (agent.RunReport, error) {
	base := []tools.Tool{
		tools.ReadFile, tools.EditFile, tools.ListDir, tools.Grep, tools.Recall,
	}
	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 20
	}
	return RunPhase(ctx, RunPhaseConfig{
		Phase:    "analyzer",
		Chat:     chat,
		Tools:    base,
		Events:   events,
		MaxSteps: maxSteps,
		Slots: map[string]any{
			"exp_id":          a.ExpID,
			"slug":            a.Slug,
			"status":          a.Status,
			"final_loss":      a.FinalLoss,
			"best_val_loss":   a.BestValLoss,
			"val_loss_delta":  a.ValLossDelta,
		},
	})
}
