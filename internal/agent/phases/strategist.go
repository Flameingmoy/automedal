// Strategist phase — curates knowledge.md and writes the next 5 queue entries.
package phases

import (
	"context"

	"github.com/Flameingmoy/automedal/internal/advisor"
	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

// StrategistArgs is the slot dict for the strategist phase.
type StrategistArgs struct {
	ExpID        string
	Iteration    int
	MaxIters     int
	Stagnating   any
	BestValLoss  any
	Pending      int
	Reflective   string
	Ranked       string
	AdvisorNote  string
	MaxSteps     int
	ConsultFunc  tools.ConsultFunc // wired by cmd layer; nil disables consult_advisor
}

// RunStrategist runs the strategist phase.
func RunStrategist(ctx context.Context, chat agent.ChatStreamFunc, events *agent.EventSink, a StrategistArgs) (agent.RunReport, error) {
	base := []tools.Tool{
		tools.ReadFile, tools.WriteFile, tools.EditFile, tools.ListDir, tools.Grep, tools.Recall,
	}

	// Per-phase session dict — scopes plan state to this invocation.
	session := map[string]any{}

	extraFactory := func(sink *agent.EventSink) []tools.Tool {
		extras := []tools.Tool{tools.MakePlanTool(session, sink)}
		if advisor.IsEnabled("tool") && a.ConsultFunc != nil {
			extras = append(extras, tools.MakeAdvisorTool(a.ConsultFunc, 1))
		}
		return extras
	}

	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 30
	}

	return RunPhase(ctx, RunPhaseConfig{
		Phase:    "strategist",
		Chat:     chat,
		Tools:    base,
		Events:   events,
		MaxSteps: maxSteps,
		Slots: map[string]any{
			"exp_id":        a.ExpID,
			"iteration":     a.Iteration,
			"max_iters":     a.MaxIters,
			"stagnating":    a.Stagnating,
			"best_val_loss": a.BestValLoss,
			"pending":       a.Pending,
			"reflective":    a.Reflective,
			"ranked":        a.Ranked,
			"advisor_note":  a.AdvisorNote,
		},
		ExtraToolsFactory: extraFactory,
	})
}
