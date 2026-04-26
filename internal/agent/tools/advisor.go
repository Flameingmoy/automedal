// consult_advisor — worker-triggered advisor consult, one per phase.
//
// Mirrors automedal/agent/tools/advisor.py. The actual call into the
// advisor LLM is supplied by the cmd layer as a ConsultFunc, so this
// package stays free of an internal/advisor import.
package tools

import (
	"context"
)

// AdvisorOpinion is the value returned by a ConsultFunc. Mirrors
// internal/advisor.AdvisorOpinion (kept in this package to avoid the
// import — the wire shape is the same).
type AdvisorOpinion struct {
	Text     string
	Skipped  bool
	Reason   string
	InTokens int
	OutTokens int
}

// ConsultFunc is the kernel-tool view of the advisor. The cmd layer
// wires this from internal/advisor.Consult.
type ConsultFunc func(ctx context.Context, purpose, question, contextHint string) AdvisorOpinion

// MakeAdvisorTool returns a fresh consult_advisor Tool with its own
// per-phase use counter (closed over). Phases get a new instance every
// iteration via run_phase.
func MakeAdvisorTool(consult ConsultFunc, maxUses int) Tool {
	if maxUses <= 0 {
		maxUses = 1
	}
	desc := "Ask a frontier model (configured advisor, e.g. Kimi K2.6) for a second " +
		"opinion on a hard design decision. Expensive — use only when the choice " +
		"materially affects outcome. At most one call per phase."
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"question": map[string]any{
				"type": "string",
				"description": "The specific decision or tradeoff you want adjudicated. " +
					"Be concrete — name the model, feature, or hyperparameter.",
			},
			"context_hint": map[string]any{
				"type": "string",
				"description": "Short summary of the relevant code or state the advisor " +
					"should consider. Include the failing/contested fragment verbatim " +
					"if small; otherwise paraphrase.",
			},
		},
		"required": []string{"question", "context_hint"},
	}

	uses := 0
	run := func(ctx context.Context, args map[string]any) (ToolResult, error) {
		if uses >= maxUses {
			return Error("Budget exhausted for this phase — proceed without advisor."), nil
		}
		question := StrArg(args, "question", "")
		if question == "" {
			return MissingArg("question"), nil
		}
		ctxHint := StrArg(args, "context_hint", "")
		if ctxHint == "" {
			return MissingArg("context_hint"), nil
		}
		uses++
		if consult == nil {
			return Error("Advisor unavailable (no consult function wired) — proceed without advisor."), nil
		}
		op := consult(ctx, "tool", question, ctxHint)
		if op.Skipped {
			reason := op.Reason
			if reason == "" {
				reason = "unavailable"
			}
			return Error("Advisor unavailable (%s) — proceed without advisor.", reason), nil
		}
		return Result(op.Text), nil
	}
	return Tool{Name: "consult_advisor", Description: desc, Schema: schema, Run: run}
}
