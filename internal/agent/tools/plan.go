// plan tool — lets the strategist maintain a structured todo list.
//
// Each call replaces the entire plan and emits a `notice` event tagged
// `plan_update` so the TUI can render the current state in a side-panel.
//
// Status vocabulary mirrors ml-intern/agent/tools/plan_tool.py:
//
//	pending | in_progress | completed
//
// Validation rules:
//   - id    — non-empty string, unique within the call
//   - content — non-empty string, ≤280 chars
//   - status — one of the vocabulary above
//   - at most one item may be in_progress at a time
package tools

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// PlanItem is one row of the strategist's plan.
type PlanItem struct {
	ID      string
	Content string
	Status  string
}

// Notifier is the subset of EventSink the plan tool depends on. We model
// it as an interface so we don't import the events package (which would
// cycle through providers/kernel).
type Notifier interface {
	Notice(tag, message string)
}

var validStatuses = map[string]bool{
	"pending":     true,
	"in_progress": true,
	"completed":   true,
}

// MakePlanTool returns a tool that mutates `session["plan"]` and emits
// a plan_update notice on the supplied sink. session must be non-nil.
func MakePlanTool(session map[string]any, events Notifier) Tool {
	var mu sync.Mutex
	desc := "Maintain a structured plan of what you intend to do. Each call replaces " +
		"the entire plan. Use status='in_progress' for the single item you are " +
		"currently working on (at most one), 'pending' for future items, and " +
		"'completed' for finished items. Keep content short — one line each."
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"items": map[string]any{
				"type":        "array",
				"description": "Full replacement plan — every call overwrites the previous one.",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":      map[string]any{"type": "string", "description": "Unique short identifier."},
						"content": map[string]any{"type": "string", "description": "One-line task description."},
						"status":  map[string]any{"type": "string", "enum": []string{"completed", "in_progress", "pending"}},
					},
					"required": []string{"id", "content", "status"},
				},
			},
		},
		"required": []string{"items"},
	}
	run := func(ctx context.Context, args map[string]any) (ToolResult, error) {
		raw, ok := args["items"]
		if !ok {
			return Error("error: items must be a non-empty array"), nil
		}
		rawList, ok := raw.([]any)
		if !ok || len(rawList) == 0 {
			return Error("error: items must be a non-empty array"), nil
		}
		seen := map[string]bool{}
		inProgress := 0
		normalized := make([]PlanItem, 0, len(rawList))
		for i, x := range rawList {
			it, ok := x.(map[string]any)
			if !ok {
				return Error("error: items[%d] is not an object", i), nil
			}
			id := strings.TrimSpace(StrArg(it, "id", ""))
			if id == "" {
				return Error("error: items[%d].id must be a non-empty string", i), nil
			}
			if seen[id] {
				return Error("error: duplicate id %q", id), nil
			}
			seen[id] = true
			content := strings.TrimSpace(StrArg(it, "content", ""))
			if content == "" {
				return Error("error: items[%d].content must be a non-empty string", i), nil
			}
			if len(content) > 280 {
				return Error("error: items[%d].content exceeds 280 chars", i), nil
			}
			status := StrArg(it, "status", "")
			if !validStatuses[status] {
				return Error("error: items[%d].status must be one of [completed in_progress pending]", i), nil
			}
			if status == "in_progress" {
				inProgress++
			}
			normalized = append(normalized, PlanItem{ID: id, Content: content, Status: status})
		}
		if inProgress > 1 {
			return Error("error: at most one item may be status='in_progress'"), nil
		}

		mu.Lock()
		session["plan"] = normalized
		mu.Unlock()

		if events != nil {
			pieces := make([]string, len(normalized))
			for i, it := range normalized {
				pieces[i] = fmt.Sprintf("[%s] %s", string(it.Status[0]), it.Content)
			}
			events.Notice("plan_update", fmt.Sprintf("%d items: %s", len(normalized), strings.Join(pieces, " | ")))
		}

		counts := map[string]int{}
		for _, it := range normalized {
			counts[it.Status]++
		}
		return Result(fmt.Sprintf(
			"plan updated (%d items: %d pending, %d in_progress, %d completed)",
			len(normalized), counts["pending"], counts["in_progress"], counts["completed"],
		)), nil
	}
	return Tool{Name: "plan", Description: desc, Schema: schema, Run: run}
}
