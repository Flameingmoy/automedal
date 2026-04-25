// Package providers — pluggable LLM provider adapters.
//
// Internal message shape mirrors automedal/agent/providers/base.py:
//
//	{role: "user",      content: <string or []Block>}
//	{role: "assistant", content: []Block — text/thinking/tool_use}
//	{role: "tool",      tool_use_id: <id>, content: <string>}
//
// Each provider translates between this shape and its own wire format.
package providers

import (
	"context"

	"github.com/Flameingmoy/automedal/internal/agent"
)

// ToolCall is one tool invocation requested by the model.
type ToolCall struct {
	ID   string
	Name string
	Args map[string]any
}

// ToolSpec is what providers send to the LLM as a callable tool.
// (kernel.Tool wraps the actual implementation; this is just the shape.)
type ToolSpec struct {
	Name        string
	Description string
	Schema      map[string]any
}

// ChatTurn is the result of one provider.ChatStream call.
//
// AssistantBlocks is the raw block list (per provider's native format
// when echoed back, e.g. Anthropic's blocks include thinking blocks
// with signatures). The kernel must echo this list back verbatim on
// the next turn so tool_use ids line up.
type ChatTurn struct {
	AssistantBlocks []agent.Block
	AssistantText   string
	ToolCalls       []ToolCall
	Usage           agent.Usage
	StopReason      string
}

// ChatProvider is the uniform interface every adapter implements.
type ChatProvider interface {
	// Model returns the model id (e.g. "claude-sonnet-4-5", "minimax-m2.7").
	Model() string

	// ChatStream runs one streaming turn against the provider, forwarding
	// per-token deltas to events.Delta and (optionally) thinking blocks
	// to events.Thinking. Returns the final ChatTurn.
	ChatStream(ctx context.Context, in ChatRequest) (*ChatTurn, error)
}

// ChatRequest bundles the inputs to ChatStream so the signature stays
// stable as we add fields.
type ChatRequest struct {
	System   string
	Messages []agent.Message
	Tools    []ToolSpec
	Events   *agent.EventSink // nullable; emitted-to when non-nil
}
