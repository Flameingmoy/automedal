// Anthropic provider — wraps github.com/anthropics/anthropic-sdk-go.
//
// Handles three sources behind the same code path:
//
//	BaseURL=""                              → Anthropic direct
//	BaseURL="https://opencode.ai/zen/go"    → opencode-go
//	BaseURL=<custom>                        → any Anthropic-compatible gateway
//
// Streaming via client.Messages.NewStreaming. Per-token text deltas are
// forwarded to events.Delta. Thinking blocks are surfaced via
// events.Thinking and echoed verbatim on the next turn so extended-
// thinking continuity holds.
//
// Tool messages from our internal shape (role="tool") are repacked into
// Anthropic's user-role tool_result blocks before sending.
package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Flameingmoy/automedal/internal/agent"
	anthropic "github.com/anthropics/anthropic-sdk-go"
	anthropicopt "github.com/anthropics/anthropic-sdk-go/option"
)

// AnthropicProvider talks to anthropic.com or any Anthropic-compatible
// endpoint via BaseURL.
type AnthropicProvider struct {
	ModelName string
	APIKey    string
	BaseURL   string
	MaxTokens int64
	Timeout   time.Duration
}

// Model satisfies ChatProvider.
func (p *AnthropicProvider) Model() string { return p.ModelName }

func (p *AnthropicProvider) client() anthropic.Client {
	opts := []anthropicopt.RequestOption{anthropicopt.WithAPIKey(p.APIKey)}
	if p.BaseURL != "" {
		opts = append(opts, anthropicopt.WithBaseURL(p.BaseURL))
	}
	if p.Timeout > 0 {
		opts = append(opts, anthropicopt.WithRequestTimeout(p.Timeout))
	}
	return anthropic.NewClient(opts...)
}

func (p *AnthropicProvider) maxTokens() int64 {
	if p.MaxTokens > 0 {
		return p.MaxTokens
	}
	return 4096
}

// ChatStream sends one turn through the Anthropic streaming endpoint.
func (p *AnthropicProvider) ChatStream(ctx context.Context, in ChatRequest) (*ChatTurn, error) {
	c := p.client()
	apiMsgs, err := toAnthropicMessages(in.Messages)
	if err != nil {
		return nil, err
	}
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(p.ModelName),
		MaxTokens: p.maxTokens(),
		Messages:  apiMsgs,
	}
	if in.System != "" {
		params.System = []anthropic.TextBlockParam{{Text: in.System}}
	}
	if len(in.Tools) > 0 {
		params.Tools = toAnthropicTools(in.Tools)
	}

	final, err := agent.WithRetry(ctx, func() (anthropic.Message, error) {
		return p.runStream(ctx, &c, params, in.Events)
	}, agent.RetryOpts{
		Sink:  in.Events,
		Label: "anthropic.chat_stream model=" + p.ModelName,
	})
	if err != nil {
		return nil, err
	}
	return finalToChatTurn(&final, in.Events), nil
}

// runStream executes one streaming attempt + accumulates the events into
// a final Message via the SDK's Accumulate helper. Forwards text + thinking
// deltas to the EventSink as they arrive.
func (p *AnthropicProvider) runStream(
	ctx context.Context,
	c *anthropic.Client,
	params anthropic.MessageNewParams,
	events *agent.EventSink,
) (anthropic.Message, error) {
	stream := c.Messages.NewStreaming(ctx, params)
	defer stream.Close()

	var msg anthropic.Message
	for stream.Next() {
		ev := stream.Current()
		if err := msg.Accumulate(ev); err != nil {
			return anthropic.Message{}, err
		}
		// Forward text deltas live to the EventSink.
		if events == nil {
			continue
		}
		if cb, ok := ev.AsAny().(anthropic.ContentBlockDeltaEvent); ok {
			switch d := cb.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				if d.Text != "" {
					events.Delta(d.Text)
				}
			case anthropic.ThinkingDelta:
				if d.Thinking != "" {
					events.Thinking(d.Thinking)
				}
			}
		}
	}
	if err := stream.Err(); err != nil {
		return anthropic.Message{}, err
	}
	return msg, nil
}

// ── outbound message translation ────────────────────────────────────────

func toAnthropicMessages(internal []agent.Message) ([]anthropic.MessageParam, error) {
	out := []anthropic.MessageParam{}
	pendingResults := []anthropic.ContentBlockParamUnion{}

	flushTools := func() {
		if len(pendingResults) > 0 {
			out = append(out, anthropic.NewUserMessage(pendingResults...))
			pendingResults = nil
		}
	}

	for _, m := range internal {
		switch m.Role {
		case "tool":
			content, _ := m.Content.(string)
			pendingResults = append(pendingResults, anthropic.NewToolResultBlock(m.ToolUseID, content, m.IsError))
			continue
		}
		flushTools()

		switch m.Role {
		case "user":
			switch c := m.Content.(type) {
			case string:
				out = append(out, anthropic.NewUserMessage(anthropic.NewTextBlock(c)))
			case []agent.Block:
				blocks, err := blocksToParams(c)
				if err != nil {
					return nil, err
				}
				out = append(out, anthropic.NewUserMessage(blocks...))
			default:
				return nil, fmt.Errorf("unexpected user content type %T", m.Content)
			}
		case "assistant":
			blocks, err := blocksToParams(m.Content.([]agent.Block))
			if err != nil {
				return nil, err
			}
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		default:
			return nil, fmt.Errorf("unknown role %q", m.Role)
		}
	}
	flushTools()
	return out, nil
}

func blocksToParams(blocks []agent.Block) ([]anthropic.ContentBlockParamUnion, error) {
	out := make([]anthropic.ContentBlockParamUnion, 0, len(blocks))
	for _, b := range blocks {
		switch b.Type {
		case "text":
			out = append(out, anthropic.NewTextBlock(b.Text))
		case "thinking":
			out = append(out, anthropic.NewThinkingBlock(b.Signature, b.Thinking))
		case "redacted_thinking":
			out = append(out, anthropic.NewRedactedThinkingBlock(b.Data))
		case "tool_use":
			out = append(out, anthropic.NewToolUseBlock(b.ID, b.Input, b.Name))
		case "tool_result":
			s, _ := b.Content.(string)
			out = append(out, anthropic.NewToolResultBlock(b.ToolUseID, s, b.IsError))
		default:
			return nil, fmt.Errorf("unknown block type %q", b.Type)
		}
	}
	return out, nil
}

func toAnthropicTools(specs []ToolSpec) []anthropic.ToolUnionParam {
	out := make([]anthropic.ToolUnionParam, 0, len(specs))
	for _, s := range specs {
		props, _ := s.Schema["properties"]
		req, _ := s.Schema["required"].([]string)
		if req == nil {
			if rawReq, ok := s.Schema["required"].([]any); ok {
				req = make([]string, len(rawReq))
				for i, v := range rawReq {
					req[i], _ = v.(string)
				}
			}
		}
		tp := anthropic.ToolParam{
			Name: s.Name,
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: props,
				Required:   req,
			},
		}
		if s.Description != "" {
			tp.Description = anthropic.String(s.Description)
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &tp})
	}
	return out
}

// ── inbound (final → ChatTurn) ──────────────────────────────────────────

func finalToChatTurn(m *anthropic.Message, _ *agent.EventSink) *ChatTurn {
	turn := &ChatTurn{
		Usage: agent.Usage{
			InTokens:  int(m.Usage.InputTokens),
			OutTokens: int(m.Usage.OutputTokens),
		},
		StopReason: string(m.StopReason),
	}
	for _, cb := range m.Content {
		switch cb.Type {
		case "text":
			turn.AssistantBlocks = append(turn.AssistantBlocks, agent.Block{
				Type: "text",
				Text: cb.Text,
			})
			turn.AssistantText += cb.Text
		case "thinking":
			turn.AssistantBlocks = append(turn.AssistantBlocks, agent.Block{
				Type:      "thinking",
				Thinking:  cb.Thinking,
				Signature: cb.Signature,
			})
		case "redacted_thinking":
			turn.AssistantBlocks = append(turn.AssistantBlocks, agent.Block{
				Type: "redacted_thinking",
				Data: cb.Data,
			})
		case "tool_use":
			var input map[string]any
			if len(cb.Input) > 0 {
				_ = json.Unmarshal(cb.Input, &input)
			}
			turn.AssistantBlocks = append(turn.AssistantBlocks, agent.Block{
				Type:  "tool_use",
				ID:    cb.ID,
				Name:  cb.Name,
				Input: input,
			})
			turn.ToolCalls = append(turn.ToolCalls, ToolCall{
				ID:   cb.ID,
				Name: cb.Name,
				Args: input,
			})
		}
	}
	return turn
}
