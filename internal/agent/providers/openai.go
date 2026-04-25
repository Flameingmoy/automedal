// OpenAI provider — wraps github.com/openai/openai-go.
//
// Used for OpenAI direct, Ollama (/v1), OpenRouter, Groq, and any other
// OpenAI-compatible endpoint via BaseURL.
//
// Translates between our internal Anthropic-flavoured message shape and
// OpenAI's chat-completions schema:
//
//	internal "tool" role  → OpenAI "tool" role with tool_call_id
//	internal assistant tool_use blocks → assistant message with tool_calls
package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Flameingmoy/automedal/internal/agent"
	openai "github.com/openai/openai-go"
	openaiopt "github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

// OpenAIProvider talks to api.openai.com or any OpenAI-compatible
// endpoint via BaseURL.
type OpenAIProvider struct {
	ModelName string
	APIKey    string
	BaseURL   string
	Timeout   time.Duration
}

func (p *OpenAIProvider) Model() string { return p.ModelName }

func (p *OpenAIProvider) client() openai.Client {
	opts := []openaiopt.RequestOption{openaiopt.WithAPIKey(p.APIKey)}
	if p.BaseURL != "" {
		opts = append(opts, openaiopt.WithBaseURL(p.BaseURL))
	}
	if p.Timeout > 0 {
		opts = append(opts, openaiopt.WithRequestTimeout(p.Timeout))
	}
	return openai.NewClient(opts...)
}

// ChatStream sends one streaming turn through the chat-completions API.
func (p *OpenAIProvider) ChatStream(ctx context.Context, in ChatRequest) (*ChatTurn, error) {
	c := p.client()
	apiMsgs, err := toOpenAIMessages(in.System, in.Messages)
	if err != nil {
		return nil, err
	}
	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(p.ModelName),
		Messages: apiMsgs,
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		},
	}
	if len(in.Tools) > 0 {
		params.Tools = toOpenAITools(in.Tools)
	}

	final, err := agent.WithRetry(ctx, func() (openai.ChatCompletion, error) {
		return p.runStream(ctx, &c, params, in.Events)
	}, agent.RetryOpts{
		Sink:  in.Events,
		Label: "openai.chat_stream model=" + p.ModelName,
	})
	if err != nil {
		return nil, err
	}
	return openAIFinalToTurn(&final), nil
}

func (p *OpenAIProvider) runStream(
	ctx context.Context,
	c *openai.Client,
	params openai.ChatCompletionNewParams,
	events *agent.EventSink,
) (openai.ChatCompletion, error) {
	stream := c.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()

	acc := openai.ChatCompletionAccumulator{}
	for stream.Next() {
		chunk := stream.Current()
		if !acc.AddChunk(chunk) {
			return openai.ChatCompletion{}, fmt.Errorf("openai accumulator rejected chunk")
		}
		// Live text deltas to the EventSink.
		if events != nil && len(chunk.Choices) > 0 {
			if d := chunk.Choices[0].Delta.Content; d != "" {
				events.Delta(d)
			}
		}
	}
	if err := stream.Err(); err != nil {
		return openai.ChatCompletion{}, err
	}
	return acc.ChatCompletion, nil
}

// ── outbound message translation ────────────────────────────────────────

func toOpenAIMessages(system string, internal []agent.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	var out []openai.ChatCompletionMessageParamUnion
	if system != "" {
		out = append(out, openai.SystemMessage(system))
	}

	for _, m := range internal {
		switch m.Role {
		case "tool":
			content, _ := m.Content.(string)
			out = append(out, openai.ToolMessage(content, m.ToolUseID))

		case "user":
			switch c := m.Content.(type) {
			case string:
				out = append(out, openai.UserMessage(c))
			case []agent.Block:
				// Anthropic-shape tool_result blocks → flatten as tool messages.
				flushed := false
				for _, b := range c {
					if b.Type == "tool_result" {
						s, _ := b.Content.(string)
						out = append(out, openai.ToolMessage(s, b.ToolUseID))
						flushed = true
					}
				}
				if !flushed {
					var text string
					for _, b := range c {
						if b.Type == "text" {
							text += b.Text
						}
					}
					out = append(out, openai.UserMessage(text))
				}
			default:
				return nil, fmt.Errorf("unexpected user content type %T", m.Content)
			}

		case "assistant":
			blocks, _ := m.Content.([]agent.Block)
			ap := openai.ChatCompletionAssistantMessageParam{}
			var text string
			for _, b := range blocks {
				switch b.Type {
				case "text":
					text += b.Text
				case "tool_use":
					argsJSON, err := json.Marshal(b.Input)
					if err != nil {
						return nil, err
					}
					ap.ToolCalls = append(ap.ToolCalls, openai.ChatCompletionMessageToolCallParam{
						ID: b.ID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      b.Name,
							Arguments: string(argsJSON),
						},
					})
					// thinking / redacted_thinking: no OpenAI analog; drop.
				}
			}
			if text != "" {
				ap.Content.OfString = openai.String(text)
			}
			out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: &ap})

		default:
			return nil, fmt.Errorf("unknown role %q", m.Role)
		}
	}
	return out, nil
}

func toOpenAITools(specs []ToolSpec) []openai.ChatCompletionToolParam {
	out := make([]openai.ChatCompletionToolParam, 0, len(specs))
	for _, s := range specs {
		fd := shared.FunctionDefinitionParam{
			Name:       s.Name,
			Parameters: shared.FunctionParameters(s.Schema),
		}
		if s.Description != "" {
			fd.Description = openai.String(s.Description)
		}
		out = append(out, openai.ChatCompletionToolParam{Function: fd})
	}
	return out
}

// ── inbound (final → ChatTurn) ──────────────────────────────────────────

func openAIFinalToTurn(cc *openai.ChatCompletion) *ChatTurn {
	turn := &ChatTurn{
		Usage: agent.Usage{
			InTokens:  int(cc.Usage.PromptTokens),
			OutTokens: int(cc.Usage.CompletionTokens),
		},
	}
	if len(cc.Choices) == 0 {
		return turn
	}
	choice := cc.Choices[0]
	turn.StopReason = string(choice.FinishReason)

	if text := choice.Message.Content; text != "" {
		turn.AssistantBlocks = append(turn.AssistantBlocks, agent.Block{Type: "text", Text: text})
		turn.AssistantText = text
	}
	for _, tc := range choice.Message.ToolCalls {
		var args map[string]any
		if tc.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				args = map[string]any{"_raw": tc.Function.Arguments}
			}
		}
		turn.ToolCalls = append(turn.ToolCalls, ToolCall{
			ID:   tc.ID,
			Name: tc.Function.Name,
			Args: args,
		})
		turn.AssistantBlocks = append(turn.AssistantBlocks, agent.Block{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: args,
		})
	}
	return turn
}
