// AgentKernel — the bespoke tool-call loop.
//
// One kernel = one phase invocation. Build a fresh kernel per phase.
// The loop is provider-agnostic: it speaks the internal Anthropic-flavoured
// message shape (agent.Message + agent.Block) and lets each provider
// adapter translate at the wire.
//
// Port of automedal/agent/kernel.py:AgentKernel.
package agent

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

// stop reasons the kernel surfaces
const (
	StopAssistantDone = "assistant_done"
	StopMaxSteps      = "max_steps"
	StopProviderError = "provider_error"
)

var lengthStopReasons = map[string]bool{
	"length":            true,
	"max_tokens":        true,
	"max_output_tokens": true,
}

// isLengthStop tracks the divergent stop_reason wording across providers.
func isLengthStop(s string) bool { return lengthStopReasons[strings.ToLower(s)] }

// RunReport summarises one kernel run.
type RunReport struct {
	Stop       string    // StopAssistantDone | StopMaxSteps | StopProviderError
	FinalText  string    // last assistant text (or empty)
	Messages   []Message // full transcript including tool turns
	Steps      int
	UsageTotal Usage
	Error      string // populated when Stop == StopProviderError
}

// ToolCall is one tool invocation requested by the model. Mirrors
// providers.ToolCall — duplicated here to avoid an import cycle.
type ToolCall struct {
	ID   string
	Name string
	Args map[string]any
}

// ChatTurn is what providers return. Duplicated from providers for the
// same reason as ToolCall — kernel cannot import providers.
type ChatTurn struct {
	AssistantBlocks []Block
	AssistantText   string
	ToolCalls       []ToolCall
	Usage           Usage
	StopReason      string
}

// ChatStreamFunc is the kernel's view of a provider — a single function
// closing over the underlying provider + its model. The cmd layer wires
// this from providers.ChatProvider.
type ChatStreamFunc func(
	ctx context.Context,
	system string,
	messages []Message,
	toolList []tools.Tool,
	events *EventSink,
) (*ChatTurn, error)

// AgentKernel is a single-phase tool-call loop.
type AgentKernel struct {
	Chat               ChatStreamFunc
	SystemPrompt       string
	Tools              []tools.Tool
	Events             *EventSink
	MaxSteps           int
	ParallelToolCalls  bool

	byName map[string]tools.Tool
}

// NewKernel constructs a kernel with sensible defaults.
func NewKernel(chat ChatStreamFunc, system string, ts []tools.Tool, events *EventSink) *AgentKernel {
	k := &AgentKernel{
		Chat:              chat,
		SystemPrompt:      system,
		Tools:             ts,
		Events:            events,
		MaxSteps:          50,
		ParallelToolCalls: true,
		byName:            make(map[string]tools.Tool, len(ts)),
	}
	for _, t := range ts {
		k.byName[t.Name] = t
	}
	return k
}

// Run executes the kernel against `userMessage`. Returns a RunReport.
func (k *AgentKernel) Run(ctx context.Context, userMessage string) RunReport {
	messages := []Message{{Role: "user", Content: userMessage}}
	usageTotal := Usage{}

	for step := 1; step <= k.MaxSteps; step++ {
		if k.Events != nil {
			k.Events.StepAdvance()
		}

		// Self-heal: patch any unanswered tool_use blocks left by an
		// interrupted prior step so the provider doesn't 400.
		if stubs := PatchDanglingToolCalls(&messages); stubs > 0 && k.Events != nil {
			k.Events.Notice("self_heal", fmt.Sprintf("patched %d dangling tool_use block(s)", stubs))
		}

		// Doom-loop guard: detect repetition / cycles in recent tool calls.
		if doom := CheckDoomLoop(messages); doom != "" {
			messages = append(messages, Message{Role: "user", Content: doom})
			if k.Events != nil {
				k.Events.Notice("doom_loop", doom)
			}
		}

		turn, err := k.Chat(ctx, k.SystemPrompt, messages, k.Tools, k.Events)
		if err != nil {
			if k.Events != nil {
				k.Events.Error(fmt.Sprintf("provider.chat_stream step=%d", step), err)
			}
			return RunReport{
				Stop:       StopProviderError,
				Messages:   messages,
				Steps:      step,
				UsageTotal: usageTotal,
				Error:      FormatError(err),
			}
		}

		usageTotal.InTokens += turn.Usage.InTokens
		usageTotal.OutTokens += turn.Usage.OutTokens
		if k.Events != nil && (turn.Usage.InTokens > 0 || turn.Usage.OutTokens > 0) {
			k.Events.Usage(turn.Usage.InTokens, turn.Usage.OutTokens)
		}

		// Truncation handler: model ran out of output budget mid-tool-call.
		// JSON arguments are garbage; drop the calls, keep any text prefix,
		// inject a hint, and re-loop.
		if isLengthStop(turn.StopReason) && len(turn.ToolCalls) > 0 {
			dropped := make([]string, 0, len(turn.ToolCalls))
			for _, tc := range turn.ToolCalls {
				dropped = append(dropped, tc.Name)
			}
			textBlocks := make([]Block, 0, len(turn.AssistantBlocks))
			for _, b := range turn.AssistantBlocks {
				if b.Type == "text" {
					textBlocks = append(textBlocks, b)
				}
			}
			messages = append(messages, Message{Role: "assistant", Content: textBlocks})
			hint := fmt.Sprintf(
				"Your previous response was truncated by the output token limit, so "+
					"the following tool calls were dropped: %v. Do NOT retry with the "+
					"same large content. For 'write_file' use bash with cat<<'HEREDOC', "+
					"or split into multiple smaller edit_file calls.",
				dropped,
			)
			messages = append(messages, Message{Role: "user", Content: hint})
			if k.Events != nil {
				k.Events.Notice("truncation", fmt.Sprintf("stop=length; dropped %v", dropped))
			}
			continue
		}

		// Echo assistant blocks back into the transcript verbatim.
		messages = append(messages, Message{Role: "assistant", Content: turn.AssistantBlocks})

		if len(turn.ToolCalls) == 0 {
			return RunReport{
				Stop:       StopAssistantDone,
				FinalText:  turn.AssistantText,
				Messages:   messages,
				Steps:      step,
				UsageTotal: usageTotal,
			}
		}

		results := k.executeTools(ctx, turn.ToolCalls)
		for i, call := range turn.ToolCalls {
			r := results[i]
			msg := Message{
				Role:      "tool",
				ToolUseID: call.ID,
				Content:   r.Text,
			}
			if !r.OK {
				msg.IsError = true
			}
			messages = append(messages, msg)
		}
	}

	return RunReport{
		Stop:       StopMaxSteps,
		Messages:   messages,
		Steps:      k.MaxSteps,
		UsageTotal: usageTotal,
	}
}

// executeTools dispatches every tool call (in parallel when batched) and
// returns results in the same order as `calls`.
func (k *AgentKernel) executeTools(ctx context.Context, calls []ToolCall) []tools.ToolResult {
	out := make([]tools.ToolResult, len(calls))

	dispatch := func(i int) {
		call := calls[i]
		if k.Events != nil {
			k.Events.ToolStart(call.ID, call.Name, call.Args)
		}
		t, ok := k.byName[call.Name]
		var res tools.ToolResult
		if !ok {
			res = tools.Error("error: unknown tool %q", call.Name)
		} else {
			res = t.Invoke(ctx, call.Args)
		}
		if k.Events != nil {
			k.Events.ToolEnd(call.ID, call.Name, res.OK, res.Text)
		}
		out[i] = res
	}

	if k.ParallelToolCalls && len(calls) > 1 {
		var wg sync.WaitGroup
		wg.Add(len(calls))
		// Stable order in `out`; goroutines write disjoint indices.
		idxs := make([]int, len(calls))
		for i := range calls {
			idxs[i] = i
		}
		sort.Ints(idxs)
		for _, i := range idxs {
			i := i
			go func() {
				defer wg.Done()
				dispatch(i)
			}()
		}
		wg.Wait()
		return out
	}
	for i := range calls {
		dispatch(i)
	}
	return out
}
