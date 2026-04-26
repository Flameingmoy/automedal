// Advisor LLM client — one non-streaming Chat Completions round-trip.
//
// The advisor is invoked against an OpenAI-compatible endpoint (default
// https://opencode.ai/zen/go/v1, which also hosts the executor's
// minimax-m2.7 behind the same OPENCODE_API_KEY). The default model is
// kimi-k2.6.
//
// Mirrors automedal/advisor/client.py. Never raises — every failure
// returns AdvisorOpinion{Skipped: true, Reason: ...}.
package advisor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Flameingmoy/automedal/internal/advisor/prompts"
)

// AdvisorOpinion is one consult result. Mirrors the Python dataclass.
type AdvisorOpinion struct {
	Text      string
	InTokens  int
	OutTokens int
	Skipped   bool
	Reason    string
}

// AdvisorEvents is the narrow recorder Consult uses for one
// advisor_consult event per call. The cmd layer adapts agent.EventSink
// to satisfy this (see internal/advisor/sinkadapter via its AdvisorConsult
// method).
type AdvisorEvents interface {
	RecordAdvisorConsult(purpose, model, reason, preview string, inTokens, outTokens int, skipped bool)
}

func shortText(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func emit(evt AdvisorEvents, purpose, model string, op AdvisorOpinion) {
	if evt == nil {
		return
	}
	defer func() { _ = recover() }()
	evt.RecordAdvisorConsult(purpose, model, op.Reason, shortText(op.Text, 280),
		op.InTokens, op.OutTokens, op.Skipped)
}

// Consult asks the advisor model. Never raises.
//
// `purpose` controls (a) the prompt template and (b) the junction
// allowlist gate. Use the canonical set: stagnation, audit, tool.
func Consult(ctx context.Context, purpose, question, contextHint string, events AdvisorEvents) AdvisorOpinion {
	model := envStr(envModel, defaultModel)

	if !IsEnabled(purpose) {
		op := AdvisorOpinion{Skipped: true, Reason: "disabled:" + purpose}
		emit(events, purpose, model, op)
		return op
	}

	maxTokens := envInt(envCapPerConsult, defaultCapPerConsult)
	rem := RemainingTokens()
	if rem <= 0 {
		op := AdvisorOpinion{Skipped: true, Reason: "budget:iter"}
		emit(events, purpose, model, op)
		return op
	}
	if maxTokens > rem {
		maxTokens = rem
	}

	if question == "" {
		question = "(no question)"
	}
	if contextHint == "" {
		contextHint = "(no context)"
	}
	prompt, err := prompts.Render(purpose, map[string]any{
		"question": question,
		"context":  contextHint,
	})
	if err != nil {
		op := AdvisorOpinion{Skipped: true, Reason: fmt.Sprintf("template:%T", err)}
		emit(events, purpose, model, op)
		return op
	}

	apiKey := envStr(envAPIKey, "")
	if apiKey == "" {
		op := AdvisorOpinion{Skipped: true, Reason: "no_api_key"}
		emit(events, purpose, model, op)
		return op
	}

	baseURL := envStr(envBaseURL, defaultBaseURL)
	resp, err := chatCompletion(ctx, baseURL, apiKey, model, prompt, maxTokens)
	if err != nil {
		op := AdvisorOpinion{Skipped: true, Reason: "error:" + errKind(err)}
		emit(events, purpose, model, op)
		return op
	}

	text := resp.Text
	in, out := resp.InTokens, resp.OutTokens
	ConsumeTokens(in + out)

	if text == "" {
		op := AdvisorOpinion{InTokens: in, OutTokens: out, Skipped: true, Reason: "empty"}
		emit(events, purpose, model, op)
		return op
	}

	op := AdvisorOpinion{Text: text, InTokens: in, OutTokens: out}
	emit(events, purpose, model, op)
	return op
}

// chatCompletion does one non-streaming /v1/chat/completions POST.
type chatResp struct {
	Text      string
	InTokens  int
	OutTokens int
}

func errKind(err error) string {
	// Mirror Python's type(exc).__name__ shape — just the last segment.
	s := fmt.Sprintf("%T", err)
	if i := strings.LastIndex(s, "."); i >= 0 {
		s = s[i+1:]
	}
	return strings.TrimPrefix(s, "*")
}

type ccReq struct {
	Model     string      `json:"model"`
	MaxTokens int         `json:"max_tokens"`
	Messages  []ccMessage `json:"messages"`
}

type ccMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ccUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type ccChoice struct {
	Message ccMessage `json:"message"`
}

type ccResponse struct {
	Choices []ccChoice `json:"choices"`
	Usage   ccUsage    `json:"usage"`
}

func chatCompletion(ctx context.Context, baseURL, apiKey, model, prompt string, maxTokens int) (chatResp, error) {
	cctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	body, _ := json.Marshal(ccReq{
		Model:     model,
		MaxTokens: maxTokens,
		Messages:  []ccMessage{{Role: "user", Content: prompt}},
	})
	req, err := http.NewRequestWithContext(cctx, http.MethodPost,
		strings.TrimRight(baseURL, "/")+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return chatResp{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	r, err := http.DefaultClient.Do(req)
	if err != nil {
		return chatResp{}, err
	}
	defer r.Body.Close()
	if r.StatusCode/100 != 2 {
		b, _ := io.ReadAll(r.Body)
		return chatResp{}, fmt.Errorf("HTTP %d: %s", r.StatusCode, string(b))
	}
	rb, err := io.ReadAll(r.Body)
	if err != nil {
		return chatResp{}, err
	}
	var cr ccResponse
	if err := json.Unmarshal(rb, &cr); err != nil {
		return chatResp{}, fmt.Errorf("bad JSON: %w", err)
	}
	out := chatResp{InTokens: cr.Usage.PromptTokens, OutTokens: cr.Usage.CompletionTokens}
	if len(cr.Choices) > 0 {
		out.Text = strings.TrimSpace(cr.Choices[0].Message.Content)
	}
	return out, nil
}
