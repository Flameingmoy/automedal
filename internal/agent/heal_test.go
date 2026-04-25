package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

// ── retry ───────────────────────────────────────────────────────────────

func TestIsTransientErrorPatternMatches(t *testing.T) {
	for _, msg := range []string{
		"503 Service Unavailable",
		"Connection reset by peer",
		"HTTP 429 — rate_limit_exceeded",
		"upstream EOF",
	} {
		if !IsTransientError(errors.New(msg)) {
			t.Errorf("transient miss: %q", msg)
		}
	}
}

func TestIsTransientErrorIgnoresAuth(t *testing.T) {
	for _, msg := range []string{
		"401 Unauthorized",
		"insufficient credits",
		"model_not_found",
	} {
		if IsTransientError(errors.New(msg)) {
			t.Errorf("non-transient mis-classified: %q", msg)
		}
	}
}

func TestWithRetrySucceedsAfterTransient(t *testing.T) {
	calls := 0
	out, err := WithRetry(context.Background(), func() (string, error) {
		calls++
		if calls < 3 {
			return "", errors.New("503 service unavailable")
		}
		return "ok", nil
	}, RetryOpts{
		Attempts: 3,
		Delays:   []time.Duration{0, 0, 0},
		Label:    "test",
	})
	if err != nil {
		t.Fatal(err)
	}
	if out != "ok" || calls != 3 {
		t.Errorf("calls=%d out=%q", calls, out)
	}
}

func TestWithRetryGivesUpOnNonTransient(t *testing.T) {
	calls := 0
	_, err := WithRetry(context.Background(), func() (int, error) {
		calls++
		return 0, errors.New("401 unauthorized")
	}, RetryOpts{Attempts: 3, Delays: []time.Duration{0}, Label: "x"})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Errorf("non-transient should not retry, got %d calls", calls)
	}
}

func TestWithRetryRespectsContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := WithRetry(ctx, func() (int, error) {
		return 0, errors.New("503")
	}, RetryOpts{Attempts: 3, Delays: []time.Duration{time.Hour}, Label: "x"})
	if err == nil {
		t.Fatal("expected ctx cancel")
	}
}

// ── friendly errors ─────────────────────────────────────────────────────

func TestFriendlyErrorAuth(t *testing.T) {
	for _, msg := range []string{
		"401 Unauthorized",
		"Invalid x-api-key",
		"invalid api key",
	} {
		got := FriendlyError(errors.New(msg))
		if !strings.Contains(got, "OPENCODE_API_KEY") {
			t.Errorf("auth msg miss for %q", msg)
		}
	}
}

func TestFriendlyErrorCredits(t *testing.T) {
	got := FriendlyError(errors.New("insufficient credits remaining"))
	if !strings.Contains(strings.ToLower(got), "credits") {
		t.Errorf("credits miss: %q", got)
	}
}

func TestFriendlyErrorModelNotFound(t *testing.T) {
	got := FriendlyError(errors.New("model_not_found"))
	if !strings.Contains(got, "automedal models") {
		t.Errorf("model-not-found miss: %q", got)
	}
}

func TestFriendlyErrorContext(t *testing.T) {
	got := FriendlyError(errors.New("context_length_exceeded: too many tokens"))
	if got == "" {
		t.Error("context-exceed miss")
	}
}

func TestFriendlyErrorUnknownReturnsEmpty(t *testing.T) {
	if FriendlyError(errors.New("totally unknown failure")) != "" {
		t.Error("unknown error should map to empty")
	}
}

func TestFormatErrorAlwaysContainsRaw(t *testing.T) {
	out := FormatError(errors.New("401 Unauthorized"))
	if !strings.Contains(out, "[raw]") || !strings.Contains(out, "OPENCODE_API_KEY") {
		t.Errorf("missing parts: %q", out)
	}
}

// ── messages: PatchDanglingToolCalls ────────────────────────────────────

func TestPatchDanglingToolCallsCleanIsNoop(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "do it"},
		{Role: "assistant", Content: []Block{
			{Type: "tool_use", ID: "t1", Name: "bash"},
		}},
		{Role: "tool", ToolUseID: "t1", Content: "ok"},
		{Role: "assistant", Content: []Block{{Type: "text", Text: "done"}}},
	}
	if PatchDanglingToolCalls(&msgs) != 0 {
		t.Error("clean transcript should patch nothing")
	}
}

func TestPatchDanglingToolCallsFillsGaps(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "do it"},
		{Role: "assistant", Content: []Block{
			{Type: "tool_use", ID: "t1", Name: "bash"},
			{Type: "tool_use", ID: "t2", Name: "bash"},
		}},
		{Role: "tool", ToolUseID: "t1", Content: "ok"},
	}
	got := PatchDanglingToolCalls(&msgs)
	if got != 1 {
		t.Errorf("want 1, got %d", got)
	}
	last := msgs[len(msgs)-1]
	if last.Role != "tool" || last.ToolUseID != "t2" || !last.IsError {
		t.Errorf("stub wrong: %+v", last)
	}
}

func TestPatchAcceptsAnthropicShapeAnswers(t *testing.T) {
	msgs := []Message{
		{Role: "assistant", Content: []Block{
			{Type: "tool_use", ID: "t1", Name: "bash"},
		}},
		{Role: "user", Content: []Block{
			{Type: "tool_result", ToolUseID: "t1", Content: "ok"},
		}},
	}
	if PatchDanglingToolCalls(&msgs) != 0 {
		t.Error("Anthropic-shape tool_result should count as answered")
	}
}

func TestPatchIdempotent(t *testing.T) {
	msgs := []Message{
		{Role: "assistant", Content: []Block{
			{Type: "tool_use", ID: "t1", Name: "bash"},
		}},
	}
	if PatchDanglingToolCalls(&msgs) != 1 {
		t.Fatal("first call should patch once")
	}
	if PatchDanglingToolCalls(&msgs) != 0 {
		t.Error("second call should be a no-op")
	}
}

// ── doom-loop ───────────────────────────────────────────────────────────

func toolUseTurn(name string, args map[string]any, id string) Message {
	return Message{Role: "assistant", Content: []Block{
		{Type: "tool_use", ID: id, Name: name, Input: args},
	}}
}

func toolResult(id string) Message {
	return Message{Role: "tool", ToolUseID: id, Content: "ok"}
}

func TestDoomLoopIdenticalTriple(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "x"}}
	for i := 0; i < 3; i++ {
		msgs = append(msgs,
			toolUseTurn("bash", map[string]any{"cmd": "echo hi"}, fmt.Sprintf("c%d", i)),
			toolResult(fmt.Sprintf("c%d", i)),
		)
	}
	got := CheckDoomLoop(msgs)
	if got == "" || !strings.Contains(got, "bash") {
		t.Errorf("no doom-loop fired: %q", got)
	}
}

func TestDoomLoopDifferentArgsNoFire(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "x"}}
	for i := 0; i < 3; i++ {
		msgs = append(msgs,
			toolUseTurn("bash", map[string]any{"cmd": fmt.Sprintf("echo %d", i)}, fmt.Sprintf("c%d", i)),
			toolResult(fmt.Sprintf("c%d", i)),
		)
	}
	if got := CheckDoomLoop(msgs); got != "" {
		t.Errorf("false positive: %q", got)
	}
}

func TestDoomLoopABABCycle(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "x"}}
	pairs := []struct{ name, val string }{
		{"a", "1"}, {"b", "2"}, {"a", "1"}, {"b", "2"},
	}
	for i, p := range pairs {
		msgs = append(msgs,
			toolUseTurn(p.name, map[string]any{"x": p.val}, fmt.Sprintf("c%d", i)),
			toolResult(fmt.Sprintf("c%d", i)),
		)
	}
	got := CheckDoomLoop(msgs)
	if got == "" || !strings.Contains(strings.ToLower(got), "cycling") {
		t.Errorf("AB cycle missed: %q", got)
	}
}

func TestDoomLoopEnvKillSwitch(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "x"}}
	for i := 0; i < 4; i++ {
		msgs = append(msgs, toolUseTurn("bash", map[string]any{"cmd": "x"}, "c"))
	}
	t.Setenv("AUTOMEDAL_DOOM_LOOP", "0")
	if got := CheckDoomLoop(msgs); got != "" {
		t.Errorf("kill switch ignored: %q", got)
	}
}

func TestDoomLoopEmptyMessages(t *testing.T) {
	if CheckDoomLoop(nil) != "" {
		t.Error("empty should be empty")
	}
}
