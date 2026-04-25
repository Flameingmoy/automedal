package agent

import (
	"fmt"
	"strings"
)

// FriendlyError returns a user-facing fix string for known provider error
// patterns, or an empty string when none match. Port of
// automedal/agent/errors.py:friendly_error.
func FriendlyError(err error) string {
	if err == nil {
		return ""
	}
	s := strings.ToLower(err.Error())

	if containsAny(s, "unauthorized", "invalid x-api-key", "invalid api key", "401") {
		return "Authentication failed — your API key is missing or invalid.\n\n" +
			"Fix:\n" +
			"  • opencode-go:  export OPENCODE_API_KEY=...\n" +
			"  • anthropic:    export ANTHROPIC_API_KEY=sk-ant-...\n" +
			"  • openai:       export OPENAI_API_KEY=sk-...\n\n" +
			"Add it to ~/.automedal/.env or a project .env file if you want it persistent."
	}
	if (strings.Contains(s, "insufficient") && strings.Contains(s, "credit")) ||
		strings.Contains(s, "insufficient_quota") || strings.Contains(s, "402") {
		return "Out of credits at the provider. Check your balance at\n" +
			"  • opencode.ai dashboard (for OPENCODE_API_KEY)\n" +
			"  • console.anthropic.com (for ANTHROPIC_API_KEY)"
	}
	if strings.Contains(s, "model_not_found") ||
		(strings.Contains(s, "model") && (strings.Contains(s, "not found") || strings.Contains(s, "does not exist"))) {
		return "Model id not recognized by the provider.\n" +
			"  • `automedal models` lists available models cached from opencode-go.\n" +
			"  • For anthropic, use a current id (e.g. claude-opus-4-7, claude-sonnet-4-6)."
	}
	if strings.Contains(s, "not supported by provider") || strings.Contains(s, "no provider supports") {
		return "This model isn't served by the provider you pinned.\n" +
			"Drop any `:provider` suffix to let routing pick automatically."
	}
	if strings.Contains(s, "context") &&
		(strings.Contains(s, "exceed") || strings.Contains(s, "too long") || strings.Contains(s, "too many tokens")) {
		return "Context window exceeded. The conversation is longer than the model accepts.\n" +
			"Tier 2 context compaction will fix this automatically. For now, start a fresh run."
	}
	return ""
}

// FormatError composes a user-facing message: friendly explanation +
// raw error line. Always returns a non-empty string.
func FormatError(err error) string {
	if err == nil {
		return ""
	}
	raw := fmt.Sprintf("%T: %v", err, err)
	friendly := FriendlyError(err)
	if friendly == "" {
		return raw
	}
	return friendly + "\n\n[raw] " + raw
}

func containsAny(s string, needles ...string) bool {
	for _, n := range needles {
		if strings.Contains(s, n) {
			return true
		}
	}
	return false
}
