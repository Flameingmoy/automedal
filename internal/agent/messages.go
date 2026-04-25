package agent

// Message is the internal Anthropic-flavoured message shape used by the
// kernel and providers. role ∈ {user, assistant, tool}.
//
//	{role: "user",      content: "<string>" or []Block}
//	{role: "assistant", content: []Block}
//	{role: "tool",      tool_use_id: "<id>", content: "<string>", is_error?: bool}
//
// Content blocks have type ∈ {text, thinking, redacted_thinking, tool_use, tool_result}.
type Message struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"` // string or []Block
	ToolUseID  string         `json:"tool_use_id,omitempty"`
	IsError    bool           `json:"is_error,omitempty"`
}

// Block is one piece of structured assistant/user content.
type Block struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	Thinking  string         `json:"thinking,omitempty"`
	Signature string         `json:"signature,omitempty"`
	Data      string         `json:"data,omitempty"` // for redacted_thinking
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
	// tool_result fields (when present in user-shaped content)
	ToolUseID string `json:"tool_use_id,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`
	Content   any    `json:"content,omitempty"` // for tool_result blocks
}

// PatchDanglingToolCalls appends stub tool-result entries for any
// assistant tool_use blocks that don't have a matching tool message.
// Self-heals an interrupted-tool transcript so the next chat_stream
// doesn't 400 with "tool_use_id has no matching tool_result".
//
// Returns the number of stubs appended (0 if the transcript was clean).
// Port of automedal/agent/messages.py:patch_dangling_tool_calls.
func PatchDanglingToolCalls(messages *[]Message) int {
	answered := answeredIDs(*messages)
	appended := 0

	// Walk backwards to find the most recent assistant turn carrying tool_use.
	for i := len(*messages) - 1; i >= 0; i-- {
		msg := (*messages)[i]
		if msg.Role != "assistant" {
			continue
		}
		blocks, ok := msg.Content.([]Block)
		if !ok {
			return appended
		}
		for _, b := range blocks {
			if b.Type != "tool_use" || b.ID == "" || answered[b.ID] {
				continue
			}
			*messages = append(*messages, Message{
				Role:      "tool",
				ToolUseID: b.ID,
				Content:   "Tool was not executed (interrupted or error).",
				IsError:   true,
			})
			answered[b.ID] = true
			appended++
		}
		return appended // only patch most recent assistant turn
	}
	return appended
}

// answeredIDs collects every tool_use_id that already has a matching
// tool message — either a role=tool entry or a tool_result block under
// a role=user message (Anthropic-shape).
func answeredIDs(messages []Message) map[string]bool {
	ids := map[string]bool{}
	for _, m := range messages {
		switch m.Role {
		case "tool":
			if m.ToolUseID != "" {
				ids[m.ToolUseID] = true
			}
		case "user":
			if blocks, ok := m.Content.([]Block); ok {
				for _, b := range blocks {
					if b.Type == "tool_result" && b.ToolUseID != "" {
						ids[b.ToolUseID] = true
					}
				}
			}
		}
	}
	return ids
}
