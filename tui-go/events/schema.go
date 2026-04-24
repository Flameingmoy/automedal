// Package events mirrors the JSONL schema written by
// automedal/agent/events.py:EventSink.  Kept permissive on purpose —
// the Python side owns the spec, we just read what it emits.
package events

import (
	"encoding/json"
)

// Event is one JSON line. Fields that don't appear are zero-valued.
//
// Top-level always-present fields: t, phase, step, depth, kind.
// Kind-specific fields follow.
type Event struct {
	// Always present on every record.
	T     string `json:"t"` // ISO-8601 UTC
	Phase string `json:"phase,omitempty"`
	Step  int    `json:"step,omitempty"`
	Depth int    `json:"depth,omitempty"`
	Kind  string `json:"kind"` // phase_start|phase_end|delta|thinking|tool_start|tool_end|usage|subagent_start|subagent_end|advisor_consult|error|notice

	// --- phase_end / usage ---
	Stop  string `json:"stop,omitempty"`
	Usage *Usage `json:"usage,omitempty"`
	InT   int    `json:"in,omitempty"`  // when `usage` is inlined
	OutT  int    `json:"out,omitempty"` // when `usage` is inlined

	// --- delta / thinking ---
	Text string `json:"text,omitempty"`

	// --- tool_start / tool_end ---
	CallID  string          `json:"call_id,omitempty"`
	Name    string          `json:"name,omitempty"`
	Args    json.RawMessage `json:"args,omitempty"`
	OK      bool            `json:"ok,omitempty"`
	Preview string          `json:"preview,omitempty"`

	// --- subagent_start / subagent_end ---
	Label  string `json:"label,omitempty"`
	Prompt string `json:"prompt,omitempty"`

	// --- advisor_consult ---
	Purpose string `json:"purpose,omitempty"`
	Model   string `json:"model,omitempty"`
	Skipped bool   `json:"skipped,omitempty"`
	Reason  string `json:"reason,omitempty"`

	// --- error ---
	Where string `json:"where,omitempty"`
	Type  string `json:"type,omitempty"`
	Msg   string `json:"msg,omitempty"`

	// --- notice / retry ---
	Tag     string `json:"tag,omitempty"`
	Message string `json:"message,omitempty"`
}

// Usage is emitted either inline in `phase_end` under `usage`, or as its
// own "usage" record with flat `in`/`out` ints.
type Usage struct {
	In  int `json:"in"`
	Out int `json:"out"`
}

// InTokens / OutTokens normalize the two usage shapes we see in the wild.
func (e *Event) InTokens() int {
	if e.Usage != nil {
		return e.Usage.In
	}
	return e.InT
}
func (e *Event) OutTokens() int {
	if e.Usage != nil {
		return e.Usage.Out
	}
	return e.OutT
}

// BasePhase strips the ">subagent" suffix (EventSink.child_subagent
// prepends the parent's phase as "parent>child").
func (e *Event) BasePhase() string {
	for i := 0; i < len(e.Phase); i++ {
		if e.Phase[i] == '>' {
			return e.Phase[:i]
		}
	}
	return e.Phase
}
