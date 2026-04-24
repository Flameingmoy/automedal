package events

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// Format renders an Event into a single display line matching the
// conventions of tui/sources/events_jsonl.py:_render. Returns ("", false)
// when the event shouldn't be shown (successful tool_end, empty deltas,
// raw usage records).
func Format(e *Event) (string, bool) {
	ts := shortTime(e.T)
	indent := strings.Repeat("  ", e.Depth+1)
	tag := ""
	if e.Phase != "" {
		tag = "[" + e.Phase + "]"
	}

	switch e.Kind {
	case "phase_start":
		return fmt.Sprintf("%s %s ── start ──", ts, tag), true
	case "phase_end":
		utxt := ""
		if in := e.InTokens(); in > 0 || e.OutTokens() > 0 {
			utxt = fmt.Sprintf(" usage=%d/%d", in, e.OutTokens())
		}
		return fmt.Sprintf("%s %s ── end (stop=%s%s) ──", ts, tag, e.Stop, utxt), true
	case "tool_start":
		return fmt.Sprintf("%s %s[%s] %s", ts, indent, e.Name, formatArgs(e.Args)), true
	case "tool_end":
		if e.OK {
			return "", false
		}
		prev := clip(strings.ReplaceAll(e.Preview, "\n", " "), 100)
		return fmt.Sprintf("%s %s[%s] ERROR: %s", ts, indent, e.Name, prev), true
	case "delta":
		txt := strings.ReplaceAll(e.Text, "\n", " ")
		if strings.TrimSpace(txt) == "" {
			return "", false
		}
		return fmt.Sprintf("%s %s", ts, clip(txt, 200)), true
	case "thinking":
		return fmt.Sprintf("%s %s[thinking] (%d chars)", ts, indent, len(e.Text)), true
	case "subagent_start":
		return fmt.Sprintf("%s %s[subagent:%s] start — %s", ts, indent, e.Label, clip(e.Prompt, 100)), true
	case "subagent_end":
		return fmt.Sprintf("%s %s[subagent:%s] end ok=%v", ts, indent, e.Label, e.OK), true
	case "advisor_consult":
		if e.Skipped {
			r := e.Reason
			if r == "" {
				r = "no_reason"
			}
			return fmt.Sprintf("%s %s[advisor:%s] skipped (%s)", ts, indent, e.Purpose, r), true
		}
		return fmt.Sprintf(
			"%s %s[advisor:%s] %s (%d/%d) — %s",
			ts, indent, e.Purpose, e.Model, e.InTokens(), e.OutTokens(), clip(e.Preview, 120),
		), true
	case "usage":
		return "", false
	case "error":
		return fmt.Sprintf("%s %s[error] %s: %s", ts, indent, e.Where, e.Msg), true
	case "notice":
		return fmt.Sprintf("%s %s[%s] %s", ts, indent, e.Tag, e.Message), true
	}
	return "", false
}

func formatArgs(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return clip(string(raw), 80)
	}
	var parts []string
	for k, v := range m {
		sv := fmt.Sprintf("%v", v)
		sv = strings.ReplaceAll(sv, "\n", " ")
		parts = append(parts, fmt.Sprintf("%s=%s", k, clip(sv, 60)))
	}
	return strings.Join(parts, ", ")
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

// shortTime renders the HH:MM:SS out of an ISO-8601 UTC string. Empty → "--:--:--".
func shortTime(iso string) string {
	if iso == "" {
		return "--:--:--"
	}
	t, err := time.Parse("2006-01-02T15:04:05Z", iso)
	if err != nil {
		return "--:--:--"
	}
	return t.Local().Format("15:04:05")
}
