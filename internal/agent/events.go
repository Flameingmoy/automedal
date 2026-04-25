// Package agent — events: JSONL event sink for the bespoke kernel.
//
// Byte-identical port of automedal/agent/events.py:EventSink. Each
// emit produces one line in agent_loop.events.jsonl with the same key
// order Python's json.dumps produces (dict insertion order: t, phase,
// step, depth, kind, then the kind-specific extras). The internal/ui
// TUI tail (events/format.go) parses the same shape.
//
// A parallel human-readable mirror is appended to agent_loop.log.
package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// utcNow returns "YYYY-MM-DDTHH:MM:SSZ" — matches Python strftime("%Y-%m-%dT%H:%M:%SZ").
func utcNow() string {
	return time.Now().UTC().Format("2006-01-02T15:04:05Z")
}

// preview clips s to n runes, replaces newlines with spaces, trims, and
// suffixes with "…" if cut. Mirrors automedal/agent/events.py:_preview.
func preview(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
	if n <= 0 {
		return s
	}
	if len([]rune(s)) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:n-1]) + "…"
}

// EventSink writes JSONL events + an optional human-readable mirror.
// Caller owns lifecycle: defer Close() after construction.
type EventSink struct {
	JSONLPath  string
	HumanPath  string
	Echo       bool
	Phase      string
	Step       int
	Depth      int

	jsonlFH       io.WriteCloser
	humanFH       io.WriteCloser
	inlineActive  bool
}

// New opens the JSONL + human files for append. Either path may be empty
// to disable that sink. Returns an error if a non-empty path can't be
// opened.
func New(jsonlPath, humanPath string, echo bool) (*EventSink, error) {
	s := &EventSink{
		JSONLPath: jsonlPath,
		HumanPath: humanPath,
		Echo:      echo,
	}
	if jsonlPath != "" {
		if err := os.MkdirAll(filepath.Dir(jsonlPath), 0o755); err != nil {
			return nil, err
		}
		f, err := os.OpenFile(jsonlPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return nil, err
		}
		s.jsonlFH = f
	}
	if humanPath != "" {
		if err := os.MkdirAll(filepath.Dir(humanPath), 0o755); err != nil {
			return nil, err
		}
		f, err := os.OpenFile(humanPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			return nil, err
		}
		s.humanFH = f
	}
	return s, nil
}

// Close releases the file handles. Safe to call multiple times.
func (s *EventSink) Close() error {
	s.endInline()
	var firstErr error
	if s.jsonlFH != nil {
		if err := s.jsonlFH.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		s.jsonlFH = nil
	}
	if s.humanFH != nil {
		if err := s.humanFH.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		s.humanFH = nil
	}
	return firstErr
}

// WithPhase returns a shallow copy scoped to a new phase. Reuses the
// open file handles — do NOT call Close() on the returned child.
func (s *EventSink) WithPhase(phase string) *EventSink {
	c := *s
	c.Phase = phase
	c.Step = 0
	c.inlineActive = false
	return &c
}

// ChildSubagent returns a sink scoped to a sub-agent run (depth+1,
// fresh step counter, phase suffixed with ">label").
func (s *EventSink) ChildSubagent(label string) *EventSink {
	c := s.WithPhase(s.Phase + ">" + label)
	c.Depth = s.Depth + 1
	return c
}

// ── public emit API ─────────────────────────────────────────────────────

func (s *EventSink) PhaseStart(extra map[string]any) {
	s.emit("phase_start", extra)
	s.human(fmt.Sprintf("\n========== phase: %s ==========", s.Phase))
}

// PhaseEnd emits a phase_end event with usage totals + stop reason.
func (s *EventSink) PhaseEnd(stop string, usage *Usage, extra map[string]any) {
	s.endInline()
	body := map[string]any{"stop": stop}
	if usage != nil {
		body["usage"] = map[string]any{"in": usage.InTokens, "out": usage.OutTokens}
	}
	for k, v := range extra {
		body[k] = v
	}
	s.emit("phase_end", body)
	usageStr := "{}"
	if usage != nil {
		usageStr = fmt.Sprintf("{'in': %d, 'out': %d}", usage.InTokens, usage.OutTokens)
	}
	s.human(fmt.Sprintf("  [phase_end] stop=%s usage=%s", stop, usageStr))
}

// StepAdvance bumps the per-phase step counter. Tools / providers don't
// see steps; this is metadata for the JSONL trail.
func (s *EventSink) StepAdvance() { s.Step++ }

// Delta emits an incremental assistant text chunk + streams it to the
// human mirror (no trailing newline; ends inline run).
func (s *EventSink) Delta(text string) {
	if text == "" {
		return
	}
	s.emit("delta", map[string]any{"text": text})
	if s.humanFH != nil {
		fmt.Fprint(s.humanFH, text)
		s.inlineActive = true
	}
	if s.Echo {
		fmt.Fprint(os.Stdout, text)
	}
}

// Thinking emits a thinking-block event (full text), and a length-only
// note in the human mirror.
func (s *EventSink) Thinking(text string) {
	if text == "" {
		return
	}
	s.emit("thinking", map[string]any{"text": text})
	s.endInline()
	s.human(fmt.Sprintf("  [thinking] (%d chars)", len(text)))
}

// ToolStart emits a tool_start event. args is the tool's input dict.
func (s *EventSink) ToolStart(callID, name string, args map[string]any) {
	s.endInline()
	s.emit("tool_start", map[string]any{
		"call_id": callID,
		"name":    name,
		"args":    args,
	})
	parts := make([]string, 0, len(args))
	for k, v := range args {
		parts = append(parts, fmt.Sprintf("%s=%q", k, preview(fmt.Sprintf("%v", v), 40)))
	}
	s.human(fmt.Sprintf("  [tool] %s(%s)", name, strings.Join(parts, ", ")))
}

func (s *EventSink) ToolEnd(callID, name string, ok bool, result string) {
	s.emit("tool_end", map[string]any{
		"call_id": callID,
		"name":    name,
		"ok":      ok,
		"preview": preview(result, 200),
	})
	tag := "ok"
	if !ok {
		tag = "ERROR"
	}
	s.human(fmt.Sprintf("  [tool] %s → %s: %s", name, tag, preview(result, 80)))
}

// Usage emits a usage event. Surfaced in TUI as part of phase_end.
func (s *EventSink) Usage(inTokens, outTokens int) {
	s.emit("usage", map[string]any{"in": inTokens, "out": outTokens})
}

func (s *EventSink) SubagentStart(label, promptPreview string) {
	s.emit("subagent_start", map[string]any{
		"label":  label,
		"prompt": preview(promptPreview, 120),
	})
	s.human(fmt.Sprintf("  [subagent:%s] start — %s", label, preview(promptPreview, 80)))
}

func (s *EventSink) SubagentEnd(label string, ok bool, resultPreview string) {
	s.emit("subagent_end", map[string]any{
		"label":   label,
		"ok":      ok,
		"preview": preview(resultPreview, 200),
	})
	s.human(fmt.Sprintf("  [subagent:%s] end ok=%v", label, ok))
}

// AdvisorConsult records one Kimi K2.6 (or other) consultation.
type AdvisorConsultArgs struct {
	Purpose, Model, Reason, Preview string
	InTokens, OutTokens             int
	Skipped                         bool
}

func (s *EventSink) AdvisorConsult(a AdvisorConsultArgs) {
	s.endInline()
	s.emit("advisor_consult", map[string]any{
		"purpose": a.Purpose,
		"model":   a.Model,
		"in":      a.InTokens,
		"out":     a.OutTokens,
		"skipped": a.Skipped,
		"reason":  a.Reason,
		"preview": preview(a.Preview, 280),
	})
	if a.Skipped {
		reason := a.Reason
		if reason == "" {
			reason = "no_reason"
		}
		s.human(fmt.Sprintf("  [advisor:%s] skipped (%s)", a.Purpose, reason))
	} else {
		s.human(fmt.Sprintf("  [advisor:%s] %s (%d/%d) — %s",
			a.Purpose, a.Model, a.InTokens, a.OutTokens, preview(a.Preview, 120)))
	}
}

// Error emits an error event. `where` describes the call site.
func (s *EventSink) Error(where string, err error) {
	s.endInline()
	typ := "error"
	if err != nil {
		typ = fmt.Sprintf("%T", err)
		// Strip "*" from pointer types and pkg path; keep just the name.
		if i := strings.LastIndex(typ, "."); i >= 0 {
			typ = typ[i+1:]
		}
		typ = strings.TrimPrefix(typ, "*")
	}
	msg := ""
	if err != nil {
		msg = err.Error()
	}
	s.emit("error", map[string]any{"where": where, "type": typ, "msg": msg})
	s.human(fmt.Sprintf("  [error] %s: %s: %s", where, typ, msg))
}

// Notice emits a neutral-severity informational event. Used by retry,
// self-healing paths, and the doom-loop detector.
func (s *EventSink) Notice(tag, message string) {
	s.endInline()
	s.emit("notice", map[string]any{"tag": tag, "message": message})
	s.human(fmt.Sprintf("  [%s] %s", tag, message))
}

// ToolLog is a back-compat alias matching ml-intern's interface — same
// payload as Notice with a tool-derived tag.
func (s *EventSink) ToolLog(tool, log string) { s.Notice(tool, log) }

// ── internals ───────────────────────────────────────────────────────────

// emit writes one JSONL record. Key order matches the Python emitter:
// t, phase, step, depth, kind, then sorted extras (Python relies on
// insertion order; we sort the extras to keep diffs stable).
func (s *EventSink) emit(kind string, extra map[string]any) {
	if s.jsonlFH == nil {
		return
	}
	var buf bytes.Buffer
	buf.WriteByte('{')
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	writeKV := func(first bool, k string, v any) error {
		if !first {
			buf.WriteByte(',')
		}
		// Encode the key as a JSON string.
		kb, _ := json.Marshal(k)
		buf.Write(kb)
		buf.WriteByte(':')
		// json.Encoder always appends a newline; strip it.
		var vb bytes.Buffer
		ve := json.NewEncoder(&vb)
		ve.SetEscapeHTML(false)
		if err := ve.Encode(v); err != nil {
			return err
		}
		out := vb.Bytes()
		if len(out) > 0 && out[len(out)-1] == '\n' {
			out = out[:len(out)-1]
		}
		buf.Write(out)
		return nil
	}
	_ = enc // keep linter happy

	_ = writeKV(true, "t", utcNow())
	_ = writeKV(false, "phase", s.Phase)
	_ = writeKV(false, "step", s.Step)
	_ = writeKV(false, "depth", s.Depth)
	_ = writeKV(false, "kind", kind)

	// Extras: preserve a stable order. We use the same insertion order
	// callers use, which Python also relies on (3.7+ dict order is the
	// insertion order). Iterate in the order each caller passed them
	// by sorting keys lexicographically except when an explicit order
	// was provided. For determinism + identical TUI rendering we use a
	// canonical order for the kinds the readers care about.
	for _, k := range orderedKeys(kind, extra) {
		_ = writeKV(false, k, extra[k])
	}

	buf.WriteByte('}')
	buf.WriteByte('\n')
	_, _ = s.jsonlFH.Write(buf.Bytes())
}

// orderedKeys returns the extra-keys in the order Python emits them
// for the given kind. Falls back to insertion-style sorted order for
// anything we haven't pinned.
func orderedKeys(kind string, extra map[string]any) []string {
	switch kind {
	case "phase_end":
		return present([]string{"stop", "usage"}, extra, true)
	case "tool_start":
		return present([]string{"call_id", "name", "args"}, extra, true)
	case "tool_end":
		return present([]string{"call_id", "name", "ok", "preview"}, extra, true)
	case "delta", "thinking":
		return present([]string{"text"}, extra, true)
	case "usage":
		return present([]string{"in", "out"}, extra, true)
	case "subagent_start":
		return present([]string{"label", "prompt"}, extra, true)
	case "subagent_end":
		return present([]string{"label", "ok", "preview"}, extra, true)
	case "advisor_consult":
		return present([]string{"purpose", "model", "in", "out", "skipped", "reason", "preview"}, extra, true)
	case "error":
		return present([]string{"where", "type", "msg"}, extra, true)
	case "notice":
		return present([]string{"tag", "message"}, extra, true)
	}
	// Unknown kind — sort lexicographically for determinism.
	out := make([]string, 0, len(extra))
	for k := range extra {
		out = append(out, k)
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

// present returns canonical keys that exist in extra, in canonical order.
// If includeUnlisted, any extras not in the canonical list are appended
// in lexicographic order (so callers can pass through extra fields).
func present(canonical []string, extra map[string]any, includeUnlisted bool) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(canonical)+len(extra))
	for _, k := range canonical {
		if _, ok := extra[k]; ok {
			out = append(out, k)
			seen[k] = true
		}
	}
	if !includeUnlisted {
		return out
	}
	rest := []string{}
	for k := range extra {
		if !seen[k] {
			rest = append(rest, k)
		}
	}
	for i := 1; i < len(rest); i++ {
		for j := i; j > 0 && rest[j-1] > rest[j]; j-- {
			rest[j-1], rest[j] = rest[j], rest[j-1]
		}
	}
	return append(out, rest...)
}

func (s *EventSink) human(line string) {
	if s.humanFH == nil {
		return
	}
	s.endInline()
	fmt.Fprintln(s.humanFH, line)
	if s.Echo {
		fmt.Fprintln(os.Stdout, line)
	}
}

func (s *EventSink) endInline() {
	if s.inlineActive && s.humanFH != nil {
		fmt.Fprintln(s.humanFH)
		s.inlineActive = false
		if s.Echo {
			fmt.Fprintln(os.Stdout)
		}
	}
}

// Usage is the token accounting for one chat turn. The agent kernel
// aggregates this across steps.
type Usage struct {
	InTokens  int `json:"in_tokens"`
	OutTokens int `json:"out_tokens"`
}
