package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// readJSONL parses a JSONL file into a slice of generic maps.
func readJSONL(t *testing.T, path string) []map[string]any {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimRight(string(b), "\n"), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("bad JSON %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

func TestEventSinkPhaseLifecycle(t *testing.T) {
	dir := t.TempDir()
	jpath := filepath.Join(dir, "events.jsonl")
	hpath := filepath.Join(dir, "log.txt")
	s, err := New(jpath, hpath, false)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	p := s.WithPhase("strategist")
	p.PhaseStart(nil)
	p.StepAdvance()
	p.ToolStart("c1", "write_file", map[string]any{"path": "knowledge.md", "content": "x"})
	p.ToolEnd("c1", "write_file", true, "wrote knowledge.md")
	p.Usage(120, 45)
	p.PhaseEnd("end_turn", &Usage{InTokens: 120, OutTokens: 45}, nil)
	s.Close()

	events := readJSONL(t, jpath)
	if len(events) != 5 {
		t.Fatalf("want 5 events, got %d", len(events))
	}
	wantKinds := []string{"phase_start", "tool_start", "tool_end", "usage", "phase_end"}
	for i, want := range wantKinds {
		if events[i]["kind"] != want {
			t.Errorf("event[%d].kind = %v, want %s", i, events[i]["kind"], want)
		}
		if events[i]["phase"] != "strategist" {
			t.Errorf("event[%d].phase missing", i)
		}
	}
	// tool_end fields
	if events[2]["ok"] != true || events[2]["name"] != "write_file" {
		t.Errorf("tool_end shape wrong: %v", events[2])
	}
	// phase_end usage nested
	usage, ok := events[4]["usage"].(map[string]any)
	if !ok || usage["in"].(float64) != 120 {
		t.Errorf("phase_end.usage shape wrong: %v", events[4]["usage"])
	}
}

func TestEventSinkAdvisorConsult(t *testing.T) {
	dir := t.TempDir()
	jpath := filepath.Join(dir, "events.jsonl")
	s, _ := New(jpath, "", false)
	defer s.Close()

	s.AdvisorConsult(AdvisorConsultArgs{
		Purpose:   "stagnation",
		Model:     "kimi-k2.6",
		InTokens:  1500,
		OutTokens: 320,
		Skipped:   false,
		Preview:   "Switch to dart-style diff edits.",
	})
	s.AdvisorConsult(AdvisorConsultArgs{
		Purpose: "audit", Model: "kimi-k2.6", Skipped: true, Reason: "budget:iter",
	})
	s.Close()

	events := readJSONL(t, jpath)
	if len(events) != 2 {
		t.Fatalf("want 2, got %d", len(events))
	}
	if events[0]["model"] != "kimi-k2.6" || events[0]["in"].(float64) != 1500 {
		t.Errorf("consult shape wrong: %v", events[0])
	}
	if events[1]["skipped"] != true || events[1]["reason"] != "budget:iter" {
		t.Errorf("skipped consult shape wrong: %v", events[1])
	}
}

func TestEventSinkSubagentDepth(t *testing.T) {
	dir := t.TempDir()
	jpath := filepath.Join(dir, "events.jsonl")
	s, _ := New(jpath, "", false)
	defer s.Close()

	parent := s.WithPhase("researcher")
	parent.PhaseStart(nil)
	child := parent.ChildSubagent("webfetch")
	child.PhaseStart(nil)
	child.PhaseEnd("end_turn", nil, nil)
	parent.PhaseEnd("end_turn", nil, nil)
	s.Close()

	events := readJSONL(t, jpath)
	if len(events) != 4 {
		t.Fatalf("want 4 events, got %d", len(events))
	}
	if events[0]["depth"].(float64) != 0 || events[1]["depth"].(float64) != 1 {
		t.Errorf("depth wrong: outer=%v inner=%v", events[0]["depth"], events[1]["depth"])
	}
	if events[1]["phase"] != "researcher>webfetch" {
		t.Errorf("nested phase name wrong: %v", events[1]["phase"])
	}
}

func TestEventSinkNoticeAndToolLogAlias(t *testing.T) {
	dir := t.TempDir()
	jpath := filepath.Join(dir, "events.jsonl")
	s, _ := New(jpath, "", false)
	defer s.Close()

	s.Notice("retry", "transient timeout — backing off 5s")
	s.ToolLog("doom_loop", "cycle [a,b] detected")
	s.Close()

	events := readJSONL(t, jpath)
	for i, want := range []struct {
		tag string
	}{{tag: "retry"}, {tag: "doom_loop"}} {
		if events[i]["kind"] != "notice" || events[i]["tag"] != want.tag {
			t.Errorf("event[%d] wrong: %v", i, events[i])
		}
	}
}

func TestEventSinkDeltaInlinesHumanLog(t *testing.T) {
	dir := t.TempDir()
	jpath := filepath.Join(dir, "events.jsonl")
	hpath := filepath.Join(dir, "log.txt")
	s, _ := New(jpath, hpath, false)

	s.Delta("hello ")
	s.Delta("world")
	s.Notice("end", "finalize") // ends inline run with newline before [end]
	s.Close()

	body, _ := os.ReadFile(hpath)
	str := string(body)
	if !strings.Contains(str, "hello world") {
		t.Errorf("delta inline wrong: %q", str)
	}
	if !strings.Contains(str, "[end] finalize") {
		t.Errorf("notice not rendered: %q", str)
	}
	// The inline run ends with a newline before the indented [end] line.
	if !strings.Contains(str, "\n  [end]") {
		t.Errorf("missing newline before indented [end]: %q", str)
	}
}

func TestEventSinkPreserveKeyOrder(t *testing.T) {
	dir := t.TempDir()
	jpath := filepath.Join(dir, "events.jsonl")
	s, _ := New(jpath, "", false)

	s.PhaseStart(nil)
	s.Close()

	b, _ := os.ReadFile(jpath)
	line := strings.TrimSpace(string(b))
	// Required leading order: t, phase, step, depth, kind.
	if !strings.HasPrefix(line, `{"t":`) {
		t.Errorf("first key not 't': %q", line)
	}
	for _, want := range []string{`"phase":`, `"step":`, `"depth":`, `"kind":`} {
		if !strings.Contains(line, want) {
			t.Errorf("missing %s in %q", want, line)
		}
	}
	tIdx := strings.Index(line, `"t":`)
	pIdx := strings.Index(line, `"phase":`)
	sIdx := strings.Index(line, `"step":`)
	dIdx := strings.Index(line, `"depth":`)
	kIdx := strings.Index(line, `"kind":`)
	if !(tIdx < pIdx && pIdx < sIdx && sIdx < dIdx && dIdx < kIdx) {
		t.Errorf("wrong key order: %q", line)
	}
}

func TestPreviewClipping(t *testing.T) {
	if got := preview("hello\nworld", 0); got != "hello world" {
		t.Errorf("zero n: %q", got)
	}
	long := strings.Repeat("x", 200)
	got := preview(long, 50)
	if len([]rune(got)) != 50 {
		t.Errorf("rune len: %d", len([]rune(got)))
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("ellipsis missing: %q", got)
	}
}
