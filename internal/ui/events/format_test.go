package events

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFormatPhaseStart(t *testing.T) {
	e := &Event{T: "2026-04-24T10:32:14Z", Phase: "strategist", Kind: "phase_start"}
	got, ok := Format(e)
	if !ok {
		t.Fatal("expected ok=true for phase_start")
	}
	if !strings.Contains(got, "strategist") || !strings.Contains(got, "start") {
		t.Errorf("bad format: %q", got)
	}
}

func TestFormatToolEndSuccessSuppressed(t *testing.T) {
	e := &Event{Kind: "tool_end", Name: "write_file", OK: true, Preview: "ok"}
	_, ok := Format(e)
	if ok {
		t.Error("successful tool_end should not render")
	}
}

func TestFormatToolEndFailureVisible(t *testing.T) {
	e := &Event{Kind: "tool_end", Name: "write_file", OK: false, Preview: "boom"}
	got, ok := Format(e)
	if !ok || !strings.Contains(got, "ERROR") {
		t.Errorf("failed tool_end should render ERROR: %q (%v)", got, ok)
	}
}

func TestFormatAdvisorSkipped(t *testing.T) {
	e := &Event{Kind: "advisor_consult", Purpose: "audit", Model: "kimi", Skipped: true, Reason: "budget:iter"}
	got, ok := Format(e)
	if !ok || !strings.Contains(got, "skipped (budget:iter)") {
		t.Errorf("advisor-skipped: %q (%v)", got, ok)
	}
}

func TestFormatAdvisorConsult(t *testing.T) {
	e := &Event{Kind: "advisor_consult", Purpose: "audit", Model: "kimi", InT: 1200, OutT: 512, Preview: "switch to dart"}
	got, ok := Format(e)
	if !ok {
		t.Fatal("advisor_consult should render")
	}
	for _, sub := range []string{"audit", "kimi", "1200/512", "switch"} {
		if !strings.Contains(got, sub) {
			t.Errorf("missing %q in %q", sub, got)
		}
	}
}

func TestFormatUsageSuppressed(t *testing.T) {
	e := &Event{Kind: "usage", InT: 100, OutT: 50}
	_, ok := Format(e)
	if ok {
		t.Error("usage events should not render (surfaced in phase_end)")
	}
}

func TestUsageInlineVsRecord(t *testing.T) {
	// phase_end style: usage nested
	body := []byte(`{"kind":"phase_end","phase":"x","stop":"end_turn","usage":{"in":1,"out":2}}`)
	var ev Event
	if err := json.Unmarshal(body, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.InTokens() != 1 || ev.OutTokens() != 2 {
		t.Errorf("nested usage: got (%d,%d)", ev.InTokens(), ev.OutTokens())
	}

	// usage-kind record: flat. Use a fresh Event — InTokens() prefers the
	// nested Usage pointer when set, which would leak from the previous
	// case if we re-used the variable. (Real code always unmarshals into
	// a fresh Event per line; see tail.go:emit.)
	ev = Event{}
	body = []byte(`{"kind":"usage","in":11,"out":22}`)
	if err := json.Unmarshal(body, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.InTokens() != 11 || ev.OutTokens() != 22 {
		t.Errorf("flat usage: got (%d,%d)", ev.InTokens(), ev.OutTokens())
	}
}

func TestBasePhaseStripsSubagent(t *testing.T) {
	e := &Event{Phase: "researcher>webfetch"}
	if got := e.BasePhase(); got != "researcher" {
		t.Errorf("want researcher, got %q", got)
	}
}
