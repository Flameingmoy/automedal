package events

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestTailReadsAppendedLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "agent_loop.events.jsonl")
	if err := os.WriteFile(path, []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := Tail(ctx, path, TailOpts{FromStart: true})
	if err != nil {
		t.Fatal(err)
	}

	// Write two records.
	go func() {
		time.Sleep(100 * time.Millisecond)
		f, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
		defer f.Close()
		f.WriteString(`{"kind":"phase_start","phase":"s","t":"2026-04-24T10:00:00Z"}` + "\n")
		f.WriteString(`{"kind":"phase_end","phase":"s","stop":"x","t":"2026-04-24T10:00:01Z"}` + "\n")
	}()

	got := 0
	deadline := time.After(3 * time.Second)
	for got < 2 {
		select {
		case <-deadline:
			t.Fatalf("only read %d events", got)
		case ev, ok := <-ch:
			if !ok {
				t.Fatal("channel closed early")
			}
			if ev.Kind == "" {
				t.Error("empty event kind")
			}
			got++
		}
	}
}

func TestTailHandlesPartialLine(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "agent_loop.events.jsonl")
	if err := os.WriteFile(path, []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ch, err := Tail(ctx, path, TailOpts{FromStart: true})
	if err != nil {
		t.Fatal(err)
	}

	// Write half a line first, then complete it later.
	go func() {
		time.Sleep(50 * time.Millisecond)
		f, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
		f.WriteString(`{"kind":"phase`)
		f.Close()
		time.Sleep(120 * time.Millisecond)
		f2, _ := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
		f2.WriteString(`_start","phase":"s","t":"2026-04-24T10:00:00Z"}` + "\n")
		f2.Close()
	}()

	select {
	case ev := <-ch:
		if ev.Kind != "phase_start" {
			t.Errorf("want phase_start, got %q", ev.Kind)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("never got the completed line")
	}
}
