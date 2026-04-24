// Package models holds the three Bubbletea tea.Model implementations
// (Home, Dashboard, Run) and the cross-model message types that glue them
// together.
package models

import (
	"github.com/cdharmaraj/automedal-tui/events"
	"github.com/cdharmaraj/automedal-tui/proc"
)

// SwitchScreenMsg tells main.go's router to swap the active model.
// The root model listens for this and transitions.
type SwitchScreenMsg struct {
	To      Screen
	Verb    string   // for ScreenRun: the subprocess argv[1]
	Args    []string // for ScreenRun: argv[2:]
}

// Screen identifies one of our top-level screens.
type Screen int

const (
	ScreenHome Screen = iota
	ScreenDash
	ScreenRun
	ScreenHelp
	ScreenKnowledge
)

// EventMsg wraps a JSONL event on its way into the dashboard's Update.
type EventMsg struct{ Ev events.Event }

// TickMsg is what we inject for periodic polling (GPU, leaderboard re-read).
type TickMsg struct{ Kind string }

// RunLineMsg is one line of subprocess output.
type RunLineMsg struct{ Line proc.Line }

// RunExitMsg is the final "child exited" signal.
type RunExitMsg struct{ Exit proc.ExitMsg }
