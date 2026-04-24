// Command automedal-tui is the Go shell for the AutoMedal workflow.  See
// docs/plans/go-tui-migration.md for the why.  It does three things:
//   1. Render the Home / Dashboard / Run / Help / Knowledge screens.
//   2. Tail agent_loop.events.jsonl with fsnotify.
//   3. Spawn `automedal <cmd>` as a subprocess for anything that isn't UI.
//
// It never imports Python, never speaks to a provider directly, never
// touches Kaggle. The Python kernel owns all of that.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cdharmaraj/automedal-tui/models"
	tea "github.com/charmbracelet/bubbletea"
)

// Root is the top-level tea.Model. It holds the active child model and
// forwards WindowSize / Key / custom messages to it, swapping children on
// SwitchScreenMsg.
type Root struct {
	active   models.Screen
	home     tea.Model
	dash     tea.Model
	run      tea.Model
	help     tea.Model
	knowBase tea.Model

	width, height int
}

func newRoot(initial models.Screen) *Root {
	return &Root{
		active:   initial,
		home:     models.NewHome(),
		help:     models.NewHelp(),
		knowBase: models.NewKnowledge(),
	}
}

func (r *Root) Init() tea.Cmd {
	switch r.active {
	case models.ScreenHome:
		return r.home.Init()
	case models.ScreenDash:
		r.dash = models.NewDash()
		return r.dash.Init()
	}
	return r.home.Init()
}

// current returns a pointer to the tea.Model field matching the current
// screen, or nil if we haven't instantiated it yet.
func (r *Root) current() tea.Model {
	switch r.active {
	case models.ScreenHome:
		return r.home
	case models.ScreenDash:
		return r.dash
	case models.ScreenRun:
		return r.run
	case models.ScreenHelp:
		return r.help
	case models.ScreenKnowledge:
		return r.knowBase
	}
	return r.home
}

func (r *Root) setCurrent(m tea.Model) {
	switch r.active {
	case models.ScreenHome:
		r.home = m
	case models.ScreenDash:
		r.dash = m
	case models.ScreenRun:
		r.run = m
	case models.ScreenHelp:
		r.help = m
	case models.ScreenKnowledge:
		r.knowBase = m
	}
}

func (r *Root) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		r.width, r.height = msg.Width, msg.Height
		// Forward to all instantiated children so their sizes stay in sync.
		var cmds []tea.Cmd
		for _, k := range []models.Screen{models.ScreenHome, models.ScreenDash, models.ScreenRun, models.ScreenHelp, models.ScreenKnowledge} {
			was := r.active
			r.active = k
			m := r.current()
			if m == nil {
				r.active = was
				continue
			}
			nm, cmd := m.Update(msg)
			r.setCurrent(nm)
			if cmd != nil {
				cmds = append(cmds, cmd)
			}
			r.active = was
		}
		return r, tea.Batch(cmds...)

	case models.SwitchScreenMsg:
		// Tear down old state lazily; instantiate new if needed.
		r.active = msg.To
		switch msg.To {
		case models.ScreenDash:
			if r.dash == nil {
				r.dash = models.NewDash()
			}
			init := r.dash.Init()
			// Resend the window-size so the new model lays out correctly.
			resize := func() tea.Msg { return tea.WindowSizeMsg{Width: r.width, Height: r.height} }
			return r, tea.Batch(init, resize)
		case models.ScreenRun:
			r.run = models.NewRun(msg.Verb, msg.Args)
			init := r.run.Init()
			resize := func() tea.Msg { return tea.WindowSizeMsg{Width: r.width, Height: r.height} }
			return r, tea.Batch(init, resize)
		case models.ScreenKnowledge:
			r.knowBase = models.NewKnowledge()
			init := r.knowBase.Init()
			resize := func() tea.Msg { return tea.WindowSizeMsg{Width: r.width, Height: r.height} }
			return r, tea.Batch(init, resize)
		case models.ScreenHelp:
			r.help = models.NewHelp()
			resize := func() tea.Msg { return tea.WindowSizeMsg{Width: r.width, Height: r.height} }
			return r, resize
		case models.ScreenHome:
			// Home is always alive; just re-focus.
			return r, nil
		}
	}

	// Default: forward to current screen.
	cur := r.current()
	if cur == nil {
		return r, nil
	}
	nm, cmd := cur.Update(msg)
	r.setCurrent(nm)
	return r, cmd
}

func (r *Root) View() string {
	cur := r.current()
	if cur == nil {
		return "loading…"
	}
	return cur.View()
}

func main() {
	screen := flag.String("screen", "home", "initial screen: home|dashboard")
	version := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *version {
		fmt.Println("automedal-tui v0.1.0")
		return
	}

	initial := models.ScreenHome
	if *screen == "dashboard" || *screen == "dash" {
		initial = models.ScreenDash
	}

	p := tea.NewProgram(newRoot(initial),
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "tui error:", err)
		os.Exit(1)
	}
}
