// Command automedal-tui is the v2 Go shell for AutoMedal.  See
// design/AutoMedal TUI v2.html for the visual reference.  Every screen
// shares: a top spring-nav strip, a status bar, and a footer hint row.
//
// Responsibilities:
//   1. Render Home / Dashboard / Run / Timeline / Config / Knowledge / Help.
//   2. Tail agent_loop.events.jsonl with fsnotify.
//   3. Spawn `automedal <cmd>` as a subprocess for any non-UI command.
//
// It never imports Python, never speaks to a provider directly, never
// touches Kaggle.  The Go control plane owns all of that.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/Flameingmoy/automedal/internal/ui/components"
	"github.com/Flameingmoy/automedal/internal/ui/models"
	"github.com/Flameingmoy/automedal/internal/ui/theme"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Root holds the active child model, instantiates lazy ones on first
// nav, and renders the chrome (nav strip) above whatever the child
// returned.
type Root struct {
	active models.Screen

	home     models.HomeModel
	dash     tea.Model
	run      tea.Model
	help     models.HelpModel
	knowBase tea.Model
	timeline tea.Model
	config   tea.Model

	width, height int

	nav *components.NavBar
}

func newRoot(initial models.Screen) *Root {
	nav := components.NewNavBar([]components.NavTab{
		{ID: "home", Label: "home"},
		{ID: "dashboard", Label: "dashboard"},
		{ID: "run", Label: "run"},
		{ID: "timeline", Label: "timeline"},
		{ID: "config", Label: "config"},
		{ID: "help", Label: "help"},
	})
	nav.SetActive(navIDFor(initial))
	return &Root{
		active:   initial,
		home:     models.NewHome(),
		help:     models.NewHelp(),
		knowBase: models.NewKnowledge(),
		nav:      nav,
	}
}

func navIDFor(s models.Screen) string {
	switch s {
	case models.ScreenDash:
		return "dashboard"
	case models.ScreenRun:
		return "run"
	case models.ScreenHelp:
		return "help"
	case models.ScreenKnowledge:
		return "home"
	case models.ScreenTimeline:
		return "timeline"
	case models.ScreenConfig:
		return "config"
	}
	return "home"
}

func (r *Root) Init() tea.Cmd {
	cmds := []tea.Cmd{components.NavTickCmd()}
	switch r.active {
	case models.ScreenHome:
		cmds = append(cmds, r.home.Init())
	case models.ScreenDash:
		r.dash = models.NewDash()
		cmds = append(cmds, r.dash.Init())
	}
	return tea.Batch(cmds...)
}

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
	case models.ScreenTimeline:
		return r.timeline
	case models.ScreenConfig:
		return r.config
	}
	return r.home
}

func (r *Root) setCurrent(m tea.Model) {
	switch r.active {
	case models.ScreenHome:
		r.home = m.(models.HomeModel)
	case models.ScreenDash:
		r.dash = m
	case models.ScreenRun:
		r.run = m
	case models.ScreenHelp:
		r.help = m.(models.HelpModel)
	case models.ScreenKnowledge:
		r.knowBase = m
	case models.ScreenTimeline:
		r.timeline = m
	case models.ScreenConfig:
		r.config = m
	}
}

func (r *Root) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		r.width, r.height = msg.Width, msg.Height
		r.nav.SetWidth(msg.Width)
		// Pass child the inner size (header eats 2 rows).
		inner := tea.WindowSizeMsg{Width: msg.Width, Height: msg.Height - 2}
		var cmds []tea.Cmd
		for _, k := range []models.Screen{
			models.ScreenHome, models.ScreenDash, models.ScreenRun,
			models.ScreenHelp, models.ScreenKnowledge,
			models.ScreenTimeline, models.ScreenConfig,
		} {
			was := r.active
			r.active = k
			m := r.current()
			if m == nil {
				r.active = was
				continue
			}
			nm, cmd := m.Update(inner)
			r.setCurrent(nm)
			if cmd != nil {
				cmds = append(cmds, cmd)
			}
			r.active = was
		}
		return r, tea.Batch(cmds...)

	case components.NavTickMsg:
		moved := r.nav.Tick()
		if moved {
			return r, components.NavTickCmd()
		}
		// Spring at rest — keep ticking at low rate to stay responsive
		// when the user switches tabs again.
		return r, components.NavTickCmd()

	case models.SwitchScreenMsg:
		r.active = msg.To
		r.nav.SetActive(navIDFor(msg.To))
		// Instantiate (or refresh) the target screen, then forward the
		// current size SYNCHRONOUSLY so the very first View() has
		// non-zero width — otherwise children fall back to their
		// "loading…" guard for one frame.
		switch msg.To {
		case models.ScreenDash:
			if r.dash == nil {
				r.dash = models.NewDash()
			}
		case models.ScreenRun:
			r.run = models.NewRun(msg.Verb, msg.Args)
		case models.ScreenKnowledge:
			r.knowBase = models.NewKnowledge()
		case models.ScreenHelp:
			r.help = models.NewHelp()
		case models.ScreenTimeline:
			r.timeline = models.NewTimeline()
		case models.ScreenConfig:
			r.config = models.NewConfig()
		case models.ScreenHome:
			return r, components.NavTickCmd()
		}
		cur := r.current()
		var initCmd tea.Cmd
		if cur != nil {
			initCmd = cur.Init()
			// Sync size into the new screen before the next paint.
			if r.width > 0 && r.height > 0 {
				inner := tea.WindowSizeMsg{Width: r.width, Height: r.height - 2}
				nm, _ := cur.Update(inner)
				r.setCurrent(nm)
			}
		}
		return r, tea.Batch(initCmd, components.NavTickCmd())
	}

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
	bg := lipgloss.NewStyle().Background(lipgloss.Color(theme.ColorBg))
	return bg.Render(r.nav.Render() + "\n" + cur.View())
}

func main() {
	screen := flag.String("screen", "home", "initial screen: home|dashboard|timeline|config")
	version := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *version {
		fmt.Println("automedal-tui v2.0.0")
		return
	}

	initial := models.ScreenHome
	switch *screen {
	case "dashboard", "dash":
		initial = models.ScreenDash
	case "timeline":
		initial = models.ScreenTimeline
	case "config":
		initial = models.ScreenConfig
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
