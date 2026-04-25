// Package paths resolves every file path AutoMedal touches — both in dev
// mode (running from a repo checkout) and user mode (installed binary in
// a competition directory).
//
// Port of automedal/paths.py:Layout. See the Python docstring for the
// historical design rationale. Key differences from Python:
//
//   - The Go binary is statically linked; there is no "installed package"
//     to find at runtime. Prompts and templates are embedded with go:embed
//     in their respective packages, so the legacy PromptsDir / TemplatesDir
//     / HarnessDir / ScoutDir properties are removed.
//   - Mode detection still looks for dev-repo markers so AutoMedal running
//     inside its own source tree keeps flat layout.
package paths

import (
	"os"
	"path/filepath"
)

// Mode is either "dev" (repo checkout, flat layout) or "user" (installed,
// data under .automedal/).
type Mode string

const (
	ModeDev  Mode = "dev"
	ModeUser Mode = "user"
)

// detectMode returns "dev" if cwd looks like the AutoMedal source tree,
// "user" otherwise. Honours AUTOMEDAL_DEV (explicit dev override) and
// AUTOMEDAL_MODE (explicit override of either mode).
func detectMode(cwd string) Mode {
	if m := os.Getenv("AUTOMEDAL_MODE"); m == string(ModeDev) || m == string(ModeUser) {
		return Mode(m)
	}
	if os.Getenv("AUTOMEDAL_DEV") != "" {
		return ModeDev
	}
	// Repo markers — either the Python package or the Go control plane.
	if exists(filepath.Join(cwd, "pyproject.toml")) &&
		exists(filepath.Join(cwd, "automedal", "run_loop.py")) {
		return ModeDev
	}
	if exists(filepath.Join(cwd, "go.mod")) &&
		exists(filepath.Join(cwd, "cmd", "automedal", "main.go")) {
		return ModeDev
	}
	return ModeUser
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// Layout is a centralised path resolver.
type Layout struct {
	Cwd  string
	Mode Mode
}

// New auto-detects mode from os.Getwd(). Returns a layout whose paths
// reflect the current working directory and detected mode.
func New() (*Layout, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	abs, err := filepath.Abs(cwd)
	if err != nil {
		return nil, err
	}
	return &Layout{Cwd: abs, Mode: detectMode(abs)}, nil
}

// NewAt constructs a Layout with explicit cwd + optional mode override.
// Pass an empty Mode to auto-detect.
func NewAt(cwd string, mode Mode) (*Layout, error) {
	abs, err := filepath.Abs(cwd)
	if err != nil {
		return nil, err
	}
	m := mode
	if m == "" {
		m = detectMode(abs)
	}
	return &Layout{Cwd: abs, Mode: m}, nil
}

// ── always visible (same in both modes) ──────────────────────────────────

func (l *Layout) DataDir() string        { return filepath.Join(l.Cwd, "data") }
func (l *Layout) SubmissionsDir() string { return filepath.Join(l.Cwd, "submissions") }
func (l *Layout) JournalDir() string     { return filepath.Join(l.Cwd, "journal") }
func (l *Layout) KnowledgeMD() string    { return filepath.Join(l.Cwd, "knowledge.md") }
func (l *Layout) QueueMD() string        { return filepath.Join(l.Cwd, "experiment_queue.md") }
func (l *Layout) ResearchMD() string     { return filepath.Join(l.Cwd, "research_notes.md") }

// ResultsTSV: agent/results.tsv in dev; results.tsv at root in user mode.
func (l *Layout) ResultsTSV() string {
	if l.Mode == ModeDev {
		return filepath.Join(l.Cwd, "agent", "results.tsv")
	}
	return filepath.Join(l.Cwd, "results.tsv")
}

// ── hidden in user mode; flat in dev mode ────────────────────────────────

func (l *Layout) HiddenRoot() string {
	if l.Mode == ModeUser {
		return filepath.Join(l.Cwd, ".automedal")
	}
	return l.Cwd
}

func (l *Layout) AgentDir() string {
	if l.Mode == ModeUser {
		return filepath.Join(l.HiddenRoot(), "agent")
	}
	return filepath.Join(l.Cwd, "agent")
}

func (l *Layout) TrainPy() string   { return filepath.Join(l.AgentDir(), "train.py") }
func (l *Layout) PreparePy() string { return filepath.Join(l.AgentDir(), "prepare.py") }

func (l *Layout) ConfigYAML() string {
	if l.Mode == ModeUser {
		return filepath.Join(l.HiddenRoot(), "configs", "competition.yaml")
	}
	return filepath.Join(l.Cwd, "configs", "competition.yaml")
}

func (l *Layout) AgentsMD() string {
	if l.Mode == ModeUser {
		return filepath.Join(l.HiddenRoot(), "AGENTS.md")
	}
	return filepath.Join(l.Cwd, "AGENTS.md")
}

// LogFile: hidden logs/ dir in user mode; AUTOMEDAL_LOG_FILE or LOG_FILE
// env override in dev mode, falling back to cwd/agent_loop.log.
func (l *Layout) LogFile() string {
	if l.Mode == ModeUser {
		return filepath.Join(l.HiddenRoot(), "logs", "agent_loop.log")
	}
	if v := os.Getenv("AUTOMEDAL_LOG_FILE"); v != "" {
		return v
	}
	if v := os.Getenv("LOG_FILE"); v != "" {
		return v
	}
	return filepath.Join(l.Cwd, "agent_loop.log")
}

// EventsFile: JSONL event sink. Env override AUTOMEDAL_EVENTS_FILE in dev;
// hidden in user mode.
func (l *Layout) EventsFile() string {
	if l.Mode == ModeUser {
		return filepath.Join(l.HiddenRoot(), "logs", "agent_loop.events.jsonl")
	}
	if v := os.Getenv("AUTOMEDAL_EVENTS_FILE"); v != "" {
		return v
	}
	return filepath.Join(l.Cwd, "agent_loop.events.jsonl")
}

// LastTrainingOutput: scratch path the runloop uses to cache the most
// recent train.py stdout for verify_iteration post-hoc inspection.
func (l *Layout) LastTrainingOutput() string {
	if l.Mode == ModeUser {
		return filepath.Join(l.HiddenRoot(), "cache", ".last_training_output")
	}
	return filepath.Join(l.Cwd, "harness", ".last_training_output")
}

// AsEnv returns the AUTOMEDAL_* env map to inject into subprocesses. The
// legacy Python consumers depend on these keys.
func (l *Layout) AsEnv() map[string]string {
	return map[string]string{
		"AUTOMEDAL_CWD":                  l.Cwd,
		"AUTOMEDAL_MODE":                 string(l.Mode),
		"AUTOMEDAL_DATA_DIR":             l.DataDir(),
		"AUTOMEDAL_SUBMISSIONS_DIR":      l.SubmissionsDir(),
		"AUTOMEDAL_JOURNAL_DIR":          l.JournalDir(),
		"AUTOMEDAL_KNOWLEDGE_MD":         l.KnowledgeMD(),
		"AUTOMEDAL_QUEUE_MD":             l.QueueMD(),
		"AUTOMEDAL_RESEARCH_MD":          l.ResearchMD(),
		"AUTOMEDAL_RESULTS_TSV":          l.ResultsTSV(),
		"AUTOMEDAL_HIDDEN_ROOT":          l.HiddenRoot(),
		"AUTOMEDAL_AGENT_DIR":            l.AgentDir(),
		"AUTOMEDAL_TRAIN_PY":             l.TrainPy(),
		"AUTOMEDAL_PREPARE_PY":           l.PreparePy(),
		"AUTOMEDAL_CONFIG_YAML":          l.ConfigYAML(),
		"AUTOMEDAL_LOG_FILE":             l.LogFile(),
		"AUTOMEDAL_EVENTS_FILE":          l.EventsFile(),
		"AUTOMEDAL_LAST_TRAINING_OUTPUT": l.LastTrainingOutput(),
		"LOG_FILE":                       l.LogFile(),
	}
}
