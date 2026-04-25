package paths

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDevModeDetection(t *testing.T) {
	dir := t.TempDir()
	// Drop the Go dev markers into tmp.
	writeFile(t, filepath.Join(dir, "go.mod"), "module x")
	os.MkdirAll(filepath.Join(dir, "cmd", "automedal"), 0o755)
	writeFile(t, filepath.Join(dir, "cmd", "automedal", "main.go"), "package main")

	t.Setenv("AUTOMEDAL_MODE", "")
	t.Setenv("AUTOMEDAL_DEV", "")
	l, err := NewAt(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if l.Mode != ModeDev {
		t.Errorf("want dev, got %q", l.Mode)
	}
}

func TestUserModeFallback(t *testing.T) {
	dir := t.TempDir() // empty — no markers
	t.Setenv("AUTOMEDAL_MODE", "")
	t.Setenv("AUTOMEDAL_DEV", "")
	l, err := NewAt(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	if l.Mode != ModeUser {
		t.Errorf("want user, got %q", l.Mode)
	}
}

func TestExplicitModeOverride(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("AUTOMEDAL_MODE", "dev")
	l, _ := NewAt(dir, "")
	if l.Mode != ModeDev {
		t.Errorf("env override ignored, got %q", l.Mode)
	}
	t.Setenv("AUTOMEDAL_MODE", "user")
	l, _ = NewAt(dir, "")
	if l.Mode != ModeUser {
		t.Errorf("env override ignored, got %q", l.Mode)
	}
}

func TestAutomedalDevEnvVar(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("AUTOMEDAL_MODE", "")
	t.Setenv("AUTOMEDAL_DEV", "1")
	l, _ := NewAt(dir, "")
	if l.Mode != ModeDev {
		t.Errorf("AUTOMEDAL_DEV ignored, got %q", l.Mode)
	}
}

func TestPythonRepoMarkers(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "pyproject.toml"), "[project]")
	os.MkdirAll(filepath.Join(dir, "automedal"), 0o755)
	writeFile(t, filepath.Join(dir, "automedal", "run_loop.py"), "# py")
	t.Setenv("AUTOMEDAL_MODE", "")
	t.Setenv("AUTOMEDAL_DEV", "")
	l, _ := NewAt(dir, "")
	if l.Mode != ModeDev {
		t.Errorf("python markers not detected, got %q", l.Mode)
	}
}

func TestUserModeHiddenLayout(t *testing.T) {
	l := &Layout{Cwd: "/proj", Mode: ModeUser}
	if l.HiddenRoot() != "/proj/.automedal" {
		t.Errorf("hidden_root: %q", l.HiddenRoot())
	}
	if l.AgentDir() != "/proj/.automedal/agent" {
		t.Errorf("agent_dir: %q", l.AgentDir())
	}
	if l.ConfigYAML() != "/proj/.automedal/configs/competition.yaml" {
		t.Errorf("config_yaml: %q", l.ConfigYAML())
	}
	if l.ResultsTSV() != "/proj/results.tsv" {
		t.Errorf("results_tsv: %q", l.ResultsTSV())
	}
}

func TestDevModeFlatLayout(t *testing.T) {
	l := &Layout{Cwd: "/proj", Mode: ModeDev}
	if l.HiddenRoot() != "/proj" {
		t.Errorf("hidden_root: %q", l.HiddenRoot())
	}
	if l.AgentDir() != "/proj/agent" {
		t.Errorf("agent_dir: %q", l.AgentDir())
	}
	if l.ResultsTSV() != "/proj/agent/results.tsv" {
		t.Errorf("results_tsv: %q", l.ResultsTSV())
	}
}

func TestLogFileEnvOverrideDevOnly(t *testing.T) {
	l := &Layout{Cwd: "/proj", Mode: ModeDev}
	t.Setenv("AUTOMEDAL_LOG_FILE", "/custom/path.log")
	if l.LogFile() != "/custom/path.log" {
		t.Errorf("env override ignored in dev")
	}
	// User mode ignores env override — always hidden path.
	l.Mode = ModeUser
	if l.LogFile() != "/proj/.automedal/logs/agent_loop.log" {
		t.Errorf("user mode respected env override: %q", l.LogFile())
	}
}

func TestEventsFileEnvOverrideDevOnly(t *testing.T) {
	l := &Layout{Cwd: "/proj", Mode: ModeDev}
	t.Setenv("AUTOMEDAL_EVENTS_FILE", "/events/log.jsonl")
	if l.EventsFile() != "/events/log.jsonl" {
		t.Errorf("env override ignored in dev")
	}
}

func TestAsEnvContainsAllKeys(t *testing.T) {
	l := &Layout{Cwd: "/proj", Mode: ModeUser}
	env := l.AsEnv()
	required := []string{
		"AUTOMEDAL_CWD", "AUTOMEDAL_MODE", "AUTOMEDAL_DATA_DIR",
		"AUTOMEDAL_HIDDEN_ROOT", "AUTOMEDAL_KNOWLEDGE_MD",
		"AUTOMEDAL_EVENTS_FILE", "LOG_FILE",
	}
	for _, k := range required {
		if v, ok := env[k]; !ok || v == "" {
			t.Errorf("missing env key %q", k)
		}
	}
	if !strings.Contains(env["AUTOMEDAL_HIDDEN_ROOT"], ".automedal") {
		t.Errorf("user-mode hidden root should end in .automedal")
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
