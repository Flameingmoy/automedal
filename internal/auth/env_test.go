package auth

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestSaveKeyAndLoad(t *testing.T) {
	dir := t.TempDir()
	envfile := filepath.Join(dir, ".env")

	if _, err := SaveKey("opencode-go", "sk-test-123", envfile); err != nil {
		t.Fatal(err)
	}
	// Mode 0600.
	info, _ := os.Stat(envfile)
	if info.Mode().Perm() != 0o600 {
		t.Errorf("want mode 0600, got %o", info.Mode().Perm())
	}
	// Process env set.
	if os.Getenv("OPENCODE_API_KEY") != "sk-test-123" {
		t.Errorf("env not updated")
	}
	os.Unsetenv("OPENCODE_API_KEY")

	// Roundtrip: LoadEnv should restore it.
	ok, err := LoadEnv(envfile)
	if err != nil || !ok {
		t.Fatalf("load failed: ok=%v err=%v", ok, err)
	}
	if os.Getenv("OPENCODE_API_KEY") != "sk-test-123" {
		t.Errorf("LoadEnv didn't set the key")
	}
}

func TestSaveKeyUnknownProviderRejected(t *testing.T) {
	envfile := filepath.Join(t.TempDir(), ".env")
	if _, err := SaveKey("nope", "k", envfile); err == nil {
		t.Error("expected error for unknown provider")
	}
	if _, err := SaveKey("ollama", "k", envfile); err == nil {
		t.Error("expected error — ollama has no key")
	}
}

func TestSaveKeyOverwritesExisting(t *testing.T) {
	dir := t.TempDir()
	envfile := filepath.Join(dir, ".env")
	SaveKey("opencode-go", "sk-old", envfile)
	SaveKey("opencode-go", "sk-new", envfile)

	pairs, _ := readPairs(envfile)
	if pairs["OPENCODE_API_KEY"] != "sk-new" {
		t.Errorf("want sk-new, got %q", pairs["OPENCODE_API_KEY"])
	}
	// Only one entry — no duplicate lines.
	b, _ := os.ReadFile(envfile)
	count := 0
	for _, c := range b {
		if c == '\n' {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected one line, got %d", count)
	}
}

func TestConfiguredProvidersExplicitEnv(t *testing.T) {
	env := map[string]string{
		"OPENCODE_API_KEY":  "x",
		"ANTHROPIC_API_KEY": "y",
	}
	got := ConfiguredProviders(env)
	want := []string{"opencode-go", "anthropic"}
	if !slices.Equal(got, want) {
		t.Errorf("want %v, got %v", want, got)
	}
}

func TestConfiguredProvidersOllamaNeedsHost(t *testing.T) {
	env := map[string]string{"OLLAMA_HOST": "http://localhost:11434"}
	if !slices.Contains(ConfiguredProviders(env), "ollama") {
		t.Error("ollama should be configured when OLLAMA_HOST is set")
	}
	env = map[string]string{}
	if slices.Contains(ConfiguredProviders(env), "ollama") {
		t.Error("ollama should not show without OLLAMA_HOST")
	}
}

func TestLoadEnvMissingFileIsNoError(t *testing.T) {
	ok, err := LoadEnv(filepath.Join(t.TempDir(), "does-not-exist"))
	if err != nil {
		t.Errorf("unexpected err: %v", err)
	}
	if ok {
		t.Error("want ok=false for missing file")
	}
}

func TestLoadEnvDoesNotOverrideExisting(t *testing.T) {
	dir := t.TempDir()
	envfile := filepath.Join(dir, ".env")
	os.WriteFile(envfile, []byte("OPENCODE_API_KEY=from_file\n"), 0o600)

	os.Setenv("OPENCODE_API_KEY", "from_process")
	defer os.Unsetenv("OPENCODE_API_KEY")
	LoadEnv(envfile)
	if os.Getenv("OPENCODE_API_KEY") != "from_process" {
		t.Errorf("env override wrongly applied")
	}
}

func TestLoadEnvStripsSingleAndDoubleQuotes(t *testing.T) {
	dir := t.TempDir()
	envfile := filepath.Join(dir, ".env")
	os.WriteFile(envfile, []byte(
		`OPENAI_API_KEY="sk-double"`+"\n"+
			`ANTHROPIC_API_KEY='sk-single'`+"\n",
	), 0o600)

	os.Unsetenv("OPENAI_API_KEY")
	os.Unsetenv("ANTHROPIC_API_KEY")
	LoadEnv(envfile)
	if os.Getenv("OPENAI_API_KEY") != "sk-double" {
		t.Errorf("double-quoted: %q", os.Getenv("OPENAI_API_KEY"))
	}
	if os.Getenv("ANTHROPIC_API_KEY") != "sk-single" {
		t.Errorf("single-quoted: %q", os.Getenv("ANTHROPIC_API_KEY"))
	}
}
