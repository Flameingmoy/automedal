// Package auth manages provider credentials in ~/.automedal/.env (mode 0600).
//
// Port of automedal/auth.py. Minimal pure-Go dotenv implementation —
// we only need KEY=value round-trip with no quoting (quote_mode="never"
// in the Python original). For anything fancier, switch to joho/godotenv.
package auth

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// EnvFile is the canonical credential store path.
func EnvFile() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".automedal", ".env")
}

// ProviderEnv maps canonical provider name → env-var name holding the
// API key. An empty string means the provider has no key (local endpoint).
// Order matters — the wizard and doctor display providers in this order.
var ProviderEnv = [...]struct {
	Provider, Var string
}{
	{"opencode-go", "OPENCODE_API_KEY"},
	{"anthropic", "ANTHROPIC_API_KEY"},
	{"openai", "OPENAI_API_KEY"},
	{"openrouter", "OPENROUTER_API_KEY"},
	{"groq", "GROQ_API_KEY"},
	{"mistral", "MISTRAL_API_KEY"},
	{"gemini", "GEMINI_API_KEY"},
	{"xai", "XAI_API_KEY"},
	{"cerebras", "CEREBRAS_API_KEY"},
	{"zai", "ZAI_API_KEY"},
	{"ollama", ""},
}

// ProviderDefaultModel maps canonical provider → default model slug
// used by the setup wizard and doctor smoke test.
var ProviderDefaultModel = map[string]string{
	"opencode-go": "opencode-go/minimax-m2.7",
	"anthropic":   "anthropic/claude-sonnet-4-5",
	"openai":      "openai/gpt-4o",
	"openrouter":  "openrouter/openai/gpt-4o-mini",
	"groq":        "groq/llama-3.3-70b-versatile",
	"mistral":     "mistral/mistral-large-latest",
	"gemini":      "gemini/gemini-2.0-flash-exp",
	"xai":         "xai/grok-2",
	"cerebras":    "cerebras/llama-3.3-70b",
	"zai":         "zai/glm-4.6",
	"ollama":      "ollama/llama3.2",
}

func varFor(provider string) (string, bool) {
	for _, p := range ProviderEnv {
		if p.Provider == provider {
			return p.Var, true
		}
	}
	return "", false
}

// LoadEnv reads `path` (defaults to EnvFile()) and sets each KEY=value
// into os.Getenv if not already set (override=false semantics). Returns
// true if the file existed.
func LoadEnv(path string) (bool, error) {
	if path == "" {
		path = EnvFile()
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq <= 0 {
			continue
		}
		k := strings.TrimSpace(line[:eq])
		v := strings.TrimSpace(line[eq+1:])
		// Strip a single layer of matching quotes if present.
		if len(v) >= 2 && (v[0] == '"' || v[0] == '\'') && v[0] == v[len(v)-1] {
			v = v[1 : len(v)-1]
		}
		if os.Getenv(k) == "" {
			_ = os.Setenv(k, v)
		}
	}
	if err := sc.Err(); err != nil {
		return true, err
	}
	return true, nil
}

// SaveKey writes a provider's API key into `path` (defaults to EnvFile()).
// Creates the parent dir + sets mode 0600. Idempotent. Updates os.Environ
// so smoke-tests in the same process see the new key.
func SaveKey(provider, key, path string) (string, error) {
	v, ok := varFor(provider)
	if !ok {
		return "", fmt.Errorf("unknown provider: %s", provider)
	}
	if v == "" {
		return "", fmt.Errorf("provider %q does not take an API key", provider)
	}
	if path == "" {
		path = EnvFile()
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", err
	}

	existing, _ := readPairs(path)
	existing[v] = key

	if err := writePairs(path, existing); err != nil {
		return "", err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		return "", err
	}
	_ = os.Setenv(v, key)
	return path, nil
}

// ConfiguredProviders returns the providers whose credentials are
// available in the given environment. Pass nil to inspect os.Environ.
func ConfiguredProviders(env map[string]string) []string {
	getenv := func(k string) string {
		if env != nil {
			return env[k]
		}
		return os.Getenv(k)
	}
	out := []string{}
	for _, p := range ProviderEnv {
		if p.Var == "" {
			if p.Provider == "ollama" && getenv("OLLAMA_HOST") != "" {
				out = append(out, p.Provider)
			}
			continue
		}
		if getenv(p.Var) != "" {
			out = append(out, p.Provider)
		}
	}
	return out
}

// NeedsSetup returns true iff no provider has credentials available.
// Checks env, then ~/.automedal/.env, then legacy ~/.pi/agent/auth.json.
func NeedsSetup() bool {
	if len(ConfiguredProviders(nil)) > 0 {
		return false
	}
	if ok, _ := LoadEnv(""); ok {
		if len(ConfiguredProviders(nil)) > 0 {
			return false
		}
	}
	// Legacy pi auth — treat as configured if any provider has a key.
	home, _ := os.UserHomeDir()
	b, err := os.ReadFile(filepath.Join(home, ".pi", "agent", "auth.json"))
	if err != nil {
		return true
	}
	var data map[string]json.RawMessage
	if json.Unmarshal(b, &data) != nil {
		return true
	}
	for _, raw := range data {
		var entry map[string]any
		if json.Unmarshal(raw, &entry) == nil {
			if k, ok := entry["key"].(string); ok && k != "" {
				return false
			}
		}
	}
	return true
}

// ImportPiAuth imports keys from ~/.pi/agent/auth.json into path
// (defaults to EnvFile()). Returns provider names imported. Idempotent.
func ImportPiAuth(path string) ([]string, error) {
	home, _ := os.UserHomeDir()
	b, err := os.ReadFile(filepath.Join(home, ".pi", "agent", "auth.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var data map[string]json.RawMessage
	if err := json.Unmarshal(b, &data); err != nil {
		return nil, err
	}

	imported := []string{}
	for provider, raw := range data {
		var entry map[string]any
		if json.Unmarshal(raw, &entry) != nil {
			continue
		}
		key, _ := entry["key"].(string)
		if key == "" {
			continue
		}
		v, ok := varFor(provider)
		if !ok || v == "" {
			continue
		}
		if _, err := SaveKey(provider, key, path); err != nil {
			return imported, err
		}
		imported = append(imported, provider)
	}
	sort.Strings(imported)
	return imported, nil
}

// ── dotenv I/O (unquoted KEY=value lines) ────────────────────────────────

func readPairs(path string) (map[string]string, error) {
	out := map[string]string{}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq <= 0 {
			continue
		}
		k := strings.TrimSpace(line[:eq])
		v := strings.TrimSpace(line[eq+1:])
		if len(v) >= 2 && (v[0] == '"' || v[0] == '\'') && v[0] == v[len(v)-1] {
			v = v[1 : len(v)-1]
		}
		out[k] = v
	}
	return out, sc.Err()
}

func writePairs(path string, pairs map[string]string) error {
	keys := make([]string, 0, len(pairs))
	for k := range pairs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var b strings.Builder
	for _, k := range keys {
		b.WriteString(k)
		b.WriteByte('=')
		b.WriteString(pairs[k])
		b.WriteByte('\n')
	}
	return os.WriteFile(path, []byte(b.String()), 0o600)
}
