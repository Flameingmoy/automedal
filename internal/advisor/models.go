// Live model-list fetch for advisor autocompletion.
//
// Pulls GET <base_url>/models from the configured advisor endpoint
// (default opencode-go) and caches the result on disk so subsequent
// calls are free. Returns the cached list on any network failure so
// the TUI never blocks. Mirrors automedal/advisor/models.py.
package advisor

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const cacheTTL = time.Hour

// fallbackModels is used only when both (a) disk cache is empty and
// (b) the live fetch failed. Keep it small + advisor-grade.
var fallbackModels = []string{
	"kimi-k2.6",
	"minimax-m2.7",
	"glm-4.6",
	"glm-4.5-air",
	"mimo-7b",
	"claude-sonnet-4-5",
	"claude-opus-4-7",
	"gpt-5",
	"gpt-5-mini",
}

func cachePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".automedal", "models_cache.json")
}

// CachePath exposes the on-disk cache location for the TUI.
func CachePath() string { return cachePath() }

type modelsCache struct {
	BaseURL   string   `json:"base_url"`
	FetchedAt float64  `json:"fetched_at"`
	Models    []string `json:"models"`
}

func readCache() modelsCache {
	var c modelsCache
	b, err := os.ReadFile(cachePath())
	if err != nil {
		return c
	}
	_ = json.Unmarshal(b, &c)
	return c
}

func writeCache(c modelsCache) error {
	if err := os.MkdirAll(filepath.Dir(cachePath()), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(cachePath(), b, 0o644)
}

func fetchRemote(baseURL, apiKey string, timeout time.Duration) ([]string, error) {
	url := baseURL + "/models"
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	c := &http.Client{Timeout: timeout}
	r, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()
	if r.StatusCode/100 != 2 {
		return nil, fmt.Errorf("models: HTTP %d", r.StatusCode)
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	// Accept either {"data":[...]} or [...] shapes.
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err == nil {
		if d, ok := raw["data"]; ok {
			return extractIDs(d), nil
		}
		// Fall through to array decode.
	}
	var arr []any
	if err := json.Unmarshal(body, &arr); err != nil {
		return nil, fmt.Errorf("models: bad JSON: %w", err)
	}
	return extractIDs(arr), nil
}

func extractIDs(d any) []string {
	seen := map[string]bool{}
	out := []string{}
	if list, ok := d.([]any); ok {
		for _, e := range list {
			switch v := e.(type) {
			case string:
				if v != "" && !seen[v] {
					seen[v] = true
					out = append(out, v)
				}
			case map[string]any:
				if id, ok := v["id"].(string); ok && id != "" && !seen[id] {
					seen[id] = true
					out = append(out, id)
				}
			}
		}
	}
	sort.Strings(out)
	return out
}

// ListModels returns the cached model list, refreshing if stale or forced.
// Never errors. Falls back through three sources: fresh cache → stale cache
// → built-in.
func ListModels(forceRefresh bool) []string {
	c := readCache()
	baseURL := envStr(envBaseURL, defaultBaseURL)
	fresh := time.Since(time.Unix(int64(c.FetchedAt), 0)) < cacheTTL
	sameEndpoint := c.BaseURL == baseURL
	if len(c.Models) > 0 && fresh && sameEndpoint && !forceRefresh {
		out := append([]string(nil), c.Models...)
		return out
	}
	apiKey := os.Getenv(envAPIKey)
	if apiKey == "" {
		if len(c.Models) > 0 {
			return append([]string(nil), c.Models...)
		}
		return append([]string(nil), fallbackModels...)
	}
	ids, err := fetchRemote(baseURL, apiKey, 10*time.Second)
	if err != nil {
		if len(c.Models) > 0 {
			return append([]string(nil), c.Models...)
		}
		return append([]string(nil), fallbackModels...)
	}
	_ = writeCache(modelsCache{
		BaseURL:   baseURL,
		FetchedAt: float64(time.Now().Unix()),
		Models:    ids,
	})
	return ids
}

// RefreshModels forces a fetch. Returns (count, source-or-error-message).
func RefreshModels() (int, string) {
	baseURL := envStr(envBaseURL, defaultBaseURL)
	apiKey := os.Getenv(envAPIKey)
	if apiKey == "" {
		return 0, "OPENCODE_API_KEY not set"
	}
	ids, err := fetchRemote(baseURL, apiKey, 10*time.Second)
	if err != nil {
		return 0, err.Error()
	}
	_ = writeCache(modelsCache{
		BaseURL:   baseURL,
		FetchedAt: float64(time.Now().Unix()),
		Models:    ids,
	})
	return len(ids), baseURL
}
