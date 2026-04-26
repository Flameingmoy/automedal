// Per-iteration token budget + junction gating for the advisor.
//
// Process-global state: ResetIterationBudget() at the top of each loop
// iteration; ConsumeTokens(n) inside Consult after each successful call.
// RemainingTokens() short-circuits further calls once the cap is reached.
//
// Mirrors automedal/advisor/budget.py.
package advisor

import (
	"os"
	"strconv"
	"strings"
	"sync"
)

const (
	envEnabled       = "AUTOMEDAL_ADVISOR"
	envJunctions     = "AUTOMEDAL_ADVISOR_JUNCTIONS"
	envCapPerIter    = "AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER"
	envCapPerConsult = "AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT"
	envBaseURL       = "AUTOMEDAL_ADVISOR_BASE_URL"
	envModel         = "AUTOMEDAL_ADVISOR_MODEL"
	envAPIKey        = "OPENCODE_API_KEY"
)

const (
	defaultJunctions     = "stagnation,audit,tool"
	defaultCapPerIter    = 8000
	defaultCapPerConsult = 2000
	defaultBaseURL       = "https://opencode.ai/zen/go/v1"
	defaultModel         = "kimi-k2.6"
)

var (
	budgetMu      sync.Mutex
	usedThisIter  int
)

func envBool(key string, def bool) bool {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return def
	}
	switch strings.ToLower(raw) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func envInt(key string, def int) int {
	raw := os.Getenv(key)
	if raw == "" {
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return def
	}
	return n
}

func envStr(key, def string) string {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	return v
}

// ResetIterationBudget zeros the per-iteration token tally.
func ResetIterationBudget() {
	budgetMu.Lock()
	defer budgetMu.Unlock()
	usedThisIter = 0
}

// ConsumeTokens adds n to the per-iteration tally (clamped ≥0).
func ConsumeTokens(n int) {
	if n < 0 {
		n = 0
	}
	budgetMu.Lock()
	defer budgetMu.Unlock()
	usedThisIter += n
}

// RemainingTokens returns max(0, cap - used).
func RemainingTokens() int {
	cap := envInt(envCapPerIter, defaultCapPerIter)
	budgetMu.Lock()
	defer budgetMu.Unlock()
	r := cap - usedThisIter
	if r < 0 {
		return 0
	}
	return r
}

// BudgetState reports (used, cap, remaining).
func BudgetState() map[string]int {
	cap := envInt(envCapPerIter, defaultCapPerIter)
	budgetMu.Lock()
	defer budgetMu.Unlock()
	rem := cap - usedThisIter
	if rem < 0 {
		rem = 0
	}
	return map[string]int{
		"used_this_iter": usedThisIter,
		"cap_per_iter":   cap,
		"remaining":      rem,
	}
}

func junctionsAllowed() map[string]bool {
	raw := envStr(envJunctions, defaultJunctions)
	out := map[string]bool{}
	for _, s := range strings.Split(raw, ",") {
		if s = strings.TrimSpace(s); s != "" {
			out[s] = true
		}
	}
	return out
}

// IsEnabled reports whether the advisor is on at all (and, if junction
// is non-empty, whether that junction is in the allowlist).
func IsEnabled(junction string) bool {
	if !envBool(envEnabled, false) {
		return false
	}
	if junction == "" {
		return true
	}
	return junctionsAllowed()[junction]
}
