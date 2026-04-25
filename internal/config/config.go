// Package config is the single source of truth for every AUTOMEDAL_* env
// var the control plane consumes. Defaults + parsing live here; callers
// receive a populated Config and never read os.Getenv directly.
//
// Port of the env-var surface from automedal/run_loop.py docstring +
// automedal/dispatch.py. README keeps a user-facing table of the same
// vars — keep the two in sync.
package config

import (
	"os"
	"strconv"
	"strings"
)

// Config is the typed runtime configuration.
type Config struct {
	// Loop pacing
	MaxIterations      int // default 50 (caller sets from positional arg)
	StagnationK        int
	ResearchEvery      int
	CooldownSecs       int
	TrainBudgetMinutes int

	// Provider selection
	Provider string
	Model    string

	// Behaviour flags
	Analyzer        bool
	QuickReject     bool
	Dedupe          bool
	DedupeThreshold float64
	RegressionGate  string // "warn" | "strict"
	DoomLoop        bool   // env kill-switch; default true

	// Advisor
	Advisor                       bool
	AdvisorModel                  string
	AdvisorBaseURL                string
	AdvisorJunctions              map[string]bool // set of {"stagnation","audit","tool"}
	AdvisorMaxTokensPerConsult    int
	AdvisorMaxTokensPerIter       int
	AdvisorAuditEvery             int
	AdvisorStagnationEvery        int

	// Logging
	LogFile    string
	EventsFile string
}

// Defaults returns a Config populated with every default. Safe to use
// without env vars set.
func Defaults() Config {
	return Config{
		MaxIterations:              50,
		StagnationK:                3,
		ResearchEvery:              10,
		CooldownSecs:               1,
		TrainBudgetMinutes:         10,
		Provider:                   "opencode-go",
		Model:                      "minimax-m2.7",
		Analyzer:                   true,
		QuickReject:                false,
		Dedupe:                     true,
		DedupeThreshold:            5.0,
		RegressionGate:             "warn",
		DoomLoop:                   true,
		Advisor:                    false,
		AdvisorModel:               "kimi-k2.6",
		AdvisorBaseURL:             "https://opencode.ai/zen/go/v1",
		AdvisorJunctions:           map[string]bool{"stagnation": true, "audit": true, "tool": true},
		AdvisorMaxTokensPerConsult: 2000,
		AdvisorMaxTokensPerIter:    8000,
		AdvisorAuditEvery:          5,
		AdvisorStagnationEvery:     5,
	}
}

// Load returns a Config populated from os.Environ, falling back to
// Defaults() for anything unset. LogFile / EventsFile are left empty
// when not set — the Layout resolver owns their paths.
func Load() Config {
	c := Defaults()

	c.StagnationK = envInt("STAGNATION_K", c.StagnationK)
	c.ResearchEvery = envInt("RESEARCH_EVERY", c.ResearchEvery)
	c.CooldownSecs = envInt("COOLDOWN_SECS", c.CooldownSecs)
	c.TrainBudgetMinutes = envInt("TRAIN_BUDGET_MINUTES", c.TrainBudgetMinutes)

	// Back-compat: MODEL="provider/model-id" splits into the two vars.
	if slug := os.Getenv("MODEL"); slug != "" {
		if p, m, ok := strings.Cut(slug, "/"); ok {
			// Only apply when the dedicated vars aren't already set,
			// so explicit overrides still win.
			if os.Getenv("AUTOMEDAL_PROVIDER") == "" {
				c.Provider = p
			}
			if os.Getenv("AUTOMEDAL_MODEL") == "" {
				c.Model = m
			}
		}
	}
	c.Provider = envStr("AUTOMEDAL_PROVIDER", c.Provider)
	c.Model = envStr("AUTOMEDAL_MODEL", c.Model)

	c.Analyzer = envBool("AUTOMEDAL_ANALYZER", c.Analyzer)
	c.QuickReject = envBool("AUTOMEDAL_QUICK_REJECT", c.QuickReject)
	c.Dedupe = envBool("AUTOMEDAL_DEDUPE", c.Dedupe)
	c.DedupeThreshold = envFloat("AUTOMEDAL_DEDUPE_THRESHOLD", c.DedupeThreshold)
	c.RegressionGate = envStr("AUTOMEDAL_REGRESSION_GATE", c.RegressionGate)
	c.DoomLoop = envBool("AUTOMEDAL_DOOM_LOOP", c.DoomLoop)

	c.Advisor = envBool("AUTOMEDAL_ADVISOR", c.Advisor)
	c.AdvisorModel = envStr("AUTOMEDAL_ADVISOR_MODEL", c.AdvisorModel)
	c.AdvisorBaseURL = envStr("AUTOMEDAL_ADVISOR_BASE_URL", c.AdvisorBaseURL)
	if raw := os.Getenv("AUTOMEDAL_ADVISOR_JUNCTIONS"); raw != "" {
		c.AdvisorJunctions = parseSet(raw)
	}
	c.AdvisorMaxTokensPerConsult = envInt("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT", c.AdvisorMaxTokensPerConsult)
	c.AdvisorMaxTokensPerIter = envInt("AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER", c.AdvisorMaxTokensPerIter)
	c.AdvisorAuditEvery = envInt("AUTOMEDAL_ADVISOR_AUDIT_EVERY", c.AdvisorAuditEvery)
	c.AdvisorStagnationEvery = envInt("AUTOMEDAL_ADVISOR_STAGNATION_EVERY", c.AdvisorStagnationEvery)

	c.LogFile = firstNonEmpty(os.Getenv("AUTOMEDAL_LOG_FILE"), os.Getenv("LOG_FILE"))
	c.EventsFile = os.Getenv("AUTOMEDAL_EVENTS_FILE")

	return c
}

// ── typed env readers ────────────────────────────────────────────────────

func envStr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		return def
	}
	return n
}

func envFloat(k string, def float64) float64 {
	v := os.Getenv(k)
	if v == "" {
		return def
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		return def
	}
	return f
}

// envBool reads a Python-flavoured bool: "1"/"true"/"yes"/"on" → true,
// "0"/"false"/"no"/"off" → false. Anything else falls back to def.
func envBool(k string, def bool) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(k)))
	if v == "" {
		return def
	}
	switch v {
	case "1", "true", "yes", "on", "y", "t":
		return true
	case "0", "false", "no", "off", "n", "f":
		return false
	}
	return def
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if x != "" {
			return x
		}
	}
	return ""
}

func parseSet(raw string) map[string]bool {
	out := map[string]bool{}
	for _, p := range strings.Split(raw, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out[p] = true
		}
	}
	return out
}
