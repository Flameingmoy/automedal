package config

import (
	"testing"
)

func TestDefaultsRoundtrip(t *testing.T) {
	clearEnv(t)
	c := Load()
	d := Defaults()
	if c.MaxIterations != d.MaxIterations ||
		c.StagnationK != d.StagnationK ||
		c.Provider != d.Provider ||
		c.Model != d.Model ||
		c.DedupeThreshold != d.DedupeThreshold ||
		c.AdvisorModel != d.AdvisorModel {
		t.Errorf("defaults drift: got %+v", c)
	}
	if c.Analyzer != true || c.Dedupe != true || c.DoomLoop != true {
		t.Error("default-true flags flipped")
	}
	if c.Advisor != false || c.QuickReject != false {
		t.Error("default-false flags flipped")
	}
	wantJ := []string{"stagnation", "audit", "tool"}
	for _, j := range wantJ {
		if !c.AdvisorJunctions[j] {
			t.Errorf("default junction %q missing", j)
		}
	}
}

func TestModelSlugSplits(t *testing.T) {
	clearEnv(t)
	t.Setenv("MODEL", "anthropic/claude-sonnet-4-5")
	c := Load()
	if c.Provider != "anthropic" || c.Model != "claude-sonnet-4-5" {
		t.Errorf("slug split: %+v", c)
	}
}

func TestExplicitVarsWinOverMODEL(t *testing.T) {
	clearEnv(t)
	t.Setenv("MODEL", "anthropic/claude-sonnet-4-5")
	t.Setenv("AUTOMEDAL_MODEL", "minimax-m2.7")
	c := Load()
	if c.Provider != "anthropic" || c.Model != "minimax-m2.7" {
		t.Errorf("explicit override not respected: %+v", c)
	}
}

func TestBooleanParsing(t *testing.T) {
	for _, v := range []string{"1", "true", "yes", "on", "TRUE", "Yes"} {
		t.Setenv("AUTOMEDAL_ADVISOR", v)
		if !Load().Advisor {
			t.Errorf("%q not parsed as true", v)
		}
	}
	for _, v := range []string{"0", "false", "no", "off"} {
		t.Setenv("AUTOMEDAL_ANALYZER", v)
		if Load().Analyzer {
			t.Errorf("%q not parsed as false", v)
		}
	}
}

func TestAdvisorJunctionsAllowlist(t *testing.T) {
	clearEnv(t)
	t.Setenv("AUTOMEDAL_ADVISOR_JUNCTIONS", "stagnation, audit")
	c := Load()
	if !c.AdvisorJunctions["stagnation"] || !c.AdvisorJunctions["audit"] {
		t.Error("missing expected junctions")
	}
	if c.AdvisorJunctions["tool"] {
		t.Error("tool junction should have been filtered out")
	}
}

func TestIntAndFloatParsing(t *testing.T) {
	clearEnv(t)
	t.Setenv("STAGNATION_K", "7")
	t.Setenv("AUTOMEDAL_DEDUPE_THRESHOLD", "8.25")
	c := Load()
	if c.StagnationK != 7 {
		t.Errorf("int: %d", c.StagnationK)
	}
	if c.DedupeThreshold != 8.25 {
		t.Errorf("float: %v", c.DedupeThreshold)
	}
}

func TestInvalidNumberFallsBackToDefault(t *testing.T) {
	clearEnv(t)
	t.Setenv("STAGNATION_K", "not-an-int")
	if Load().StagnationK != Defaults().StagnationK {
		t.Error("invalid int should fall back to default")
	}
}

func TestLogFilePrecedence(t *testing.T) {
	clearEnv(t)
	t.Setenv("LOG_FILE", "/from/log_file.log")
	t.Setenv("AUTOMEDAL_LOG_FILE", "/from/automedal.log")
	if Load().LogFile != "/from/automedal.log" {
		t.Error("AUTOMEDAL_LOG_FILE should win over LOG_FILE")
	}
	t.Setenv("AUTOMEDAL_LOG_FILE", "")
	if Load().LogFile != "/from/log_file.log" {
		t.Error("LOG_FILE should be used when AUTOMEDAL_LOG_FILE is unset")
	}
}

// clearEnv clears every variable Config.Load touches so a test starts from Defaults().
func clearEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"STAGNATION_K", "RESEARCH_EVERY", "COOLDOWN_SECS", "TRAIN_BUDGET_MINUTES",
		"AUTOMEDAL_PROVIDER", "AUTOMEDAL_MODEL", "MODEL",
		"AUTOMEDAL_ANALYZER", "AUTOMEDAL_QUICK_REJECT", "AUTOMEDAL_DEDUPE",
		"AUTOMEDAL_DEDUPE_THRESHOLD", "AUTOMEDAL_REGRESSION_GATE", "AUTOMEDAL_DOOM_LOOP",
		"AUTOMEDAL_ADVISOR", "AUTOMEDAL_ADVISOR_MODEL", "AUTOMEDAL_ADVISOR_BASE_URL",
		"AUTOMEDAL_ADVISOR_JUNCTIONS",
		"AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT",
		"AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER",
		"AUTOMEDAL_ADVISOR_AUDIT_EVERY", "AUTOMEDAL_ADVISOR_STAGNATION_EVERY",
		"LOG_FILE", "AUTOMEDAL_LOG_FILE", "AUTOMEDAL_EVENTS_FILE",
	} {
		t.Setenv(k, "")
	}
}
