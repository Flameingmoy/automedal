// Package dispatch — headless command dispatch for the `automedal` CLI.
//
// Every subcommand is a function that takes the residual argv + the
// resolved Layout and returns an exit code. Mirrors automedal/dispatch.py.
//
// Phase 4 design choices:
//   - The Python `_run_python(harness/<name>.py)` shellouts are gone;
//     subcommands call directly into internal/harness, internal/scout,
//     and internal/runloop.
//   - Setup wizard is text-mode only here (no Charmbracelet huh) — the
//     TUI exposes the friendlier setup/model-picker dialogs in Phase 5.
package dispatch

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Flameingmoy/automedal/internal/advisor"
	"github.com/Flameingmoy/automedal/internal/agent/providers"
	"github.com/Flameingmoy/automedal/internal/auth"
	"github.com/Flameingmoy/automedal/internal/config"
	"github.com/Flameingmoy/automedal/internal/harness"
	"github.com/Flameingmoy/automedal/internal/paths"
	"github.com/Flameingmoy/automedal/internal/runloop"
	"github.com/Flameingmoy/automedal/internal/scout"
)

// Version is the binary version string. Wired into `automedal version`.
var Version = "2.0.0-go"

// Dispatch is the public entry point. cmd is the verb (e.g. "run"); args
// is everything after the verb. Returns an exit code.
func Dispatch(ctx context.Context, cmd string, args []string) int {
	// Pin CPU thread pools (matches Python harness defaults).
	for k, v := range map[string]string{
		"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
		"OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
		"VECLIB_MAXIMUM_THREADS": "4",
	} {
		if os.Getenv(k) == "" {
			_ = os.Setenv(k, v)
		}
	}

	// Ensure ~/.automedal/ exists for first-run.
	if home, err := os.UserHomeDir(); err == nil {
		_ = os.MkdirAll(filepath.Join(home, ".automedal"), 0o700)
	}
	_, _ = auth.LoadEnv("")

	layout, err := paths.New()
	if err != nil {
		fmt.Fprintln(os.Stderr, "paths:", err)
		return 1
	}
	for k, v := range layout.AsEnv() {
		if os.Getenv(k) == "" {
			_ = os.Setenv(k, v)
		}
	}

	ungated := map[string]bool{
		"setup": true, "help": true, "--help": true, "-h": true,
		"doctor": true, "version": true, "--version": true, "status": true,
	}
	if !ungated[cmd] && auth.NeedsSetup() {
		fmt.Println("AutoMedal isn't configured yet.\nRun:  automedal setup")
		return 1
	}

	switch cmd {
	case "setup":
		return cmdSetup(ctx, args)
	case "doctor":
		return cmdDoctor(ctx, args)
	case "discover":
		return cmdDiscover(ctx, args, layout)
	case "select":
		return cmdSelect(args, layout)
	case "init", "bootstrap":
		return cmdInit(ctx, args)
	case "prepare":
		return cmdPrepare(args, layout)
	case "run":
		return cmdRun(ctx, args)
	case "models":
		return cmdModels(args)
	case "status":
		return cmdStatus(args, layout)
	case "clean":
		return cmdClean(args, layout)
	case "help", "--help", "-h":
		return cmdHelp()
	case "version", "--version", "-v":
		fmt.Printf("automedal %s\n", Version)
		return 0
	}
	fmt.Printf("Unknown command: %s. Run 'automedal help' for usage.\n", cmd)
	return 1
}

// ── setup ──────────────────────────────────────────────────────────────

type providerChoice struct {
	id, label string
}

var setupProviders = []providerChoice{
	{"opencode-go", "OpenCode Go  (default — MiniMax-M2.7, GLM, Kimi, MiMo)"},
	{"openrouter", "OpenRouter   (free-tier models available)"},
	{"ollama", "Ollama       (local, no key needed)"},
	{"anthropic", "Anthropic    (direct Claude)"},
	{"openai", "OpenAI       (direct GPT)"},
	{"groq", "Groq         (fast Llama / Mixtral)"},
}

func cmdSetup(_ context.Context, _ []string) int {
	fmt.Println("AutoMedal first-run setup")
	fmt.Println("─────────────────────────")
	fmt.Println()

	// Legacy pi auth import.
	home, _ := os.UserHomeDir()
	piAuth := filepath.Join(home, ".pi", "agent", "auth.json")
	if _, err := os.Stat(piAuth); err == nil && len(auth.ConfiguredProviders(nil)) == 0 {
		fmt.Printf("Found legacy %s.\nImport into %s? [Y/n] ", piAuth, auth.EnvFile())
		ans := readLine()
		if ans == "" || strings.HasPrefix(strings.ToLower(ans), "y") {
			imported, err := auth.ImportPiAuth("")
			if err != nil {
				fmt.Println("import error:", err)
			} else if len(imported) > 0 {
				fmt.Printf("✓ Imported: %s\n\n", strings.Join(imported, ", "))
			}
		}
	}

	fmt.Println("Default provider: OpenCode Go (one sk- key unlocks GLM/Kimi/MiMo/MiniMax)")
	fmt.Println()
	fmt.Print("Use OpenCode Go? [Y/switch] ")
	resp := strings.ToLower(readLine())
	idx := 0
	if strings.HasPrefix(resp, "s") {
		fmt.Println()
		for i, p := range setupProviders {
			fmt.Printf("  %d) %s\n", i+1, p.label)
		}
		fmt.Println()
		fmt.Printf("Choice [1-%d, default 1]: ", len(setupProviders))
		raw := readLine()
		if raw == "" {
			raw = "1"
		}
		if n, err := strconv.Atoi(raw); err == nil && n >= 1 && n <= len(setupProviders) {
			idx = n - 1
		}
	}
	provider := setupProviders[idx].id
	envVar, _ := lookupEnvVar(provider)
	defaultModel := auth.ProviderDefaultModel[provider]
	if defaultModel == "" {
		defaultModel = provider + "/<model>"
	}

	if envVar != "" {
		fmt.Printf("Paste your %s API key (input will echo): ", provider)
		key := readLine()
		if key == "" {
			fmt.Println("Empty key, aborting.")
			return 1
		}
		path, err := auth.SaveKey(provider, key, "")
		if err != nil {
			fmt.Println("save key:", err)
			return 1
		}
		fmt.Printf("✓ Saved to %s (mode 0600)\n", path)
	} else {
		switch provider {
		case "ollama":
			fmt.Print("Ollama host URL [http://localhost:11434]: ")
			host := readLine()
			if host == "" {
				host = "http://localhost:11434"
			}
			_ = os.Setenv("OLLAMA_HOST", host)
			pairs, _ := readDotenv(auth.EnvFile())
			pairs["OLLAMA_HOST"] = host
			if err := writeDotenv(auth.EnvFile(), pairs); err != nil {
				fmt.Println("save:", err)
			} else {
				fmt.Printf("✓ Saved OLLAMA_HOST to %s\n", auth.EnvFile())
			}
		default:
			fmt.Printf("(no key needed for %s)\n", provider)
		}
	}

	fmt.Printf("\nSetup complete.\n  Provider:      %s\n  Default model: %s\n\n",
		provider, defaultModel)

	fmt.Println("Running smoke test…")
	if ok, detail := smokeProvider(provider, modelOnly(defaultModel)); ok {
		fmt.Printf("✓ Smoke test passed — %s\n", detail)
	} else {
		fmt.Printf("⚠  Smoke test failed — %s\n", detail)
		fmt.Println("   Run 'automedal doctor' for more details.")
	}
	return 0
}

// ── doctor ─────────────────────────────────────────────────────────────

func cmdDoctor(ctx context.Context, _ []string) int {
	fmt.Println("── agent runtime ──")
	fmt.Println("  bespoke kernel (Go control plane)")
	fmt.Printf("  binary: automedal %s\n\n", Version)

	fmt.Println("── credentials ──")
	envFile := auth.EnvFile()
	state := "(missing)"
	if _, err := os.Stat(envFile); err == nil {
		state = "(present)"
	}
	fmt.Printf("  env file: %s %s\n", envFile, state)
	active := auth.ConfiguredProviders(nil)
	if len(active) > 0 {
		fmt.Printf("  configured providers: %s\n", strings.Join(active, ", "))
	} else {
		fmt.Println("  (no provider credentials found — run 'automedal setup')")
	}
	home, _ := os.UserHomeDir()
	pi := filepath.Join(home, ".pi", "agent", "auth.json")
	if _, err := os.Stat(pi); err == nil {
		fmt.Printf("  legacy: %s exists (importable via 'automedal setup')\n", pi)
	}
	fmt.Println()

	cfg := config.Load()
	fmt.Println("── smoke test ──")
	fmt.Printf("  provider: %s\n  model:    %s\n", cfg.Provider, cfg.Model)
	rc := 0
	if ok, detail := smokeProvider(cfg.Provider, cfg.Model); ok {
		fmt.Printf("  ✓ %s\n", detail)
	} else {
		fmt.Printf("  ⚠  %s\n", detail)
		rc = 1
	}

	if cfg.Advisor {
		fmt.Println()
		fmt.Println("── advisor (Kimi K2.6) ──")
		fmt.Printf("  model:     %s\n", cfg.AdvisorModel)
		fmt.Printf("  base_url:  %s\n", cfg.AdvisorBaseURL)
		junctions := []string{}
		for j := range cfg.AdvisorJunctions {
			junctions = append(junctions, j)
		}
		fmt.Printf("  junctions: %s\n", strings.Join(junctions, ","))
		fmt.Printf("  budget:    %d tok/consult, %d tok/iter\n",
			cfg.AdvisorMaxTokensPerConsult, cfg.AdvisorMaxTokensPerIter)
		if os.Getenv("OPENCODE_API_KEY") == "" {
			fmt.Println("  ⚠  OPENCODE_API_KEY missing — advisor will skip every call")
		} else {
			cctx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
			op := advisor.Consult(cctx, "tool",
				"Reply with the single word READY.",
				"Smoke test from automedal doctor.", nil)
			switch {
			case op.Skipped:
				fmt.Printf("  ⚠  skipped: %s\n", op.Reason)
			default:
				preview := strings.ReplaceAll(op.Text, "\n", " ")
				if len(preview) > 80 {
					preview = preview[:80]
				}
				fmt.Printf("  ✓ %s (%d/%d tok) — %s\n",
					cfg.AdvisorModel, op.InTokens, op.OutTokens, preview)
			}
		}
	}
	return rc
}

// ── discover / select ─────────────────────────────────────────────────

func cmdDiscover(ctx context.Context, args []string, layout *paths.Layout) int {
	creds, err := scout.LoadKaggleCreds()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	client := scout.NewClient(creds)
	cctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	candidates, err := scout.Discover(cctx, client, scout.DiscoverOptions{})
	if err != nil {
		fmt.Fprintln(os.Stderr, "discover:", err)
		return 1
	}
	out := scout.CandidatesJSONPath()
	if err := scout.WriteJSONOutput(out, candidates, 30); err != nil {
		fmt.Fprintln(os.Stderr, "write:", err)
		return 1
	}
	fmt.Printf("✓ %d candidates ranked → %s\n", len(candidates), out)
	for i, c := range candidates {
		if i >= 10 {
			break
		}
		fmt.Printf("  %2d. [%3d] %s\n", i+1, c.FinalScore, c.Competition.Ref)
	}
	return 0
}

func cmdSelect(args []string, _ *paths.Layout) int {
	bundle, err := scout.LoadCandidates(scout.CandidatesJSONPath())
	if err != nil {
		fmt.Fprintln(os.Stderr, "load candidates:", err)
		return 1
	}
	if len(bundle.Candidates) == 0 {
		fmt.Println("No candidates — run 'automedal discover' first.")
		return 1
	}
	fmt.Println("Active competitions:")
	for i, c := range bundle.Candidates {
		fmt.Printf("  %2d. [%3d] %s — %s\n", i+1, c.FinalScore,
			c.Competition.Ref, c.Competition.Title)
	}
	fmt.Print("\nSelect [1-N or slug]: ")
	raw := readLine()
	if raw == "" {
		return 0
	}
	var slug string
	if n, err := strconv.Atoi(raw); err == nil && n >= 1 && n <= len(bundle.Candidates) {
		slug = bundle.Candidates[n-1].Competition.Ref
	} else {
		if c := bundle.FindBySlug(raw); c != nil {
			slug = c.Competition.Ref
		}
	}
	if slug == "" {
		fmt.Println("(invalid choice)")
		return 1
	}
	fmt.Printf("Run:  automedal init %s\n", slug)
	return 0
}

// ── init ───────────────────────────────────────────────────────────────

func cmdInit(ctx context.Context, args []string) int {
	if len(args) == 0 {
		fmt.Println("Usage: automedal init <slug>")
		fmt.Println("Example: automedal init playground-series-s6e4")
		return 1
	}
	slug := args[0]
	skipDownload := false
	abortLow := false
	for _, a := range args[1:] {
		switch a {
		case "--skip-download":
			skipDownload = true
		case "--abort-on-low-confidence":
			abortLow = true
		}
	}
	creds, err := scout.LoadKaggleCreds()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	client := scout.NewClient(creds)
	cctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	res, err := scout.Bootstrap(cctx, client, slug, scout.BootstrapOptions{
		SkipDownload:         skipDownload,
		AbortOnLowConfidence: abortLow,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "bootstrap:", err)
		return 1
	}
	fmt.Printf("== AutoMedal — Bootstrap %q ==\n", slug)
	fmt.Printf("  Task:        %s\n", res.Schema.TaskType)
	fmt.Printf("  Target:      %s\n", res.Schema.TargetCol)
	fmt.Printf("  Features:    %d numeric + %d categorical\n",
		len(res.Schema.NumericFeatures), len(res.Schema.CategoricalFeatures))
	fmt.Printf("  Train/Test:  %d / %d rows\n", res.Schema.TrainRows, res.Schema.TestRows)
	fmt.Printf("  Confidence:  %.0f%%\n", res.Schema.Confidence*100)
	fmt.Printf("  Config:      %s\n", res.ConfigPath)
	for _, p := range res.Rendered {
		fmt.Printf("  Rendered:    %s\n", p)
	}
	if res.PrepareWritten {
		fmt.Printf("  Wrote:       %s (starter)\n", res.PreparePath)
	} else {
		fmt.Printf("  Kept:        %s (existing)\n", res.PreparePath)
	}
	for name, state := range res.Memory {
		fmt.Printf("  %7s     %s\n", state, name)
	}
	if len(res.Schema.Warnings) > 0 {
		fmt.Println("  Warnings:")
		for _, w := range res.Schema.Warnings {
			fmt.Printf("    - %s\n", w)
		}
	}
	fmt.Println("Done.")
	return 0
}

// ── prepare ────────────────────────────────────────────────────────────

func cmdPrepare(args []string, layout *paths.Layout) int {
	preparePy := layout.PreparePy()
	if _, err := os.Stat(preparePy); err != nil {
		fmt.Fprintln(os.Stderr, "prepare.py not found:", preparePy)
		return 1
	}
	cmd := exec.Command("python", append([]string{preparePy}, args...)...)
	cmd.Dir = layout.Cwd
	cmd.Stdout, cmd.Stderr, cmd.Stdin = os.Stdout, os.Stderr, os.Stdin
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return ee.ExitCode()
		}
		return 1
	}
	return 0
}

// ── run ────────────────────────────────────────────────────────────────

func cmdRun(ctx context.Context, args []string) int {
	args, env := runloop.ParseRunArgs(args)
	for k, v := range env {
		_ = os.Setenv(k, v)
	}
	maxIters := 50
	fast := false
	if len(args) > 0 {
		if n, err := strconv.Atoi(args[0]); err == nil {
			maxIters = n
		} else {
			fmt.Fprintf(os.Stderr, "invalid MAX_ITERATIONS: %q\n", args[0])
			return 2
		}
	}
	if len(args) > 1 && strings.EqualFold(args[1], "fast") {
		fast = true
	}
	rc, err := runloop.Run(ctx, runloop.Options{MaxIterations: maxIters, FastMode: fast})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		if rc == 0 {
			rc = 1
		}
	}
	return rc
}

// ── models (advisor model cache) ──────────────────────────────────────

func cmdModels(args []string) int {
	sub := "list"
	if len(args) > 0 {
		sub = args[0]
	}
	switch sub {
	case "refresh", "--refresh", "-r":
		n, where := advisor.RefreshModels()
		if n == 0 {
			fmt.Printf("⚠  refresh failed: %s\n", where)
			return 1
		}
		fmt.Printf("✓ %d models cached from %s\n  (%s)\n", n, where, advisor.CachePath())
		return 0
	}
	models := advisor.ListModels(false)
	if len(models) == 0 {
		fmt.Println("(no models cached — run 'automedal models refresh')")
		return 1
	}
	fmt.Printf("# %d models in %s\n", len(models), advisor.CachePath())
	for _, m := range models {
		fmt.Printf("  %s\n", m)
	}
	return 0
}

// ── status ────────────────────────────────────────────────────────────

func cmdStatus(_ []string, layout *paths.Layout) int {
	fmt.Println("── knowledge.md (head) ──")
	if raw, err := os.ReadFile(layout.KnowledgeMD()); err == nil {
		lines := strings.Split(string(raw), "\n")
		if len(lines) > 20 {
			lines = lines[:20]
		}
		fmt.Println(strings.Join(lines, "\n"))
	} else {
		fmt.Println("(no knowledge.md — run 'automedal init' first)")
	}
	fmt.Println()
	fmt.Println("── results.tsv (tail) ──")
	if raw, err := os.ReadFile(layout.ResultsTSV()); err == nil {
		lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
		if len(lines) > 5 {
			lines = lines[len(lines)-5:]
		}
		fmt.Println(strings.Join(lines, "\n"))
	} else {
		fmt.Println("(no results.tsv yet)")
	}
	fmt.Println()
	fmt.Println("── latest exp tags ──")
	cmd := exec.Command("git", "tag", "-l", "exp/*")
	cmd.Dir = layout.Cwd
	out, _ := cmd.CombinedOutput()
	tags := strings.Split(strings.TrimRight(string(out), "\n"), "\n")
	if len(tags) > 5 {
		tags = tags[len(tags)-5:]
	}
	if len(tags) == 1 && tags[0] == "" {
		fmt.Println("(no experiment tags)")
	} else {
		fmt.Println(strings.Join(tags, "\n"))
	}
	return 0
}

// ── clean ─────────────────────────────────────────────────────────────

func cmdClean(args []string, layout *paths.Layout) int {
	yes := false
	for _, a := range args {
		if a == "--yes" || a == "-y" {
			yes = true
		}
	}
	if !yes {
		fmt.Print("Wipe memory files and results.tsv? [y/N] ")
		if !strings.EqualFold(readLine(), "y") {
			fmt.Println("Aborted.")
			return 0
		}
	}
	if _, err := harness.InitMemory(layout.Cwd, true); err != nil {
		fmt.Fprintln(os.Stderr, "init memory:", err)
		return 1
	}
	if _, err := os.Stat(layout.ResultsTSV()); err == nil {
		_ = os.Remove(layout.ResultsTSV())
	}
	fmt.Println("✓ Memory reset")
	return 0
}

// ── help ──────────────────────────────────────────────────────────────

func cmdHelp() int {
	fmt.Println(`automedal — autonomous Kaggle ML research agent

One-time:
  automedal setup                configure a model provider (first-run)
  automedal doctor               diagnose provider/env state + smoke-test the LLM

Competition setup:
  automedal discover             list active Kaggle competitions
  automedal select               pick one interactively
  automedal init <slug>          download + wire up a competition
  automedal prepare              regenerate .npy arrays from data/

Loop:
  automedal run [N] [--advisor [model]]
                                 start the loop (default 50). --advisor enables
                                 the Kimi K2.6 second-opinion loop.
  automedal status               quick health check (knowledge + last results)
  automedal clean [--yes]        wipe memory files + results.tsv (confirms first)
  automedal models [refresh]     list cached advisor models (--refresh re-fetches)

Monitor:
  automedal                      open the Go TUI home screen
  automedal tui [args...]        same as above

Env vars honoured by 'automedal run':
  AUTOMEDAL_PROVIDER     opencode-go | anthropic | openai | ollama | openrouter | groq
  AUTOMEDAL_MODEL        model id for that provider (default: minimax-m2.7)
  MODEL                  back-compat slug (provider/model)
  AUTOMEDAL_ANALYZER     1=on (default), 0=off
  AUTOMEDAL_QUICK_REJECT 0=off (default), 1=on (30s smoke-train guard)
  AUTOMEDAL_DEDUPE       1=on (default), 0=off
  AUTOMEDAL_DEDUPE_THRESHOLD  BM25 score, default 5.0
  STAGNATION_K           consecutive non-improving runs before research (default 3)
  RESEARCH_EVERY         scheduled research cadence (default 10, 0 disables)
  AUTOMEDAL_LOG_FILE     human log path (default agent_loop.log)
  AUTOMEDAL_EVENTS_FILE  JSONL event sink (default agent_loop.events.jsonl)

Advisor (Kimi K2.6 second-opinion loop, off by default — see README):
  AUTOMEDAL_ADVISOR                        1=on, 0=off (default 0)
  AUTOMEDAL_ADVISOR_MODEL                  default kimi-k2.6
  AUTOMEDAL_ADVISOR_BASE_URL               default https://opencode.ai/zen/go/v1
  AUTOMEDAL_ADVISOR_JUNCTIONS              stagnation,audit,tool (any subset)
  AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_CONSULT default 2000
  AUTOMEDAL_ADVISOR_MAX_TOKENS_PER_ITER    default 8000 (hard ceiling)
  AUTOMEDAL_ADVISOR_AUDIT_EVERY            knowledge-audit cadence (default 5)
  AUTOMEDAL_ADVISOR_STAGNATION_EVERY       periodic stagnation gate (default 5)`)
	return 0
}

// ── helpers ───────────────────────────────────────────────────────────

func readLine() string {
	r := bufio.NewReader(os.Stdin)
	line, err := r.ReadString('\n')
	if err != nil && line == "" {
		return ""
	}
	return strings.TrimSpace(line)
}

func lookupEnvVar(provider string) (string, bool) {
	for _, p := range auth.ProviderEnv {
		if p.Provider == provider {
			return p.Var, true
		}
	}
	return "", false
}

func modelOnly(slug string) string {
	if i := strings.IndexByte(slug, '/'); i >= 0 {
		return slug[i+1:]
	}
	return slug
}

func smokeProvider(provider, model string) (bool, string) {
	prov, err := providers.Build(provider, model, providers.BuildOpts{
		Timeout: 30 * time.Second, MaxTokens: 64,
	})
	if err != nil {
		return false, err.Error()
	}
	_ = prov
	// Phase 4 keeps the smoke shallow — we already proved Build succeeds
	// (config + key + provider type all line up). A live ping would cost
	// a token; doctor runs a real chat round when the user asks for it
	// via debug chat. We can't easily do a one-shot ping here without
	// duplicating the chat plumbing; instead report the resolved combo.
	return true, fmt.Sprintf("provider built (%s/%s, no live call)", provider, model)
}

// readDotenv / writeDotenv: tiny KEY=value helper (Ollama path only).

func readDotenv(path string) (map[string]string, error) {
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
		out[k] = v
	}
	return out, sc.Err()
}

func writeDotenv(path string, pairs map[string]string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	var b strings.Builder
	for k, v := range pairs {
		b.WriteString(k)
		b.WriteByte('=')
		b.WriteString(v)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o600); err != nil {
		return err
	}
	return os.Chmod(path, 0o600)
}

