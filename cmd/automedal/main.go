// automedal — single CLI entry point.
//
// Phase 1 build only wires the deterministic harness commands and a
// --version stub. The TUI hand-off, agent kernel, run-loop, and
// dispatch verbs land in subsequent phases per
// /home/chinmay/.claude/plans/stateful-dancing-peacock.md.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"context"
	"time"

	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/phases"
	"github.com/Flameingmoy/automedal/internal/agent/providers"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
	"github.com/Flameingmoy/automedal/internal/auth"
	"github.com/Flameingmoy/automedal/internal/config"
	"github.com/Flameingmoy/automedal/internal/harness"
	"github.com/Flameingmoy/automedal/internal/paths"
	"github.com/Flameingmoy/automedal/internal/scout"
)

const Version = "2.0.0-go-phase1"

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		// In Phase 5 this hands off to the Go TUI. For now: print usage.
		usage()
		os.Exit(1)
	}

	switch args[0] {
	case "--version", "-v", "version":
		fmt.Printf("automedal %s\n", Version)
		return
	case "--help", "-h", "help":
		usage()
		return
	case "harness":
		runHarness(args[1:])
		return
	case "debug":
		runDebug(args[1:])
		return
	case "init":
		runInit(args[1:])
		return
	}
	fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", args[0])
	usage()
	os.Exit(2)
}

func usage() {
	fmt.Fprint(os.Stderr, `automedal — autonomous Kaggle ML research agent (Go control plane)

Usage:
  automedal <command> [args...]

Available commands (Phase 1):
  --version              print the binary version
  harness <subcommand>   deterministic helpers (no LLM):
    next-exp-id [dir]               next 4-digit experiment id (default: ./journal)
    stagnation [--k N] [--print-best|--both]   read agent/results.tsv
    rank [--m N] [--k N] [--journal-dir D]     top-K by learning value
    trace [--n N] [--journal-dir D]            chronological trace block
    init-memory [--force] [--root D]           create memory files
  debug <subcommand>     dev-facing smoke tests:
    chat [--system S] [--no-events] <prompt>   one streaming turn against the
                                               configured provider
    run-phase <name> [--max-steps N]           run one full phase end-to-end via
                                               the kernel (smoke test)
  init <slug> [--skip-download] [--abort-on-low-confidence]
                         download + sniff + render the named Kaggle competition

Coming in later phases: setup, doctor, run, dispatch, tui.
`)
}

// ── harness subcommand router ────────────────────────────────────────────

func runHarness(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: automedal harness <subcommand>")
		os.Exit(2)
	}
	switch args[0] {
	case "next-exp-id":
		harnessNextExpID(args[1:])
	case "stagnation":
		harnessStagnation(args[1:])
	case "rank":
		harnessRank(args[1:])
	case "trace":
		harnessTrace(args[1:])
	case "init-memory":
		harnessInitMemory(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "unknown harness subcommand: %s\n", args[0])
		os.Exit(2)
	}
}

func harnessNextExpID(args []string) {
	dir := defaultJournalDir()
	if len(args) > 0 {
		dir = args[0]
	}
	id, err := harness.NextExpID(dir)
	failOn(err)
	fmt.Println(id)
}

func harnessStagnation(args []string) {
	fs := flag.NewFlagSet("stagnation", flag.ExitOnError)
	k := fs.Int("k", 3, "stagnation window size")
	printBest := fs.Bool("print-best", false, "print best val_loss instead")
	both := fs.Bool("both", false, "print '<flag> <best>' on one line")
	tsvOverride := fs.String("results", "", "explicit results.tsv path")
	fs.Parse(args)

	tsv := *tsvOverride
	if tsv == "" {
		tsv = defaultResultsTSV()
	}
	losses, err := harness.ReadValLosses(tsv)
	failOn(err)

	if *both {
		flag := "0"
		if harness.IsStagnating(*k, losses) {
			flag = "1"
		}
		best := harness.BestValLoss(losses)
		bestStr := "inf"
		if !isInf(best) {
			bestStr = fmt.Sprintf("%.6f", best)
		}
		fmt.Printf("%s %s\n", flag, bestStr)
		return
	}
	if *printBest {
		best := harness.BestValLoss(losses)
		if isInf(best) {
			fmt.Println("inf")
			return
		}
		fmt.Printf("%.6f\n", best)
		return
	}
	if harness.IsStagnating(*k, losses) {
		fmt.Println("1")
	} else {
		fmt.Println("0")
	}
}

func harnessRank(args []string) {
	fs := flag.NewFlagSet("rank", flag.ExitOnError)
	m := fs.Int("m", 30, "max journals to read")
	k := fs.Int("k", 10, "top-K to output")
	dir := fs.String("journal-dir", "", "path to journal/ (default: ./journal)")
	fs.Parse(args)

	d := *dir
	if d == "" {
		d = defaultJournalDir()
	}
	out, err := harness.RankJournals(d, *m, *k)
	failOn(err)
	fmt.Println(out)
}

func harnessTrace(args []string) {
	fs := flag.NewFlagSet("trace", flag.ExitOnError)
	n := fs.Int("n", 3, "number of recent journal entries")
	dir := fs.String("journal-dir", "", "path to journal/ (default: ./journal)")
	fs.Parse(args)

	d := *dir
	if d == "" {
		d = defaultJournalDir()
	}
	out, err := harness.BuildTrace(d, *n)
	failOn(err)
	fmt.Println(out)
}

func harnessInitMemory(args []string) {
	fs := flag.NewFlagSet("init-memory", flag.ExitOnError)
	force := fs.Bool("force", false, "overwrite existing memory files")
	root := fs.String("root", "", "project root (default: layout-resolved cwd)")
	fs.Parse(args)

	r := *root
	if r == "" {
		l, err := paths.New()
		failOn(err)
		r = l.Cwd
	}
	res, err := harness.InitMemory(r, *force)
	failOn(err)
	for name, state := range res {
		fmt.Printf("  %7s  %s\n", state, name)
	}
}

// ── debug subcommand router ──────────────────────────────────────────────

func runDebug(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: automedal debug <subcommand>")
		os.Exit(2)
	}
	switch args[0] {
	case "chat":
		debugChat(args[1:])
	case "run-phase":
		debugRunPhase(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "unknown debug subcommand: %s\n", args[0])
		os.Exit(2)
	}
}

// ── init subcommand (scout bootstrap) ────────────────────────────────────

func runInit(args []string) {
	fs := flag.NewFlagSet("init", flag.ExitOnError)
	skipDownload := fs.Bool("skip-download", false, "skip download (data already in data/)")
	abortLow := fs.Bool("abort-on-low-confidence", false, "abort when sniff confidence < 0.7")
	fs.Parse(args)
	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: automedal init <slug>")
		os.Exit(2)
	}
	slug := fs.Arg(0)

	creds, err := scout.LoadKaggleCreds()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	client := scout.NewClient(creds)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	fmt.Printf("== AutoMedal — Bootstrap %q ==\n", slug)
	res, err := scout.Bootstrap(ctx, client, slug, scout.BootstrapOptions{
		SkipDownload:         *skipDownload,
		AbortOnLowConfidence: *abortLow,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "bootstrap:", err)
		os.Exit(1)
	}

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
}

// debugRunPhase runs one full phase end-to-end via the kernel — smoke test
// proving the kernel + provider + tools + prompts wire together.
func debugRunPhase(args []string) {
	fs := flag.NewFlagSet("run-phase", flag.ExitOnError)
	maxSteps := fs.Int("max-steps", 10, "kernel max steps")
	fs.Parse(args)
	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: automedal debug run-phase <researcher|strategist|experimenter|experimenter_eval|analyzer>")
		os.Exit(2)
	}
	name := fs.Arg(0)

	_, _ = auth.LoadEnv("")
	cfg := config.Load()
	prov, err := providers.Build(cfg.Provider, cfg.Model, providers.BuildOpts{
		Timeout: 120 * time.Second, MaxTokens: 4096,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, agent.FormatError(err))
		os.Exit(1)
	}

	l, _ := paths.New()
	sink, err := agent.New(l.EventsFile(), l.LogFile(), true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "events:", err)
		os.Exit(1)
	}
	defer sink.Close()

	chat := agent.ChatStreamFunc(func(ctx context.Context, system string,
		msgs []agent.Message, ts []tools.Tool, ev *agent.EventSink) (*agent.ChatTurn, error) {
		specs := make([]providers.ToolSpec, 0, len(ts))
		for _, t := range ts {
			specs = append(specs, providers.ToolSpec{
				Name: t.Name, Description: t.Description, Schema: t.Schema,
			})
		}
		turn, err := prov.ChatStream(ctx, providers.ChatRequest{
			System: system, Messages: msgs, Tools: specs, Events: ev,
		})
		if err != nil {
			return nil, err
		}
		out := &agent.ChatTurn{
			AssistantBlocks: turn.AssistantBlocks,
			AssistantText:   turn.AssistantText,
			Usage:           turn.Usage,
			StopReason:      turn.StopReason,
		}
		for _, c := range turn.ToolCalls {
			out.ToolCalls = append(out.ToolCalls, agent.ToolCall{
				ID: c.ID, Name: c.Name, Args: c.Args,
			})
		}
		return out, nil
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	expID, _ := harness.NextExpID(defaultJournalDir())
	bestStr := "inf"
	if losses, _ := harness.ReadValLosses(defaultResultsTSV()); len(losses) > 0 {
		bestStr = fmt.Sprintf("%.6f", harness.BestValLoss(losses))
	}

	var report agent.RunReport
	switch name {
	case "researcher":
		report, err = phases.RunResearcher(ctx, chat, sink, phases.ResearcherArgs{
			ExpID: expID, Trigger: "manual", Stagnating: false, ScheduledResearch: 1,
			BestValLoss: bestStr, MaxSteps: *maxSteps,
		})
	case "strategist":
		report, err = phases.RunStrategist(ctx, chat, sink, phases.StrategistArgs{
			ExpID: expID, Iteration: 1, MaxIters: 1, Stagnating: false, BestValLoss: bestStr,
			Pending: 0, Reflective: "", Ranked: "", MaxSteps: *maxSteps,
		})
	case "experimenter":
		report, err = phases.RunExperimenterEdit(ctx, chat, sink, phases.ExperimenterEditArgs{
			ExpID: expID, BestValLoss: bestStr, MaxSteps: *maxSteps,
		})
	case "experimenter_eval":
		report, err = phases.RunExperimenterEval(ctx, chat, sink, phases.ExperimenterEvalArgs{
			ExpID: expID, BestValLoss: bestStr, TrainRC: 0, FinalLoss: bestStr, MaxSteps: *maxSteps,
		})
	case "analyzer":
		report, err = phases.RunAnalyzer(ctx, chat, sink, phases.AnalyzerArgs{
			ExpID: expID, Slug: "manual", Status: "improved", FinalLoss: bestStr,
			BestValLoss: bestStr, ValLossDelta: 0, MaxSteps: *maxSteps,
		})
	default:
		fmt.Fprintf(os.Stderr, "unknown phase: %s\n", name)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, agent.FormatError(err))
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "\n[stop=%s steps=%d usage=%d/%d]\n",
		report.Stop, report.Steps, report.UsageTotal.InTokens, report.UsageTotal.OutTokens)
}


// debugChat runs one streaming chat turn against the configured provider,
// printing the model's response to stdout and (unless --no-events) writing
// JSONL events to AUTOMEDAL_EVENTS_FILE so the Go TUI tail picks it up.
func debugChat(args []string) {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	system := fs.String("system", "Reply concisely.", "system prompt")
	noEvents := fs.Bool("no-events", false, "don't write to the JSONL events file")
	provider := fs.String("provider", "", "override AUTOMEDAL_PROVIDER")
	model := fs.String("model", "", "override AUTOMEDAL_MODEL")
	fs.Parse(args)

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: automedal debug chat [flags] <prompt>")
		os.Exit(2)
	}
	prompt := strings.Join(fs.Args(), " ")

	// Load .env so smoke tests pick up keys without manual export.
	_, _ = auth.LoadEnv("")
	cfg := config.Load()
	if *provider != "" {
		cfg.Provider = *provider
	}
	if *model != "" {
		cfg.Model = *model
	}

	prov, err := providers.Build(cfg.Provider, cfg.Model, providers.BuildOpts{
		Timeout:   30 * time.Second,
		MaxTokens: 1024,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, agent.FormatError(err))
		os.Exit(1)
	}

	var sink *agent.EventSink
	jsonlPath, humanPath := "", ""
	if !*noEvents {
		l, _ := paths.New()
		jsonlPath = l.EventsFile()
		humanPath = l.LogFile()
	}
	// Echo=true streams deltas to stdout; sink without paths is in-memory.
	sink, err = agent.New(jsonlPath, humanPath, true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "cannot open events file:", err)
		os.Exit(1)
	}
	defer sink.Close()
	sink = sink.WithPhase("debug-chat")
	sink.PhaseStart(map[string]any{"provider": cfg.Provider, "model": cfg.Model})
	sink.StepAdvance()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	turn, err := prov.ChatStream(ctx, providers.ChatRequest{
		System: *system,
		Messages: []agent.Message{
			{Role: "user", Content: prompt},
		},
		Events: sink,
	})
	if err != nil {
		if sink != nil {
			sink.Error("debug.ChatStream", err)
			sink.PhaseEnd("provider_error", nil, nil)
		}
		fmt.Fprintln(os.Stderr, agent.FormatError(err))
		os.Exit(1)
	}

	if sink != nil {
		sink.Usage(turn.Usage.InTokens, turn.Usage.OutTokens)
		sink.PhaseEnd(turn.StopReason, &turn.Usage, nil)
	}

	// Newline after streamed deltas, then a usage tag.
	fmt.Println()
	fmt.Fprintf(os.Stderr, "\n[%s/%s usage=%d/%d stop=%s]\n",
		cfg.Provider, cfg.Model, turn.Usage.InTokens, turn.Usage.OutTokens, turn.StopReason)
}

// ── helpers ──────────────────────────────────────────────────────────────

func defaultJournalDir() string {
	l, err := paths.New()
	if err != nil {
		return filepath.Join(".", "journal")
	}
	return l.JournalDir()
}

func defaultResultsTSV() string {
	l, err := paths.New()
	if err != nil {
		return filepath.Join("agent", "results.tsv")
	}
	return l.ResultsTSV()
}

func failOn(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func isInf(f float64) bool {
	return f > 1e308 || strings.HasPrefix(fmt.Sprintf("%g", f), "+Inf")
}
