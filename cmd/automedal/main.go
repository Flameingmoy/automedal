// automedal — single CLI entry point.
//
// Top-level routing:
//   - "harness" / "debug" — internal dev/test helpers handled in this file.
//   - everything else — delegated to internal/dispatch.Dispatch (the
//     port of automedal/dispatch.py).
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/phases"
	"github.com/Flameingmoy/automedal/internal/agent/providers"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
	"github.com/Flameingmoy/automedal/internal/auth"
	"github.com/Flameingmoy/automedal/internal/config"
	"github.com/Flameingmoy/automedal/internal/dispatch"
	"github.com/Flameingmoy/automedal/internal/harness"
	"github.com/Flameingmoy/automedal/internal/paths"
)

const Version = "2.0.0-go"

func main() {
	dispatch.Version = Version

	args := os.Args[1:]
	if len(args) == 0 {
		// No verb → Phase 5 will hand off to the TUI. Until then, print
		// usage so users know what's available.
		fmt.Fprintln(os.Stderr, "automedal: no command — try `automedal help`")
		os.Exit(1)
	}

	// Internal dev verbs route locally; everything else goes to dispatch.
	switch args[0] {
	case "harness":
		runHarness(args[1:])
		return
	case "debug":
		runDebug(args[1:])
		return
	}

	ctx, cancel := signalContext()
	defer cancel()
	os.Exit(dispatch.Dispatch(ctx, args[0], args[1:]))
}

// signalContext returns a cancellable context that fires on SIGINT/SIGTERM.
func signalContext() (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-ch
		cancel()
	}()
	return ctx, cancel
}

// ── harness subcommand router (dev) ──────────────────────────────────────

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

// debugRunPhase runs one full phase end-to-end via the kernel — smoke
// test proving the kernel + provider + tools + prompts wire together.
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

// debugChat runs one streaming chat turn against the configured provider.
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

	jsonlPath, humanPath := "", ""
	if !*noEvents {
		l, _ := paths.New()
		jsonlPath = l.EventsFile()
		humanPath = l.LogFile()
	}
	sink, err := agent.New(jsonlPath, humanPath, true)
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
