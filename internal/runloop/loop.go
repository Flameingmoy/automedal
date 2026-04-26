// Package runloop — AutoMedal four-phase loop orchestrator (Go port).
//
// Replaces automedal/run_loop.py. Phases per iteration:
//
//	Researcher    (only on stagnation or scheduled cadence)
//	Strategist    (only when queue empty or stagnating; followed by dedupe)
//	Quick-reject  (optional 30s smoke train guard, opt-in)
//	Experimenter  (edit)
//	[training]    (subprocess, runloop-managed)
//	Experimenter  (eval)
//	Analyzer      (per-iteration knowledge compression, default ON)
//	Verify        (regression gate + success_criteria)
//
// The run_loop.py `_verify` shell-out is replaced by direct calls into
// internal/harness — no `python harness/verify_iteration.py` subprocess.
package runloop

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/Flameingmoy/automedal/internal/advisor"
	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/phases"
	"github.com/Flameingmoy/automedal/internal/agent/providers"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
	"github.com/Flameingmoy/automedal/internal/config"
	"github.com/Flameingmoy/automedal/internal/harness"
	"github.com/Flameingmoy/automedal/internal/paths"
)

// Options override what would otherwise be derived from env. Zero values
// fall back to env / config defaults.
type Options struct {
	MaxIterations int
	FastMode      bool
}

// Run is the public entry point — the Go replacement for `python -m
// automedal.run_loop`.
func Run(ctx context.Context, opts Options) (int, error) {
	cfg := config.Load()
	if opts.MaxIterations > 0 {
		cfg.MaxIterations = opts.MaxIterations
	}
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 50
	}
	// Pin CPU thread pools; matches Python harness defaults.
	for k, v := range map[string]string{
		"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
		"OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
		"VECLIB_MAXIMUM_THREADS": "4",
	} {
		if os.Getenv(k) == "" {
			_ = os.Setenv(k, v)
		}
	}
	// Echo train budget into env so the train.py honours it.
	_ = os.Setenv("TRAIN_BUDGET_MINUTES", strconv.Itoa(cfg.TrainBudgetMinutes))

	layout, err := paths.New()
	if err != nil {
		return 1, fmt.Errorf("paths: %w", err)
	}
	for k, v := range layout.AsEnv() {
		if os.Getenv(k) == "" {
			_ = os.Setenv(k, v)
		}
	}

	logFile := cfg.LogFile
	if logFile == "" {
		logFile = layout.LogFile()
	}
	eventsFile := cfg.EventsFile
	if eventsFile == "" {
		eventsFile = layout.EventsFile()
	}

	hl, err := newHarnessLog(logFile)
	if err != nil {
		return 1, err
	}
	defer hl.close()

	prov, err := providers.Build(cfg.Provider, cfg.Model, providers.BuildOpts{
		Timeout: 120 * time.Second, MaxTokens: 8192,
	})
	if err != nil {
		return 1, err
	}
	chat := makeChatStream(prov)

	sink, err := agent.New(eventsFile, logFile, false)
	if err != nil {
		return 1, fmt.Errorf("events: %w", err)
	}
	defer sink.Close()

	hl.banner([]string{
		"==================================================",
		"  AutoMedal — Four-Phase Loop (Go control plane)",
		fmt.Sprintf("  Provider:       %s", cfg.Provider),
		fmt.Sprintf("  Model:          %s", cfg.Model),
		fmt.Sprintf("  Iterations:     %d", cfg.MaxIterations),
		fmt.Sprintf("  Stagnation K:   %d", cfg.StagnationK),
		fmt.Sprintf("  Research every: %d", cfg.ResearchEvery),
		fmt.Sprintf("  Analyzer:       %s", onOff(cfg.Analyzer)),
		fmt.Sprintf("  Quick-reject:   %s", onOff(cfg.QuickReject)),
		fmt.Sprintf("  Dedupe:         %s", onOff(cfg.Dedupe)),
		fmt.Sprintf("  Advisor:        %s", advisorBanner(cfg)),
		fmt.Sprintf("  Cooldown:       %ds", cfg.CooldownSecs),
		fmt.Sprintf("  Train budget:   %dm", cfg.TrainBudgetMinutes),
		fmt.Sprintf("  Log:            %s", logFile),
		fmt.Sprintf("  Events:         %s", eventsFile),
		fmt.Sprintf("  Started:        %s", time.Now().Format(time.RFC3339)),
		"==================================================",
	})

	rootDir := paths.RepoRoot()
	dataDir := layout.DataDir()
	preparePy := layout.PreparePy()
	trainPy := layout.TrainPy()
	queueMD := layout.QueueMD()
	journalDir := layout.JournalDir()
	knowledgeMD := layout.KnowledgeMD()
	resultsTSV := layout.ResultsTSV()
	lastTrainOut := layout.LastTrainingOutput()

	// Prepare data on first run if missing.
	if _, err := os.Stat(filepath.Join(dataDir, "X_train.npy")); err != nil {
		hl.write("Preparing data first...")
		if _, statErr := os.Stat(preparePy); statErr == nil {
			out, _ := exec.Command("python", preparePy).CombinedOutput()
			if len(out) > 0 {
				hl.write(strings.TrimRight(string(out), "\n"))
			}
		}
	}

	// Ensure memory files exist.
	if _, err := harness.InitMemory(rootDir, false); err != nil {
		hl.harness(fmt.Sprintf("init_memory failed: %v", err))
	}

	// Cancel flag — SIGTERM/SIGINT lets the current iteration finish.
	var cancelled atomic.Bool
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() {
		<-sigCh
		hl.harness("SIGTERM received — finishing this iteration then stopping")
		cancelled.Store(true)
	}()

	for i := 1; i <= cfg.MaxIterations; i++ {
		if cancelled.Load() {
			hl.harness("cancel requested — stopping cleanly")
			break
		}

		expID, _ := harness.NextExpID(journalDir)
		hl.iterationStart(i, cfg.MaxIterations, expID)
		advisor.ResetIterationBudget()

		stagnating, bestStr := stagnationSnapshot(resultsTSV, cfg.StagnationK)
		scheduledResearch := cfg.ResearchEvery > 0 && i%cfg.ResearchEvery == 0
		hl.harness(fmt.Sprintf("stagnating=%s scheduled_research=%d best=%s",
			boolDigit(stagnating), btoi(scheduledResearch), bestStr))

		// ── Researcher ──────────────────────────────────────────────────
		if stagnating || scheduledResearch {
			trigger := "stagnation"
			if !stagnating {
				trigger = "scheduled"
			}
			hl.harness(fmt.Sprintf("dispatching Researcher (%s)", trigger))
			rep, err := phases.RunResearcher(ctx, chat, sink, phases.ResearcherArgs{
				ExpID: expID, Trigger: trigger,
				Stagnating: stagnating, ScheduledResearch: scheduledResearch,
				BestValLoss: bestStr,
			})
			if err != nil {
				hl.write(fmt.Sprintf("  [WARN] Researcher error: %v", err))
			} else if rep.Stop != agent.StopAssistantDone {
				hl.write(fmt.Sprintf("  [WARN] Researcher stopped %s: %s", rep.Stop, rep.Error))
			}
			emitWarnings(hl, "researcher", harness.VerifyResearcher())
		}

		// ── Strategist ──────────────────────────────────────────────────
		pending := countPendingQueue(queueMD)
		if pending == 0 || stagnating {
			reflective, _ := harness.BuildTrace(journalDir, 3)
			ranked, _ := harness.RankJournals(journalDir, 30, 10)

			advisorNote := ""
			if advisor.IsEnabled("stagnation") &&
				(stagnating || (i > 1 && i%cfg.AdvisorStagnationEvery == 0)) {
				hl.harness(fmt.Sprintf("dispatching Advisor (stagnation, model=%s)", cfg.AdvisorModel))
				ctxStr := buildStagnationContext(journalDir, queueMD, bestStr, 3)
				op := advisor.Consult(ctx, "stagnation",
					"Val loss has not improved for several iterations. "+
						"What concrete levers should the next Strategist pull?",
					ctxStr,
					sinkAdapter(sink.WithPhase("advisor")),
				)
				if op.Skipped {
					hl.harness(fmt.Sprintf("advisor (stagnation) skipped — %s", op.Reason))
				} else {
					advisorNote = op.Text
					hl.harness(fmt.Sprintf("advisor (stagnation): %d/%d tok",
						op.InTokens, op.OutTokens))
				}
			}

			advisorTag := "no"
			if advisorNote != "" {
				advisorTag = "yes"
			}
			hl.harness(fmt.Sprintf(
				"dispatching Strategist (queue_pending=%d, stagnating=%s, advisor=%s)",
				pending, boolDigit(stagnating), advisorTag))

			rep, err := phases.RunStrategist(ctx, chat, sink, phases.StrategistArgs{
				ExpID: expID, Iteration: i, MaxIters: cfg.MaxIterations,
				Stagnating: stagnating, BestValLoss: bestStr,
				Pending:     pending,
				Reflective:  reflective,
				Ranked:      ranked,
				AdvisorNote: advisorNote,
				ConsultFunc: makeConsultFunc(ctx, sink),
			})
			if err != nil {
				hl.write(fmt.Sprintf("  [WARN] Strategist error: %v", err))
			} else if rep.Stop != agent.StopAssistantDone {
				hl.write(fmt.Sprintf("  [WARN] Strategist stopped %s: %s", rep.Stop, rep.Error))
			}
			emitWarnings(hl, "strategist", harness.VerifyStrategist())

			if cfg.Dedupe {
				summary := ApplyDedupe(queueMD, journalDir, cfg.DedupeThreshold)
				hl.harness(fmt.Sprintf(
					"dedupe: scanned=%d marked=%d threshold=%.2f journal_n=%d",
					summary.Scanned, summary.Marked, summary.Threshold, summary.JournalN))
			}
		}

		// Tag exp BEFORE Experimenter so journal has a stable ref.
		tagExp(rootDir, expID, hl)

		// ── Quick-reject (optional pre-train guard) ────────────────────
		if cfg.QuickReject {
			hl.harness("running quick-reject smoke train (budget=30s)")
			ok, reason := SmokeTrain(ctx, QuickRejectOpts{TrainPy: trainPy})
			tag := "PASS"
			if !ok {
				tag = "FAIL"
			}
			hl.harness(fmt.Sprintf("quick-reject: %s — %s", tag, reason))
			if !ok {
				hl.iterationEnd(i, expID)
				if !opts.FastMode && i < cfg.MaxIterations && cfg.CooldownSecs > 0 {
					sleepCancel(ctx, time.Duration(cfg.CooldownSecs)*time.Second)
				}
				continue
			}
		}

		// ── Experimenter (edit) ────────────────────────────────────────
		hl.harness("dispatching Experimenter (edit)")
		rep, err := phases.RunExperimenterEdit(ctx, chat, sink, phases.ExperimenterEditArgs{
			ExpID: expID, BestValLoss: bestStr,
			ConsultFunc: makeConsultFunc(ctx, sink),
		})
		if err != nil {
			hl.write(fmt.Sprintf("  [WARN] Experimenter (edit) error: %v", err))
		} else if rep.Stop != agent.StopAssistantDone {
			hl.write(fmt.Sprintf("  [WARN] Experimenter (edit) stopped %s: %s", rep.Stop, rep.Error))
		}

		// ── Training (subprocess, no agent) ────────────────────────────
		trainTimeout := cfg.TrainBudgetMinutes*60 + 30
		hl.harness(fmt.Sprintf("running training (budget=%dm, timeout=%ds)...",
			cfg.TrainBudgetMinutes, trainTimeout))

		// If prepare.py changed, re-run prepare first.
		if cmd := exec.Command("git", "diff", "--quiet", "--", preparePy); cmd != nil {
			cmd.Dir = rootDir
			if err := cmd.Run(); err != nil {
				if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() != 0 {
					hl.harness("prepare.py changed — running prepare first")
					_ = exec.Command("python", preparePy).Run()
				}
			}
		}

		trainRC, trainOut := runTrain(ctx, trainPy, lastTrainOut, trainTimeout, hl)
		finalLoss := parseFinalValLoss(trainOut)
		hl.harness(fmt.Sprintf("training done: val_loss=%s exit=%d", finalLoss, trainRC))

		// ── Experimenter (eval) ────────────────────────────────────────
		hl.harness("dispatching Experimenter (eval)")
		rep, err = phases.RunExperimenterEval(ctx, chat, sink, phases.ExperimenterEvalArgs{
			ExpID: expID, BestValLoss: bestStr,
			TrainRC: trainRC, FinalLoss: finalLoss,
		})
		if err != nil {
			hl.write(fmt.Sprintf("  [WARN] Experimenter (eval) error: %v", err))
		} else if rep.Stop != agent.StopAssistantDone {
			hl.write(fmt.Sprintf("  [WARN] Experimenter (eval) stopped %s: %s", rep.Stop, rep.Error))
		}

		_, bestAfter := stagnationSnapshot(resultsTSV, cfg.StagnationK)
		bestForVerify := bestAfter
		if bestForVerify == "" || bestForVerify == "inf" {
			bestForVerify = bestStr
		}

		// ── Verify gate ────────────────────────────────────────────────
		warnings := harness.VerifyExperimenter(expID)
		regCode := 0
		if vl, ok := parseFloat(finalLoss); ok {
			if bb, ok := parseFloat(bestStr); ok {
				warnings = append(warnings, harness.CheckRegression(vl, bb)...)
				if len(harness.CheckRegression(vl, bb)) > 0 {
					regCode = 2
				}
			}
		}
		nearMiss := false
		if vl, ok := parseFloat(finalLoss); ok {
			bsf, _ := parseFloat(bestForVerify)
			passed, near, crit := harness.CheckSuccessCriteria(expID, vl, bsf)
			if !passed && !near && crit != "" {
				warnings = append(warnings,
					fmt.Sprintf("success_criteria not met: %s (val=%.4f best=%.4f)",
						crit, vl, bsf))
			}
			if near {
				nearMiss = true
			}
		}
		emitWarnings(hl, "experimenter", warnings)

		if regCode == 2 && cfg.RegressionGate == "strict" {
			hl.harness(fmt.Sprintf("strict regression gate triggered — reverting git tag exp/%s", expID))
			cmd := exec.Command("git", "tag", "-d", "exp/"+expID)
			cmd.Dir = rootDir
			_ = cmd.Run()
		}

		if nearMiss {
			hl.harness("success_criteria near-miss — attempting one retry edit")
			rep, err := phases.RunExperimenterEdit(ctx, chat, sink, phases.ExperimenterEditArgs{
				ExpID: expID, BestValLoss: bestStr,
				Retry: true, PrevLoss: finalLoss,
				ConsultFunc: makeConsultFunc(ctx, sink),
			})
			if err != nil {
				hl.write(fmt.Sprintf("  [WARN] Experimenter (retry) error: %v", err))
			} else if rep.Stop != agent.StopAssistantDone {
				hl.write(fmt.Sprintf("  [WARN] Experimenter (retry) stopped %s: %s", rep.Stop, rep.Error))
			}
			retryRC, retryOut := runTrain(ctx, trainPy, lastTrainOut, trainTimeout, hl)
			retryLoss := parseFinalValLoss(retryOut)
			hl.harness(fmt.Sprintf("retry training done: val_loss=%s", retryLoss))
			rep, err = phases.RunExperimenterEval(ctx, chat, sink, phases.ExperimenterEvalArgs{
				ExpID: expID, BestValLoss: bestStr,
				TrainRC: retryRC, FinalLoss: retryLoss,
			})
			if err != nil {
				hl.write(fmt.Sprintf("  [WARN] Experimenter (eval-retry) error: %v", err))
			} else if rep.Stop != agent.StopAssistantDone {
				hl.write(fmt.Sprintf("  [WARN] Experimenter (eval-retry) stopped %s: %s", rep.Stop, rep.Error))
			}
			emitWarnings(hl, "experimenter", harness.VerifyExperimenter(expID))
			finalLoss = retryLoss
		}

		// ── Analyzer ───────────────────────────────────────────────────
		if cfg.Analyzer {
			fm := parseJournalAfter(journalDir, expID)
			slug := fm["slug"]
			status := defaultIfEmpty(fm["status"], "unknown")
			delta := defaultIfEmpty(fm["val_loss_delta"], "0.0")
			hl.harness(fmt.Sprintf("dispatching Analyzer (status=%s)", status))
			rep, err := phases.RunAnalyzer(ctx, chat, sink, phases.AnalyzerArgs{
				ExpID: expID, Slug: slug, Status: status,
				FinalLoss: finalLoss, BestValLoss: bestForVerify,
				ValLossDelta: delta,
			})
			if err != nil {
				hl.write(fmt.Sprintf("  [WARN] Analyzer error: %v", err))
			} else if rep.Stop != agent.StopAssistantDone {
				hl.write(fmt.Sprintf("  [WARN] Analyzer stopped %s: %s", rep.Stop, rep.Error))
			}
		}

		// ── Advisor (knowledge audit, periodic) ─────────────────────────
		if advisor.IsEnabled("audit") && i > 0 && i%cfg.AdvisorAuditEvery == 0 {
			hl.harness(fmt.Sprintf("dispatching Advisor (audit, model=%s)", cfg.AdvisorModel))
			ctxStr := buildAuditContext(knowledgeMD, journalDir, 5)
			op := advisor.Consult(ctx, "audit",
				"Review knowledge.md for contradictions, stale claims, or missing "+
					"signals from the last few experiments.",
				ctxStr,
				sinkAdapter(sink.WithPhase("advisor")),
			)
			if op.Skipped {
				hl.harness(fmt.Sprintf("advisor (audit) skipped — %s", op.Reason))
			} else {
				n := appendAdvisorAudit(knowledgeMD, op.Text)
				hl.harness(fmt.Sprintf("advisor (audit): %d/%d tok, %d comment(s) appended",
					op.InTokens, op.OutTokens, n))
			}
		}

		hl.iterationEnd(i, expID)

		if !opts.FastMode && i < cfg.MaxIterations && cfg.CooldownSecs > 0 {
			sleepCancel(ctx, time.Duration(cfg.CooldownSecs)*time.Second)
		}
	}

	hl.banner([]string{
		"",
		"==================================================",
		fmt.Sprintf("  AutoMedal complete — %d iterations", cfg.MaxIterations),
		fmt.Sprintf("  Finished: %s", time.Now().Format(time.RFC3339)),
		fmt.Sprintf("  Results:  cat %s", resultsTSV),
		"  Memory:   knowledge.md / experiment_queue.md / journal/",
		"==================================================",
	})
	return 0, nil
}

// ── chat-stream adapter ─────────────────────────────────────────────────

func makeChatStream(prov providers.ChatProvider) agent.ChatStreamFunc {
	return func(ctx context.Context, system string, msgs []agent.Message,
		ts []tools.Tool, ev *agent.EventSink) (*agent.ChatTurn, error) {
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
	}
}

func makeConsultFunc(_ context.Context, sink *agent.EventSink) tools.ConsultFunc {
	return func(ctx context.Context, purpose, question, contextHint string) tools.AdvisorOpinion {
		op := advisor.Consult(ctx, purpose, question, contextHint,
			sinkAdapter(sink.WithPhase("advisor")))
		return tools.AdvisorOpinion{
			Text: op.Text, Skipped: op.Skipped, Reason: op.Reason,
			InTokens: op.InTokens, OutTokens: op.OutTokens,
		}
	}
}

func sinkAdapter(s *agent.EventSink) advisor.AdvisorEvents {
	return advisor.SinkFunc(func(purpose, model, reason, preview string,
		inTokens, outTokens int, skipped bool) {
		if s == nil {
			return
		}
		s.AdvisorConsult(agent.AdvisorConsultArgs{
			Purpose: purpose, Model: model, Reason: reason, Preview: preview,
			InTokens: inTokens, OutTokens: outTokens, Skipped: skipped,
		})
	})
}

// ── harness state helpers ───────────────────────────────────────────────

// stagnationSnapshot returns (stagnating, bestStr).
func stagnationSnapshot(resultsTSV string, k int) (bool, string) {
	losses, _ := harness.ReadValLosses(resultsTSV)
	stag := harness.IsStagnating(k, losses)
	best := harness.BestValLoss(losses)
	if math.IsInf(best, 1) || math.IsNaN(best) || best == 0 || best > 1e308 {
		return stag, "nan"
	}
	return stag, fmt.Sprintf("%.6f", best)
}

func countPendingQueue(queueMD string) int {
	raw, err := os.ReadFile(queueMD)
	if err != nil {
		return 0
	}
	return strings.Count(string(raw), "[STATUS: pending]")
}

func tagExp(rootDir, expID string, hl *harnessLog) {
	check := exec.Command("git", "rev-parse", "exp/"+expID)
	check.Dir = rootDir
	if err := check.Run(); err == nil {
		return
	}
	cmd := exec.Command("git", "tag", "exp/"+expID, "HEAD")
	cmd.Dir = rootDir
	out, _ := cmd.CombinedOutput()
	if len(out) > 0 {
		hl.write(strings.TrimRight(string(out), "\n"))
	}
}

func runTrain(ctx context.Context, trainPy, outPath string, timeoutSecs int, hl *harnessLog) (int, string) {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		hl.harness(fmt.Sprintf("could not create %s: %v", filepath.Dir(outPath), err))
	}

	out, err := os.Create(outPath)
	if err != nil {
		hl.harness(fmt.Sprintf("could not create %s: %v", outPath, err))
		return 1, ""
	}
	defer out.Close()

	cctx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSecs)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(cctx, "python", trainPy)
	cmd.Dir = paths.RepoRoot()
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		hl.harness(fmt.Sprintf("stdout pipe: %v", err))
		return 1, ""
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		hl.write(fmt.Sprintf("  [harness] train.py launch failed: %v", err))
		return 127, ""
	}

	var buf strings.Builder
	mw := io.MultiWriter(out, &buf, os.Stdout)
	if hl.fh != nil {
		mw = io.MultiWriter(out, &buf, os.Stdout, hl.fh)
	}
	go func() { _, _ = io.Copy(mw, pipe) }()

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		rc := 0
		if err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				rc = ee.ExitCode()
			} else {
				rc = 1
			}
		}
		return rc, buf.String()
	case <-cctx.Done():
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			<-done
		}
		return 124, buf.String()
	}
}

var finalLossLineRE = regexp.MustCompile(`final_val_loss=([0-9.]+)`)

func parseFinalValLoss(text string) string {
	m := finalLossLineRE.FindStringSubmatch(text)
	if m == nil {
		return "nan"
	}
	return m[1]
}

// buildStagnationContext snapshots recent journals + queue + best for the
// stagnation advisor consult.
func buildStagnationContext(journalDir, queueMD, best string, lastN int) string {
	parts := []string{"current best val_loss: " + best}
	if entries, err := filepath.Glob(filepath.Join(journalDir, "*.md")); err == nil {
		sort.Strings(entries)
		if len(entries) > lastN {
			entries = entries[len(entries)-lastN:]
		}
		for _, p := range entries {
			raw, err := os.ReadFile(p)
			if err != nil {
				continue
			}
			s := string(raw)
			if len(s) > 600 {
				s = s[:600]
			}
			parts = append(parts, fmt.Sprintf("\n--- %s ---\n%s", filepath.Base(p), s))
		}
	}
	if raw, err := os.ReadFile(queueMD); err == nil {
		s := string(raw)
		if len(s) > 2000 {
			s = s[:2000]
		}
		parts = append(parts, "\n--- experiment_queue.md (current) ---\n"+s)
	} else {
		parts = append(parts, "\n--- experiment_queue.md (missing) ---")
	}
	return strings.Join(parts, "\n")
}

func buildAuditContext(knowledgeMD, journalDir string, lastN int) string {
	parts := []string{}
	if raw, err := os.ReadFile(knowledgeMD); err == nil {
		s := string(raw)
		if len(s) > 6000 {
			s = s[:6000]
		}
		parts = append(parts, "--- knowledge.md (current) ---\n"+s)
	} else {
		parts = append(parts, "--- knowledge.md (missing) ---")
	}
	if entries, err := filepath.Glob(filepath.Join(journalDir, "*.md")); err == nil {
		sort.Strings(entries)
		if len(entries) > lastN {
			entries = entries[len(entries)-lastN:]
		}
		for _, p := range entries {
			raw, err := os.ReadFile(p)
			if err != nil {
				continue
			}
			s := string(raw)
			if len(s) > 500 {
				s = s[:500]
			}
			parts = append(parts, fmt.Sprintf("\n--- %s ---\n%s", filepath.Base(p), s))
		}
	}
	return strings.Join(parts, "\n")
}

// appendAdvisorAudit appends advisor lines as HTML comments. Returns the
// number of comment lines written (0 if no-op).
func appendAdvisorAudit(knowledgeMD, text string) int {
	body := strings.TrimSpace(text)
	if body == "" || strings.EqualFold(body, "no-op") {
		return 0
	}
	lines := []string{"\n<!-- advisor audit -->\n"}
	n := 0
	for _, ln := range strings.Split(body, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		lines = append(lines, "<!-- advisor: "+ln+" -->\n")
		n++
	}
	if n == 0 {
		return 0
	}
	if err := os.MkdirAll(filepath.Dir(knowledgeMD), 0o755); err != nil {
		return 0
	}
	f, err := os.OpenFile(knowledgeMD, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return 0
	}
	defer f.Close()
	for _, ln := range lines {
		_, _ = f.WriteString(ln)
	}
	return n
}

// parseJournalAfter reads the journal entry just written for expID and
// returns the YAML frontmatter as a flat map.
func parseJournalAfter(journalDir, expID string) map[string]string {
	entries, err := filepath.Glob(filepath.Join(journalDir, expID+"-*.md"))
	if err != nil || len(entries) == 0 {
		return map[string]string{}
	}
	sort.Strings(entries)
	raw, err := os.ReadFile(entries[len(entries)-1])
	if err != nil {
		return map[string]string{}
	}
	return harness.ReadFrontmatter(string(raw))
}

// ── small utilities ─────────────────────────────────────────────────────

func emitWarnings(hl *harnessLog, phase string, warnings []string) {
	for _, w := range warnings {
		hl.write(fmt.Sprintf("  [verify:%s] %s", phase, w))
	}
}

func parseFloat(s string) (float64, bool) {
	if s == "" || s == "nan" {
		return 0, false
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil || math.IsNaN(v) {
		return 0, false
	}
	return v, true
}

func defaultIfEmpty(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

func boolDigit(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

func advisorBanner(c config.Config) string {
	if c.Advisor {
		return "on (" + c.AdvisorModel + ")"
	}
	return "off"
}

func sleepCancel(ctx context.Context, d time.Duration) {
	select {
	case <-time.After(d):
	case <-ctx.Done():
	}
}

// ── harnessLog: marker lines that mirror Python `_HarnessLog` shape ─────

type harnessLog struct {
	path string
	fh   *os.File
	bw   *bufio.Writer
}

func newHarnessLog(path string) (*harnessLog, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &harnessLog{path: path, fh: f, bw: bufio.NewWriter(f)}, nil
}

func (h *harnessLog) close() {
	if h == nil || h.fh == nil {
		return
	}
	_ = h.bw.Flush()
	_ = h.fh.Close()
	h.fh = nil
}

func (h *harnessLog) write(line string) {
	if h == nil {
		return
	}
	if h.bw != nil {
		_, _ = h.bw.WriteString(line + "\n")
		_ = h.bw.Flush()
	}
	_, _ = os.Stdout.WriteString(line + "\n")
}

func (h *harnessLog) harness(msg string)            { h.write("  [harness] " + msg) }
func (h *harnessLog) banner(lines []string)         { for _, l := range lines { h.write(l) } }
func (h *harnessLog) iterationStart(i, total int, expID string) {
	h.write("")
	h.write(fmt.Sprintf("========== Iteration %d / %d  exp=%s  [%s] ==========",
		i, total, expID, time.Now().Format("15:04:05")))
}
func (h *harnessLog) iterationEnd(i int, expID string) {
	h.write(fmt.Sprintf("--- Iteration %d complete  exp=%s  [%s] ---",
		i, expID, time.Now().Format("15:04:05")))
}
