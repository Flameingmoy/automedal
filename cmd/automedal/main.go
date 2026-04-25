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

	"github.com/Flameingmoy/automedal/internal/harness"
	"github.com/Flameingmoy/automedal/internal/paths"
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

Coming in later phases: setup, doctor, init, run, dispatch, tui.
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
