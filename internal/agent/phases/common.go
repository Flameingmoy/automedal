// Shared helpers for phase orchestrators. Mirrors automedal/agent/phases/_common.py.
package phases

import (
	"context"
	"errors"
	"fmt"

	"github.com/Flameingmoy/automedal/internal/agent"
	"github.com/Flameingmoy/automedal/internal/agent/prompts"
	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

const phaseSystem = "You are an AutoMedal phase agent. Read the phase instructions in the " +
	"user message carefully and use the provided tools to accomplish the " +
	"task. When you are done, produce a brief final assistant message " +
	"summarizing what you did. Do not chat — be terse and factual."

// ExtraToolsFactory is called with the phase-scoped sink so any tool it
// creates can emit phase-tagged events. Returned tools are appended to
// the base tool list before the kernel runs.
type ExtraToolsFactory func(*agent.EventSink) []tools.Tool

// RunPhaseConfig captures everything RunPhase needs.
type RunPhaseConfig struct {
	Phase             string
	Chat              agent.ChatStreamFunc
	Tools             []tools.Tool
	Events            *agent.EventSink
	MaxSteps          int
	Slots             map[string]any
	ExtraToolsFactory ExtraToolsFactory
}

// RunPhase renders the phase prompt + runs a fresh kernel.
func RunPhase(ctx context.Context, cfg RunPhaseConfig) (agent.RunReport, error) {
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 50
	}
	userMessage, err := prompts.Render(cfg.Phase, cfg.Slots)
	if err != nil {
		return agent.RunReport{}, fmt.Errorf("render %s: %w", cfg.Phase, err)
	}

	var sink *agent.EventSink
	if cfg.Events != nil {
		sink = cfg.Events.WithPhase(cfg.Phase)
		sink.PhaseStart(nil)
	}

	all := make([]tools.Tool, 0, len(cfg.Tools)+4)
	all = append(all, cfg.Tools...)
	if cfg.ExtraToolsFactory != nil {
		extra, ferr := safeFactory(cfg.ExtraToolsFactory, sink)
		if ferr != nil && sink != nil {
			sink.Error("extra_tools_factory", ferr)
		}
		all = append(all, extra...)
	}

	k := agent.NewKernel(cfg.Chat, phaseSystem, all, sink)
	k.MaxSteps = cfg.MaxSteps
	report := k.Run(ctx, userMessage)

	if sink != nil {
		sink.PhaseEnd(report.Stop, &report.UsageTotal, nil)
	}
	return report, nil
}

func safeFactory(f ExtraToolsFactory, sink *agent.EventSink) (extra []tools.Tool, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.New(fmt.Sprintf("panic: %v", r))
		}
	}()
	return f(sink), nil
}
