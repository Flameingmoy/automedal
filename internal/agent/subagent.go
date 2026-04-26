// spawn_subagent — runs a focused sub-task in a fresh AgentKernel.
//
// Lives in the agent package (not internal/agent/tools) to avoid the
// cycle: kernel.go → tools, so tools cannot import the kernel. Phases
// build this tool and add it to the parent kernel's toolset.
//
// Mirrors automedal/agent/tools/subagent.py.
package agent

import (
	"context"
	"fmt"
	"sort"

	"github.com/Flameingmoy/automedal/internal/agent/tools"
)

var defaultSubagentAllowlist = []string{"read_file", "grep", "list_dir", "recall", "arxiv_search"}

// SubagentConfig captures the knobs for MakeSubagentTool.
type SubagentConfig struct {
	Chat            ChatStreamFunc
	ParentTools     []tools.Tool
	Events          *EventSink
	Depth           int
	MaxDepth        int
	DefaultMaxSteps int
	Allowlist       []string // overrides the built-in default if non-nil
}

// MakeSubagentTool returns a `spawn_subagent` Tool bound to the parent's
// provider + toolset. The sub-agent gets a restricted allowlist and a
// fresh message history.
func MakeSubagentTool(cfg SubagentConfig) tools.Tool {
	if cfg.MaxDepth <= 0 {
		cfg.MaxDepth = 2
	}
	if cfg.DefaultMaxSteps <= 0 {
		cfg.DefaultMaxSteps = 20
	}
	defaultSet := cfg.Allowlist
	if defaultSet == nil {
		defaultSet = defaultSubagentAllowlist
	}

	byName := make(map[string]tools.Tool, len(cfg.ParentTools))
	for _, t := range cfg.ParentTools {
		byName[t.Name] = t
	}

	desc := "Spawn a focused sub-agent on `prompt`. The sub-agent gets a " +
		"restricted tool allowlist (intersected against the parent's). " +
		"Use for parallel fanout (e.g., 3 arxiv queries in parallel)."
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"prompt": map[string]any{"type": "string"},
			"tools": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "Tool name allowlist; subset of the parent's tools.",
			},
			"max_steps": map[string]any{"type": "integer", "default": cfg.DefaultMaxSteps, "minimum": 1, "maximum": 50},
			"label":     map[string]any{"type": "string", "default": "subagent"},
		},
		"required": []string{"prompt"},
	}

	run := func(ctx context.Context, args map[string]any) (tools.ToolResult, error) {
		prompt := tools.StrArg(args, "prompt", "")
		if prompt == "" {
			return tools.MissingArg("prompt"), nil
		}
		if cfg.Depth >= cfg.MaxDepth {
			return tools.Error("error: subagent depth cap reached (depth=%d, max=%d)", cfg.Depth, cfg.MaxDepth), nil
		}
		label := tools.StrArg(args, "label", "subagent")
		maxSteps := tools.IntArg(args, "max_steps", cfg.DefaultMaxSteps)
		if maxSteps < 1 {
			maxSteps = 1
		}
		if maxSteps > 50 {
			maxSteps = 50
		}

		wanted := tools.StrListArg(args, "tools", nil)
		if len(wanted) == 0 {
			wanted = defaultSet
		}
		allowed := make([]tools.Tool, 0, len(wanted))
		for _, n := range wanted {
			if t, ok := byName[n]; ok {
				allowed = append(allowed, t)
			}
		}
		if len(allowed) == 0 {
			parentNames := make([]string, 0, len(byName))
			for n := range byName {
				parentNames = append(parentNames, n)
			}
			sort.Strings(parentNames)
			return tools.Error("error: none of the requested tools %v are available to the parent (parent has %v)",
				wanted, parentNames), nil
		}

		var sink *EventSink
		if cfg.Events != nil {
			sink = cfg.Events.ChildSubagent(label)
			sink.SubagentStart(label, prompt)
		}

		// Allow nested subagents up to max_depth.
		subTools := make([]tools.Tool, 0, len(allowed)+1)
		subTools = append(subTools, allowed...)
		subTools = append(subTools, MakeSubagentTool(SubagentConfig{
			Chat:            cfg.Chat,
			ParentTools:     allowed,
			Events:          sink,
			Depth:           cfg.Depth + 1,
			MaxDepth:        cfg.MaxDepth,
			DefaultMaxSteps: cfg.DefaultMaxSteps,
			Allowlist:       defaultSet,
		}))

		k := NewKernel(
			cfg.Chat,
			"You are a focused sub-agent. Complete the requested sub-task "+
				"with the tools available, then summarize the result in your "+
				"final assistant message. Do not chat — be terse and factual.",
			subTools,
			sink,
		)
		k.MaxSteps = maxSteps
		report := k.Run(ctx, prompt)

		ok := report.Stop == StopAssistantDone
		if sink != nil {
			sink.SubagentEnd(label, ok, report.FinalText)
			// Do NOT close — child sinks share file handles with the parent.
		}
		if ok {
			text := report.FinalText
			if text == "" {
				text = "(empty)"
			}
			return tools.Result(text), nil
		}
		return tools.Error("subagent stopped with %s: %s", report.Stop, report.Error), nil
	}

	return tools.Tool{Name: "spawn_subagent", Description: desc, Schema: schema, Run: run}
}

// silence unused import when tests strip log
var _ = fmt.Sprintf
