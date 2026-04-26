// Shell tool — `run_shell` executes bash -lc bound to RepoRoot.
//
// Combined stdout+stderr capped at 8KB. Defense-in-depth, not a security
// boundary: the model already has direct file-write access via fs tools.
package tools

import (
	"context"
	"fmt"
	"os/exec"
	"time"
)

const shellOutputCap = 8000

func runShell(ctx context.Context, args map[string]any) (ToolResult, error) {
	command := StrArg(args, "command", "")
	if command == "" {
		return MissingArg("command"), nil
	}
	timeout := IntArg(args, "timeout", 120)
	if timeout < 1 {
		timeout = 1
	}
	if timeout > 600 {
		timeout = 600
	}

	cctx, cancel := context.WithTimeout(ctx, time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(cctx, "bash", "-lc", command)
	cmd.Dir = RepoRoot()
	out, err := cmd.CombinedOutput()
	if cctx.Err() == context.DeadlineExceeded {
		return Error("error: command timed out after %ds", timeout), nil
	}
	s := string(out)
	truncated := false
	if len(s) > shellOutputCap {
		s = s[:shellOutputCap]
		truncated = true
	}
	if truncated {
		s += "\n... (truncated)"
	}
	if err != nil {
		// non-zero rc surfaces as ok=false; output already captured.
		return ToolResult{Text: s, OK: false}, nil
	}
	return ToolResult{Text: s, OK: true}, nil
}

var RunShell = Tool{
	Name: "run_shell",
	Description: "Run a shell command via bash -lc, anchored at the repo root. " +
		"Returns combined stdout+stderr (capped at 8KB).",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command": map[string]any{"type": "string"},
			"timeout": map[string]any{"type": "integer", "default": 120, "minimum": 1, "maximum": 600},
		},
		"required": []string{"command"},
	},
	Run: runShell,
}

// _used keeps fmt import in tact when no Errorf calls remain after edits.
var _ = fmt.Sprintf
