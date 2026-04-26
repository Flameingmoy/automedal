// Package tools — primitives shared by every tool.
//
// Each tool exposes:
//
//	Name        — string identifier the model sees
//	Description — natural-language summary
//	Schema      — JSON-Schema input shape (Anthropic/OpenAI tool spec)
//	Run         — invoked by the kernel with the model-supplied args
//
// All filesystem and shell tools resolve paths relative to RepoRoot
// (driven by the AUTOMEDAL_CWD env var, with cwd fallback) and reject
// any path that would escape it.
package tools

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// ── repo-root resolution ─────────────────────────────────────────────

var (
	repoRootOnce sync.Once
	repoRootVal  string
)

// RepoRoot returns the absolute path tools are anchored at.
// Set AUTOMEDAL_CWD to override; otherwise os.Getwd().
func RepoRoot() string {
	repoRootOnce.Do(func() {
		if v := os.Getenv("AUTOMEDAL_CWD"); v != "" {
			abs, err := filepath.Abs(v)
			if err == nil {
				repoRootVal = abs
				return
			}
		}
		cwd, err := os.Getwd()
		if err != nil {
			repoRootVal = "."
			return
		}
		repoRootVal = cwd
	})
	return repoRootVal
}

// SetRepoRootForTest overrides the resolved root. Tests only.
func SetRepoRootForTest(p string) {
	abs, _ := filepath.Abs(p)
	repoRootOnce.Do(func() {})
	repoRootVal = abs
}

// ErrPathEscape is returned by Safe when the resolved path would leave RepoRoot.
var ErrPathEscape = errors.New("path escapes repo")

// Safe resolves p relative to RepoRoot and returns it iff it stays inside.
// Mirrors automedal/agent/tools/base.py:_safe.
func Safe(p string) (string, error) {
	root := RepoRoot()
	abs := p
	if !filepath.IsAbs(p) {
		abs = filepath.Join(root, p)
	}
	resolved, err := filepath.Abs(filepath.Clean(abs))
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil {
		return "", err
	}
	if rel == ".." || len(rel) >= 3 && rel[:3] == ".."+string(filepath.Separator) {
		return "", fmt.Errorf("%w: %s", ErrPathEscape, p)
	}
	return resolved, nil
}

// ── ToolResult / Tool ────────────────────────────────────────────────

// ToolResult is what a tool returns. Text is fed back to the model verbatim.
type ToolResult struct {
	Text string
	OK   bool
}

// Result builds an ok ToolResult.
func Result(text string) ToolResult { return ToolResult{Text: text, OK: true} }

// Error builds a !ok ToolResult.
func Error(format string, a ...any) ToolResult {
	return ToolResult{Text: fmt.Sprintf(format, a...), OK: false}
}

// RunFunc is the underlying tool callable. Args are the model-supplied JSON
// inputs (already validated against Schema). ctx carries the kernel's deadline.
type RunFunc func(ctx context.Context, args map[string]any) (ToolResult, error)

// Tool is a single tool the agent kernel can dispatch.
type Tool struct {
	Name        string
	Description string
	Schema      map[string]any
	Run         RunFunc
}

// Invoke runs the tool with panic + error → ToolResult conversion.
func (t Tool) Invoke(ctx context.Context, args map[string]any) (res ToolResult) {
	defer func() {
		if r := recover(); r != nil {
			res = Error("error: panic: %v", r)
		}
	}()
	if args == nil {
		args = map[string]any{}
	}
	out, err := t.Run(ctx, args)
	if err != nil {
		if errors.Is(err, ErrPathEscape) {
			return Error("error: %s", err.Error())
		}
		return Error("error: %s", err.Error())
	}
	return out
}

// ── arg helpers (typed extraction with defaults) ─────────────────────

// StrArg returns args[k] as a string (or def if missing/non-string).
func StrArg(args map[string]any, k, def string) string {
	v, ok := args[k]
	if !ok {
		return def
	}
	if s, ok := v.(string); ok {
		return s
	}
	return def
}

// IntArg returns args[k] as an int (or def if missing/non-numeric).
func IntArg(args map[string]any, k string, def int) int {
	v, ok := args[k]
	if !ok {
		return def
	}
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	case float32:
		return int(x)
	}
	return def
}

// BoolArg returns args[k] as a bool (or def if missing/non-bool).
func BoolArg(args map[string]any, k string, def bool) bool {
	v, ok := args[k]
	if !ok {
		return def
	}
	if b, ok := v.(bool); ok {
		return b
	}
	return def
}

// StrListArg returns args[k] as []string (or def if missing/wrong type).
func StrListArg(args map[string]any, k string, def []string) []string {
	v, ok := args[k]
	if !ok {
		return def
	}
	switch x := v.(type) {
	case []string:
		return x
	case []any:
		out := make([]string, 0, len(x))
		for _, e := range x {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return def
}

// MissingArg returns an error-ToolResult for a required arg.
func MissingArg(name string) ToolResult { return Error("error: missing required arg %q", name) }
