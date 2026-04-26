// Filesystem tools — read_file, write_file, edit_file, list_dir, grep.
//
// All paths are resolved relative to RepoRoot and rejected if they escape it.
// Tool surface mirrors automedal/agent/tools/fs.py.
package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// ── implementations ──────────────────────────────────────────────────

func runReadFile(ctx context.Context, args map[string]any) (ToolResult, error) {
	path := StrArg(args, "path", "")
	if path == "" {
		return MissingArg("path"), nil
	}
	p, err := Safe(path)
	if err != nil {
		return ToolResult{}, err
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return Error("error: %s", err.Error()), nil
	}
	return Result(string(b)), nil
}

func runWriteFile(ctx context.Context, args map[string]any) (ToolResult, error) {
	path := StrArg(args, "path", "")
	if path == "" {
		return MissingArg("path"), nil
	}
	content := StrArg(args, "content", "")
	p, err := Safe(path)
	if err != nil {
		return ToolResult{}, err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return Error("error: %s", err.Error()), nil
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		return Error("error: %s", err.Error()), nil
	}
	return Result(fmt.Sprintf("wrote %s (%d chars)", path, len(content))), nil
}

func runEditFile(ctx context.Context, args map[string]any) (ToolResult, error) {
	path := StrArg(args, "path", "")
	if path == "" {
		return MissingArg("path"), nil
	}
	old := StrArg(args, "old", "")
	if old == "" {
		return MissingArg("old"), nil
	}
	newStr := StrArg(args, "new", "")
	p, err := Safe(path)
	if err != nil {
		return ToolResult{}, err
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return Error("error: %s", err.Error()), nil
	}
	txt := string(b)
	n := strings.Count(txt, old)
	if n != 1 {
		return Error("error: old string appears %d times (needs exactly 1)", n), nil
	}
	if err := os.WriteFile(p, []byte(strings.Replace(txt, old, newStr, 1)), 0o644); err != nil {
		return Error("error: %s", err.Error()), nil
	}
	return Result(fmt.Sprintf("edited %s", path)), nil
}

func runListDir(ctx context.Context, args map[string]any) (ToolResult, error) {
	path := StrArg(args, "path", ".")
	p, err := Safe(path)
	if err != nil {
		return ToolResult{}, err
	}
	info, err := os.Stat(p)
	if err != nil {
		return Error("error: %s", err.Error()), nil
	}
	if !info.IsDir() {
		return Error("error: %s is not a directory", path), nil
	}
	entries, err := os.ReadDir(p)
	if err != nil {
		return Error("error: %s", err.Error()), nil
	}
	sort.Slice(entries, func(i, j int) bool {
		di, dj := entries[i].IsDir(), entries[j].IsDir()
		if di != dj {
			return di // dirs first
		}
		return strings.ToLower(entries[i].Name()) < strings.ToLower(entries[j].Name())
	})
	var lines []string
	for _, e := range entries {
		mark := "f"
		if e.IsDir() {
			mark = "d"
		}
		lines = append(lines, fmt.Sprintf("%s  %s", mark, e.Name()))
	}
	return Result(strings.Join(lines, "\n")), nil
}

func runGrep(ctx context.Context, args map[string]any) (ToolResult, error) {
	pattern := StrArg(args, "pattern", "")
	if pattern == "" {
		return MissingArg("pattern"), nil
	}
	rx, err := regexp.Compile(pattern)
	if err != nil {
		return Error("error: bad regex: %s", err.Error()), nil
	}
	path := StrArg(args, "path", ".")
	glob := StrArg(args, "glob", "*")
	root, err := Safe(path)
	if err != nil {
		return ToolResult{}, err
	}
	repoRoot := RepoRoot()

	hits := make([]string, 0, 80)
	walkOne := func(f string) (stop bool) {
		base := filepath.Base(f)
		if matched, _ := filepath.Match(glob, base); !matched {
			return false
		}
		b, err := os.ReadFile(f)
		if err != nil {
			return false
		}
		if !isText(b) {
			return false
		}
		rel, _ := filepath.Rel(repoRoot, f)
		for i, line := range strings.Split(string(b), "\n") {
			if rx.MatchString(line) {
				hits = append(hits, fmt.Sprintf("%s:%d: %s", rel, i+1, strings.TrimRight(line, "\r ")))
				if len(hits) >= 80 {
					return true
				}
			}
		}
		return false
	}

	info, err := os.Stat(root)
	if err != nil {
		return Error("error: %s", err.Error()), nil
	}
	if !info.IsDir() {
		walkOne(root)
	} else {
		filepath.Walk(root, func(p string, fi os.FileInfo, err error) error {
			if err != nil || fi.IsDir() {
				return nil
			}
			if walkOne(p) {
				return filepath.SkipAll
			}
			return nil
		})
	}
	if len(hits) == 0 {
		return Result("(no matches)"), nil
	}
	out := strings.Join(hits, "\n")
	if len(hits) >= 80 {
		out += "\n... (truncated at 80 matches)"
	}
	return Result(out), nil
}

// isText is a small heuristic: if the first 1KB has no NULL bytes and is
// valid UTF-8 enough we treat it as text.
func isText(b []byte) bool {
	n := len(b)
	if n > 1024 {
		n = 1024
	}
	for i := 0; i < n; i++ {
		if b[i] == 0 {
			return false
		}
	}
	return true
}

// ── Tool definitions ─────────────────────────────────────────────────

var ReadFile = Tool{
	Name:        "read_file",
	Description: "Read a UTF-8 file. Path is resolved relative to the repo root.",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "description": "Relative path"},
		},
		"required": []string{"path"},
	},
	Run: runReadFile,
}

var WriteFile = Tool{
	Name:        "write_file",
	Description: "Create or overwrite a UTF-8 file relative to the repo root.",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":    map[string]any{"type": "string"},
			"content": map[string]any{"type": "string"},
		},
		"required": []string{"path", "content"},
	},
	Run: runWriteFile,
}

var EditFile = Tool{
	Name: "edit_file",
	Description: "Replace `old` with `new` in `path`. `old` must appear EXACTLY once " +
		"or the call fails.",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
			"old":  map[string]any{"type": "string"},
			"new":  map[string]any{"type": "string"},
		},
		"required": []string{"path", "old", "new"},
	},
	Run: runEditFile,
}

var ListDir = Tool{
	Name:        "list_dir",
	Description: "List immediate children of a directory relative to the repo root.",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "default": "."},
		},
		"required": []string{},
	},
	Run: runListDir,
}

var Grep = Tool{
	Name: "grep",
	Description: "Search for `pattern` (regex) in files matching `glob` under `path`. " +
		"Returns up to 80 'relpath:lineno: line' matches.",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"pattern": map[string]any{"type": "string"},
			"path":    map[string]any{"type": "string", "default": "."},
			"glob":    map[string]any{"type": "string", "default": "*"},
		},
		"required": []string{"pattern"},
	},
	Run: runGrep,
}

// FSTools is the canonical filesystem tool bundle, in the order phases use.
var FSTools = []Tool{ReadFile, WriteFile, EditFile, ListDir, Grep}
