package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func setupRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	SetRepoRootForTest(dir)
	return dir
}

func TestSafeStaysInsideRoot(t *testing.T) {
	root := setupRoot(t)
	p, err := Safe("a/b.txt")
	if err != nil {
		t.Fatalf("safe: %v", err)
	}
	if !strings.HasPrefix(p, root) {
		t.Errorf("expected inside root, got %q", p)
	}
	if _, err := Safe("../escape"); err == nil {
		t.Error("expected escape to fail")
	}
}

func TestReadWriteEdit(t *testing.T) {
	setupRoot(t)
	ctx := context.Background()

	r := WriteFile.Invoke(ctx, map[string]any{"path": "x.txt", "content": "hello"})
	if !r.OK {
		t.Fatalf("write: %v", r.Text)
	}
	r = ReadFile.Invoke(ctx, map[string]any{"path": "x.txt"})
	if !r.OK || r.Text != "hello" {
		t.Fatalf("read: %#v", r)
	}
	r = EditFile.Invoke(ctx, map[string]any{"path": "x.txt", "old": "hello", "new": "world"})
	if !r.OK {
		t.Fatalf("edit: %v", r.Text)
	}
	r = ReadFile.Invoke(ctx, map[string]any{"path": "x.txt"})
	if r.Text != "world" {
		t.Errorf("after edit: %q", r.Text)
	}
	// edit fails when old appears !=1 times
	r = EditFile.Invoke(ctx, map[string]any{"path": "x.txt", "old": "missing", "new": "x"})
	if r.OK {
		t.Error("expected edit to fail when old missing")
	}
}

func TestListDir(t *testing.T) {
	root := setupRoot(t)
	os.MkdirAll(filepath.Join(root, "sub"), 0o755)
	os.WriteFile(filepath.Join(root, "a.txt"), []byte("x"), 0o644)
	r := ListDir.Invoke(context.Background(), map[string]any{"path": "."})
	if !r.OK {
		t.Fatal(r.Text)
	}
	if !strings.Contains(r.Text, "d  sub") || !strings.Contains(r.Text, "f  a.txt") {
		t.Errorf("unexpected listing:\n%s", r.Text)
	}
}

func TestGrep(t *testing.T) {
	root := setupRoot(t)
	os.WriteFile(filepath.Join(root, "a.txt"), []byte("foo\nbar baz\n"), 0o644)
	r := Grep.Invoke(context.Background(), map[string]any{"pattern": "bar"})
	if !r.OK {
		t.Fatal(r.Text)
	}
	if !strings.Contains(r.Text, "a.txt:2:") {
		t.Errorf("unexpected grep:\n%s", r.Text)
	}
}

func TestRunShell(t *testing.T) {
	setupRoot(t)
	r := RunShell.Invoke(context.Background(), map[string]any{"command": "echo hi"})
	if !r.OK || !strings.Contains(r.Text, "hi") {
		t.Fatalf("shell: %#v", r)
	}
	r = RunShell.Invoke(context.Background(), map[string]any{"command": "exit 3"})
	if r.OK {
		t.Errorf("expected non-zero rc to be ok=false")
	}
}
