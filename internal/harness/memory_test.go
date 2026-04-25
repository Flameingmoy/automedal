package harness

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInitMemoryCreatesAllArtifacts(t *testing.T) {
	dir := t.TempDir()
	res, err := InitMemory(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"knowledge.md", "experiment_queue.md", "research_notes.md", "journal/"} {
		if res[name] != "created" {
			t.Errorf("%s: want 'created', got %q", name, res[name])
		}
	}
	for _, name := range []string{"knowledge.md", "experiment_queue.md", "research_notes.md"} {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Errorf("%s missing: %v", name, err)
		}
		if !strings.Contains(string(b), "AutoMedal") &&
			!strings.Contains(string(b), "Experiment") &&
			!strings.Contains(string(b), "Research") {
			t.Errorf("%s body looks wrong: %q", name, string(b))
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "journal", ".gitkeep")); err != nil {
		t.Errorf(".gitkeep missing: %v", err)
	}
}

func TestInitMemoryKeepsExistingWithoutForce(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "knowledge.md"), []byte("# user content"), 0o644)

	res, err := InitMemory(dir, false)
	if err != nil {
		t.Fatal(err)
	}
	if res["knowledge.md"] != "kept" {
		t.Errorf("want 'kept', got %q", res["knowledge.md"])
	}
	b, _ := os.ReadFile(filepath.Join(dir, "knowledge.md"))
	if string(b) != "# user content" {
		t.Errorf("user content overwritten: %q", string(b))
	}
}

func TestInitMemoryForceResets(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "knowledge.md"), []byte("# user content"), 0o644)

	res, err := InitMemory(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if res["knowledge.md"] != "reset" {
		t.Errorf("want 'reset', got %q", res["knowledge.md"])
	}
	b, _ := os.ReadFile(filepath.Join(dir, "knowledge.md"))
	if !strings.Contains(string(b), "AutoMedal Knowledge Base") {
		t.Error("force=true should overwrite to header content")
	}
}

func TestInitMemoryGitkeepKept(t *testing.T) {
	dir := t.TempDir()
	InitMemory(dir, false)
	res, _ := InitMemory(dir, false)
	if res["journal/"] != "kept" {
		t.Errorf("second call: want 'kept', got %q", res["journal/"])
	}
}
