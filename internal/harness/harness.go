// Package harness ports the deterministic (LLM-free) Python helpers in
// harness/*.py to Go: stagnation detection, experiment ID allocation,
// journal ranking, reflective trace building, and memory init.
//
// These helpers read results.tsv + journal/*.md and emit plain text or
// JSON. They are what the run-loop calls between phase invocations.
package harness

import (
	"bufio"
	"os"
	"strings"
)

// ReadFrontmatter returns the YAML frontmatter of a markdown file as a
// flat map[string]string. Expects `---` fences. Ported verbatim from
// harness/_frontmatter helper in rank_journals.py + build_trace_trailer.py.
func ReadFrontmatter(body string) map[string]string {
	fm := map[string]string{}
	lines := strings.Split(body, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return fm
	}
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			break
		}
		if idx := strings.IndexByte(lines[i], ':'); idx > 0 {
			k := strings.TrimSpace(lines[i][:idx])
			v := strings.TrimSpace(lines[i][idx+1:])
			fm[k] = v
		}
	}
	return fm
}

// ExtractSection returns the body of the first H2 section whose name
// case-insensitively matches `name`. Stops at the next `## ` heading.
func ExtractSection(body, name string) string {
	want := "## " + strings.ToLower(name)
	inSection := false
	var out []string
	for _, line := range strings.Split(body, "\n") {
		trim := strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToLower(trim), want) {
			inSection = true
			continue
		}
		if inSection {
			if strings.HasPrefix(line, "## ") {
				break
			}
			out = append(out, line)
		}
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}

// readFileLines reads a file and returns its lines. Missing file → nil, nil.
func readFileLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		out = append(out, sc.Text())
	}
	return out, sc.Err()
}
