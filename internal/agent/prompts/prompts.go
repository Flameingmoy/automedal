// Package prompts renders the per-phase agent prompts from embedded
// Go text/template files.
//
// Port of automedal/agent/prompts/__init__.py:render_prompt — same
// phase set, same slot semantics, byte-identical output.
package prompts

import (
	"bytes"
	"embed"
	"fmt"
	"strings"
	"text/template"

	"github.com/Flameingmoy/automedal/internal/templates"
)

//go:embed *.tmpl
var fs embed.FS

// Phases lists every phase template available. Matches PHASES in the
// Python module so callers can iterate identically.
var Phases = []string{
	"researcher",
	"strategist",
	"experimenter",
	"experimenter_eval",
	"analyzer",
}

// Render returns the phase prompt rendered with the given slot map.
// Missing slot keys produce "<no value>" — matches Go template's
// default "missingkey=invalid" behaviour, mirroring Jinja's
// StrictUndefined raising. To enforce strict behaviour we use Option.
func Render(phase string, slots map[string]any) (string, error) {
	if !known(phase) {
		return "", fmt.Errorf("unknown phase %q; expected one of %v", phase, Phases)
	}
	body, err := fs.ReadFile(phase + ".tmpl")
	if err != nil {
		return "", fmt.Errorf("read %s.tmpl: %w", phase, err)
	}
	t, err := template.New(phase).
		Option("missingkey=error").
		Funcs(templates.Funcs()).
		Parse(string(body))
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := t.Execute(&buf, slots); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// AvailablePhases mirrors the Python helper of the same name.
func AvailablePhases() []string {
	out := make([]string, len(Phases))
	copy(out, Phases)
	return out
}

func known(phase string) bool {
	for _, p := range Phases {
		if p == phase {
			return true
		}
	}
	return false
}

// FilesEmbedded returns the names of every .tmpl file shipped with this
// package. Useful for tests that want to enumerate.
func FilesEmbedded() []string {
	entries, _ := fs.ReadDir(".")
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".tmpl") {
			out = append(out, e.Name())
		}
	}
	return out
}
