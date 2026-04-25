// Package prompts renders advisor consult prompts (Kimi K2.6 by default).
// Three templates: stagnation, audit, tool. Slots: {question, context}.
package prompts

import (
	"bytes"
	"embed"
	"fmt"
	"text/template"

	"github.com/Flameingmoy/automedal/internal/templates"
)

//go:embed *.tmpl
var fs embed.FS

// Junctions matches the advisor allowlist used by automedal/advisor/client.py.
var Junctions = []string{"stagnation", "audit", "tool"}

// Render returns the advisor prompt for `junction` with the given slots.
func Render(junction string, slots map[string]any) (string, error) {
	if !knownJunction(junction) {
		return "", fmt.Errorf("unknown junction %q; expected %v", junction, Junctions)
	}
	body, err := fs.ReadFile(junction + ".tmpl")
	if err != nil {
		return "", fmt.Errorf("read %s.tmpl: %w", junction, err)
	}
	t, err := template.New(junction).
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

func knownJunction(j string) bool {
	for _, p := range Junctions {
		if p == j {
			return true
		}
	}
	return false
}
