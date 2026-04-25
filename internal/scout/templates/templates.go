// Package templates renders the scout-side bootstrap files (AGENTS.md,
// program.md, prepare_starter.py) from embedded Go text/template files.
package templates

import (
	"bytes"
	"embed"
	"fmt"
	"text/template"

	commonfuncs "github.com/Flameingmoy/automedal/internal/templates"
)

//go:embed *.tmpl
var fs embed.FS

// Names enumerates the .tmpl basenames (without .tmpl suffix) shipped here.
var Names = []string{"AGENTS", "program", "prepare_starter.py"}

// Render returns the named template rendered with the given slot map.
// Pass `name` without the ".tmpl" suffix.
func Render(name string, slots map[string]any) (string, error) {
	body, err := fs.ReadFile(name + ".tmpl")
	if err != nil {
		return "", fmt.Errorf("read %s.tmpl: %w", name, err)
	}
	t, err := template.New(name).
		Option("missingkey=error").
		Funcs(commonfuncs.Funcs()).
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
