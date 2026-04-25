// Package templates provides the shared FuncMap that mirrors the Jinja
// filters our prompt templates rely on.
//
// The four Jinja filters in use across our 11 templates:
//
//	join('sep')   → strings.Join(arr, sep)
//	capitalize    → upper-case the first rune
//	default('?')  → return def when value is nil/empty/zero
//	length        → use Go's builtin `len`
//
// Whitespace control ({%- -%}) maps 1:1 to Go's {{- -}}.
package templates

import (
	"fmt"
	"strings"
	"text/template"
	"unicode"
)

// Funcs returns the FuncMap installed on every prompt template.
func Funcs() template.FuncMap {
	return template.FuncMap{
		"join":       joinAny,
		"capitalize": capitalize,
		"default":    defaultIfEmpty,
		// pybool prints booleans the way Python's str() does — "True" /
		// "False" — so prompts rendered from Go match the Jinja output
		// byte-for-byte. Non-bool values are stringified verbatim.
		"pybool": func(v any) string {
			switch x := v.(type) {
			case bool:
				if x {
					return "True"
				}
				return "False"
			case nil:
				return "None"
			default:
				return fmt.Sprintf("%v", x)
			}
		},
		// length is unused — Go has builtin `len`. Provide it for parity
		// with Jinja in case a template uses `| length` directly.
		"length": func(v any) int {
			switch x := v.(type) {
			case nil:
				return 0
			case string:
				return len(x)
			case []any:
				return len(x)
			}
			return 0
		},
	}
}

// joinAny accepts a separator + a slice of anything and emits "%v" joined.
// Mirrors Jinja's `arr | join('sep')` behavior on lists of mixed types.
func joinAny(sep string, items any) string {
	switch xs := items.(type) {
	case nil:
		return ""
	case []string:
		return strings.Join(xs, sep)
	case []any:
		parts := make([]string, len(xs))
		for i, v := range xs {
			parts[i] = fmt.Sprintf("%v", v)
		}
		return strings.Join(parts, sep)
	}
	// Fallback: stringify whatever it is.
	return fmt.Sprintf("%v", items)
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	rs := []rune(s)
	rs[0] = unicode.ToUpper(rs[0])
	return string(rs)
}

// defaultIfEmpty returns def when v is nil, empty string, empty slice,
// or empty map. Otherwise returns v. Argument order matches Jinja's
// `value | default(fallback)` — but Go templates pipe LHS into the
// LAST arg, so we put def FIRST in the function signature and call it
// like `{{ default "?" .x }}`.
func defaultIfEmpty(def any, v any) any {
	switch x := v.(type) {
	case nil:
		return def
	case string:
		if x == "" {
			return def
		}
	case []any:
		if len(x) == 0 {
			return def
		}
	case map[string]any:
		if len(x) == 0 {
			return def
		}
	}
	return v
}
