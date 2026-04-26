// Package runloop is the 4-phase loop orchestrator. args.go is the
// shared argv parser used by both the CLI dispatch (`automedal run`)
// and the TUI spawn — it strips recognized flags from the positional
// "[N] [fast]" tail so the rest of the loop can stay flag-free.
//
// Mirrors automedal/run_args.py.
package runloop

import (
	"strings"
	"unicode"
)

// DefaultAdvisorModel is the canonical model used when --advisor is
// passed without an explicit model id.
const DefaultAdvisorModel = "kimi-k2.6"

// ParseRunArgs walks args and pulls out flags it recognises:
//
//	--advisor [model]   →  AUTOMEDAL_ADVISOR=1 (+ AUTOMEDAL_ADVISOR_MODEL=model
//	                      when the next token isn't another flag or a digit)
//
// Unrecognised tokens are passed through positionally so the existing
// `automedal run [N] [fast]` shape is preserved.
func ParseRunArgs(args []string) (remaining []string, env map[string]string) {
	env = map[string]string{}
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		tok := args[i]
		if tok == "--advisor" {
			env["AUTOMEDAL_ADVISOR"] = "1"
			next := ""
			if i+1 < len(args) {
				next = args[i+1]
			}
			if next != "" && !strings.HasPrefix(next, "--") && !isAllDigits(next) {
				env["AUTOMEDAL_ADVISOR_MODEL"] = next
				i++
			} else {
				env["AUTOMEDAL_ADVISOR_MODEL"] = DefaultAdvisorModel
			}
			continue
		}
		out = append(out, tok)
	}
	return out, env
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}
