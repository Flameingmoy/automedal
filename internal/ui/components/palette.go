package components

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/Flameingmoy/automedal/internal/ui/util"
)

// Commands mirrors tui/widgets/command_input.py:COMMANDS — keep in sync.
var Commands = []string{
	"run", "init", "discover", "select", "doctor", "status",
	"clean", "prepare", "render", "setup", "models", "help", "quit",
}

// QuitAliases — never spawn a subprocess for these, call tea.Quit directly.
// Mirrors tui/screens/home.py:_QUIT_ALIASES.
var QuitAliases = map[string]struct{}{
	"q": {}, "quit": {}, "exit": {}, ":q": {}, ":quit": {}, ":wq": {},
}

// IsQuit returns true if the (lowercased) first word is a quit alias.
func IsQuit(text string) bool {
	parts := strings.Fields(strings.TrimSpace(text))
	if len(parts) == 0 {
		return false
	}
	_, ok := QuitAliases[strings.ToLower(parts[0])]
	return ok
}

var trailingDigits = regexp.MustCompile(`^([a-zA-Z]+)(\d+)$`)

// Normalize: 'run30' → 'run 30'. Mirrors command_input.py:normalize.
func Normalize(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	first, rest, hasRest := strings.Cut(text, " ")
	m := trailingDigits.FindStringSubmatch(first)
	if m != nil && containsLower(Commands, strings.ToLower(m[1])) {
		fixed := m[1] + " " + m[2]
		if hasRest {
			return strings.TrimSpace(fixed + " " + rest)
		}
		return fixed
	}
	return text
}

// SplitForAdvisor mirrors _split_for_advisor — detect `--advisor <prefix>`
// at the end of the typed text. Returns (isAdvisor, prefix, head).
// `head` includes `--advisor ` so the completion can splice.
func SplitForAdvisor(text string) (bool, string, string) {
	parts := strings.Split(text, " ")
	advIdx := -1
	for i, p := range parts {
		if p == "--advisor" {
			advIdx = i
		}
	}
	if advIdx < 0 {
		return false, "", text
	}
	// Cursor sits right after `--advisor ` with no prefix yet.
	if advIdx == len(parts)-1 {
		head := strings.Join(parts[:advIdx+1], " ") + " "
		return true, "", head
	}
	if advIdx+1 == len(parts)-1 {
		prefix := parts[advIdx+1]
		head := strings.Join(parts[:advIdx+1], " ") + " "
		return true, prefix, head
	}
	return false, "", text
}

// ModelsForAutocomplete reads ~/.automedal/models_cache.json and returns
// the cached advisor model ids. Empty on any error — callers show
// "no models cached".
func ModelsForAutocomplete() []string {
	b, err := os.ReadFile(util.ModelsCachePath())
	if err != nil {
		return nil
	}
	// The cache file is a JSON list-of-strings per automedal/advisor/models.py.
	// Tolerate both `["id",...]` and `{"models": [...]}`.
	var arr []string
	if err := json.Unmarshal(b, &arr); err == nil {
		return arr
	}
	var wrapped struct {
		Models []string `json:"models"`
	}
	if err := json.Unmarshal(b, &wrapped); err == nil {
		return wrapped.Models
	}
	return nil
}

// Autocomplete returns the new text and whether a completion fired.
// Mirrors the single-match-append-trailing-space behavior we just landed
// in command_input.py:_autocomplete (2026-04-24).
func Autocomplete(text string) (string, bool) {
	text = Normalize(text)
	if text == "" {
		return text, false
	}

	// Model completion after `--advisor `.
	if isAdv, prefix, head := SplitForAdvisor(text); isAdv {
		models := ModelsForAutocomplete()
		var matches []string
		for _, m := range models {
			if strings.HasPrefix(m, prefix) {
				matches = append(matches, m)
			}
		}
		if len(matches) == 1 {
			return head + matches[0], true
		}
		if len(matches) > 1 {
			// Longest common prefix so a second Tab narrows further.
			lcp := matches[0]
			for _, m := range matches[1:] {
				for !strings.HasPrefix(m, lcp) {
					lcp = lcp[:len(lcp)-1]
					if lcp == "" {
						break
					}
				}
				if lcp == "" {
					break
				}
			}
			if lcp != "" && lcp != prefix {
				return head + lcp, true
			}
		}
		return text, false
	}

	// Command completion on first word.
	first, rest, hasRest := strings.Cut(text, " ")
	word := strings.ToLower(first)
	var matches []string
	for _, c := range Commands {
		if strings.HasPrefix(c, word) {
			matches = append(matches, c)
		}
	}
	if len(matches) == 1 {
		completed := matches[0]
		if hasRest {
			return completed + " " + rest, true
		}
		// Trailing space so user can start typing args immediately.
		return completed + " ", true
	}
	return text, false
}

// HintLine renders the muted suggestion row under the prompt.
func HintLine(text string) string {
	text = Normalize(text)
	if text == "" {
		return theme.Muted.Render("  " + strings.Join(Commands[:6], "  "))
	}
	if isAdv, prefix, _ := SplitForAdvisor(text); isAdv {
		models := ModelsForAutocomplete()
		if len(models) == 0 {
			return theme.Muted.Render("  (no models cached — run 'automedal models refresh')")
		}
		var matches []string
		for _, m := range models {
			if prefix == "" || strings.HasPrefix(m, prefix) {
				matches = append(matches, m)
			}
		}
		if len(matches) == 0 {
			return theme.Muted.Render("  (no model starts with " + prefix + ")")
		}
		head := matches
		tail := ""
		if len(matches) > 6 {
			head = matches[:6]
			tail = "  +" + itoa(len(matches)-6)
		}
		return theme.Muted.Render("  " + strings.Join(head, "  ") + tail)
	}
	first, _, _ := strings.Cut(text, " ")
	word := strings.ToLower(first)
	if word == "" {
		return theme.Muted.Render("  " + strings.Join(Commands[:6], "  "))
	}
	var matches []string
	for _, c := range Commands {
		if strings.HasPrefix(c, word) {
			matches = append(matches, c)
		}
	}
	if len(matches) == 0 {
		return theme.Muted.Render("  (no matches)")
	}
	if len(matches) > 6 {
		matches = matches[:6]
	}
	return theme.Muted.Render("  " + strings.Join(matches, "  "))
}

func containsLower(xs []string, x string) bool {
	for _, s := range xs {
		if s == x {
			return true
		}
	}
	return false
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
