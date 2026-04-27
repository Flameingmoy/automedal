package components

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// AUTOMEDAL wordmark in the FIGlet "ANSI Shadow" font — the same chunky
// cube-letter style Charm's crush uses for its splash.  Pre-rendered
// here so we apply ONE foreground style per row at render time and
// avoid the per-cell ANSI nesting that mangled the previous pixel-font
// implementation.
const automedalArt = ` █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ███╗███████╗██████╗  █████╗ ██╗
██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗ ████║██╔════╝██╔══██╗██╔══██╗██║
███████║██║   ██║   ██║   ██║   ██║██╔████╔██║█████╗  ██║  ██║███████║██║
██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╔╝██║██╔══╝  ██║  ██║██╔══██║██║
██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗██████╔╝██║  ██║███████╗
╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝╚══════╝`

// generic ANSI Shadow glyphs for any other word a caller passes in.
// Used as a fallback so Banner("X") still produces something readable
// (lower fidelity than the curated AUTOMEDAL art).
var ansiShadowGlyphs = map[byte][6]string{
	'A': {
		" █████╗ ",
		"██╔══██╗",
		"███████║",
		"██╔══██║",
		"██║  ██║",
		"╚═╝  ╚═╝",
	},
	'U': {
		"██╗   ██╗",
		"██║   ██║",
		"██║   ██║",
		"██║   ██║",
		"╚██████╔╝",
		" ╚═════╝ ",
	},
	'T': {
		"████████╗",
		"╚══██╔══╝",
		"   ██║   ",
		"   ██║   ",
		"   ██║   ",
		"   ╚═╝   ",
	},
	'O': {
		" ██████╗ ",
		"██╔═══██╗",
		"██║   ██║",
		"██║   ██║",
		"╚██████╔╝",
		" ╚═════╝ ",
	},
	'M': {
		"███╗   ███╗",
		"████╗ ████║",
		"██╔████╔██║",
		"██║╚██╔╝██║",
		"██║ ╚═╝ ██║",
		"╚═╝     ╚═╝",
	},
	'E': {
		"███████╗",
		"██╔════╝",
		"█████╗  ",
		"██╔══╝  ",
		"███████╗",
		"╚══════╝",
	},
	'D': {
		"██████╗ ",
		"██╔══██╗",
		"██║  ██║",
		"██║  ██║",
		"██████╔╝",
		"╚═════╝ ",
	},
	'L': {
		"██╗     ",
		"██║     ",
		"██║     ",
		"██║     ",
		"███████╗",
		"╚══════╝",
	},
	' ': {
		"  ",
		"  ",
		"  ",
		"  ",
		"  ",
		"  ",
	},
}

// bannerRows returns the 6 raw (un-styled) rows of the wordmark for `text`.
// For "AUTOMEDAL" we use the curated art constant; for anything else we
// stitch glyphs together from ansiShadowGlyphs with a 1-col gap.
func bannerRows(text string) []string {
	upper := strings.ToUpper(strings.TrimSpace(text))
	if upper == "AUTOMEDAL" {
		return strings.Split(automedalArt, "\n")
	}
	const rows = 6
	var rb [rows]strings.Builder
	for i := 0; i < len(upper); i++ {
		g, ok := ansiShadowGlyphs[upper[i]]
		if !ok {
			continue
		}
		for r := 0; r < rows; r++ {
			rb[r].WriteString(g[r])
			if i < len(upper)-1 {
				rb[r].WriteByte(' ')
			}
		}
	}
	out := make([]string, rows)
	for r := 0; r < rows; r++ {
		out[r] = rb[r].String()
	}
	return out
}

// Banner renders `text` as an ANSI-Shadow wordmark with a per-row
// gradient (jade → cyan → neon-blue, top → bottom).  Each row is wrapped
// in exactly one lipgloss style — total ANSI cost is roughly two
// escape sequences per row, which keeps every terminal happy.
func Banner(text string) string {
	rows := bannerRows(text)
	if len(rows) == 0 {
		return ""
	}
	colors := theme.GradientRows(len(rows))
	out := make([]string, len(rows))
	for i, r := range rows {
		st := lipgloss.NewStyle().
			Foreground(colors[i]).
			Bold(true)
		out[i] = st.Render(r)
	}
	return strings.Join(out, "\n")
}

// BannerWidth returns the printable width of the longest row of
// Banner(text), in cells.  Safe for callers that need to centre or
// decide a side-by-side layout — counts runes (not bytes).
func BannerWidth(text string) int {
	rows := bannerRows(text)
	max := 0
	for _, r := range rows {
		w := lipgloss.Width(r)
		if w > max {
			max = w
		}
	}
	return max
}
