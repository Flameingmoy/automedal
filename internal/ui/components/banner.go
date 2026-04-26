package components

import (
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/charmbracelet/lipgloss"
)

// 6-row chunky block-letter font for the Home banner.  Each glyph is
// 6 rows tall × 5 cols wide.  Fill cell = '█', empty = ' '.
// Designed to render with a per-column blue→cyan→jade gradient — see
// AutoMedal TUI v2 design notes.
var glyphs = map[byte][6]string{
	'A': {
		" ███ ",
		"█   █",
		"█   █",
		"█████",
		"█   █",
		"█   █",
	},
	'U': {
		"█   █",
		"█   █",
		"█   █",
		"█   █",
		"█   █",
		" ███ ",
	},
	'T': {
		"█████",
		"  █  ",
		"  █  ",
		"  █  ",
		"  █  ",
		"  █  ",
	},
	'O': {
		" ███ ",
		"█   █",
		"█   █",
		"█   █",
		"█   █",
		" ███ ",
	},
	'M': {
		"█   █",
		"██ ██",
		"█ █ █",
		"█   █",
		"█   █",
		"█   █",
	},
	'E': {
		"█████",
		"█    ",
		"████ ",
		"█    ",
		"█    ",
		"█████",
	},
	'D': {
		"████ ",
		"█   █",
		"█   █",
		"█   █",
		"█   █",
		"████ ",
	},
	'L': {
		"█    ",
		"█    ",
		"█    ",
		"█    ",
		"█    ",
		"█████",
	},
	' ': {
		"     ",
		"     ",
		"     ",
		"     ",
		"     ",
		"     ",
	},
}

// Banner renders `text` in the chunky pixel font with a per-column
// blue→cyan→jade gradient (crush-inspired).  Returns one styled string
// containing 6 lines joined by '\n'.
func Banner(text string) string {
	text = strings.ToUpper(text)

	const rows = 6
	const colsPerGlyph = 5
	const gap = 1

	width := 0
	for i := 0; i < len(text); i++ {
		if _, ok := glyphs[text[i]]; ok {
			width += colsPerGlyph
			if i < len(text)-1 {
				width += gap
			}
		}
	}
	if width == 0 {
		return ""
	}

	colors := theme.GradientColors(width)

	var rowsBuf [rows]strings.Builder
	col := 0
	for i := 0; i < len(text); i++ {
		g, ok := glyphs[text[i]]
		if !ok {
			continue
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < colsPerGlyph; c++ {
				ch := g[r][c]
				color := colors[col+c]
				if ch == ' ' {
					rowsBuf[r].WriteByte(' ')
				} else {
					st := lipgloss.NewStyle().Foreground(color).Bold(true)
					rowsBuf[r].WriteString(st.Render(string(ch)))
				}
			}
		}
		col += colsPerGlyph
		if i < len(text)-1 {
			for r := 0; r < rows; r++ {
				rowsBuf[r].WriteByte(' ')
			}
			col += gap
		}
	}

	out := make([]string, rows)
	for r := 0; r < rows; r++ {
		out[r] = rowsBuf[r].String()
	}
	return strings.Join(out, "\n")
}

// BannerWidth returns the printable width of Banner(text) so callers can
// centre or measure it without re-running the renderer.
func BannerWidth(text string) int {
	width := 0
	for i := 0; i < len(text); i++ {
		if _, ok := glyphs[text[i]]; ok {
			width += 5
			if i < len(text)-1 {
				width++
			}
		}
	}
	return width
}
