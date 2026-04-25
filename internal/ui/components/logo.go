// Package components — small UI primitives (logo, status bar, recent
// activity, sparkline, leaderboard). Deliberately dumb: no tea.Model,
// just functions that return styled strings.
package components

import (
	"bytes"
	"fmt"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/Flameingmoy/automedal/internal/ui/util"
	"github.com/charmbracelet/lipgloss"
)

// Logo renders the AutoMedal logo as Unicode block-chars by downsampling
// tui/assets/logo/automedal.png at startup. If the PNG is missing we
// fall back to the spaced text we've always used.
//
// We cache the rendered string so repeated calls are free.
var _logoCache string

func Logo(width int) string {
	if _logoCache == "" {
		_logoCache = renderLogo()
	}
	return theme.Logo.Copy().Width(width).Render(_logoCache)
}

// LogoBytes returns the raw (unstyled) logo so the caller can embed it
// inside another style.
func LogoBytes() string {
	if _logoCache == "" {
		_logoCache = renderLogo()
	}
	return _logoCache
}

func renderLogo() string {
	path := filepath.Join(util.RepoRoot(), "tui", "assets", "logo", "automedal.png")
	img, err := loadPNG(path)
	if err != nil {
		return fallbackLogo()
	}
	return pngToBlocks(img, 48, 10)
}

func loadPNG(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

// pngToBlocks downsamples an image to (cols × rows) cells and picks a
// block character per cell by average luminance.  We use the "shade"
// glyphs (U+2591..U+2588) because they render more evenly at varying
// terminal DPIs than the two-row half-block trick.
func pngToBlocks(img image.Image, cols, rows int) string {
	if img == nil {
		return fallbackLogo()
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 {
		return fallbackLogo()
	}
	palette := []rune{' ', '░', '▒', '▓', '█'}
	var buf bytes.Buffer
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Average luminance over the source region this cell covers.
			x0 := bounds.Min.X + x*w/cols
			x1 := bounds.Min.X + (x+1)*w/cols
			y0 := bounds.Min.Y + y*h/rows
			y1 := bounds.Min.Y + (y+1)*h/rows
			if x1 <= x0 {
				x1 = x0 + 1
			}
			if y1 <= y0 {
				y1 = y0 + 1
			}
			var sum uint32
			var n uint32
			var sumAlpha uint32
			for py := y0; py < y1; py++ {
				for px := x0; px < x1; px++ {
					r, g, b, a := img.At(px, py).RGBA()
					sum += (r + g + b) / 3
					sumAlpha += a
					n++
				}
			}
			if n == 0 {
				buf.WriteRune(' ')
				continue
			}
			avg := sum / n
			alpha := sumAlpha / n
			// Transparent or near-black → space.
			if alpha < 0x4000 {
				buf.WriteRune(' ')
				continue
			}
			idx := int(avg*uint32(len(palette)-1)) / 0xFFFF
			if idx < 0 {
				idx = 0
			}
			if idx >= len(palette) {
				idx = len(palette) - 1
			}
			buf.WriteRune(palette[idx])
		}
		buf.WriteByte('\n')
	}
	return strings.TrimRight(buf.String(), "\n")
}

func fallbackLogo() string {
	return lipgloss.NewStyle().
		Foreground(lipgloss.Color(theme.ColorLogo)).
		Bold(true).
		Render("A U T O M E D A L")
}

// ShortBanner is the small title line under the logo.
func ShortBanner(subtitle string) string {
	return theme.Muted.Render(fmt.Sprintf("  ▸ %s", subtitle))
}
