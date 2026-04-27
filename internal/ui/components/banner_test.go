package components

import (
	"regexp"
	"strings"
	"testing"
)

// Strips all CSI escape sequences so we can count cells, not bytes.
var ansiRE = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)

func TestBannerHasSixRows(t *testing.T) {
	out := Banner("AUTOMEDAL")
	rows := strings.Split(out, "\n")
	if len(rows) != 6 {
		t.Fatalf("Banner(AUTOMEDAL) rows = %d, want 6", len(rows))
	}
}

func TestBannerWidthMatchesLongestRow(t *testing.T) {
	w := BannerWidth("AUTOMEDAL")
	if w <= 0 {
		t.Fatalf("BannerWidth = %d, want > 0", w)
	}
	out := Banner("AUTOMEDAL")
	for _, row := range strings.Split(out, "\n") {
		// Strip ANSI; count runes.
		stripped := ansiRE.ReplaceAllString(row, "")
		got := 0
		for range stripped {
			got++
		}
		if got > w {
			t.Fatalf("row width %d exceeds BannerWidth %d", got, w)
		}
	}
}

func TestBannerAnsiDensityIsTight(t *testing.T) {
	// One style per row → 2 escape sequences (open + reset) per row.
	// Allow a small slack (4) for lipgloss internals; the bar is just
	// "no per-cell nesting" — anything > 8 means we regressed to the
	// hand-coded pixel font.
	out := Banner("AUTOMEDAL")
	for i, row := range strings.Split(out, "\n") {
		matches := ansiRE.FindAllString(row, -1)
		if len(matches) > 8 {
			t.Fatalf("row %d has %d ANSI escapes — per-cell styling regression",
				i, len(matches))
		}
	}
}
