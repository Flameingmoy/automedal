package components

import (
	"strings"

	"github.com/cdharmaraj/automedal-tui/theme"
)

// Sparkline renders `points` as a one-line mini-chart using the eight
// Unicode bar glyphs U+2581..U+2588. Empty → a muted placeholder.
func Sparkline(points []float64, width int) string {
	if width <= 0 {
		width = 40
	}
	if len(points) == 0 {
		return theme.Muted.Render(strings.Repeat("·", width))
	}
	// Downsample if we have more points than columns.
	pts := points
	if len(pts) > width {
		pts = pts[len(pts)-width:]
	}

	// Normalize to [0,1]; INVERTED so lower-loss plots as taller bars
	// (down = better is confusing in a bar chart).
	min, max := pts[0], pts[0]
	for _, v := range pts {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	span := max - min
	if span == 0 {
		span = 1
	}

	bars := []rune{'▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'}
	var sb strings.Builder
	for _, v := range pts {
		norm := 1.0 - ((v - min) / span)
		idx := int(norm * float64(len(bars)-1))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(bars) {
			idx = len(bars) - 1
		}
		sb.WriteRune(bars[idx])
	}
	padded := sb.String()
	if len(padded) < width {
		padded = strings.Repeat(" ", width-len(padded)) + padded
	}
	return theme.Accent.Render(padded)
}

// LossPanel wraps the sparkline with a title and the current loss value.
func LossPanel(state interface {
	Series() []float64
	Last() (float64, bool)
}, width int) string {
	last, ok := state.Last()
	title := theme.Accent.Render("val_loss")
	subtitle := theme.Muted.Render("— no points yet —")
	if ok {
		subtitle = theme.Muted.Render("last:") + " " + theme.OKStyle.Render(f4(last))
	}
	spark := Sparkline(state.Series(), width-4)
	body := title + "   " + subtitle + "\n" + spark
	return theme.Panel.Copy().Width(width - 2).Render(body)
}

func f4(v float64) string {
	// small, deliberate float→string that doesn't pull in fmt for a hot path
	// (called on every render). Keep 4 decimals to match existing TSV.
	return fmt4(v)
}

// fmt4 is a tiny formatter — we call it a lot and want zero allocations
// past its single string build. Delegates to strconv via fmt in the
// rare case of non-finite.
func fmt4(v float64) string {
	// Fine to use strconv for correctness; avoid fmt.Sprintf hot-path alloc.
	return strconvF4(v)
}
