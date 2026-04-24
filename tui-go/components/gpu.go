package components

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/cdharmaraj/automedal-tui/theme"
)

// GpuSample is what `nvidia-smi` gives us in one poll.
type GpuSample struct {
	Util     int    // 0..100
	MemUsed  int    // MiB
	MemTotal int    // MiB
	Name     string // GPU product name
	OK       bool   // false if nvidia-smi failed / missing
}

// Poll runs nvidia-smi once with a short timeout. Returns OK=false on any
// failure (driver missing, binary missing, parse error) — caller hides
// the panel.
func Poll() GpuSample {
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()

	cmd := exec.CommandContext(ctx,
		"nvidia-smi",
		"--query-gpu=utilization.gpu,memory.used,memory.total,name",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		return GpuSample{}
	}
	line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	parts := strings.Split(line, ", ")
	if len(parts) < 4 {
		return GpuSample{}
	}
	util, err1 := strconv.Atoi(parts[0])
	used, err2 := strconv.Atoi(parts[1])
	total, err3 := strconv.Atoi(parts[2])
	if err1 != nil || err2 != nil || err3 != nil {
		return GpuSample{}
	}
	return GpuSample{
		Util: util, MemUsed: used, MemTotal: total,
		Name: parts[3], OK: true,
	}
}

// GpuPanel renders a small two-bar panel (util + memory). Empty string
// when sample is not OK — the dashboard collapses the row.
func GpuPanel(s GpuSample, width int) string {
	if !s.OK {
		return ""
	}
	title := theme.Accent.Render("gpu") + "  " + theme.Muted.Render(s.Name)
	barWidth := width - 18
	if barWidth < 10 {
		barWidth = 10
	}

	util := bar(s.Util, 100, barWidth)
	var memPct int
	if s.MemTotal > 0 {
		memPct = 100 * s.MemUsed / s.MemTotal
	}
	mem := bar(memPct, 100, barWidth)

	body := title + "\n" +
		"  util  " + util + "  " + padLeft(strconv.Itoa(s.Util)+"%", 5) + "\n" +
		"  mem   " + mem + "  " + padLeft(strconv.Itoa(s.MemUsed)+"/"+strconv.Itoa(s.MemTotal)+"M", 12)
	return theme.Panel.Copy().Width(width - 2).Render(body)
}

func bar(cur, max, width int) string {
	if max == 0 {
		return strings.Repeat(" ", width)
	}
	filled := cur * width / max
	if filled < 0 {
		filled = 0
	}
	if filled > width {
		filled = width
	}
	return theme.OKStyle.Render(strings.Repeat("█", filled)) +
		theme.Muted.Render(strings.Repeat("░", width-filled))
}

func padLeft(s string, w int) string {
	if len(s) >= w {
		return s
	}
	return strings.Repeat(" ", w-len(s)) + s
}
