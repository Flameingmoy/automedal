package components

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/Flameingmoy/automedal/internal/ui/theme"
	"github.com/Flameingmoy/automedal/internal/ui/util"
)

// RecentEntry is one row in the "recent activity" panel.
type RecentEntry struct {
	ID   string // e.g. "0042"
	Slug string // filename stem minus the id
}

// ReadRecent looks at `journal/NNNN-*.md` and returns up to `n` newest entries
// (sorted by numeric id, descending). Errors are swallowed — the panel
// shows "(no experiments yet)" in that case.
func ReadRecent(n int) []RecentEntry {
	dir := util.JournalDir()
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	re := regexp.MustCompile(`^(\d{4})-(.+)\.md$`)
	var out []RecentEntry
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		m := re.FindStringSubmatch(e.Name())
		if len(m) != 3 {
			continue
		}
		out = append(out, RecentEntry{ID: m[1], Slug: m[2]})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID > out[j].ID })
	if len(out) > n {
		out = out[:n]
	}
	return out
}

// RecentPanel renders the fixed-size box shown below the status strip.
func RecentPanel(n int, width int) string {
	rows := ReadRecent(n)
	var lines []string
	lines = append(lines, theme.Accent.Render("recent activity"))
	if len(rows) == 0 {
		lines = append(lines, theme.Muted.Render("(no experiments yet)"))
	} else {
		for _, r := range rows {
			slug := r.Slug
			if len(slug) > 40 {
				slug = slug[:37] + "…"
			}
			lines = append(lines, fmt.Sprintf("  #%s  %s", r.ID, slug))
		}
	}
	body := strings.Join(lines, "\n")
	if width <= 0 {
		return theme.Panel.Render(body)
	}
	// Account for border+padding: lipgloss subtracts automatically, we just
	// clamp the inner width so long lines wrap cleanly.
	return theme.Panel.Copy().Width(width - 2).Render(body)
}

// JournalTitle reads the first `# ` header of an entry file for the
// detail view. (Kept here because recent.go is already journal-aware.)
func JournalTitle(id string) string {
	dir := util.JournalDir()
	matches, _ := filepath.Glob(filepath.Join(dir, id+"-*.md"))
	if len(matches) == 0 {
		return ""
	}
	b, err := os.ReadFile(matches[0])
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "# ") {
			return strings.TrimSpace(strings.TrimPrefix(line, "# "))
		}
	}
	return ""
}
