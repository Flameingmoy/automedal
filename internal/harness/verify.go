// AutoMedal — Iteration Invariant Checker (Go port).
//
// Mirrors harness/verify_iteration.py. Public API:
//
//	report := harness.VerifyResearcher()
//	report := harness.VerifyStrategist()
//	report := harness.VerifyExperimenter(expID)
//	regWarnings := harness.CheckRegression(valLoss, bestBefore)
//	passed, near, crit := harness.CheckSuccessCriteria(expID, valLoss, bestSoFar)
//
// Each returns warnings as a []string. The cmd layer maps warnings →
// exit codes (1 = warnings, 2 = regression-strict, 3 = near-miss).
package harness

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/Flameingmoy/automedal/internal/paths"
)

// Public path helpers — mirror Python module-level constants but resolve
// from paths.RepoRoot() so user-mode + tests work transparently.
func KnowledgePath() string  { return filepath.Join(paths.RepoRoot(), "knowledge.md") }
func QueuePath() string      { return filepath.Join(paths.RepoRoot(), "experiment_queue.md") }
func ResearchPath() string   { return filepath.Join(paths.RepoRoot(), "research_notes.md") }
func JournalDir() string     { return filepath.Join(paths.RepoRoot(), "journal") }
func ResultsTSVPath() string { return filepath.Join(paths.RepoRoot(), "agent", "results.tsv") }

const (
	knowledgeBulletCap = 80
	queueEntryCount    = 5
	queueMaxPerAxis    = 2
)

var (
	validAxes = map[string]bool{
		"preprocessing": true, "feature-eng": true, "HPO": true,
		"new-model": true, "ensembling": true, "pseudo-label": true, "architecture": true,
	}
	validQueueStatuses    = map[string]bool{"pending": true, "running": true, "done": true}
	validJournalStatuses  = map[string]bool{"improved": true, "no_change": true, "worse": true, "crashed": true}
	requiredJournalKeys   = []string{"id", "slug", "timestamp", "git_tag", "queue_entry", "status", "val_loss", "val_accuracy", "best_so_far"}
	requiredJournalSects  = []string{"Hypothesis", "What I changed", "Result", "What I learned"}
	queueEntryRE          = regexp.MustCompile(`(?i)^##\s+\d+\.\s+(?P<slug>[a-z0-9-]+)\s+\[axis:\s*(?P<axis>[a-zA-Z-]+)\]\s+\[STATUS:\s*(?P<status>[a-zA-Z]+)\]`)
	queueHeaderRE         = regexp.MustCompile(`^##\s+\d+\.`)
	expCiteRE             = regexp.MustCompile(`(?i)\bexps?\s*0*\d+`)
	researchEntryHeaderRE = regexp.MustCompile(`(?m)^##\s+exp\s+\d+`)
	paperBulletRE         = regexp.MustCompile(`(?m)^-\s+Paper:`)
	scRE                  = regexp.MustCompile(`(?i)val_loss\s*(?P<op><=|>=|<|>|==)\s*(?P<val>[0-9.]+)(?:\s+or\s+val_loss\s*(?P<op2><=|>=|<|>|==)\s*(?P<expr2>best_so_far\s*\*\s*[0-9.]+|[0-9.]+))?`)
)

// readFile returns nil + ok=false if the file doesn't exist.
func readFile(path string) (string, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	return string(b), true
}

// parseKnowledgeBullets returns (section, bullet) pairs.
func parseKnowledgeBullets(text string) [][2]string {
	var out [][2]string
	current := ""
	for _, line := range strings.Split(text, "\n") {
		s := strings.TrimSpace(line)
		if strings.HasPrefix(s, "## ") {
			current = strings.TrimSpace(s[3:])
			continue
		}
		if strings.HasPrefix(s, "- ") {
			out = append(out, [2]string{current, strings.TrimSpace(s[2:])})
		}
	}
	return out
}

// knowledgeSections returns section title → []bullet.
func knowledgeSections(text string) map[string][]string {
	out := map[string][]string{}
	current := ""
	for _, line := range strings.Split(text, "\n") {
		s := strings.TrimSpace(line)
		if strings.HasPrefix(s, "## ") {
			current = strings.TrimSpace(s[3:])
			if _, ok := out[current]; !ok {
				out[current] = nil
			}
			continue
		}
		if strings.HasPrefix(s, "- ") && current != "" {
			out[current] = append(out[current], strings.TrimSpace(s[2:]))
		}
	}
	return out
}

type queueEntry struct {
	slug, axis, status              string
	hasHypothesis, hasSketch, hasExpected bool
}

func parseQueueEntries(text string) []queueEntry {
	var entries []queueEntry
	var current *queueEntry
	for _, line := range strings.Split(text, "\n") {
		s := strings.TrimSpace(line)
		if m := queueEntryRE.FindStringSubmatch(s); m != nil {
			if current != nil {
				entries = append(entries, *current)
			}
			current = &queueEntry{
				slug:   m[queueEntryRE.SubexpIndex("slug")],
				axis:   m[queueEntryRE.SubexpIndex("axis")],
				status: strings.ToLower(m[queueEntryRE.SubexpIndex("status")]),
			}
			continue
		}
		if current == nil {
			continue
		}
		lower := strings.ToLower(s)
		switch {
		case strings.HasPrefix(lower, "**hypothesis:**"):
			current.hasHypothesis = true
		case strings.HasPrefix(lower, "**sketch:**"):
			current.hasSketch = true
		case strings.HasPrefix(lower, "**expected:**"):
			current.hasExpected = true
		}
	}
	if current != nil {
		entries = append(entries, *current)
	}
	return entries
}

// parseJournalFrontmatter returns nil if the file lacks `---` framing.
func parseJournalFrontmatter(text string) map[string]string {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return nil
	}
	fm := map[string]string{}
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			break
		}
		if idx := strings.Index(lines[i], ":"); idx >= 0 {
			key := strings.TrimSpace(lines[i][:idx])
			val := strings.TrimSpace(lines[i][idx+1:])
			fm[key] = val
		}
	}
	return fm
}

// journalSections returns section title → trimmed body.
func journalSections(text string) map[string]string {
	out := map[string]string{}
	current := ""
	var body []string
	flush := func() {
		if current != "" {
			out[current] = strings.TrimSpace(strings.Join(body, "\n"))
		}
	}
	for _, line := range strings.Split(text, "\n") {
		s := strings.TrimSpace(line)
		if strings.HasPrefix(s, "## ") {
			flush()
			current = strings.TrimSpace(s[3:])
			body = nil
			continue
		}
		if current != "" {
			body = append(body, line)
		}
	}
	flush()
	return out
}

// ── phase checks ───────────────────────────────────────────────────────

// VerifyResearcher returns the warning list for the researcher phase.
func VerifyResearcher() []string {
	var warnings []string
	text, ok := readFile(ResearchPath())
	if !ok {
		return append(warnings, "research_notes.md does not exist")
	}
	parts := researchEntryHeaderRE.Split(text, -1)
	num := len(parts) - 1
	if num <= 0 {
		return append(warnings, "research_notes.md has no entries after header")
	}
	last := parts[len(parts)-1]
	papers := paperBulletRE.FindAllString(last, -1)
	if !(len(papers) >= 2 && len(papers) <= 3) {
		warnings = append(warnings, fmt.Sprintf(
			"last research_notes.md entry has %d papers (expected 2-3)", len(papers)))
	}
	if !strings.Contains(strings.ToLower(last), "query:") {
		warnings = append(warnings, "last research_notes.md entry missing 'query:' header marker")
	}
	return warnings
}

// VerifyStrategist returns the warning list for the strategist phase.
func VerifyStrategist() []string {
	var warnings []string

	if kbText, ok := readFile(KnowledgePath()); !ok {
		warnings = append(warnings, "knowledge.md does not exist")
	} else {
		bullets := parseKnowledgeBullets(kbText)
		if len(bullets) > knowledgeBulletCap {
			warnings = append(warnings, fmt.Sprintf(
				"knowledge.md has %d bullets (cap is %d)", len(bullets), knowledgeBulletCap))
		}
		sections := knowledgeSections(kbText)
		if oq, ok := sections["Open questions"]; !ok || len(oq) == 0 {
			warnings = append(warnings, "knowledge.md missing non-empty 'Open questions' section")
		}
		for _, b := range bullets {
			section, bullet := b[0], b[1]
			if section == "Open questions" {
				continue
			}
			if !expCiteRE.MatchString(bullet) {
				preview := bullet
				if len(preview) > 60 {
					preview = preview[:60]
				}
				warnings = append(warnings, fmt.Sprintf(
					"knowledge.md bullet missing exp citation: %q", preview))
				break
			}
		}
	}

	qText, ok := readFile(QueuePath())
	if !ok {
		warnings = append(warnings, "experiment_queue.md does not exist")
		return warnings
	}
	entries := parseQueueEntries(qText)
	if len(entries) != queueEntryCount {
		warnings = append(warnings, fmt.Sprintf(
			"experiment_queue.md has %d entries (expected %d)", len(entries), queueEntryCount))
	}
	axisCounts := map[string]int{}
	for _, e := range entries {
		if !validAxes[e.axis] {
			warnings = append(warnings, fmt.Sprintf(
				"queue entry '%s' has invalid axis: %s", e.slug, e.axis))
		}
		if !validQueueStatuses[e.status] {
			warnings = append(warnings, fmt.Sprintf(
				"queue entry '%s' has invalid status: %s", e.slug, e.status))
		}
		if !(e.hasHypothesis && e.hasSketch && e.hasExpected) {
			warnings = append(warnings, fmt.Sprintf(
				"queue entry '%s' missing Hypothesis/Sketch/Expected", e.slug))
		}
		axisCounts[e.axis]++
	}
	// Iterate axes in sorted order for stable output.
	axisKeys := make([]string, 0, len(axisCounts))
	for k := range axisCounts {
		axisKeys = append(axisKeys, k)
	}
	sort.Strings(axisKeys)
	for _, axis := range axisKeys {
		if c := axisCounts[axis]; c > queueMaxPerAxis {
			warnings = append(warnings, fmt.Sprintf(
				"queue has %d entries on axis '%s' (max %d)", c, axis, queueMaxPerAxis))
		}
	}
	return warnings
}

// VerifyExperimenter returns the warning list for the experimenter phase.
func VerifyExperimenter(expID string) []string {
	var warnings []string
	if expID == "" {
		return append(warnings, "experimenter check requires --exp-id")
	}

	var journals []string
	if dirEntries, err := os.ReadDir(JournalDir()); err == nil {
		prefix := expID + "-"
		for _, e := range dirEntries {
			if e.IsDir() {
				continue
			}
			n := e.Name()
			if strings.HasPrefix(n, prefix) && strings.HasSuffix(n, ".md") {
				journals = append(journals, n)
			}
		}
	}
	if len(journals) == 0 {
		return append(warnings, fmt.Sprintf("no journal entry found for exp %s", expID))
	}
	if len(journals) > 1 {
		warnings = append(warnings, fmt.Sprintf("multiple journal entries for exp %s: %v", expID, journals))
	}

	jPath := filepath.Join(JournalDir(), journals[0])
	text, _ := readFile(jPath)
	fm := parseJournalFrontmatter(text)
	if fm == nil {
		return append(warnings, fmt.Sprintf("journal %s missing frontmatter", journals[0]))
	}
	missing := []string{}
	for _, k := range requiredJournalKeys {
		if _, ok := fm[k]; !ok {
			missing = append(missing, k)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		warnings = append(warnings, fmt.Sprintf(
			"journal %s missing frontmatter keys: %v", journals[0], missing))
	}
	if !validJournalStatuses[fm["status"]] {
		warnings = append(warnings, fmt.Sprintf(
			"journal %s has invalid status: %q", journals[0], fm["status"]))
	}
	sections := journalSections(text)
	for _, name := range requiredJournalSects {
		if v, ok := sections[name]; !ok || v == "" {
			warnings = append(warnings, fmt.Sprintf(
				"journal %s missing section: %q", journals[0], name))
		}
	}
	kbText, _ := readFile(KnowledgePath())
	if len(parseKnowledgeBullets(kbText)) > 0 {
		if v, ok := sections["KB entries consulted"]; !ok || v == "" {
			warnings = append(warnings, fmt.Sprintf(
				"journal %s missing 'KB entries consulted' (KB is non-empty)", journals[0]))
		}
	}
	if _, err := os.Stat(ResultsTSVPath()); err != nil {
		warnings = append(warnings, "results.tsv missing")
	}
	return warnings
}

// CheckRegression returns a non-empty list when val_loss exceeds
// best_before by more than 1%.
func CheckRegression(valLoss, bestBefore float64) []string {
	if math.IsNaN(valLoss) || math.IsNaN(bestBefore) || bestBefore <= 0 {
		return nil
	}
	if valLoss > bestBefore*1.01 {
		pct := 100 * (valLoss/bestBefore - 1)
		return []string{fmt.Sprintf(
			"REGRESSION: val_loss %.4f exceeds best_before %.4f by %.1f%% (>1%% threshold)",
			valLoss, bestBefore, pct)}
	}
	return nil
}

// ── success_criteria ───────────────────────────────────────────────────

func evalCriterion(criteriaStr string, valLoss, bestSoFar float64) (passed, nearMiss bool) {
	if criteriaStr == "" {
		return true, false
	}
	m := scRE.FindStringSubmatch(criteriaStr)
	if m == nil {
		return true, false
	}
	op1 := m[scRE.SubexpIndex("op")]
	val1, ok1 := targetVal(m[scRE.SubexpIndex("val")], bestSoFar)
	if !ok1 {
		return true, false
	}
	op2 := m[scRE.SubexpIndex("op2")]
	expr2 := m[scRE.SubexpIndex("expr2")]

	clause1 := compareNum(valLoss, op1, val1)
	clause2 := false
	val2 := 0.0
	hasClause2 := false
	if op2 != "" && expr2 != "" {
		var ok2 bool
		val2, ok2 = targetVal(expr2, bestSoFar)
		if ok2 {
			clause2 = compareNum(valLoss, op2, val2)
			hasClause2 = true
		}
	}
	if clause1 || (hasClause2 && clause2) {
		return true, false
	}
	adj := valLoss * 0.99
	if compareNum(adj, op1, val1) {
		return false, true
	}
	if hasClause2 && compareNum(adj, op2, val2) {
		return false, true
	}
	return false, false
}

func targetVal(raw string, bestSoFar float64) (float64, bool) {
	raw = strings.TrimSpace(raw)
	if strings.Contains(strings.ToLower(raw), "best_so_far") {
		factor := 1.0
		if mf := regexp.MustCompile(`\*\s*([0-9.]+)`).FindStringSubmatch(raw); mf != nil {
			if f, err := strconv.ParseFloat(mf[1], 64); err == nil {
				factor = f
			}
		}
		return bestSoFar * factor, true
	}
	if v, err := strconv.ParseFloat(raw, 64); err == nil {
		return v, true
	}
	return 0, false
}

func compareNum(vl float64, op string, target float64) bool {
	switch op {
	case "<=":
		return vl <= target
	case ">=":
		return vl >= target
	case "<":
		return vl < target
	case ">":
		return vl > target
	case "==":
		return math.Abs(vl-target) < 1e-8
	}
	return false
}

// CheckSuccessCriteria parses the running queue entry's success_criteria
// and evaluates it. Returns (passed, nearMiss, criteriaStr).
func CheckSuccessCriteria(expID string, valLoss, bestSoFar float64) (bool, bool, string) {
	qText, ok := readFile(QueuePath())
	if !ok || qText == "" {
		return true, false, ""
	}
	current := false
	inEntry := false
	for _, line := range strings.Split(qText, "\n") {
		s := strings.TrimSpace(line)
		if queueHeaderRE.MatchString(s) {
			inEntry = true
		}
		if inEntry && strings.Contains(strings.ToLower(s), "running") {
			current = true
			break
		}
	}
	if !current {
		return true, false, ""
	}
	criteriaStr := ""
	for _, line := range strings.Split(qText, "\n") {
		ls := strings.ToLower(strings.TrimSpace(line))
		if strings.HasPrefix(ls, "success_criteria:") {
			if i := strings.Index(line, ":"); i >= 0 {
				criteriaStr = strings.TrimSpace(line[i+1:])
				break
			}
		}
	}
	if criteriaStr == "" {
		return true, false, ""
	}
	passed, nearMiss := evalCriterion(criteriaStr, valLoss, bestSoFar)
	return passed, nearMiss, criteriaStr
}
