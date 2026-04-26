// Two-stage heuristic scoring. Mirrors scout/scoring.py.
package scout

import (
	"regexp"
	"strings"
	"time"
)

// Competition is the metadata shape used by scoring + select. Mirrors the
// dict produced by scout/discover.py:_competition_to_dict.
type Competition struct {
	Ref                      string     `json:"ref"`
	Title                    string     `json:"title"`
	Description              string     `json:"description"`
	URL                      string     `json:"url"`
	Category                 string     `json:"category"`
	Reward                   any        `json:"reward,omitempty"`
	Deadline                 string     `json:"deadline,omitempty"`
	TeamCount                int        `json:"teamCount"`
	IsKernelsSubmissionsOnly bool       `json:"isKernelsSubmissionsOnly"`
	EvaluationMetric         string     `json:"evaluationMetric"`
	MaxDailySubmissions      any        `json:"maxDailySubmissions,omitempty"`
	MaxTeamSize              any        `json:"maxTeamSize,omitempty"`
	EnabledDate              string     `json:"enabledDate,omitempty"`
	Tags                     []string   `json:"tags"`
	Files                    []FileInfo `json:"files,omitempty"`
}

// FileInfo describes one competition file.
type FileInfo struct {
	Name       string `json:"name"`
	TotalBytes int64  `json:"totalBytes"`
}

var (
	knownTabularMetrics = []string{
		"accuracy", "logloss", "log_loss", "categoricalcrossentropy",
		"categoricalaccuracy", "auc", "rmse", "mae", "r2", "f1",
		"rootmeansquarederror", "rootmeansquaredlogerror",
		"meanabsoluteerror", "meansquarederror", "mse",
		"multiclassloss", "mapk", "map@k", "quadraticweightedkappa",
		"weightedlogloss", "binarycrossentropy", "f1score",
		"rmsle", "medianabsoluteerror", "spearman",
	}
	nonTabularTags = map[string]bool{
		"nlp": true, "computer vision": true, "image classification": true,
		"object detection": true, "text": true, "audio": true, "video": true,
		"segmentation": true, "image": true,
		"natural language processing": true, "speech": true,
		"generative ai": true, "large language models": true,
		"diffusion": true, "gan": true,
	}
	nonTabularDescPatterns = []*regexp.Regexp{
		regexp.MustCompile(`\bimage\b`), regexp.MustCompile(`\bnlp\b`),
		regexp.MustCompile(`\btext classification\b`), regexp.MustCompile(`\bsegmentation\b`),
		regexp.MustCompile(`\bobject detection\b`), regexp.MustCompile(`\bcomputer vision\b`),
		regexp.MustCompile(`\bspeech\b`), regexp.MustCompile(`\baudio\b`),
		regexp.MustCompile(`\bvideo\b`), regexp.MustCompile(`\bpixel\b`),
		regexp.MustCompile(`\btoken\b`), regexp.MustCompile(`\btransformer\b`),
		regexp.MustCompile(`\bbert\b`), regexp.MustCompile(`\bgpt\b`),
		regexp.MustCompile(`\bllm\b`), regexp.MustCompile(`\bdiffusion\b`),
	}
	nonTabularExtensions = map[string]bool{
		".jpg": true, ".jpeg": true, ".png": true, ".gif": true, ".bmp": true,
		".tiff": true, ".webp": true, ".mp3": true, ".wav": true, ".flac": true,
		".mp4": true, ".avi": true, ".mov": true,
		".json": true, ".jsonl": true, ".txt": true, ".parquet": true,
	}
	nonAlnumRE = regexp.MustCompile(`[^a-z0-9]`)
)

func normalizeMetric(s string) string {
	if s == "" {
		return ""
	}
	return nonAlnumRE.ReplaceAllString(strings.ToLower(s), "")
}

// daysUntilDeadline returns the day count (negative when past). -1 means
// unparseable; the caller treats -1 like nil in the Python impl.
func daysUntilDeadline(s string) (days int, ok bool) {
	if s == "" {
		return 0, false
	}
	cleaned := strings.ReplaceAll(s, "Z", "+00:00")
	t, err := time.Parse(time.RFC3339, cleaned)
	if err != nil {
		t, err = time.Parse("2006-01-02T15:04:05.999999-07:00", cleaned)
		if err != nil {
			return 0, false
		}
	}
	return int(time.Until(t.UTC()).Hours() / 24), true
}

func tagsLower(tags []string) map[string]bool {
	out := map[string]bool{}
	for _, t := range tags {
		if t = strings.ToLower(strings.TrimSpace(t)); t != "" {
			out[t] = true
		}
	}
	return out
}

// ScoreStage1 scores a competition by metadata only. Returns (score,
// reasons, disqualified).
func ScoreStage1(c Competition) (int, []string, bool) {
	score := 0
	var reasons []string

	if c.IsKernelsSubmissionsOnly {
		return 0, []string{"DISQUALIFIED: kernels-only submissions"}, true
	}
	if days, ok := daysUntilDeadline(c.Deadline); ok && days < 7 {
		return 0, []string{stringf("DISQUALIFIED: deadline in %d days", days)}, true
	}

	if strings.Contains(strings.ToLower(c.Ref), "playground-series") {
		score += 40
		reasons = append(reasons, "+40: Playground Series (slug match)")
	}
	cat := strings.ToLower(c.Category)
	switch cat {
	case "getting started":
		score += 30
		reasons = append(reasons, "+30: Getting Started category")
	case "playground":
		score += 25
		reasons = append(reasons, "+25: Playground category")
	case "featured":
		score += 5
		reasons = append(reasons, "+5: Featured category")
	}

	metricNorm := normalizeMetric(c.EvaluationMetric)
	knownNorms := map[string]bool{}
	for _, m := range knownTabularMetrics {
		knownNorms[normalizeMetric(m)] = true
	}
	if metricNorm != "" && knownNorms[metricNorm] {
		score += 20
		reasons = append(reasons, stringf("+20: known tabular metric (%s)", c.EvaluationMetric))
	}

	tl := tagsLower(c.Tags)
	if tl["tabular"] {
		score += 20
		reasons = append(reasons, "+20: tagged 'tabular'")
	}

	switch {
	case c.TeamCount > 2000:
		score += 15
		reasons = append(reasons, stringf("+15: very active (%d teams)", c.TeamCount))
	case c.TeamCount > 500:
		score += 10
		reasons = append(reasons, stringf("+10: active (%d teams)", c.TeamCount))
	case c.TeamCount > 100:
		score += 5
		reasons = append(reasons, stringf("+5: moderate activity (%d teams)", c.TeamCount))
	}

	if days, ok := daysUntilDeadline(c.Deadline); ok && days > 30 {
		score += 5
		reasons = append(reasons, stringf("+5: comfortable deadline (%d days)", days))
	}

	badTags := []string{}
	for t := range tl {
		if nonTabularTags[t] {
			badTags = append(badTags, t)
		}
	}
	if len(badTags) > 0 {
		score -= 50
		reasons = append(reasons, stringf("-50: non-tabular tags (%s)", strings.Join(badTags, ", ")))
	}

	descLower := strings.ToLower(c.Description)
	hasDescMatch := false
	for _, re := range nonTabularDescPatterns {
		if re.FindStringIndex(descLower) != nil {
			hasDescMatch = true
			break
		}
	}
	if hasDescMatch && len(badTags) == 0 {
		score -= 30
		reasons = append(reasons, "-30: non-tabular description keywords")
	}

	if c.EvaluationMetric == "" {
		score -= 10
		reasons = append(reasons, "-10: no evaluation metric specified")
	} else if metricNorm != "" && !knownNorms[metricNorm] {
		score -= 5
		reasons = append(reasons, stringf("-5: unknown metric (%s)", c.EvaluationMetric))
	}

	if cat == "research" {
		score -= 20
		reasons = append(reasons, "-20: Research category (non-standard)")
	}

	return score, reasons, false
}

// ScoreStage2 scores a competition by file listing. (score, reasons).
func ScoreStage2(files []FileInfo) (int, []string) {
	if len(files) == 0 {
		return 0, []string{"+0: no file listing available"}
	}
	score := 0
	var reasons []string

	totalBytes := int64(0)
	exts := map[string]bool{}
	names := []string{}
	for _, f := range files {
		n := strings.ToLower(f.Name)
		names = append(names, n)
		totalBytes += f.TotalBytes
		if i := strings.LastIndex(n, "."); i >= 0 {
			exts["."+n[i+1:]] = true
		}
	}

	hasName := func(targets ...string) bool {
		for _, n := range names {
			for _, t := range targets {
				if n == t {
					return true
				}
			}
		}
		return false
	}

	if hasName("train.csv", "train.csv.zip", "train.csv.gz") {
		score += 25
		reasons = append(reasons, "+25: train.csv found")
	}
	if hasName("test.csv", "test.csv.zip", "test.csv.gz") {
		score += 15
		reasons = append(reasons, "+15: test.csv found")
	}
	hasSample := false
	for _, n := range names {
		if strings.Contains(n, "sample") && strings.Contains(n, "submission") {
			hasSample = true
			break
		}
	}
	if hasSample {
		score += 10
		reasons = append(reasons, "+10: sample_submission.csv found")
	}

	csvOK := map[string]bool{".csv": true, ".csv.zip": true, ".csv.gz": true, ".zip": true}
	allCSV := len(names) > 0
	for e := range exts {
		if e == "" {
			continue
		}
		if !csvOK[e] {
			allCSV = false
			break
		}
	}
	if allCSV {
		score += 10
		reasons = append(reasons, "+10: all files are CSV/zip")
	}

	bad := []string{}
	for e := range exts {
		if nonTabularExtensions[e] {
			bad = append(bad, e)
		}
	}
	if len(bad) > 0 {
		score -= 50
		reasons = append(reasons, stringf("-50: non-tabular files (%s)", strings.Join(bad, ", ")))
	}

	sizeGB := float64(totalBytes) / (1024 * 1024 * 1024)
	if sizeGB > 5 {
		score -= 15
		reasons = append(reasons, stringf("-15: large dataset (%.1fGB)", sizeGB))
	}
	return score, reasons
}

// ComputeFinalScore clamps to [0, 100].
func ComputeFinalScore(s1, s2 int) int {
	v := s1 + s2
	if v < 0 {
		return 0
	}
	if v > 100 {
		return 100
	}
	return v
}
