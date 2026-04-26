// Competition discovery via the Kaggle public API. No SDK — just
// net/http + basic auth (KAGGLE_USERNAME:KAGGLE_KEY or ~/.kaggle/kaggle.json).
//
// Endpoints used:
//
//	GET  /api/v1/competitions/list?page=N
//	GET  /api/v1/competitions/data/list/{slug}
//
// Mirrors scout/discover.py at the dict shape level so the downstream
// scoring + select modules consume the same structure.
package scout

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const (
	kaggleAPIBase = "https://www.kaggle.com/api/v1"
	stage2TopN    = 20
	minFinalScore = 30
	maxPages      = 20
)

// KaggleCreds holds basic-auth values resolved from env or kaggle.json.
type KaggleCreds struct {
	Username, Key string
}

// LoadKaggleCreds resolves Kaggle credentials from env vars first, then
// ~/.kaggle/kaggle.json. Returns ErrNoKaggleAuth if neither is set.
func LoadKaggleCreds() (*KaggleCreds, error) {
	if u, k := os.Getenv("KAGGLE_USERNAME"), os.Getenv("KAGGLE_KEY"); u != "" && k != "" {
		return &KaggleCreds{Username: u, Key: k}, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, ErrNoKaggleAuth
	}
	for _, p := range []string{
		filepath.Join(home, ".kaggle", "kaggle.json"),
		filepath.Join(os.Getenv("USERPROFILE"), ".kaggle", "kaggle.json"),
	} {
		if p == "" {
			continue
		}
		b, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var raw struct {
			Username, Key string
		}
		if err := json.Unmarshal(b, &raw); err == nil && raw.Username != "" && raw.Key != "" {
			return &KaggleCreds{Username: raw.Username, Key: raw.Key}, nil
		}
	}
	return nil, ErrNoKaggleAuth
}

// ErrNoKaggleAuth is returned by LoadKaggleCreds when no creds are available.
var ErrNoKaggleAuth = fmt.Errorf("kaggle: no credentials found (set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json)")

// Client is a minimal Kaggle API client.
type Client struct {
	HTTP    *http.Client
	Creds   *KaggleCreds
	BaseURL string
}

// NewClient constructs a default client (60s HTTP timeout).
func NewClient(creds *KaggleCreds) *Client {
	return &Client{
		HTTP:    &http.Client{Timeout: 60 * time.Second},
		Creds:   creds,
		BaseURL: kaggleAPIBase,
	}
}

func (c *Client) get(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+path, nil)
	if err != nil {
		return err
	}
	req.SetBasicAuth(c.Creds.Username, c.Creds.Key)
	req.Header.Set("Accept", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("kaggle %s: HTTP %d: %s", path, resp.StatusCode, string(b))
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	return json.Unmarshal(body, out)
}

// rawCompetition is what /competitions/list returns. Matches the SDK
// camelCase shape but we tolerate snake_case from older endpoints.
type rawCompetition struct {
	Ref                      string `json:"ref"`
	Title                    string `json:"title"`
	Description              string `json:"description"`
	URL                      string `json:"url"`
	Category                 string `json:"category"`
	Reward                   any    `json:"reward,omitempty"`
	Deadline                 string `json:"deadline,omitempty"`
	TeamCount                int    `json:"teamCount"`
	IsKernelsSubmissionsOnly bool   `json:"isKernelsSubmissionsOnly"`
	EvaluationMetric         string `json:"evaluationMetric"`
	MaxDailySubmissions      any    `json:"maxDailySubmissions,omitempty"`
	MaxTeamSize              any    `json:"maxTeamSize,omitempty"`
	EnabledDate              string `json:"enabledDate,omitempty"`
	Tags                     []any  `json:"tags"`
}

func (r rawCompetition) toCompetition() Competition {
	c := Competition{
		Ref:                      r.Ref,
		Title:                    r.Title,
		Description:              r.Description,
		URL:                      r.URL,
		Category:                 r.Category,
		Reward:                   r.Reward,
		Deadline:                 r.Deadline,
		TeamCount:                r.TeamCount,
		IsKernelsSubmissionsOnly: r.IsKernelsSubmissionsOnly,
		EvaluationMetric:         r.EvaluationMetric,
		MaxDailySubmissions:      r.MaxDailySubmissions,
		MaxTeamSize:              r.MaxTeamSize,
		EnabledDate:              r.EnabledDate,
	}
	if c.URL == "" && c.Ref != "" {
		c.URL = "https://www.kaggle.com/competitions/" + c.Ref
	}
	for _, t := range r.Tags {
		switch v := t.(type) {
		case string:
			if v != "" {
				c.Tags = append(c.Tags, v)
			}
		case map[string]any:
			if name, ok := v["name"].(string); ok && name != "" {
				c.Tags = append(c.Tags, name)
			} else if ref, ok := v["ref"].(string); ok && ref != "" {
				c.Tags = append(c.Tags, ref)
			}
		}
	}
	return c
}

// ListCompetitions paginates through /competitions/list until the API
// returns an empty page or maxPages is hit.
func (c *Client) ListCompetitions(ctx context.Context) ([]Competition, error) {
	var out []Competition
	for page := 1; page <= maxPages; page++ {
		var batch []rawCompetition
		if err := c.get(ctx, fmt.Sprintf("/competitions/list?page=%d", page), &batch); err != nil {
			return out, err
		}
		if len(batch) == 0 {
			break
		}
		for _, r := range batch {
			out = append(out, r.toCompetition())
		}
	}
	return out, nil
}

// rawFile mirrors /competitions/data/list response entries.
type rawFile struct {
	Name           string `json:"name"`
	TotalBytes     int64  `json:"totalBytes"`
	TotalBytesAlt  int64  `json:"total_bytes,omitempty"`
}

// ListCompetitionFiles fetches the file listing for one competition.
func (c *Client) ListCompetitionFiles(ctx context.Context, slug string) ([]FileInfo, error) {
	var raw []rawFile
	if err := c.get(ctx, "/competitions/data/list/"+slug, &raw); err != nil {
		return nil, err
	}
	out := make([]FileInfo, 0, len(raw))
	for _, f := range raw {
		size := f.TotalBytes
		if size == 0 {
			size = f.TotalBytesAlt
		}
		out = append(out, FileInfo{Name: f.Name, TotalBytes: size})
	}
	return out, nil
}

// DiscoverOptions tunes Discover.
type DiscoverOptions struct {
	Stage2TopN    int
	MinFinalScore int
}

// Discover fetches all competitions, applies the two-stage scoring, and
// returns the ranked shortlist.
func Discover(ctx context.Context, c *Client, opt DiscoverOptions) ([]Candidate, error) {
	if opt.Stage2TopN <= 0 {
		opt.Stage2TopN = stage2TopN
	}
	if opt.MinFinalScore == 0 {
		opt.MinFinalScore = minFinalScore
	}
	all, err := c.ListCompetitions(ctx)
	if err != nil {
		return nil, err
	}
	candidates := make([]Candidate, 0, len(all))
	for _, comp := range all {
		s, reasons, dq := ScoreStage1(comp)
		if dq {
			continue
		}
		candidates = append(candidates, Candidate{
			Competition:   comp,
			Stage1Score:   s,
			Stage1Reasons: reasons,
			FinalScore:    s,
		})
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Stage1Score > candidates[j].Stage1Score
	})

	top := candidates
	if len(top) > opt.Stage2TopN {
		top = top[:opt.Stage2TopN]
	}
	for i := range top {
		files, err := c.ListCompetitionFiles(ctx, top[i].Competition.Ref)
		if err != nil {
			top[i].Stage2Reasons = []string{stringf("Error checking files: %s", err.Error())}
			continue
		}
		top[i].Competition.Files = files
		s2, r2 := ScoreStage2(files)
		top[i].Stage2Score = s2
		top[i].Stage2Reasons = r2
		top[i].FinalScore = ComputeFinalScore(top[i].Stage1Score, s2)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].FinalScore > candidates[j].FinalScore
	})
	out := make([]Candidate, 0, len(candidates))
	for _, c := range candidates {
		if c.FinalScore >= opt.MinFinalScore {
			out = append(out, c)
		}
	}
	return out, nil
}

// WriteJSONOutput writes the ranked shortlist as a JSON bundle.
func WriteJSONOutput(path string, candidates []Candidate, minScore int) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	bundle := CandidateBundle{
		GeneratedAt:       time.Now().UTC().Format(time.RFC3339),
		TotalCandidates:   len(candidates),
		MinScoreThreshold: minScore,
		Candidates:        candidates,
	}
	b, err := json.MarshalIndent(bundle, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
