// Competition picker — loads the ranked shortlist from
// scout/outputs/competition_candidates.json. Mirrors scout/select.py.
package scout

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/Flameingmoy/automedal/internal/paths"
)

// CandidatesJSONPath is the canonical location of the discover output.
func CandidatesJSONPath() string {
	return filepath.Join(paths.RepoRoot(), "scout", "outputs", "competition_candidates.json")
}

// Candidate is one row of the ranked shortlist.
type Candidate struct {
	Competition    Competition `json:"competition"`
	Stage1Score    int         `json:"stage1_score"`
	Stage1Reasons  []string    `json:"stage1_reasons"`
	Stage2Score    int         `json:"stage2_score"`
	Stage2Reasons  []string    `json:"stage2_reasons"`
	FinalScore     int         `json:"final_score"`
}

// CandidateBundle is the wrapper saved by discover.
type CandidateBundle struct {
	GeneratedAt       string      `json:"generated_at"`
	TotalCandidates   int         `json:"total_candidates"`
	MinScoreThreshold int         `json:"min_score_threshold"`
	Candidates        []Candidate `json:"candidates"`
}

// LoadCandidates reads the JSON shortlist. Returns nil + nil error when
// the file does not exist (call sites print their own help message).
func LoadCandidates(path string) (*CandidateBundle, error) {
	if path == "" {
		path = CandidatesJSONPath()
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var bundle CandidateBundle
	if err := json.Unmarshal(b, &bundle); err != nil {
		return nil, fmt.Errorf("decode candidates: %w", err)
	}
	return &bundle, nil
}

// FindBySlug returns the first candidate whose Competition.Ref matches
// `slug`, or nil if not found.
func (b *CandidateBundle) FindBySlug(slug string) *Candidate {
	if b == nil {
		return nil
	}
	for i := range b.Candidates {
		if b.Candidates[i].Competition.Ref == slug {
			return &b.Candidates[i]
		}
	}
	return nil
}
