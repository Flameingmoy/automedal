// Thin Go wrapper over `python -m sniff <data_dir>`. The shim lives in
// py-shim/sniff/ and emits one JSON blob to stdout matching the Schema
// shape below. Mirrors the dict scout/sniff.py:sniff_schema returns.
package scout

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// SniffSubmission mirrors the `submission` sub-dict.
type SniffSubmission struct {
	IDCol     string `json:"id_col"`
	TargetCol string `json:"target_col"`
	Format    string `json:"format"`
}

// Schema is the structured result of one sniff run. `Error` is set when
// the shim produced an error envelope; callers must check it first.
type Schema struct {
	Error               string          `json:"error,omitempty"`
	TargetCol           string          `json:"target_col"`
	IDCol               string          `json:"id_col"`
	NumericFeatures     []string        `json:"numeric_features"`
	CategoricalFeatures []string        `json:"categorical_features"`
	TaskType            string          `json:"task_type"`
	NumClasses          *int            `json:"num_classes"`
	ClassNames          []any           `json:"class_names"`
	Submission          SniffSubmission `json:"submission"`
	TrainRows           int             `json:"train_rows"`
	TestRows            int             `json:"test_rows"`
	Confidence          float64         `json:"confidence"`
	Warnings            []string        `json:"warnings"`
}

// Sniff runs `python -m sniff <data_dir>` and decodes the JSON it writes
// to stdout. Times out at 60s. Returns a Schema (Error field set on shim
// failure envelopes) or a Go error on transport failure.
func Sniff(ctx context.Context, dataDir string) (*Schema, error) {
	cctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	pyBin := os.Getenv("AUTOMEDAL_PYTHON")
	if pyBin == "" {
		pyBin = "python"
	}
	cmd := exec.CommandContext(cctx, pyBin, "-m", "sniff", dataDir)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("sniff: %w; stderr: %s", err, strings.TrimSpace(errBuf.String()))
	}

	var s Schema
	if err := json.Unmarshal(out.Bytes(), &s); err != nil {
		return nil, fmt.Errorf("sniff: bad JSON: %w; stdout: %s", err, strings.TrimSpace(out.String()))
	}
	return &s, nil
}
