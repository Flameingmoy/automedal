// Bootstrap orchestrates: download → sniff → render → write config →
// reset results.tsv → init memory. Mirrors scout/bootstrap.py.
package scout

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Flameingmoy/automedal/internal/harness"
	"github.com/Flameingmoy/automedal/internal/paths"
)

// objectiveMap encodes per-task XGBoost/LightGBM/CatBoost objectives.
// Mirrors OBJECTIVE_MAP in scout/bootstrap.py.
var objectiveMap = map[string]map[string]string{
	"multiclass": {
		"xgboost":      "multi:softprob",
		"xgboost_eval": "mlogloss",
		"lightgbm":     "multiclass",
		"catboost":     "MultiClass",
	},
	"binary": {
		"xgboost":      "binary:logistic",
		"xgboost_eval": "logloss",
		"lightgbm":     "binary",
		"catboost":     "Logloss",
	},
	"regression": {
		"xgboost":      "reg:squarederror",
		"xgboost_eval": "rmse",
		"lightgbm":     "regression",
		"catboost":     "RMSE",
	},
}

// metricMap encodes (kaggle, proxy) per task type.
var metricMap = map[string][2]string{
	"multiclass": {"accuracy", "log_loss"},
	"binary":     {"auc", "log_loss"},
	"regression": {"rmse", "rmse"},
}

// CompetitionMetadata is the small set of fields bootstrap reads from
// the Kaggle API to enrich the config.
type CompetitionMetadata struct {
	Title             string
	Deadline          string
	EvalMetricKaggle  string
}

// FetchCompetitionMetadata calls /competitions/list?search=<slug> and
// returns the matching entry's metadata. Falls back to {Title: slug,
// Deadline: "unknown"} on any failure.
func FetchCompetitionMetadata(ctx context.Context, c *Client, slug string) CompetitionMetadata {
	def := CompetitionMetadata{Title: slug, Deadline: "unknown"}
	if c == nil {
		return def
	}
	var batch []rawCompetition
	if err := c.get(ctx, "/competitions/list?search="+slug, &batch); err != nil {
		return def
	}
	for _, r := range batch {
		if r.Ref == slug {
			deadline := r.Deadline
			if t, err := time.Parse(time.RFC3339, strings.ReplaceAll(deadline, "Z", "+00:00")); err == nil {
				deadline = t.Format("2006-01-02")
			}
			if deadline == "" {
				deadline = "unknown"
			}
			title := r.Title
			if title == "" {
				title = slug
			}
			return CompetitionMetadata{
				Title:            title,
				Deadline:         deadline,
				EvalMetricKaggle: r.EvaluationMetric,
			}
		}
	}
	return def
}

// DownloadCompetitionData fetches competition files via the Kaggle
// `data/download/<slug>` endpoint. The endpoint returns a single
// .zip blob; we extract it into <root>/data/.
func DownloadCompetitionData(ctx context.Context, c *Client, slug string, root string) ([]string, error) {
	dataDir := filepath.Join(root, "data")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil, err
	}

	url := c.BaseURL + "/competitions/data/download-all/" + slug
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.SetBasicAuth(c.Creds.Username, c.Creds.Key)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("kaggle download: HTTP %d: %s", resp.StatusCode, string(body))
	}

	zipPath := filepath.Join(dataDir, slug+".zip")
	zf, err := os.Create(zipPath)
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(zf, resp.Body); err != nil {
		zf.Close()
		return nil, err
	}
	zf.Close()

	names, err := unzipAll(zipPath, dataDir)
	if err != nil {
		return nil, err
	}
	_ = os.Remove(zipPath)
	return names, nil
}

func unzipAll(src, dest string) ([]string, error) {
	r, err := zip.OpenReader(src)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	var names []string
	for _, f := range r.File {
		dst := filepath.Join(dest, f.Name)
		if !strings.HasPrefix(filepath.Clean(dst)+string(os.PathSeparator), filepath.Clean(dest)+string(os.PathSeparator)) {
			return names, fmt.Errorf("zip entry escapes dest: %s", f.Name)
		}
		if f.FileInfo().IsDir() {
			os.MkdirAll(dst, 0o755)
			continue
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return names, err
		}
		out, err := os.Create(dst)
		if err != nil {
			return names, err
		}
		in, err := f.Open()
		if err != nil {
			out.Close()
			return names, err
		}
		_, err = io.Copy(out, in)
		in.Close()
		out.Close()
		if err != nil {
			return names, err
		}
		names = append(names, f.Name)
	}
	return names, nil
}

// BuildConfigSlots produces the slot dict expected by the AGENTS.md +
// program.md templates from the sniffed schema + Kaggle metadata.
// The shape mirrors scout/bootstrap.py:build_config.
func BuildConfigSlots(slug string, schema *Schema, meta CompetitionMetadata) map[string]any {
	taskType := schema.TaskType
	if taskType == "" {
		taskType = "multiclass"
	}
	pair := metricMap[taskType]
	if pair == [2]string{} {
		pair = metricMap["multiclass"]
	}
	evalKaggle := meta.EvalMetricKaggle
	proxy := pair[1]
	if evalKaggle == "" {
		evalKaggle = pair[0]
	}
	objectives := objectiveMap[taskType]
	if objectives == nil {
		objectives = objectiveMap["multiclass"]
	}
	title := meta.Title
	subtitle := title
	for _, sep := range []string{" - ", " — "} {
		if i := strings.Index(title, sep); i > 0 {
			subtitle = strings.TrimSpace(title[i+len(sep):])
			title = strings.TrimSpace(title[:i])
			break
		}
	}
	deadline := meta.Deadline
	if deadline == "" {
		deadline = "unknown"
	}
	return map[string]any{
		"competition": map[string]any{
			"slug":     slug,
			"title":    title,
			"subtitle": subtitle,
			"url":      "https://www.kaggle.com/competitions/" + slug,
			"deadline": deadline,
		},
		"task": map[string]any{
			"type":               taskType,
			"target_col":         schema.TargetCol,
			"id_col":             schema.IDCol,
			"class_names":        schema.ClassNames,
			"num_classes":        schema.NumClasses,
			"eval_metric_kaggle": evalKaggle,
			"eval_metric_proxy":  proxy,
		},
		"dataset": map[string]any{
			"train_rows":           schema.TrainRows,
			"test_rows":            schema.TestRows,
			"numeric_features":     schema.NumericFeatures,
			"categorical_features": schema.CategoricalFeatures,
		},
		"submission": map[string]any{
			"id_col":     schema.Submission.IDCol,
			"target_col": schema.Submission.TargetCol,
			"format":     schema.Submission.Format,
		},
		"objectives": objectives,
		"meta": map[string]any{
			"bootstrapped_at":  time.Now().UTC().Format(time.RFC3339),
			"sniff_confidence": schema.Confidence,
			"human_verified":   false,
		},
	}
}

// WriteConfigYAML writes the YAML representation of `slots` to
// configs/competition.yaml under root.
func WriteConfigYAML(root string, slots map[string]any) (string, error) {
	full := filepath.Join(root, "configs", "competition.yaml")
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return "", err
	}
	yaml, err := encodeYAML(slots)
	if err != nil {
		return "", err
	}
	header := "# configs/competition.yaml — Single source of truth for active competition\n"
	if meta, ok := slots["meta"].(map[string]any); ok {
		if ts, ok := meta["bootstrapped_at"].(string); ok {
			header += fmt.Sprintf("# Bootstrapped: %s\n\n", ts)
		}
	}
	if err := os.WriteFile(full, []byte(header+yaml), 0o644); err != nil {
		return "", err
	}
	return full, nil
}

// ResetResultsTSV truncates agent/results.tsv to header only.
func ResetResultsTSV(root string) (string, error) {
	full := filepath.Join(root, "agent", "results.tsv")
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return "", err
	}
	return full, os.WriteFile(full,
		[]byte("timestamp\tmethod\ttrials\tval_loss\tval_accuracy\tsubmission\tnotes\n"),
		0o644)
}

// BootstrapResult bundles the artifacts created during one bootstrap.
type BootstrapResult struct {
	Slug       string
	Schema     *Schema
	ConfigPath string
	Rendered   []string
	PreparePath string
	PrepareWritten bool
	Memory     map[string]string
	DataFiles  []string
}

// BootstrapOptions controls the bootstrap flow.
type BootstrapOptions struct {
	SkipDownload     bool
	AbortOnLowConfidence bool
}

// Bootstrap runs the full pipeline. Returns the result + a non-nil
// error on any hard failure. Low-confidence sniff results are surfaced
// in Result.Schema.Warnings and (with AbortOnLowConfidence) cause an
// error.
func Bootstrap(ctx context.Context, c *Client, slug string, opt BootstrapOptions) (*BootstrapResult, error) {
	root := paths.RepoRoot()
	res := &BootstrapResult{Slug: slug}

	if !opt.SkipDownload {
		files, err := DownloadCompetitionData(ctx, c, slug, root)
		if err != nil {
			return res, fmt.Errorf("download: %w", err)
		}
		res.DataFiles = files
	}

	schema, err := Sniff(ctx, filepath.Join(root, "data"))
	if err != nil {
		return res, fmt.Errorf("sniff: %w", err)
	}
	if schema.Error != "" {
		return res, fmt.Errorf("sniff: %s", schema.Error)
	}
	res.Schema = schema
	if schema.Confidence < 0.7 && opt.AbortOnLowConfidence {
		return res, fmt.Errorf("sniff confidence %.0f%% — aborting (warnings: %v)",
			schema.Confidence*100, schema.Warnings)
	}

	meta := FetchCompetitionMetadata(ctx, c, slug)
	slots := BuildConfigSlots(slug, schema, meta)
	cfgPath, err := WriteConfigYAML(root, slots)
	if err != nil {
		return res, fmt.Errorf("write config: %w", err)
	}
	res.ConfigPath = cfgPath

	rendered, err := RenderTemplates(root, slots)
	if err != nil {
		return res, fmt.Errorf("render templates: %w", err)
	}
	res.Rendered = rendered

	prepPath, written, err := RenderPrepareStarter(root, slots)
	if err != nil {
		return res, fmt.Errorf("render prepare.py: %w", err)
	}
	res.PreparePath = prepPath
	res.PrepareWritten = written

	if _, err := ResetResultsTSV(root); err != nil {
		return res, fmt.Errorf("reset results.tsv: %w", err)
	}

	mem, err := harness.InitMemory(root, true)
	if err != nil {
		return res, fmt.Errorf("init memory: %w", err)
	}
	res.Memory = mem
	return res, nil
}
