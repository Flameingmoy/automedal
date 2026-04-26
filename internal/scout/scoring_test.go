package scout

import (
	"strings"
	"testing"
	"time"
)

func TestScoreStage1_PlaygroundSeries(t *testing.T) {
	c := Competition{
		Ref:              "playground-series-s6e4",
		Title:            "PSx",
		Category:         "Playground",
		EvaluationMetric: "accuracy",
		Tags:             []string{"tabular"},
		TeamCount:        1500,
		Deadline:         time.Now().AddDate(0, 0, 60).UTC().Format(time.RFC3339),
	}
	score, reasons, dq := ScoreStage1(c)
	if dq {
		t.Fatal("should not be disqualified")
	}
	if score < 80 {
		t.Errorf("expected high score; got %d (%v)", score, reasons)
	}
}

func TestScoreStage1_KernelsOnlyDQ(t *testing.T) {
	c := Competition{Ref: "x", IsKernelsSubmissionsOnly: true}
	_, _, dq := ScoreStage1(c)
	if !dq {
		t.Error("kernels-only should disqualify")
	}
}

func TestScoreStage1_NonTabularTags(t *testing.T) {
	c := Competition{
		Ref:  "vision-x",
		Tags: []string{"computer vision"},
	}
	score, reasons, _ := ScoreStage1(c)
	if score >= 0 {
		t.Errorf("non-tabular tags should drag score below zero; got %d (%v)", score, reasons)
	}
	matched := false
	for _, r := range reasons {
		if strings.Contains(r, "non-tabular tags") {
			matched = true
		}
	}
	if !matched {
		t.Errorf("expected non-tabular tags reason; got %v", reasons)
	}
}

func TestScoreStage2_HappyPath(t *testing.T) {
	files := []FileInfo{
		{Name: "train.csv", TotalBytes: 100_000_000},
		{Name: "test.csv", TotalBytes: 30_000_000},
		{Name: "sample_submission.csv", TotalBytes: 20_000},
	}
	score, reasons := ScoreStage2(files)
	if score != 60 { // 25 + 15 + 10 + 10
		t.Errorf("expected 60; got %d (%v)", score, reasons)
	}
}

func TestScoreStage2_NonTabularPenalty(t *testing.T) {
	files := []FileInfo{
		{Name: "train.jpg"},
		{Name: "labels.csv"},
	}
	score, reasons := ScoreStage2(files)
	hit := false
	for _, r := range reasons {
		if strings.Contains(r, "non-tabular files") {
			hit = true
		}
	}
	if !hit || score >= 0 {
		t.Errorf("expected non-tabular penalty; got %d (%v)", score, reasons)
	}
}

func TestComputeFinalScore_Clamps(t *testing.T) {
	if v := ComputeFinalScore(80, 30); v != 100 {
		t.Errorf("expected 100; got %d", v)
	}
	if v := ComputeFinalScore(-50, -20); v != 0 {
		t.Errorf("expected 0; got %d", v)
	}
}
