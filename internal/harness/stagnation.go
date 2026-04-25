package harness

import (
	"encoding/csv"
	"math"
	"os"
	"strconv"
	"strings"
)

// ReadValLosses parses `results.tsv` and returns val_loss floats in row
// order. Missing file → empty slice. Missing/invalid val_loss column in
// a row → that row is silently skipped.
func ReadValLosses(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = '\t'
	r.FieldsPerRecord = -1

	header, err := r.Read()
	if err != nil {
		return nil, nil // empty or unreadable → treat as no data
	}
	col := -1
	for i, h := range header {
		if strings.TrimSpace(h) == "val_loss" {
			col = i
			break
		}
	}
	if col < 0 {
		return nil, nil
	}

	var losses []float64
	for {
		row, err := r.Read()
		if err != nil {
			break
		}
		if col >= len(row) {
			continue
		}
		raw := strings.TrimSpace(row[col])
		if raw == "" {
			continue
		}
		v, perr := strconv.ParseFloat(raw, 64)
		if perr != nil {
			continue
		}
		losses = append(losses, v)
	}
	return losses, nil
}

// IsStagnating reports whether the best val_loss in the last k rows is
// not strictly better than the minimum seen before the window. Needs at
// least k+1 data points to fire — ties don't count as improvement.
func IsStagnating(k int, losses []float64) bool {
	if len(losses) < k+1 {
		return false
	}
	bestBefore := minSlice(losses[:len(losses)-k])
	bestInWindow := minSlice(losses[len(losses)-k:])
	return bestInWindow >= bestBefore
}

// BestValLoss returns the smallest val_loss seen, or +Inf if none.
func BestValLoss(losses []float64) float64 {
	if len(losses) == 0 {
		return math.Inf(1)
	}
	return minSlice(losses)
}

func minSlice(xs []float64) float64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}
