package harness

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

var journalPattern = regexp.MustCompile(`^(\d{4})-[a-z0-9-]+\.md$`)

// NextExpID returns the next experiment ID, zero-padded to 4 digits.
// Uses journalDir/.last_exp_id as a fast-path sentinel; falls back to
// directory scan when the sentinel is missing or invalid.
//
// Idempotent writer: bumps the sentinel by 1 per call. Atomic via
// os.Rename over a tmp file.
func NextExpID(journalDir string) (string, error) {
	sentinel := filepath.Join(journalDir, ".last_exp_id")

	var highest int
	if b, err := os.ReadFile(sentinel); err == nil {
		if n, perr := strconv.Atoi(strings.TrimSpace(string(b))); perr == nil {
			highest = n
		} else {
			highest = scanJournal(journalDir)
		}
	} else {
		if info, err := os.Stat(journalDir); err == nil && info.IsDir() {
			highest = scanJournal(journalDir)
		} else {
			if err := os.MkdirAll(journalDir, 0o755); err != nil {
				return "", err
			}
		}
	}

	next := highest + 1
	tmp := sentinel + ".tmp"
	if err := os.WriteFile(tmp, []byte(strconv.Itoa(next)), 0o644); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, sentinel); err != nil {
		return "", err
	}
	return fmt.Sprintf("%04d", next), nil
}

func scanJournal(dir string) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	highest := 0
	for _, e := range entries {
		m := journalPattern.FindStringSubmatch(e.Name())
		if m == nil {
			continue
		}
		n, _ := strconv.Atoi(m[1])
		if n > highest {
			highest = n
		}
	}
	return highest
}
