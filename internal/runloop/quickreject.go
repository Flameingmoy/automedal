// Pre-train smoke guard.
//
// Runs `python <train.py>` for at most budgetSecs seconds with the
// AUTOMEDAL_QUICK_REJECT=1 environment variable set so the script can
// short-circuit (e.g., one tiny epoch / one HPO trial) and still print
// a final_val_loss line. Accept the run if:
//   - the process exited with code 0, AND
//   - we found a finite `final_val_loss=` line in stdout, AND
//   - the loss is not absurd (configurable upper bound).
//
// If the process is still running at the budget, kill it and accept
// (slow ≠ broken). Catches obvious blunders like lr=1e10, mismatched
// feature shapes, or import-time exceptions.
//
// Mirrors automedal/quick_reject.py.
package runloop

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"syscall"
	"time"

	"github.com/Flameingmoy/automedal/internal/paths"
)

var finalLossRE = regexp.MustCompile(`final_val_loss=([0-9.eE+-]+)`)

// QuickRejectOpts tunes the smoke guard.
type QuickRejectOpts struct {
	BudgetSecs        int     // max wall time, default 30
	MaxAcceptableLoss float64 // reject if val_loss exceeds this, default 100
	TrainPy           string  // override path to train.py (defaults to AUTOMEDAL_TRAIN_PY or <root>/agent/train.py)
}

// SmokeTrain runs one short-circuited train.py and reports whether the
// upcoming iteration should proceed. accepted=false means the iteration
// should be aborted before the full training run.
func SmokeTrain(ctx context.Context, opts QuickRejectOpts) (accepted bool, reason string) {
	budget := opts.BudgetSecs
	if budget <= 0 {
		budget = 30
	}
	maxLoss := opts.MaxAcceptableLoss
	if maxLoss <= 0 {
		maxLoss = 100
	}
	trainPy := opts.TrainPy
	if trainPy == "" {
		if v := os.Getenv("AUTOMEDAL_TRAIN_PY"); v != "" {
			trainPy = v
		} else {
			trainPy = paths.RepoRoot() + "/agent/train.py"
		}
	}
	if _, err := os.Stat(trainPy); err != nil {
		return true, fmt.Sprintf("train.py not found at %s; skipping quick-reject", trainPy)
	}

	cctx, cancel := context.WithTimeout(ctx, time.Duration(budget)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(cctx, "python", trainPy)
	cmd.Dir = paths.RepoRoot()
	env := os.Environ()
	env = append(env, "AUTOMEDAL_QUICK_REJECT=1")
	if os.Getenv("TRAIN_BUDGET_MINUTES") == "" {
		env = append(env, "TRAIN_BUDGET_MINUTES=1")
	}
	cmd.Env = env
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	if err := cmd.Start(); err != nil {
		return true, fmt.Sprintf("could not launch train.py: %v; skipping quick-reject", err)
	}

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	select {
	case err := <-done:
		stdout := buf.String()
		if err != nil {
			rc := -1
			if ee, ok := err.(*exec.ExitError); ok {
				rc = ee.ExitCode()
			}
			tail := tailLines(stdout, 12)
			return false, fmt.Sprintf("smoke train exited code=%d\n--- last lines ---\n%s", rc, tail)
		}
		m := finalLossRE.FindStringSubmatch(stdout)
		if m == nil {
			return false, "smoke train produced no `final_val_loss=` line"
		}
		loss, err := strconv.ParseFloat(m[1], 64)
		if err != nil {
			return false, fmt.Sprintf("smoke train produced unparseable val_loss=%q", m[1])
		}
		if math.IsNaN(loss) {
			return false, "smoke train val_loss is NaN"
		}
		if loss > maxLoss {
			return false, fmt.Sprintf("smoke train val_loss=%.4f > %g (likely diverged)", loss, maxLoss)
		}
		return true, fmt.Sprintf("smoke train ok (val_loss=%.4f, exit=0)", loss)

	case <-cctx.Done():
		// Budget exceeded. SIGTERM the process group; kill on follow-up.
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			<-done
		}
		return true, fmt.Sprintf("smoke train still running at %ds budget (slow ≠ broken)", budget)
	}
}

func tailLines(s string, n int) string {
	if s == "" {
		return "(no stdout)"
	}
	lines := bytes.Split([]byte(s), []byte("\n"))
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return string(bytes.Join(lines, []byte("\n")))
}
