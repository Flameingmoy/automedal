// Package proc spawns an `automedal` subprocess and bridges its stdout
// into the Bubbletea message loop as one tea.Msg per line.
package proc

import (
	"bufio"
	"context"
	"io"
	"os"
	"os/exec"
	"regexp"
	"syscall"
	"time"
)

// Line is one decoded line from the child.
type Line struct {
	Text string
	IsErr bool // from stderr channel
}

// ExitMsg closes out the run — sent once the child has exited.
type ExitMsg struct {
	ExitCode int
	Err      error
}

// Handle is an opaque handle that the RunModel holds onto. Cancel() is
// idempotent and safe to call from Bubbletea's Update.
type Handle struct {
	cancel context.CancelFunc
	cmd    *exec.Cmd
	lines  <-chan Line
	exit   <-chan ExitMsg
}

func (h *Handle) Lines() <-chan Line  { return h.lines }
func (h *Handle) Exit() <-chan ExitMsg { return h.exit }

// Cancel asks the child to exit. We try SIGTERM first; if it's still alive
// after 2 s the context deadline kills it.
func (h *Handle) Cancel() {
	if h == nil || h.cmd == nil || h.cmd.Process == nil {
		return
	}
	_ = h.cmd.Process.Signal(syscall.SIGTERM)
	go func() {
		select {
		case <-time.After(2 * time.Second):
			h.cancel()
		}
	}()
}

// ansiRE strips the VT escape sequences and OSC strings we sometimes see.
var ansiRE = regexp.MustCompile("\x1b\\[[0-?]*[ -/]*[@-~]|\x1b\\][^\x07]*\x07")

// Spawn forks `automedal <cmd> <args...>` with PYTHONUNBUFFERED=1 so we
// get line-by-line stdout. Returns a Handle; the caller reads from
// Handle.Lines() until Handle.Exit() fires.
func Spawn(parent context.Context, verb string, args []string) (*Handle, error) {
	ctx, cancel := context.WithCancel(parent)

	argv := append([]string{verb}, args...)
	cmd := exec.CommandContext(ctx, "automedal", argv...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1", "NO_COLOR=1")

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, err
	}

	lines := make(chan Line, 256)
	exitCh := make(chan ExitMsg, 1)

	go pump(stdout, lines, false)
	go pump(stderr, lines, true)
	go func() {
		err := cmd.Wait()
		code := 0
		if ee, ok := err.(*exec.ExitError); ok {
			code = ee.ExitCode()
		} else if err != nil {
			code = -1
		}
		// Give the pumpers a beat to drain whatever's left.
		time.Sleep(50 * time.Millisecond)
		close(lines)
		exitCh <- ExitMsg{ExitCode: code, Err: err}
		close(exitCh)
	}()

	return &Handle{cancel: cancel, cmd: cmd, lines: lines, exit: exitCh}, nil
}

func pump(r io.Reader, ch chan<- Line, isErr bool) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := ansiRE.ReplaceAllString(sc.Text(), "")
		ch <- Line{Text: line, IsErr: isErr}
	}
}
