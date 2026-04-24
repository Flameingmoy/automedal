package events

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/fsnotify/fsnotify"
)

// TailOpts controls how Tail behaves. Zero value is fine — defaults below.
type TailOpts struct {
	// FromStart replays the whole file on attach (tests). Default = EOF.
	FromStart bool
	// Buffer is the channel capacity; overflow drops oldest. Default 256.
	Buffer int
}

// Tail starts a goroutine that watches `path` and sends parsed events on
// the returned channel. Stop by cancelling ctx.
//
// Log rotation / truncation is handled: if the file shrinks or is
// replaced, we re-open and re-seek. Partial trailing lines wait for the
// next write.
func Tail(ctx context.Context, path string, opts TailOpts) (<-chan Event, error) {
	if opts.Buffer <= 0 {
		opts.Buffer = 256
	}
	ch := make(chan Event, opts.Buffer)

	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	// Watch the parent dir so CREATE events land after a rotation.
	if err := w.Add(filepath.Dir(path)); err != nil {
		_ = w.Close()
		return nil, err
	}

	go tailLoop(ctx, path, w, ch, opts)
	return ch, nil
}

func tailLoop(ctx context.Context, path string, w *fsnotify.Watcher, ch chan<- Event, opts TailOpts) {
	defer close(ch)
	defer w.Close()

	var f *os.File
	var rdr *bufio.Reader
	var pending []byte
	var err error

	open := func() {
		if f != nil {
			_ = f.Close()
			f = nil
		}
		f, err = os.Open(path)
		if err != nil {
			return
		}
		if !opts.FromStart {
			_, _ = f.Seek(0, io.SeekEnd)
		}
		rdr = bufio.NewReaderSize(f, 64*1024)
		pending = pending[:0]
	}

	readLines := func() {
		if rdr == nil {
			return
		}
		for {
			line, err := rdr.ReadBytes('\n')
			if len(line) > 0 {
				pending = append(pending, line...)
			}
			if err != nil {
				if !errors.Is(err, io.EOF) {
					return
				}
				break
			}
			if n := len(pending); n > 0 && pending[n-1] == '\n' {
				emit(pending[:n-1], ch)
				pending = pending[:0]
			}
		}
	}

	// Detect truncation / rotation by comparing current size to our offset.
	checkRotate := func() {
		if f == nil {
			open()
			return
		}
		st, err := os.Stat(path)
		if err != nil {
			// File gone (rename-then-create). Wait for CREATE event.
			return
		}
		cur, _ := f.Seek(0, io.SeekCurrent)
		if st.Size() < cur {
			// Truncated — re-open from start so we don't miss the new tail.
			_, _ = f.Seek(0, io.SeekStart)
			pending = pending[:0]
			rdr = bufio.NewReaderSize(f, 64*1024)
		}
		// Inode swap (rotation): compare by ino when we can.
		if fi, err := f.Stat(); err == nil {
			if !sameFile(fi, st) {
				open()
			}
		}
	}

	open()

	// First drain (nothing pending if we seeked to EOF, but the From-start
	// case relies on this).
	readLines()

	// Fallback poller — some filesystems (NFS, overlay) drop inotify
	// events. 200 ms is cheap and still feels live.
	poll := time.NewTicker(200 * time.Millisecond)
	defer poll.Stop()

	for {
		select {
		case <-ctx.Done():
			if f != nil {
				_ = f.Close()
			}
			return
		case ev, ok := <-w.Events:
			if !ok {
				return
			}
			if filepath.Clean(ev.Name) != filepath.Clean(path) {
				continue
			}
			if ev.Op&(fsnotify.Create|fsnotify.Rename|fsnotify.Remove) != 0 {
				open()
			}
			if ev.Op&fsnotify.Write != 0 {
				checkRotate()
				readLines()
			}
		case err, ok := <-w.Errors:
			if !ok {
				return
			}
			_ = err // swallow; poller will recover
		case <-poll.C:
			checkRotate()
			readLines()
		}
	}
}

// emit parses a single JSON line and sends it on the channel. On overflow
// we drop the new message — the UI is cosmetic; correctness lives on
// disk. (Drop-newest is simpler than drop-oldest in Go because a
// send-only channel can't be receive-drained from the producer side.)
func emit(line []byte, ch chan<- Event) {
	var ev Event
	if err := json.Unmarshal(line, &ev); err != nil {
		return
	}
	select {
	case ch <- ev:
	default:
	}
}
