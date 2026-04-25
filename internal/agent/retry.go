package agent

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"time"
)

const (
	defaultRetryAttempts = 3
)

var defaultRetryDelays = []time.Duration{5 * time.Second, 15 * time.Second, 30 * time.Second}

// transient patterns matched against the lowercased exception text — same
// list as automedal/agent/retry.py:_TRANSIENT_PATTERNS.
var transientPatterns = []string{
	"timeout", "timed out",
	"429", "rate limit", "rate_limit",
	"503", "service unavailable",
	"502", "bad gateway",
	"500", "internal server error",
	"504", "gateway timeout",
	"overloaded", "capacity",
	"connection reset", "connection refused", "connection error",
	"eof", "broken pipe",
	"remote end closed",
}

// IsTransientError reports whether err matches a known transient
// network/provider pattern that's worth retrying.
func IsTransientError(err error) bool {
	if err == nil {
		return false
	}
	// Hard-typed transients first.
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return true
	}
	s := strings.ToLower(err.Error())
	for _, p := range transientPatterns {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}

// RetryOpts customises the backoff policy.
type RetryOpts struct {
	Attempts int
	Delays   []time.Duration
	Sink     *EventSink // optional, for retry-attempt notices
	Label    string     // identifies the call site in notices
}

// WithRetry runs `fn` up to opts.Attempts times, sleeping the corresponding
// opts.Delays[i-1] (clamped to last) between attempts. Non-transient errors
// bubble immediately. Honors ctx cancellation.
//
// Mirrors automedal/agent/retry.py:with_retry. Identical posture: 3 ×
// [5,15,30]s by default.
func WithRetry[T any](ctx context.Context, fn func() (T, error), opts RetryOpts) (T, error) {
	if opts.Attempts <= 0 {
		opts.Attempts = defaultRetryAttempts
	}
	if len(opts.Delays) == 0 {
		opts.Delays = defaultRetryDelays
	}
	var zero T
	var lastErr error
	for attempt := 1; attempt <= opts.Attempts; attempt++ {
		out, err := fn()
		if err == nil {
			return out, nil
		}
		lastErr = err
		if attempt >= opts.Attempts || !IsTransientError(err) {
			return zero, err
		}
		idx := attempt - 1
		if idx >= len(opts.Delays) {
			idx = len(opts.Delays) - 1
		}
		delay := opts.Delays[idx]
		if opts.Sink != nil {
			opts.Sink.ToolLog("retry", fmt.Sprintf(
				"%s: transient error (attempt %d/%d) — retrying in %s: %s",
				opts.Label, attempt, opts.Attempts, delay, err,
			))
		}
		select {
		case <-ctx.Done():
			return zero, ctx.Err()
		case <-time.After(delay):
		}
	}
	return zero, lastErr
}
