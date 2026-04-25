//go:build !linux && !darwin && !freebsd && !netbsd && !openbsd
// +build !linux,!darwin,!freebsd,!netbsd,!openbsd

package events

import "os"

// On non-Unix we fall back to "always same file"; the poller + content-size
// check still detects truncation.
func sameFile(a, b os.FileInfo) bool { return true }
