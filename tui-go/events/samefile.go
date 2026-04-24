//go:build linux || darwin || freebsd || netbsd || openbsd
// +build linux darwin freebsd netbsd openbsd

package events

import (
	"os"
	"syscall"
)

// sameFile compares two stat results by inode (device+ino) on Unix.
func sameFile(a, b os.FileInfo) bool {
	ax, ok1 := a.Sys().(*syscall.Stat_t)
	bx, ok2 := b.Sys().(*syscall.Stat_t)
	if !ok1 || !ok2 {
		return true // fall back to "same"; polling will recover us if wrong
	}
	return ax.Dev == bx.Dev && ax.Ino == bx.Ino
}
