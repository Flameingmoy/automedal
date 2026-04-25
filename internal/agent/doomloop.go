package agent

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const (
	doomLookback         = 30
	doomIdenticalThresh  = 3
	doomMinPattern       = 2
	doomMaxPattern       = 3
	doomPatternRepeats   = 2
)

type doomSig struct {
	Name string
	Hash string
}

func toolSig(name string, args map[string]any) doomSig {
	blob, err := json.Marshal(args)
	if err != nil {
		blob = []byte(fmt.Sprintf("%v", args))
	}
	// Match the Python sort_keys=True behavior: re-encode through a
	// canonical map by parsing back and emitting with sorted keys via
	// json.Marshal of map[string]any (Go marshals map[string]any with
	// sorted keys as of stdlib).
	var canon map[string]any
	if json.Unmarshal(blob, &canon) == nil {
		blob, _ = json.Marshal(canon)
	}
	h := sha1.Sum(blob)
	return doomSig{Name: name, Hash: hex.EncodeToString(h[:6])}
}

// recentToolSignatures extracts the last `lookback` tool_use signatures
// from assistant turns in chronological order.
func recentToolSignatures(messages []Message, lookback int) []doomSig {
	out := []doomSig{}
	for i := len(messages) - 1; i >= 0; i-- {
		m := messages[i]
		if m.Role != "assistant" {
			continue
		}
		blocks, ok := m.Content.([]Block)
		if !ok {
			continue
		}
		for j := len(blocks) - 1; j >= 0; j-- {
			b := blocks[j]
			if b.Type != "tool_use" {
				continue
			}
			out = append(out, toolSig(b.Name, b.Input))
			if len(out) >= lookback {
				break
			}
		}
		if len(out) >= lookback {
			break
		}
	}
	// Reverse to chronological.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

func detectIdentical(sigs []doomSig, threshold int) (doomSig, bool) {
	if len(sigs) < threshold {
		return doomSig{}, false
	}
	tail := sigs[len(sigs)-threshold:]
	first := tail[0]
	for _, s := range tail[1:] {
		if s != first {
			return doomSig{}, false
		}
	}
	return first, true
}

func detectCycle(sigs []doomSig, minLen, maxLen, repeats int) []doomSig {
	for length := minLen; length <= maxLen; length++ {
		needed := length * repeats
		if len(sigs) < needed {
			continue
		}
		tail := sigs[len(sigs)-needed:]
		pattern := tail[:length]
		// Reject all-equal patterns (covered by detectIdentical).
		allEq := true
		for _, s := range pattern[1:] {
			if s != pattern[0] {
				allEq = false
				break
			}
		}
		if allEq {
			continue
		}
		ok := true
		for r := 1; r < repeats; r++ {
			chunk := tail[r*length : (r+1)*length]
			for i := range pattern {
				if chunk[i] != pattern[i] {
					ok = false
					break
				}
			}
			if !ok {
				break
			}
		}
		if ok {
			return pattern
		}
	}
	return nil
}

// CheckDoomLoop returns a corrective user message when the recent tool
// signatures show identical-3 repetition or a short cycle. Returns "" when
// the transcript is healthy or the env kill-switch is set.
//
// Port of automedal/agent/doom_loop.py:check_for_doom_loop.
func CheckDoomLoop(messages []Message) string {
	if os.Getenv("AUTOMEDAL_DOOM_LOOP") == "0" {
		return ""
	}
	sigs := recentToolSignatures(messages, doomLookback)
	if len(sigs) == 0 {
		return ""
	}
	if hit, ok := detectIdentical(sigs, doomIdenticalThresh); ok {
		return fmt.Sprintf(
			"[SYSTEM] You have called tool %q %d times with identical "+
				"arguments. The response isn't changing — repeating won't help. "+
				"Try a different tool, change the arguments, or produce a final "+
				"assistant message with the conclusion you currently have.",
			hit.Name, doomIdenticalThresh,
		)
	}
	if pattern := detectCycle(sigs, doomMinPattern, doomMaxPattern, doomPatternRepeats); pattern != nil {
		names := make([]string, len(pattern))
		for i, p := range pattern {
			names[i] = p.Name
		}
		return fmt.Sprintf(
			"[SYSTEM] You appear to be cycling between tool calls [%s] without "+
				"making progress. Break out of the loop: pick one path and follow "+
				"it, or stop calling tools and produce your best final answer.",
			strings.Join(names, " → "),
		)
	}
	return ""
}
