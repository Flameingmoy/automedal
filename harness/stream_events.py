#!/usr/bin/env python3
"""
AutoMedal — Pi JSON Event Stream → Terminal Output
=====================================================
Reads JSON lines from pi's --mode json output on stdin and prints
human-readable summaries to stdout in real time.

Usage (called by run.sh, not directly):
    pi --mode json -p "..." | tee -a log | python3 -u harness/stream_events.py
"""

import json
import sys


def _truncate(s, maxlen=120):
    s = str(s).replace("\n", " ")
    return s[:maxlen] + "..." if len(s) > maxlen else s


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        t = event.get("type", "")

        if t == "tool_execution_start":
            tool = event.get("toolName", "?")
            args = event.get("args", {})
            if isinstance(args, dict):
                # Show the most useful arg for common tools
                if "command" in args:
                    summary = _truncate(args["command"])
                elif "file_path" in args or "path" in args:
                    summary = args.get("file_path") or args.get("path", "")
                else:
                    summary = _truncate(args)
            else:
                summary = _truncate(args)
            print(f"  [{tool}] {summary}", flush=True)

        elif t == "tool_execution_end":
            tool = event.get("toolName", "?")
            is_err = event.get("isError", False)
            if is_err:
                result = _truncate(event.get("result", ""), 200)
                print(f"  [{tool}] ERROR: {result}", flush=True)

        elif t == "message_update":
            ame = event.get("assistantMessageEvent", {})
            if ame.get("type") == "text_delta":
                delta = ame.get("delta", "")
                print(delta, end="", flush=True)

        elif t == "turn_end":
            print("", flush=True)

        elif t == "agent_end":
            print("", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
