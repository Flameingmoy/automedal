"""Entry point: `python -m sniff <data_dir>` → JSON on stdout.

The Go control plane spawns this as a subprocess to run pandas-backed
schema inference. It is the only Python in the AutoMedal control plane
after the Phase 4 port.
"""

from __future__ import annotations

import json
import sys

from sniff.sniff import sniff_schema


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    try:
        result = sniff_schema(data_dir)
    except Exception as exc:
        json.dump(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "confidence": 0.0,
                "warnings": [str(exc)],
            },
            sys.stdout,
        )
        return 1
    json.dump(result, sys.stdout, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
