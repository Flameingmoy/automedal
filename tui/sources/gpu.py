"""nvidia-smi poll. Silent if unavailable."""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import Optional

from tui.bus import EventBus
from tui.events import GpuSample


def _available() -> bool:
    return shutil.which("nvidia-smi") is not None


async def _sample() -> Optional[GpuSample]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except (FileNotFoundError, OSError):
        return None
    if proc.returncode != 0 or not out:
        return None
    line = out.decode("utf-8", errors="replace").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 4:
        return None
    try:
        return GpuSample(
            util_pct=float(parts[0]),
            mem_used_mb=float(parts[1]),
            mem_total_mb=float(parts[2]),
            temp_c=float(parts[3]),
            ts=time.time(),
        )
    except ValueError:
        return None


async def run(bus: EventBus, *, poll_interval: float = 2.0) -> None:
    if not _available():
        return
    while True:
        try:
            s = await _sample()
            if s is not None:
                bus.publish_nowait(s)
            await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(poll_interval)
