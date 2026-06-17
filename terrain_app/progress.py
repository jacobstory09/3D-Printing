"""Job progress snapshots for async processing."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


def _existing_started_at(job_dir: Path) -> float | None:
    path = job_dir / "progress.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("started_at")
        return float(raw) if raw is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def write_progress(
    job_dir: Path,
    *,
    status: str = "running",
    step: str = "",
    message: str = "",
    percent: int = 0,
    error: str | None = None,
    started_at: float | None = None,
) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if started_at is None:
        started_at = _existing_started_at(job_dir)
    if started_at is None:
        started_at = now
    payload: dict[str, Any] = {
        "status": status,
        "step": step,
        "message": message,
        "percent": int(max(0, min(100, percent))),
        "error": error,
        "started_at": started_at,
        "updated_at": now,
    }
    path = job_dir / "progress.json"
    tmp = job_dir / "progress.json.tmp"
    data = json.dumps(payload)
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def load_progress(cache_root: Path, job_id: str) -> dict[str, Any]:
    p = cache_root / job_id / "progress.json"
    if not p.is_file():
        raise FileNotFoundError(job_id)
    last_err: json.JSONDecodeError | None = None
    for attempt in range(5):
        try:
            text = p.read_text(encoding="utf-8").strip()
            if not text:
                raise json.JSONDecodeError("empty progress file", text, 0)
            return json.loads(text)
        except json.JSONDecodeError as exc:
            last_err = exc
            if attempt < 4:
                time.sleep(0.03)
    return {
        "status": "running",
        "step": "busy",
        "message": "Working…",
        "percent": 0,
        "error": None,
        "updated_at": time.time(),
        "progress_read_error": str(last_err) if last_err else "invalid progress.json",
    }


def progress_reporter(job_dir: Path) -> Callable[[str, str, int], None]:
    """Return ``report(step, message, percent)`` for :func:`process_kml`."""

    def report(step: str, message: str, percent: int) -> None:
        write_progress(job_dir, status="running", step=step, message=message, percent=percent)

    return report


def progress_reporter_with_timeout(
    job_dir: Path,
    *,
    max_runtime_sec: float,
    started_at: float | None = None,
) -> Callable[[str, str, int], None]:
    """Like :func:`progress_reporter` but raises :class:`~terrain_app.job_cleanup.JobTimeoutError`."""
    from terrain_app.job_cleanup import JobTimeoutError

    job_started = started_at if started_at is not None else time.time()

    def report(step: str, message: str, percent: int) -> None:
        elapsed = time.time() - job_started
        if elapsed > max_runtime_sec:
            limit = int(max_runtime_sec)
            raise JobTimeoutError(f"Job exceeded maximum runtime ({limit}s)")
        write_progress(
            job_dir,
            status="running",
            step=step,
            message=message,
            percent=percent,
            started_at=job_started,
        )

    return report
