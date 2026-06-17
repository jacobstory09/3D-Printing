"""Job cache TTL, stale-job removal, and cooperative runtime limits."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Defaults: 1 h max runtime, 24 h cache retention, 2 h without progress → orphan.
DEFAULT_MAX_RUNTIME_SEC = 3600.0
DEFAULT_CACHE_TTL_SEC = 86_400.0
DEFAULT_STALE_RUNNING_SEC = 7200.0
DEFAULT_CLEANUP_INTERVAL_SEC = 900.0


class JobTimeoutError(TimeoutError):
    """Raised when a job exceeds :data:`JobSettings.max_runtime_sec`."""


@dataclass(frozen=True)
class JobSettings:
    max_runtime_sec: float = DEFAULT_MAX_RUNTIME_SEC
    cache_ttl_sec: float = DEFAULT_CACHE_TTL_SEC
    stale_running_sec: float = DEFAULT_STALE_RUNNING_SEC
    cleanup_interval_sec: float = DEFAULT_CLEANUP_INTERVAL_SEC


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def job_settings() -> JobSettings:
    return JobSettings(
        max_runtime_sec=_env_float("TERRAIN_JOB_MAX_RUNTIME_SEC", DEFAULT_MAX_RUNTIME_SEC),
        cache_ttl_sec=_env_float("TERRAIN_JOB_CACHE_TTL_SEC", DEFAULT_CACHE_TTL_SEC),
        stale_running_sec=_env_float("TERRAIN_JOB_STALE_RUNNING_SEC", DEFAULT_STALE_RUNNING_SEC),
        cleanup_interval_sec=_env_float(
            "TERRAIN_JOB_CLEANUP_INTERVAL_SEC", DEFAULT_CLEANUP_INTERVAL_SEC
        ),
    )


def _read_progress_file(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None


def prune_job_artifacts(job_dir: Path) -> None:
    """Drop cached outputs but keep ``progress.json`` so clients can read the error."""
    if not job_dir.is_dir():
        return
    for entry in job_dir.iterdir():
        if entry.name == "progress.json":
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def remove_job_dir(job_dir: Path) -> None:
    if job_dir.is_dir():
        shutil.rmtree(job_dir, ignore_errors=True)


def job_should_expire(
    progress: dict | None,
    *,
    dir_mtime: float,
    now: float,
    settings: JobSettings,
    job_id: str,
    active_job_ids: set[str] | frozenset[str],
) -> bool:
    if job_id in active_job_ids:
        return False
    if progress is None:
        return (now - dir_mtime) >= settings.cache_ttl_sec
    status = str(progress.get("status") or "running")
    updated_at = float(progress.get("updated_at") or dir_mtime)
    started_at = float(progress.get("started_at") or updated_at)
    if status in ("done", "error"):
        return (now - updated_at) >= settings.cache_ttl_sec
    if status == "running":
        if (now - updated_at) >= settings.stale_running_sec:
            return True
        if (now - started_at) >= settings.max_runtime_sec:
            return True
    return False


def cleanup_expired_jobs(
    cache_root: Path,
    *,
    settings: JobSettings | None = None,
    now: float | None = None,
    active_job_ids: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Remove expired job directories under ``cache_root``. Returns removed job ids."""
    if settings is None:
        settings = job_settings()
    if now is None:
        now = time.time()
    active = active_job_ids if active_job_ids is not None else frozenset()
    if not cache_root.is_dir():
        return []

    removed: list[str] = []
    for job_dir in sorted(cache_root.iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        try:
            dir_mtime = job_dir.stat().st_mtime
        except OSError:
            continue
        progress = _read_progress_file(job_dir / "progress.json")
        if job_should_expire(
            progress,
            dir_mtime=dir_mtime,
            now=now,
            settings=settings,
            job_id=job_id,
            active_job_ids=active,
        ):
            remove_job_dir(job_dir)
            removed.append(job_id)
    return removed


def start_cleanup_scheduler(
    cache_root: Path,
    settings: JobSettings,
    get_active_job_ids: Callable[[], frozenset[str]],
    logger: logging.Logger | None = None,
) -> None:
    """Run cleanup on startup and periodically in a daemon thread."""
    log = logger or logging.getLogger(__name__)

    def run_once() -> None:
        try:
            removed = cleanup_expired_jobs(
                cache_root,
                settings=settings,
                active_job_ids=get_active_job_ids(),
            )
            if removed:
                log.info(
                    "Removed %d expired job(s) from cache",
                    len(removed),
                )
        except Exception:
            log.exception("job cache cleanup failed")

    run_once()

    def loop() -> None:
        while True:
            time.sleep(max(60.0, settings.cleanup_interval_sec))
            run_once()

    threading.Thread(target=loop, daemon=True, name="terrain-job-cleanup").start()
