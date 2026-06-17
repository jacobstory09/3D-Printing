"""Tests for job cache expiry and runtime limits."""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

from terrain_app.job_cleanup import (
    JobSettings,
    JobTimeoutError,
    cleanup_expired_jobs,
    job_should_expire,
    prune_job_artifacts,
    remove_job_dir,
)
from terrain_app.progress import progress_reporter_with_timeout, write_progress


class TestJobCleanup(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self._testMethodName + "_cache")
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        remove_job_dir(self.tmp)

    def _job_dir(self, job_id: str) -> Path:
        d = self.tmp / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_progress(
        self,
        job_dir: Path,
        *,
        status: str = "running",
        updated_at: float,
        started_at: float | None = None,
    ) -> None:
        payload = {
            "status": status,
            "step": "test",
            "message": "test",
            "percent": 0,
            "error": None,
            "started_at": started_at if started_at is not None else updated_at,
            "updated_at": updated_at,
        }
        (job_dir / "progress.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_removes_finished_jobs_past_cache_ttl(self) -> None:
        now = 1_000_000.0
        job_dir = self._job_dir("done-job")
        self._write_progress(job_dir, status="done", updated_at=now - 100_000)
        (job_dir / "terrain.glb").write_bytes(b"glb")
        settings = JobSettings(cache_ttl_sec=3600, stale_running_sec=7200, max_runtime_sec=3600)
        removed = cleanup_expired_jobs(self.tmp, settings=settings, now=now)
        self.assertEqual(removed, ["done-job"])
        self.assertFalse(job_dir.exists())

    def test_keeps_recent_finished_jobs(self) -> None:
        now = 1_000_000.0
        job_dir = self._job_dir("fresh-job")
        self._write_progress(job_dir, status="done", updated_at=now - 60)
        settings = JobSettings(cache_ttl_sec=3600)
        removed = cleanup_expired_jobs(self.tmp, settings=settings, now=now)
        self.assertEqual(removed, [])
        self.assertTrue((job_dir / "progress.json").is_file())

    def test_removes_stale_running_jobs(self) -> None:
        now = 1_000_000.0
        job_dir = self._job_dir("orphan")
        self._write_progress(
            job_dir,
            status="running",
            updated_at=now - 10_000,
            started_at=now - 10_000,
        )
        settings = JobSettings(stale_running_sec=3600, cache_ttl_sec=86_400, max_runtime_sec=86_400)
        removed = cleanup_expired_jobs(self.tmp, settings=settings, now=now)
        self.assertEqual(removed, ["orphan"])

    def test_skips_active_jobs(self) -> None:
        now = 1_000_000.0
        job_dir = self._job_dir("active")
        self._write_progress(
            job_dir,
            status="running",
            updated_at=now - 10_000,
            started_at=now - 10_000,
        )
        settings = JobSettings(stale_running_sec=60, max_runtime_sec=60)
        removed = cleanup_expired_jobs(
            self.tmp, settings=settings, now=now, active_job_ids={"active"}
        )
        self.assertEqual(removed, [])
        self.assertTrue(job_dir.is_dir())

    def test_prune_job_artifacts_keeps_progress_only(self) -> None:
        job_dir = self._job_dir("timed-out")
        write_progress(job_dir, status="error", step="timeout", message="timed out", percent=100)
        (job_dir / "terrain.glb").write_bytes(b"glb")
        (job_dir / "dem.npy").write_bytes(b"dem")
        prune_job_artifacts(job_dir)
        self.assertTrue((job_dir / "progress.json").is_file())
        self.assertFalse((job_dir / "terrain.glb").exists())
        self.assertFalse((job_dir / "dem.npy").exists())

    def test_progress_reporter_with_timeout_raises(self) -> None:
        job_dir = self._job_dir("timeout")
        report = progress_reporter_with_timeout(
            job_dir,
            max_runtime_sec=10,
            started_at=time.time() - 20,
        )
        with self.assertRaises(JobTimeoutError):
            report("mesh", "Building…", 50)

    def test_job_should_expire_without_progress_uses_dir_mtime(self) -> None:
        settings = JobSettings(cache_ttl_sec=100)
        self.assertTrue(
            job_should_expire(
                None,
                dir_mtime=0.0,
                now=200.0,
                settings=settings,
                job_id="x",
                active_job_ids=set(),
            )
        )


if __name__ == "__main__":
    unittest.main()
