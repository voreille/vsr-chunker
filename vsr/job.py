"""Job manifest: per-chunk state, checksums, resume logic."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .plan import Chunk

MANIFEST = "manifest.json"
VERSION = 1
STATUSES = ("pending", "running", "done", "failed")


class JobError(RuntimeError):
    pass


def sha256(path: str | Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bufsize):
            h.update(b)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Manifest:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    # --- persistence -------------------------------------------------------------
    @classmethod
    def load(cls, job_dir: Path) -> Manifest:
        path = job_dir / MANIFEST
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise JobError(f"no manifest at {path}") from None
        except json.JSONDecodeError as e:
            raise JobError(f"corrupt manifest {path}: {e}") from None
        if data.get("version") != VERSION:
            raise JobError(f"{path}: unsupported manifest version {data.get('version')}")
        return cls(path, data)

    @classmethod
    def open(cls, job_dir: Path, *, job_id: str, args: dict, plan_info: dict, chunks: list[Chunk]) -> Manifest:
        """Load the job's manifest, or create it. An existing manifest must describe the same plan."""
        path = job_dir / MANIFEST
        want = [{"start": c.start, "length": c.length, "pad": c.pad} for c in chunks]
        if path.exists():
            m = cls.load(job_dir)
            have = [{k: c[k] for k in ("start", "length", "pad")} for c in m.chunks]
            if have != want:
                raise JobError(
                    f"job {job_id} exists with a different chunk plan ({len(have)} chunks vs {len(want)} now; "
                    f"the profile's chunk settings probably changed). Delete {job_dir} to start over."
                )
            m.data["args"] = args
            for c in m.chunks:
                if c["status"] == "running":  # interrupted mid-chunk
                    c["status"] = "pending"
            m.save()
            return m
        job_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": VERSION, "job_id": job_id, "created": now(), "updated": now(),
            "args": args, **plan_info,
            "chunks": [{"index": i, **w, "status": "pending", "attempts": 0, "input": None, "output": None,
                        "duration_s": None, "error": None} for i, w in enumerate(want)],
            "master": None,
        }
        m = cls(path, data)
        m.save()
        return m

    def save(self) -> None:
        self.data["updated"] = now()
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)

    # --- accessors ---------------------------------------------------------------
    @property
    def chunks(self) -> list[dict]:
        return self.data["chunks"]

    def chunk(self, i: int) -> dict:
        return self.chunks[i]

    def redo(self, indices: list[int]) -> None:
        n = len(self.chunks)
        bad = [i for i in indices if not 0 <= i < n]
        if bad:
            raise JobError(f"--redo: chunk index out of range {bad} (job has chunks 0..{n - 1})")
        for i in indices:
            c = self.chunks[i]
            c.update(status="pending", output=None, error=None)
        if indices:
            self.data["master"] = None
        self.save()

    def is_done(self, i: int) -> bool:
        """Done, and the output file still exists with the recorded checksum."""
        c = self.chunks[i]
        if c["status"] != "done" or not c.get("output"):
            return False
        p = Path(c["output"]["path"])
        return p.is_file() and sha256(p) == c["output"]["sha256"]

    def mark_running(self, i: int) -> None:
        c = self.chunks[i]
        c.update(status="running", error=None, started=now())
        c["attempts"] = c.get("attempts", 0) + 1
        self.save()

    def mark_input(self, i: int, path: Path) -> None:
        self.chunks[i]["input"] = {"path": str(path), "sha256": sha256(path)}
        self.save()

    def mark_done(self, i: int, path: Path, *, frames: int, width: int, height: int, duration: float) -> None:
        self.chunks[i].update(
            status="done", error=None, duration_s=round(duration, 3), finished=now(),
            output={"path": str(path), "sha256": sha256(path), "frames": frames, "width": width, "height": height},
        )
        self.save()

    def mark_failed(self, i: int, error: str, duration: float | None = None) -> None:
        self.chunks[i].update(status="failed", error=error, finished=now(),
                              duration_s=round(duration, 3) if duration is not None else None)
        self.save()

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for c in self.chunks:
            out[c["status"]] += 1
        return out


def parse_redo(spec: str | None) -> list[int]:
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        except ValueError:
            raise JobError(f"--redo expects chunk indices like '3,7' or '2-5', got {spec!r}") from None
    return sorted(set(out))


class Eta:
    """Seconds-per-frame estimate from chunks processed in this session."""

    def __init__(self):
        self.seconds = 0.0
        self.frames = 0

    def add(self, frames: int, seconds: float) -> None:
        self.frames += frames
        self.seconds += seconds

    def remaining(self, frames_left: int) -> float | None:
        return self.seconds / self.frames * frames_left if self.frames else None
