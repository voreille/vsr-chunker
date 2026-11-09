import json
import os
import shutil

import pytest

from vsr import cli
from vsr.comfy import ComfyError, ComfyFatalError
from vsr.job import Eta, JobError, Manifest, parse_redo
from vsr.pipeline import PipelineError, run_job
from vsr.plan import plan_chunks
from vsr.probe import probe


def test_parse_redo():
    assert parse_redo("3,7") == [3, 7]
    assert parse_redo("2-4, 1") == [1, 2, 3, 4]
    assert parse_redo(None) == []
    with pytest.raises(JobError):
        parse_redo("a")


def test_eta():
    e = Eta()
    assert e.remaining(10) is None
    e.add(10, 20.0)
    assert e.remaining(5) == 10.0


def test_manifest_lifecycle(tmp_path):
    chunks = plan_chunks(100, 41, 8, "4n+1")
    m = Manifest.open(tmp_path, job_id="j", args={"a": 1}, plan_info={"seed": 5}, chunks=chunks)
    assert m.counts()["pending"] == 3 and m.data["seed"] == 5
    out = tmp_path / "o.bin"
    out.write_bytes(b"x")
    m.mark_running(0)
    m.mark_done(0, out, frames=41, width=2, height=2, duration=1.0)
    m.mark_running(1)  # simulate a crash while chunk 1 runs
    m2 = Manifest.open(tmp_path, job_id="j", args={}, plan_info={}, chunks=chunks)
    assert [c["status"] for c in m2.chunks] == ["done", "pending", "pending"]
    assert m2.is_done(0)
    out.write_bytes(b"changed")
    assert not m2.is_done(0)  # checksum mismatch
    with pytest.raises(JobError, match="different chunk plan"):
        Manifest.open(tmp_path, job_id="j", args={}, plan_info={}, chunks=plan_chunks(100, 33, 8, "4n+1"))
    with pytest.raises(JobError, match="out of range"):
        m2.redo([9])


class Proc:
    """Identity fake model that records calls and can fail on given chunks."""

    def __init__(self, fail=None, exc=ComfyError):
        self.calls, self.fail, self.exc = [], set(fail or ()), exc

    def __call__(self, i, src, dst):
        self.calls.append(i)
        if i in self.fail:
            raise self.exc(f"boom on chunk {i}")
        shutil.copyfile(src, dst)
        return dst


def make_plan(clip):
    args = cli.build_parser().parse_args(["run", str(clip), "--profile", "seedvr2_3b", "--frames", "41", "--overlap", "8"])
    return cli.build_plan(args)


def go(plan, tmp_path, proc, **kw):
    return run_job(plan, job_dir=tmp_path / "job", master_path=tmp_path / "m.mov", processor=proc, **kw)


def statuses(tmp_path):
    return [c["status"] for c in Manifest.load(tmp_path / "job").chunks]


def test_resume_after_failure_and_redo(clip, tmp_path):
    plan = make_plan(clip)
    p1 = Proc(fail={1})
    with pytest.raises(PipelineError, match="(?s)1 chunk\\(s\\) failed.*chunk 1: boom"):
        go(plan, tmp_path, p1)
    assert p1.calls == [0, 1, 2]  # kept going after the failure
    assert statuses(tmp_path) == ["done", "failed", "done"]
    assert Manifest.load(tmp_path / "job").chunk(1)["error"] == "boom on chunk 1"

    p2 = Proc()
    master = go(plan, tmp_path, p2)
    assert p2.calls == [1]  # only the failed chunk
    assert probe(master).frame_count == 100
    m = Manifest.load(tmp_path / "job")
    assert m.data["master"]["frames"] == 100 and m.chunk(1)["attempts"] == 2

    p3 = Proc()
    go(plan, tmp_path, p3, redo=[0, 2])
    assert p3.calls == [0, 2]

    # a done chunk whose output was modified is redone
    out0 = tmp_path / "job" / "out" / "chunk_000.mp4"
    out0.write_bytes(out0.read_bytes() + b"x")
    p4 = Proc()
    go(plan, tmp_path, p4)
    assert p4.calls == [0]


def test_fatal_error_stops_immediately(clip, tmp_path):
    plan = make_plan(clip)
    p = Proc(fail={1}, exc=ComfyFatalError)
    with pytest.raises(ComfyFatalError):
        go(plan, tmp_path, p)
    assert p.calls == [0, 1]
    assert statuses(tmp_path) == ["done", "failed", "pending"]


def test_interrupt_leaves_chunk_pending(clip, tmp_path):
    plan = make_plan(clip)
    p = Proc(fail={1}, exc=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        go(plan, tmp_path, p)
    assert statuses(tmp_path) == ["done", "pending", "pending"]


def test_cli_run_status_resume(clip, tmp_path, capsys):
    jobs = tmp_path / "jobs"
    base = ["--profile", "seedvr2_3b", "--fake", "identity", "--frames", "41", "--jobs-dir", str(jobs), "--out", str(tmp_path)]
    assert cli.main(["run", str(clip), *base]) == 0
    (job_id,) = [p.name for p in jobs.iterdir()]
    capsys.readouterr()

    assert cli.main(["status", job_id, "--jobs-dir", str(jobs)]) == 0
    out = capsys.readouterr().out
    assert "3 done" in out and "clip_seedvr2_3b_320x240.mov" in out

    assert cli.main(["resume", job_id, "--jobs-dir", str(jobs), "--redo", "1"]) == 0
    out = capsys.readouterr().out
    assert "resuming: 2/3 chunk(s) already done" in out and "master OK" in out

    # same `run` command again -> same job, nothing to process
    assert cli.main(["run", str(clip), *base]) == 0
    assert "resuming: 3/3" in capsys.readouterr().out

    assert cli.main(["resume", "nope", "--jobs-dir", str(jobs)]) == 1
    assert job_id in capsys.readouterr().err  # lists known jobs


def test_resume_refuses_changed_source(clip, tmp_path, capsys):
    src = tmp_path / "s.mp4"
    shutil.copyfile(clip, src)
    jobs = tmp_path / "jobs"
    assert cli.main(["run", str(src), "--profile", "seedvr2_3b", "--fake", "identity", "--jobs-dir", str(jobs),
                     "--out", str(tmp_path)]) == 0
    (job_id,) = [p.name for p in jobs.iterdir()]
    st = os.stat(src)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    capsys.readouterr()
    assert cli.main(["resume", job_id, "--jobs-dir", str(jobs)]) == 1
    assert "changed since the job was created" in capsys.readouterr().err
    assert [p.name for p in jobs.iterdir()] == [job_id]  # no stray job dir created
