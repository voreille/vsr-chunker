"""Run a job: cut -> process -> verify -> stitch + master -> verify."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from . import ffmpeg as ff
from .probe import VideoInfo, probe
from .stitch import FrameWriter, StitchError, read_frames, stitch_frames

# processor(index, input_chunk_path, output_path) -> path of the processed chunk
Processor = Callable[[int, Path, Path], Path]


class PipelineError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def run_ffmpeg(cmd: list[str], logfile: Path, what: str) -> None:
    with open(logfile, "ab") as f:
        f.write(("\n$ " + " ".join(cmd) + "\n").encode())
        f.flush()
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=f)
    if res.returncode != 0:
        tail = logfile.read_text(errors="replace").strip().splitlines()[-5:]
        raise PipelineError(f"{what} failed (ffmpeg exit {res.returncode}); see {logfile}\n    " + "\n    ".join(tail))


def fake_processor(kind: str, logfile: Path) -> Processor:
    def identity(i: int, src: Path, dst: Path) -> Path:
        shutil.copyfile(src, dst)
        return dst

    def ffmpeg_model(i: int, src: Path, dst: Path) -> Path:
        run_ffmpeg(ff.fake_model_command(kind, src, dst), logfile, f"fake model {kind} on chunk {i}")
        return dst

    if kind == "identity":
        return identity
    if kind == "scale2":
        return ffmpeg_model
    raise PipelineError(f"unknown fake model {kind!r} (use identity or scale2)")


def fmt_dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def run_job(plan, *, job_dir: Path, master_path: Path, processor: Processor,
            codec: str = "dnxhr", ten_bit: bool = False) -> Path:
    """``plan`` is a cli.RunPlan. Returns the path of the verified master."""
    info: VideoInfo = plan.info
    chunks_dir = job_dir / "chunks"
    out_dir = job_dir / "out"
    logs = job_dir / "logs"
    for d in (chunks_dir, out_dir, logs):
        d.mkdir(parents=True, exist_ok=True)
    ext = ff.DEFAULT_CHUNK_CODEC.ext
    n = len(plan.chunks)

    # 1. cut
    t0 = time.monotonic()
    inputs = []
    for i, c in enumerate(plan.chunks):
        dst = chunks_dir / f"chunk_{i:03d}{ext}"
        run_ffmpeg(ff.cut_command(plan.frame_source, dst, plan.range_start + c.start, c.length, c.pad),
                   logs / "cut.log", f"cutting chunk {i}")
        got = probe(dst).frame_count
        if got != c.padded_length:
            raise PipelineError(f"chunk {i}: cut produced {got} frames, expected {c.padded_length}")
        inputs.append(dst)
    log(f"cut {n} chunk(s) in {fmt_dur(time.monotonic() - t0)}")

    # 2. process
    outputs: list[Path] = []
    t_start = time.monotonic()
    for i, (c, src) in enumerate(zip(plan.chunks, inputs)):
        t = time.monotonic()
        out = processor(i, src, out_dir / f"chunk_{i:03d}{ext}")
        oi = probe(out)
        if oi.frame_count != c.padded_length:
            raise PipelineError(
                f"chunk {i}: model output {out} has {oi.frame_count} frames, expected {c.padded_length} "
                f"(source frames {plan.range_start + c.start}..{plan.range_start + c.end}, pad {c.pad})"
            )
        outputs.append(out)
        dt = time.monotonic() - t
        done = i + 1
        eta = (time.monotonic() - t_start) / done * (n - done)
        log(f"chunk {done}/{n}: {c.padded_length} frames -> {oi.width}x{oi.height} in {fmt_dur(dt)}"
            + (f", ETA {fmt_dur(eta)}" if done < n else ""))

    sizes = {(o.width, o.height) for o in map(probe, outputs)}
    if len(sizes) != 1:
        raise PipelineError(f"processed chunks have different sizes: {sorted(sizes)}")
    (w, h), = sizes

    # 3. stitch straight into the master encoder
    final = master_path.with_name(ff.master_output_name(info, plan.profile.name, w, h)) \
        if master_path.name.endswith("_WxH.mov") else master_path
    final.parent.mkdir(parents=True, exist_ok=True)
    cmd, warns = ff.master_command(
        None, info, final, codec=codec, ten_bit=ten_bit,
        range_start_frame=plan.range_start, range_frames=plan.range_frames,
        raw=(w, h, ff.STITCH_PIX_FMT),
    )
    for wmsg in warns:
        log(f"warning: {wmsg}")
    t = time.monotonic()
    (logs / "master.log").write_text("$ " + " ".join(cmd) + "\n")
    writer = FrameWriter(cmd, logs / "master.log")
    readers = (read_frames(ff.decode_command(o, info), ff.stitch_frame_samples(w, h), logs / "decode.log") for o in outputs)
    try:
        written = stitch_frames(plan.chunks, readers, writer)
        writer.close()
    except StitchError as e:
        writer.abort()
        raise PipelineError(str(e)) from None
    except BaseException:
        writer.abort()
        raise
    log(f"stitched {written} frames into {final.name} in {fmt_dur(time.monotonic() - t)}")

    # 4. verify
    mi = probe(final)
    if mi.frame_count != plan.range_frames:
        raise PipelineError(f"master {final} has {mi.frame_count} frames, expected {plan.range_frames}")
    if (mi.width, mi.height) != (w, h):
        raise PipelineError(f"master {final} is {mi.width}x{mi.height}, expected {w}x{h}")
    log(f"master OK: {final}  ({w}x{h}, {mi.frame_count} frames, {mi.duration:.3f}s)")
    return final
