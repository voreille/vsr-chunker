"""Run a job: cut -> process -> verify -> stitch + master -> verify."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from . import ffmpeg as ff
from .job import Eta, Manifest, now
from .probe import ProbeError, VideoInfo, probe
from .plan import delay_extra
from .stitch import FrameWriter, StitchError, read_frames, stitch_frames, window

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
    def delayed(i: int, src: Path, dst: Path) -> Path:
        run_ffmpeg(ff.fake_model_command(kind, src, dst, frames=probe(src).frame_count), logfile,
                   f"fake model {kind} on chunk {i}")
        return dst

    if kind == "scale2":
        return ffmpeg_model
    if kind == "delay1":
        return delayed
    raise PipelineError(f"unknown fake model {kind!r} (use identity, scale2 or delay1)")


def comfy_processor(client, profile, job_id: str, seed: int, *, timeout: float = 3600.0) -> Processor:
    """Process a chunk through ComfyUI: upload -> patch -> queue -> wait -> download."""
    from .comfy import ComfyError
    from .profiles import patch_workflow

    def proc(i: int, src: Path, dst: Path) -> Path:
        name = f"vsr_{job_id}_{i:03d}{src.suffix}"
        value = client.upload(src, name=name, subfolder="vsr")
        wf = patch_workflow(profile, value, f"vsr/vsr_{job_id}_{i:03d}", seed)
        prompt_id = client.queue(wf)
        try:
            entry = client.wait(prompt_id, timeout=timeout)
        except KeyboardInterrupt:
            try:
                log(f"chunk {i}: prompt {prompt_id} {client.cancel(prompt_id)}")
            except ComfyError:
                pass
            raise
        files = client.outputs(entry, profile.output_node)
        videos = [f for f in files if Path(f.filename).suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi")]
        if len(videos) != 1:
            raise ComfyError(
                f"chunk {i}: expected one video from output node {profile.output_node}, got "
                f"{[(f.kind, f.subfolder, f.filename) for f in files]}"
            )
        return client.download(videos[0], dst.with_suffix(Path(videos[0].filename).suffix))

    return proc


def fmt_dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def run_job(plan, *, job_dir: Path, master_path: Path, processor: Processor,
            codec: str = "dnxhr", ten_bit: bool = False, args_record: dict | None = None,
            redo: list[int] | None = None) -> Path:
    """``plan`` is a cli.RunPlan. Resumable: chunks marked done in ``job_dir/manifest.json``
    (with an intact output file) are skipped. Returns the path of the verified master."""
    from .comfy import ComfyError, ComfyFatalError

    info: VideoInfo = plan.info
    chunks_dir = job_dir / "chunks"
    out_dir = job_dir / "out"
    logs = job_dir / "logs"
    for d in (chunks_dir, out_dir, logs):
        d.mkdir(parents=True, exist_ok=True)
    ext = ff.DEFAULT_CHUNK_CODEC.ext
    n = len(plan.chunks)
    delay = plan.profile.frame_delay
    fm = plan.profile.frame_multiple
    # frames fed to / expected back from the model per chunk (more than padded_length if it lags)
    model_len = [c.padded_length + delay_extra(c.padded_length, delay, fm) for c in plan.chunks]

    m = Manifest.open(job_dir, job_id=plan.job_id, args=args_record or {}, chunks=plan.chunks, plan_info={
        "source": info.path, "profile": plan.profile.name, "seed": plan.seed,
        "range_start": plan.range_start, "range_frames": plan.range_frames,
        "frames": plan.frames, "overlap": plan.overlap,
    })
    if redo:
        m.redo(redo)
        log(f"redo: chunk(s) {', '.join(map(str, redo))} reset to pending")
    todo = []
    for i in range(n):
        if m.is_done(i):
            continue
        if m.chunk(i)["status"] == "done":
            log(f"chunk {i}: marked done but its output is missing or changed; redoing it")
            m.chunk(i)["status"] = "pending"
        todo.append(i)
    if len(todo) < n:
        log(f"resuming: {n - len(todo)}/{n} chunk(s) already done")

    # 1. cut (only what still needs processing; cheap, so always recut)
    t0 = time.monotonic()
    inputs: dict[int, Path] = {}
    for i in todo:
        c = plan.chunks[i]
        dst = chunks_dir / f"chunk_{i:03d}{ext}"
        extra = model_len[i] - c.padded_length
        run_ffmpeg(ff.cut_command(plan.frame_source, dst, plan.range_start + c.start, c.length, c.pad + extra),
                   logs / "cut.log", f"cutting chunk {i}")
        got = probe(dst).frame_count
        if got != model_len[i]:
            raise PipelineError(f"chunk {i}: cut produced {got} frames, expected {model_len[i]}")
        m.mark_input(i, dst)
        inputs[i] = dst
    if todo:
        log(f"cut {len(todo)} chunk(s) in {fmt_dur(time.monotonic() - t0)}")

    # 2. process
    eta = Eta()
    frames_left = sum(model_len[i] for i in todo)
    for k, i in enumerate(todo, 1):
        c = plan.chunks[i]
        m.mark_running(i)
        t = time.monotonic()
        try:
            out = processor(i, inputs[i], out_dir / f"chunk_{i:03d}{ext}")
            oi = probe(out)
            if oi.frame_count != model_len[i]:
                raise PipelineError(
                    f"chunk {i}: model output {out} has {oi.frame_count} frames, expected {model_len[i]} "
                    f"(source frames {plan.range_start + c.start}..{plan.range_start + c.end}, pad {c.pad}"
                    + (f", +{model_len[i] - c.padded_length} for frame_delay {delay}" if delay else "") + ")"
                )
        except ComfyFatalError as e:
            m.mark_failed(i, str(e), time.monotonic() - t)
            raise
        except (ComfyError, PipelineError, ProbeError) as e:
            m.mark_failed(i, str(e), time.monotonic() - t)
            log(f"chunk {i} FAILED: {e}")
            frames_left -= model_len[i]
            continue
        except BaseException:  # Ctrl-C etc.: leave it to be redone
            m.chunk(i).update(status="pending", error="interrupted")
            m.save()
            raise
        dt = time.monotonic() - t
        m.mark_done(i, out, frames=oi.frame_count, width=oi.width, height=oi.height, duration=dt)
        eta.add(model_len[i], dt)
        frames_left -= model_len[i]
        rem = eta.remaining(frames_left)
        log(f"chunk {i} ({k}/{len(todo)}): {model_len[i]} frames -> {oi.width}x{oi.height} in {fmt_dur(dt)} "
            f"({dt / model_len[i]:.2f} s/frame)" + (f", ETA {fmt_dur(rem)}" if frames_left and rem else ""))

    failed = [c for c in m.chunks if c["status"] != "done"]
    if failed:
        lines = "\n".join(f"  chunk {c['index']}: {c['error']}" for c in failed)
        raise PipelineError(
            f"{len(failed)} chunk(s) failed; not stitching.\n{lines}\n"
            f"Fix the cause, then rerun: vsr resume {plan.job_id}"
        )

    outputs = [Path(c["output"]["path"]) for c in m.chunks]
    sizes = {(c["output"]["width"], c["output"]["height"]) for c in m.chunks}
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
    samples = ff.stitch_frame_samples(w, h)
    readers = (window(read_frames(ff.decode_command(o, info), samples, logs / "decode.log"), delay, c.padded_length)
               for o, c in zip(outputs, plan.chunks))
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
    m.data["master"] = {"path": str(final), "frames": mi.frame_count, "width": w, "height": h, "finished": now()}
    m.save()
    log(f"master OK: {final}  ({w}x{h}, {mi.frame_count} frames, {mi.duration:.3f}s)")
    return final
