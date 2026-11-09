"""vsr command line."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import shlex
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import ffmpeg as ff
from .plan import Chunk, overlaps, plan_chunks
from .comfy import ComfyClient, ComfyError, ComfyFatalError
from .job import MANIFEST, JobError, Manifest, parse_redo
from .pipeline import PipelineError, comfy_processor, fake_processor, fmt_dur, run_ffmpeg, run_job
from .probe import ProbeError, VideoInfo, check_tools, probe
from .profiles import PROJECT_ROOT, Profile, ProfileError, describe_workflow, load_profile, load_workflow

DEFAULT_SERVER = "http://127.0.0.1:8188"
JOBS_DIR = PROJECT_ROOT / "jobs"


class CliError(RuntimeError):
    pass


def parse_range(spec: str) -> tuple[float, float | None]:
    """'START-END' in seconds; END may be empty ('12.5-') for 'to the end'."""
    if "-" not in spec:
        raise CliError(f"--range must look like START-END in seconds (e.g. 10-15.5), got {spec!r}")
    a, b = spec.split("-", 1)
    try:
        start = float(a) if a.strip() else 0.0
        end = float(b) if b.strip() else None
    except ValueError:
        raise CliError(f"--range must look like START-END in seconds (e.g. 10-15.5), got {spec!r}") from None
    if start < 0 or (end is not None and end <= start):
        raise CliError(f"--range: need 0 <= START < END, got {spec!r}")
    return start, end


@dataclass
class RunPlan:
    info: VideoInfo
    profile: Profile
    range_start: int  # absolute source frame
    range_frames: int
    chunks: list[Chunk]
    frames: int
    overlap: int
    seed: int
    job_id: str
    warnings: list[str]
    frame_source: str  # file the chunks are cut from (the source, or its CFR conversion)


def make_job_id(info: VideoInfo, profile: Profile, frames: int, overlap: int, rs: int, rn: int, seed: int | None) -> str:
    src = Path(info.path).resolve()
    st = src.stat()
    key = f"{src}|{st.st_size}|{st.st_mtime_ns}|{profile.name}|{frames}|{overlap}|{rs}|{rn}|{seed}"
    return f"{src.stem[:40]}_{profile.name}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def build_plan(args, info: VideoInfo | None = None, frames_info: VideoInfo | None = None) -> RunPlan:
    """``info`` is the source (audio, timecode, color). ``frames_info`` is the file chunks are
    cut from, when it differs (CFR conversion of a VFR source)."""
    info = info or probe(args.video)
    profile = load_profile(args.profile)
    warnings = list(info.warnings)
    fsrc = frames_info or info

    frames = args.frames if args.frames is not None else profile.frames
    overlap = args.overlap if args.overlap is not None else profile.overlap

    total = fsrc.frame_count
    rs, rn = 0, total
    if args.range:
        start_s, end_s = parse_range(args.range)
        rs = fsrc.frame_at(start_s)
        re_ = total if end_s is None else min(fsrc.frame_at(end_s), total)
        if rs >= total:
            raise CliError(f"--range start {start_s}s is past the end of the clip ({info.duration:.3f}s)")
        if end_s is not None and fsrc.frame_at(end_s) > total:
            warnings.append(f"--range end {end_s}s is past the end of the clip; clamped to {total} frames")
        rn = re_ - rs

    try:
        chunks = plan_chunks(rn, frames, overlap, profile.frame_multiple)
    except ValueError as e:
        raise CliError(f"chunk plan: {e}") from None

    longest = max(c.padded_length for c in chunks) / info.fps
    if profile.max_seconds and longest > profile.max_seconds:
        warnings.append(
            f"chunks are {float(longest):.2f}s long, above this profile's max_seconds={profile.max_seconds} "
            "(see CLAUDE.md 'Known model constraints'); lower --frames"
        )
    if overlap < 4 and len(chunks) > 1:
        warnings.append(f"overlap={overlap} is small; diffusion models need a crossfade of several frames")

    job_id = make_job_id(info, profile, frames, overlap, rs, rn, args.seed)
    seed = args.seed if args.seed is not None else int(job_id.rsplit("_", 1)[1], 16) % 2**31
    return RunPlan(info, profile, rs, rn, chunks, frames, overlap, seed, job_id, warnings, fsrc.path)


def _fmt_tc(frame: int, fps: Fraction) -> str:
    s = float(frame / fps)
    return f"{int(s // 60):02d}:{s % 60:06.3f}"


def print_dry_run(p: RunPlan, args) -> None:
    info, prof = p.info, p.profile
    fps = info.fps
    out = print

    out("== Source")
    out(f"  file       {info.path}")
    out(f"  video      {info.codec} {info.width}x{info.height} {info.pix_fmt}"
        + (f" ({info.bit_depth}-bit)" if info.bit_depth else ""))
    out(f"  fps        {ff.fps_str(fps)} ({float(fps):.3f}){'  VFR!' if info.vfr else ''}")
    out(f"  frames     {info.frame_count}  ({info.duration:.3f}s)")
    out(f"  color      range={info.color_range} primaries={info.color_primaries} "
        f"transfer={info.color_transfer} matrix={info.color_space}")
    out(f"  timecode   {info.timecode or '-'}")
    if info.audio:
        for a in info.audio:
            out(f"  audio #{a.index}   {a.codec} {a.channels}ch {a.sample_rate} Hz" + (f" [{a.language}]" if a.language else ""))
    else:
        out("  audio      none")

    out("\n== Profile")
    out(f"  name       {prof.name}  ({prof.path})")
    out(f"  workflow   {prof.workflow_path}")
    wf = prof.workflow
    out(f"  input      [{prof.input_node}] {wf[prof.input_node]['class_type']}.{prof.input_field}")
    out(f"  output     [{prof.output_node}] {wf[prof.output_node]['class_type']} "
        f"({prof.output_prefix_field}=vsr_{p.job_id}_<chunk>)")
    for nid, fields in prof.overrides.items():
        out(f"  override   [{nid}] {wf[nid]['class_type']}: {fields}")
    seeds = prof.seed_fields()
    out(f"  seed       {p.seed}" + (" (--seed)" if args.seed is not None else " (derived from job id)")
        + (f" -> {', '.join(f'[{n}].{f}' for n, f in seeds)}" if seeds else "  (no seed field found in workflow)"))
    out(f"  server     {args.server}  (not contacted in --dry-run)")

    out("\n== Chunk plan")
    out(f"  range      frames {p.range_start}..{p.range_start + p.range_frames - 1} "
        f"({p.range_frames} frames, {float(p.range_frames / fps):.3f}s)")
    out(f"  settings   frames={p.frames} overlap={p.overlap} frame_multiple={prof.frame_multiple}")
    out(f"  {'#':>4} {'start':>7} {'end':>7} {'len':>5} {'pad':>4} {'sent':>5} {'time':>18}")
    for i, c in enumerate(p.chunks):
        a = p.range_start + c.start
        out(f"  {i:>4} {a:>7} {a + c.length - 1:>7} {c.length:>5} {c.pad:>4} {c.padded_length:>5} "
            f"{_fmt_tc(a, fps):>9}-{_fmt_tc(a + c.length, fps)}")
    ov = overlaps(p.chunks)
    sent = sum(c.padded_length for c in p.chunks)
    out(f"  {len(p.chunks)} chunk(s), overlaps {sorted(set(ov)) if ov else '-'}, "
        f"{sent} frames sent to the model ({sent / p.range_frames:.2f}x the range)")

    job_dir = Path(args.jobs_dir) / p.job_id
    out(f"\n== Commands  (job {p.job_id}, dir {job_dir})")
    src = info.path
    if info.vfr:
        cfr = job_dir / f"source_cfr{ff.DEFAULT_CHUNK_CODEC.ext}"
        out("  # convert VFR source to CFR first")
        out("  " + shlex.join(ff.cfr_command(src, cfr, ff.cfr_target_fps(fps))))
        out("  # (chunk plan below is from the VFR packet count; it is recomputed after conversion)")
        src = str(cfr)
    for i, c in enumerate(p.chunks):
        dst = job_dir / "chunks" / f"chunk_{i:03d}{ff.DEFAULT_CHUNK_CODEC.ext}"
        out("  " + shlex.join(ff.cut_command(src, dst, p.range_start + c.start, c.length, c.pad)))
    out("  # ... upload / queue / download each chunk via ComfyUI, then decode each output to")
    out(f"  # {ff.STITCH_PIX_FMT}, crossfade the overlaps (numpy, streamed), drop pad frames, and pipe into:")
    out_dir = Path(args.out) if args.out else Path(info.path).resolve().parent
    master = out_dir / ff.master_output_name(info, prof.name, None, None)
    cmd, mw = ff.master_command(
        None, info, master, codec=args.codec, ten_bit=args.ten_bit,
        range_start_frame=p.range_start, range_frames=p.range_frames,
        raw=("W", "H", ff.STITCH_PIX_FMT),
    )
    p.warnings.extend(mw)
    out("  " + shlex.join(cmd))
    out("  (WxH in the master name is filled in from the real model output)")

    if p.warnings:
        out("\n== Warnings")
        for w in p.warnings:
            out(f"  ! {w}")


# --- commands -----------------------------------------------------------------

def cmd_inspect(args) -> int:
    wf = load_workflow(args.workflow)
    print("\n".join(describe_workflow(wf)))
    return 0


def cmd_run(args) -> int:
    check_tools()
    info = probe(args.video)
    plan = build_plan(args, info)
    if getattr(args, "expect_job_id", None) and plan.job_id != args.expect_job_id:
        raise CliError(
            f"cannot resume {args.expect_job_id}: the same arguments now give job {plan.job_id} "
            f"(the source file {info.path} changed since the job was created)"
        )
    if args.dry_run:
        print_dry_run(plan, args)
        return 0
    client = None
    if not args.fake:
        client = ComfyClient(args.server)
        stats = client.system_stats()  # fail fast, before cutting anything
        dev = (stats.get("devices") or [{}])[0]
        print(f"ComfyUI {stats.get('system', {}).get('comfyui_version', '?')} at {args.server}: "
              f"{dev.get('name', '?')}, {dev.get('vram_free', 0) / 2**30:.1f} GiB VRAM free", flush=True)

    job_dir = Path(args.jobs_dir) / plan.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    if info.vfr:
        target = ff.cfr_target_fps(info.fps)
        cfr = job_dir / f"source_cfr{ff.DEFAULT_CHUNK_CODEC.ext}"
        print(f"warning: variable frame rate source; converting to CFR {ff.fps_str(target)} "
              f"({float(target):.3f} fps) -> {cfr}", flush=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        run_ffmpeg(ff.cfr_command(info.path, cfr, target), job_dir / "logs" / "cfr.log", "CFR conversion")
        # audio/timecode/color still come from the original file; frames and fps from the CFR copy
        cfr_info = dataclasses.replace(info, fps=target, vfr=False, warnings=[])
        replanned = build_plan(args, cfr_info, probe(cfr))
        replanned.job_id, replanned.seed = plan.job_id, plan.seed  # keep the id of the source job
        plan = replanned

    for w in plan.warnings:
        print(f"warning: {w}", flush=True)
    print(f"job {plan.job_id}: {len(plan.chunks)} chunk(s), frames {plan.range_start}.."
          f"{plan.range_start + plan.range_frames - 1}, "
          + (f"fake model '{args.fake}'" if args.fake else f"profile {plan.profile.name}, seed {plan.seed}"), flush=True)
    out_dir = Path(args.out) if args.out else Path(info.path).resolve().parent
    master = out_dir / ff.master_output_name(info, plan.profile.name, None, None)
    if client:
        processor = comfy_processor(client, plan.profile, plan.job_id, plan.seed, timeout=args.timeout)
    else:
        processor = fake_processor(args.fake, job_dir / "logs" / "model.log")
    print(f"(resume later with: vsr resume {plan.job_id})", flush=True)
    try:
        run_job(plan, job_dir=job_dir, master_path=master, processor=processor,
                codec=args.codec, ten_bit=args.ten_bit, args_record=run_args_record(args),
                redo=parse_redo(args.redo))
    finally:
        if client:
            client.close()
    return 0


# Arguments of `vsr run` stored in the manifest, so `vsr resume <id>` can replay them.
RUN_ARG_KEYS = ("video", "profile", "out", "server", "seed", "frames", "overlap", "range",
                "codec", "ten_bit", "timeout", "fake")


def run_args_record(args) -> dict:
    rec = {k: getattr(args, k) for k in RUN_ARG_KEYS}
    rec["video"] = str(Path(args.video).resolve())
    if Path(args.profile).is_file():
        rec["profile"] = str(Path(args.profile).resolve())
    if args.out:
        rec["out"] = str(Path(args.out).resolve())
    return rec


def _job_dir(args) -> Path:
    d = Path(args.jobs_dir) / args.job_id
    if not (d / MANIFEST).is_file():
        known = sorted(p.parent.name for p in Path(args.jobs_dir).glob(f"*/{MANIFEST}"))
        raise CliError(f"no job {args.job_id!r} in {args.jobs_dir}" + (f". Known jobs: {', '.join(known)}" if known else ""))
    return d


def cmd_resume(args) -> int:
    m = Manifest.load(_job_dir(args))
    stored = m.data.get("args") or {}
    if not stored.get("video"):
        raise CliError(f"job {args.job_id} has no stored run arguments; rerun it with 'vsr run'")
    run_args = build_parser().parse_args(["run", stored["video"], "--profile", stored["profile"]])
    for k in RUN_ARG_KEYS:
        if k in stored:
            setattr(run_args, k, stored[k])
    if args.server:
        run_args.server = args.server
    if args.timeout:
        run_args.timeout = args.timeout
    run_args.redo = args.redo
    run_args.jobs_dir = args.jobs_dir
    run_args.expect_job_id = args.job_id
    return cmd_run(run_args)


def cmd_status(args) -> int:
    m = Manifest.load(_job_dir(args))
    d = m.data
    counts = m.counts()
    print(f"job      {d['job_id']}  (created {d['created']}, updated {d['updated']})")
    print(f"source   {d.get('source')}")
    print(f"profile  {d.get('profile')}  seed {d.get('seed')}  frames {d.get('frames')} overlap {d.get('overlap')}  "
          f"range {d.get('range_start')}+{d.get('range_frames')} frames")
    print("chunks   " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    print(f"  {'#':>4} {'status':8} {'start':>7} {'len':>5} {'pad':>4} {'tries':>5} {'time':>8}  output / error")
    for c in m.chunks:
        dur = fmt_dur(c["duration_s"]) if c.get("duration_s") is not None else "-"
        extra = c["error"] if c.get("error") else (Path(c["output"]["path"]).name if c.get("output") else "")
        print(f"  {c['index']:>4} {c['status']:8} {c['start']:>7} {c['length']:>5} {c['pad']:>4} "
              f"{c.get('attempts', 0):>5} {dur:>8}  {extra.splitlines()[0] if extra else ''}")
    done = [c for c in m.chunks if c["status"] == "done" and c.get("duration_s")]
    if done and counts["done"] < len(m.chunks):
        spf = sum(c["duration_s"] for c in done) / sum(c["length"] + c["pad"] for c in done)
        left = sum(c["length"] + c["pad"] for c in m.chunks if c["status"] != "done")
        print(f"estimate {fmt_dur(spf * left)} left at {spf:.2f} s/frame")
    master = d.get("master")
    print(f"master   {master['path']} ({master['width']}x{master['height']}, {master['frames']} frames)"
          if master else "master   not written yet")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vsr", description="Chunked video upscaling through ComfyUI")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="list the nodes of an API-format workflow")
    p.add_argument("workflow")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("run", help="upscale a video")
    p.add_argument("video")
    p.add_argument("--profile", required=True, help="profile name (profiles/<name>.yaml) or path")
    p.add_argument("--out", help="output directory for the master (default: next to the source)")
    p.add_argument("--server", default=DEFAULT_SERVER, help=f"ComfyUI URL (default {DEFAULT_SERVER})")
    p.add_argument("--seed", type=int, help="fixed seed for every chunk (default: derived from the job id)")
    p.add_argument("--frames", type=int, help="frames per chunk (overrides the profile)")
    p.add_argument("--overlap", type=int, help="overlap frames between chunks (overrides the profile)")
    p.add_argument("--range", help="only process START-END seconds of the source, e.g. 10-15.5")
    p.add_argument("--codec", choices=["dnxhr", "prores"], default="dnxhr", help="master codec")
    p.add_argument("--10bit", dest="ten_bit", action="store_true", help="10-bit master (DNxHR HQX)")
    p.add_argument("--dry-run", action="store_true", help="probe + plan + print commands; don't touch ComfyUI")
    p.add_argument("--timeout", type=float, default=3600.0, help="max seconds per chunk in ComfyUI (default 3600)")
    p.add_argument("--jobs-dir", default=str(JOBS_DIR), help=f"job state directory (default {JOBS_DIR})")
    p.add_argument("--redo", help="reprocess these chunks even if done, e.g. 3,7 or 2-5")
    p.add_argument("--fake", choices=["identity", "scale2"],
                   help="testing: replace the ComfyUI model with a local fake (copy, or lanczos x2)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="continue a job with its stored arguments (skips done chunks)")
    p.add_argument("job_id")
    p.add_argument("--server", help="override the stored ComfyUI URL")
    p.add_argument("--timeout", type=float, help="override the stored per-chunk timeout")
    p.add_argument("--redo", help="reprocess these chunks, e.g. 3,7 or 2-5")
    p.add_argument("--jobs-dir", default=str(JOBS_DIR))
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("status", help="show a job's chunk states")
    p.add_argument("job_id")
    p.add_argument("--jobs-dir", default=str(JOBS_DIR))
    p.set_defaults(func=cmd_status)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ComfyFatalError as e:
        print(f"fatal ComfyUI error: {e}", file=sys.stderr)
        return 2
    except (CliError, ProbeError, ProfileError, PipelineError, ComfyError, JobError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; done chunks are kept. Continue with 'vsr resume <job_id>'.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
