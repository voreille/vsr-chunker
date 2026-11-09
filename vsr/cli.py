"""vsr command line."""

from __future__ import annotations

import argparse
import hashlib
import shlex
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import ffmpeg as ff
from .plan import Chunk, overlaps, plan_chunks
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


def make_job_id(info: VideoInfo, profile: Profile, frames: int, overlap: int, rs: int, rn: int, seed: int | None) -> str:
    src = Path(info.path).resolve()
    st = src.stat()
    key = f"{src}|{st.st_size}|{st.st_mtime_ns}|{profile.name}|{frames}|{overlap}|{rs}|{rn}|{seed}"
    return f"{src.stem[:40]}_{profile.name}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def build_plan(args) -> RunPlan:
    info = probe(args.video)
    profile = load_profile(args.profile)
    warnings = list(info.warnings)

    frames = args.frames if args.frames is not None else profile.frames
    overlap = args.overlap if args.overlap is not None else profile.overlap

    rs, rn = 0, info.frame_count
    if args.range:
        start_s, end_s = parse_range(args.range)
        rs = info.frame_at(start_s)
        re_ = info.frame_count if end_s is None else min(info.frame_at(end_s), info.frame_count)
        if rs >= info.frame_count:
            raise CliError(f"--range start {start_s}s is past the end of the clip ({info.duration:.3f}s)")
        if end_s is not None and info.frame_at(end_s) > info.frame_count:
            warnings.append(f"--range end {end_s}s is past the end of the clip; clamped to {info.frame_count} frames")
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
    return RunPlan(info, profile, rs, rn, chunks, frames, overlap, seed, job_id, warnings)


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

    job_dir = JOBS_DIR / p.job_id
    out(f"\n== Commands  (job {p.job_id}, dir {job_dir})")
    src = info.path
    if info.vfr:
        cfr = job_dir / f"source_cfr{ff.DEFAULT_CHUNK_CODEC.ext}"
        out("  # convert VFR source to CFR first")
        out("  " + shlex.join(ff.cfr_command(src, cfr, fps)))
        src = str(cfr)
    for i, c in enumerate(p.chunks):
        dst = job_dir / "chunks" / f"chunk_{i:03d}{ff.DEFAULT_CHUNK_CODEC.ext}"
        out("  " + shlex.join(ff.cut_command(src, dst, p.range_start + c.start, c.length, c.pad)))
    out("  # ... upload / queue / download each chunk via ComfyUI, then crossfade-stitch to:")
    stitched = job_dir / "stitched.mkv"
    out_dir = Path(args.out) if args.out else Path(info.path).resolve().parent
    master = out_dir / ff.master_output_name(info, prof.name, None, None)
    cmd, mw = ff.master_command(
        stitched, info, master, codec=args.codec, ten_bit=args.ten_bit,
        range_start_frame=p.range_start, range_frames=p.range_frames,
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
    plan = build_plan(args)
    if args.dry_run:
        print_dry_run(plan, args)
        return 0
    raise CliError("processing is not implemented yet (milestone 2+). Use --dry-run.")


def cmd_not_yet(args) -> int:
    raise CliError(f"'vsr {args.command}' is not implemented yet (milestone 4)")


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
    p.set_defaults(func=cmd_run)

    for name in ("resume", "status"):
        p = sub.add_parser(name, help=f"{name} a job (not implemented yet)")
        p.add_argument("job_id")
        p.set_defaults(func=cmd_not_yet)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (CliError, ProbeError, ProfileError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
