"""ffprobe wrapper."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path


class ProbeError(RuntimeError):
    pass


def check_tools(tools: tuple[str, ...] = ("ffmpeg", "ffprobe")) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise ProbeError(
            f"required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install them (Ubuntu: sudo apt install ffmpeg)."
        )


@dataclass
class AudioStream:
    index: int
    codec: str
    channels: int | None
    sample_rate: int | None
    language: str | None = None


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Fraction  # constant frame rate used for frame <-> time conversion
    frame_count: int  # counted packets on the first video stream
    duration: float  # seconds (container/stream)
    codec: str
    pix_fmt: str
    bit_depth: int | None
    color_range: str | None
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    timecode: str | None
    video_stream_index: int
    audio: list[AudioStream] = field(default_factory=list)
    vfr: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fps"] = f"{self.fps.numerator}/{self.fps.denominator}"
        return d

    def frame_at(self, seconds: float) -> int:
        """Nearest frame index for a time in seconds."""
        return round(Fraction(seconds).limit_denominator(1_000_000) * self.fps)


def _rational(s: str | None) -> Fraction | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        f = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return f if f > 0 else None


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clean(v: str | None) -> str | None:
    return None if v in (None, "", "unknown", "N/A") else v


def _run_ffprobe(args: list[str]) -> dict:
    cmd = ["ffprobe", "-v", "error", "-of", "json", *args]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise ProbeError("ffprobe not found on PATH") from e
    if res.returncode != 0:
        raise ProbeError(f"ffprobe failed ({res.returncode}): {res.stderr.strip()}")
    return json.loads(res.stdout or "{}")


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise ProbeError(f"input file not found: {path}")

    data = _run_ffprobe(["-show_format", "-show_streams", str(path)])
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    videos = [
        s for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    if not videos:
        raise ProbeError(f"no video stream in {path}")
    v = videos[0]
    warnings: list[str] = []

    r_rate = _rational(v.get("r_frame_rate"))
    avg_rate = _rational(v.get("avg_frame_rate"))
    fps = avg_rate or r_rate
    if fps is None:
        raise ProbeError(f"could not determine frame rate of {path}")

    # Exact frame count: count packets of the video stream (demux only, no decode).
    counted = _run_ffprobe([
        "-select_streams", str(v["index"]),
        "-count_packets", "-show_entries", "stream=nb_read_packets", str(path),
    ])
    frame_count = _int((counted.get("streams") or [{}])[0].get("nb_read_packets"))
    header_frames = _int(v.get("nb_frames"))
    if frame_count is None:
        frame_count = header_frames
    if not frame_count:
        raise ProbeError(f"could not determine frame count of {path}")
    if header_frames and header_frames != frame_count:
        warnings.append(f"container says {header_frames} frames, counted {frame_count} packets; using {frame_count}")

    duration = _float(v.get("duration")) or _float(fmt.get("duration")) or float(frame_count / fps)

    # VFR heuristic: r_frame_rate (lowest common rate) and avg_frame_rate disagree,
    # or the counted frames don't match duration * fps.
    vfr = False
    if r_rate and avg_rate and abs(float(r_rate - avg_rate)) / float(avg_rate) > 0.001:
        vfr = True
        warnings.append(
            f"variable frame rate suspected (r_frame_rate={v.get('r_frame_rate')}, "
            f"avg_frame_rate={v.get('avg_frame_rate')}); it will be converted to CFR {fps} first"
        )
    expected = duration * float(fps)
    if not vfr and abs(expected - frame_count) > max(2.0, 0.01 * frame_count):
        warnings.append(
            f"frame count {frame_count} doesn't match duration {duration:.3f}s x {float(fps):.3f} fps "
            f"(~{expected:.1f}); timing may be irregular"
        )

    tags = {k.lower(): val for k, val in (v.get("tags") or {}).items()}
    ftags = {k.lower(): val for k, val in (fmt.get("tags") or {}).items()}
    timecode = tags.get("timecode") or ftags.get("timecode")
    if not timecode:
        for s in streams:  # QuickTime tmcd data track
            t = {k.lower(): val for k, val in (s.get("tags") or {}).items()}
            if t.get("timecode"):
                timecode = t["timecode"]
                break

    audio = [
        AudioStream(
            index=s["index"],
            codec=s.get("codec_name", "?"),
            channels=_int(s.get("channels")),
            sample_rate=_int(s.get("sample_rate")),
            language=(s.get("tags") or {}).get("language"),
        )
        for s in streams
        if s.get("codec_type") == "audio"
    ]

    width, height = int(v["width"]), int(v["height"])
    if width % 2 or height % 2:
        warnings.append(f"odd dimensions {width}x{height}: yuv420p chunk intermediates need even sizes")

    return VideoInfo(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        codec=v.get("codec_name", "?"),
        pix_fmt=v.get("pix_fmt", "?"),
        bit_depth=_int(v.get("bits_per_raw_sample")),
        color_range=_clean(v.get("color_range")),
        color_primaries=_clean(v.get("color_primaries")),
        color_transfer=_clean(v.get("color_transfer")),
        color_space=_clean(v.get("color_space")),
        timecode=timecode,
        video_stream_index=v["index"],
        audio=audio,
        vfr=vfr,
        warnings=warnings,
    )
