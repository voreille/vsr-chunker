"""ffmpeg command builders (pure: they return argv lists, they don't run anything)."""

from __future__ import annotations

import functools
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .probe import VideoInfo

FFMPEG = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> tuple[int, int]:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return (0, 0)
    m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 0)  # git builds: assume recent


def vsync(mode: str) -> list[str]:
    """-fps_mode (ffmpeg >= 5.1) or the older -vsync spelling (ffmpeg 4.x, Ubuntu 22.04)."""
    return ["-fps_mode", mode] if ffmpeg_version() >= (5, 1) else ["-vsync", mode]


@dataclass
class ChunkCodec:
    """Lossless intermediate used for chunk files sent to ComfyUI."""

    ext: str = ".mp4"
    args: tuple[str, ...] = ("-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p")


DEFAULT_CHUNK_CODEC = ChunkCodec()


def fps_str(fps: Fraction) -> str:
    return f"{fps.numerator}/{fps.denominator}"


STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                  Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60)]


def cfr_target_fps(avg: Fraction) -> Fraction:
    """Constant rate for a VFR source: the nearest standard rate within 2%, else ``avg``
    rounded to 3 decimals."""
    best = min(STANDARD_RATES, key=lambda r: abs(r - avg))
    if abs(best - avg) / avg <= Fraction(2, 100):
        return best
    return Fraction(round(avg * 1000), 1000)


def cfr_command(src: str | Path, dst: str | Path, fps: Fraction, codec: ChunkCodec = DEFAULT_CHUNK_CODEC) -> list[str]:
    """Convert a VFR source to constant frame rate (video only, lossless intermediate)."""
    return [
        *FFMPEG, "-i", str(src), "-map", "0:v:0", "-map_metadata", "-1", "-an", "-sn", "-dn",
        *vsync("cfr"), "-r", fps_str(fps), *codec.args, str(dst),
    ]


def prescale_filter(w: int, h: int) -> str:
    """Antialiased downscale (swscale widens the Lanczos kernel by the scale factor)."""
    return f"scale={w}:{h}:flags=lanczos+accurate_rnd+full_chroma_int"


def cut_command(
    src: str | Path,
    dst: str | Path,
    start: int,
    length: int,
    pad: int = 0,
    codec: ChunkCodec = DEFAULT_CHUNK_CODEC,
    prescale: tuple[int, int] | None = None,
) -> list[str]:
    """Frame-accurate cut of frames [start, start+length) by frame number, plus ``pad``
    copies of the last frame, optionally downscaled to ``prescale`` (w, h) first."""
    vf = f"trim=start_frame={start}:end_frame={start + length},setpts=PTS-STARTPTS"
    if prescale:
        vf += f",{prescale_filter(*prescale)}"
    if pad:
        vf += f",tpad=stop_mode=clone:stop={pad}"
    return [
        *FFMPEG, "-i", str(src), "-map", "0:v:0", "-map_metadata", "-1", "-an", "-sn", "-dn",
        "-vf", vf, *vsync("passthrough"), "-frames:v", str(length + pad),
        *codec.args, str(dst),
    ]


# --- decode for stitching ------------------------------------------------------

# Planar 4:2:2 16-bit: every master codec is 4:2:2, so the only chroma resampling happens when
# decoding the model output (blending in YUV needs no matrix conversion; it's linear either way).
STITCH_PIX_FMT = "yuv422p16le"


def stitch_frame_samples(width: int, height: int) -> int:
    """uint16 samples per STITCH_PIX_FMT frame (Y + two half-width chroma planes)."""
    return width * height + 2 * ((width + 1) // 2) * height

_MATRIX = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "fcc": "fcc",
           "smpte240m": "smpte240m", "bt2020nc": "bt2020", "bt2020c": "bt2020"}


def decode_command(src: str | Path, info: VideoInfo | None = None, pix_fmt: str = STITCH_PIX_FMT) -> list[str]:
    """Decode a video to raw frames on stdout. If the source tags are known, pin the YUV
    matrix/range so an RGB model output is converted back with the source's matrix
    (swscale would otherwise default to bt601)."""
    vf = []
    if info is not None:
        opts = []
        if info.color_space in _MATRIX:
            opts.append(f"out_color_matrix={_MATRIX[info.color_space]}")
        if info.color_range in ("tv", "pc"):
            opts.append(f"out_range={info.color_range}")
        if opts:
            vf = ["-vf", "scale=" + ":".join(opts)]
    return ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(src), "-map", "0:v:0",
            *vf, *vsync("passthrough"), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]


# --- fake models (tests without ComfyUI) -------------------------------------

def fake_model_command(kind: str, src: str | Path, dst: str | Path, codec: ChunkCodec = DEFAULT_CHUNK_CODEC,
                       frames: int | None = None) -> list[str]:
    if kind == "scale2":
        return [*FFMPEG, "-i", str(src), "-vf", "scale=iw*2:ih*2:flags=lanczos", *vsync("passthrough"),
                *codec.args, str(dst)]
    if kind.startswith("delay") and kind[5:].isdigit() and frames is not None:
        # same frame count, shifted late by k frames (first frame repeated), like the FlashVSR port
        k = int(kind[5:])
        return [*FFMPEG, "-i", str(src), "-vf", f"tpad=start={k}:start_mode=clone", *vsync("passthrough"),
                "-frames:v", str(frames), *codec.args, str(dst)]
    raise ValueError(f"unknown fake model {kind!r}")


# --- master -----------------------------------------------------------------

MASTER_CODECS = {
    # (codec, 10bit) -> encoder args
    ("dnxhr", False): ["-c:v", "dnxhd", "-profile:v", "dnxhr_hq", "-pix_fmt", "yuv422p"],
    ("dnxhr", True): ["-c:v", "dnxhd", "-profile:v", "dnxhr_hqx", "-pix_fmt", "yuv422p10le"],
    ("prores", False): ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
    ("prores", True): ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
}

# Audio codecs a .mov can carry as-is.
MOV_AUDIO_COPY = {"aac", "alac", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_s16be", "pcm_s24be", "pcm_f32le"}

_TC_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})([:;.])(\d{2})$")


def offset_timecode(tc: str, frames: int, fps: Fraction) -> tuple[str, str | None]:
    """Add ``frames`` to a non-drop-frame timecode. Returns (timecode, warning)."""
    if frames == 0:
        return tc, None
    m = _TC_RE.match(tc)
    if not m:
        return tc, f"unrecognized timecode {tc!r}; copied unchanged"
    hh, mm, ss, sep, ff = m.groups()
    if sep in ";.":
        return tc, f"drop-frame timecode {tc} not offset for --range; copied unchanged"
    base = round(fps)
    total = ((int(hh) * 60 + int(mm)) * 60 + int(ss)) * base + int(ff) + frames
    ff2 = total % base
    s = total // base
    return f"{(s // 3600) % 24:02d}:{(s // 60) % 60:02d}:{s % 60:02d}:{ff2:02d}", None


def color_args(info: VideoInfo) -> list[str]:
    args = []
    for flag, val in (
        ("-color_range", info.color_range),
        ("-color_primaries", info.color_primaries),
        ("-color_trc", info.color_transfer),
        ("-colorspace", info.color_space),
    ):
        if val:
            args += [flag, val]
    return args


def master_output_name(info: VideoInfo, profile_name: str, width: int | None, height: int | None) -> str:
    size = f"{width}x{height}" if width and height else "WxH"
    return f"{Path(info.path).stem}_{profile_name}_{size}.mov"


def master_command(
    video: str | Path,
    info: VideoInfo,
    dst: str | Path,
    *,
    codec: str = "dnxhr",
    ten_bit: bool = False,
    range_start_frame: int = 0,
    range_frames: int | None = None,
    raw: tuple[int, int, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Encode the stitched ``video`` and mux the source's audio + timecode + color tags.

    With ``raw=(width, height, pix_fmt)``, the video is read as rawvideo from stdin
    (``video`` is ignored), at the source frame rate.

    Returns (argv, warnings)."""
    warnings: list[str] = []
    try:
        vargs = MASTER_CODECS[(codec, ten_bit)]
    except KeyError:
        raise ValueError(f"unknown master codec {codec!r} (use dnxhr or prores)") from None

    if raw:
        w, h, pix = raw
        cmd = [*FFMPEG, "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", fps_str(info.fps), "-i", "-"]
    else:
        cmd = [*FFMPEG, "-i", str(video)]
    trimmed = range_start_frame != 0 or (range_frames is not None and range_frames != info.frame_count)
    if trimmed:
        start_s = Fraction(range_start_frame) / info.fps
        cmd += ["-ss", f"{float(start_s):.6f}"]
        if range_frames is not None:
            cmd += ["-t", f"{float(Fraction(range_frames) / info.fps):.6f}"]
    cmd += ["-i", info.path, "-map", "0:v:0", "-map", "1:a?"]
    cmd += vargs + color_args(info)

    if info.audio:
        codecs = {a.codec for a in info.audio}
        if not trimmed and codecs <= MOV_AUDIO_COPY:
            cmd += ["-c:a", "copy"]
        elif codecs & {"pcm_s24le", "pcm_s24be", "pcm_s32le", "pcm_f32le"}:
            cmd += ["-c:a", "pcm_s24le"]
        else:
            cmd += ["-c:a", "pcm_s16le"]

    if info.timecode:
        tc, w = offset_timecode(info.timecode, range_start_frame, info.fps)
        if w:
            warnings.append(w)
        cmd += ["-timecode", tc]

    # No -shortest: it could drop video frames, and the frame count must match the source.
    cmd.append(str(dst))
    return cmd, warnings
