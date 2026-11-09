"""Estimate a source's real resolution (was it upscaled from something smaller?).

Method: for every even height h in a range, downscale sample frames (luma) to h and scale them
back up (Lanczos), and measure the round-trip PSNR. For a clip upscaled from h0, the round trip
at exactly h0 is much better than at its neighbours h0 +- 2 px: a sharp peak in the curve.
Native footage gives a smooth curve (small bumps < ~0.6 dB at simple ratios like 3/4).

Measured on real 720p/544p footage and controls upscaled from 600/540/480/432/360 (bicubic,
bilinear, Lanczos, h264 crf 20-28): every control's true height was the top peak; sharpness
1.1-2.1 dB (bilinear: 0.64-0.81, depending on the sampled frames); native footage <= 0.52 dB.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg import FFMPEG
from .probe import VideoInfo

# The crop avoids pixels at or above this luma (8-bit 225): burned-in subtitles and logos may be
# drawn at full resolution on top of an upscaled picture. The detection itself uses every pixel:
# masking them was tried and lost every control (their scaled text carried most of the signal),
# so subtitles burned in *after* an upscale can make a clip look native.
OVERLAY_LUMA = 225 * 257

LIKELY = 1.0  # sharpness (dB) above which the source was clearly upscaled from that height
POSSIBLE = 0.6


class DetectError(RuntimeError):
    pass


@dataclass
class Detection:
    width: int
    height: int
    frames: int
    heights: np.ndarray  # candidate heights, descending
    psnr: np.ndarray  # round-trip PSNR (dB) per height
    sharpness: np.ndarray  # how much each height stands above its +-4 px neighbours (dB)

    def peaks(self, n: int = 3) -> list[tuple[int, float]]:
        idx = np.argsort(self.sharpness)[::-1][:n]
        return [(int(self.heights[i]), float(self.sharpness[i])) for i in idx]

    @property
    def best(self) -> tuple[int, float]:
        return self.peaks(1)[0]

    @property
    def verdict(self) -> str:
        s = self.best[1]
        return "upscaled" if s >= LIKELY else "probably upscaled" if s >= POSSIBLE else "no sign of upscaling"


def sample_frames(info: VideoInfo, n: int, start_s: float = 0.0,
                  end_s: float | None = None) -> tuple[np.ndarray, list[float]]:
    """``n`` luma frames evenly spread over [start_s, end_s), as uint16 (H, W) arrays, and their times."""
    end_s = info.duration if end_s is None else min(end_s, info.duration)
    w, h = info.width, info.height
    out, times = [], []
    for k in range(n):
        t = start_s + (end_s - start_s) * (k + 0.5) / n
        raw = subprocess.run(
            [*FFMPEG, "-ss", f"{t:.3f}", "-i", info.path, "-map", "0:v:0", "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
            capture_output=True,
        ).stdout
        if len(raw) == w * h * 2:
            out.append(np.frombuffer(raw, np.uint16).reshape(h, w))
            times.append(t)
    if not out:
        raise DetectError(f"could not decode any frame from {info.path}")
    return np.stack(out), times


def roundtrip(frames: np.ndarray, h: int) -> np.ndarray:
    """Downscale to height ``h`` (width from the aspect ratio, even) and back, Lanczos."""
    n, H, W = frames.shape
    w = max(2, round(W * h / H / 2) * 2)
    cmd = [*FFMPEG, "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{W}x{H}", "-i", "-",
           "-vf", f"scale={w}:{h}:flags=lanczos+accurate_rnd,scale={W}:{H}:flags=lanczos+accurate_rnd",
           "-f", "rawvideo", "-pix_fmt", "gray16le", "-"]
    res = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if res.returncode != 0 or len(res.stdout) != frames.nbytes:
        raise DetectError(f"round trip at height {h} failed: {res.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(res.stdout, np.uint16).reshape(frames.shape)


def overlay_free(frames: np.ndarray, grow: int = 6) -> np.ndarray:
    """Mask of pixels away (``grow`` px) from near-white ones (subtitles, logos)."""
    bright = frames >= OVERLAY_LUMA
    near = bright.copy()
    for ax in (1, 2):
        for d in range(1, grow + 1):
            near |= np.roll(bright, d, axis=ax) | np.roll(bright, -d, axis=ax)
    if (~near).mean() < 0.3:  # mostly bright content (snow, sky): keep everything
        return np.ones_like(bright)
    return ~near


def detect(frames: np.ndarray, lo: float = 0.3, hi: float = 0.92, workers: int = 8) -> Detection:
    n, H, W = frames.shape
    ref = frames.astype(np.float64)
    if ref.std() < 257:  # < 1 level of 8-bit contrast overall
        raise DetectError("sample frames are almost flat (black?); try another --range")
    heights = np.arange(int(H * hi) // 2 * 2, max(16, int(H * lo)) - 1, -2)
    if len(heights) < 12:
        raise DetectError(f"source height {H} is too small to analyse")
    peak = 65535.0 ** 2

    def psnr(h: int) -> float:
        mse = np.mean((roundtrip(frames, int(h)).astype(np.float64) - ref) ** 2)
        return 10 * np.log10(peak / max(mse, 1e-9))

    with ThreadPoolExecutor(workers) as ex:
        db = np.array(list(ex.map(psnr, heights)))
    # residual against the local trend (median of +-4..12 px), then how much it beats its +-4 px neighbours
    resid = np.array([db[i] - np.median(np.r_[db[max(0, i - 6):max(0, i - 1)], db[i + 2:i + 7]])
                      for i in range(len(heights))])
    sharp = np.array([resid[i] - max(resid[max(0, i - 2):i].tolist() + resid[i + 1:i + 3].tolist())
                      for i in range(len(heights))])
    sharp[[0, -1]] = 0.0  # edges have one-sided neighbourhoods
    return Detection(W, H, n, heights, db, sharp)


# --- visual check ------------------------------------------------------------------

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def detail_window(frames: np.ndarray, cw: int, ch: int) -> tuple[int, int, int]:
    """(frame index, x, y) of the ``cw x ch`` window with the most fine detail (gradient energy)."""
    best = (-1.0, 0, 0, 0)
    keep = overlay_free(frames)
    for i, f in enumerate(frames.astype(np.float32)):
        g = np.abs(np.diff(f, axis=0))[:, :-1] + np.abs(np.diff(f, axis=1))[:-1, :]
        g *= keep[i][:-1, :-1]  # not the subtitles
        ii = np.pad(g.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        H, W = g.shape
        ys, xs = np.arange(0, H - ch, 8), np.arange(0, W - cw, 8)
        if not len(ys) or not len(xs):
            return 0, 0, 0
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        e = ii[Y + ch, X + cw] - ii[Y, X + cw] - ii[Y + ch, X] + ii[Y, X]
        k = np.unravel_index(np.argmax(e), e.shape)
        if e[k] > best[0]:
            best = (float(e[k]), i, int(xs[k[1]]), int(ys[k[0]]))
    _, i, x, y = best
    return i, x - x % 2, y - y % 2


def crop_heights(d: Detection) -> list[int]:
    """Heights to show: the best peak if it scored, plus a spread of common fractions."""
    H = d.height
    h, s = d.best
    hs = [h, int(round(h * 0.85 / 2)) * 2] if s >= POSSIBLE else []
    for f in (0.9, 0.75, 0.667, 0.6, 0.5):
        v = int(round(H * f / 2)) * 2
        if len(hs) < 5 and all(abs(v - x) > H * 0.04 for x in hs):
            hs.append(v)
    return sorted(hs, reverse=True)


def crops_command(info: VideoInfo, t: float, x: int, y: int, cw: int, ch: int, heights: list[int],
                  dst: str, zoom: int = 2) -> list[str]:
    """One PNG: the source crop, then the same crop after a round trip through each height,
    each enlarged ``zoom`` x (nearest neighbour) and labelled, in a 3-column grid."""
    W, H = info.width, info.height
    n = len(heights) + 1
    label = Path(FONT).is_file()
    parts = [f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n))]
    for i in range(n):
        chain = "" if i == 0 else (f"scale={max(2, round(W * heights[i - 1] / H / 2) * 2)}:{heights[i - 1]}:"
                                   f"flags=lanczos+accurate_rnd,scale={W}:{H}:flags=lanczos+accurate_rnd,")
        text = f"source {H}p" if i == 0 else f"via {heights[i - 1]}p"
        chain += f"crop={cw}:{ch}:{x}:{y},scale=iw*{zoom}:ih*{zoom}:flags=neighbor"
        if label:
            chain += (f",drawtext=fontfile={FONT}:text='{text}':x=8:y=8:fontsize=20:fontcolor=white:"
                      "box=1:boxcolor=black@0.6:boxborderw=4")
        parts.append(f"[s{i}]{chain}[c{i}]")
    cols = 3
    while n % cols:  # pad the grid with black tiles
        parts.append(f"color=black:s={cw * zoom}x{ch * zoom}:d=1[c{n}]")
        n += 1
    tw, th = cw * zoom, ch * zoom
    layout = "|".join(f"{(i % cols) * tw}_{(i // cols) * th}" for i in range(n))
    parts.append("".join(f"[c{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={layout}[out]")
    return [*FFMPEG, "-ss", f"{t:.3f}", "-i", info.path, "-filter_complex", ";".join(parts),
            "-map", "[out]", "-frames:v", "1", "-update", "1", str(dst)]


def write_crops(info: VideoInfo, frames: np.ndarray, times: list[float], d: Detection, dst: str) -> list[int]:
    """Write the comparison PNG; returns the heights shown."""
    W, H = info.width, info.height
    cw, ch = max(64, W // 5 // 2 * 2), max(36, H // 5 // 2 * 2)
    i, x, y = detail_window(frames, cw, ch)
    heights = crop_heights(d)
    res = subprocess.run(crops_command(info, times[i], x, y, cw, ch, heights, dst), capture_output=True)
    if res.returncode != 0 or not Path(dst).is_file():
        raise DetectError(f"writing {dst} failed: {res.stderr.decode(errors='replace')[-400:]}")
    return heights
