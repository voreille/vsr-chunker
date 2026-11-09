"""Chunk math. Pure functions, no I/O."""

from __future__ import annotations

import re
from typing import NamedTuple


class Chunk(NamedTuple):
    start: int  # first source frame (relative to the processed range)
    length: int  # real frames taken from the source
    pad: int = 0  # extra frames appended (last frame repeated) to satisfy frame_multiple

    @property
    def end(self) -> int:
        """Last source frame, inclusive."""
        return self.start + self.length - 1

    @property
    def padded_length(self) -> int:
        return self.length + self.pad


class FrameMultiple(NamedTuple):
    """Valid lengths are ``k*n + r`` with n >= 0 and the length >= 1. ``k == 1`` means any."""

    k: int
    r: int

    def is_valid(self, length: int) -> bool:
        return length >= 1 and length >= self.r and (length - self.r) % self.k == 0

    def round_up(self, length: int) -> int:
        n = max(length, self.r, 1)
        while not self.is_valid(n):
            n += 1
        return n

    def round_down(self, length: int) -> int | None:
        n = length
        while n >= 1 and not self.is_valid(n):
            n -= 1
        return n if n >= 1 else None

    def __str__(self) -> str:
        if self.k == 1:
            return "any"
        return f"{self.k}n" + (f"+{self.r}" if self.r else "")


_FM_RE = re.compile(r"^\s*(\d+)\s*n\s*(?:\+\s*(\d+))?\s*$")


def parse_frame_multiple(spec: str | int | None) -> FrameMultiple:
    """Parse ``"any"``, ``"4n+1"``, ``"8n"``, or an int ``k`` (same as ``"kn"``)."""
    if spec is None:
        return FrameMultiple(1, 0)
    if isinstance(spec, bool):
        raise ValueError(f"invalid frame_multiple: {spec!r}")
    if isinstance(spec, int):
        if spec < 1:
            raise ValueError(f"frame_multiple must be >= 1, got {spec}")
        return FrameMultiple(spec, 0)
    s = str(spec).strip().lower()
    if s in ("any", "", "1", "1n"):
        return FrameMultiple(1, 0)
    m = _FM_RE.match(s)
    if not m:
        raise ValueError(f"invalid frame_multiple {spec!r}: expected 'any', 'Kn' or 'Kn+R' (e.g. '4n+1')")
    k = int(m.group(1))
    r = int(m.group(2) or 0)
    if k < 1:
        raise ValueError(f"invalid frame_multiple {spec!r}: K must be >= 1")
    return FrameMultiple(k, r)


def plan_chunks(
    total_frames: int,
    frames: int,
    overlap: int,
    frame_multiple: str | int | FrameMultiple | None = "any",
) -> list[Chunk]:
    """Split ``total_frames`` into overlapping chunks.

    * ``frames`` is rounded *down* to the nearest valid length for ``frame_multiple``
      (down, so the configured value stays a safe upper bound for VRAM).
    * Chunk i starts at ``i * (frames - overlap)``.
    * The last chunk may be shorter. If its length isn't valid, it gets ``pad`` extra
      frames (the tool repeats the last frame, then trims them after processing).
    * A trailing chunk is only emitted if it contributes at least one new frame, so
      every chunk after the first has ``length > overlap``.
    """
    fm = frame_multiple if isinstance(frame_multiple, FrameMultiple) else parse_frame_multiple(frame_multiple)
    if total_frames < 1:
        raise ValueError(f"total_frames must be >= 1, got {total_frames}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    chunk_len = fm.round_down(frames)
    if chunk_len is None:
        raise ValueError(f"frames={frames} has no valid length for frame_multiple {fm}")
    if chunk_len <= overlap:
        raise ValueError(
            f"chunk length {chunk_len} (frames={frames} rounded to {fm}) must be greater than overlap={overlap}"
        )
    if total_frames > chunk_len and chunk_len < 2 * overlap:
        raise ValueError(
            f"chunk length {chunk_len} must be at least 2 x overlap ({2 * overlap}) so that a chunk's "
            "two crossfade regions don't overlap each other"
        )
    stride = chunk_len - overlap

    chunks: list[Chunk] = []
    start = 0
    while True:
        length = min(chunk_len, total_frames - start)
        chunks.append(Chunk(start, length, fm.round_up(length) - length))
        if start + length >= total_frames:
            break
        start += stride
    return chunks


def overlaps(chunks: list[Chunk]) -> list[int]:
    """Number of frames shared between chunk i and chunk i+1."""
    return [a.start + a.length - b.start for a, b in zip(chunks, chunks[1:])]


def delay_extra(padded_length: int, delay: int, frame_multiple: FrameMultiple) -> int:
    """Extra input frames to feed a model whose output lags its input by ``delay`` frames,
    so that after dropping the first ``delay`` output frames every chunk frame is still there.
    Keeps the input length valid for ``frame_multiple``."""
    if delay <= 0:
        return 0
    return frame_multiple.round_up(padded_length + delay) - padded_length
