"""Crossfade-stitch processed chunks, streaming one frame at a time."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from .plan import Chunk, overlaps


class StitchError(RuntimeError):
    pass


def crossfade_weights(overlap: int) -> list[float]:
    """Weight of the *incoming* chunk at each overlap frame, strictly between 0 and 1,
    rising linearly (the outgoing chunk gets 1 - w, falling from ~1 to ~0)."""
    return [(j + 1) / (overlap + 1) for j in range(overlap)]


def blend(outgoing: np.ndarray, incoming: np.ndarray, w_in: float) -> np.ndarray:
    out = outgoing.astype(np.float32) * (1.0 - w_in) + incoming.astype(np.float32) * w_in
    info = np.iinfo(outgoing.dtype) if np.issubdtype(outgoing.dtype, np.integer) else None
    if info:
        return np.clip(np.rint(out), info.min, info.max).astype(outgoing.dtype)
    return out.astype(outgoing.dtype)


def stitch_frames(
    chunks: list[Chunk],
    readers: Iterable[Iterator[np.ndarray]],
    write: Callable[[np.ndarray], None],
) -> int:
    """Core stitching logic. ``readers`` yields one frame iterator per chunk (opened lazily,
    in order); each must produce exactly ``chunk.padded_length`` frames. Pad frames are
    dropped. Holds at most ``overlap`` frames in memory. Returns frames written."""
    ovs = overlaps(chunks)
    tail: list[np.ndarray] = []  # last `overlap` frames of the previous chunk, not yet written
    written = 0
    shape = None
    readers = iter(readers)
    for i, chunk in enumerate(chunks):
        try:
            frames = next(readers)
        except StopIteration:
            raise StitchError(f"no frames source for chunk {i}") from None
        ov_prev = ovs[i - 1] if i > 0 else 0
        ov_next = ovs[i] if i < len(ovs) else 0
        weights = crossfade_weights(ov_prev)
        new_tail: list[np.ndarray] = []
        n = 0
        for j, frame in enumerate(frames):
            n += 1
            if shape is None:
                shape = (frame.shape, frame.dtype)
            elif (frame.shape, frame.dtype) != shape:
                raise StitchError(
                    f"chunk {i} frame {j}: shape {frame.shape} {frame.dtype} differs from "
                    f"chunk 0 ({shape[0]} {shape[1]}); all chunks must have the same output size"
                )
            if j >= chunk.length:
                continue  # padding added for frame_multiple, trimmed here
            if j < ov_prev:
                write(blend(tail[j], frame, weights[j]))
                written += 1
            elif j >= chunk.length - ov_next:
                new_tail.append(frame)
            else:
                write(frame)
                written += 1
        if n != chunk.padded_length:
            raise StitchError(
                f"chunk {i}: model output has {n} frames, expected {chunk.padded_length} "
                f"({chunk.length} + {chunk.pad} pad)"
            )
        tail = new_tail
    if tail:
        raise StitchError("internal error: frames left in the overlap buffer")
    return written


# --- ffmpeg-backed I/O ----------------------------------------------------------

def read_frames(cmd: list[str], samples: int, log: Path | None = None) -> Iterator[np.ndarray]:
    """Run a decoder ``cmd`` writing 16-bit raw frames to stdout; yield flat uint16 frames
    of ``samples`` values each."""
    frame_bytes = samples * 2
    errf = open(log, "ab") if log else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf:
                break
            if len(buf) != frame_bytes:
                raise StitchError(f"truncated frame from decoder ({len(buf)} of {frame_bytes} bytes): {cmd[-1]}")
            yield np.frombuffer(buf, dtype=np.uint16)
        if proc.wait() != 0:
            raise StitchError(f"decoder failed (exit {proc.returncode}): {' '.join(cmd)}" + (f"; see {log}" if log else ""))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if log:
            errf.close()


class FrameWriter:
    """Feeds raw frames to an encoder process on stdin."""

    def __init__(self, cmd: list[str], log: Path):
        self.cmd = cmd
        self.log = log
        self._errf = open(log, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._errf)

    def __call__(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).data)
        except BrokenPipeError:
            self.proc.wait()
            raise StitchError(f"encoder exited early (exit {self.proc.returncode}); see {self.log}") from None

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = self.proc.wait()
        self._errf.close()
        if rc != 0:
            raise StitchError(f"encoder failed (exit {rc}); see {self.log}")

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self._errf.close()
