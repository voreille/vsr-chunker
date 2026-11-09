import numpy as np
import pytest

from vsr.plan import Chunk, plan_chunks
from vsr.stitch import StitchError, blend, crossfade_weights, stitch_frames


def run(chunks, frame_lists):
    out = []
    n = stitch_frames(chunks, (iter(f) for f in frame_lists), out.append)
    assert n == len(out)
    return out


def source_frames(chunks, value_of):
    """Frames each chunk would get from a fake model: value_of(chunk_index, source_frame)."""
    lists = []
    for i, c in enumerate(chunks):
        idx = list(range(c.start, c.start + c.length)) + [c.start + c.length - 1] * c.pad
        lists.append([np.full(4, value_of(i, f), dtype=np.uint16) for f in idx])
    return lists


def test_weights():
    assert crossfade_weights(3) == [0.25, 0.5, 0.75]
    assert crossfade_weights(0) == []


def test_identity_reconstructs_source_exactly():
    chunks = plan_chunks(200, 33, 8, "4n+1")
    out = run(chunks, source_frames(chunks, lambda i, f: f))
    assert [int(x[0]) for x in out] == list(range(200))


def test_crossfade_values():
    # chunk i is a flat value 1000*i: overlaps must ramp linearly between them
    chunks = plan_chunks(30, 13, 4, "any")
    assert chunks == [Chunk(0, 13), Chunk(9, 13), Chunk(18, 12)]
    out = [int(x[0]) for x in run(chunks, source_frames(chunks, lambda i, f: 1000 * i))]
    assert len(out) == 30
    assert out[:9] == [0] * 9
    assert out[9:13] == [200, 400, 600, 800]   # w_in = 1/5 .. 4/5
    assert out[13:18] == [1000] * 5
    assert out[18:22] == [1200, 1400, 1600, 1800]
    assert out[22:30] == [2000] * 8


def test_pad_frames_dropped():
    chunks = plan_chunks(50, 97, 8, "4n+1")
    assert chunks[0].pad == 3
    out = run(chunks, source_frames(chunks, lambda i, f: f))
    assert len(out) == 50


def test_wrong_frame_count_is_an_error():
    chunks = plan_chunks(100, 41, 8, "4n+1")
    lists = source_frames(chunks, lambda i, f: f)
    lists[1] = lists[1][:-1]
    with pytest.raises(StitchError, match="chunk 1: model output has 40 frames, expected 41"):
        run(chunks, lists)


def test_size_mismatch_is_an_error():
    chunks = plan_chunks(60, 41, 8, "4n+1")
    lists = source_frames(chunks, lambda i, f: f)
    lists[1] = [np.zeros(8, np.uint16) for _ in lists[1]]
    with pytest.raises(StitchError, match="same output size"):
        run(chunks, lists)


def test_blend_rounding_and_dtype():
    a = np.array([0, 65535], np.uint16)
    b = np.array([65535, 0], np.uint16)
    r = blend(a, b, 0.5)
    assert r.dtype == np.uint16 and r.tolist() == [32768, 32768]
