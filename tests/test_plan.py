import pytest

from vsr.plan import Chunk, FrameMultiple, overlaps, parse_frame_multiple, plan_chunks


def covered(chunks):
    return sorted({f for c in chunks for f in range(c.start, c.start + c.length)})


@pytest.mark.parametrize("spec,k,r", [("4n+1", 4, 1), ("8n", 8, 0), ("any", 1, 0), (None, 1, 0), (4, 4, 0), (" 16n + 1 ", 16, 1)])
def test_parse_frame_multiple(spec, k, r):
    assert parse_frame_multiple(spec) == FrameMultiple(k, r)


@pytest.mark.parametrize("bad", ["4x+1", "n+1", "foo", 0, -4, True])
def test_parse_frame_multiple_invalid(bad):
    with pytest.raises(ValueError):
        parse_frame_multiple(bad)


def test_frame_multiple_rounding():
    fm = parse_frame_multiple("4n+1")
    assert [fm.round_up(n) for n in (1, 2, 5, 6, 9)] == [1, 5, 5, 9, 9]
    assert [fm.round_down(n) for n in (1, 4, 5, 100)] == [1, 1, 5, 97]
    assert parse_frame_multiple("8n").round_down(7) is None


def test_basic_plan():
    chunks = plan_chunks(200, 97, 8, "4n+1")
    assert chunks == [Chunk(0, 97, 0), Chunk(89, 97, 0), Chunk(178, 22, 3)]
    assert overlaps(chunks) == [8, 8]
    assert covered(chunks) == list(range(200))


def test_shorter_than_one_chunk():
    assert plan_chunks(50, 97, 8, "4n+1") == [Chunk(0, 50, 3)]
    assert plan_chunks(1, 97, 8, "4n+1") == [Chunk(0, 1, 0)]
    assert plan_chunks(50, 97, 8, "any") == [Chunk(0, 50, 0)]


def test_exactly_one_chunk():
    assert plan_chunks(97, 97, 8, "4n+1") == [Chunk(0, 97, 0)]


def test_exact_multiple_no_tiny_tail():
    # 97 + 89 = 186 frames: two full chunks, no third chunk made only of overlap
    chunks = plan_chunks(186, 97, 8, "4n+1")
    assert chunks == [Chunk(0, 97, 0), Chunk(89, 97, 0)]


def test_one_frame_past_exact_multiple():
    chunks = plan_chunks(187, 97, 8, "4n+1")
    assert chunks[-1] == Chunk(178, 9, 0)  # 8 overlap + 1 new frame, 9 = 4*2+1
    assert covered(chunks) == list(range(187))


def test_frames_rounded_down_to_pattern():
    chunks = plan_chunks(300, 100, 8, "4n+1")  # 100 -> 97
    assert chunks[0] == Chunk(0, 97, 0)
    assert all(c.length <= 97 for c in chunks)


def test_padding_makes_valid_lengths():
    fm = parse_frame_multiple("4n+1")
    for total in range(1, 400):
        chunks = plan_chunks(total, 81, 8, fm)
        assert covered(chunks) == list(range(total))
        for c in chunks:
            assert fm.is_valid(c.padded_length)
            assert 0 <= c.pad < 4
        for c in chunks[1:]:
            assert c.length > 8  # every later chunk adds new frames
        assert all(o == 8 for o in overlaps(chunks))


def test_zero_overlap():
    chunks = plan_chunks(10, 4, 0, "any")
    assert chunks == [Chunk(0, 4), Chunk(4, 4), Chunk(8, 2)]


@pytest.mark.parametrize("total,frames,overlap,fm", [(0, 97, 8, "any"), (10, 8, 8, "any"), (10, 8, 6, "4n+1"), (10, 5, -1, "any"), (10, 7, 0, "8n")])
def test_invalid(total, frames, overlap, fm):
    with pytest.raises(ValueError):
        plan_chunks(total, frames, overlap, fm)


def test_chunk_must_hold_two_overlaps():
    with pytest.raises(ValueError, match="2 x overlap"):
        plan_chunks(100, 13, 8, "4n+1")
    assert plan_chunks(10, 13, 8, "4n+1") == [Chunk(0, 10, 3)]  # single chunk: no crossfade, fine
