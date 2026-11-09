from fractions import Fraction

import pytest

from conftest import make_clip
from vsr.probe import ProbeError, probe


def test_probe_clip(clip):
    info = probe(clip)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == Fraction(25)
    assert info.frame_count == 100
    assert info.pix_fmt == "yuv420p"
    assert info.color_primaries == "bt709" and info.color_space == "bt709"
    assert info.timecode == "10:00:00:00"
    assert len(info.audio) == 1 and info.audio[0].codec == "aac"
    assert not info.vfr
    assert info.frame_at(1.0) == 25


def test_probe_ntsc_no_audio(tmp_path, need_ffmpeg):
    info = probe(make_clip(tmp_path / "ntsc.mkv", seconds=2, rate="30000/1001", audio=False))
    assert info.fps == Fraction(30000, 1001)
    assert info.frame_count == 60
    assert info.audio == [] and info.timecode is None


def test_probe_missing(tmp_path):
    with pytest.raises(ProbeError, match="not found"):
        probe(tmp_path / "nope.mp4")


def test_probe_not_a_video(tmp_path, need_ffmpeg):
    p = tmp_path / "x.mp4"
    p.write_text("hello")
    with pytest.raises(ProbeError):
        probe(p)
