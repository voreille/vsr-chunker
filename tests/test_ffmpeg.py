import subprocess
from fractions import Fraction

from vsr import ffmpeg as ff
from vsr.probe import probe


def test_offset_timecode():
    assert ff.offset_timecode("01:00:00:00", 60, Fraction(30000, 1001))[0] == "01:00:02:00"
    assert ff.offset_timecode("00:59:59:24", 1, Fraction(25))[0] == "01:00:00:00"
    tc, warn = ff.offset_timecode("01:00:00;00", 10, Fraction(30000, 1001))
    assert tc == "01:00:00;00" and "drop-frame" in warn


def test_cut_is_frame_exact_with_padding(clip, tmp_path):
    out = tmp_path / "c.mp4"
    subprocess.run(ff.cut_command(clip, out, 90, 10, 3), check=True, capture_output=True)
    info = probe(out)
    assert info.frame_count == 13 and info.audio == [] and info.timecode is None

    def frame(path, n):
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"trim=start_frame={n}:end_frame={n + 1}",
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
            check=True, capture_output=True).stdout

    assert frame(out, 0) == frame(clip, 90)
    assert frame(out, 9) == frame(clip, 99)
    assert frame(out, 12) == frame(clip, 99)  # padding repeats the last frame


def test_master_command(clip):
    info = probe(clip)
    cmd, warns = ff.master_command("st.mkv", info, "m.mov", range_start_frame=25, range_frames=50)
    s = " ".join(cmd)
    assert "-profile:v dnxhr_hq" in s and "-ss 1.000000 -t 2.000000" in s
    assert "-timecode 10:00:01:00" in s and "-colorspace bt709" in s and "-c:a pcm_s16le" in s
    cmd, _ = ff.master_command("st.mkv", info, "m.mov", codec="prores", ten_bit=True)
    s = " ".join(cmd)
    assert "prores_ks" in s and "-c:a copy" in s and "-ss" not in s
