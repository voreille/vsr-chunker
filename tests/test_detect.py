"""Real-resolution detection on synthetic clips (noisy testsrc2, optionally pre-upscaled)."""

import subprocess

import pytest

from vsr.cli import main
from vsr.detect import detect, sample_frames
from vsr.probe import probe


def _clip(path, vf=""):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=4",
                    "-vf", "noise=alls=6:allf=t" + (f",{vf}" if vf else ""),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.mark.parametrize("vf,expect,verdict", [
    ("", None, "no sign of upscaling"),
    ("scale=-2:160:flags=bicubic,scale=320:240:flags=bicubic", 160, "upscaled"),
    ("scale=-2:180:flags=bilinear,scale=320:240:flags=bilinear", 180, "probably upscaled"),
])
def test_detect(tmp_path, need_ffmpeg, vf, expect, verdict):
    d = detect(sample_frames(probe(_clip(tmp_path / "c.mp4", vf)), 4)[0])
    assert d.verdict == verdict
    if expect:
        assert d.best[0] == expect


def test_detect_res_cli(tmp_path, need_ffmpeg, capsys):
    src = _clip(tmp_path / "c.mp4", "scale=-2:160:flags=bicubic,scale=320:240:flags=bicubic")
    assert main(["detect-res", str(src), "--frames", "4", "-v"]) == 0
    out = capsys.readouterr().out
    assert "looks upscaled from 160 lines" in out and "--prescale 160" in out
    assert "round-trip PSNR" in out
    png = tmp_path / "c_detect-res.png"
    assert f"crops      {png}" in out and "160p" in out
    from vsr.probe import probe as p2
    sheet = p2(png)
    assert (sheet.width, sheet.height) == (3 * 64 * 2, 2 * 48 * 2)  # 3x2 tiles of 64x48 crops at 2x
    assert main(["detect-res", str(src), "--frames", "4", "--no-crops", "--crops", str(tmp_path / "x.png")]) == 0
    assert not (tmp_path / "x.png").exists()
