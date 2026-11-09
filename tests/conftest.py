import shutil
import subprocess

import pytest


def make_clip(path, seconds=4, rate="25", size="320x240", audio=True, timecode=None, extra=()):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    cmd += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", *extra]
    if audio:
        cmd += ["-c:a", "aac"]
    if timecode:
        cmd += ["-timecode", timecode]
    cmd.append(str(path))
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture(scope="session")
def need_ffmpeg():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")


@pytest.fixture(scope="session")
def clip(tmp_path_factory, need_ffmpeg):
    return make_clip(
        tmp_path_factory.mktemp("media") / "clip.mp4", seconds=4, rate="25", timecode="10:00:00:00",
        extra=("-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"),
    )
