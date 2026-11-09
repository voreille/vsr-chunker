"""End-to-end cut -> fake model -> stitch -> master, no GPU / ComfyUI."""

import subprocess

import numpy as np
import pytest

from conftest import make_clip
from vsr import cli
from vsr import ffmpeg as ff
from vsr.pipeline import PipelineError, run_job
from vsr.probe import probe
from vsr.stitch import crossfade_weights
from vsr.plan import overlaps


def run_cli(*argv):
    assert cli.main(["run", *map(str, argv)]) == 0


def test_identity_full_clip(clip, tmp_path):
    run_cli(clip, "--profile", "seedvr2_3b", "--fake", "identity", "--frames", "41", "--overlap", "8",
            "--out", tmp_path, "--jobs-dir", tmp_path / "jobs")
    src = probe(clip)
    m = probe(tmp_path / "clip_seedvr2_3b_320x240.mov")
    assert m.frame_count == src.frame_count == 100
    assert abs(m.duration - src.duration) < 1e-3
    assert m.codec == "dnxhd" and m.pix_fmt == "yuv422p"
    assert m.timecode == "10:00:00:00"
    assert m.color_primaries == "bt709" and m.color_space == "bt709"
    assert len(m.audio) == 1


def test_scale2_with_range(clip, tmp_path):
    run_cli(clip, "--profile", "seedvr2_3b", "--fake", "scale2", "--range", "1-3.5", "--frames", "21",
            "--overlap", "4", "--out", tmp_path, "--jobs-dir", tmp_path / "jobs", "--codec", "prores")
    m = probe(tmp_path / "clip_seedvr2_3b_640x480.mov")
    assert (m.width, m.height) == (640, 480)
    assert m.frame_count == 63  # frames 25..87 at 25 fps
    assert m.codec == "prores" and m.timecode == "10:00:01:00"
    assert abs(m.audio and m.duration - 63 / 25) < 0.01


def test_vfr_source(tmp_path, need_ffmpeg):
    src = make_clip(tmp_path / "vfr.mp4", seconds=4, rate="30", audio=True,
                    extra=("-vf", "select='not(eq(mod(n,3),2))'", "-vsync", "vfr"))
    assert probe(src).vfr
    run_cli(src, "--profile", "seedvr2_3b", "--fake", "identity", "--out", tmp_path, "--jobs-dir", tmp_path / "jobs")
    cfr = next((tmp_path / "jobs").glob("*/source_cfr.mp4"))
    m = probe(next(tmp_path.glob("vfr_*.mov")))
    assert m.frame_count == probe(cfr).frame_count


def _flat_processor(values, fps):
    """Fake model: chunk i becomes a flat gray of luma values[i] (same frame count/size)."""
    def proc(i, src, dst):
        info = probe(src)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"lutyuv=y={values[i]}:u=128:v=128",
             "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(dst)], check=True)
        assert probe(dst).frame_count == info.frame_count
        return dst
    return proc


def test_crossfade_lands_on_the_right_frames(clip, tmp_path):
    args = cli.build_parser().parse_args(
        ["run", str(clip), "--profile", "seedvr2_3b", "--frames", "33", "--overlap", "8", "--fake", "identity"])
    plan = cli.build_plan(args)
    values = [40, 200, 80, 160][: len(plan.chunks)]
    master = run_job(plan, job_dir=tmp_path / "job", master_path=tmp_path / "m.mov",
                     processor=_flat_processor(values, plan.info.fps))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(master), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         check=True, capture_output=True).stdout  # not "gray": that would expand tv range
    y = np.frombuffer(raw, np.uint8).reshape(-1, 240 * 320 * 3 // 2)[:, : 240 * 320].mean(axis=1)

    expected = np.zeros(plan.range_frames)
    ovs = overlaps(plan.chunks)
    for i, c in enumerate(plan.chunks):
        expected[c.start:c.start + c.length] = values[i]
        if i:
            for j, w in enumerate(crossfade_weights(ovs[i - 1])):
                expected[c.start + j] = values[i - 1] * (1 - w) + values[i] * w
    assert len(y) == len(expected) == 100
    assert np.abs(y - expected).max() < 1.5, np.round(y - expected, 1)


def test_model_frame_mismatch_is_reported(clip, tmp_path):
    args = cli.build_parser().parse_args(["run", str(clip), "--profile", "seedvr2_3b", "--frames", "41"])
    plan = cli.build_plan(args)

    def drop_one(i, src, dst):
        n = probe(src).frame_count
        subprocess.run([*ff.FFMPEG, "-v", "error", "-i", str(src), "-frames:v", str(n - 1), str(dst)], check=True)
        return dst

    with pytest.raises(PipelineError, match="chunk 0: model output .* has 40 frames, expected 41"):
        run_job(plan, job_dir=tmp_path / "job", master_path=tmp_path / "m.mov", processor=drop_one)


def test_frame_delay_realigned(clip, tmp_path):
    """A model that lags by one frame (FlashVSR port) is realigned by the profile's frame_delay:
    the master must match the identity run exactly."""
    from vsr.profiles import load_profile
    from vsr.plan import delay_extra, parse_frame_multiple
    assert load_profile("flashvsr_1_3b").frame_delay == 1
    fm = parse_frame_multiple("4n+1")
    assert delay_extra(41, 1, fm) == 4 and delay_extra(41, 0, fm) == 0
    assert delay_extra(10, 2, parse_frame_multiple("any")) == 2

    common = ["--frames", "41", "--overlap", "8", "--codec", "prores", "--10bit"]
    run_cli(clip, "--profile", "flashvsr_1_3b", "--fake", "delay1", *common, "--out", tmp_path / "d",
            "--jobs-dir", tmp_path / "jobs")
    run_cli(clip, "--profile", "seedvr2_3b", "--fake", "identity", *common, "--out", tmp_path / "i",
            "--jobs-dir", tmp_path / "jobs")
    d = next((tmp_path / "d").glob("*.mov"))
    i = next((tmp_path / "i").glob("*.mov"))
    assert probe(d).frame_count == probe(i).frame_count == 100

    def frames(p):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(100, -1).astype(int)
    assert np.abs(frames(d) - frames(i)).max() <= 1

    # without the correction the lag shows: fake delay1 with a profile that has no frame_delay
    run_cli(clip, "--profile", "seedvr2_3b", "--fake", "delay1", *common, "--out", tmp_path / "x",
            "--jobs-dir", tmp_path / "jobs_x")  # own jobs dir: same job id as the identity run
    x = next((tmp_path / "x").glob("*.mov"))
    assert np.abs(frames(x) - frames(i)).max() > 10
