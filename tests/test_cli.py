import pytest

from vsr.cli import CliError, main, parse_range


def test_parse_range():
    assert parse_range("1.5-3") == (1.5, 3.0)
    assert parse_range("2-") == (2.0, None)
    for bad in ("3", "3-2", "a-b"):
        with pytest.raises(CliError):
            parse_range(bad)


def test_dry_run(clip, capsys):
    assert main(["run", str(clip), "--profile", "seedvr2_3b", "--dry-run", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert "frames     100" in out and "seed       7 (--seed) -> [66:54].seed" in out
    # 100 frames, 97/8 4n+1: [0..96] and [89..99] (11 frames + 2 pad = 13)
    assert "2 chunk(s)" in out
    assert "trim=start_frame=89:end_frame=100,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop=2" in out
    assert "-f rawvideo -pix_fmt yuv422p16le -s WxH" in out


def test_dry_run_range_and_overrides(clip, capsys):
    assert main(["run", str(clip), "--profile", "seedvr2_3b", "--dry-run",
                 "--range", "1-3", "--frames", "21", "--overlap", "4"]) == 0
    out = capsys.readouterr().out
    assert "frames 25..74 (50 frames" in out
    assert "3 chunk(s)" in out  # 21 + 17 + 17 > 50
    assert "-timecode 10:00:01:00" in out


def test_errors_are_clean(clip, capsys, tmp_path):
    assert main(["run", str(clip), "--profile", "nope", "--dry-run"]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["run", str(tmp_path / "missing.mp4"), "--profile", "seedvr2_3b", "--dry-run"]) == 1
    assert main(["run", str(clip), "--profile", "seedvr2_3b", "--dry-run", "--range", "9-10"]) == 1
    assert "past the end" in capsys.readouterr().err


def test_inspect(capsys):
    assert main(["inspect", "workflows/utility_seedvr2_3b_int8_upscale_video.json"]) == 0
    assert "[66:54] KSampler" in capsys.readouterr().out


def test_output_size_options(clip, capsys, tmp_path):
    assert main(["run", str(clip), "--profile", "seedvr2_3b", "--dry-run"]) == 0
    assert "size       1440x1080 -> [66:57]" in capsys.readouterr().out  # 320x240 -> default 1080p
    assert main(["run", str(clip), "--profile", "seedvr2_3b", "--dry-run", "--height", "480"]) == 0
    assert "size       640x480" in capsys.readouterr().out
    assert main(["run", str(clip), "--profile", "flashvsr_1_3b", "--dry-run", "--scale", "1.5"]) == 0
    assert "size       480x352" in capsys.readouterr().out  # multiple of 16
    # a profile without a resize section rejects --height
    import re
    from vsr.profiles import PROJECT_ROOT
    text = (PROJECT_ROOT / "profiles/seedvr2_3b.yaml").read_text()
    text = re.sub(r"(?s)resize:.*?multiple: 2\n", "", text)
    (tmp_path / "noresize.yaml").write_text(text.replace("workflows/", str(PROJECT_ROOT / "workflows") + "/"))
    assert main(["run", str(clip), "--profile", str(tmp_path / "noresize.yaml"), "--dry-run"]) == 0
    assert "fixed by the workflow" in capsys.readouterr().out
    assert main(["run", str(clip), "--profile", str(tmp_path / "noresize.yaml"), "--dry-run", "--height", "720"]) == 1
    assert "no 'resize' section" in capsys.readouterr().err
