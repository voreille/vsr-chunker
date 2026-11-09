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
