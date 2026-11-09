import socket

import pytest

from mock_comfy import MockComfy
from vsr import cli
from vsr.comfy import ComfyClient, ComfyError, ComfyFatalError
from vsr.pipeline import comfy_processor, run_job
from vsr.probe import probe


@pytest.fixture
def mock():
    with MockComfy("73", "file", "76") as m:
        yield m


@pytest.fixture
def client(mock):
    with ComfyClient(mock.url) as c:
        c.poll = 0.01
        yield c


def plan_for(clip, *extra):
    args = cli.build_parser().parse_args(["run", str(clip), "--profile", "seedvr2_3b", "--frames", "41",
                                          "--overlap", "8", "--seed", "1234", *extra])
    return cli.build_plan(args)


def test_upload_and_free(client, mock, tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert client.upload(f, name="x.mp4", subfolder="vsr") == "vsr/x.mp4"
    assert mock.files[("input", "vsr", "x.mp4")] == b"hello"
    client.free()
    assert mock.frees == [{"unload_models": True, "free_memory": True}]


@pytest.mark.parametrize("key,ext", [("images", ".mov"), ("gifs", ".mp4"), ("videos", ".mp4")])
def test_full_job_through_mock(clip, client, mock, tmp_path, key, ext):
    mock.output_key, mock.output_ext = key, ext
    plan = plan_for(clip)
    master = run_job(plan, job_dir=tmp_path / "job", master_path=tmp_path / "m.mov",
                     processor=comfy_processor(client, plan.profile, plan.job_id, plan.seed))
    assert probe(master).frame_count == 100
    assert len(mock.prompts) == len(plan.chunks) == 3
    for i, p in enumerate(mock.prompts):
        assert p["73"]["inputs"]["file"] == f"vsr/vsr_{plan.job_id}_{i:03d}.mp4"
        assert p["76"]["inputs"]["filename_prefix"] == f"vsr/vsr_{plan.job_id}_{i:03d}"
        assert p["66:54"]["inputs"]["seed"] == 1234  # same seed for every chunk
        assert p["76"]["inputs"]["format"] == "mov"  # profile override applied
        assert p["66:59"]["inputs"]["color_correction_method"] == "lab"
    assert (tmp_path / "job" / "out" / f"chunk_000{ext}").is_file()


def test_cuda_error_is_fatal(clip, client, mock, tmp_path):
    mock.mode = "cuda"
    plan = plan_for(clip)
    with pytest.raises(ComfyFatalError, match="(?s)illegal memory access.*must be restarted"):
        run_job(plan, job_dir=tmp_path / "job", master_path=tmp_path / "m.mov",
                processor=comfy_processor(client, plan.profile, plan.job_id, plan.seed))
    assert len(mock.prompts) == 1  # stopped after the first failure


def test_non_fatal_execution_error(clip, client, mock, tmp_path):
    mock.mode = "oom"
    plan = plan_for(clip)
    with pytest.raises(ComfyError, match="node 66:54 \\(KSampler\\).*exceed") as e:
        run_job(plan, job_dir=tmp_path / "job", master_path=tmp_path / "m.mov",
                processor=comfy_processor(client, plan.profile, plan.job_id, plan.seed))
    assert not isinstance(e.value, ComfyFatalError)


def test_rejected_workflow(client, mock):
    mock.mode = "reject"
    with pytest.raises(ComfyError, match="node 73 \\(LoadVideo\\): Value not in list"):
        client.queue({"73": {"class_type": "LoadVideo", "inputs": {"file": "x"}},
                      "76": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "p"}}})


def test_no_output(clip, client, mock, tmp_path):
    mock.mode = "nothing"
    plan = plan_for(clip)
    with pytest.raises(ComfyError, match="output node 76 produced nothing"):
        run_job(plan, job_dir=tmp_path / "job", master_path=tmp_path / "m.mov",
                processor=comfy_processor(client, plan.profile, plan.job_id, plan.seed))


def test_server_down(clip, tmp_path, capsys):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    rc = cli.main(["run", str(clip), "--profile", "seedvr2_3b", "--server", f"http://127.0.0.1:{port}",
                   "--jobs-dir", str(tmp_path / "jobs"), "--out", str(tmp_path)])
    assert rc == 2
    assert "cannot connect to ComfyUI" in capsys.readouterr().err
    assert not (tmp_path / "jobs").exists()  # failed before cutting anything


def test_cli_against_mock(clip, mock, tmp_path, capsys):
    rc = cli.main(["run", str(clip), "--profile", "seedvr2_3b", "--server", mock.url, "--frames", "41",
                   "--jobs-dir", str(tmp_path / "jobs"), "--out", str(tmp_path)])
    assert rc == 0, capsys.readouterr().err
    assert probe(tmp_path / "clip_seedvr2_3b_320x240.mov").frame_count == 100
