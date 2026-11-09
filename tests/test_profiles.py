import json

import pytest

from vsr.profiles import ProfileError, describe_workflow, load_profile, load_workflow

WF = {
    "1": {"class_type": "LoadVideo", "inputs": {"file": "x.mp4"}, "_meta": {"title": "Load"}},
    "2": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 1, "model": ["3", 0]}},
    "3": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "out", "video": ["2", 0]}},
}


def write(tmp_path, profile: str, wf=WF):
    (tmp_path / "workflows").mkdir(exist_ok=True)
    (tmp_path / "workflows" / "wf.json").write_text(json.dumps(wf))
    (tmp_path / "profiles").mkdir(exist_ok=True)
    p = tmp_path / "profiles" / "p.yaml"
    p.write_text(profile)
    return p


GOOD = """
name: test
workflow: workflows/wf.json
input: {node_id: "1", field: file}
output: {node_id: "3"}
overrides: {"2": {steps: 2}}
chunk: {frames: 81, overlap: 8, frame_multiple: 4n+1}
"""


def test_load_good(tmp_path):
    prof = load_profile(str(write(tmp_path, GOOD)))
    assert prof.name == "test" and prof.input_node == "1" and prof.output_node == "3"
    assert prof.frames == 81 and str(prof.frame_multiple) == "4n+1"
    assert prof.seed_fields() == [("2", "seed")]


@pytest.mark.parametrize("old,new,msg", [
    ('node_id: "1", field: file', 'node_id: "9", field: file', "input.node_id"),
    ("field: file", "field: filename", "input.field"),
    ('{"2": {steps: 2}}', '{"2": {stepz: 2}}', "stepz"),
    ("frames: 81", "frames: 0", "chunk.frames"),
    ("4n+1", "4x", "frame_multiple"),
])
def test_load_bad(tmp_path, old, new, msg):
    with pytest.raises(ProfileError, match=msg):
        load_profile(str(write(tmp_path, GOOD.replace(old, new))))


def test_ui_format_rejected(tmp_path):
    p = tmp_path / "ui.json"
    p.write_text(json.dumps({"nodes": [], "links": []}))
    with pytest.raises(ProfileError, match="Export \\(API\\)"):
        load_workflow(p)


def test_repo_profiles_are_valid():
    prof = load_profile("seedvr2_3b")
    assert ("66:54", "seed") in prof.seed_fields()


def test_describe():
    text = "\n".join(describe_workflow(WF))
    assert '[1] LoadVideo  "Load"' in text and "model <- [3].0" in text
