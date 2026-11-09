"""Load and validate YAML workflow profiles (patching comes with the ComfyUI client)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .plan import FrameMultiple, parse_frame_multiple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
SEED_FIELDS = ("seed", "noise_seed")


class ProfileError(ValueError):
    pass


@dataclass
class Profile:
    name: str
    path: Path
    workflow_path: Path
    workflow: dict
    input_node: str
    input_field: str
    output_node: str
    output_prefix_field: str
    frames: int
    overlap: int
    frame_multiple: FrameMultiple
    overrides: dict[str, dict] = field(default_factory=dict)
    max_seconds: float | None = None  # warn if a chunk is longer (e.g. SeedVR2 INT8 crash limit)

    def seed_fields(self) -> list[tuple[str, str]]:
        """(node_id, field) pairs holding a literal seed value in the workflow."""
        out = []
        for nid, node in self.workflow.items():
            for f in SEED_FIELDS:
                v = node.get("inputs", {}).get(f)
                if isinstance(v, int) and not isinstance(v, bool):
                    out.append((nid, f))
        return out


def load_workflow(path: str | Path) -> dict:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise ProfileError(f"workflow not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ProfileError(f"workflow {path} is not valid JSON: {e}") from None
    if not isinstance(data, dict) or "nodes" in data or "links" in data:
        raise ProfileError(
            f"{path} looks like a UI-format workflow. In ComfyUI use Workflow > Export (API) "
            "and save that file in workflows/."
        )
    bad = [k for k, v in data.items() if not isinstance(v, dict) or "class_type" not in v]
    if bad:
        raise ProfileError(f"{path} is not an API-format workflow (entries without class_type: {bad[:5]})")
    return data


def resolve_profile_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        if not p.is_file():
            raise ProfileError(f"profile file not found: {p}")
        return p
    for ext in (".yaml", ".yml"):
        cand = PROFILES_DIR / f"{name_or_path}{ext}"
        if cand.is_file():
            return cand
    available = sorted(x.stem for x in PROFILES_DIR.glob("*.y*ml")) if PROFILES_DIR.is_dir() else []
    raise ProfileError(f"profile {name_or_path!r} not found in {PROFILES_DIR}. Available: {available or 'none'}")


def _resolve_workflow_path(raw: str, profile_path: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    for base in (Path.cwd(), profile_path.parent.parent, PROJECT_ROOT):
        if (base / p).is_file():
            return base / p
    return PROJECT_ROOT / p


def _require(d: dict, key: str, ctx: str):
    if not isinstance(d, dict) or key not in d:
        raise ProfileError(f"{ctx}: missing required key '{key}'")
    return d[key]


def load_profile(name_or_path: str) -> Profile:
    path = resolve_profile_path(name_or_path)
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ProfileError(f"{path}: invalid YAML: {e}") from None
    ctx = str(path)

    wf_path = _resolve_workflow_path(str(_require(raw, "workflow", ctx)), path)
    wf = load_workflow(wf_path)

    inp = _require(raw, "input", ctx)
    out = _require(raw, "output", ctx)
    ch = _require(raw, "chunk", ctx)
    input_node = str(_require(inp, "node_id", f"{ctx} input"))
    input_field = str(_require(inp, "field", f"{ctx} input"))
    output_node = str(_require(out, "node_id", f"{ctx} output"))
    prefix_field = str(out.get("prefix_field", "filename_prefix"))

    errors = []
    if input_node not in wf:
        errors.append(f"input.node_id {input_node!r} not in workflow")
    elif input_field not in wf[input_node].get("inputs", {}):
        errors.append(
            f"input.field {input_field!r} not an input of node {input_node} "
            f"({wf[input_node]['class_type']}); has {list(wf[input_node].get('inputs', {}))}"
        )
    if output_node not in wf:
        errors.append(f"output.node_id {output_node!r} not in workflow")
    elif prefix_field not in wf[output_node].get("inputs", {}):
        errors.append(f"output.prefix_field {prefix_field!r} not an input of node {output_node}")

    overrides = {str(k): v for k, v in (raw.get("overrides") or {}).items()}
    for nid, fields in overrides.items():
        if nid not in wf:
            errors.append(f"overrides: node {nid!r} not in workflow")
            continue
        if not isinstance(fields, dict):
            errors.append(f"overrides: node {nid!r} must map to {{field: value}}")
            continue
        for f in fields:
            if f not in wf[nid].get("inputs", {}):
                errors.append(f"overrides: {f!r} not an input of node {nid} ({wf[nid]['class_type']})")

    try:
        fm = parse_frame_multiple(ch.get("frame_multiple", "any"))
    except ValueError as e:
        errors.append(f"chunk.frame_multiple: {e}")
        fm = FrameMultiple(1, 0)
    frames, overlap = ch.get("frames"), ch.get("overlap", 0)
    if not isinstance(frames, int) or frames < 1:
        errors.append(f"chunk.frames must be a positive integer, got {frames!r}")
    if not isinstance(overlap, int) or overlap < 0:
        errors.append(f"chunk.overlap must be a non-negative integer, got {overlap!r}")

    max_seconds = ch.get("max_seconds")
    if max_seconds is not None and (not isinstance(max_seconds, (int, float)) or max_seconds <= 0):
        errors.append(f"chunk.max_seconds must be a positive number, got {max_seconds!r}")

    if errors:
        raise ProfileError(f"{ctx}:\n  - " + "\n  - ".join(errors))

    return Profile(
        name=str(raw.get("name") or path.stem),
        path=path,
        workflow_path=wf_path,
        workflow=wf,
        input_node=input_node,
        input_field=input_field,
        output_node=output_node,
        output_prefix_field=prefix_field,
        frames=frames,
        overlap=overlap,
        frame_multiple=fm,
        overrides=overrides,
        max_seconds=float(max_seconds) if max_seconds is not None else None,
    )


def describe_workflow(wf: dict) -> list[str]:
    """Human-readable node listing for `vsr inspect`."""
    lines = []

    def key(k: str):
        return [int(p) if p.isdigit() else p for p in k.split(":")]

    for nid in sorted(wf, key=key):
        node = wf[nid]
        title = node.get("_meta", {}).get("title", "")
        lines.append(f"[{nid}] {node['class_type']}" + (f"  \"{title}\"" if title else ""))
        for name, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[1], int):
                lines.append(f"    {name} <- [{val[0]}].{val[1]}")
            else:
                lines.append(f"    {name} = {val!r}")
    return lines
