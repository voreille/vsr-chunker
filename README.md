# vsr-chunker

Upscale long videos through ComfyUI workflows (SeedVR2, Wan 2.2, FlashVSR, ...) in overlapping
chunks, stitch them with crossfades, and write a Resolve-friendly master (DNxHR/ProRes) with the
original audio and timecode. Status: milestone 2 (probe, plan, cut, stitch, master; ComfyUI processing not wired yet).

## Install

Needs `ffmpeg`/`ffprobe` on PATH and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run vsr --help
```

## Exporting a workflow (API format)

In ComfyUI: open the workflow, then **Workflow → Export (API)**, and save the JSON in `workflows/`.
The normal UI save (with `nodes`/`links`) is rejected.

## Writing a profile

List the nodes to find the input/output IDs:

```
uv run vsr inspect workflows/utility_seedvr2_3b_int8_upscale_video.json
```

Then create `profiles/<name>.yaml` (see `profiles/seedvr2_3b.yaml`):

```yaml
name: seedvr2_3b
workflow: workflows/utility_seedvr2_3b_int8_upscale_video.json
input:  { node_id: "73", field: "file" }               # LoadVideo
output: { node_id: "76", prefix_field: "filename_prefix" }  # SaveVideo
overrides: {}                 # node_id -> {field: value}
chunk:
  frames: 97                  # rounded down to frame_multiple
  overlap: 8
  frame_multiple: 4n+1        # or "any", "8n", ...
  max_seconds: 4.5            # optional: warn when chunks are longer
```

Seed fields (`seed`, `noise_seed`) are detected and set to one fixed seed per job.

## Examples

```
uv run vsr run clip.mp4 --profile seedvr2_3b --dry-run
uv run vsr run clip.mp4 --profile seedvr2_3b --dry-run --range 10-15 --frames 81 --overlap 8
# whole pipeline with a fake model instead of ComfyUI (identity copy, or lanczos x2):
uv run vsr run clip.mp4 --profile seedvr2_3b --fake scale2 --out /tmp/test --10bit
```

## Tests

```
uv run pytest -q
```
