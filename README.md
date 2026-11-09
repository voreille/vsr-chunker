# vsr-chunker

Upscale long videos through ComfyUI workflows (SeedVR2, Wan 2.2, FlashVSR, ...) in overlapping
chunks, stitch them with crossfades, and write a Resolve-friendly master (DNxHR/ProRes) with the
original audio and timecode. Status: milestone 4 (single-profile runs through ComfyUI, resumable).

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

Optional `resize:` section: the node whose width/height fields set the output size. The tool fills
them from `--height N` (default 1080, width from the source aspect ratio) or `--scale F`, rounded to
`multiple` (2 by default; 16 for FlashVSR):

```yaml
resize: { node_id: "66:57", width_field: "resize_type.width", height_field: "resize_type.height", multiple: 2 }
```

If a model's output lags its input (output frame n shows input frame n-k), set
`output.frame_delay: k`: the tool sends extra frames and drops the first k output frames
(the FlashVSR profile uses 1).

## Examples

```
vsr run clip.mp4 --profile seedvr2_3b                 # 1080p output (default)
vsr run clip.mp4 --profile seedvr2_3b --height 1440   # or --scale 2
uv run vsr run clip.mp4 --profile seedvr2_3b                      # real run, master next to clip.mp4
uv run vsr run clip.mp4 --profile seedvr2_3b --range 10-14 --seed 42 --server http://127.0.0.1:8188
uv run vsr run clip.mp4 --profile seedvr2_3b --dry-run
uv run vsr run clip.mp4 --profile seedvr2_3b --dry-run --range 10-15 --frames 81 --overlap 8
# whole pipeline with a fake model instead of ComfyUI (identity copy, or lanczos x2):
uv run vsr run clip.mp4 --profile seedvr2_3b --fake scale2 --out /tmp/test --10bit
```

Jobs are resumable. Rerunning the same command, or `vsr resume <job_id>`, skips finished chunks:

```
uv run vsr status <job_id>                 # chunk table, errors, time left
uv run vsr resume <job_id> [--redo 3,7]    # continue / reprocess specific chunks
```

Ctrl-C is safe: finished chunks are kept and the running ComfyUI prompt is cancelled.
A CUDA error stops the job (exit 2): restart ComfyUI, then `vsr resume`.

Uploads go to ComfyUI's `input/vsr/`, outputs to `output/vsr/` (named `vsr_<job>_<chunk>`).
They are not deleted (the API can't); clean those folders by hand from time to time.

## Tests

```
uv run pytest -q        # no GPU needed: fake models + a mock ComfyUI server
```

### Manual end-to-end test (real ComfyUI, ~3.5 min on a 3090)

```
ffmpeg -f lavfi -i testsrc2=size=854x480:rate=24 -f lavfi -i sine=sample_rate=48000 -t 4 \
  -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -timecode 01:00:00:00 /tmp/e2e.mp4
uv run vsr run /tmp/e2e.mp4 --profile seedvr2_3b --frames 49 --overlap 8 --seed 42 --out /tmp/e2e_out
```

Expected: 3 chunks (49, 49, 14+3 pad frames), ~1:25 per 49-frame chunk, then
`master OK: /tmp/e2e_out/e2e_seedvr2_3b_1708x960.mov (1708x960, 96 frames, 4.000s)`.
