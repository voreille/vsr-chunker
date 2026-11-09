# vsr-chunker — chunked video upscaling through ComfyUI

## Goal

A Python CLI that upscales long videos with ComfyUI workflows (SeedVR2, Wan 2.2 detailer, FlashVSR, ...) by:

1. splitting the source into overlapping chunks,
2. sending each chunk through a saved ComfyUI workflow via its HTTP API,
3. stitching the results back with crossfaded overlaps,
4. writing a Resolve-friendly master (DNxHR / ProRes) with the original audio and timecode.

Typical job: 480p/720p source → 1080p, 5–60 s clips, run unattended, resumable after a crash.

## Environment

- Ubuntu 22.04, NVIDIA RTX 3090 (24 GB VRAM), 32 GB RAM.
- ComfyUI runs separately at `~/python_wkspce/ComfyUI` (own uv venv, Python 3.12). **This tool never imports ComfyUI code and never modifies the ComfyUI install, its models, or its custom nodes.** It only talks to the server over HTTP.
- ComfyUI URL must be configurable (default `http://127.0.0.1:8188`). Claude Code may run inside a container, so do **not** assume a shared filesystem with ComfyUI: transfer files via the API (`/upload/image` to send, `/view` to fetch outputs).
- Required system tools: `ffmpeg`, `ffprobe`. Check for them at startup and fail with a clear message.
- Use `uv` for the project (`uv init`, `uv add ...`). Python 3.12. Keep dependencies minimal: `httpx` or `requests`, `websocket-client` (optional), `pyyaml`, `typer` or `argparse`, `numpy` only if needed for blending.

## ComfyUI API notes

- Workflows must be exported from ComfyUI in **API format** ("Export (API)"). The UI-format JSON (with `nodes`/`links`/subgraphs) is NOT usable directly. Store API-format files in `workflows/`.
- `POST /upload/image` (multipart, field `image`, plus `overwrite=true`) puts any file, videos included, into ComfyUI's `input/` folder.
- `POST /prompt` with `{"prompt": <api_json>, "client_id": <uuid>}` returns `prompt_id`.
- Completion: poll `GET /history/{prompt_id}` (simplest), or listen on `ws://host/ws?clientId=<uuid>`.
- Outputs are listed in the history entry under each output node. Depending on the save node, files may appear under `images`, `gifs` (VideoHelperSuite Video Combine) or `videos`. **Inspect a real history response first** and handle all of these keys.
- Download with `GET /view?filename=...&subfolder=...&type=output`.
- `POST /free` with `{"unload_models": true, "free_memory": true}` frees VRAM between different workflows.
- Detect failures: the history entry has a `status` with `status_str` / error messages. Treat errors such as "CUDA error: an illegal memory access" as fatal for the server process: report clearly that ComfyUI must be restarted. Don't keep queueing into a broken server.

## Workflow profiles (config, not code)

Never hardcode node IDs. Each workflow gets a YAML profile in `profiles/`:

```yaml
name: seedvr2_3b
workflow: workflows/seedvr2_3b_api.json
input:
  node_id: "73"          # LoadVideo (or VHS_LoadVideo) node
  field: "file"          # widget holding the filename — verify in the API JSON
output:
  node_id: "88"          # save node whose output we download
overrides:               # optional: node_id -> {field: value}
  "12": { seed: 42 }
chunk:
  frames: 97             # frames per chunk (before rounding)
  overlap: 8             # frames shared between consecutive chunks
  frame_multiple: 4n+1   # round chunk length to this pattern (or "any")
```

Provide a helper command `vsr inspect workflows/x.json` that lists every node (id, class_type, title, inputs) to make writing profiles easy.

Always set a **fixed seed** per job (use the overrides, or detect seed fields and set them), so all chunks share the same noise.

## Known model constraints (from testing on this machine)

- **SeedVR2 INT8** (`*_int8_convrot`) crashes with `CUDA error: illegal memory access` in `comfy_kitchen ... int8_linear` on long inputs (a 5 s 1080p clip works, a 20 s one crashes). That's likely a 32-bit overflow. Keep SeedVR2 chunks at **≤ 4–5 s** of output at 1080p. FP16/FP8 checkpoints avoid that kernel.
- **SeedVR2**: don't use FlashAttention (black output with quantized checkpoints on the 3090).
- **Wan 2.2 detailer**: ≤ 81 frames per chunk is safe on 24 GB (121 max). Frame counts must be 4n+1. Width and height must be divisible by 16 (the workflow resizes internally, so the tool doesn't need to).
- **Wan "Utility Video Upscale" template**: contains an `ImageFromBatch` node that limits processing to 48 frames. Its profile must override `batch_index=0` and `length` to the full chunk length (or that node must be removed from the API JSON).
- Diffusion-based detailers produce slightly different texture per chunk, so **crossfading overlaps is required**, not optional.

## Pipeline design

### 1. Probe
- `ffprobe` the source: width, height, fps (as a rational), frame count, pixel format, color metadata (range, primaries, transfer, matrix), audio streams, timecode.
- If the source is variable frame rate, convert it to constant frame rate first, and warn.

### 2. Chunk plan
- Pure, unit-tested function: `plan_chunks(total_frames, frames, overlap, frame_multiple) -> list[(start, length)]`.
- Chunk i starts at `i * (frames - overlap)`. The last chunk may be shorter; pad it if the model requires 4n+1 (repeat the last frame), and trim the padding after processing.
- Cut **frame-accurately** using frame numbers, not timestamps: `-vf "select='between(n,START,END)',setpts=PTS-STARTPTS"`, or decode once to a lossless intermediate and slice that.
- Chunk files: lossless intermediate that ComfyUI's loader can read. Default is x264 lossless (`-c:v libx264 -crf 0 -preset veryfast -pix_fmt yuv420p`) in `.mp4`; make the codec configurable. Video only, no audio.

### 3. Process
- For each chunk: upload → patch the workflow JSON (input filename, overrides, output filename prefix `vsr_<job>_<chunk>`) → queue → wait → download the output into `jobs/<job_id>/out/`.
- Verify the output frame count matches the input chunk frame count (or chunk × the model's temporal factor, if any). Mismatch → mark the chunk failed with a clear message.
- **Resumable**: keep `jobs/<job_id>/manifest.json` with per-chunk status (`pending/running/done/failed`), paths, durations and checksums. Rerunning the same job skips `done` chunks. `--redo 3,7` reprocesses specific chunks.
- Log the time per chunk and an ETA.

### 4. Stitch
- Concatenate the processed chunks. In each overlap region, linearly crossfade frame-by-frame (weight from 1→0 on the outgoing chunk). Implement it with ffmpeg (`xfade` with a duration of `overlap / fps`, or the `blend` filter) or with numpy on decoded frames, whichever is simpler and exact. The output frame count must equal the source frame count exactly.
- Stream frames; never load a whole clip into RAM (1080p float frames add up fast).

### 5. Master
- Encode the final video as DNxHR HQ by default (`-c:v dnxhd -profile:v dnxhr_hq -pix_fmt yuv422p`, `.mov`). Options: `--codec prores` (prores_ks, profile 3) and `--10bit` (DNxHR HQX, `yuv422p10le`).
- Mux the **original audio** from the source, untouched (`-map 1:a? -c:a pcm_s16le` or copy when the container allows).
- Copy the source timecode (`-timecode`) and the color metadata tags so Resolve relinks and interprets the file correctly.
- Output name: `<source_stem>_<profile>_<WxH>.mov` next to the source, or in `--out`.

## CLI

```
vsr inspect <workflow_api.json>
vsr run <video> --profile seedvr2_3b [--out DIR] [--server URL] [--seed N]
         [--frames N --overlap N] [--range START-END (seconds)] [--dry-run]
vsr resume <job_id>
vsr status <job_id>
```

- `--dry-run`: probe + chunk plan + print commands, without touching ComfyUI.
- `--range`: process only part of the source, for quick tests and for upscaling only what's used in the Resolve edit (with handles).
- Chaining profiles (e.g. SeedVR2 → Wan detailer) can come later. Design `run` so a job can take an ordered list of profiles, each stage's stitched output feeding the next, with `/free` between stages.

## Project layout

```
vsr/
  cli.py        # commands
  probe.py      # ffprobe wrapper
  plan.py       # chunk math (pure)
  ffmpeg.py     # cut / stitch / master command builders
  comfy.py      # API client: upload, queue, wait, download, free
  profiles.py   # load + validate YAML profiles, patch workflow JSON
  job.py        # manifest + resume logic
profiles/
workflows/      # API-format JSON exported from ComfyUI
tests/
jobs/           # runtime state (gitignored)
```

## Testing

- Unit tests for `plan.py` (edge cases: clip shorter than one chunk, exact multiples, 4n+1 rounding, last-chunk padding).
- Stitch test **without a GPU**: generate synthetic clips with `ffmpeg -f lavfi -i testsrc2`, run them through an identity "fake model" (just copy each chunk), stitch, and assert that the frame count and duration equal the source.
- A mock ComfyUI server (small FastAPI or `http.server`) for `comfy.py` tests: upload, prompt, history, view.
- One manual end-to-end test against the real server with a 3–5 s clip, documented in `README.md`.

## Rules for working on this repo

- Build in milestones, in this order. Finish and test each one before starting the next:
  1. probe + chunk plan + `--dry-run`
  2. cut + stitch + master with the identity fake model (no ComfyUI)
  3. ComfyUI client + `inspect` + one real profile (SeedVR2)
  4. manifest / resume / ETA
  5. Wan detailer profile; multi-stage chaining
- Keep `HANDOFF.md` up to date at the end of each work session: current milestone, what's done, what's next, open problems, and exact commands to rerun the tests. Prefer small focused sessions, and read `HANDOFF.md` first when starting.
- Don't download models, change ComfyUI settings, or install custom nodes. If something is needed on the ComfyUI side, write it in `HANDOFF.md` for the user.
- Prefer clear error messages over silent fallbacks, especially for frame-count mismatches and ComfyUI errors.
- Keep `README.md` short: install, how to export a workflow in API format, how to write a profile, example commands.
- Commits must not be attributed to Claude: no `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" line, and no Claude author or committer. Use the repo's configured git user only.

