# HANDOFF

## Current milestone: 4 done (manifest, resume, redo, status, ETA) + FlashVSR profile + output size. Next: milestone 5.

### Done (session 2026-11-10, prescale)
- `vsr run --prescale HEIGHT`: chunks are downscaled while cutting (`ffmpeg.prescale_filter`, Lanczos,
  antialiased) for sources whose real detail is below their pixel size. `--scale` is relative to the
  prescaled size; the output width always follows the source aspect (no 1922x1080). Cut size checked.
  In the job id. Verified on the server: 1280x720 --prescale 480 -> 1920x1080.
- Open: last real runs were very slow (7 s/frame for 25 frames at 1080p vs ~1.6 normally) while
  the disk was full; re-measure now that there is space.

### Done (session 2026-11-09, output size)
- Profile `resize:` section (`profiles.Resize`: node_id, width_field, height_field, multiple).
  `vsr run --height N` (default 1080) / `--scale F` -> `profiles.output_size()` -> patched into the
  workflow; the job id includes the size; each chunk's output size is checked (`expect_size`).
  A profile without `resize` keeps the workflow's size and rejects --height/--scale.
- Both workflows' "Resize Image/Mask" nodes now use "scale dimensions" (was "scale by multiplier" 2).
  FlashVSR: multiple 16 (1080 -> 1088); the extra ImageResizeKJv2 node was removed.
- Verified on the server: 1280x720 -> SeedVR2 1920x1080, FlashVSR 1920x1088.
- ComfyUI now runs with `--fast fp16_accumulation` (user restarted it). 97-frame chunk: SeedVR2 INT8
  186.6 -> 154.7 s, output unchanged (50.6 dB vs before); fp16 checkpoint 512 tiles 170.5 s, 1024
  tiles 136.1 s / 17.1 GiB; FlashVSR 97.4 -> 109.4 s (single run, maybe noise).

### Done (session 2026-11-09, FlashVSR profile)
- `profiles/flashvsr_1_3b.yaml` + `workflows/flashvsr_1_3b_api.json` (Kijai WanVideoWrapper port, x2).
  Benchmark on a 97-frame 960x544 chunk: SeedVR2 3B 186.6 s / 14.3 GiB; FlashVSR sdpa 144.9 s;
  FlashVSR sageattn 97.4 s / 11.9 GiB (profile override `"8": attention_mode: sageattn`).
  Output is smoother than SeedVR2 (luma fidelity vs source 27.0 vs 31.6 dB).
- The port's output lags by one frame. New profile key `output.frame_delay` (`Profile.frame_delay`):
  cut sends `delay_extra()` more frames (kept 4n+1, e.g. 97 -> 101), the model output must have that
  many, stitching skips the first `frame_delay` frames (`stitch.window`). Job id includes it when
  non-zero. Fake model `--fake delay1` + `test_frame_delay_realigned`. Verified on the real server:
  best alignment at shift 0 after the fix.
- FlashVSR needs output width/height divisible by 16: node 16 `ImageResizeKJv2` (KJNodes) rounds
  the x2 size down (stretch; no-op for 960x544 -> 1920x1088, 854x480 -> 1696x960).

### Done (session 2026-11-09, milestone 4)
- `vsr/job.py`: `Manifest` (`jobs/<id>/manifest.json`, atomic write): run args, plan, per-chunk
  status pending/running/done/failed, attempts, duration, error, input/output sha256 + size;
  master info. Reopening checks the chunk plan is unchanged; `running` -> `pending` (crash).
  A chunk is skipped only if `done` AND its output file still matches its checksum.
- `run_job`: cuts and processes only unfinished chunks. Non-fatal chunk errors -> `failed`,
  keep going, then stop before stitching with the list of errors + `vsr resume <id>`.
  `ComfyFatalError` -> stop at once. Ctrl-C -> chunk back to `pending`, our ComfyUI prompt is
  removed from the queue or interrupted (`ComfyClient.cancel`, only if it is ours), exit 130.
- ETA weighted by frames (s/frame of chunks done this session x frames left).
- CLI: `--redo 3,7` / `2-5` on run and resume; `vsr resume <id>` replays the stored args
  (`--server`, `--timeout` overridable; refuses if the source file changed, since the job id
  then differs); `vsr status <id>` prints the chunk table, estimate, master.
  Rerunning the same `vsr run` command continues the same job.
- Real-server test: interrupted a 3-chunk SeedVR2 job during chunk 1 -> ComfyUI queue empty;
  `vsr resume` redid chunks 1-2 only, master 96 frames. ETA predicted 0:28, actual 0:28.

### Done (session 2026-11-09, milestone 3)
- `vsr/comfy.py`: httpx client. upload (`/upload/image`, subfolder `vsr`, overwrite), queue
  (`/prompt`, readable node_errors), wait (polls `/history/{id}`, checks `/queue` every 30 s for a
  vanished prompt), outputs (any list of `{filename, ...}` under the output node, whatever the key),
  download (`/view`, streamed, `.part` then rename), free. CUDA-type errors -> `ComfyFatalError`
  with "ComfyUI must be restarted"; CLI exits 2. Connection refused is fatal too, checked with
  `/system_stats` before cutting anything.
- `profiles.patch_workflow`: input value, `vsr/vsr_<job>_<chunk>` prefix, overrides, seed fields.
- `pipeline.comfy_processor`: upload -> patch -> queue -> wait -> download into `jobs/<id>/out/`
  (keeps the server's extension). `vsr run` without `--fake` uses it. `--timeout` per chunk.
- Mock ComfyUI server (`tests/mock_comfy.py`, http.server): upload/prompt/history/queue/view/free,
  modes ok/cuda/oom/reject/nothing, output under images/gifs/videos. `tests/test_comfy.py`.
- Real run (ComfyUI 0.44.0): 4 s 854x480 24 fps testsrc2, 3 chunks of 49 -> 1708x960 master,
  96 frames, ~1:25 per 49-frame chunk. Real history shape: `outputs["76"]["images"] =
  [{filename, subfolder: "vsr", type: "output", metadata: {frame_count, fps, width, height}}]`
  plus `"animated": [true]`; status messages execution_start/cached/success.
- Profile changes: SaveVideo -> `format: mov`, `format.codec: prores` (gives ProRes 4444
  yuv444p12le, tagged bt709 / transfer sRGB; we tag the master from the source anyway).
  SeedVR2 Post-Process `color_correction_method: lab` (template had `none`): measured mean drift
  vs source Y/U/V +4.6/-1.7/-2.8 with none, +0.7/+0.5/+1.2 with lab (PSNR Y 26.4 -> 29.1 dB).
  The drift is in ComfyUI's raw output; our stitch + master add < 0.2 levels.
- Matrix check: master vs source 26.4 dB; same master forced through a 709->601 mismatch 19.0 dB,
  so the color matrix is right.

### Done (session 2026-11-09, milestone 2)
- `vsr run ... --fake identity|scale2` runs the whole pipeline without ComfyUI:
  CFR conversion (if VFR) -> cut chunks into `jobs/<id>/chunks/` (frame count verified) -> fake model
  into `jobs/<id>/out/` (frame count verified, clear error naming the chunk) -> stitch -> master
  -> master verified (frame count == range frames, size). Logs per step in `jobs/<id>/logs/`.
- `vsr/stitch.py`: `stitch_frames()` (pure core, tested with numpy arrays) + ffmpeg pipe reader and
  writer. Each model output is decoded to `yuv422p16le`; overlaps blend with incoming weight
  (j+1)/(ov+1) (strictly between 0 and 1); pad frames dropped; at most `overlap` frames in RAM.
  Frames are piped straight into the master encoder: there is no stitched intermediate file.
  Why 4:2:2: all master codecs are 4:2:2, so chroma is resampled once. (4:4:4 cost ~5 dB chroma
  PSNR vs a direct encode; 4:2:2 matches a direct DNxHR encode: Y 61.6 / U 45.1 dB on testsrc2.)
- Decode pins `out_color_matrix`/`out_range` from the source tags, so an RGB model output is
  converted back with the source matrix rather than swscale's bt601 default.
- VFR: converted to CFR at the nearest standard rate (within 2%, else avg rounded to 0.001),
  plan recomputed from the CFR copy, audio/timecode/color still from the original. Job id kept.
- `plan_chunks` now also requires chunk length >= 2 x overlap when there is more than one chunk.
- `--jobs-dir` (default `jobs/`), used by tests to stay out of the repo.
- 1080p 10 s identity job: 4.7 s total, stitch ~100 fps, peak RSS 1.3 GB (ffmpeg children).
- Not done yet: manifest / resume / skipping existing chunks (milestone 4). A rerun redoes all.


### Done (session 2026-11-09)
- uv project (Python 3.12, deps: pyyaml; dev: pytest). Entry point `vsr` -> `vsr/cli.py`. argparse, no typer.
- `vsr/probe.py`: ffprobe wrapper. Exact frame count by `-count_packets` (warns if the header
  `nb_frames` disagrees); VFR heuristic (r_frame_rate vs avg_frame_rate); color tags, audio streams,
  timecode (stream, format, or tmcd track). `check_tools()` fails clearly without ffmpeg/ffprobe.
- `vsr/plan.py`: `plan_chunks(total, frames, overlap, frame_multiple) -> list[Chunk(start, length, pad)]`.
  `frames` is rounded DOWN to the pattern; the last chunk is padded UP (repeat the last frame); no
  trailing chunk that is only overlap. Exhaustively tested for 1..399 frames.
- `vsr/profiles.py`: YAML profile loading + validation against the workflow JSON (nodes/fields
  exist, UI-format rejected), seed-field detection, `describe_workflow` for `vsr inspect`.
  Extra optional keys: `output.prefix_field` (default `filename_prefix`), `output.frame_delay`, `resize`, `chunk.max_seconds`.
- `vsr/ffmpeg.py`: argv builders for CFR conversion, frame-exact cut
  (`trim=start_frame:end_frame,setpts,tpad=clone`), and master (DNxHR HQ/HQX, ProRes; audio
  copy or PCM when trimmed; timecode offset for --range; color tags).
  Cuts verified bit-exact against source frames (test_ffmpeg.py).
- `vsr/cli.py`: `inspect` and `run --dry-run` work. `run` without --dry-run, `resume` and `status`
  exit with "not implemented yet". Job id = `<stem>_<profile>_<sha1[:8]>` of
  (path, size, mtime, profile, frames, overlap, range, seed); default seed is derived from it.
- `profiles/seedvr2_3b.yaml` for `workflows/utility_seedvr2_3b_int8_upscale_video.json`
  (LoadVideo 73.file -> SaveVideo 76, KSampler seed at 66:54).
- `inspect` was done early (milestone 3 item) since it needs no server.

### Rerun tests
```
uv sync && uv run pytest -q          # 79 tests, ~21 s, needs ffmpeg, no GPU (mock ComfyUI)
uv run vsr run <clip> --profile seedvr2_3b --dry-run
uv run vsr run <clip> --profile seedvr2_3b --fake identity --out /tmp/x   # full pipeline, no ComfyUI
# real end-to-end test: see README "Manual end-to-end test"
```

### Next (milestone 5: Wan detailer profile; multi-stage chaining)
- Needs from the user: the Wan 2.2 detailer workflow exported in API format, in `workflows/`.
  Then `vsr inspect` it and write `profiles/wan22_detailer.yaml` (frames <= 81, 4n+1; override
  `ImageFromBatch` batch_index=0 / length=chunk length if it is the "Utility Video Upscale" one).
  The `length` override depends on each chunk, so profiles need a way to say "chunk length"
  (e.g. a `$chunk_frames` placeholder in overrides).
- Chaining: `--profile a,b`. Stage k's stitched output is a lossless intermediate (FFV1 .mkv or
  x264 lossless) fed to stage k+1 as its source, `/free` between stages, master only after the
  last stage. Each stage = its own job dir/manifest under a parent job.

### ComfyUI-side changes made with the user's explicit approval (2026-11-09)
- CLAUDE.md forbids downloading models; the user explicitly asked for this exception.
  Downloaded from huggingface.co/Kijai/WanVideo_comfy (sha256 verified against HF LFS oids):
  - `models/diffusion_models/WanVideo/FlashVSR/Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors` (5.68 GB)
  - `models/diffusion_models/WanVideo/FlashVSR/Wan2_1_FlashVSR_LQ_proj_model_bf16.safetensors` (0.58 GB)
  - `models/vae/Wan2_1_FlashVSR_TCDecoder_fp32.safetensors` (0.18 GB)
  Not downloaded: `umt5-xxl-enc-bf16` (11.4 GB); the existing `umt5_xxl_fp16` works with Kijai's T5 loader.
  No settings changed, no custom nodes installed. Disk was 98% full afterwards (24 GB free).

### Notes / open questions
- System ffmpeg is 4.4 (Ubuntu 22.04): no `-fps_mode`; `ffmpeg.vsync()` picks `-vsync` on < 5.1.
- Each cut re-decodes the source from frame 0 (`trim`): fine for <= 60 s (1080p 10 s: 2 s for 3
  cuts). For long sources, decode once to a lossless intermediate and slice that.
- Chunk intermediates are yuv420p 8-bit: loses chroma/bit depth for 10-bit or 4:4:4 sources.
  Codec is configurable in code (`ChunkCodec`) but not yet from the CLI/profile.
- ComfyUI 0.44.0 reachable at http://127.0.0.1:8188 from this environment (checked
  /system_stats, /queue, /history). History was empty, so the real output-entry shape
  (`images`/`gifs`/`videos`) still has to be checked on a real run in milestone 3.
- For the user: chunk uploads pile up in ComfyUI `input/vsr/` and outputs in `output/vsr/` (no
  delete endpoint). Clean them by hand. They also appear in LoadVideo's file list in the UI.
- SeedVR2 is slow here: ~1.7 s/frame at 1708x960 (tiled VAE, INT8). Not investigated.
- Not tested on a real server yet: the CUDA-crash path (only via the mock) and long chunks.
- Unknown: whether SeedVR2 output is bit-identical for the same seed (matters for --redo
  consistency at chunk borders; the crossfade hides small differences anyway).
- The master is re-stitched on every successful run/resume, even if it already exists (cheap).
