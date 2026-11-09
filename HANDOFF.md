# HANDOFF

## Current milestone: 2 done (cut + stitch + master, fake models). Next: milestone 3.

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
  Extra optional keys: `output.prefix_field` (default `filename_prefix`), `chunk.max_seconds`.
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
uv sync && uv run pytest -q          # 60 tests, ~8 s, needs ffmpeg, no GPU
uv run vsr run <clip> --profile seedvr2_3b --dry-run
uv run vsr run <clip> --profile seedvr2_3b --fake identity --out /tmp/x   # full pipeline, no ComfyUI
```

### Next (milestone 3: ComfyUI client + one real profile)
- `vsr/comfy.py`: upload (`/upload/image`), patch workflow (input file, overrides, seed fields,
  `vsr_<job>_<chunk>` prefix), `/prompt`, poll `/history/{id}`, download via `/view`, `/free`.
  Mock server tests (http.server).
- Implement it as a `Processor` (`pipeline.py`: `(index, in_path, out_path) -> out_path`) and use
  it in `cmd_run` when `--fake` isn't given. Everything after the processor already works.
- Inspect a real history entry first (images/gifs/videos keys). Fatal CUDA errors -> stop + tell
  the user to restart ComfyUI.
- Manual e2e: 3-5 s clip through seedvr2_3b, document in README.

### Notes / open questions
- System ffmpeg is 4.4 (Ubuntu 22.04): no `-fps_mode`; `ffmpeg.vsync()` picks `-vsync` on < 5.1.
- Each cut re-decodes the source from frame 0 (`trim`): fine for <= 60 s (1080p 10 s: 2 s for 3
  cuts). For long sources, decode once to a lossless intermediate and slice that.
- Chunk intermediates are yuv420p 8-bit: loses chroma/bit depth for 10-bit or 4:4:4 sources.
  Codec is configurable in code (`ChunkCodec`) but not yet from the CLI/profile.
- ComfyUI 0.44.0 reachable at http://127.0.0.1:8188 from this environment (checked
  /system_stats, /queue, /history). History was empty, so the real output-entry shape
  (`images`/`gifs`/`videos`) still has to be checked on a real run in milestone 3.
- SaveVideo (node 76) uses format/codec `auto` (likely lossy h264). Its schema offers other codecs
  (ProRes -> MOV). In milestone 3, consider an override for a higher-quality output codec.
- The SeedVR2 workflow has LoadVideo -> GetVideoComponents -> CreateVideo with the input fps; check
  that the output frame count equals the input (the 4n+1 rule is assumed from the model).
