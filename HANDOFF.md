# HANDOFF

## Current milestone: 1 done (probe + chunk plan + --dry-run). Next: milestone 2.

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
uv sync && uv run pytest -q          # 47 tests, ~1 s, needs ffmpeg
uv run vsr run <clip> --profile seedvr2_3b --dry-run
```

### Next (milestone 2: cut + stitch + master with identity fake model)
- Run the cut commands into jobs/<id>/chunks/. Each cut re-decodes from frame 0 (trim): fine for
  <= 60 s; otherwise decode once into a lossless intermediate and slice that.
- Stitch: stream decoded frames (rawvideo pipes or numpy), crossfade overlap linearly, drop pad
  frames, assert output frame count == range frames. Write stitched intermediate (e.g. FFV1 .mkv).
- Master: when the stitched frames are RGB, set the YUV matrix explicitly
  (`scale=out_color_matrix=bt709:out_range=tv`) or swscale defaults to bt601.
- Identity fake-model test via `ffmpeg -f lavfi -i testsrc2`.

### Notes / open questions
- System ffmpeg is 4.4 (Ubuntu 22.04): no `-fps_mode`; `ffmpeg.vsync()` picks `-vsync` on < 5.1.
- Chunk intermediates are yuv420p 8-bit: loses chroma/bit depth for 10-bit or 4:4:4 sources.
  Codec is configurable in code (`ChunkCodec`) but not yet from the CLI/profile.
- ComfyUI 0.44.0 reachable at http://127.0.0.1:8188 from this environment (checked
  /system_stats, /queue, /history). History was empty, so the real output-entry shape
  (`images`/`gifs`/`videos`) still has to be checked on a real run in milestone 3.
- SaveVideo (node 76) uses format/codec `auto` (likely lossy h264). Its schema offers other codecs
  (ProRes -> MOV). In milestone 3, consider an override for a higher-quality output codec.
- The SeedVR2 workflow has LoadVideo -> GetVideoComponents -> CreateVideo with the input fps; check
  that the output frame count equals the input (the 4n+1 rule is assumed from the model).
