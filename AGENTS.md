# AGENTS.md

## Project

GemmaClip is a Track 2 AMD Developer Hackathon video-captioning agent. It reads video-captioning tasks from `/input/tasks.json` and writes valid JSON to `/output/results.json`.

Primary goal: maximize caption accuracy and style match for four styles:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

The system must make Gemma 4 central to the pipeline.

## Competition constraints

- Container must read `/input/tasks.json` on startup.
- Container must write `/output/results.json` before exiting.
- Exit code must be `0` on success.
- Runtime limit is 10 minutes.
- Output must be valid JSON.
- Missing requested styles score zero for that clip.
- Do not hardcode sample-clip answers.
- All responses must be in English.
- Docker image must support `linux/amd64`.

## Target pipeline

Implement in this order:

1. Input/output harness
2. Video download
3. Video metadata probing
4. Frame extraction
5. Gemma 4 factual video summary
6. Gemma 4 caption generation
7. Gemma 4 verifier/reranker
8. JSON validation and fallback repair

Recommended runtime flow:

```text
/input/tasks.json
  -> parse tasks
  -> download each video_url
  -> probe duration, fps, resolution
  -> extract candidate frames
  -> select representative frames
  -> Gemma 4 factual evidence summary
  -> Gemma 4 caption candidates per style
  -> Gemma 4 rubric verification
  -> /output/results.json
```

## Downloader and frame extraction choice

Use Python `httpx` or `requests` for downloading remote MP4 files. Do not use ffmpeg as the primary downloader.

Use `ffprobe` for metadata and `ffmpeg` for frame extraction. Prefer ffmpeg over OpenCV for the first production path because it is more reliable inside Docker for MP4 decoding.

Recommended dependency split:

- Download: `httpx`
- Metadata: `ffprobe`
- Frame extraction: `ffmpeg`
- Image handling and optional scoring: `Pillow`, `opencv-python-headless`, `numpy`

## Frame extraction plan

Start with a reliable baseline, then upgrade.

### Baseline

Extract uniformly spaced frames:

- 30 to 45 seconds: 10 to 12 frames
- 45 to 90 seconds: 12 to 16 frames
- 90 to 120 seconds: 16 to 20 frames

### Upgrade

Implement AKS-lite candidate selection:

- uniform coverage
- scene-change detection
- motion score
- sharpness score
- visual diversity

Do not implement a trainable frame selector until the full pipeline works.

## Captioning rules

Always create a factual evidence object before writing styled captions.

Suggested evidence schema:

```json
{
  "scene": "",
  "main_subjects": [],
  "actions": [],
  "setting": "",
  "visible_objects": [],
  "mood": "",
  "camera_notes": "",
  "uncertain_details": []
}
```

Caption constraints:

- 12 to 25 words per caption
- faithful to the video evidence
- no invented objects, people, speech, brands, or locations
- sarcastic must be dry and lightly ironic, not mean
- humorous_tech must include technology or programming-flavored humor
- humorous_non_tech must avoid technical jargon

## Output format

For each task, output exactly:

```json
{
  "task_id": "v1",
  "captions": {
    "formal": "...",
    "sarcastic": "...",
    "humorous_tech": "...",
    "humorous_non_tech": "..."
  }
}
```

Only include styles requested in the task, unless tests explicitly require all four.

## Repository structure to create

```text
GemmaClip/
  AGENTS.md
  README.md
  pyproject.toml
  Dockerfile
  src/gemmaclip/
    __init__.py
    main.py
    io.py
    download.py
    video.py
    frames.py
    gemma_client.py
    prompts.py
    captioner.py
    validate.py
  tests/
    test_io.py
    test_validate.py
  examples/
    tasks.json
```

## Implementation rules

- Keep functions small and testable.
- Avoid global mutable state.
- Never commit API keys or `.env` files.
- Use clear error messages.
- If one video fails, return safe fallback captions for that task rather than crashing the entire batch.
- Validate JSON before writing output.
- Prefer deterministic settings for final caption generation unless candidate generation is intentionally enabled.

## First milestone

Create a working container that:

1. Reads `/input/tasks.json`.
2. Downloads each `video_url`.
3. Extracts frames to `/tmp/gemmaclip/frames/<task_id>/`.
4. Writes placeholder captions to `/output/results.json`.
5. Exits successfully.

After this milestone, replace placeholder captions with Gemma 4 factual summaries and generated captions.
