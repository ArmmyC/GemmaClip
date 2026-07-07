# AGENTS.md

## Project

GemmaClip is a Track 2 AMD Developer Hackathon video-captioning agent. It reads video-captioning tasks from `/input/tasks.json` and writes valid JSON to `/output/results.json`.

Primary goal: maximize caption accuracy and style match for four styles:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

The system must make Gemma 4 central to the final captioning pipeline because the team is also targeting the Gemma challenge.

Before making changes, read:

1. `docs/track2-guide.md`
2. `docs/codex-prompts.md`

## Competition constraints for Track 2

- Container must read `/input/tasks.json` on startup.
- Container must write `/output/results.json` before exiting.
- Exit code must be `0` on success.
- Maximum runtime is 10 minutes.
- No inference log is required for Track 2.
- Track 2 does not inject API keys or model restrictions.
- Track 2 allows any model, API, or framework, but this repo should use Gemma 4 as the main reasoning/captioning model.
- Hidden videos are 30 seconds to 2 minutes long.
- Output must include a caption for every requested style for every clip.
- Missing requested styles score zero for that clip.
- `/output/results.json` must be valid JSON.
- All responses must be in English.
- Do not hardcode or cache answers to specific clips.
- Evaluation uses unseen clips, not only the public examples.
- Docker image compressed size must not exceed 10GB.
- Container image must be publicly pullable at submission time.
- The judging VM runs `linux/amd64`; Docker images must include a `linux/amd64` manifest.

## Scoring target

Each caption is judged on:

1. **Caption accuracy**: faithfulness to the video content.
2. **Style match**: fit to the requested tone.

Every implementation choice should improve one of these two metrics or protect valid output generation.

## Target pipeline

Implement in this order:

1. Input/output harness
2. Video download
3. Video metadata probing
4. Uniform frame extraction
5. Placeholder output and validation
6. Gemma 4 factual video summary
7. Gemma 4 caption generation
8. Gemma 4 verifier/reranker
9. AKS-lite adaptive frame selection
10. Docker and CI hardening

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

Use Python `httpx` or `requests` for downloading remote MP4 files. Do not use `ffmpeg` as the primary downloader.

Use `ffprobe` for metadata and `ffmpeg` for frame extraction. Prefer `ffmpeg` over OpenCV for the first production extraction path because it is more reliable inside Docker for MP4 decoding.

Recommended dependency split:

- Download: `httpx`
- Metadata: `ffprobe`
- Frame extraction: `ffmpeg`
- Optional AKS-lite scoring: `Pillow`, `opencv-python-headless`, `numpy`

## Frame extraction plan

Start with a reliable baseline, then upgrade.

### Baseline

Extract uniformly spaced frames by timestamp:

- 30 to 45 seconds: 10 to 12 frames
- 45 to 90 seconds: 12 to 16 frames
- 90 to 120 seconds: 16 to 20 frames

### Upgrade

Implement AKS-lite candidate selection:

- uniform temporal coverage
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

- 12 to 25 words per caption.
- Faithfulness beats humor.
- No invented objects, people, speech, brands, locations, identities, or events.
- `sarcastic` must be dry and lightly ironic, not hostile.
- `humorous_tech` must include technology or programming-flavored humor.
- `humorous_non_tech` must avoid technical jargon.

## Output format

For each task, output:

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

Only include styles requested in the task unless tests explicitly require all four.

## Repository structure to create

```text
GemmaClip/
  AGENTS.md
  README.md
  pyproject.toml
  Dockerfile
  docs/
    track2-guide.md
    codex-prompts.md
  src/gemmaclip/
    __init__.py
    main.py
    io.py
    download.py
    video.py
    frames.py
    aks_lite.py
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
- Do not specialize prompts or code for the public example clips.

## First milestone

Create a working container that:

1. Reads `/input/tasks.json`.
2. Downloads each `video_url`.
3. Extracts frames to `/tmp/gemmaclip/frames/<task_id>/`.
4. Writes placeholder captions to `/output/results.json`.
5. Exits successfully.

After this milestone, replace placeholder captions with Gemma 4 factual summaries and generated captions.
