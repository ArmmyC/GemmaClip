# GemmaClip Track 2 Implementation Guide

This document turns the AMD Developer Hackathon ACT II participant guide into implementation requirements for GemmaClip.

## Selected track

GemmaClip targets **Track 2: Video Captioning Agent**.

The agent receives short video clips and must generate captions in requested styles. The hidden evaluation clips are unseen, so the system must generalize beyond the public examples.

## Required input

At container startup, read tasks from:

```text
/input/tasks.json
```

Expected shape:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://storage.example.com/clips/clip1.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

## Required output

Before exiting, write results to:

```text
/output/results.json
```

Expected shape:

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

For every task, output a caption for every style requested in that task. Missing styles score zero for that clip.

## Supported styles

| Style | Required behavior |
| --- | --- |
| `formal` | Professional, objective, factual tone |
| `sarcastic` | Dry, ironic, lightly mocking |
| `humorous_tech` | Funny with technology or programming references |
| `humorous_non_tech` | Funny everyday humor with no technical jargon |

## Scoring target

Each caption is scored by an LLM judge on:

1. **Caption accuracy**: how faithfully the caption reflects the video.
2. **Style match**: how well the caption matches the requested tone.

The final score is a weighted average across clips and styles. Local design decisions should be judged against these two dimensions.

## Track 2 rules

- Exit code `0` on success.
- Maximum runtime: 10 minutes.
- No model restriction for Track 2.
- No API key or model restriction is injected for Track 2.
- You may call any model, API, or framework, but Gemma 4 should be central because this repo is also targeting the Gemma challenge.
- Hidden evaluation videos are 30 seconds to 2 minutes long.
- `/output/results.json` must be valid JSON.
- Docker compressed image size must not exceed 10GB.
- Submissions are rate-limited to 10 per hour per team.
- Responses must be in English.
- Do not hardcode or cache answers to public example clips.
- The judging VM runs `linux/amd64`; Docker images must include a `linux/amd64` manifest.

## Public example clips

Use these only for development and smoke tests. Do not specialize prompts, code, or constants to these clips.

| Clip | Content |
| --- | --- |
| `v1` | Urban autumn boulevard with golden trees and city traffic |
| `v2` | Orange kitten among green foliage in a garden |
| `v3` | Office worker at a desktop computer in a modern open-plan office |

## Production pipeline

```text
/input/tasks.json
  -> parse tasks
  -> download video_url
  -> probe video metadata
  -> extract representative frames
  -> Gemma 4 factual video evidence
  -> Gemma 4 caption candidate generation
  -> Gemma 4 verifier/reranker
  -> validate output JSON
  -> /output/results.json
```

## First milestone

Implement this first before adding model calls:

1. Read `/input/tasks.json`.
2. Download every `video_url`.
3. Probe metadata with `ffprobe`.
4. Extract uniformly spaced frames with `ffmpeg`.
5. Generate placeholder captions for every requested style.
6. Validate output structure.
7. Write `/output/results.json`.
8. Exit with code `0`.

## Downloader and media tooling

Use this split:

- Download remote files with `httpx` or `requests`.
- Probe media with `ffprobe`.
- Extract frames with `ffmpeg`.
- Use OpenCV later only for AKS-lite scoring, such as motion, scene change, sharpness, and diversity.

Do not use `ffmpeg` as the primary HTTP downloader. Downloading in Python gives clearer retries, timeout handling, status checks, and file-size control.

## Frame extraction plan

### Phase 1: uniform baseline

Use timestamp-based extraction.

Suggested frame counts:

| Duration | Frames |
| ---: | ---: |
| 30 to 45 sec | 10 to 12 |
| 45 to 90 sec | 12 to 16 |
| 90 to 120 sec | 16 to 20 |

### Phase 2: AKS-lite

Add adaptive selection after the full pipeline works.

Signals:

- uniform temporal coverage
- scene-change score
- motion score
- sharpness score
- diversity filter

Purpose: avoid missing short important moments while avoiding redundant frames.

## Gemma 4 usage plan

Gemma 4 should not be a thin final rewriter. Use it in multiple stages:

1. Frame-level understanding.
2. Factual video evidence generation.
3. Four-style caption candidate generation.
4. Rubric-based verification and reranking.

Suggested factual evidence schema:

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
- Accurate before funny.
- Use only visual or audio evidence.
- Do not invent speech, brands, locations, identities, objects, or events.
- Sarcasm must stay light, not hostile.
- `humorous_tech` must include a clear tech/programming flavor.
- `humorous_non_tech` must avoid tech jargon.

## Failure policy

If a task fails internally, do not crash the whole batch. Return safe fallback captions for that task and continue.

Fallback captions must:

- include all requested styles;
- be valid English;
- avoid pretending to know specific details not observed.

Example generic fallback shape:

```json
{
  "formal": "The video shows a short scene with visible activity and environmental details.",
  "sarcastic": "A brief video appears, bravely asking us to understand it without extra context.",
  "humorous_tech": "The clip renders a real-world scene while our captioning pipeline tries not to segfault.",
  "humorous_non_tech": "The clip shows a moment doing its best to become a tiny story."
}
```

Use fallback only for hard failures. A generic fallback will likely score lower than grounded captions.

## Local test command pattern

```bash
mkdir -p input output
python -m gemmaclip.main --input input/tasks.json --output output/results.json
```

Docker build for Apple Silicon or cross-platform environments:

```bash
docker buildx build --platform linux/amd64 --tag gemmaclip:latest .
```

Public submission images should be pushed with a `linux/amd64` manifest.
