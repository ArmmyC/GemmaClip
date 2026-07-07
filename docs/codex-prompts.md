# Codex Prompt Plan

Use these prompts sequentially. Do not ask Codex to implement the full Gemma 4 pipeline at once.

## Prompt 1: IO, downloader, metadata, and placeholder output

```text
Read AGENTS.md and docs/track2-guide.md first.

Implement GemmaClip milestone 1.

Create a Python package under src/gemmaclip with:
- __init__.py
- main.py
- io.py
- download.py
- video.py
- frames.py
- validate.py

Add:
- pyproject.toml
- Dockerfile
- examples/tasks.json
- tests/test_io.py
- tests/test_validate.py

Behavior:
- Default input path: /input/tasks.json
- Default output path: /output/results.json
- Allow CLI overrides: --input and --output
- Read tasks shaped like [{task_id, video_url, styles}]
- Download each video_url using httpx streaming
- Use clear timeouts and HTTP error handling
- Save videos to /tmp/gemmaclip/videos/<safe_task_id>.mp4
- Use ffprobe to return duration, fps, width, height, and frame_count when available
- Use ffmpeg to extract uniformly spaced JPEG frames into /tmp/gemmaclip/frames/<safe_task_id>/
- Select frame counts based on duration:
  - <=45s: 12 frames
  - <=90s: 16 frames
  - otherwise: 20 frames
- Generate placeholder captions for every requested style
- Validate that every output item has task_id and captions
- Validate that every requested style exists
- Write valid JSON to the output path
- Exit 0 if all tasks are processed or fallbacks are written
- If one task fails, write fallback captions for that task and continue

Docker:
- Use python:3.12-slim or similar
- Install ffmpeg through apt
- Make the container run python -m gemmaclip.main
- Ensure it is suitable for linux/amd64 builds

Tests:
- Test reading valid tasks
- Test rejecting malformed tasks
- Test output validation catches missing styles
- Test output validation accepts a correct result

Do not implement Gemma API calls yet.
Do not implement AKS-lite yet.
Do not hardcode example clip answers.
```

## Prompt 2: Gemma 4 client and factual summary prompts

```text
Read AGENTS.md and docs/track2-guide.md first.

Add Gemma 4 model integration while preserving milestone 1 behavior.

Create:
- src/gemmaclip/gemma_client.py
- src/gemmaclip/prompts.py
- src/gemmaclip/captioner.py

Requirements:
- Read model/API config from environment variables.
- Do not commit or require .env files for submission.
- Support a dry-run mode that uses placeholder captions without API calls.
- Convert selected frame JPEGs to base64 image inputs if the configured Gemma endpoint supports image messages.
- First ask Gemma 4 for a structured factual evidence JSON object.
- Then ask Gemma 4 to generate captions only from that evidence.
- Return captions only for requested styles.
- Preserve the fallback behavior if Gemma calls fail.

Factual evidence schema:
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

Caption rules:
- 12 to 25 words per caption
- accurate before funny
- no invented objects, speech, brands, locations, or events
- formal: professional, objective, factual
- sarcastic: dry, ironic, lightly mocking, not mean
- humorous_tech: funny with programming or technology references
- humorous_non_tech: funny everyday humor, no technical jargon

Add tests for parsing model JSON and fallback behavior.
```

## Prompt 3: AKS-lite adaptive frame selection

```text
Read AGENTS.md and docs/track2-guide.md first.

Add AKS-lite frame selection as an optional upgrade after uniform extraction.

Create:
- src/gemmaclip/aks_lite.py

Behavior:
- Start from candidate frames extracted at approximately 1 FPS, capped at 48 candidates.
- Score frames using:
  - scene-change score based on histogram difference
  - motion score based on pixel difference from previous candidate
  - sharpness score using Laplacian variance
  - diversity penalty to avoid near-duplicates
- Always preserve beginning, middle, and end coverage.
- Select final frame count based on video duration:
  - <=45s: 12 frames
  - <=90s: 16 frames
  - otherwise: 20 frames
- Sort selected frames by timestamp before sending to Gemma.
- Keep uniform extraction as a fallback.

Add tests for:
- selecting no more than k frames
- preserving chronological order
- handling small candidate lists
```

## Prompt 4: caption candidate generation and verifier

```text
Read AGENTS.md and docs/track2-guide.md first.

Upgrade caption generation to produce multiple candidates and rerank them.

Requirements:
- For each requested style, generate 3 candidate captions.
- Add a Gemma 4 verifier prompt that scores each candidate on:
  - accuracy
  - style_match
  - hallucination_risk
  - length_ok
- Select the best caption per style.
- If verifier output is invalid, choose the first valid candidate.
- Enforce no missing requested styles.
- Repair captions that are empty, too long, or obviously wrong style.

Keep runtime under the 10-minute limit for about 12 clips.
```

## Prompt 5: Docker and CI hardening

```text
Read AGENTS.md and docs/track2-guide.md first.

Harden the project for submission.

Requirements:
- Add GitHub Actions for tests and Docker build.
- Confirm Docker build supports linux/amd64.
- Add README usage instructions.
- Add local test command using examples/tasks.json.
- Add a makefile or scripts for:
  - test
  - run-local
  - docker-build
  - docker-run-example
- Ensure no API keys, .env files, downloaded videos, extracted frames, or output files are committed.
- Add .gitignore.
```
