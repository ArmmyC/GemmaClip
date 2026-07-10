# GemmaClip

GemmaClip is a Track 2 AMD Developer Hackathon video captioning agent. It reads `/input/tasks.json`, downloads each video, probes metadata with `ffprobe`, extracts representative frames with `AKS-lite`, builds factual evidence, generates styled captions, optionally verifies them, and writes `/output/results.json`.

## Runtime limit

The container is bounded to 590 seconds, including startup, downloads, media processing, and model calls. The application writes valid fallback captions before it starts remote work, uses a 570-second soft batch budget, limits each download to 30 seconds, and limits each `ffprobe` or individual `ffmpeg` command to 15 seconds. If time runs low or a task operation fails, it preserves valid fallback captions for unfinished tasks rather than risking the 10-minute competition limit.

Runtime model configuration:

- Default provider selection:
  - `google` when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is present.
  - `fireworks` otherwise, using the existing Fireworks/Gemma path.
- Override provider selection with `GEMMACLIP_PROVIDER=google` or `GEMMACLIP_PROVIDER=fireworks`.
- Google provider:
  - Uses Gemma 4 on the Gemini API
  - API keys: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - Default model: `gemma-4-26b-a4b-it`
  - Optional stronger model: `gemma-4-31b-it`
  - Override with `GEMINI_MODEL`
- Fireworks provider:
  - API keys: `GEMMA_API_KEY` or `FIREWORKS_API_KEY`
  - Default vision model: `accounts/fireworks/models/qwen3p7-plus`
  - Default text model: `accounts/fireworks/models/deepseek-v4-pro`
  - Set `GEMMA_MODEL` to override both vision and text model selection, which is useful for local Gemma 4 deployment testing.
  - `GEMMA_VISION_MODEL` and `GEMMA_TEXT_MODEL` can override the default split model choices independently.
  - If a configured model is unavailable to the runtime key, GemmaClip tries fallback models from `GEMMA_FALLBACK_MODELS`, or `accounts/fireworks/models/kimi-k2p6` by default.
- Fireworks judge provider:
  - Set `GEMMACLIP_PROVIDER=fireworks_judge` to use six separate 512px frames, direct caption generation, and a visual judge/rewrite pass.
  - Required: `FIREWORKS_API_KEY`.
  - Optional: `FIREWORKS_BASE_URL`, `FIREWORKS_VISION_MODEL`, and `FIREWORKS_FALLBACK_VISION_MODEL`.
  - Defaults: `accounts/fireworks/models/minimax-m3` followed by `accounts/fireworks/models/qwen3p7-plus`.
  - Replace `FIREWORKS_VISION_MODEL` with a Gemma model identifier later without changing the pipeline.
- `GEMMACLIP_DISABLE_VERIFIER=true` skips the optional verifier/refiner pass.
- `GEMMACLIP_FORCE_PLACEHOLDER=true` and `GEMMACLIP_FORCE_FALLBACK=true` remain available for control runs.
- No literal API keys are committed in the Dockerfile, README examples, source code, or tests.

Local run:

```bash
python -m pip install -e .[dev]
python -m gemmaclip.main --input examples/tasks.json --output output/results.json
```

Frame debug artifacts:

```bash
python -m gemmaclip.main \
  --input examples/tasks.json \
  --output output/results.json \
  --workdir /tmp/gemmaclip \
  --debug-dir debug-frames
```

This stores downloaded videos under `<workdir>/videos/`, extracted frames under `<workdir>/frames/<task_id>/`, and writes `<workdir>/frame_manifest.json`. When `--debug-dir` is set, GemmaClip also copies frames into `<debug-dir>/<task_id>/` and writes `<debug-dir>/<task_id>_contact_sheet.jpg`.

PowerShell venv activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Make targets:

```bash
make install-dev
make test
make run-local INPUT=examples/tasks.json OUTPUT=output/results.json
```

Docker build and push:

```bash
docker buildx build --platform linux/amd64 --tag ghcr.io/armmyc/gemmaclip:v3 --push .
```

Temporary Gemini build-arg example for leaderboard submission:

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-arg GEMINI_API_KEY="$GEMINI_API_KEY" \
  --tag ghcr.io/armmyc/gemmaclip:v3 \
  --push .
```

Warning: the build-arg pattern above copies the temporary key into the image environment. Use it only for the event flow, and revoke that key after the event.
