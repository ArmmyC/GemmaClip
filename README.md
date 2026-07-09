# GemmaClip

GemmaClip is a Track 2 AMD Developer Hackathon video captioning agent. It reads `/input/tasks.json`, downloads each video, probes metadata with `ffprobe`, extracts representative frames with `AKS-lite`, builds factual evidence, generates styled captions, optionally verifies them, and writes `/output/results.json`.

Runtime model configuration:

- Default provider selection:
  - `google` when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is present.
  - `fireworks` otherwise, using the existing Fireworks/Gemma path.
- Override provider selection with `GEMMACLIP_PROVIDER=google` or `GEMMACLIP_PROVIDER=fireworks`.
- Google provider:
  - API keys: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - Default model: `gemini-3.5-flash`
  - Override with `GEMINI_MODEL`
- Fireworks provider:
  - API keys: `GEMMA_API_KEY` or `FIREWORKS_API_KEY`
  - Default vision model: `accounts/fireworks/models/qwen3p7-plus`
  - Default text model: `accounts/fireworks/models/deepseek-v4-pro`
  - Set `GEMMA_MODEL` to override both vision and text model selection, which is useful for local Gemma 4 deployment testing.
  - `GEMMA_VISION_MODEL` and `GEMMA_TEXT_MODEL` can override the default split model choices independently.
  - If a configured model is unavailable to the runtime key, GemmaClip tries fallback models from `GEMMA_FALLBACK_MODELS`, or `accounts/fireworks/models/kimi-k2p6` by default.
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
