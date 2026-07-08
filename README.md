# GemmaClip

GemmaClip is a Track 2 AMD Developer Hackathon video captioning agent. It reads `/input/tasks.json`, downloads each video, probes metadata with `ffprobe`, extracts representative frames with `AKS-lite`, uses Fireworks-hosted multimodel inference to build factual evidence and captions, optionally verifies/refines them, and writes `/output/results.json`.

Runtime model configuration:

- `FIREWORKS_API_KEY` or `GEMMA_API_KEY` must be injected at runtime.
- No secrets are baked into the Docker image, README examples, source code, or tests.
- Default vision model: `accounts/fireworks/models/qwen3p7-plus`.
- Default text model: `accounts/fireworks/models/deepseek-v4-pro`.
- Set `GEMMA_MODEL` to override both vision and text model selection, which is useful for local Gemma 4 deployment testing.
- `GEMMA_VISION_MODEL` and `GEMMA_TEXT_MODEL` can override the default split model choices independently.
- If a configured model is unavailable to the runtime key, GemmaClip tries fallback models from `GEMMA_FALLBACK_MODELS`, or `accounts/fireworks/models/kimi-k2p6` by default.
- `GEMMACLIP_DISABLE_VERIFIER=true` skips the optional verifier/refiner pass.

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
docker buildx build --platform linux/amd64 --tag ghcr.io/armmyc/gemmaclip:v2 --push .
```
