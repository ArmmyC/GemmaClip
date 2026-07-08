# GemmaClip

GemmaClip is a Track 2 AMD Developer Hackathon baseline agent. This milestone reads `/input/tasks.json`, downloads each video, probes metadata with `ffprobe`, extracts uniform frames with `ffmpeg`, and writes placeholder captions to `/output/results.json`.

Runtime model configuration:

- `FIREWORKS_API_KEY` or `GEMMA_API_KEY` must be injected at runtime.
- No secrets are baked into the Docker image, README examples, source code, or tests.
- GemmaClip tries Gemma 4 first with `accounts/fireworks/models/gemma-4-31b-it`.
- If that model is unavailable to the runtime key, GemmaClip tries fallback models from `GEMMA_FALLBACK_MODELS`, or `accounts/fireworks/models/kimi-k2p6` by default.
- Set `GEMMA_MODEL` to point at a different primary model, including a local or dedicated Fireworks deployment path.

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

Docker build:

```bash
docker buildx build --platform linux/amd64 --tag gemmaclip:latest .
```
