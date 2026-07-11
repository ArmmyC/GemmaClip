# GemmaClip

GemmaClip has two independent surfaces: the Track 2 leaderboard CLI and an optional FastAPI + React web demo. The web demo provides a real Quick Caption workflow through the same routed Gemma pipeline and opens its stored results in Gemma Lab.

See [docs/WEB_DEVELOPMENT.md](docs/WEB_DEVELOPMENT.md) for installation, environment variables, run storage, validation, and the manual Lab stage flow.

Gemma Lab supports two explicit paths after upload: `Open manual Lab` probes metadata and waits for stage actions; `Run automatically` starts the existing Quick Caption flow. Manual stages persist Frames, Audio, Evidence, and Captions through the shared Python services. Experiments are immutable snapshots that can be compared from the Compare stage. Fireworks audio-visual inference remains optional; failed audio inference drops audio before the Google Gemma 4 31B visual fallback.

Web runs record whether captions were `model_generated`, produced through an `evidence_fallback`, or rejected as a `deterministic_fallback`. Evidence fallbacks remain inspectable with a degraded-result notice; deterministic fallback is never presented as successful Gemma output. Demo run retention defaults to 24 hours and is configurable with `GEMMACLIP_WEB_RUN_TTL_SECONDS`.

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

## Why Gemma is essential

The `routed_gemma` provider makes Gemma 4 the reasoning and writing core. Gemma 4 26B A4B produces visual evidence when audio is absent, silent, or unsafe to process. Gemma 4 12B Unified produces routed audio-visual evidence when a bounded audio window is useful and runtime permits. Gemma 4 31B then receives the six chronological hybrid frames, structured evidence, requested styles, and exact output schema to produce final captions.

Fireworks is the primary endpoint for each role. Visual evidence falls back to Google Gemma 4 31B. Fireworks audio-visual inference is optional: if it is unavailable or invalid, GemmaClip drops audio and continues through Google Gemma 4 31B using visual frames only. Caption synthesis falls back from Fireworks 31B to Google 31B. No non-Gemma model is silently substituted, and audio is never sent to Google 31B in this fallback configuration.

```text
GOOGLE_GEMMA_VISUAL_MODEL=gemma-4-31b-it
GOOGLE_GEMMA_CAPTION_MODEL=gemma-4-31b-it
```

Routed calls recheck the live deadline before every Fireworks or Google attempt. Audio preprocessing is skipped below its runtime threshold, and the full degradation ladder is reapplied afterward. Final synthesis falls back to grounded evidence captions when fewer than 70 seconds remain, while optional repair preserves valid captions and fills missing styles locally below the same threshold. RMS is only an energy heuristic, not confirmation of speech.

Routed stage temperatures are configurable and are conservative starting values, not proven optima: `GEMMACLIP_ROUTED_EVIDENCE_TEMPERATURE=0.0`, `GEMMACLIP_ROUTED_CAPTION_TEMPERATURE=0.4`, `GEMMACLIP_ROUTED_REPAIR_TEMPERATURE=0.25`, and `GEMMACLIP_ROUTED_SINGLE_CALL_TEMPERATURE=0.4`.

Example without embedded credentials:

```bash
export GEMMACLIP_PROVIDER=routed_gemma
export GEMMACLIP_AUDIO_MODE=auto
export FIREWORKS_API_KEY="${FIREWORKS_API_KEY}"
export GOOGLE_API_KEY="${GOOGLE_API_KEY}"
python -m gemmaclip.main --input examples/tasks.json --output output/results.json
```

See [Gemma routing](docs/GEMMA_ROUTING.md) and [Gemma audio](docs/GEMMA_AUDIO.md) for configuration and failure behavior.

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
