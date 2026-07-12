# Web development

GemmaClip's web demo is an additional entrypoint. The leaderboard CLI remains Python-only and continues to read `/input/tasks.json` and write `/output/results.json` without Node, FastAPI, or frontend assets.

## Install and run

```powershell
python -m pip install -e ".[web,dev]"
python -m gemmaclip.web
```

In another terminal:

```powershell
cd web
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. Production builds use `npm run build` and default to the real API.

## Configuration

`GEMMACLIP_WEB_RUN_TTL_SECONDS` controls retention in seconds and defaults to `86400` (24 hours). A value of `0` or less disables automatic deletion.

- `GEMMACLIP_WEB_RUNS_DIR` — filesystem run root; default `.gemmaclip/runs`.
- `GEMMACLIP_WEB_MAX_UPLOAD_BYTES` — upload limit; default 200 MiB.
- `GEMMACLIP_WEB_HOST` and `GEMMACLIP_WEB_PORT` — API bind address and port.
- `GEMMACLIP_WEB_CORS_ORIGINS` — comma-separated development origins; CORS is disabled when unset.
- Existing routed-Gemma variables, including `FIREWORKS_API_KEY` or `GOOGLE_API_KEY`, configure captioning and are never returned by the API.
- `GOOGLE_GEMMA_VISUAL_MODEL=gemma-4-31b-it` and `GOOGLE_GEMMA_CAPTION_MODEL=gemma-4-31b-it` configure the default Google visual and writing roles.
- `VITE_GEMMACLIP_API_BASE_URL` — optional public API origin; empty uses same-origin `/api`.

The production client does not silently fall back to mock data. The preserved `lovable/` directory remains the design snapshot.

Gemma Lab now supports a manual stage flow in addition to Quick Caption. `/lab` creates a run, probes metadata, and waits for the user to run Frames, Audio, Evidence, and Captions individually. Each stage persists its configuration and artifact through the API, invalidates only its downstream dependents, and exposes a recoverable stage error. `Save Experiment` stores an immutable snapshot; Compare reads two real snapshots from the same run.

Manual stage endpoints are `POST /api/runs/{run_id}/metadata`, `POST /api/runs/{run_id}/frames`, `PATCH /api/runs/{run_id}/frames/selection`, `POST /api/runs/{run_id}/audio`, `POST /api/runs/{run_id}/evidence`, `POST /api/runs/{run_id}/captions`, `POST /api/runs/{run_id}/experiments`, and `GET /api/runs/{run_id}/compare?left={experiment_id}&right={experiment_id}`.

Audio supports the MVP's highest-energy and first-window (first-N-seconds) selectors. Custom waveform ranges remain intentionally unavailable until a persisted range model is added.

Interactive evidence uses a consistent six-frame minimum. The Fast preset therefore selects six uniform frames with audio disabled, and the API rejects smaller frame sets before creating a stage job.

Stage jobs use a single in-memory run lock. Only one job may mutate a run at a time, conflicting requests return `409`, active runs cannot be deleted, and interrupted processing is recovered as a safe error on restart. This remains intentionally single-process for the demo; use an external queue and shared lock before production deployment.

Each run stores its upload, six selected frames, sanitized evidence, and captions below a server-generated run ID. Generated media is ignored by Git and never stored under frontend static assets.

Runs expose one generation outcome:

- `model_generated`: a remote Gemma call produced caption output. A small number of missing styles may still be completed locally from evidence.
- `evidence_fallback`: Gemma produced valid structured evidence, but final writing failed or became runtime-unsafe. The run remains ready, is marked degraded, and the UI explains the fallback.
- `deterministic_fallback`: no valid Gemma evidence or captions succeeded. The web job becomes an error and does not publish generic leaderboard fallback captions as successful demo output.

At API startup, interrupted `processing` runs are marked error without deleting artifacts. Startup and new uploads remove expired pending, ready, or error runs; active jobs are retained and cannot be deleted through the API. The executor is intentionally single-process and in-memory: jobs are not resumed after restart, and internally owned worker threads shut down through FastAPI lifespan handling.

Caption cards report whether visual and caption-safe audio grounding context was available. Exact per-caption or sentence-level attribution is not tracked and is not claimed.

Evidence results report the actual safe provider, model, and modality plus whether an audio fallback occurred. A successful Fireworks-to-Google fallback remains `model_generated` and is not degraded. When Fireworks audio-visual inference is unavailable, the web run explains that audio was dropped and Google Gemma 4 31B continued with frames only.

## Current scope

The working slice supports both `upload → metadata → manual Frames → manual Audio → manual Evidence → manual Captions → immutable experiments → Compare` and the original Quick Caption automatic flow. Both flows call the same Python stage services. Temporary audio candidates are removed after Audio inspection and Evidence generation; the Audio page exposes safe metadata and waveform data but does not retain a playable private artifact.

## Verification

```powershell
python -m compileall src tests
pytest
cd web
npm run typecheck
npm run lint
npm test
npm run build
```
