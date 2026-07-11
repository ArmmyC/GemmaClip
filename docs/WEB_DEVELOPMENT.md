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

- `GEMMACLIP_WEB_RUNS_DIR` — filesystem run root; default `.gemmaclip/runs`.
- `GEMMACLIP_WEB_MAX_UPLOAD_BYTES` — upload limit; default 200 MiB.
- `GEMMACLIP_WEB_HOST` and `GEMMACLIP_WEB_PORT` — API bind address and port.
- `GEMMACLIP_WEB_CORS_ORIGINS` — comma-separated development origins; CORS is disabled when unset.
- Existing routed-Gemma variables, including `FIREWORKS_API_KEY` or `GOOGLE_API_KEY`, configure captioning and are never returned by the API.
- `VITE_GEMMACLIP_API_BASE_URL` — optional public API origin; empty uses same-origin `/api`.

The production client does not silently fall back to mock data. The preserved `lovable/` directory remains the design snapshot.

Each run stores its upload, six selected frames, sanitized evidence, and captions below a server-generated run ID. Generated media is ignored by Git and never stored under frontend static assets.

## Current scope

The working slice is upload → metadata → existing hybrid extraction → existing routed Gemma pipeline → Quick results → inspection of that same run in Gemma Lab. Interactive Lab reruns and experiment comparison are visibly disabled until persistent stage APIs exist. Temporary audio is cleaned by the existing pipeline, so the Audio page truthfully reports that no playable artifact is available.

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
