<div align="center">

# GemmaClip

</div>

<p align="center">
  <strong>Grounded video captions with an inspectable Gemma 4 pipeline.</strong><br />
  Turn video captioning from a black box into a glass box.
</p>

<p align="center">
  <a href="https://gemmaclip.kamolpop.dev/">Live demo</a> ·
  <a href="#quick-caption">Quick Caption</a> ·
  <a href="#gemma-lab">Gemma Lab</a> ·
  <a href="#leaderboard-cli">Leaderboard CLI</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12 or newer" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-TypeScript-3178C6?style=flat-square&logo=react&logoColor=white" alt="React and TypeScript" />
  <img src="https://img.shields.io/badge/Docker-linux%2Famd64-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker linux amd64" />
</p>

> [!NOTE]
> GemmaClip has two independent delivery surfaces: a public web application for interactive captioning and a Python-only leaderboard container. Web changes must not make the competition entrypoint depend on Node, FastAPI, or frontend assets.

## Overview

GemmaClip gives everyday users a fast captioning flow and gives technical users a transparent lab for inspecting every important decision:

~~~text
Video -> Frames -> Audio -> Evidence -> Captions -> Compare
~~~

The web application is built around the existing Python pipeline. The browser never receives credentials, private provider URLs, raw media payloads, or hidden reasoning.

## Quick Caption

Quick Caption is the one-upload path for ordinary users:

1. Upload an MP4, WebM, or MOV video.
2. GemmaClip selects representative moments.
3. A bounded audio candidate is considered when useful.
4. Gemma chooses the safest evidence route.
5. Grounded captions are generated in four styles.

The Balanced preset uses:

- six chronological Hybrid frames;
- four temporal anchors plus two high-change frames;
- automatic audio routing;
- a maximum selected audio window of 30 seconds;
- automatic evidence routing and safe fallbacks.

Default caption styles are formal, sarcastic, humorous-tech, and humorous-non-tech.

## Gemma Lab

Gemma Lab exposes the same services stage by stage:

| Stage | What you can inspect |
| --- | --- |
| Video | Metadata, codec, duration, resolution, and audio presence |
| Frames | Uniform, AKS-Lite, or Hybrid selection with timestamps and reasons |
| Audio | Selected energy window, waveform, RMS, and playable audio |
| Evidence | Route decision, safe provider/model labels, structured evidence, and safe JSON |
| Captions | Styles, temperature, grounding status, and model metadata |
| Compare | Immutable snapshots and side-by-side experiment results |

Changing an upstream configuration invalidates only its dependent stages. A completed Quick Caption run can be opened in Gemma Lab without re-uploading or repeating completed work.

## Model routing

The public Gemma path uses GEMMACLIP_PROVIDER=routed_gemma.

~~~text
Visual route:
  Gemma 4 26B A4B evidence -> Gemma 4 31B captions

Audio-visual route:
  Gemma 4 12B Unified evidence -> Gemma 4 31B captions
~~~

The audio-visual evidence role may use an OpenAI-compatible AMD Cloud deployment through the AMD_GEMMA_AUDIO_VISUAL_* variables. If that endpoint is unavailable, GemmaClip removes the temporary audio candidate and continues with a visual-only route. Audio energy is only an energy candidate; it does not prove speech.

The pipeline protects grounding with:

- chronological frame ordering;
- bounded FFmpeg and FFprobe work;
- live runtime checks before provider attempts;
- normalized evidence and caption-safe audio facts;
- exact requested caption keys;
- valid partial captions and evidence-based fallbacks;
- progressive leaderboard result writes.

## Quick start with Docker

### 1. Configure providers

~~~bash
cp .env.example .env
~~~

On PowerShell:

~~~powershell
Copy-Item .env.example .env
~~~

Set the provider credentials and model IDs required by your deployment. The public routed setup commonly uses:

~~~text
GEMMACLIP_PROVIDER=routed_gemma
GEMMACLIP_AUDIO_MODE=auto
GOOGLE_API_KEY=...
AMD_GEMMA_AUDIO_VISUAL_API_KEY=...
AMD_GEMMA_AUDIO_VISUAL_BASE_URL=...
AMD_GEMMA_AUDIO_VISUAL_MODEL=...
~~~

Never commit .env, API keys, authorization headers, uploaded videos, extracted audio, generated frames, or run directories.

### 2. Start the web application

~~~bash
docker compose -f docker-compose.web.yml up --build
~~~

Open http://127.0.0.1:8000.

Check service readiness without exposing secrets:

~~~bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/config
~~~

The health endpoint checks storage, FFmpeg, FFprobe, provider configuration, and the job manager. It does not make a live provider request.

Stop the application with:

~~~bash
docker compose -f docker-compose.web.yml down
~~~

Run artifacts are stored in the named gemmaclip-runs volume, outside the static frontend image.

## Local development

### Backend

Requirements: Python 3.12 or newer and FFmpeg/FFprobe on PATH.

~~~bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# PowerShell
\.venv\Scripts\Activate.ps1

python -m pip install -e ".[web,dev]"
python -m gemmaclip.web
~~~

The FastAPI service listens on http://127.0.0.1:8000 by default.

### Frontend

In a second terminal:

~~~bash
cd web
npm install
npm run dev
~~~

Vite serves the React application and proxies /api to the local FastAPI service. Production builds use VITE_GEMMACLIP_API_BASE_URL, or same-origin /api when unset.

## Production deployment

The production web image is separate from the leaderboard image:

~~~bash
GEMMACLIP_WEB_PORT=8011 docker compose -f docker-compose.web.yml up -d --build
~~~

This runs Dockerfile.web, which builds the React app, installs FastAPI with FFmpeg/FFprobe, runs as an unprivileged user, and stores run artifacts in /data/runs.

### Reverse proxy

The repository includes an Nginx example at deploy/nginx/gemmaclip.kamolpop.dev.conf. It forwards:

~~~text
public HTTP/HTTPS -> Nginx -> 127.0.0.1:8011 -> GemmaClip container:8000
~~~

### Cloudflare Tunnel

When the host is behind CGNAT, inbound router forwarding cannot work. A Cloudflare Tunnel can expose the local web port without opening ports on the router:

~~~yaml
ingress:
  - hostname: gemmaclip.example.com
    service: http://127.0.0.1:8011
  - service: http_status:404
~~~

Create the DNS route with Cloudflare, run the tunnel on the server, and point the public hostname at 127.0.0.1:8011. Keep tunnel credentials outside this repository.

## Leaderboard CLI

The competition surface remains Python-only and independent from the web app. It reads /input/tasks.json and writes /output/results.json while preserving the batch timeout and safe fallback contract.

Install the package:

~~~bash
python -m pip install -e ".[dev]"
~~~

Run the mounted-contract-shaped example:

~~~bash
python -m gemmaclip.main \
  --input examples/tasks.json \
  --output output/results.json
~~~

Build and run the competition image:

~~~bash
docker build -f Dockerfile -t gemmaclip-cli:latest .
docker run --rm \
  -v "$PWD/examples:/input:ro" \
  -v "$PWD/output:/output" \
  gemmaclip-cli:latest
~~~

The leaderboard image does not require Node, React, FastAPI, or frontend assets.

## Web routes

| Route | Purpose |
| --- | --- |
| / | Quick Caption upload |
| /quick | Quick Caption processing and result view |
| /lab | Start a Gemma Lab run |
| /lab/:runId/video | Inspect source metadata and choose a preset |
| /lab/:runId/frames | Configure and review representative frames |
| /lab/:runId/audio | Inspect bounded audio selection and RMS energy |
| /lab/:runId/evidence | Review route and structured evidence |
| /lab/:runId/captions | Generate grounded captions |
| /lab/:runId/compare | Compare saved experiments |
| /api/docs | FastAPI OpenAPI documentation |

## Configuration reference

| Variable | Purpose |
| --- | --- |
| GEMMACLIP_PROVIDER | Set to routed_gemma for the public Gemma pipeline. |
| GEMMACLIP_AUDIO_MODE | off, auto, or always; auto is recommended. |
| FIREWORKS_API_KEY | Fireworks credential for configured routed roles. |
| GOOGLE_API_KEY / GEMINI_API_KEY | Google Gemma credential and fallback path. |
| AMD_GEMMA_AUDIO_VISUAL_API_KEY | Optional AMD Cloud credential for audio-visual evidence. |
| AMD_GEMMA_AUDIO_VISUAL_BASE_URL | OpenAI-compatible AMD Cloud base URL. |
| AMD_GEMMA_AUDIO_VISUAL_MODEL | Model ID exposed by the AMD deployment. |
| GEMMACLIP_WEB_RUNS_DIR | Filesystem root for web runs. |
| GEMMACLIP_WEB_MAX_UPLOAD_BYTES | Maximum upload size; default is 200 MiB. |
| GEMMACLIP_WEB_RUN_TTL_SECONDS | Retention for inactive runs; default is 24 hours. |
| GEMMACLIP_LOG_FORMAT | Set to json for safe allow-listed lifecycle logs. |

See .env.example for the complete list of supported settings.

## Security and privacy

- Credentials remain server-side and are never returned by /api/config.
- Uploads are size-limited, filenames are sanitized, and run IDs are generated server-side.
- Media is stored outside the static frontend directory.
- FFmpeg and FFprobe use argument arrays and bounded timeouts; shell execution is not used.
- Raw base64, authorization headers, signed URLs, private endpoints, and provider responses are not exposed.
- Logs contain only safe lifecycle metadata such as run ID, stage, route, provider label, and elapsed time.
- Audio is temporary and bounded; only normalized allowed_caption_facts can influence captions.
- Hidden reasoning is never displayed. Structured evidence is an observable output, not chain-of-thought.

## Project layout

~~~text
GemmaClip/
├── src/gemmaclip/          # Shared pipeline, routing, providers, and web services
├── web/                    # React, TypeScript, and Vite frontend
├── tests/                  # Python backend and API tests
├── examples/               # Small task fixtures
├── scripts/                # Production runners and media utilities
├── Dockerfile              # Python-only leaderboard image
├── Dockerfile.web          # FastAPI + compiled React web image
├── docker-compose.web.yml  # Web production orchestration
└── deploy/nginx/           # Reverse-proxy deployment example
~~~

## Validation

Run the relevant checks before committing:

~~~bash
python -m compileall src tests
pytest
git diff --check
~~~

For frontend changes:

~~~bash
cd web
npm run typecheck
npm test
npm run build
~~~

Generate a small local media fixture when needed:

~~~bash
python scripts/create_demo_videos.py
~~~

The generated tone clip contains a sine wave, not speech, and is not a caption-quality benchmark.

## The idea

> GemmaClip turns video captioning from a black box into a glass box.

Upload a video, see what the system selected, understand why it chose a route, inspect the evidence, and compare the result against another experiment.
