# GemmaClip Web App Specification

> Implementation status (July 2026): `web/` and the optional FastAPI backend implement the real Quick Caption vertical slice and an interactive, stage-driven Gemma Lab. Manual Frames, Audio, Evidence, and Captions jobs persist configuration and artifacts; experiments are immutable snapshots compared from stored values. Audio candidates are temporary and cleaned after inspection or evidence generation. A separate `Dockerfile.web` serves the compiled SPA and FastAPI together, with a truthful `/api/health` endpoint and safe lifecycle logging. See `docs/WEB_DEVELOPMENT.md` and `docs/ARCHITECTURE.md`.

> Hardening note: public runs distinguish model generation, grounded evidence fallback, and deterministic fallback. Only the first two may become ready; grounded fallback is visibly degraded. Interrupted processing is recovered as a safe error on restart, expired inactive runs are cleaned according to `GEMMACLIP_WEB_RUN_TTL_SECONDS`, and active runs cannot be deleted. Caption cards describe grounding availability, not unsupported exact per-caption attribution. This remains a single-process demo architecture, not a public-scale job system.

## 1. Product vision

GemmaClip should be both:

1. a simple video-captioning product for non-technical users, and
2. a transparent multimodal experimentation lab for developers, researchers, judges, and curious users.

The core product promise is:

> GemmaClip is simple for everyone and transparent for builders.

The website must avoid becoming another generic upload-and-caption demo. Its differentiator is that users can inspect and control the complete multimodal pipeline: frame selection, audio extraction, Gemma routing, structured evidence, caption generation, and experiment comparison.

The public demo must make Gemma 4 visibly central. The leaderboard container and the web demo are separate delivery surfaces. Do not change the existing leaderboard CLI behavior merely to satisfy web-demo requirements.

## 2. Product modes

### 2.1 Quick Caption

For normal users who want a result with no configuration.

Flow:

```text
Upload video
  -> automatic frame selection
  -> automatic audio selection
  -> automatic Gemma route
  -> grounded captions
  -> copy or download result
```

The user should not need to understand frame extraction, RMS, temperature, providers, or model routing.

### 2.2 Gemma Lab

For technical users who want to inspect and alter each stage.

Flow:

```text
Video
  -> Frames
  -> Audio
  -> Evidence
  -> Captions
  -> Compare
```

Each stage must show:

- configuration controls,
- a clear explanation of what the setting changes,
- the generated artifact,
- runtime metadata,
- Previous and Next navigation,
- an explicit Run or Extract action,
- downstream invalidation when an upstream setting changes.

## 3. Product principles

1. **Progressive disclosure**: Quick Caption stays simple. Gemma Lab exposes complexity gradually.
2. **Glass-box AI**: show inputs, selected artifacts, route decisions, evidence, and final outputs.
3. **Grounding before style**: factual evidence must be visible before styled captions.
4. **Gemma is the star**: model roles and routing decisions should be visible without becoming marketing clutter.
5. **Safe transparency**: never expose credentials, authorization headers, raw base64, private endpoints, or hidden chain-of-thought.
6. **Reproducible experiments**: every run stores its configuration and outputs.
7. **Fast demo path**: the happy path must work with one short video and no manual setup beyond server environment variables.
8. **Preserve the competition CLI**: web code must reuse pipeline components without breaking `/input/tasks.json` to `/output/results.json` behavior.

## 4. Recommended implementation stack

Use the existing Python 3.12 package as the pipeline backend.

### Backend

- FastAPI
- Pydantic models
- Existing `gemmaclip` modules for video, frames, audio, routing, evidence, and captions
- Filesystem run storage for the MVP
- `uvicorn` for local serving
- `ffmpeg` and `ffprobe` from the existing runtime image

### Frontend

- React
- TypeScript
- Vite
- React Router
- TanStack Query or a small typed fetch wrapper
- Plain CSS variables, CSS modules, or Tailwind CSS

Do not introduce a second Python pipeline implementation. The web backend must call reusable service functions that wrap the existing `src/gemmaclip` logic.

Recommended structure:

```text
GemmaClip/
  AGENTS.md
  docs/
    WEB_APP_SPEC.md
  src/gemmaclip/
    ...existing competition code...
    web/
      __init__.py
      app.py
      api.py
      models.py
      services.py
      storage.py
      media.py
  web/
    package.json
    vite.config.ts
    src/
      main.tsx
      app/
      components/
      pages/
      api/
      styles/
  tests/
    ...existing tests...
    test_web_api.py
    test_web_storage.py
```

## 5. Information architecture

Use these routes:

```text
/                       Landing page
/quick                  Quick Caption flow
/lab                    Gemma Lab start or recent runs
/lab/:runId/video       Video inspection
/lab/:runId/frames      Frame extraction
/lab/:runId/audio       Audio extraction
/lab/:runId/evidence    Evidence extraction
/lab/:runId/captions    Caption generation
/lab/:runId/compare     Experiment comparison
```

The landing page should present one upload area and two clear paths rather than two disconnected products.

## 6. Landing page

### Hero copy

```text
GemmaClip

Video captioning powered by pure Gemma.

Drop a video. Get grounded captions.
```

Primary interaction:

```text
[ Drop video or choose file ]
```

After a file is selected, show two actions:

```text
[ Generate captions ]
[ Open in Gemma Lab ]
```

Secondary section:

```text
Are you a nerd?
Inspect every frame, audio segment, model route, evidence object, and generation setting.

[ Open Gemma Lab ]
```

Use “Open Gemma Lab” as the actual call to action. “Are you a nerd?” is supporting personality, not the only navigation label.

### Landing-page requirements

- Drag-and-drop and file-picker support.
- Accept MP4, WebM, and MOV when ffmpeg can decode them.
- Show the configured maximum upload size.
- Show an example-video option when bundled examples are available.
- Do not request API keys in the browser.
- Explain that uploaded media is stored only for the current demo run.

## 7. Quick Caption experience

### 7.1 Processing screen

Show a progress timeline with user-friendly labels:

```text
Preparing video
Selecting important moments
Checking audio
Understanding the scene with Gemma
Writing captions with Gemma
Complete
```

The frontend may poll a run-status endpoint. The backend must store stage status and errors.

Do not fabricate precise progress percentages. Stage-based progress is sufficient.

### 7.2 Result screen

Layout:

- video player on the left or top,
- caption cards on the right or below,
- one card per requested style.

Default styles:

- formal,
- sarcastic,
- humorous_tech,
- humorous_non_tech.

Each caption card must support:

- copy,
- regenerate that style,
- show grounding context availability without claiming exact per-caption use,
- display model and temperature metadata in a collapsible details section.

Page actions:

```text
[ Download JSON ]
[ Start another video ]
[ Open this run in Gemma Lab ]
```

“Open this run in Gemma Lab” must reuse all artifacts already produced. It must not restart processing unnecessarily.

## 8. Gemma Lab shell

Use a persistent stepper:

```text
1 Video -> 2 Frames -> 3 Audio -> 4 Evidence -> 5 Captions -> 6 Compare
```

Desktop layout:

- left column: controls and explanations,
- right column: artifact preview and metadata.

Mobile layout:

- controls first,
- preview second,
- sticky Previous and Next actions.

Every stage needs these states:

- not started,
- running,
- complete,
- stale because an upstream configuration changed,
- failed with a recoverable message.

A user may navigate backward at any time. Navigating forward to a stale stage should require rerunning that stage.

## 9. Step 1: Video inspection

Show:

- filename,
- duration,
- resolution,
- frame rate,
- codec when available,
- audio stream detected or not detected,
- file size,
- video player.

Preset control:

```text
Fast
Balanced
Maximum detail
Custom
```

Recommended preset behavior:

### Fast

- 4 uniform frames,
- audio off,
- visual evidence route,
- one or two model calls depending on runtime.

### Balanced

- existing six-frame hybrid extraction,
- audio auto,
- automatic Gemma route,
- current routed temperatures.

### Maximum detail

- 8 to 12 frames subject to provider limits,
- audio auto,
- automatic Gemma route,
- no claim that this preset is always better.

### Custom

Reveal all controls in later steps.

The Balanced preset should be selected by default.

## 10. Step 2: Frame extraction

### 10.1 Methods

Expose three methods:

1. Uniform
2. AKS-Lite
3. Hybrid: anchors plus AKS-Lite

Map these to existing reusable extraction logic. Do not duplicate ffmpeg extraction code in the web layer.

### 10.2 Controls

Uniform controls:

- total frame count, minimum 2, practical maximum 16,
- optional start and end trim,
- chronological ordering always enabled.

AKS-Lite controls:

- selected frame count,
- candidate scan count or scan interval,
- minimum temporal spacing,
- change-sensitivity control mapped to a documented backend parameter.

Hybrid controls:

- anchor count,
- high-change count,
- total shown as derived value,
- minimum temporal spacing,
- optional sensitivity.

Default hybrid configuration:

```text
4 anchor frames
2 high-change frames
6 total frames
```

### 10.3 Extraction output

Show a horizontal or responsive grid of frame cards.

Each card must include:

- image preview,
- frame number,
- timestamp,
- selection reason: anchor, uniform, or high change,
- change score when available,
- include or exclude toggle for expert runs.

Show a video timeline with selected frame markers. For AKS-Lite or Hybrid, show a simple visual-change chart when scoring data exists.

Actions:

```text
[ Extract frames ]
[ Reset to preset ]
[ Previous ]
[ Next: Audio ]
```

Changing frame settings invalidates:

- frames,
- evidence,
- captions,
- comparisons derived from those captions.

It does not invalidate uploaded video metadata or an independently extracted audio artifact.

## 11. Step 3: Audio extraction

### 11.1 Controls

Audio mode:

```text
Disabled
Automatic
Always analyze
```

Additional controls:

- maximum selected window, 1 to 30 seconds,
- sample rate, with 16 kHz default,
- minimum RMS energy,
- selection method:
  - highest-energy window,
  - first N seconds,
  - custom range when implemented.

For the first MVP, highest-energy and first-N-seconds are enough. Custom waveform-range selection is a later enhancement.

### 11.2 Audio output

Show:

- whether the source contains audio,
- waveform or amplitude envelope,
- selected region,
- start and end timestamps,
- duration,
- RMS energy,
- audio player for the selected WAV,
- routing-candidate status,
- human-readable selection reason.

Use the term `energy candidate`, never `speech candidate`.

Display this explanation near the waveform:

> Audio energy is used only to choose a potentially useful segment. Loud audio does not prove speech. Gemma determines whether speech or relevant sound is actually present.

Changing audio settings invalidates:

- audio artifact,
- evidence when the evidence route may use audio,
- captions,
- comparisons derived from those captions.

It does not invalidate frames.

## 12. Step 4: Evidence extraction

This is the most important Gemma demonstration stage.

### 12.1 Route controls

Expose:

```text
Automatic routing
Visual only: Gemma 4 26B A4B
Audio + visual: Gemma 4 12B Unified
```

Automatic routing should use existing route logic and show the selected result.

Display a route-decision card:

```text
Selected route: Gemma 4 12B Unified
Reason: A non-silent audio window was selected and the runtime budget was sufficient.
```

or:

```text
Selected route: Gemma 4 26B A4B
Reason: No usable audio artifact was available, so visual evidence was used.
```

### 12.2 Expert controls

- provider: automatic, Fireworks, or Google when configured,
- model role chosen from server-approved configured models,
- temperature,
- maximum output tokens when supported,
- strict JSON toggle should remain enabled by default,
- show request structure toggle.

Never expose API keys or private provider endpoints. The frontend should receive only provider names and safe model labels.

### 12.3 Safe request transparency

The UI may show:

- ordered content-part types,
- number of images,
- frame timestamps,
- whether audio is attached,
- system-purpose summary,
- user prompt text after secret-safe sanitization,
- requested schema.

The UI must not show:

- authorization headers,
- API keys,
- signed upload URLs,
- raw base64 image or audio content,
- private endpoints,
- hidden chain-of-thought,
- full raw provider diagnostics containing secrets.

### 12.4 Evidence output

Render readable sections:

- scene,
- main subjects,
- actions,
- setting,
- visible objects,
- mood,
- camera notes,
- temporal progression,
- verified description,
- possible misreads,
- unsupported claim types,
- style hooks.

Render audio evidence separately:

- status,
- speech present,
- language,
- transcript,
- visual consistency,
- caption-safe facts.

Provide a collapsible raw JSON viewer.

Changing evidence configuration invalidates:

- evidence,
- captions,
- comparisons derived from those captions.

It does not invalidate frames or audio.

## 13. Step 5: Caption generation

### 13.1 Controls

Model section:

- Gemma 4 31B caption role,
- safe provider selector,
- temperature,
- maximum output tokens when supported.

Style selection:

- formal,
- sarcastic,
- humorous_tech,
- humorous_non_tech,
- optional custom style name and instruction as a later enhancement.

Caption constraints:

- minimum words,
- maximum words,
- strict grounding enabled by default,
- audio evidence enabled only through normalized `allowed_caption_facts`,
- focused repair enabled by default.

Default generation temperature: use the existing routed default rather than inventing a new value.

### 13.2 Output

Each result card should show:

- style label,
- caption,
- copy button,
- regenerate-style button,
- model/provider metadata,
- temperature,
- elapsed time,
- whether focused repair was used,
- grounding summary.

Do not claim exact token-level or per-caption attribution unless the backend records it. The current availability summary is:

```text
Grounding available: visual evidence, caption-safe audio evidence
```

The output must always contain exactly the requested styles. Missing styles must use the existing evidence-based fallback behavior.

Changing caption settings invalidates only caption output and comparisons.

## 14. Step 6: Experiment comparison

The MVP must support comparing two saved snapshots from the same uploaded video.

A snapshot stores:

- frame configuration,
- selected frame metadata,
- audio configuration,
- selected audio metadata,
- evidence route and settings,
- normalized evidence,
- caption settings,
- captions,
- elapsed stage times,
- provider and safe model labels.

Comparison table:

| Setting | Experiment A | Experiment B |
| --- | --- | --- |
| Frame method | Hybrid | Uniform |
| Frame count | 6 | 4 |
| Audio mode | Auto | Off |
| Evidence route | Gemma 4 12B | Gemma 4 26B |
| Evidence temperature | 0.0 | 0.0 |
| Caption temperature | 0.4 | 0.8 |
| Runtime | value | value |

Below the table, show caption cards side by side by style.

Preset comparison actions:

```text
Compare frame methods
Compare frame counts
Compare temperature
Compare visual-only vs audio-visual
```

For MVP, these actions may simply duplicate the current configuration and change the named setting. Do not automatically run expensive experiments without a confirmation action.

## 15. Run state and dependency model

Each run has these artifacts:

```text
video
metadata
frames
frame_scores
audio
route_decision
evidence
captions
snapshots
```

Each artifact has:

```text
status: not_started | running | complete | stale | failed
config_hash
created_at
elapsed_seconds
error_message
```

Dependency graph:

```text
video -> metadata
video + frame_config -> frames
video + audio_config -> audio
frames + audio + evidence_config -> evidence
evidence + frames + caption_config -> captions
all current artifacts -> snapshot
```

When configuration changes, invalidate only dependent downstream artifacts.

Examples:

- frame-count change invalidates frames, evidence, captions, and snapshots based on them;
- audio-threshold change invalidates audio, evidence when audio is involved, captions, and related snapshots;
- caption-temperature change invalidates captions only;
- changing the caption style set invalidates captions only.

## 16. Backend API

All responses should use JSON except media streaming endpoints.

### 16.1 Health and configuration

```text
GET /api/health
GET /api/config
```

`GET /api/config` returns safe information only:

```json
{
  "max_upload_bytes": 209715200,
  "supported_extensions": ["mp4", "webm", "mov"],
  "providers": ["automatic", "fireworks", "google"],
  "models": {
    "visual_evidence": ["Gemma 4 26B A4B"],
    "audio_visual_evidence": ["Gemma 4 12B Unified"],
    "captions": ["Gemma 4 31B"]
  },
  "default_styles": ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]
}
```

Only include providers and roles that are configured on the server.

### 16.2 Runs

```text
POST /api/runs
GET /api/runs/{run_id}
DELETE /api/runs/{run_id}
```

`POST /api/runs` accepts multipart upload and creates the run directory.

Response:

```json
{
  "run_id": "run_...",
  "status": "created",
  "next": "/lab/run_.../video"
}
```

### 16.3 Quick mode

```text
POST /api/runs/{run_id}/quick
```

Runs the Balanced preset through the complete pipeline. It may respond synchronously for the first implementation or return a job status that the frontend polls.

### 16.4 Metadata

```text
POST /api/runs/{run_id}/metadata
```

### 16.5 Frames

```text
POST /api/runs/{run_id}/frames
GET /api/runs/{run_id}/frames
GET /api/runs/{run_id}/frames/{frame_id}
```

Request example:

```json
{
  "method": "hybrid",
  "total_frames": 6,
  "anchor_count": 4,
  "change_count": 2,
  "minimum_spacing_seconds": 1.0,
  "change_sensitivity": 0.5
}
```

### 16.6 Audio

```text
POST /api/runs/{run_id}/audio
GET /api/runs/{run_id}/audio
GET /api/runs/{run_id}/audio/file
GET /api/runs/{run_id}/audio/waveform
```

Request example:

```json
{
  "mode": "auto",
  "max_seconds": 30,
  "sample_rate": 16000,
  "minimum_rms": 0.01,
  "selection_method": "highest_energy"
}
```

Waveform endpoint may return a downsampled numeric envelope rather than a generated image.

### 16.7 Evidence

```text
POST /api/runs/{run_id}/evidence
GET /api/runs/{run_id}/evidence
```

Request example:

```json
{
  "route": "automatic",
  "provider": "automatic",
  "temperature": 0.0,
  "max_tokens": 2048,
  "show_safe_request": true
}
```

### 16.8 Captions

```text
POST /api/runs/{run_id}/captions
GET /api/runs/{run_id}/captions
POST /api/runs/{run_id}/captions/{style}/regenerate
```

Request example:

```json
{
  "provider": "automatic",
  "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
  "temperature": 0.4,
  "minimum_words": 18,
  "maximum_words": 35,
  "focused_repair": true
}
```

### 16.9 Snapshots and comparison

```text
POST /api/runs/{run_id}/snapshots
GET /api/runs/{run_id}/snapshots
GET /api/runs/{run_id}/compare?left={snapshot_id}&right={snapshot_id}
```

### 16.10 Status polling

```text
GET /api/runs/{run_id}/status
```

Response example:

```json
{
  "active_stage": "evidence",
  "stages": {
    "metadata": {"status": "complete", "elapsed_seconds": 0.4},
    "frames": {"status": "complete", "elapsed_seconds": 2.1},
    "audio": {"status": "complete", "elapsed_seconds": 1.8},
    "evidence": {"status": "running"},
    "captions": {"status": "not_started"}
  }
}
```

## 17. Run storage

Use local filesystem storage for the MVP:

```text
.gemmaclip/runs/{run_id}/
  run.json
  source/
    video.ext
  frames/
    frame_*.jpg
    scores.json
  audio/
    selected.wav
    waveform.json
  evidence/
    evidence.json
    safe_request.json
  captions/
    captions.json
  snapshots/
    {snapshot_id}.json
```

Rules:

- Generate run IDs server-side.
- Sanitize original filenames.
- Never use user-controlled paths directly.
- Write JSON atomically using a temporary file and rename.
- Prevent path traversal.
- Deleting a run removes its full directory.
- Add an age-based cleanup utility for demo deployments.
- Do not store credentials in run files.

## 18. Service-layer requirements

Create service functions independent of FastAPI route objects:

```text
create_run
probe_run_video
extract_run_frames
extract_run_audio
generate_run_evidence
generate_run_captions
create_run_snapshot
compare_snapshots
invalidate_downstream
```

The API layer should validate requests and call these services. Tests should be able to call service functions without starting a server.

Reuse existing functions from:

- `video.py`,
- `frames.py`,
- `audio.py`,
- `routed.py`,
- `captioner.py`,
- `gemma_client.py`.

Where existing functions are too CLI-specific, extract a shared pure function rather than copying code.

## 19. Quick mode preset

The Quick Caption Balanced preset must use:

- existing six-frame hybrid extraction,
- four anchors and two high-change frames,
- chronological ordering,
- audio mode `auto`,
- maximum selected audio window of 30 seconds,
- automatic evidence routing,
- visual route using Gemma 4 26B A4B,
- audio-visual route using Gemma 4 12B Unified,
- final captions using Gemma 4 31B,
- existing safe evidence normalization,
- existing evidence-based fallback behavior,
- existing routed stage temperatures.

Do not introduce a hidden non-Gemma model into the web demo’s Gemma path.

## 20. Frontend component plan

Recommended shared components:

```text
AppShell
UploadDropzone
VideoPlayer
PipelineStepper
StageStatusBadge
ConfigPanel
ArtifactPanel
FrameGrid
FrameCard
TimelineMarkers
ChangeScoreChart
AudioWaveform
AudioPlayer
RouteDecisionCard
EvidenceViewer
RawJsonViewer
CaptionCard
ExperimentTable
ErrorNotice
EmptyState
LoadingStage
```

Use semantic HTML and keyboard-accessible controls.

## 21. Visual direction

The interface should feel like a polished AI laboratory, not an admin dashboard.

Recommended characteristics:

- dark neutral background with high-contrast cards,
- one primary accent color and one secondary data accent,
- large, confident type on the landing page,
- compact monospace labels for timestamps and model metadata,
- rounded panels but not excessive glass effects,
- visible pipeline connectors and stage status,
- waveform, frame timeline, and evidence cards as the visual focus.

Avoid:

- excessive gradients,
- fake terminal output,
- tiny dense controls on the landing page,
- unexplained model jargon in Quick Caption mode,
- animations that slow navigation.

## 22. Accessibility

- All controls require labels.
- Keyboard navigation must work through the stepper and form controls.
- Frame previews need alt text with timestamp and selection reason.
- Do not rely on color alone for stage status.
- Use visible focus states.
- Caption copy buttons need accessible names.
- Audio controls must use native controls or accessible equivalents.
- Respect reduced-motion preferences.

## 23. Error handling

User-facing errors must be actionable.

Examples:

```text
This video could not be decoded. Try MP4 with H.264 video.
No audio stream was found. GemmaClip will continue with visual evidence.
The selected provider is not configured on this server.
Evidence generation failed. You can retry or continue with a visual fallback.
Caption generation timed out. Grounded fallback captions were preserved.
```

Never show raw stack traces in the browser.

A failed stage should not delete previously valid upstream artifacts.

## 24. Security and privacy

- Enforce upload-size limits before reading the complete file into memory.
- Store uploads outside the static frontend directory.
- Use generated filenames.
- Validate MIME type and extension, but trust ffprobe for actual decoding capability.
- Run subprocesses without `shell=True`.
- Set ffmpeg and ffprobe timeouts.
- Do not accept arbitrary provider base URLs from the browser.
- Do not expose environment variables.
- Do not log raw media, base64 content, prompts containing secrets, or provider responses containing credentials.
- Apply basic same-origin CORS defaults.
- Add a run cleanup mechanism.

## 25. Testing requirements

### Backend unit tests

- create and delete run,
- upload validation,
- path traversal rejection,
- metadata probing adapter,
- uniform frame config mapping,
- AKS-Lite config mapping,
- hybrid config mapping,
- frame artifact serialization,
- audio config mapping,
- waveform downsampling,
- route-decision serialization,
- evidence normalization,
- exact requested caption styles,
- downstream invalidation,
- snapshot creation and comparison,
- no credentials in API responses,
- existing CLI tests remain green.

### API tests

Use FastAPI test client or `httpx` ASGI transport.

Test:

- health,
- safe config,
- multipart upload,
- run retrieval,
- each stage endpoint,
- media file endpoints,
- invalid run ID,
- stale-stage behavior,
- failure responses,
- run deletion.

Mock model calls. Unit tests must not require live provider credentials.

### Frontend tests

At minimum:

- landing upload interaction,
- Quick versus Lab navigation,
- stepper navigation,
- frame-method form behavior,
- audio mode controls,
- evidence viewer rendering,
- caption copy action,
- stale-stage warning,
- comparison rendering.

### End-to-end smoke test

Use a tiny local fixture video and mocked Gemma responses:

```text
upload -> frames -> audio -> evidence -> captions -> snapshot -> compare
```

## 26. Environment variables

Keep existing model variables. Add web-specific variables only when needed:

```text
GEMMACLIP_WEB_HOST=0.0.0.0
GEMMACLIP_WEB_PORT=8000
GEMMACLIP_WEB_RUN_DIR=.gemmaclip/runs
GEMMACLIP_WEB_MAX_UPLOAD_BYTES=209715200
GEMMACLIP_WEB_RUN_TTL_SECONDS=86400
GEMMACLIP_WEB_EXAMPLE_DIR=examples/videos
```

Do not add browser-exposed API keys.

## 27. Development commands

Codex should add clear scripts, for example:

```bash
# backend
python -m gemmaclip.web.app

# frontend
cd web
npm install
npm run dev

# backend tests
pytest

# frontend tests
cd web
npm test

# production frontend build
cd web
npm run build
```

A root development helper is acceptable, but do not make the existing competition entrypoint depend on Node.

## 28. Production serving

For MVP production deployment:

1. Build the React frontend to `web/dist`.
2. Serve the static build through FastAPI or a small reverse proxy.
3. Keep API routes under `/api`.
4. Keep uploaded media behind authenticated-by-run-id API routes rather than exposing the run directory directly.

Do not alter the competition Docker entrypoint until the web image is intentionally separated or selected by a dedicated target.

Recommended Docker strategy:

- preserve the existing leaderboard image behavior,
- add a separate `web` build target or `Dockerfile.web`,
- avoid inflating the leaderboard image with unnecessary frontend build tooling.

The release uses `Dockerfile.web`, a multi-stage Node 22 plus Python 3.12 image. FastAPI serves the compiled `index.html` for `/`, `/quick`, `/lab`, and direct run-stage routes; `/assets/*` serves only built frontend assets; `/api/*` never falls through to the SPA. The image installs ffmpeg/ffprobe, runs as an unprivileged user, and stores runs in the mounted `GEMMACLIP_WEB_RUNS_DIR` volume. Runtime secrets are passed through the environment, never Docker build arguments. `docker-compose.web.yml` and the platform-specific production scripts provide the launch path.

`GET /api/health` checks storage writability, ffmpeg, ffprobe, provider credential presence, and job-manager availability without making a provider request. It returns `ok`, `degraded`, or `unavailable` with safe status fields only. Set `GEMMACLIP_LOG_FORMAT=json` to emit allow-listed lifecycle events; captions, evidence, prompts, provider responses, credentials, and private URLs are excluded.

## 29. Implementation phases

### Phase 1: foundations

- FastAPI app,
- safe config endpoint,
- run storage,
- upload endpoint,
- metadata endpoint,
- React shell and routing,
- landing page,
- video-inspection page.

### Phase 2: visible preprocessing

- frame endpoint and frame gallery,
- timeline markers,
- audio endpoint,
- selected audio player,
- waveform data,
- stale-stage invalidation.

### Phase 3: Gemma pipeline

- evidence endpoint using existing routed logic,
- route-decision card,
- structured evidence viewer,
- caption endpoint,
- result cards,
- Quick Caption orchestration.

### Phase 4: experimentation

- snapshots,
- compare page,
- duplicate configuration,
- temperature comparison,
- visual-only versus audio-visual comparison.

### Phase 5: hardening

- accessibility pass,
- cleanup job,
- upload limits,
- Docker web target,
- end-to-end tests,
- demo fixtures,
- documentation.

Codex should complete one phase with tests before moving to the next. Avoid a giant untested implementation commit.

## 30. MVP acceptance criteria

The MVP is complete when all of the following work:

1. A user can upload a video from the landing page.
2. Quick Caption produces all four default caption styles.
3. The result can be opened in Gemma Lab without re-uploading.
4. Gemma Lab shows video metadata.
5. The user can choose Uniform, AKS-Lite, or Hybrid frame extraction.
6. Extracted frames are shown with timestamps and selection reasons.
7. The user can run automatic audio extraction and play the selected window.
8. The UI clearly states that RMS indicates energy, not confirmed speech.
9. The user can choose automatic, visual-only, or audio-visual evidence routing.
10. The selected Gemma route and reason are visible.
11. Structured evidence and safe raw JSON are visible.
12. The user can change caption temperature and requested styles.
13. Captions are generated with exactly the requested keys.
14. Two experiment snapshots can be compared side by side.
15. Upstream changes correctly mark downstream stages stale.
16. No credentials, raw base64, or hidden reasoning appear in the browser or logs.
17. Existing leaderboard CLI tests still pass.
18. Frontend and backend have documented local-development commands.

## 31. Demo script

The final hackathon demo should follow this sequence:

1. Drop a short video on the landing page.
2. Use Quick Caption and show the immediate captions.
3. Click “Open this run in Gemma Lab.”
4. Show the six selected hybrid frames and why each was selected.
5. Play the extracted audio window and point out that energy does not equal speech.
6. Show the automatic route decision between Gemma 4 26B and Gemma 4 12B.
7. Show the structured evidence object.
8. Change caption temperature and regenerate.
9. Compare the original and changed experiment side by side.
10. End with the message: “GemmaClip turns video captioning from a black box into a glass box.”

## 32. Non-goals for the first build

Do not block the MVP on:

- user accounts,
- cloud database,
- payments,
- collaborative workspaces,
- arbitrary custom provider endpoints,
- editable system prompts,
- token-level attribution,
- full-video transcription,
- multi-window audio analysis,
- frame-by-frame video annotation,
- mobile-native applications,
- production-scale distributed workers.

## 33. Definition of done for Codex

Before reporting completion, Codex must:

1. run existing Python tests,
2. run new backend tests,
3. run frontend tests,
4. build the frontend,
5. run `git diff --check`,
6. verify no credentials or media blobs are committed,
7. document commands and environment variables,
8. report files changed and any intentionally deferred items,
9. avoid claiming live provider compatibility unless a real smoke test was executed,
10. leave the repository in a clean state after committing.
