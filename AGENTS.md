# AGENTS.md

## Project overview

GemmaClip is an AMD Developer Hackathon video-captioning project with two distinct delivery surfaces:

1. **Leaderboard container**: reads `/input/tasks.json`, processes hidden videos, and writes `/output/results.json`.
2. **Public web demo**: offers a simple Quick Caption experience and an expert Gemma Lab that exposes the complete multimodal pipeline.

The leaderboard submission may use any permitted model configuration. The public web demo must make Gemma 4 visibly central and must not silently substitute a non-Gemma model in its Gemma pipeline.

Before changing code, read:

1. `docs/WEB_APP_SPEC.md`
2. `docs/GEMMA_ROUTING.md`
3. `docs/GEMMA_AUDIO.md`
4. `README.md`

Treat `docs/WEB_APP_SPEC.md` as the source of truth for the website product, architecture, API, UI, and acceptance criteria.

## Current competition pipeline

The existing routed Gemma architecture is:

```text
Visual route:
Gemma 4 26B A4B evidence
  -> Gemma 4 31B final captions

Audio-visual route:
Gemma 4 12B Unified evidence
  -> Gemma 4 31B final captions
```

The normal successful route uses two remote calls. A third call is allowed only for focused repair of missing or invalid caption styles when runtime permits.

Current preprocessing behavior includes:

- six chronological hybrid frames,
- four anchor frames,
- two high-change frames,
- uniform fallback,
- optional bounded audio extraction,
- maximum selected audio window of 30 seconds,
- RMS-based energy selection,
- explicit audio safety normalization,
- evidence-based caption fallback,
- live runtime checks before provider attempts.

Do not weaken these safeguards while adding the website.

## Non-negotiable leaderboard behavior

Preserve the existing container contract:

- read `/input/tasks.json` on startup,
- write valid `/output/results.json`,
- include every requested caption style,
- return safe fallbacks instead of crashing the full batch,
- preserve the 570-second application budget,
- preserve the 590-second container timeout,
- preserve progressive result writes,
- preserve `linux/amd64` compatibility,
- do not require Node or frontend assets for the leaderboard entrypoint,
- do not change provider behavior unless the task explicitly requires it.

Web work must not make the competition CLI slower, less reliable, or dependent on a web server.

## Product goal

Build two connected website modes.

### Quick Caption

For ordinary users:

```text
Upload video
  -> automatic processing
  -> grounded captions
```

No technical configuration should be required.

### Gemma Lab

For technical users:

```text
Video
  -> Frames
  -> Audio
  -> Evidence
  -> Captions
  -> Compare
```

Users must be able to inspect and configure each stage without exposing credentials, private endpoints, raw base64, or hidden reasoning.

The key product message is:

> GemmaClip turns video captioning from a black box into a glass box.

## Implementation strategy

Use the existing Python package as the single pipeline implementation.

Recommended stack:

### Backend

- FastAPI
- Pydantic
- existing `gemmaclip` modules
- filesystem run storage for MVP
- ffmpeg and ffprobe

### Frontend

- React
- TypeScript
- Vite
- React Router
- typed API client

Recommended locations:

```text
src/gemmaclip/web/
web/
tests/test_web_*.py
```

Do not reimplement frame extraction, audio extraction, route selection, evidence normalization, or caption cleanup in TypeScript.

## Build order

Implement in small, tested phases.

### Phase 1: foundation

- FastAPI application
- health endpoint
- safe configuration endpoint
- run storage
- upload endpoint
- metadata endpoint
- React shell
- landing page
- video inspection page

### Phase 2: preprocessing visibility

- Uniform frame extraction UI
- AKS-Lite frame extraction UI
- Hybrid frame extraction UI
- frame gallery and timeline
- audio extraction UI
- waveform data
- selected audio playback
- dependency invalidation

### Phase 3: Gemma pipeline

- evidence endpoint
- automatic route decision
- visual-only route
- audio-visual route
- structured evidence viewer
- caption endpoint
- Quick Caption orchestration

### Phase 4: experimentation

- snapshots
- side-by-side comparison
- temperature comparison
- frame-method comparison
- visual-only versus audio-visual comparison

### Phase 5: hardening

- accessibility
- upload limits
- cleanup utility
- web Docker target
- end-to-end smoke test
- documentation

Complete and test each phase before starting a broad refactor of the next phase.

## Shared service rules

Create framework-independent service functions for web operations.

Expected responsibilities:

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

FastAPI route handlers should validate input, call services, and serialize output. They should not contain pipeline logic.

When existing functions are too CLI-specific, extract reusable pure functions. Do not copy large blocks of pipeline code into `src/gemmaclip/web`.

## Run artifact model

A run contains:

```text
video
metadata
frames
frame scores
audio
route decision
evidence
captions
snapshots
```

Each stage tracks:

```text
status
configuration hash
created time
elapsed seconds
safe error message
```

Supported status values:

```text
not_started
running
complete
stale
failed
```

Upstream configuration changes must invalidate only dependent downstream stages.

Examples:

- frame settings invalidate frames, evidence, captions, and dependent snapshots;
- audio settings invalidate audio, audio-dependent evidence, captions, and dependent snapshots;
- caption temperature invalidates captions only.

## Frame extraction requirements

Expose:

- Uniform
- AKS-Lite
- Hybrid: anchors plus AKS-Lite

Default Balanced configuration:

```text
4 anchors
2 high-change frames
6 chronological frames
```

Frame cards should include:

- preview,
- timestamp,
- selection reason,
- change score when available.

Always preserve chronological order before model calls.

## Audio requirements

Expose:

- off,
- auto,
- always.

Default behavior:

- mono 16 kHz WAV,
- maximum 30-second selected window,
- highest-energy selection,
- configurable minimum RMS.

Use `energy candidate`, not `speech candidate`.

The UI must explain that RMS measures energy and does not prove speech.

Only normalized facts in `audio.allowed_caption_facts` may be used in captions when audio status is usable and visual consistency is not contradictory.

## Evidence requirements

Expose these routes:

- automatic,
- visual only with Gemma 4 26B A4B,
- audio plus visual with Gemma 4 12B Unified.

Show:

- selected route,
- route reason,
- safe model label,
- provider label,
- temperature,
- structured evidence,
- safe raw JSON,
- safe request structure when requested.

Never show hidden reasoning. The evidence schema is an observable output, not chain-of-thought.

## Caption requirements

Use the Gemma 4 31B caption role in the public Gemma demo.

Default styles:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

The response must contain exactly the requested style keys.

Preserve:

- grounding rules,
- audio claim gates,
- valid partial captions,
- focused repair only for missing or invalid styles,
- evidence-based fallback behavior.

Do not discard a valid caption because another requested style failed.

## Quick Caption preset

Quick Caption should use the Balanced configuration:

- six-frame Hybrid extraction,
- four anchors and two high-change frames,
- audio auto,
- automatic evidence routing,
- existing routed temperatures,
- existing caption styles,
- existing fallback behavior.

A Quick Caption result must be openable in Gemma Lab without re-uploading or repeating completed stages unnecessarily.

## Experiment comparison

The MVP must compare two snapshots for the same source video.

A snapshot includes configuration, artifacts, safe model metadata, captions, and elapsed stage times.

Comparison should support:

- frame method,
- frame count,
- audio mode,
- evidence route,
- evidence temperature,
- caption temperature,
- runtime,
- captions by style.

Do not automatically launch expensive comparison runs without a clear user action.

## Security and privacy

Never commit or expose:

- API keys,
- `.env` files,
- authorization headers,
- private endpoints,
- signed provider URLs,
- raw base64 media,
- uploaded user videos,
- extracted audio,
- generated frame artifacts,
- raw provider responses containing sensitive data.

Web requirements:

- enforce upload-size limits,
- generate server-side run IDs,
- sanitize filenames,
- reject path traversal,
- store media outside the static frontend directory,
- use subprocess argument arrays,
- never use `shell=True`,
- apply ffmpeg and ffprobe timeouts,
- return safe errors instead of stack traces,
- delete run directories through a controlled storage service.

Do not accept arbitrary model endpoints from the browser.

## Logging

Logs may contain:

- run ID,
- stage,
- route,
- provider,
- safe model label,
- status,
- elapsed time,
- remaining runtime,
- artifact counts,
- safe failure class.

Logs must not contain credentials, raw media, base64 data, signed URLs, or complete secret-bearing payloads.

## UI rules

Quick Caption should use plain language.

Gemma Lab may use technical terms, but every important term needs an explanation or tooltip.

The interface should:

- be keyboard accessible,
- have visible focus states,
- use labels for every control,
- avoid relying on color alone,
- provide frame alt text,
- use accessible audio controls,
- respect reduced motion.

Do not build a fake terminal interface. The frames, waveform, route decision, evidence, and captions are the primary visuals.

## Testing requirements

All existing tests must continue to pass.

Add backend tests for:

- run storage,
- upload validation,
- path safety,
- metadata,
- frame config mapping,
- audio config mapping,
- evidence route serialization,
- exact caption styles,
- invalidation,
- snapshots,
- comparison,
- credential-safe API output.

Add API tests for every endpoint.

Mock model calls in unit tests. Do not require live credentials.

Add frontend tests for:

- upload,
- Quick versus Lab navigation,
- stepper behavior,
- frame controls,
- audio controls,
- evidence rendering,
- caption rendering,
- stale-stage messages,
- comparison.

Add an end-to-end smoke path with a small fixture and mocked Gemma output.

## Validation commands

Before committing implementation work, run the relevant available commands:

```bash
python -m compileall src tests
pytest
git diff --check
```

For the frontend:

```bash
cd web
npm test
npm run build
```

Also run configured lint and type checks when added.

Do not claim a command passed unless it was actually executed successfully.

## Git workflow

- Work on a feature branch, not `main`.
- Keep commits focused by phase.
- Do not combine unrelated leaderboard-model changes with web UI work.
- Inspect the diff before staging.
- Do not commit generated frontend build output unless the deployment design explicitly requires it.
- Do not commit uploaded media or run directories.
- Keep the worktree clean after the final commit.

## Definition of done

Website work is complete only when:

1. upload works,
2. Quick Caption returns all default styles,
3. completed Quick runs open in Gemma Lab,
4. frame methods are configurable and previewed,
5. selected audio can be inspected and played,
6. route decisions are visible,
7. structured Gemma evidence is visible,
8. caption settings are configurable,
9. two experiments can be compared,
10. downstream invalidation works,
11. no secrets or hidden reasoning are exposed,
12. existing leaderboard tests remain green,
13. frontend tests pass,
14. frontend production build succeeds,
15. documentation explains local development and deployment.

When implementation must be reduced for time, prioritize the complete vertical path:

```text
upload -> frames -> audio -> evidence -> captions -> compare
```

A smaller working glass-box pipeline is better than many unfinished controls.
