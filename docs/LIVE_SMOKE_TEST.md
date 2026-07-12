# Live-provider smoke-test template

This is a sanitized checklist for a human operator. It contains no fabricated results. Run it only with a controlled, legally usable video and runtime credentials supplied outside the repository. Never attach raw provider responses, API headers, uploaded videos, audio, frames, or signed URLs to the result.

Record one row per scenario:

| Date | Commit | Fixture | Provider | Model | Modality | Fallback used | Generation outcome | Frames s | Audio s | Evidence s | Captions s | Word counts | Pass/fail | Notes |
|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| YYYY-MM-DD | SHA | sanitized filename | fireworks/google | safe label | visual/audio_visual | yes/no | model_generated/evidence_fallback/deterministic_fallback |  |  |  |  | formal=; sarcastic=; humorous-tech=; humorous-non-tech= |  |  |

## Matrix

- Visual-only video: verify six frames, no audio claim, visual route, all requested styles.
- Silent-audio video: verify an audio stream may exist but no useful candidate is used; do not call it speech.
- Usable-audio video: verify the bounded energy candidate, route decision, audio status, and caption-safe facts.
- Fireworks visual failure: verify Google Gemma 4 31B receives frames only and the outcome remains `model_generated` when successful.
- Fireworks audio-visual failure: verify audio is discarded, the modality becomes visual, and no Google request contains audio.
- Partial caption response: verify valid styles survive and focused repair or grounded fallback fills only missing styles.
- Low runtime budget: verify remote attempts are skipped according to the existing thresholds and grounded/deterministic outcome is honest.
- Manual Lab rerun: verify stage locking, dependency invalidation, truthful stale notices, and safe retry.
- Two saved experiments: verify labels, timestamps, provider/model/modality, fallback state, runtime, and captions compare from real snapshots.

## Operator notes

Use `scripts/create_demo_videos.py` for infrastructure checks; its tone clip is explicitly not speech. Capture only sanitized summaries in this file or an ignored local copy. A failed provider call is not permission to edit the result into success, and no CI job should run this matrix automatically.
