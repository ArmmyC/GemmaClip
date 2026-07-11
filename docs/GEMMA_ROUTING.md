# Gemma routing

Set `GEMMACLIP_PROVIDER=routed_gemma` to enable the two-stage routed pipeline.

## Architecture

Every routed task reuses the six-frame hybrid extractor: four fixed temporal anchors plus two high-change frames, sorted chronologically. If hybrid scanning or extraction fails, the existing six-frame uniform fallback remains active.

The normal visual route is Gemma 4 26B A4B evidence followed by Gemma 4 31B final caption synthesis. With selected non-silent audio and sufficient runtime, Gemma 4 12B Unified replaces only the evidence stage. When runtime is safe, the final 31B stage receives all six frames, normalized evidence JSON, requested styles, and the exact dynamic output schema. Successful normal tasks therefore use two model calls. There is no normal verifier call; a focused repair call is allowed only for missing or invalid requested styles.

The live deadline is checked before every Fireworks and Google attempt. If a primary attempt consumes the safe budget, its same-role fallback is not started. Audio preprocessing is skipped before `ffprobe` when fewer than the configured audio-route seconds remain, and the route is checked again after preprocessing. If evidence succeeds but fewer than 50 seconds remain, final synthesis is replaced by evidence-based captions. Focused repair requires 35 seconds; when unsafe, valid model captions are preserved and only missing styles are filled from evidence-based fallbacks.

## Same-role provider fallback

Each model role tries Fireworks first and then Google when configured:

| Role | Fireworks | Google fallback |
| --- | --- | --- |
| Visual evidence | `FIREWORKS_GEMMA_VISUAL_MODEL` | `GOOGLE_GEMMA_VISUAL_MODEL` |
| Audio-visual evidence | `FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL` | `GOOGLE_GEMMA_AUDIO_VISUAL_MODEL` |
| Final captions | `FIREWORKS_GEMMA_CAPTION_MODEL` | `GOOGLE_GEMMA_CAPTION_MODEL` |

Configured values are opaque callable model or deployment IDs. A Fireworks library path is not assumed to be callable. If only one provider has credentials, it is used directly. If evidence fails everywhere, the task receives deterministic validated fallback captions. If caption synthesis fails after evidence succeeds, the evidence-based fallback preserves the grounded subject, action, and setting.

## Runtime degradation

- At least 170 seconds remaining by default: audio may route to 12B Unified, then 31B.
- 130–170 seconds: 26B visual evidence, then 31B.
- 65–130 seconds: one direct 26B visual caption call.
- Below 65 seconds: deterministic fallback with no remote call.

These per-task thresholds extend rather than replace the 570-second batch guard and preserve the final output buffer. The container watchdog remains 590 seconds.

Per-attempt safeguards default to 95 seconds for evidence, 50 for final synthesis, 35 for focused repair, and 45 for the single-call route. Each provider fallback receives a fresh deadline check.

## Evidence schema

Evidence contains scene, subjects, actions, setting, objects, mood, camera notes, temporal progression, caption focus, a verified description, possible misreads, unsupported claim types, style hooks, and a nested audio object. Audio status is one of `usable`, `uncertain`, `silent`, `unavailable`, or `failed`. Sound, dialogue, speech, music, and noise remain globally unsupported unless audio is usable and not visually contradictory. Even then, only facts in `audio.allowed_caption_facts` may reach captions, and only when the fact agrees with visible frames; the full transcript is never safe by default.

## Configuration

```text
GEMMACLIP_PROVIDER=routed_gemma
GEMMACLIP_AUDIO_MODE=off|auto|always
FIREWORKS_API_KEY
GOOGLE_API_KEY or GEMINI_API_KEY
FIREWORKS_GEMMA_VISUAL_MODEL
FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL
FIREWORKS_GEMMA_CAPTION_MODEL
GOOGLE_GEMMA_VISUAL_MODEL
GOOGLE_GEMMA_AUDIO_VISUAL_MODEL
GOOGLE_GEMMA_CAPTION_MODEL
GEMMACLIP_ROUTED_EVIDENCE_TEMPERATURE=0.0
GEMMACLIP_ROUTED_CAPTION_TEMPERATURE=0.4
GEMMACLIP_ROUTED_REPAIR_TEMPERATURE=0.25
GEMMACLIP_ROUTED_SINGLE_CALL_TEMPERATURE=0.4
```

Temperatures accept finite values in the provider-supported 0–2 range; configured values are clamped and malformed values use the defaults. These defaults are safeguards, not measured optima.

Do not place real credentials in commands committed to the repository.
