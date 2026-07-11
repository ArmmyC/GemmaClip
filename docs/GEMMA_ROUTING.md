# Gemma routing

Set `GEMMACLIP_PROVIDER=routed_gemma` to enable the two-stage routed pipeline.

## Architecture

Every routed task reuses the six-frame hybrid extractor: four fixed temporal anchors plus two high-change frames, sorted chronologically. If hybrid scanning or extraction fails, the existing six-frame uniform fallback remains active.

The normal visual route is Gemma 4 26B A4B evidence followed by Gemma 4 31B final caption synthesis. With selected non-silent audio and sufficient runtime, Gemma 4 12B Unified replaces only the evidence stage. The final 31B stage always receives all six frames, normalized evidence JSON, requested styles, and the exact dynamic output schema. Successful tasks therefore use two model calls. There is no normal verifier call; a focused repair call is allowed only for missing or invalid requested styles.

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

## Evidence schema

Evidence contains scene, subjects, actions, setting, objects, mood, camera notes, temporal progression, caption focus, a verified description, possible misreads, unsupported claim types, style hooks, and a nested audio object. Audio status is one of `usable`, `uncertain`, `silent`, `unavailable`, or `failed`. Only facts in `audio.allowed_caption_facts` may reach captions, and only when status is `usable` and the fact agrees with visible frames.

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
```

Do not place real credentials in commands committed to the repository.
