# GemmaClip three-to-five-minute demo

This script assumes a short local video, a running web image, and provider credentials already configured at runtime. Do not change credentials or endpoints during the presentation.

## 1. Frame the problem — 20 seconds

Narration: “Most video captioning demos return a sentence and hide the decisions behind it. GemmaClip turns that black box into a glass box: the same run exposes the selected frames, bounded audio decision, Gemma route, structured evidence, and final styles.”

Expected screen: Landing page with `Service ready` or `Limited configuration` shown by the real health endpoint.

## 2. Quick Caption — 60 seconds

1. Drop the controlled clip into Quick Caption.
2. Point out the stage-based status: video, frames, audio, evidence, captions. Do not describe a percentage; the backend does not provide one.
3. Read one or two finished caption cards and point out the generation outcome notice.

Narration: “Quick Caption uses the Balanced preset. It selects six chronological Hybrid frames, checks a bounded energy candidate, routes evidence automatically, and asks Gemma 4 31B to write the requested styles.”

Fallback narration: “This run used grounded evidence fallback. The evidence stage succeeded, but writing was not safe within the runtime, so the displayed captions are clearly marked degraded. A provider fallback that succeeds is still model-generated.”

## 3. Open the glass box — 90 seconds

1. Select `Inspect in Gemma Lab`.
2. On Frames, show the timeline and six frame cards. Explain anchor versus high-change selection.
3. On Audio, show the candidate status and RMS. Say: “RMS measures energy; it does not prove speech.”
4. On Evidence, show the route panel, provider, safe model label, modality, fallback badge if present, verified observations, and do-not-claim section.
5. Open raw JSON only as an observable structured artifact. Do not look for hidden reasoning; it is intentionally not stored.

Fallback narration: “This video has no usable audio candidate, so the route is visual-only. That is a truthful decision, not a missing feature.”

## 4. One experiment — 45 seconds

1. Open Captions and change one supported setting, such as caption temperature.
2. Show the invalidation preview before regenerating.
3. Generate Captions and save `Experiment A` or `Experiment B` from Compare.

Narration: “Changing caption settings invalidates only Captions and Compare. Frames and Audio remain current. The stage lock prevents conflicting mutations.”

## 5. Compare — 35 seconds

1. Save two snapshots with distinct labels.
2. Choose them as Experiment A and B.
3. Show the real configuration diff, runtime, provider/model/modality, fallback state, outcome, and captions.
4. Use `Copy comparison summary` if a judge wants the data outside the browser.

Narration: “These are immutable snapshots of the same source video, not a mock chart. Differences are computed from stored experiment data.”

## Recovery and actions to avoid

- Do not intentionally break credentials; use the fallback narration if a provider call fails naturally.
- Do not paste API keys, provider URLs, prompts, raw JSON from outside the app, or private media into the browser.
- Do not start multiple expensive comparison runs during the demo.
- If a stage fails, read the sanitized error, use its Retry action once, and continue from the last completed upstream stage.
- If the health state is unavailable, click `Retry health check`, verify the container logs contain only safe lifecycle fields, and use the architecture explanation rather than claiming a completed generation.
- If provider credentials are absent, demonstrate upload, metadata, Frames, and Audio inspection only; say that Gemma generation is not configured.

Close with: “GemmaClip turns video captioning from a black box into a glass box — with Gemma visible at every meaningful model decision.”
