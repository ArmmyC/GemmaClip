# Fireworks leaderboard pipeline

The `fireworks_leaderboard` provider is a competition-only CLI path. It is
separate from the public React/FastAPI demo and does not change the demo's
routed-Gemma provider, prompts, storage, or web image.

## Provider and model order

Select it with:

```text
GEMMACLIP_PROVIDER=fireworks_leaderboard
```

The default quality-first order is:

| Role | Default model | Fallback |
| --- | --- | --- |
| Caption generation | `accounts/fireworks/models/qwen3p7-plus` | `accounts/fireworks/models/minimax-m3` |
| Independent visual review | `accounts/fireworks/models/minimax-m3` | `accounts/fireworks/models/qwen3p7-plus` |

Model identifiers are opaque runtime configuration values. Availability and
model quality require live verification with the submitting account.

Supported environment variables include:

```text
FIREWORKS_API_KEY
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_LEADERBOARD_GENERATION_MODEL=accounts/fireworks/models/qwen3p7-plus
FIREWORKS_LEADERBOARD_REVIEW_MODEL=accounts/fireworks/models/minimax-m3
FIREWORKS_LEADERBOARD_FALLBACK_MODEL=accounts/fireworks/models/minimax-m3
FIREWORKS_LEADERBOARD_REVIEW_FALLBACK_MODEL=accounts/fireworks/models/qwen3p7-plus
FIREWORKS_LEADERBOARD_ENABLE_REVIEW=true
FIREWORKS_LEADERBOARD_GENERATION_TEMPERATURE=0.35
FIREWORKS_LEADERBOARD_REPAIR_TEMPERATURE=0.20
FIREWORKS_LEADERBOARD_REVIEW_TEMPERATURE=0.0
FIREWORKS_LEADERBOARD_MIN_GENERATION_REMAINING_SECONDS=65
FIREWORKS_LEADERBOARD_MIN_REVIEW_REMAINING_SECONDS=150
GEMMA_MAX_TOKENS=2048
```

Temperatures are finite, clamped to 0–2, and use conservative defaults when
malformed. Common true/false forms are accepted for review enablement. An
empty or missing API key produces deterministic valid fallback captions and
does not attempt a remote request.

## Frames and request flow

The CLI reuses the existing Fireworks six-frame extractor. Hybrid selection
tries four temporal anchors plus two high-change moments, sorts the result
chronologically, and falls back to six deterministic uniform timestamps when
scanning or extraction fails. Short and static clips still receive six
uniform frames. Each request contains six separate JPEG data URLs; it never
sends a contact sheet, raw video, or audio.

The normal remote-call order is:

```text
Qwen generation
  -> MiniMax generation fallback when the first attempt is retryable or invalid
  -> focused repair with Qwen, then MiniMax only for styles still missing
  -> MiniMax independent visual review
  -> Qwen review fallback if review is fallback-eligible and time remains
```

A model-not-found response (`404`) advances once to the configured fallback
model even though it is not retried on the same model. Authentication or
permission failures (`401`/`403`) stop model fallback for that operation.

Valid captions are retained byte-for-byte during focused repair. A failed
review never discards valid generation output. The competition output remains
exactly `task_id` plus the requested caption keys.

## Runtime behavior

The global application budget remains 570 seconds, with the existing 20-second
final-write buffer and 590-second container watchdog. The live deadline is
checked immediately before every generation, repair, and review request.

- Fewer than 65 seconds: do not start generation or repair; use valid local
  fallback captions.
- At least 65 seconds: generation and focused repair may start.
- Fewer than 150 seconds after generation/repair: skip independent review.
- At least 150 seconds: run the independent visual review when enabled.

Progressive `/output/results.json` writes and schema validation after every
completed task remain unchanged. A task that cannot be processed still gets
valid captions so later tasks can continue.

## Local CLI

Install the package and run against the normal mounted-contract-shaped files:

```bash
python -m pip install -e .[dev]
GEMMACLIP_PROVIDER=fireworks_leaderboard \
FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
python -m gemmaclip.main \
  --input examples/tasks.json \
  --output output/results.json \
  --workdir /tmp/gemmaclip
```

Without a key, the same command is a credential-free fallback smoke test.

## Competition image

Build the Python-only competition image with its non-secret defaults:

```bash
docker build -f Dockerfile -t gemmaclip-cli:leaderboard-rc .
```

Run it with the competition mount contract:

```bash
docker run --rm \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -v "$PWD/examples:/input:ro" \
  -v "$PWD/output:/output" \
  gemmaclip-cli:leaderboard-rc
```

The image reads `/input/tasks.json` and writes `/output/results.json`. Runtime
secrets should be supplied at runtime, never committed or printed during the
build. The image retains FFmpeg/FFprobe, Python 3.12, the existing timeout
entrypoint, and `linux/amd64` compatibility.

## Sanitized live smoke

When a key is available, use controlled local fixtures covering a visual-only
clip, static or low-motion clip, multiple visible actions, a vertical video,
and a short clip. Record only the date, commit SHA, sanitized fixture name,
model names, whether fallback was used, runtime, requested styles, word
counts, pass/fail, and short sanitized notes. Never record keys, headers,
private URLs, raw responses, prompts, captions, media bytes, or base64 data.

No live provider compatibility claim is made by the unit tests or by a
credential-free container run.

## A/B model assignment

To reverse the generation and review assignments without changing code, set:

```text
FIREWORKS_LEADERBOARD_GENERATION_MODEL=accounts/fireworks/models/minimax-m3
FIREWORKS_LEADERBOARD_FALLBACK_MODEL=accounts/fireworks/models/qwen3p7-plus
FIREWORKS_LEADERBOARD_REVIEW_MODEL=accounts/fireworks/models/qwen3p7-plus
```

The review fallback remains Qwen by default; set
`FIREWORKS_LEADERBOARD_REVIEW_FALLBACK_MODEL` explicitly if the A/B test also
needs to reverse that final fallback.
