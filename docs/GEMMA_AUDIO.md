# Gemma audio

GemmaClip uses audio as a low-cost routing signal and as bounded evidence for Gemma 4 12B Unified. It does not make a separate remote speech-classification call and does not treat high energy as proof of speech.

## Preprocessing

For `auto` and `always`, GemmaClip uses `ffprobe` to detect an audio stream and `ffmpeg` to extract mono 16-bit PCM WAV. It measures deterministic RMS energy over short windows and selects the highest-energy contiguous window, capped at 30 seconds by default. Missing streams, extraction errors, command timeouts, and effectively silent audio route to visual evidence. Temporary full-length PCM is deleted immediately; the selected window is deleted after evidence generation.

`off` always uses visual evidence. `always` selects Unified whenever extraction succeeds, audio is non-silent, and the runtime threshold is met. `auto` applies the same conservative checks and requires a useful positive-duration selected window. The model—not the RMS heuristic—decides whether speech or relevant audio information actually exists.

## Settings

```text
GEMMACLIP_AUDIO_MODE=auto
GEMMACLIP_AUDIO_MAX_SECONDS=30
GEMMACLIP_AUDIO_SAMPLE_RATE=16000
GEMMACLIP_AUDIO_MIN_RMS=0.01
GEMMACLIP_AUDIO_MIN_REMAINING_SECONDS=170
```

Invalid modes safely normalize to `auto`. Invalid numeric values use conservative defaults.

## Limitations and caption safety

Only the selected window is analyzed, so speech elsewhere may be missed. Loud non-speech audio may be routed to Unified; the evidence prompt must classify it rather than assume speech. Uncertain transcripts remain uncertain. Audio facts are caption-safe only when evidence status is `usable`, the fact appears in `allowed_caption_facts`, and it does not contradict the visible frames. Otherwise final captions cannot quote dialogue or mention speech, music, noise, intent, or other audio-derived claims.

Logs contain route metadata and timing but never raw bytes, base64 payloads, signed URLs, keys, private endpoints, or full provider responses. Debug evidence and captions are sanitized structured outputs only.
