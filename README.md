# GemmaClip

GemmaClip is a Track 2 AMD Developer Hackathon baseline agent. This milestone reads `/input/tasks.json`, downloads each video, probes metadata with `ffprobe`, extracts uniform frames with `ffmpeg`, and writes placeholder captions to `/output/results.json`.

Local run:

```bash
python -m pip install -e .[dev]
python -m gemmaclip.main --input examples/tasks.json --output output/results.json
```

Frame debug artifacts:

```bash
python -m gemmaclip.main \
  --input examples/tasks.json \
  --output output/results.json \
  --workdir /tmp/gemmaclip \
  --debug-dir debug-frames
```

This stores downloaded videos under `<workdir>/videos/`, extracted frames under `<workdir>/frames/<task_id>/`, and writes `<workdir>/frame_manifest.json`. When `--debug-dir` is set, GemmaClip also copies frames into `<debug-dir>/<task_id>/` and writes `<debug-dir>/<task_id>_contact_sheet.jpg`.

PowerShell venv activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Make targets:

```bash
make install-dev
make test
make run-local INPUT=examples/tasks.json OUTPUT=output/results.json
```

Docker build:

```bash
docker buildx build --platform linux/amd64 --tag gemmaclip:latest .
```
