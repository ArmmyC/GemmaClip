# GemmaClip

GemmaClip is a Track 2 AMD Developer Hackathon baseline agent. This milestone reads `/input/tasks.json`, downloads each video, probes metadata with `ffprobe`, extracts uniform frames with `ffmpeg`, and writes placeholder captions to `/output/results.json`.

Local run:

```bash
python -m pip install -e .[dev]
python -m gemmaclip.main --input examples/tasks.json --output output/results.json
```

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
