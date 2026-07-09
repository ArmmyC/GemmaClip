FROM python:3.12-slim

ARG GEMINI_API_KEY=""
ARG GEMINI_MODEL=""
ARG GOOGLE_API_KEY=""
ARG FIREWORKS_API_KEY=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV GEMINI_API_KEY=${GEMINI_API_KEY} \
    GOOGLE_API_KEY=${GOOGLE_API_KEY} \
    FIREWORKS_API_KEY=${FIREWORKS_API_KEY} \
    GEMINI_MODEL=${GEMINI_MODEL}

ENV GEMMACLIP_DISABLE_VERIFIER=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "gemmaclip.main"]
