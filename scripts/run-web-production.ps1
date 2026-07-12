param(
  [string]$Image = $(if ($env:GEMMACLIP_WEB_IMAGE) { $env:GEMMACLIP_WEB_IMAGE } else { "gemmaclip-web:latest" }),
  [int]$Port = $(if ($env:GEMMACLIP_WEB_PORT) { [int]$env:GEMMACLIP_WEB_PORT } else { 8000 })
)

$ErrorActionPreference = "Stop"
docker build -f Dockerfile.web -t $Image .
docker volume create gemmaclip-runs | Out-Null

$envArgs = @()
if (Test-Path -LiteralPath ".env") {
  $envArgs = @("--env-file", ".env")
}

docker run --rm `
  --name gemmaclip-web `
  -p "${Port}:8000" `
  -e GEMMACLIP_WEB_HOST=0.0.0.0 `
  -e GEMMACLIP_WEB_PORT=8000 `
  -e GEMMACLIP_WEB_RUNS_DIR=/data/runs `
  @envArgs `
  -v gemmaclip-runs:/data/runs `
  $Image
