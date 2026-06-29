param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$FrontendIndex = Join-Path $Root "frontend\dist\index.html"
$Requirements = Join-Path $Root "requirements.txt"

if (-not (Test-Path -LiteralPath $FrontendIndex)) {
    Write-Error "Frontend build is missing. Run: cd frontend; npm install; npm run build"
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Python launcher 'py' was not found. Install Python 3.10 or put it on PATH."
}

Push-Location $Root
try {
    & py -3.10 -c "import fastapi, uvicorn, tensorboard" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python dependencies are missing. Run: py -3.10 -m pip install -r `"$Requirements`""
    }

    Write-Host "Starting Rnv1 ReTrain at http://$HostAddress`:$Port"
    & py -3.10 -m uvicorn backend.main:app --host $HostAddress --port $Port
}
finally {
    Pop-Location
}
