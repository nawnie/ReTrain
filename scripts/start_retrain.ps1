param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$FrontendIndex = Join-Path $Root "frontend\dist\index.html"
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $FrontendIndex)) {
    Write-Error "Frontend build is missing. Run: .\scripts\install_retrain.ps1"
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "ReTrain .venv is missing. Run: .\scripts\install_retrain.ps1"
}

Push-Location $Root
try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $dependencyCheck = & $Python -c "import fastapi, uvicorn, tensorboard, torch, transformers, accelerate, peft, bitsandbytes" 2>&1
    $dependencyExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($dependencyExitCode -ne 0) {
        $dependencyCheck | ForEach-Object { Write-Host $_ }
        Write-Error "ReTrain .venv is missing app or training dependencies. Run: .\scripts\install_retrain.ps1"
    }

    Write-Host "Starting Rnv1 ReTrain at http://$HostAddress`:$Port"
    $ErrorActionPreference = "Continue"
    & $Python -m uvicorn backend.main:app --host $HostAddress --port $Port
}
finally {
    Pop-Location
}
