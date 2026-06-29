param(
    [switch]$Recreate,
    [switch]$SkipFrontend,
    [switch]$SkipCudaCheck,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $Root ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $Root "requirements.txt"
$TrainingRequirements = Join-Path $Root "requirements-training.txt"
$TorchWheelhouse = "F:\sdks\python-wheelhouse\pytorch-cu128"
$TrainingWheelhouse = "F:\sdks\python-wheelhouse\mok-training"

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "==> $Label"
    & $Command
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Python launcher 'py' was not found. Install Python 3.10 before running this installer."
}

if ($Recreate -and (Test-Path -LiteralPath $VenvDir)) {
    Invoke-Step "Removing existing .venv" {
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    Invoke-Step "Creating ReTrain .venv with Python 3.10" {
        Push-Location $Root
        try {
            & py -3.10 -m venv .venv
        }
        finally {
            Pop-Location
        }
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Failed to create $Python"
}

$FindLinks = @()
foreach ($path in @($TorchWheelhouse, $TrainingWheelhouse)) {
    if (Test-Path -LiteralPath $path) {
        $FindLinks += @("--find-links", $path)
    }
}

$IndexArgs = @()
if ($Offline) {
    if ($FindLinks.Count -eq 0) {
        Write-Error "Offline install requested, but no local wheelhouse paths were found."
    }
    $IndexArgs += "--no-index"
}

Invoke-Step "Upgrading pip tooling" {
    & $Python -m pip install --upgrade pip
}

Invoke-Step "Installing ReTrain app and CUDA training dependencies" {
    & $Python -m pip install @IndexArgs @FindLinks -r $Requirements -r $TrainingRequirements
}

Invoke-Step "Validating Python training environment" {
    $validation = @'
import importlib.util
import sys

required = [
    "fastapi",
    "uvicorn",
    "httpx",
    "tensorboard",
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing packages: " + ", ".join(missing))

import torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda:", torch.version.cuda)
'@
    $validation | & $Python -
}

if (-not $SkipCudaCheck) {
    Invoke-Step "Checking CUDA visibility" {
        $cudaCheck = @'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible from ReTrain .venv.")
print(torch.cuda.get_device_name(0))
'@
        $cudaCheck | & $Python -
    }
}

if (-not $SkipFrontend) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Error "npm was not found. Install Node.js or rerun with -SkipFrontend."
    }
    Invoke-Step "Installing frontend dependencies" {
        Push-Location (Join-Path $Root "frontend")
        try {
            if (Test-Path -LiteralPath "package-lock.json") {
                npm ci
            }
            else {
                npm install
            }
        }
        finally {
            Pop-Location
        }
    }
    Invoke-Step "Building frontend" {
        Push-Location (Join-Path $Root "frontend")
        try {
            npm run build
        }
        finally {
            Pop-Location
        }
    }
}

Invoke-Step "Running ReTrain smoke checks" {
    Push-Location $Root
    try {
        & $Python -m compileall backend scripts
        & $Python datasets\codex_app_environment\scripts\validate_codex_app_dataset.py
        & $Python scripts\run_posttrain_bakeoff.py --dry-run --model qwen2.5-coder-1.5b --data-dir C:\Users\Shawn\Desktop\MoK-Project\training\posttrain_bakeoff\data --output-root training\runs\installer-smoke
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "ReTrain install complete."
Write-Host "Launch with: .\scripts\start_retrain.ps1"
