$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
$env:PYTHONPATH = $Root
& ".venv\Scripts\python.exe" "scripts\sync_public_data.py"
& ".venv\Scripts\python.exe" "scripts\train_conductivity_model.py"
& ".venv\Scripts\python.exe" "scripts\train_lino3_solubility_model.py"
Write-Host "Setup complete. Run .\scripts\start.ps1 to launch the application."
