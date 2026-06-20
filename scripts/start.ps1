$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = $Root
$env:IONMIX_CONDUCTIVITY_MODEL = "enabled"
& ".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
