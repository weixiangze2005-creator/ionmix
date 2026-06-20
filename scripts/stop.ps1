$ErrorActionPreference = "SilentlyContinue"
$Root = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Root ".server.pid"

if (Test-Path $PidFile) {
    $ServerPid = [int](Get-Content $PidFile)
    Stop-Process -Id $ServerPid
    Remove-Item -LiteralPath $PidFile
    Write-Host "Application stopped."
} else {
    Write-Host "No recorded application process was found."
}

