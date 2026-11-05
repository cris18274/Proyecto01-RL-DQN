# Demo corta de la canalizacion, NO evidencia de investigacion.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$out = "results/live_smoke_$(Get-Date -Format yyyyMMdd_HHmmss)"
Write-Host "Demo aislada: $out"
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m rl01.cli smoke --out $out
Write-Host "Archivo de ejemplo: $out\dqn_base_seed0007\summary.json"
