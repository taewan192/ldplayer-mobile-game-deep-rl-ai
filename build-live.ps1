param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -m PyInstaller --noconfirm --clean --onedir --windowed `
    --name MillionLordsAI `
    --add-data 'live_lords/ocr.ps1;live_lords' `
    --add-data 'live_lords/assets;live_lords/assets' `
    launch_live.py
if ($LASTEXITCODE -ne 0) { throw 'MillionLordsAI build failed.' }
Write-Output (Join-Path $PSScriptRoot 'dist\MillionLordsAI\MillionLordsAI.exe')
