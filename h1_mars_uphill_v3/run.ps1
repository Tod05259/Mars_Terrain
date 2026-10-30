# Run a Python script of this project with Isaac Lab's bundled (Kit) Python.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 scripts\preflight.py [args...]
#
# Paths come from configs\paths.json (your PC) or configs\paths.example.json (defaults).
# Relative script paths are resolved against this folder, and the working directory is this
# folder, so Isaac Lab logs land in .\logs (git-ignored).
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Script,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot

$pathsFile = Join-Path $projectRoot 'configs\paths.json'
if (-not (Test-Path $pathsFile)) { $pathsFile = Join-Path $projectRoot 'configs\paths.example.json' }
$paths = Get-Content -Raw -Encoding UTF8 $pathsFile | ConvertFrom-Json
$labRoot = [System.IO.Path]::GetFullPath($paths.isaaclab_root)
$capstoneRoot = [System.IO.Path]::GetFullPath($paths.capstone_root)
foreach ($p in @($labRoot, $capstoneRoot)) {
    if (-not (Test-Path $p)) { throw "Path not found: $p (edit $pathsFile)" }
}

$workRoot = Join-Path $projectRoot '.work'
foreach ($name in @('tmp', 'pycache', 'mpl')) {
    New-Item -ItemType Directory -Path (Join-Path $workRoot $name) -Force | Out-Null
}
# isaaclab.bat picks a conda/venv Python first; clear them so the bundled Kit Python is used.
$env:CONDA_PREFIX = ''
$env:VIRTUAL_ENV = ''
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
# Keep bytecode of the (read-only) CAPSTONE scripts out of the CAPSTONE tree.
$env:PYTHONPYCACHEPREFIX = Join-Path $workRoot 'pycache'
$env:MPLCONFIGDIR = Join-Path $workRoot 'mpl'
$env:UPHILL_ROOT = $projectRoot
$env:UPHILL_ISAACLAB = $labRoot
$env:UPHILL_CAPSTONE = $capstoneRoot
$env:UPHILL_CAPSTONE_COMMIT = $paths.capstone_commit
# Project package first, then CAPSTONE/terrain (for getup_lab).
$env:PYTHONPATH = $projectRoot + ';' + (Join-Path $capstoneRoot 'terrain') + ';' + $env:PYTHONPATH

if (-not [System.IO.Path]::IsPathRooted($Script)) { $Script = Join-Path $projectRoot $Script }
Push-Location $projectRoot
try {
    & (Join-Path $labRoot 'isaaclab.bat') -p $Script @ScriptArgs
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
