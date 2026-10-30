# Install the team's MPM-side versions into .work\mpm_site, for the CAPSTONE MPM scripts only.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_mpm_env.ps1
#
# Newton 1.6.0 / warp 1.17.0 / mujoco_warp 3.12.0 / mujoco 3.12.0 (CAPSTONE
# docs/framework_reuse_plan_2026-10-05.md), plus newton-usd-schemas (Newton 1.6's USD importer).
# Installed with --no-deps: their other requirements (numpy, absl-py, etils, glfw, pyopengl,
# usd-core, scipy, trimesh) come from the Kit Python. Isaac Lab is not affected: only
# scripts\capstone.py puts this folder on sys.path.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pathsFile = Join-Path $projectRoot 'configs\paths.json'
if (-not (Test-Path $pathsFile)) { $pathsFile = Join-Path $projectRoot 'configs\paths.example.json' }
$labRoot = (Get-Content -Raw -Encoding UTF8 $pathsFile | ConvertFrom-Json).isaaclab_root
$work = Join-Path $projectRoot '.work'
$wheels = Join-Path $work 'mpm_wheels'
$site = Join-Path $work 'mpm_site'
$env:CONDA_PREFIX = ''
$env:VIRTUAL_ENV = ''
$bat = Join-Path $labRoot 'isaaclab.bat'
& $bat -p -m pip download --no-deps --dest $wheels newton==1.6.0 warp-lang==1.17.0 mujoco-warp==3.12.0 mujoco==3.12.0 newton-usd-schemas==0.5.0
& $bat -p -m pip install --no-deps --no-index --upgrade --target $site (Get-ChildItem (Join-Path $wheels '*.whl')).FullName
Write-Host "installed into $site"

# NVIDIA's published H1 rough-terrain checkpoint: CAPSTONE 07/22 need it (git-ignored there).
$ckpt = Join-Path $work 'checkpoints\h1_published.pt'
$sha = '054f5de730d8b2fcb56a602864b5cb5d50074946cfa98278f588ea17f30b66e7'
if (-not (Test-Path $ckpt)) {
    New-Item -ItemType Directory -Force (Split-Path $ckpt) | Out-Null
    $url = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-H1-v0/checkpoint.pt'
    Invoke-WebRequest -Uri $url -OutFile $ckpt
}
if ((Get-FileHash -Algorithm SHA256 $ckpt).Hash.ToLower() -ne $sha) { throw "unexpected checkpoint hash: $ckpt" }
Write-Host "H1 published checkpoint: $ckpt"
