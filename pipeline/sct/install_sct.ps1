# Non-interactive Spinal Cord Toolbox 7.3 install, the same steps as the official
# install_sct-7.3_win.bat (which prompts for its folder and pauses at the end):
# clone the 7.3 tag, a portable Miniforge, a Python 3.10 env, requirements-freeze.txt, SCT itself,
# its Windows binaries, default data (PAM50 template included) and default models.
# Licences: SCT code LGPL-3.0; PAM50-normalized-metrics MIT; PAM50 template has no licence
# (permission pending, see LICENSING.md).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File install_sct.ps1 [-Dir <folder>]
param([string]$Dir = (Join-Path $PSScriptRoot "..\..\..\_tools\sct_7.3"))
$ErrorActionPreference = "Stop"
$Dir = [IO.Path]::GetFullPath($Dir)
function Step($m) { Write-Output "### $(Get-Date -Format HH:mm:ss) $m" }
if (-not (Test-Path "$Dir\spinalcordtoolbox\version.txt")) {
    Step "clone SCT 7.3 into $Dir"
    git clone -b 7.3 --single-branch --depth 1 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git $Dir
    if ($LASTEXITCODE) { throw "git clone failed" }
}
Set-Location $Dir
if (-not (Test-Path "$Dir\python\Scripts\conda.exe")) {
    Step "portable Miniforge 24.11.2-1"
    $mf = Join-Path $env:TEMP "miniforge-sct.exe"
    curl.exe -sSL -o $mf https://github.com/conda-forge/miniforge/releases/download/24.11.2-1/Miniforge3-Windows-x86_64.exe
    Start-Process -FilePath $mf -ArgumentList "/InstallationType=JustMe", "/AddToPath=0", "/RegisterPython=0", "/NoRegistry=1", "/S", "/D=$Dir\python" -Wait -WindowStyle Hidden
}
if (-not (Test-Path "$Dir\python\envs\venv_sct\python.exe")) {
    Step "conda env (Python 3.10)"
    & "$Dir\python\Scripts\conda.exe" create -y -p "$Dir\python\envs\venv_sct" python=3.10
    if ($LASTEXITCODE) { throw "conda create failed" }
}
$py = "$Dir\python\envs\venv_sct\python.exe"
$scripts = "$Dir\python\envs\venv_sct\Scripts"
Step "pip requirements"
& $py -m pip install -U "pip!=21.2.*"
& "$scripts\pip.exe" install -r requirements-freeze.txt
if ($LASTEXITCODE) { throw "requirements failed" }
& "$scripts\pip.exe" install -e . --use-pep517
if ($LASTEXITCODE) { throw "SCT install failed" }
Step "binaries, default data and models"
& "$scripts\sct_download_data.exe" -d binaries_win -k
& "$scripts\sct_download_data.exe" -d default -k
& $py -c "import spinalcordtoolbox.deepseg.models; spinalcordtoolbox.deepseg.models.install_default_models()"
Step "check"
& "$scripts\sct_check_dependencies.exe"
Step "done: $Dir"
