# Setup Virtual Environment and Dependencies for TrendPulse-AI (PowerShell)
Write-Host "Creating Python Virtual Environment..." -ForegroundColor Cyan
py -3.11 -m venv .venv
if (-not $?) {
    Write-Host "Defaulting to active python interpreter..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "Activating Virtual Environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip & installing dependencies..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

Write-Host "Setup Completed Successfully! Run .\scripts\run_dev.ps1 to launch application." -ForegroundColor Green
