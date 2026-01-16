# RELiSH Hub - Quick Start Script
# Save as: backend/run_hub.ps1
#
# Usage:
#   ./run_hub.ps1                    # Use localhost (testing on same machine)
#   ./run_hub.ps1 192.168.1.100      # Use LAN IP (Quest on same network)
#   ./run_hub.ps1 https://xyz.ngrok-free.app  # Use dev tunnel

param(
    [string]$BaseUrl = ""
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "RELiSH MR Hub - Starting..." -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Set base URL if provided
if ($BaseUrl -ne "") {
    Write-Host "Base URL: $BaseUrl" -ForegroundColor Yellow
    $env:RELISH_BASE_URL = $BaseUrl
} else {
    Write-Host "Base URL: (will use request URL)" -ForegroundColor Yellow
}

Write-Host ""

# Run with conda (more robust than 'conda activate' in scripts)
conda run -n relish-hub --no-capture-output uvicorn app:app --host 0.0.0.0 --port 8000 --reload