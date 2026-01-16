cd backend

# Option 1: Localhost testing (same machine)
./run_hub.ps1

# Option 2: LAN testing (Quest on same network)
# First, find your PC's LAN IP: ipconfig | findstr IPv4
./run_hub.ps1 http://192.168.1.100:8000

# Option 3: Dev tunnel (for HTTPS, required for WebRTC on LAN)
# First: Install ngrok or similar, then:
./run_hub.ps1 https://abc123.ngrok-free.app
```

**Expected output:**
```
======================================
RELiSH MR Hub - Starting...
======================================
Base URL: (will use request URL)

============================================================
RELiSH MR Hub - Phase 1
============================================================
Version: 0.1.0
Docs:    http://localhost:8000/docs
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000