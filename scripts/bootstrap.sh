#!/usr/bin/env bash
###############################################################################
# AIDAC / TLL-RDBMS BOOTSTRAP SCRIPT
# Author : William K.
# Purpose: Reproducible environment provisioning for AI-DAC, TLL pipeline,
#          meta-learning modules, Swift Hydra/MAD-GAN, and RDBMS interfaces.
# Target : Ubuntu 20.04+ / Debian-based systems / WSL2-compatible
###############################################################################

set -euo pipefail

LOGFILE="/var/log/aidac_bootstrap.log"
VENV_PATH="/opt/aidac-venv"
REPO_ROOT="$(pwd)"

###############################################################################
# Helper functions
###############################################################################
log() {
    echo "[BOOTSTRAP][$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $1" | tee -a "$LOGFILE"
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "This script must be run as root (sudo)" >&2
        exit 1
    fi
}

###############################################################################
# 0. Pre-checks
###############################################################################
require_root

log "Starting bootstrap process…"
mkdir -p /var/log/aidac

log "Updating package lists…"
apt-get update -y

###############################################################################
# 1. Install system dependencies
###############################################################################
log "Installing base system utilities…"
apt-get install -y \
    python3 python3-venv python3-pip \
    git jq curl wget unzip htop tmux \
    build-essential pkg-config \
    ca-certificates software-properties-common

###############################################################################
# 2. Install PostgreSQL & SQL Server client tools
###############################################################################
log "Installing PostgreSQL client…"
apt-get install -y postgresql-client

log "Installing SQL Server tools (msodbcsql + mssql-tools)…"
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list \
    > /etc/apt/sources.list.d/mssql-tools.list

apt-get update -y
ACCEPT_EULA=Y apt-get install -y mssql-tools msodbcsql18 unixodbc-dev
echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
source ~/.bashrc || true

###############################################################################
# 3. Python virtual environment & dependencies
###############################################################################
log "Creating Python virtual environment at $VENV_PATH…"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

log "Installing Python packages…"
if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
    pip install --upgrade pip
    pip install -r "$REPO_ROOT/requirements.txt"
else
    log "WARNING: requirements.txt not found; installing minimal dependencies…"
    pip install numpy pandas torch scikit-learn faiss-cpu shap
fi

###############################################################################
# 4. Create required directories
###############################################################################
log "Creating directory structure…"
mkdir -p /opt/aidac/{models,calibration,manifests,evidence,cache}
mkdir -p /var/aidac/{logs,artifacts}

###############################################################################
# 5. Configure monitoring agents (optional)
###############################################################################
if command -v systemctl >/dev/null 2>&1; then
    log "Systemd detected — ready for monitoring agent integration."
else
    log "Systemd not available; skipping monitoring agent provisioning."
fi

###############################################################################
# 6. Validate installation
###############################################################################
log "Validating Python environment…"
python3 - <<'EOF'
import sys, torch
print("[Validation] Python OK:", sys.version)
print("[Validation] PyTorch:", torch.__version__)
print("[Validation] CUDA available:", torch.cuda.is_available())
EOF

###############################################################################
# Finish
###############################################################################
log "Bootstrap complete. Environment ready."

echo
echo "-------------------------------------------------------"
echo " AIDAC / TLL-RDBMS Environment Successfully Prepared"
echo " Virtual Env  : $VENV_PATH"
echo " Log File     : $LOGFILE"
echo " Author       : William K."
echo "-------------------------------------------------------"
echo
