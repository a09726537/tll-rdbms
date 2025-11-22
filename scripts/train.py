#!/usr/bin/env bash
#
# ----------------------------------------------------------------------
#  File Name     : vm-init.sh
#  Description   : VM initialisation script for AI-DAC / TLL-RDBMS lab
#  Author        : William K.
#  Created       : 2025-01-01
#  Last Updated  : 2025-11-22
# ----------------------------------------------------------------------
#
#  This script prepares an Ubuntu VM for full end-to-end AI-DAC experiments.
#  It configures system limits, installs dependencies, validates datasets,
#  and ensures a fully reproducible runtime environment.
#
# ----------------------------------------------------------------------

set -euo pipefail

LOGFILE="/var/log/aidac-vm-init.log"
echo "[AIDAC] Starting VM initialisation..." | tee -a "$LOGFILE"

# ----------------------------------------------------------------------
# 1. Update system packages
# ----------------------------------------------------------------------
echo "[AIDAC] Updating package lists..." | tee -a "$LOGFILE"
sudo apt-get update -y >> "$LOGFILE" 2>&1
sudo apt-get upgrade -y >> "$LOGFILE" 2>&1

# ----------------------------------------------------------------------
# 2. Install system dependencies
# ----------------------------------------------------------------------
echo "[AIDAC] Installing required system packages..." | tee -a "$LOGFILE"
sudo apt-get install -y \
    git \
    jq \
    unzip \
    curl \
    wget \
    htop \
    lsof \
    net-tools \
    python3 \
    python3-pip \
    python3-venv \
    postgresql-client \
    docker.io \
    >> "$LOGFILE" 2>&1

# ----------------------------------------------------------------------
# 3. Configure system performance limits
# ----------------------------------------------------------------------
echo "[AIDAC] Configuring system performance limits..." | tee -a "$LOGFILE"

sudo bash -c 'cat >/etc/security/limits.conf <<EOF
* soft nofile 500000
* hard nofile 500000
* soft nproc  500000
* hard nproc  500000
EOF'

sudo bash -c 'cat >/etc/sysctl.d/80-aidac.conf <<EOF
fs.file-max = 500000
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
vm.max_map_count = 1048576
EOF'

sudo sysctl --system >> "$LOGFILE" 2>&1

# ----------------------------------------------------------------------
# 4. Create working directories
# ----------------------------------------------------------------------
echo "[AIDAC] Creating working directories..." | tee -a "$LOGFILE"

sudo mkdir -p /opt/aidac/{datasets,models,logs,artifacts}
sudo chmod -R 755 /opt/aidac

# ----------------------------------------------------------------------
# 5. Validate datasets (if present)
# ----------------------------------------------------------------------
DATASET_DIR="/opt/aidac/datasets"

echo "[AIDAC] Checking datasets..." | tee -a "$LOGFILE"

if [ -d "$DATASET_DIR" ]; then
    for f in "$DATASET_DIR"/*.zip "$DATASET_DIR"/*.tar.gz; do
        if [ -f "$f" ]; then
            echo " - Validating checksum for: $f" | tee -a "$LOGFILE"
            sha256sum "$f" >> "$LOGFILE" 2>&1 || echo "   WARNING: checksum failed for $f"
        fi
    done
else
    echo "No dataset directory found; skipped." | tee -a "$LOGFILE"
fi

# ----------------------------------------------------------------------
# 6. Enable Docker service
# ----------------------------------------------------------------------
echo "[AIDAC] Enabling Docker service..." | tee -a "$LOGFILE"
sudo systemctl enable docker >> "$LOGFILE" 2>&1
sudo systemctl start docker  >> "$LOGFILE" 2>&1

# ----------------------------------------------------------------------
# 7. Optional: Set up pipeline auto-start service
# ----------------------------------------------------------------------
echo "[AIDAC] Creating systemd service for auto-start..." | tee -a "$LOGFILE"

sudo bash -c 'cat >/etc/systemd/system/aidac-pipeline.service <<EOF
[Unit]
Description=AIDAC Pipeline Auto-Start
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/aidac/run_pipeline.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF'

sudo systemctl daemon-reload
sudo systemctl enable aidac-pipeline.service >> "$LOGFILE" 2>&1

# ----------------------------------------------------------------------
# 8. Final Summary
# ----------------------------------------------------------------------

echo
echo "-------------------------------------------------------"
echo " AIDAC / TLL-RDBMS VM Initialisation Complete"
echo " Log File     : $LOGFILE"
echo " Author       : William K."
echo "-------------------------------------------------------"
echo

exit 0
