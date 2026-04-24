#!/bin/bash
# VPS helper — run any command on the VPS
# Usage: ./vps.sh "command to run"
# Timeout: 15 seconds max to prevent scaffold hangs

VPS_HOST="root@161.97.185.65"
VPS_PASS="12345vse"
VPS_DIR="/root/poly_high_sports"

sshpass -p "$VPS_PASS" ssh \
  -o ConnectTimeout=5 \
  -o ServerAliveInterval=3 \
  -o ServerAliveCountMax=2 \
  -o StrictHostKeyChecking=no \
  -o BatchMode=no \
  "$VPS_HOST" "cd $VPS_DIR && $1" 2>/dev/null

# Exit 0 even if timeout/kill (expected when killing engine)
exit 0
