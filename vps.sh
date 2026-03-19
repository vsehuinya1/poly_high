#!/bin/bash
# VPS helper — run any command on the VPS without hanging
# Usage: ./vps.sh "command to run"
# Example: ./vps.sh "grep STATUS logs/sports_v461.log | tail -1"

VPS_HOST="root@161.97.185.65"
VPS_PASS="12345vse"
VPS_DIR="/root/poly_high_sports"

sshpass -p "$VPS_PASS" ssh -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -o StrictHostKeyChecking=no "$VPS_HOST" "cd $VPS_DIR && $1" 2>/dev/null
