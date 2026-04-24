#!/bin/bash
# Full deploy: commit → push → pull on VPS → restart engine
# Usage: ./deploy.sh "commit message"

set -e
MSG="${1:-deploy}"
DIR="/Users/MartinOile/Desktop/poly_high"
cd "$DIR"

echo "📦 Committing..."
git add -A
git diff --cached --quiet 2>/dev/null && echo "  (nothing to commit)" || git commit -m "$MSG"
git push origin main 2>/dev/null || echo "  (already pushed)"

echo "🔄 Deploy + restart (single SSH call)..."
LOG_FILE="sports_$(date -u +%Y%m%d_%H%M).log"

# Do EVERYTHING in one SSH call to avoid connection drops
sshpass -p '12345vse' ssh \
  -o ConnectTimeout=10 \
  -o ServerAliveInterval=5 \
  -o StrictHostKeyChecking=no \
  root@161.97.185.65 "
    cd /root/poly_high_sports &&
    git fetch origin main 2>/dev/null &&
    git reset --hard origin/main 2>/dev/null &&
    echo 'PULLED' &&
    pkill -9 -f 'python3.*sports.main' 2>/dev/null; sleep 2 &&
    echo 'KILLED' &&
    nohup /usr/bin/python3 -u -m sports.main > logs/${LOG_FILE} 2>&1 &
    echo 'STARTED' &&
    sleep 12 &&
    echo 'ENGINE COUNT:' &&
    pgrep -c -f 'python3.*sports.main' 2>/dev/null &&
    echo 'LATEST LOG:' &&
    grep -E 'STATUS|TENNIS_DIAG|Error|Traceback' logs/${LOG_FILE} 2>/dev/null | head -5
  " 2>/dev/null || echo "  (SSH timeout — check ./status.sh later)"

echo ""
echo "Done! Log: logs/$LOG_FILE"
