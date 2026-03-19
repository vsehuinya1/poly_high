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

echo "🔄 Pulling on VPS..."
./vps.sh "git fetch origin main && git reset --hard origin/main"

echo "🔪 Killing old engine(s)..."
# Use nohup kill to avoid SSH dying with the process
./vps.sh "nohup bash -c 'kill -9 \$(pgrep -f python3.*sports.main) 2>/dev/null' &>/dev/null &"
sleep 3

echo "🚀 Starting new engine..."
LOGNAME="sports_$(date -u +%Y%m%d_%H%M).log"
./vps.sh "nohup /usr/bin/python3 -u -m sports.main > logs/$LOGNAME 2>&1 &"
sleep 2

echo "⏳ Waiting 12s for startup..."
sleep 12

echo "✅ Verifying..."
./vps.sh "pgrep -c -f 'python3.*sports.main' 2>/dev/null; grep -E 'STATUS|Error|Traceback' logs/$LOGNAME 2>/dev/null | head -3"

echo ""
echo "Done! Log: logs/$LOGNAME"
