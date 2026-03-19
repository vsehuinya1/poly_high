#!/bin/bash
# Full deploy: commit → push → pull on VPS → restart engine
# Usage: ./deploy.sh "commit message"

set -e
MSG="${1:-deploy}"

echo "📦 Committing..."
cd /Users/MartinOile/Desktop/poly_high
git add -A && git commit -m "$MSG" && git push origin main

echo "🔄 Pulling on VPS..."
./vps.sh "git fetch origin main && git reset --hard origin/main"

echo "🔪 Killing old engine..."
./vps.sh "kill -9 \$(pgrep -f 'python3.*sports.main') 2>/dev/null; sleep 2; echo killed"

echo "🚀 Starting new engine..."
LOGFILE="logs/sports_$(date +%Y%m%d_%H%M).log"
./vps.sh "nohup /usr/bin/python3 -u -m sports.main > $LOGFILE 2>&1 &"

echo "⏳ Waiting 15s for startup..."
sleep 15

echo "✅ Verifying..."
./vps.sh "pgrep -c -f 'python3.*sports.main'; grep -E 'STATUS|Error|Traceback' $LOGFILE | head -5"

echo "Done! Log: $LOGFILE"
