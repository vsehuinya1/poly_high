#!/bin/bash
# Quick status check — single SSH call
# Usage: ./status.sh

timeout 20 sshpass -p '12345vse' ssh \
  -o ConnectTimeout=10 \
  -o ServerAliveInterval=5 \
  -o StrictHostKeyChecking=no \
  root@161.97.185.65 "
    cd /root/poly_high_sports
    echo '=== ENGINE ==='
    pgrep -c -f 'python3.*sports.main' 2>/dev/null || echo '0'
    LATEST=\$(ls -t logs/*.log 2>/dev/null | head -1)
    echo '=== STATUS ==='
    grep 'STATUS' \$LATEST 2>/dev/null | tail -1
    echo '=== TENNIS DIAG ==='
    grep 'TENNIS_DIAG' \$LATEST 2>/dev/null | tail -3
    echo '=== TENNIS SIGNALS ==='
    grep -E 'TENNIS SIGNAL|TENNIS_PENDING|TENNIS_DELAYED' \$LATEST 2>/dev/null | tail -5
    echo '=== TICK DB ==='
    ls -lh sports_data/tick_history.db 2>/dev/null
    grep 'TICK_RECORDER' \$LATEST 2>/dev/null | tail -1
    echo '=== ERRORS ==='
    grep -E 'Error|Traceback|ImportError' \$LATEST 2>/dev/null | head -5
  " 2>/dev/null || echo "(SSH timeout — VPS may be unreachable)"

echo "Done."
