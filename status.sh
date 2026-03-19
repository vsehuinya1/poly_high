#!/bin/bash
# Quick status check — shows everything in one shot
# Usage: ./status.sh

DIR="/Users/MartinOile/Desktop/poly_high"
cd "$DIR"

echo "========================================="
echo " POLY_HIGH STATUS CHECK"
echo " $(date)"
echo "========================================="

echo ""
echo "🔧 ENGINE"
./vps.sh "ps aux | grep 'sports.main' | grep -v grep | wc -l | tr -d ' '"
./vps.sh "grep 'STATUS' logs/*.log 2>/dev/null | tail -1"

echo ""
echo "📼 TICK RECORDER"
./vps.sh "ls -lh sports_data/tick_history.db 2>/dev/null; grep 'TICK_RECORDER' logs/*.log 2>/dev/null | tail -1"

echo ""
echo "🏀 NBA/NCAA PENDING"
./vps.sh "grep -h 'PENDING_ENTRY\|DELAYED_ENTRY' logs/*.log 2>/dev/null | tail -5; echo 'total:'; grep -ch 'PENDING_ENTRY' logs/*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0"

echo ""
echo "🎾 TENNIS PENDING"
./vps.sh "grep -h 'TENNIS_PENDING\|TENNIS_DELAYED\|TENNIS SIGNAL\|TENNIS PAPER' logs/*.log 2>/dev/null | tail -5; echo 'total signals:'; grep -ch 'TENNIS SIGNAL' logs/*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0"

echo ""
echo "📈 MICROSTRUCTURE"
./vps.sh "grep -h 'MICROSTRUCTURE\|MICRO_ALERT' logs/*.log 2>/dev/null | tail -2"

echo ""
echo "📁 TODAY'S DATA"
./vps.sh "ls -lh sports_data/*$(date -u +%Y%m%d)* 2>/dev/null || echo 'none today'"

echo ""
echo "🏏 CRICKET"
./vps.sh "grep -ih 'cricket' logs/*.log 2>/dev/null | tail -2"

echo ""
echo "⚽ FOOTBALL"  
./vps.sh "grep -ih 'football.*live\|FOOTBALL_FEED' logs/*.log 2>/dev/null | tail -2"

echo ""  
echo "⚠️ ERRORS"
./vps.sh "grep -ih 'Error\|Traceback\|FAIL' logs/*.log 2>/dev/null | grep -v 'polls.*errors' | tail -3"

echo ""
echo "========================================="
