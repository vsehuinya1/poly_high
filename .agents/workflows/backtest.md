---
description: Run replay backtest on VPS to test strategy parameter changes
---

# Run Backtest

// turbo-all

1. Push latest code to VPS:
```bash
cd /Users/MartinOile/Desktop/poly_high && git add -A && git commit -m "backtest update" && git push origin main
sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'cd /root/poly_high_sports && git fetch origin main && git reset --hard origin/main'
```

2. Run replay backtest with default params:
```bash
sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'cd /root/poly_high_sports && python3 replay_backtest.py --data-dir sports_data'
```

3. Run with custom params (example):
```bash
sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'cd /root/poly_high_sports && python3 replay_backtest.py --data-dir sports_data --min-hold 120 --edge 0.15 --price-floor 0.15'
```

4. Run daily report:
```bash
sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'cd /root/poly_high_sports && python3 daily_report.py'
```
