---
description: Deploy code changes to VPS and restart the sports engine
---

# Deploy to VPS

// turbo-all

1. Commit all changes:
```bash
cd /Users/MartinOile/Desktop/poly_high && git add -A && git commit -m "deploy" && git push origin main
```

2. Pull on VPS:
```bash
sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'cd /root/poly_high_sports && git fetch origin main && git reset --hard origin/main'
```

3. Kill existing engine:
```bash
sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'kill $(pgrep -f "python3.*sports.main") 2>/dev/null; sleep 2; echo killed'
```

4. Start new engine:
```bash
sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'cd /root/poly_high_sports && nohup /usr/bin/python3 -u -m sports.main > logs/sports_latest.log 2>&1 & echo started=$!'
```

5. Verify startup:
```bash
sshpass -p '12345vse' ssh -o StrictHostKeyChecking=no root@161.97.185.65 'sleep 8 && grep "STATUS" /root/poly_high_sports/logs/sports_latest.log | tail -1'
```
