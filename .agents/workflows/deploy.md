---
description: Deploy code changes to VPS and restart the sports engine
---

# Deploy to VPS

Run this in your terminal (not through the scaffold):

```bash
cd /Users/MartinOile/Desktop/poly_high && ./deploy.sh "your commit message"
```

This script handles the full pipeline:
1. `git add -A && git commit && git push`
2. `git fetch && git reset --hard` on VPS
3. Kill old engines
4. Start new engine with timestamped log
5. Wait 15s and verify startup

If you need to run a single command on the VPS:
```bash
./vps.sh "grep STATUS logs/latest.log | tail -1"
```

For a full status check:
```bash
./status.sh
```
