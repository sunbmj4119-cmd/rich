#!/usr/bin/env bash
# 평일 08:40 — GitHub 에서 어제 만든 주문 계획(30일 엔진 data/autotrade/plan.json · 거울 계좌 data/short/plan.json)을 받아
# 증권사로 보내고, 결과를 GitHub 에 올린다.
# 판단(매수/매도 결정)은 GitHub Actions(daily.yml 17:00)가 하고, 이 서버는 '고정 IP에서 전송'만 한다.
set -euo pipefail
DIR="$HOME/rich"; STATE="$HOME/.rich"
# 서버 시간대가 무엇이든 한국 시간 평일 08시에만 일한다 (cron 은 매시 40분에 깨움). 수동 실행: FORCE=1 bash run.sh
if [ "${FORCE:-0}" != 1 ]; then
  [ "$(TZ=Asia/Seoul date +%H)" = "08" ] && [ "$(TZ=Asia/Seoul date +%u)" -le 5 ] || exit 0
fi
cd "$DIR"
git pull -q --ff-only
broker=$(.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import settings; print(settings.load()['broker'])")
sbroker=$(.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import settings; print((settings.load().get('short') or {}).get('broker','paper'))")
if [ "$broker" = "paper" ] && [ "$sbroker" = "paper" ]; then
  echo "$(date '+%F %T') broker=paper · short.broker=paper — 모의 계좌는 GitHub 에서 처리하므로 이 서버는 할 일이 없습니다"; exit 0
fi
set -a; . "$STATE/secrets.env"; set +a
echo "$(date '+%F %T') 주문 전송 시작 · 30일 엔진 $broker · 단기·전천후 거울 $sbroker · 나가는 IP $(curl -s -m 5 https://ifconfig.me || echo '?')"
[ "$broker" != "paper" ] && .venv/bin/python src/autotrader.py --phase execute
# 거울 계좌(전천후 몫) — GitHub 예약 실행은 몇 시간씩 늦어 09:30 넘으면 매수를 못 보낸다 → 시간이 정확한 이 서버에서
[ "$sbroker" != "paper" ] && .venv/bin/python src/short_exec.py --phase execute --where oracle
git add data/autotrade/ docs/autotrade.json data/short/orders.csv 2>/dev/null || git add data/autotrade/
git commit -q -m "execute $(date +%F) (oracle)" || { echo "변경 없음"; exit 0; }
git pull -q --rebase && git push -q && echo "결과를 GitHub 에 올림"
find "$STATE/logs" -type f -mtime +60 -delete 2>/dev/null || true
