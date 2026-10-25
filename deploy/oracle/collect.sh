#!/usr/bin/env bash
# 평일 18:50 (한국 시간) — 토스 시세·수급 이력을 모은다. 조회(GET)만 하고 주문·잔고는 건드리지 않는다.
#   원본은 이 서버의 ~/.rich/toss/ 에만 두고, GitHub 에는 통계(data/toss/collect_status.json · lab.json)만 올린다.
#   처음(원본이 없을 때)은 2018년까지 되채우기 — 1시간 안팎. 중간에 끊겨도 다시 돌면 이어서 받는다.
#   수동 실행: FORCE=1 bash collect.sh   ·   처음부터 다시 되채우기: FORCE=1 BACKFILL=1 bash collect.sh
set -euo pipefail
DIR="$HOME/rich"; STATE="$HOME/.rich"
if [ "${FORCE:-0}" != 1 ]; then
  [ "$(TZ=Asia/Seoul date +%H)" = "18" ] && [ "$(TZ=Asia/Seoul date +%u)" -le 5 ] || exit 0
fi
mkdir -p "$STATE/logs"
exec 9>"$STATE/collect.lock"
flock -n 9 || { echo "$(date '+%F %T') 이미 수집 중 — 건너뜀"; exit 0; }
cd "$DIR"
git pull -q --ff-only || true
mode=""
[ -d "$STATE/toss/investor-trading" ] || mode="--backfill"
[ "${BACKFILL:-0}" = 1 ] && mode="--backfill"
echo "$(date '+%F %T') 토스 수집 시작 ${mode:-(매일)}"
.venv/bin/python deploy/oracle/toss_collect.py $mode
# 되채우기 직후와 매주 금요일에는 '이 자료가 주가를 미리 알려 주나' 검증
if [ -n "$mode" ] || [ "$(TZ=Asia/Seoul date +%u)" = "5" ]; then
  .venv/bin/python deploy/oracle/toss_lab.py || echo "검증 실패 (수집은 끝남)"
fi
git add data/toss/
git commit -q -m "toss collect $(date +%F) (oracle)" || { echo "올릴 변경 없음"; exit 0; }
git pull -q --rebase && git push -q && echo "$(date '+%F %T') 통계를 GitHub 에 올림"
find "$STATE/logs" -type f -mtime +60 -delete 2>/dev/null || true
