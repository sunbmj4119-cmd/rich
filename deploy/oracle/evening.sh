#!/usr/bin/env bash
# 평일 16:10 (장 마감 뒤) — 토스 잔고와 장부를 맞춰 본다(대사). 토스는 등록된 IP 에서만 되므로 이 서버가 한다.
# 결과(data/short/recon.json — 수량 차이·상태만, 금액·종목 수량 원본은 올리지 않음)를 GitHub 에 올리면
# 그날 저녁 GitHub 계산이 읽어서 어긋난 종목은 새 매수를 막는다.  수동 실행: FORCE=1 bash evening.sh
set -euo pipefail
DIR="$HOME/rich"; STATE="$HOME/.rich"
if [ "${FORCE:-0}" != 1 ]; then
  [ "$(TZ=Asia/Seoul date +%H)" = "16" ] && [ "$(TZ=Asia/Seoul date +%u)" -le 5 ] || exit 0
fi
cd "$DIR"
git pull -q --ff-only
sbroker=$(.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import settings; print((settings.load().get('short') or {}).get('broker','paper'))")
[ "$sbroker" = "paper" ] && { echo "$(date '+%F %T') short.broker=paper — 대사할 증권사 없음"; exit 0; }
set -a; . "$STATE/secrets.env"; set +a
.venv/bin/python src/short_exec.py --phase reconcile --after-plan
git add data/short/recon.json 2>/dev/null || true
git commit -q -m "reconcile $(date +%F) (oracle)" || { echo "변경 없음"; exit 0; }
git pull -q --rebase && git push -q && echo "대사 결과를 GitHub 에 올림"
