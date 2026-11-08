#!/usr/bin/env bash
# 토스 분봉 받기(옛날까지) → 불장단타왕 규칙 엔진 검증(국내·미국) → 결과만 GitHub 에 올림. 원본 분봉은 서버 ~/.rich/candles/ 에만.
#   bash ~/rich/deploy/oracle/candles.sh            (setup.sh --candles [일수] 가 뒤에서 돌림 — 기본 90일, 이미 받은 것은 이어서)
set -euo pipefail
DIR="$HOME/rich"; STATE="$HOME/.rich"
cd "$DIR"
git pull -q --ff-only || true
set -a; [ -f "$STATE/secrets.env" ] && . "$STATE/secrets.env"; set +a
.venv/bin/python -u deploy/oracle/toss_candles.py --backfill --days "${CANDLE_DAYS:-90}"
IV=$(.venv/bin/python -c "import json; print(json.load(open('data/toss/candles_status.json'))['interval'])")
.venv/bin/python -u src/intraday_lab.py --venue kr --dir "$STATE/candles/$IV/kr" --bench 069500 || echo "국내 검증 실패(계속)"
.venv/bin/python -u src/intraday_lab.py --venue us --dir "$STATE/candles/$IV/us" --bench SPY || echo "미국 검증 실패(계속)"
git add data/toss/candles_status.json data/intraday_lab.json 2>/dev/null || true
git commit -q -m "분봉 수집·검증 $(date +%F) (oracle)" || { echo "변경 없음"; exit 0; }
git pull -q --rebase && git push -q && echo "GitHub 에 올림 — 채팅으로 '분봉 끝'이라고 알려 주세요"
