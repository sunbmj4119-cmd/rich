#!/usr/bin/env bash
# 분봉 규칙 격자 탐색 (손절·익절·나눠 파는 비율·매수 자리) — 국내·미국. 원본 분봉은 서버 ~/.rich/candles/ 에만, 결과 요약만 GitHub 에.
#   setup.sh --sweep 가 뒤에서 돌림 (이미 받은 분봉을 씀 — 새로 받지 않음)
set -euo pipefail
DIR="$HOME/rich"; STATE="$HOME/.rich"
cd "$DIR"
git pull -q --ff-only || true
IV=$(.venv/bin/python -c "import json; print(json.load(open('data/toss/candles_status.json'))['interval'])")
.venv/bin/python -u src/intraday_sweep.py --venue kr --dir "$STATE/candles/$IV/kr" --bench 069500 || echo "국내 탐색 실패(계속)"
.venv/bin/python -u src/intraday_sweep.py --venue us --dir "$STATE/candles/$IV/us" --bench SPY || echo "미국 탐색 실패(계속)"
git add data/intraday_sweep.json 2>/dev/null || true
git commit -q -m "분봉 격자 탐색 $(date +%F) (oracle)" || { echo "변경 없음"; exit 0; }
git pull -q --rebase && git push -q && echo "GitHub 에 올림 — 채팅으로 '탐색 끝'이라고 알려 주세요"
