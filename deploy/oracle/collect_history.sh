#!/usr/bin/env bash
# KRX 긴 역사(2000~2018 수급·상장폐지 포함 대형주·ETF 전 기간)를 한 번 받아 GitHub 에 올린다.
# GitHub 서버는 KRX 가 IP 를 막아 못 받으므로 이 서버(또는 집 PC)에서 돌린다. 끊겨도 다시 실행하면 이어 받는다.
#   bash ~/rich/deploy/oracle/collect_history.sh
set -euo pipefail
DIR="$HOME/rich"; STATE="$HOME/.rich"
cd "$DIR"
git pull -q --ff-only
set -a; [ -f "$STATE/secrets.env" ] && . "$STATE/secrets.env"; set +a
[ -n "${KRX_ID:-}" ] || echo "주의: KRX_ID 가 secrets.env 에 없습니다 — KRX 로그인이 필요한 자료는 비어 있을 수 있습니다"
.venv/bin/pip install -q pykrx >/dev/null 2>&1 || true
mkdir -p "$STATE/logs"
.venv/bin/python src/collect_krx_history.py "$@" 2>&1 | tee "$STATE/logs/krx_history_$(date +%F).log"
git add data/history/krx_universe.csv.gz data/history/krx_ohlcv.csv.gz data/history/krx_flows.csv.gz data/history/krx_etf.csv.gz 2>/dev/null || true
git commit -q -m "KRX 긴 역사 $(date +%F) (oracle)" || { echo "변경 없음"; exit 0; }
git pull -q --rebase && git push -q && echo "GitHub 에 올림 — 다음 밤 evolve.yml 의 긴 역사 검증이 수급 모델까지 잽니다"
