#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Oracle Cloud 서버 준비 — 증권사 주문 '전송'만 이 서버가 맡는다 (판단은 GitHub Actions 그대로)
#
#   왜 서버가 필요한가: 토스증권 오픈API는 미리 등록한 IP(허용 IP)에서 온 요청만 받는다.
#   GitHub 실행기는 매번 IP가 바뀌므로, IP가 고정된 이 서버에서 주문을 보낸다.
#
#   사용:  bash setup.sh            준비 (여러 번 실행해도 안전) + 마지막에 점검 결과
#          bash setup.sh --check    점검만
#          bash setup.sh --keys     증권사 키 입력 (붙여넣기만, 화면에 안 보임 — 휴대폰에서 편집기 없이)
#          bash setup.sh --token    GitHub 쓰기 권한을 토큰으로 (휴대폰처럼 화면 글자를 복사하기 어려울 때)
#          bash setup.sh --probe    토스 연결 점검: 공식 명세 요약 + 토큰 발급 + 계좌 목록(조회만) → 결과를 GitHub 에 올림
#          bash setup.sh --collect  토스 시세·수급 이력 수집 시작 (조회만 · 처음 1시간 안팎 되채우기 · 평일 18:50 자동 등록)
#          bash setup.sh --cron     평일 08:40 주문 · 16:10 잔고 대사 자동 실행 등록 (모의 상태면 실행돼도 아무것도 안 함)
#          bash setup.sh --history  한국거래소 긴 역사(2000~2018 수급·상장폐지 포함) 받기 — 뒤에서 돌고 앱을 닫아도 계속
#          bash setup.sh --link     토스 계좌를 거울 계좌로 연결 (config short.broker: toss 로 바꾼 뒤 · 빈 계좌만)
#          bash setup.sh --live     실제 돈 주문 최종 확인값 저장 (준비도 점검표가 모두 통과한 뒤에만 의미 있음)
#          bash setup.sh --smoke    1주 실주문 시험 (평일 정규장 · 실제 돈 1주 값 + 수수료 — 묻고 나서만)
#          bash setup.sh --candles [일수]  토스 분봉 받기 (기본 90일)(국내 101·미국 104종목, 옛날까지) + 불장단타왕 규칙 검증 — 뒤에서 돌고 앱을 닫아도 계속
#          bash setup.sh --sweep    받아 둔 분봉으로 손절·익절·나눠 파는 비율·매수 자리 격자 탐색 (국내·미국, 수십 분~몇 시간) — 뒤에서 돌고 앱을 닫아도 계속
#
#   이 스크립트는 비밀 값을 화면에 출력하지 않는다. 키는 ~/.rich/secrets.env 에만 저장한다.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
REPO="sunbmj4119-cmd/rich"
DIR="$HOME/rich"
STATE="$HOME/.rich"
KEY="$HOME/.ssh/rich_deploy"
MODE="${1:-setup}"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
no()   { printf "  \033[33m✗\033[0m %s\n" "$*"; }
step() { printf "\n\033[1m▶ %s\033[0m\n" "$*"; }
# GitHub 쓰기 권한 확인. 토큰 방식이면 '없는 가지에 올려 보기'를 흉내만 내고(dry-run), 배포 키 방식이면 ssh 인사 문구로.
# (GitHub ssh 는 인증에 성공해도 종료 코드 1 을 돌려준다 → 출력 문구로 판정)
# 토큰 방식: GitHub 에 '이 토큰으로 이 저장소에 쓸 수 있나'(permissions.push)를 직접 묻는다. 토큰은 명령줄에 싣지 않는다.
token_ok() {
  tok=$(sed -n 's#^https://x-access-token:\(.*\)@github.com$#\1#p' "$STATE/git-credentials" 2>/dev/null | head -1)
  [ -n "$tok" ] || return 1
  hf=$(mktemp); chmod 600 "$hf"; printf 'Authorization: Bearer %s\n' "$tok" > "$hf"
  res=$(curl -s -m 10 -H @"$hf" -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$REPO" || true)
  rm -f "$hf"
  printf '%s' "$res" | push_allowed
}
push_allowed() { python3 -c 'import sys,json
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
sys.exit(0 if (d.get("permissions") or {}).get("push") else 1)'; }
gh_ok() {
  if [ -f "$STATE/git-credentials" ]; then
    token_ok
  else
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 -T github-rich 2>&1 || true); echo "$out" | grep -q "successfully authenticated"
  fi
}
# 값이 들어 있는 항목 이름만 (값은 절대 출력하지 않음)
filled() { { grep -E '^[A-Z_]+=.+' "$STATE/secrets.env" 2>/dev/null || true; } | cut -d= -f1 | tr '\n' ' '; }

pick_python() {
  for p in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      echo "$p"; return 0
    fi
  done
  return 1
}

install_packages() {
  step "1/6 기본 프로그램 (git · Python)"
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update -qq
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3 python3-venv python3-pip curl openssh-client cron >/dev/null
  elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y -q git curl openssh-clients cronie >/dev/null
    pick_python >/dev/null || $SUDO dnf install -y -q python3.11 python3.11-pip >/dev/null || true
  else
    no "apt / dnf 를 찾지 못했습니다 — git 과 Python 3.9 이상을 직접 설치해 주세요"
  fi
  PY="$(pick_python)" || { no "Python 3.9 이상이 없습니다"; exit 1; }
  ok "$("$PY" --version) · git $(git --version | awk '{print $3}')"
}

tune_system() {
  step "2/6 메모리 (서버의 시간대·기존 프로그램은 건드리지 않음)"
  ok "서버 시간대 $(timedatectl show -p Timezone --value 2>/dev/null || date +%Z) 그대로 — 실행 시각은 run.sh 가 한국 시간으로 판단"
  mem_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
  swap_mb=$(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo)
  if [ "$mem_mb" -lt 2048 ] && [ "$swap_mb" -lt 1024 ]; then
    $SUDO fallocate -l 2G /swapfile 2>/dev/null || $SUDO dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
    $SUDO chmod 600 /swapfile && $SUDO mkswap /swapfile >/dev/null && $SUDO swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | $SUDO tee -a /etc/fstab >/dev/null
    ok "메모리 ${mem_mb}MB → 스왑 2GB 추가 (pandas 설치·실행용)"
  else ok "메모리 ${mem_mb}MB · 스왑 ${swap_mb}MB"; fi
}

get_code() {
  step "3/6 코드 받기"
  if [ -d "$DIR/.git" ]; then
    git -C "$DIR" pull -q --ff-only && ok "$DIR 최신으로 ($(git -C "$DIR" log -1 --format=%h))"
  else
    git clone -q "https://github.com/$REPO.git" "$DIR" && ok "$DIR 에 받음"
  fi
  PY="$(pick_python)"
  [ -d "$DIR/.venv" ] || "$PY" -m venv "$DIR/.venv"
  "$DIR/.venv/bin/pip" install -q --upgrade pip >/dev/null
  "$DIR/.venv/bin/pip" install -q pandas numpy pyyaml requests >/dev/null
  ok "파이썬 환경 $DIR/.venv (pandas · numpy · pyyaml · requests)"
}

deploy_key() {
  step "4/6 GitHub 쓰기 권한 (배포 키)"
  if [ -f "$STATE/git-credentials" ]; then
    gh_ok && ok "토큰으로 GitHub 쓰기 권한 확인됨" || no "토큰이 있지만 쓰기 실패 — bash ~/rich/deploy/oracle/setup.sh --token 으로 다시 입력"
    return
  fi
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C "rich-oracle-$(hostname)" -f "$KEY"
  if ! grep -q "Host github-rich" "$HOME/.ssh/config" 2>/dev/null; then
    printf "Host github-rich\n  HostName github.com\n  User git\n  IdentityFile %s\n  IdentitiesOnly yes\n" "$KEY" >> "$HOME/.ssh/config"
    chmod 600 "$HOME/.ssh/config"
  fi
  ssh-keygen -F github.com >/dev/null 2>&1 || ssh-keyscan -t ed25519 github.com 2>/dev/null >> "$HOME/.ssh/known_hosts"
  git -C "$DIR" remote set-url --push origin "git@github-rich:$REPO.git"
  git -C "$DIR" config user.name "rich-oracle"
  git -C "$DIR" config user.email "rich-oracle@users.noreply.github.com"
  if gh_ok; then
    ok "GitHub 쓰기 권한 확인됨"
  else
    no "아직 GitHub 에 등록되지 않았습니다. 아래 '공개 키' 한 줄을 등록하세요:"
    echo
    echo "    GitHub → $REPO → Settings → Deploy keys → Add deploy key"
    echo "    Title: oracle   ·   Key: 아래 한 줄 전체   ·   ✅ Allow write access   → Add key"
    echo
    echo "    $(cat "$KEY.pub")"
    echo
    echo "    (공개 키라 보여도 안전합니다. 개인 키 $KEY 는 이 서버 밖으로 내보내지 마세요.)"
    echo "    휴대폰이라 화면 글자를 복사하기 어려우면 대신: bash ~/rich/deploy/oracle/setup.sh --token"
  fi
}

secrets_file() {
  step "5/6 비밀 키 파일"
  mkdir -p "$STATE/logs" && chmod 700 "$STATE"
  if [ ! -f "$STATE/secrets.env" ]; then
    cat > "$STATE/secrets.env" <<'TPL'
# 증권사 키 — 이 파일에만 적습니다. 채팅·GitHub 에 붙여 넣지 마세요.
# 편집: nano ~/.rich/secrets.env   (저장 Ctrl+O → Enter, 나가기 Ctrl+X)
# 값은 = 뒤에 따옴표 없이 붙여 씁니다.
TOSS_CLIENT_ID=
TOSS_CLIENT_SECRET=
TOSS_ACCOUNT=
# 실계좌 주문을 허용하려면 마지막에 I_UNDERSTAND 로 (config 의 live_trading: true 도 필요)
AUTOTRADE_CONFIRM=
TPL
  fi
  chmod 600 "$STATE/secrets.env"
  set_names=$(filled)
  ok "$STATE/secrets.env (권한 600) · 값이 들어 있는 항목: ${set_names:-없음}"
}

check() {
  step "6/6 점검 결과 — 이 표를 그대로 보내 주시면 됩니다 (비밀 값 없음)"
  . /etc/os-release 2>/dev/null || true
  out_ip=$(curl -s -m 8 https://ifconfig.me || curl -s -m 8 https://api.ipify.org || echo "확인 실패")
  py=$([ -x "$DIR/.venv/bin/python" ] && "$DIR/.venv/bin/python" --version 2>&1 || echo "없음")
  gh=$(gh_ok && echo "가능" || echo "아직 (배포 키 등록 필요)")
  names=$(filled)
  cronl=$({ crontab -l 2>/dev/null || true; } | grep -c "rich/deploy/oracle/run.sh" || true)
  crone=$({ crontab -l 2>/dev/null || true; } | grep -c "rich/deploy/oracle/evening.sh" || true)
  cronc=$({ crontab -l 2>/dev/null || true; } | grep -c "rich/deploy/oracle/collect.sh" || true)
  rawn=$(find "$STATE/toss" -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')
  echo "  ┌──────────────────────────────────────────"
  echo "  │ 운영체제      : ${PRETTY_NAME:-?} ($(uname -m))"
  echo "  │ CPU · 메모리   : $(nproc)코어 · $(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)MB · 스왑 $(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo)MB"
  echo "  │ 시간대        : $(timedatectl show -p Timezone --value 2>/dev/null || date +%Z)"
  echo "  │ 파이썬        : $py"
  echo "  │ 코드          : $(git -C "$DIR" log -1 --format='%h %cd' --date=short 2>/dev/null || echo 없음)"
  echo "  │ GitHub 쓰기   : $gh"
  echo "  │ 키 파일 항목  : ${names:-비어 있음}"
  echo "  │ 자동 실행     : $([ "$cronl" -gt 0 ] && echo '주문 08:40 등록' || echo '주문 아직') · $([ "$crone" -gt 0 ] && echo '대사 16:10 등록' || echo '대사 아직')"
  echo "  │ 긴 역사 자료  : $(ls "$DIR"/data/history/krx_*.csv.gz 2>/dev/null | wc -l | tr -d ' ')개 파일 · 받아 둔 조각 $(ls "$DIR"/data/history/krx_cache 2>/dev/null | wc -l | tr -d ' ')개 $(pgrep -f collect_krx_history.py >/dev/null 2>&1 && echo '(받는 중)' || true)"
  [ -f "$STATE/logs/krx_history.log" ] && echo "  │   마지막 줄   : $(grep -v -e 'Warning' -e '^ *df = ' "$STATE/logs/krx_history.log" | tail -n 1 | cut -c1-90)"
  echo "  │ 토스 수집     : $([ "$cronc" -gt 0 ] && echo '등록됨 (평일 18:50)' || echo '아직 안 함') · 원본 파일 ${rawn}개"
  echo "  │ 바깥으로 나가는 IP : $out_ip   ← 토스 '허용 IP'에 이 값을 등록 (보낼 때는 가려도 됩니다)"
  echo "  └──────────────────────────────────────────"
}

set_keys() {
  step "증권사 키 입력 — 붙여넣고 Enter (입력한 글자는 화면에 보이지 않습니다. 건너뛰려면 그냥 Enter)"
  [ -f "$STATE/secrets.env" ] || secrets_file
  for k in TOSS_CLIENT_ID TOSS_CLIENT_SECRET TOSS_ACCOUNT KRX_ID KRX_PW; do
    printf "  %s: " "$k"
    IFS= read -rs v || v=""
    echo
    v="$(printf '%s' "$v" | tr -d '\r' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [ -z "$v" ] && { echo "    (건너뜀)"; continue; }
    tmp="$(mktemp)"
    { grep -v "^$k=" "$STATE/secrets.env" || true; } > "$tmp"
    printf '%s=%s\n' "$k" "$v" >> "$tmp"
    mv "$tmp" "$STATE/secrets.env" && chmod 600 "$STATE/secrets.env"
    ok "$k 저장 (${#v}글자)"
  done
}

set_token() {
  step "GitHub 토큰 입력 — 붙여넣고 Enter (화면에 보이지 않습니다)"
  echo "  토큰 만들기: GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens"
  echo "  → Only select repositories: $REPO · Repository permissions → Contents: Read and write → Generate → Copy"
  printf "  토큰: "
  IFS= read -rs tok || tok=""
  echo
  tok="$(printf '%s' "$tok" | tr -d '\r\n\t ')"
  [ -z "$tok" ] && { no "입력이 없어 건너뜀"; return 0; }
  mkdir -p "$STATE" && chmod 700 "$STATE"
  ( umask 077; printf 'https://x-access-token:%s@github.com\n' "$tok" > "$STATE/git-credentials" )
  git -C "$DIR" config credential.helper "store --file $STATE/git-credentials"
  git -C "$DIR" remote set-url --push origin "https://github.com/$REPO.git"
  git -C "$DIR" config user.name "rich-oracle"
  git -C "$DIR" config user.email "rich-oracle@users.noreply.github.com"
  if gh_ok; then ok "GitHub 쓰기 권한 확인됨 (토큰 ${#tok}글자 · $STATE/git-credentials 권한 600)"
  else no "쓰기 실패 — 저장소 선택($REPO)과 Contents: Read and write 권한을 확인한 뒤 다시 --token"; fi
}

run_probe() {
  step "토스 연결 점검 — 조회만 하고 주문·잔고는 건드리지 않습니다"
  git -C "$DIR" pull -q --ff-only || true
  "$DIR/.venv/bin/python" "$DIR/deploy/oracle/toss_probe.py" || true
  git -C "$DIR" add data/toss/ 2>/dev/null || true
  if git -C "$DIR" diff --cached --quiet; then no "올릴 결과 없음"
  else
    git -C "$DIR" commit -q -m "toss probe $(date +%F)" && git -C "$DIR" pull -q --rebase && git -C "$DIR" push -q \
      && ok "결과를 GitHub 에 올렸습니다 — 이제 채팅으로 '점검 끝'이라고만 알려 주세요" || no "GitHub 에 올리기 실패 (--check 로 쓰기 권한 확인)"
  fi
}

install_collect() {
  step "토스 시세·수급 이력 수집 — 조회만 (주문·잔고 API 는 쓰지 않음)"
  git -C "$DIR" pull -q --ff-only || true
  mkdir -p "$STATE/logs"
  line="50 * * * * bash $DIR/deploy/oracle/collect.sh >> $STATE/logs/collect.log 2>&1"
  ( { crontab -l 2>/dev/null || true; } | { grep -v "rich/deploy/oracle/collect.sh" || true; } ; echo "$line" ) | crontab -
  ok "매일 수집 등록 (한국 시간 평일 18:50): $line"
  if pgrep -f "deploy/oracle/toss_collect.py" >/dev/null 2>&1; then
    ok "이미 되채우는 중입니다 — 진행 보기: tail -f $STATE/logs/collect_backfill.log"
  else
    nohup env FORCE=1 bash "$DIR/deploy/oracle/collect.sh" > "$STATE/logs/collect_backfill.log" 2>&1 &
    ok "지금 2018년까지 되채우기를 뒤에서 시작했습니다 (1시간 안팎 · 앱을 닫아도 계속됩니다)"
    echo "  진행 보기: tail -f $STATE/logs/collect_backfill.log   (그만 보기: Ctrl+C — 수집은 계속됨)"
    echo "  끝나면 통계가 GitHub 에 저절로 올라갑니다. 채팅으로 '수집 끝'이라고만 알려 주세요."
  fi
}

install_cron() {
  # 서버 시간대와 상관없이: 매시 40분·10분에 깨어나고, 스크립트가 '한국 시간 평일 08시 / 16시'일 때만 실제로 일한다
  mkdir -p "$STATE/logs"
  line="40 * * * * bash $DIR/deploy/oracle/run.sh >> $STATE/logs/cron.log 2>&1"
  line2="10 * * * * bash $DIR/deploy/oracle/evening.sh >> $STATE/logs/evening.log 2>&1"
  ( { crontab -l 2>/dev/null || true; } | { grep -v -e "rich/deploy/oracle/run.sh" -e "rich/deploy/oracle/evening.sh" || true; } ; echo "$line"; echo "$line2" ) | crontab -
  ok "자동 실행 등록: 평일 08:40 주문 · 16:10 잔고 대사 (모의 상태면 아무것도 안 함)"
}

run_history() {
  step "한국거래소 긴 역사 받기 — 뒤에서 돌고, 앱을 닫거나 접속이 끊겨도 계속됩니다 (2~3시간)"
  git -C "$DIR" pull -q --ff-only || true
  grep -q '^KRX_ID=.\+' "$STATE/secrets.env" 2>/dev/null || { no "KRX_ID 가 없습니다 — 먼저 bash ~/rich/deploy/oracle/setup.sh --keys"; return 0; }
  mkdir -p "$STATE/logs"
  if pgrep -f "collect_krx_history.py|collect_history.sh" >/dev/null 2>&1; then
    pkill -f "collect_history.sh" 2>/dev/null || true
    pkill -f "collect_krx_history.py" 2>/dev/null || true
    sleep 2
    ok "받던 것을 멈추고 최신 코드로 다시 시작합니다 (받아 둔 것은 이어서 씀)"
  fi
  nohup bash "$DIR/deploy/oracle/collect_history.sh" > "$STATE/logs/krx_history.log" 2>&1 &
  sleep 20
  echo "  ── 처음 몇 줄 ──"; tail -n 6 "$STATE/logs/krx_history.log" | sed 's/^/  /'
  ok "뒤에서 받는 중입니다. 진행 보기: tail -n 5 $STATE/logs/krx_history.log   (몇 번이든 다시 봐도 됨)"
  echo "  '0) KRX 접속 확인 ✗' 가 보이면 KRX 아이디·비밀번호 문제입니다 — 화면을 캡처해 보내 주세요."
  echo "  끝나면 저절로 GitHub 에 올라갑니다. 채팅으로 '수집 끝'이라고만 알려 주세요."
}

run_candles() {
  step "토스 분봉 받기 + 불장단타왕 규칙 검증 — 뒤에서 돌고, 앱을 닫거나 접속이 끊겨도 계속됩니다 (몇 시간)"
  git -C "$DIR" pull -q --ff-only || true
  grep -q '"intervals"' "$DIR/data/toss/probe.json" 2>/dev/null || { no "캔들 점검 전입니다 — 먼저 bash ~/rich/deploy/oracle/setup.sh --probe"; return 0; }
  mkdir -p "$STATE/logs"
  if pgrep -f "candles.sh|toss_candles.py" >/dev/null 2>&1; then
    ok "이미 받는 중입니다 — 진행 보기: tail -n 5 $STATE/logs/candles.log"
    return 0
  fi
  CANDLE_DAYS="${2:-90}" nohup bash "$DIR/deploy/oracle/candles.sh" > "$STATE/logs/candles.log" 2>&1 &
  echo "  받는 기간: 최근 ${2:-90}일 (이미 받은 것은 건너뛰고 더 옛날만)"
  sleep 20
  echo "  ── 처음 몇 줄 ──"; tail -n 6 "$STATE/logs/candles.log" | sed 's/^/  /'
  ok "뒤에서 받는 중입니다. 진행 보기: tail -n 5 $STATE/logs/candles.log"
}

run_sweep() {
  step "분봉 규칙 격자 탐색 (손절·익절·나눠 파는 비율·매수 자리) — 뒤에서 돌고, 앱을 닫거나 접속이 끊겨도 계속됩니다"
  git -C "$DIR" pull -q --ff-only || true
  [ -f "$DIR/data/toss/candles_status.json" ] || { no "받아 둔 분봉이 없습니다 — 먼저 bash ~/rich/deploy/oracle/setup.sh --candles 365"; return 0; }
  mkdir -p "$STATE/logs"
  if pgrep -f "sweep.sh|intraday_sweep.py" >/dev/null 2>&1; then
    ok "이미 탐색 중입니다 — 진행 보기: tail -n 5 $STATE/logs/sweep.log"
    return 0
  fi
  if pgrep -f "candles.sh|toss_candles.py" >/dev/null 2>&1; then
    no "분봉을 아직 받는 중입니다 — 끝난 뒤(GitHub 에 올림이 보인 뒤) 다시 실행하세요"
    return 0
  fi
  nohup bash "$DIR/deploy/oracle/sweep.sh" > "$STATE/logs/sweep.log" 2>&1 &
  sleep 30
  echo "  ── 처음 몇 줄 ──"; tail -n 6 "$STATE/logs/sweep.log" | sed 's/^/  /'
  ok "뒤에서 탐색 중입니다. 진행 보기: tail -n 5 $STATE/logs/sweep.log   (처음엔 1분봉을 5분봉으로 묶느라 몇 분 조용합니다)"
  echo "  끝나면 저절로 GitHub 에 올라갑니다. 채팅으로 '탐색 끝'이라고만 알려 주세요."
}

run_link() {
  step "토스 계좌를 거울 계좌로 연결 — 이미 있던 종목은 '직접 보유분'으로 기록하고 자동매매는 건드리지 않습니다"
  echo "  실제 돈 주문을 켜기(--live) 전까지는 다시 연결해도 안전합니다 (자동매매 금액 short.broker_capital 을 바꿨을 때 등)"
  git -C "$DIR" pull -q --ff-only || true
  set -a; . "$STATE/secrets.env"; set +a
  ( cd "$DIR" && "$DIR/.venv/bin/python" src/short_exec.py --phase link ) || true
  git -C "$DIR" add data/short/mirror_status.json 2>/dev/null || true   # 공개: 연결 여부만 (자본·보유는 ~/.rich/mirror.json 에만)
  git -C "$DIR" diff --cached --quiet || { git -C "$DIR" commit -q -m "mirror link $(date +%F) (oracle)" && git -C "$DIR" pull -q --rebase && git -C "$DIR" push -q && ok "연결 정보를 GitHub 에 올렸습니다"; }
}

set_live() {
  step "실제 돈 주문 최종 확인 — 준비도 점검표(모의투자 화면 '위원회' 탭)가 모두 통과한 뒤에만 하세요"
  echo "  이 값이 있어도 config live_trading:true · 점검표 '준비됨' 이 모두 맞아야 주문이 나갑니다."
  printf "  정말 실제 돈으로 주문을 보내려면 I_UNDERSTAND 를 입력 (취소: 그냥 Enter): "
  IFS= read -r v || v=""
  if [ "$v" = "I_UNDERSTAND" ]; then
    tmp="$(mktemp)"; { grep -v "^AUTOTRADE_CONFIRM=" "$STATE/secrets.env" || true; } > "$tmp"
    echo "AUTOTRADE_CONFIRM=I_UNDERSTAND" >> "$tmp"; mv "$tmp" "$STATE/secrets.env"; chmod 600 "$STATE/secrets.env"
    ok "저장했습니다 (되돌리기: bash ~/rich/deploy/oracle/setup.sh --unlive)"
  else
    no "취소 — 바뀐 것 없음"
  fi
}

run_smoke() {
  step "1주 실주문 시험 — KODEX 미국달러선물 1주를 사고(지정가) 체결되면 바로 팝니다 (평일 09:05~15:10)"
  echo "  실제 돈이 움직입니다: 1주 값(약 1만 4천원)으로 사고 팔며, 수수료·가격 차이로 수십~수백 원이 듭니다."
  printf "  진행하려면 yes 입력 (취소: 그냥 Enter): "
  IFS= read -r v || v=""
  [ "$v" = "yes" ] || { no "취소 — 바뀐 것 없음"; return 0; }
  git -C "$DIR" pull -q --ff-only || true
  set -a; . "$STATE/secrets.env"; set +a
  ( cd "$DIR" && "$DIR/.venv/bin/python" src/short_exec.py --phase smoke --yes ) || true
  git -C "$DIR" add data/toss/smoke.json 2>/dev/null || true
  git -C "$DIR" diff --cached --quiet || { git -C "$DIR" commit -q -m "toss smoke $(date +%F) (oracle)" && git -C "$DIR" pull -q --rebase && git -C "$DIR" push -q && ok "결과를 GitHub 에 올렸습니다"; }
}

unset_live() {
  tmp="$(mktemp)"; { grep -v "^AUTOTRADE_CONFIRM=" "$STATE/secrets.env" || true; } > "$tmp"; mv "$tmp" "$STATE/secrets.env"; chmod 600 "$STATE/secrets.env"
  ok "실제 돈 주문 확인값을 지웠습니다 — 이제 주문은 만들기만 하고 보내지 않습니다"
}

case "$MODE" in
  --check) check ;;
  --keys)  set_keys; check ;;
  --token) set_token; check ;;
  --probe) run_probe ;;
  --collect) install_collect ;;
  --cron)  install_cron; check ;;
  --history) run_history ;;
  --link)  run_link ;;
  --live)  set_live ;;
  --unlive) unset_live ;;
  --smoke) run_smoke ;;
  --candles) run_candles "$@" ;;
  --sweep) run_sweep ;;
  --*)     echo "모르는 옵션: $MODE — 철자를 확인하세요 (예: --candles · --sweep · --probe · --check · --link · --history)"; exit 1 ;;
  *)       install_packages; tune_system; get_code; deploy_key; secrets_file; check
           echo; echo "다음 단계: 위 ✗ 항목을 해결한 뒤 'bash ~/rich/deploy/oracle/setup.sh --check' 로 다시 확인하세요." ;;
esac
