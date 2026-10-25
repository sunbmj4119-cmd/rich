#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Oracle Cloud 서버 준비 — 증권사 주문 '전송'만 이 서버가 맡는다 (판단은 GitHub Actions 그대로)
#
#   왜 서버가 필요한가: 토스증권 오픈API는 미리 등록한 IP(허용 IP)에서 온 요청만 받는다.
#   GitHub 실행기는 매번 IP가 바뀌므로, IP가 고정된 이 서버에서 주문을 보낸다.
#
#   사용:  bash setup.sh            준비 (여러 번 실행해도 안전) + 마지막에 점검 결과
#          bash setup.sh --check    점검만
#          bash setup.sh --cron     평일 08:40 자동 실행 등록 (증권사 연결이 끝난 뒤에만)
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
# GitHub 은 인증에 성공해도 종료 코드 1 을 돌려준다 → 출력 문구로 판정
gh_ok() { out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 -T github-rich 2>&1 || true); echo "$out" | grep -q "successfully authenticated"; }
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
  echo "  ┌──────────────────────────────────────────"
  echo "  │ 운영체제      : ${PRETTY_NAME:-?} ($(uname -m))"
  echo "  │ CPU · 메모리   : $(nproc)코어 · $(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)MB · 스왑 $(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo)MB"
  echo "  │ 시간대        : $(timedatectl show -p Timezone --value 2>/dev/null || date +%Z)"
  echo "  │ 파이썬        : $py"
  echo "  │ 코드          : $(git -C "$DIR" log -1 --format='%h %cd' --date=short 2>/dev/null || echo 없음)"
  echo "  │ GitHub 쓰기   : $gh"
  echo "  │ 키 파일 항목  : ${names:-비어 있음}"
  echo "  │ 자동 실행     : $([ "$cronl" -gt 0 ] && echo '등록됨 (평일 08:40)' || echo '아직 안 함')"
  echo "  │ 바깥으로 나가는 IP : $out_ip   ← 토스 '허용 IP'에 이 값을 등록 (보낼 때는 가려도 됩니다)"
  echo "  └──────────────────────────────────────────"
}

install_cron() {
  # 서버 시간대와 상관없이: 매시 40분에 깨어나고, run.sh 가 '한국 시간 평일 08시'일 때만 실제로 일한다
  line="40 * * * * bash $DIR/deploy/oracle/run.sh >> $STATE/logs/cron.log 2>&1"
  ( { crontab -l 2>/dev/null || true; } | { grep -v "rich/deploy/oracle/run.sh" || true; } ; echo "$line" ) | crontab -
  ok "자동 실행 등록 (한국 시간 평일 08:40에만 주문): $line"
}

case "$MODE" in
  --check) check ;;
  --cron)  install_cron; check ;;
  *)       install_packages; tune_system; get_code; deploy_key; secrets_file; check
           echo; echo "다음 단계: 위 ✗ 항목을 해결한 뒤 'bash ~/rich/deploy/oracle/setup.sh --check' 로 다시 확인하세요." ;;
esac
