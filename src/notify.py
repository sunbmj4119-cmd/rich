"""
알림 — 판단·주문·체결·안전장치 경고를 휴대폰으로 보낸다.
(라씨 매매비서·키움 영웅문·TradingView 알림처럼 '앱을 안 열어도 알게' 하는 기능)

채널 (GitHub Secrets 로 설정. 없으면 조용히 건너뜀)
  telegram : TELEGRAM_BOT_TOKEN · TELEGRAM_CHAT_ID
             (텔레그램에서 @BotFather → /newbot → 토큰 발급, 봇에게 말 건 뒤
              https://api.telegram.org/bot<토큰>/getUpdates 에서 chat.id 확인)
  ntfy     : NTFY_TOPIC  (ntfy 앱 설치 → 아무도 못 맞힐 긴 주제 이름 구독. 계정 불필요)
             NTFY_SERVER (선택, 기본 https://ntfy.sh)

어떤 일에 알릴지는 config/autotrade.yaml 의 alerts.on 에서 고른다.
  sell · buy · fill · safety · approval · daily
"""
import json
import os
import urllib.request

TIMEOUT = 10


def _post(url, data, headers):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status


def telegram(text):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return None
    body = json.dumps({"chat_id": chat, "text": text[:4000], "disable_web_page_preview": True}).encode()
    return _post(f"https://api.telegram.org/bot{tok}/sendMessage", body,
                 {"Content-Type": "application/json"})


def ntfy(text, title="자동매매", priority=3):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return None
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    # 헤더는 latin-1만 허용 → 제목은 RFC 2047 인코딩
    import base64
    t = "=?UTF-8?B?" + base64.b64encode(title.encode()).decode() + "?="
    return _post(f"{server}/{topic}", text[:3500].encode(),
                 {"Title": t, "Priority": str(priority), "Tags": "chart_with_upwards_trend"})


def send(cfg, event, text, title="자동매매", urgent=False):
    """cfg['alerts'] 설정에 따라 보낸다. 실패해도 매매를 멈추지 않는다(경고만 출력)."""
    al = (cfg or {}).get("alerts", {}) or {}
    if not al.get("enabled") or event not in (al.get("on") or []):
        return []
    sent = []
    for ch in al.get("channels", ["telegram", "ntfy"]):
        try:
            if ch == "telegram":
                st = telegram(f"[{title}]\n{text}")
            elif ch == "ntfy":
                st = ntfy(text, title, 5 if urgent else 3)
            else:
                st = None
            if st:
                sent.append(ch)
        except Exception as e:                       # 알림 실패가 주문을 막아선 안 된다
            print(f"  (알림 실패 {ch}: {e})")
    return sent


# ── 메시지 만들기 ────────────────────────────────────────────
def fmt_decisions(plan, decisions, gate, page_url=None):
    from decision_engine import BUY, SELL, WATCH
    sells = [d for d in decisions if d["state"] == SELL]
    buys = [d for d in decisions if d["state"] == BUY]
    lines = [f"{plan['date']} 판단 · {gate['reason']}"]
    for d in sells:
        lines.append(f"🔴 {d['action']} {d['name']} — {d['reason']}")
    for d in buys:
        o = next((o for o in plan["orders"] if o["code"] == d["code"]), None)
        q = f" {o['qty']}주 @{o['price']:,}" if o and o.get("price") else ""
        lines.append(f"🟢 매수 {d['name']} {d['rank']}위{q}")
    if not sells and not buys:
        w = sum(1 for d in decisions if d["state"] == WATCH)
        lines.append(f"주문 없음 · 관망 {w}종목")
        if gate.get("need"):
            lines.append(f"매수 재개 조건: {gate['need']}")
    if page_url:
        lines.append(page_url)
    return "\n".join(lines)


def fmt_fills(fills):
    out = []
    for f in fills:
        side = "매수" if f["side"] == "buy" else "매도"
        px = f"@{f['fill']:,.0f}" if f.get("fill") else ""
        out.append(f"{'✅' if f['status'] == '체결' else '⚠️'} {side} {f['code']} {f['qty']}주 {px} {f['status']}")
    return "\n".join(out)
