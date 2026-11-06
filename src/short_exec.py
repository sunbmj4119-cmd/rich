"""
단기·전천후 몫 증권사 실행기 + 잔고 대사 — '거울 계좌' 방식.
  모의 장부(data/short/account.json)는 그대로 계속 돈다 (연구·기록이 끊기지 않게). 증권사 계좌는 정한 몫(short.broker_sleeves,
  기본 전천후만)을 '자기 돈 크기에 맞춰' 그대로 따라 한다. 아침마다 '모의 몫의 내일 보유 × 배율'을 목표로 증권사 보유와의 차이만 주문 →
  체결이 빠지거나 어긋나도 다음 날 저절로 맞춰진다.

  왜 전천후만: 긴 역사 표본 외(2005~)에서 샤프 0.97·0.94 로 두 시대가 같았고, 단기 몫은 시도 전체(약 18만 번)를 반영한
  DSR 이 32% — 운과 구분이 안 된다. 단기 몫은 앞으로의 모의 성적이 쌓일 때까지 모의로만 (설정으로 "live" 추가 가능).

  설정 (config/autotrade.yaml short:)
    broker          paper(기본·아무것도 안 보냄) | toss(토스증권 — 오라클 서버에서만, 모의투자 서버 없음) | kis_demo | kis_real
    broker_sleeves  [aw] — 증권사가 따라 할 몫 (aw = 전천후, live = 단기)
    kis_real 은 네 겹 잠금: live_trading:true + 저장소 변수 AUTOTRADE_CONFIRM=I_UNDERSTAND
                           + 준비도 점검표 '준비됨'(data/readiness.json, 아니면 short.allow_before_ready:true) + 리스크 상태 HALTED 아님
  계좌: KIS_ACCOUNT_SHORT 가 있으면 그 계좌(없으면 KIS_ACCOUNT). 30일 엔진이 같은 계좌로 증권사를 쓰면 막는다(대사가 깨짐).
  연결 (한 번, 서버): setup.sh --link → 거울 자본, 배율 = 자본 ÷ 모의 몫 평가액 (서버 ~/.rich/mirror.json — 비공개)
  주문 (아침 08:40, 시가 동시호가): 매수 = 지정가(어제 종가 × 1.12, 호가 올림 → 가격 보호) · 매도 = 시장가
    09:30 넘어 실행되면 매수는 안 보냄 · 같은 (계획 날짜·종목·방향) 성공 기록이 있으면 다시 안 보냄 (서버 ~/.rich/orders.csv)
  공개 저장소(GitHub)에는 상태·종목 코드·건수·비용(bp)만 올린다 — 현금·수량·자본은 서버 밖으로 나가지 않는다.
  증권사 주문·대사·연결은 모두 서버에서만 한다. GitHub 계산은 서버가 올린 공개 상태(recon.json)만 읽는다.
  대사 (저녁, short_trader 안): 증권사 보유 ↔ 모의 몫 보유 × 배율 — 1주·2% 넘게 다르면 그 종목 새 매수 금지,
    3종목 이상이면 계좌 정지(HALTED). 체결 조회로 실제 체결가 − 모의 체결가(시가) = 실행 비용(bp) 기록

  python src/short_exec.py --phase link | execute [--dry-run] | reconcile
"""
import argparse
import csv
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brokers  # noqa: E402
import risk as RK  # noqa: E402
import settings  # noqa: E402

PLAN = "data/short/plan.json"
ACC = "data/short/account.json"
# 공개 저장소에는 금액·수량을 올리지 않는다 — 실제 계좌의 현금·보유 수량·주문 수량·연결 자본·직접 보유분은 서버의 ~/.rich/ 에만.
PRIVATE = os.path.expanduser("~/.rich")
ORDERS = os.path.join(PRIVATE, "orders.csv")              # 보낸 주문 (수량 포함) — 비공개
MIRROR = os.path.join(PRIVATE, "mirror.json")             # 연결 자본·배율·직접 보유분 — 비공개
RECON_FULL = os.path.join(PRIVATE, "recon_full.json")     # 대사 상세 (수량·현금) — 비공개
RECON = "data/short/recon.json"                           # 공개: 상태·불일치 종목 코드·비용(bp)만
MIRROR_STATUS = "data/short/mirror_status.json"           # 공개: 연결 여부·증권사·날짜·몫만
EXEC_STATUS = "data/short/exec_status.json"               # 공개: 아침 주문 건수·성공 여부만
READY = "data/readiness.json"
LEDGER = "data/short/ledger.csv"
ORDER_COLS = ["ts", "plan_date", "broker", "sleeve", "side", "code", "name", "qty", "kind", "price", "ok", "order_no", "msg"]
LATE_BUY = "09:30"


def kst_now():
    return pd.Timestamp.now(tz="Asia/Seoul")


def make(cfg, dry=False):
    """단기 몫용 증권사. 반환 (broker 또는 None, 이름, 이유)."""
    sc = cfg.get("short") or {}
    b = sc.get("broker", "paper")
    if b == "paper":
        return None, "paper", "모의 계좌 — 증권사로 보내지 않음 (체결은 저녁 시뮬레이션)"
    acct = os.environ.get("KIS_ACCOUNT_SHORT") or os.environ.get("KIS_ACCOUNT", "")
    if cfg.get("broker", "paper").startswith("kis") and acct == os.environ.get("KIS_ACCOUNT", "") and not os.environ.get("KIS_ACCOUNT_SHORT"):
        return None, b, "30일 엔진도 같은 증권 계좌(KIS_ACCOUNT)를 씀 → 대사가 깨짐. 단기용 KIS_ACCOUNT_SHORT 를 따로 넣어야 보냄"
    if b == "kis_demo":
        return brokers.KISBroker(demo=True, live_orders=not dry, account=acct), b, "한국투자증권 모의투자"
    if b == "kis_real":
        locks = []
        if not cfg.get("live_trading"):
            locks.append("config live_trading:false")
        if os.environ.get("AUTOTRADE_CONFIRM") != "I_UNDERSTAND":
            locks.append("저장소 변수 AUTOTRADE_CONFIRM 없음")
        rd = RK.jload(READY, {}) or {}
        if rd.get("verdict") != "준비됨" and not sc.get("allow_before_ready"):
            locks.append(f"준비도 점검표 '{rd.get('verdict', '없음')}' (남은 조건 {len(rd.get('missing') or [])}개)")
        live = not locks and not dry
        return brokers.KISBroker(demo=False, live_orders=live, account=acct), b, ("실계좌 주문" if live else "실계좌 — 잠김: " + " · ".join(locks or ["dry-run"]))
    if b == "toss":
        if not on_server():
            return None, b, "토스는 등록된 IP(오라클 서버)에서만 — 여기서는 서버가 남긴 기록만 읽음"
        locks = []
        if not cfg.get("live_trading"):
            locks.append("config live_trading:false")
        if os.environ.get("AUTOTRADE_CONFIRM") != "I_UNDERSTAND":
            locks.append("서버 확인값 AUTOTRADE_CONFIRM 없음 (setup.sh --live)")
        rd = RK.jload(READY, {}) or {}
        if rd.get("verdict") != "준비됨" and not sc.get("allow_before_ready"):
            locks.append(f"준비도 점검표 '{rd.get('verdict', '없음')}' (남은 조건 {len(rd.get('missing') or [])}개)")
        live = not locks and not dry
        try:
            tb = brokers.TossBroker(live_orders=live)
        except Exception as ex:
            return None, b, f"토스 연결 준비 실패: {ex}"
        return tb, b, ("토스 실계좌 주문" if live else "토스 — 주문 잠김(만들기만 하고 안 보냄): " + " · ".join(locks or ["dry-run"]))
    return None, b, f"알 수 없는 short.broker: {b}"


def public_recon(res):
    """공개용 대사 결과 — 금액·수량을 뺀다."""
    keep = {k: res.get(k) for k in ("date", "mode", "state", "ok", "why", "blocked", "slippage_med_bp", "history")}
    keep["mismatch_codes"] = [d.get("code") for d in res.get("mismatch") or []]
    keep["slippage_bp"] = [x.get("cost_bp") for x in res.get("slippage") or []]
    return keep


def write_private(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def on_server():
    """오라클 서버인가 (토스 키 파일이 있는 곳)."""
    return os.path.exists(os.path.expanduser("~/.rich/secrets.env")) and bool(
        os.environ.get("TOSS_CLIENT_ID") or "TOSS_CLIENT_ID=" in open(os.path.expanduser("~/.rich/secrets.env"), encoding="utf-8").read())


def sent_before(plan_date, bname):
    if not os.path.exists(ORDERS):
        return set()
    o = pd.read_csv(ORDERS, dtype=str)
    o = o[(o["plan_date"] == plan_date) & (o["broker"] == bname) & (o["ok"] == "True")]
    return set(zip(o["code"], o["side"]))


def append_orders(rows):
    new = not os.path.exists(ORDERS)
    os.makedirs(os.path.dirname(ORDERS), exist_ok=True)
    if new:
        os.close(os.open(ORDERS, os.O_WRONLY | os.O_CREAT, 0o600))
    with open(ORDERS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ORDER_COLS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ORDER_COLS})


def sleeves_of(cfg):
    return list((cfg.get("short") or {}).get("broker_sleeves") or ["aw"])


def sleeve_holdings(acc, sleeves):
    """모의 장부에서 고른 몫들의 보유 수량 {code: qty}."""
    pos = {}
    if "live" in sleeves:
        for book in (acc.get("positions") or {}, acc.get("etf") or {}):
            for c, p in book.items():
                pos[c] = pos.get(c, 0) + int(p["qty"])
    if "aw" in sleeves:
        for c, p in ((acc.get("aw") or {}).get("etf") or {}).items():
            pos[c] = pos.get(c, 0) + int(p["qty"])
    return pos


def sleeve_value(acc, sleeves, price):
    v = 0.0
    if "live" in sleeves:
        v += float(acc.get("cash", 0))
    if "aw" in sleeves:
        v += float((acc.get("aw") or {}).get("cash", 0))
    for c, q in sleeve_holdings(acc, sleeves).items():
        px = price(c)
        v += q * (px or 0)
    return v


def after_plan(acc, plan, sleeves):
    """내일 계획이 다 체결됐을 때 그 몫들의 보유 (모의)."""
    pos = sleeve_holdings(acc, sleeves)
    for o in plan.get("orders") or []:
        if (o.get("sleeve") or "live") not in sleeves or o.get("session") == "close":
            continue
        q = int(o["qty"]) * (1 if o["side"] == "buy" else -1)
        pos[o["code"]] = max(0, pos.get(o["code"], 0) + q)
    return {c: q for c, q in pos.items() if q > 0}


def targets(acc, plan, mirror, sleeves):
    return {c: int(q * mirror["scale"]) for c, q in after_plan(acc, plan, sleeves).items()}


def own_positions(broker_pos, mirror):
    """증권사 보유 − 연결 때 이미 있던 '직접 보유분'(mirror.external) = 자동매매가 관리하는 수량.
    직접 산 종목은 비교하지도, 팔지도 않는다."""
    ext = (mirror or {}).get("external") or {}
    out = {}
    for c, p in (broker_pos or {}).items():
        q = float((p or {}).get("shares", 0)) - float(ext.get(c, 0))
        if q > 0:
            out[c] = {**p, "shares": q}
    return out


def diff_orders(target, broker_pos, est, collar, late=False):
    """목표 보유 − 증권사 보유 → 주문 목록 [(주문, 메모)]. 매도 먼저."""
    out = []
    codes = sorted(set(target) | set(broker_pos))
    for c in codes:
        have = float((broker_pos.get(c) or {}).get("shares", 0))
        d = int(round(float(target.get(c, 0)) - have))
        if d < 0:
            out.append(({"side": "sell", "code": c, "qty": -d, "kind": "market", "price": 0}, "시장가 매도"))
        elif d > 0:
            if late:
                out.append(({"side": "buy", "code": c, "qty": d, "kind": "skip"}, f"{LATE_BUY} 넘어 실행 — 매수 안 함"))
                continue
            p = est.get(c)
            if not p:
                out.append(({"side": "buy", "code": c, "qty": d, "kind": "skip"}, "기준 가격 없음 — 매수 안 함"))
                continue
            px = brokers.round_tick(float(p) * (1 + collar), up=True)
            out.append(({"side": "buy", "code": c, "qty": d, "kind": "limit", "price": px}, f"지정가 {px:,} (기준가 +{collar * 100:.0f}% 가격 보호)"))
    return sorted(out, key=lambda x: x[0]["side"] != "sell")


def last_prices():
    """어제 종가 (docs/short.json 의 보유·계획 + 전천후 목표) — 지정가 기준."""
    ui = RK.jload("docs/short.json", {}) or {}
    px = {}
    for p in ((ui.get("account") or {}).get("positions") or []):
        if p.get("last"):
            px[p["code"]] = float(p["last"])
    for o in ((ui.get("plan") or {}).get("orders") or []):
        if o.get("est_price"):
            px[o["code"]] = float(o["est_price"])
    try:                                               # ETF 12종 마지막 종가 (전천후 몫이 새로 살 ETF)
        e = pd.read_csv("data/etf_ohlcv.csv", dtype={"종목코드": str}).sort_values("날짜")
        e["종목코드"] = e["종목코드"].str.zfill(6)
        for c, g in e.groupby("종목코드"):
            px.setdefault(c, float(g["종가"].iloc[-1]))
    except (OSError, ValueError, KeyError):
        pass
    return px


def execute(cfg, dry=False, now=None, where=None):
    """where: 'github' | 'oracle' — 아침 주문은 한 곳에서만 보낸다(short.exec_on). 두 곳이 같이 보내면 체결 전에
    서로의 주문을 못 봐서 두 번 살 수 있다."""
    on = (cfg.get("short") or {}).get("exec_on", "github")
    if (cfg.get("short") or {}).get("broker") == "toss":
        on = "oracle"                                   # 토스는 등록된 IP 에서만
    if where and where != on:
        print(f"거울 계좌 주문은 {on} 에서 보냄 (short.exec_on) — 여기({where})서는 건너뜀")
        return []
    plan = RK.jload(PLAN, {}) or {}
    acc = RK.jload(ACC, {}) or {}
    mirror = RK.jload(MIRROR, None)
    b, bname, why = make(cfg, dry)
    print(f"거울 계좌 실행 ({plan.get('made')} 계획) · {bname}: {why}")
    if b is None:
        return []
    if not mirror or mirror.get("broker") != bname:
        print("  거울 계좌가 연결되지 않음 — control.yml short_link 를 먼저 실행")
        return []
    rk = plan.get("risk") or {}
    if rk.get("state") == "HALTED":
        print("  리스크 매니저 HALTED — 보내지 않음:", "; ".join(rk.get("why") or []))
        return []
    now = now or kst_now()
    sl = mirror.get("sleeves") or sleeves_of(cfg)
    tgt = targets(acc, plan, mirror, sl)
    bpos = own_positions(b.account()["positions"], mirror)
    est = last_prices()
    for o in plan.get("orders") or []:
        est.setdefault(o["code"], o.get("est_price"))
    for c in tgt:
        if c not in est:
            est[c] = b.price(c)
    collar = (rk.get("limits") or {}).get("collar", RK.COLLAR)
    late = now.strftime("%H:%M") > LATE_BUY or rk.get("state") == "REDUCING"
    tag = bname + ("_live" if b.is_live else "")
    done = sent_before(plan.get("made", ""), tag)
    rows = []
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    for o, note in diff_orders(tgt, bpos, est, collar, late):
        o = {**o, "sleeve": ",".join(sl), "name": o["code"]}
        if o["kind"] == "skip" or (o["code"], o["side"]) in done:
            rows.append({**o, "ts": ts, "plan_date": plan.get("made", ""), "broker": tag, "ok": False, "msg": note if o["kind"] == "skip" else "이미 보냄"})
            continue
        extra = {}
        if bname == "toss":                           # 9시 전 = 장 개시 주문(시가 단일가) · 같은 날 같은 주문은 토스가 한 번만 받게
            extra = {"tif": "OPG" if now.strftime("%H:%M") < "09:00" else "DAY",
                     "client_id": f"m{str(plan.get('made', '')).replace('-', '')}{o['side'][0]}{o['code']}"}
        try:
            r = ({"ok": False, "order_no": "", "msg": "DRY-RUN — 보내지 않음"} if dry
                 else b.order(o["side"], o["code"], int(o["qty"]), o["kind"], o.get("price"), date=plan.get("made"), **extra))
        except Exception as ex:                       # 한 주문이 실패해도 나머지는 계속 (기록은 남김)
            r = {"ok": False, "order_no": "", "msg": f"{type(ex).__name__}: {ex}"}
        rows.append({**o, "ts": ts, "plan_date": plan.get("made", ""), "broker": tag, "ok": bool(r.get("ok")),
                     "order_no": r.get("order_no", ""), "msg": (r.get("msg") or "") + f" · {note}"})
        time.sleep(0.6 if bname == "kis_demo" else 0.15)             # 모의투자 서버는 초당 호출 한도가 낮다
    append_orders(rows)
    try:                                              # 공개 상태: 건수·종목 코드·방향만 (수량·가격 없음)
        json.dump({"date": now.strftime("%Y-%m-%d %H:%M"), "broker": tag, "plan": plan.get("made"),
                   "sent": sum(1 for r in rows if r["ok"]), "failed": sum(1 for r in rows if not r["ok"] and r.get("kind") != "skip"),
                   "skipped": sum(1 for r in rows if r.get("kind") == "skip"),
                   "orders": [{"code": r["code"], "side": r["side"], "ok": bool(r["ok"])} for r in rows]},
                  open(EXEC_STATUS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except OSError:
        pass
    for r in rows:
        print(f"  {'✓' if r['ok'] else '·'} {r['side']} {r['code']} {r['qty']}주 {r.get('kind')} {r.get('price') or ''} — {r['msg']}")
    if not rows:
        print("  증권사 보유가 이미 목표와 같음 — 주문 없음")
    return rows


def compare(expect, broker_pos):
    """대사: 종목별 '모의 몫 × 배율' 수량 vs 증권사 수량. 1주·2% 넘게 다르면 불일치."""
    codes = sorted(set(expect) | set(broker_pos))
    diff = []
    for c in codes:
        e = int(expect.get(c, 0))
        h = int((broker_pos.get(c) or {}).get("shares", 0))
        if abs(h - e) > max(1, 0.02 * e):
            diff.append({"code": c, "expect": e, "broker": h})
    state, why = "ACTIVE", []
    if diff:
        state = "REDUCING"
        why.append(f"수량 불일치 {len(diff)}종목 → 그 종목 새 매수 금지")
    if len(diff) >= 3:
        state = "HALTED"
        why.append(f"불일치 {len(diff)}종목 → 계좌 정지 (점검 필요)")
    return {"state": state, "why": why or ["일치"], "mismatch": diff, "blocked": [d["code"] for d in diff]}


def slippage(fills, ledger):
    """증권사 실제 체결가 vs 모의 체결가(같은 날 시가) — bp. 매수는 비싸게 산 만큼, 매도는 싸게 판 만큼이 비용."""
    out = []
    for f in fills:
        if not f["filled"]:
            continue
        d = f["date"]
        d = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d
        x = ledger[(ledger["날짜"].astype(str) == d) & (ledger["종목코드"].astype(str) == f["code"])]
        x = x[x["구분"] == ("매수" if f["side"] == "buy" else "매도")]
        if not len(x):
            continue
        ref = float(x["가격"].iloc[0])
        bp = (f["avg"] / ref - 1) * 1e4 * (1 if f["side"] == "buy" else -1)
        out.append({"date": d, "code": f["code"], "side": f["side"], "broker_px": f["avg"], "book_px": ref, "cost_bp": round(bp, 1)})
    return out


def reconcile(cfg, acc=None, save=True, after_plan=False):
    """acc: 방금 체결을 반영한 모의 장부(short_trader 안에서 부를 때). 모의(paper)면 바로 끝.
    after_plan: 서버에서 장 마감 뒤 돌 때 — 장부는 아직 어제 저녁 상태이므로 '어제 계획이 다 체결된 뒤 보유 × 배율'과 비교.
    토스는 GitHub 에서 접속이 안 되므로 서버가 남긴 recon.json 을 읽기만 한다 (1영업일 넘게 묵으면 새 매수 금지)."""
    acc = acc if acc is not None else (RK.jload(ACC, {}) or {})
    old_rc = RK.jload(RECON, {}) or {}
    hist = old_rc.get("history") or []
    today = kst_now().strftime("%Y-%m-%d")
    mirror = RK.jload(MIRROR, None)
    sb = (cfg.get("short") or {}).get("broker", "paper")
    if sb != "paper" and not on_server():               # 증권사는 서버에서만 — 여기서는 서버가 올린 공개 상태만 읽는다
        lag = len(pd.bdate_range(pd.Timestamp(old_rc.get("date", "2000-01-01")) + pd.Timedelta(days=1), today)) if old_rc.get("date") else 99
        linked = (RK.jload(MIRROR_STATUS, {}) or {}).get("linked") or mirror
        if not linked:
            return {"date": today, "mode": sb, "state": "ACTIVE", "why": [f"{sb} 거울 계좌 연결 전 — 모의만"], "mismatch": [], "blocked": [], "ok": False}
        if lag > 1:
            return {**old_rc, "state": "REDUCING", "why": [f"증권사 잔고 대사 기록이 {lag}영업일 묵음 — 서버 저녁 대사(evening.sh) 확인 전까지 새 매수 금지"]}
        return old_rc
    b, bname, why = make(cfg, dry=True)
    if b is None:
        res = {"date": today, "mode": bname, "state": "ACTIVE", "why": [why], "mismatch": [], "blocked": [], "ok": bname == "paper"}
    elif not mirror or mirror.get("broker") != bname:
        res = {"date": today, "mode": bname, "state": "ACTIVE", "why": ["거울 계좌 연결 전 — control.yml short_link 실행"],
               "mismatch": [], "blocked": [], "ok": False}
    else:
        try:
            a = b.account()
            if bname == "toss" and not b.is_live:            # 토스는 연습 서버가 없다 — 주문이 잠긴 동안은 '조회·해석이 매일 되는가'만 본다
                res = {"date": today, "mode": "toss_dry", "state": "ACTIVE", "ok": True, "mismatch": [], "blocked": [],
                       "why": ["토스 연결 정상 (보유·매수 가능 금액 해석) — 주문 잠김 상태라 보유 비교는 실제 주문 뒤부터"]}
                raise StopIteration
            sl = mirror.get("sleeves") or sleeves_of(cfg)
            if after_plan:
                expect = targets(acc, RK.jload(PLAN, {}) or {}, mirror, sl)
            else:
                expect = {c: int(q * mirror["scale"]) for c, q in sleeve_holdings(acc, sl).items()}
            res = {"date": today, "mode": bname, **compare(expect, own_positions(a["positions"], mirror)), "broker_cash": a["cash"]}  # 현금은 비공개 파일에만
            res["ok"] = res["state"] == "ACTIVE"
            try:
                start = (pd.Timestamp(today) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                led = pd.read_csv(LEDGER, dtype={"종목코드": str})
                sp = slippage(b.daily_fills(start, today), led)
                res["slippage"] = sp
                res["slippage_med_bp"] = round(float(pd.Series([x["cost_bp"] for x in sp]).median()), 1) if sp else None
            except Exception as ex:
                res["slippage_error"] = f"{type(ex).__name__}: {ex}"
        except Exception as ex:
            res = {"date": today, "mode": bname, "state": "HALTED", "why": [f"잔고 조회 실패({type(ex).__name__}) → 정지(닫힘)"],
                   "mismatch": [], "blocked": [], "ok": False}
    hist = [h for h in hist if h.get("date") != today] + [{k: res.get(k) for k in ("date", "mode", "state", "ok")}]
    res["history"] = hist[-250:]
    if save:
        os.makedirs(os.path.dirname(RECON), exist_ok=True)
        json.dump(public_recon(res), open(RECON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if bname != "paper":
            write_private(RECON_FULL, res)
    print(f"잔고 대사 {today} · {bname}: {res['state']} — " + "; ".join(res["why"]))
    return public_recon(res) if bname != "paper" else res


def link(cfg):
    """거울 계좌 연결: 자본 = short.broker_capital (없으면 현금 전부), 배율 = 자본 ÷ 모의 몫 평가액.
    이미 있던 종목은 '직접 보유분'(external)으로 기록 — 자동매매는 그 수량을 비교도, 매도도 하지 않는다."""
    b, bname, why = make(cfg, dry=True)
    if b is None:
        print("연결할 증권사 없음:", why)
        return None
    a = b.account()
    ext = {c: float(p["shares"]) for c, p in a["positions"].items()}
    if ext:
        print(f"증권사 계좌에 이미 {len(ext)}종목이 있습니다 — '직접 보유분'으로 기록하고 자동매매는 그 수량을 건드리지 않습니다")
    acc = RK.jload(ACC, {}) or {}
    sl = sleeves_of(cfg)
    px = last_prices()
    val = sleeve_value(acc, sl, lambda c: px.get(c))
    cap = float((cfg.get("short") or {}).get("broker_capital") or a["cash"])
    cap = min(cap, float(a["cash"]))
    if not (cfg.get("short") or {}).get("broker_capital"):
        print(f"  주의: short.broker_capital 을 정하지 않아 계좌 현금 전부({cap:,.0f}원)를 자동매매 몫으로 잡습니다")
    if val <= 0 or cap <= 0:
        print("모의 몫 평가액 또는 증권사 현금이 0 — 연결 안 함")
        return None
    m = {"broker": bname, "date": kst_now().strftime("%Y-%m-%d"), "capital": round(cap), "sleeves": sl,
         "paper_value": round(val), "scale": cap / val, "external": ext}
    write_private(MIRROR, m)                                       # 자본·배율·직접 보유분 — 서버에만
    json.dump({"linked": True, "broker": bname, "date": m["date"], "sleeves": sl, "external_codes": len(ext)},
              open(MIRROR_STATUS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"거울 계좌 연결: {bname} 자본 {cap:,.0f}원 · 따라 할 몫 {sl} (모의 평가액 {val:,.0f}원) · 배율 {m['scale']:.4f} — 다음 아침부터 주문")
    return m


SMOKE = "data/toss/smoke.json"


def smoke(cfg, yes=False, code="261240", wait=90):
    """1주 실주문 시험 (토스·서버·정규장 안에서만): 지정가로 1주 사고 → 체결 확인 → 시장가로 1주 팔고 → 체결 확인.
    실제 돈(1주 값 + 수수료)이 움직인다 — yes 가 있어야만 한다. 결과(성공 여부·왕복 차이 bp)만 공개 파일로."""
    now = kst_now()
    if not yes:
        print("1주 실주문 시험은 --yes 가 있어야 합니다 (setup.sh --smoke 로 실행)")
        return None
    if not on_server():
        print("토스는 서버에서만")
        return None
    if now.weekday() > 4 or not ("09:05" <= now.strftime("%H:%M") <= "15:10"):
        print("정규장(평일 09:05~15:10)에만 합니다")
        return None
    b = brokers.TossBroker(live_orders=True)
    if not b.verified():
        print(f"토스 명세 확인 전 — 서버 {b.spec_version()} / 확인 {b.VERIFIED}")
        return None
    day = now.strftime("%Y%m%d")
    res = {"date": now.strftime("%Y-%m-%d %H:%M"), "code": code, "ok": False, "steps": []}

    def wait_fill(oid):
        for _ in range(int(wait / 3)):
            o = b.order_status(oid) or {}
            stt = o.get("status")
            if stt in ("FILLED", "CANCELED", "REJECTED"):
                return o
            time.sleep(3)
        return b.order_status(oid) or {}
    px = b.price(code)
    try:
        cash = b.account()["cash"]
    except Exception as ex:
        cash = None
        print(f"매수 가능 금액 조회 실패: {ex}")
    if not px:
        res["summary"] = "현재가 조회 실패"
    elif cash is not None and cash < px * 1.03:
        print(f"매수 가능 현금이 1주 값({px:,.0f}원)보다 적습니다 — 토스 계좌에 2만 원 이상 넣은 뒤 다시 하세요 (아무 주문도 안 보냄)")
        return None
    else:
        lim = brokers.round_tick(px * 1.01, up=True)
        r = b.order("buy", code, 1, "limit", lim, tif="DAY", client_id=f"smk{day}b")
        res["steps"].append({"step": "매수 접수", "ok": r["ok"], "msg": r["msg"][:120]})
        o = wait_fill(r["order_no"]) if r["ok"] else {}
        fb = (o.get("execution") or {}).get("averageFilledPrice")
        res["steps"].append({"step": "매수 체결", "ok": o.get("status") == "FILLED", "status": o.get("status")})
        if r["ok"] and o.get("status") != "FILLED":
            res["steps"].append({"step": "매수 취소", **b.cancel(r["order_no"])})
        if o.get("status") == "FILLED":
            r2 = b.order("sell", code, 1, "market", tif="DAY", client_id=f"smk{day}s")
            res["steps"].append({"step": "매도 접수", "ok": r2["ok"], "msg": r2["msg"][:120]})
            o2 = wait_fill(r2["order_no"]) if r2["ok"] else {}
            fs = (o2.get("execution") or {}).get("averageFilledPrice")
            res["steps"].append({"step": "매도 체결", "ok": o2.get("status") == "FILLED", "status": o2.get("status")})
            if o2.get("status") == "FILLED" and fb and fs:
                bp = (float(fb) - float(fs)) / px * 1e4
                res.update(ok=True, roundtrip_bp=round(bp, 1),
                           summary=f"사기·팔기 모두 체결 (왕복 가격 차이 {bp:.1f}bp, 수수료 별도)")
        res.setdefault("summary", "실패 — " + " · ".join(f"{x['step']} {'✓' if x.get('ok') else '✕'}" for x in res["steps"]))
    os.makedirs(os.path.dirname(SMOKE), exist_ok=True)
    json.dump(res, open(SMOKE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("1주 실주문 시험:", res["summary"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["execute", "reconcile", "link", "smoke"], required=True)
    ap.add_argument("--yes", action="store_true", help="smoke: 실제 돈으로 1주 사고팔기에 동의")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--where", choices=["github", "oracle"], default=None, help="어디서 돌리나 (short.exec_on 과 다르면 건너뜀)")
    ap.add_argument("--after-plan", action="store_true", help="서버 장 마감 뒤 대사: 어제 계획이 다 체결된 뒤 보유와 비교")
    a = ap.parse_args()
    cfg = settings.load()
    if a.phase == "execute":
        execute(cfg, a.dry_run, where=a.where)
    elif a.phase == "link":
        link(cfg)
    elif a.phase == "smoke":
        smoke(cfg, yes=a.yes)
    else:
        reconcile(cfg, after_plan=a.after_plan)


if __name__ == "__main__":
    main()
