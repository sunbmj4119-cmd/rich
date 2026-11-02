"""
단기·전천후 몫 증권사 실행기 + 잔고 대사 — '거울 계좌' 방식.
  모의 장부(data/short/account.json)는 그대로 계속 돈다 (연구·기록이 끊기지 않게). 증권사 계좌는 정한 몫(short.broker_sleeves,
  기본 전천후만)을 '자기 돈 크기에 맞춰' 그대로 따라 한다. 아침마다 '모의 몫의 내일 보유 × 배율'을 목표로 증권사 보유와의 차이만 주문 →
  체결이 빠지거나 어긋나도 다음 날 저절로 맞춰진다.

  왜 전천후만: 긴 역사 표본 외(2005~)에서 샤프 0.97·0.94 로 두 시대가 같았고, 단기 몫은 시도 전체(약 18만 번)를 반영한
  DSR 이 32% — 운과 구분이 안 된다. 단기 몫은 앞으로의 모의 성적이 쌓일 때까지 모의로만 (설정으로 "live" 추가 가능).

  설정 (config/autotrade.yaml short:)
    broker          paper(기본·아무것도 안 보냄) | kis_demo(한국투자 모의투자 서버) | kis_real(실계좌)
    broker_sleeves  [aw] — 증권사가 따라 할 몫 (aw = 전천후, live = 단기)
    kis_real 은 네 겹 잠금: live_trading:true + 저장소 변수 AUTOTRADE_CONFIRM=I_UNDERSTAND
                           + 준비도 점검표 '준비됨'(data/readiness.json, 아니면 short.allow_before_ready:true) + 리스크 상태 HALTED 아님
  계좌: KIS_ACCOUNT_SHORT 가 있으면 그 계좌(없으면 KIS_ACCOUNT). 30일 엔진이 같은 계좌로 증권사를 쓰면 막는다(대사가 깨짐).
  연결 (한 번): control.yml short_link → 증권사 현금 = 거울 자본, 배율 = 자본 ÷ 모의 몫 평가액 (data/short/mirror.json)
  주문 (아침 08:40, 시가 동시호가): 매수 = 지정가(어제 종가 × 1.12, 호가 올림 → 가격 보호) · 매도 = 시장가
    09:30 넘어 실행되면 매수는 안 보냄 · 같은 (계획 날짜·종목·방향) 성공 기록이 있으면 다시 안 보냄 (data/short/orders.csv)
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
ORDERS = "data/short/orders.csv"
RECON = "data/short/recon.json"
MIRROR = "data/short/mirror.json"
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
    return None, b, f"알 수 없는 short.broker: {b}"


def sent_before(plan_date, bname):
    if not os.path.exists(ORDERS):
        return set()
    o = pd.read_csv(ORDERS, dtype=str)
    o = o[(o["plan_date"] == plan_date) & (o["broker"] == bname) & (o["ok"] == "True")]
    return set(zip(o["code"], o["side"]))


def append_orders(rows):
    new = not os.path.exists(ORDERS)
    os.makedirs(os.path.dirname(ORDERS), exist_ok=True)
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


def diff_orders(target, broker_pos, est, collar, late=False):
    """목표 보유 − 증권사 보유 → 주문 목록 [(주문, 메모)]. 매도 먼저."""
    out = []
    codes = sorted(set(target) | set(broker_pos))
    for c in codes:
        have = int((broker_pos.get(c) or {}).get("shares", 0))
        d = int(target.get(c, 0)) - have
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
    bpos = b.account()["positions"]
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
        try:
            r = ({"ok": False, "order_no": "", "msg": "DRY-RUN — 보내지 않음"} if dry
                 else b.order(o["side"], o["code"], o["qty"], o["kind"], o.get("price"), date=plan.get("made")))
        except Exception as ex:                       # 한 주문이 실패해도 나머지는 계속 (기록은 남김)
            r = {"ok": False, "order_no": "", "msg": f"{type(ex).__name__}: {ex}"}
        rows.append({**o, "ts": ts, "plan_date": plan.get("made", ""), "broker": tag, "ok": bool(r.get("ok")),
                     "order_no": r.get("order_no", ""), "msg": (r.get("msg") or "") + f" · {note}"})
        time.sleep(0.6 if bname == "kis_demo" else 0.15)             # 모의투자 서버는 초당 호출 한도가 낮다
    append_orders(rows)
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


def reconcile(cfg, acc=None, save=True):
    """acc: 방금 체결을 반영한 모의 장부(short_trader 안에서 부를 때). 모의(paper)면 바로 끝."""
    acc = acc if acc is not None else (RK.jload(ACC, {}) or {})
    hist = (RK.jload(RECON, {}) or {}).get("history") or []
    today = kst_now().strftime("%Y-%m-%d")
    mirror = RK.jload(MIRROR, None)
    b, bname, why = make(cfg, dry=True)
    if b is None:
        res = {"date": today, "mode": bname, "state": "ACTIVE", "why": [why], "mismatch": [], "blocked": [], "ok": bname == "paper"}
    elif not mirror or mirror.get("broker") != bname:
        res = {"date": today, "mode": bname, "state": "ACTIVE", "why": ["거울 계좌 연결 전 — control.yml short_link 실행"],
               "mismatch": [], "blocked": [], "ok": False}
    else:
        try:
            a = b.account()
            sl = mirror.get("sleeves") or sleeves_of(cfg)
            expect = {c: int(q * mirror["scale"]) for c, q in sleeve_holdings(acc, sl).items()}
            res = {"date": today, "mode": bname, **compare(expect, a["positions"]), "broker_cash": a["cash"]}
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
        json.dump(res, open(RECON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"잔고 대사 {today} · {bname}: {res['state']} — " + "; ".join(res["why"]))
    return res


def link(cfg):
    """거울 계좌 연결: 증권사 현금 = 자본, 배율 = 자본 ÷ 모의 몫 평가액. 증권사에 이미 보유가 있으면 연결하지 않는다(섞임 방지)."""
    b, bname, why = make(cfg, dry=True)
    if b is None:
        print("연결할 증권사 없음:", why)
        return None
    a = b.account()
    if a["positions"]:
        print(f"증권사 계좌에 이미 {len(a['positions'])}종목 보유 — 비어 있는(현금만 있는) 계좌로 연결해야 합니다")
        return None
    acc = RK.jload(ACC, {}) or {}
    sl = sleeves_of(cfg)
    px = last_prices()
    val = sleeve_value(acc, sl, lambda c: px.get(c))
    cap = float((cfg.get("short") or {}).get("broker_capital") or a["cash"])
    cap = min(cap, float(a["cash"]))
    if val <= 0 or cap <= 0:
        print("모의 몫 평가액 또는 증권사 현금이 0 — 연결 안 함")
        return None
    m = {"broker": bname, "date": kst_now().strftime("%Y-%m-%d"), "capital": round(cap), "sleeves": sl,
         "paper_value": round(val), "scale": cap / val}
    json.dump(m, open(MIRROR, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"거울 계좌 연결: {bname} 자본 {cap:,.0f}원 · 따라 할 몫 {sl} (모의 평가액 {val:,.0f}원) · 배율 {m['scale']:.4f} — 다음 아침부터 주문")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["execute", "reconcile", "link"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--where", choices=["github", "oracle"], default=None, help="어디서 돌리나 (short.exec_on 과 다르면 건너뜀)")
    a = ap.parse_args()
    cfg = settings.load()
    if a.phase == "execute":
        execute(cfg, a.dry_run, where=a.where)
    elif a.phase == "link":
        link(cfg)
    else:
        reconcile(cfg)


if __name__ == "__main__":
    main()
