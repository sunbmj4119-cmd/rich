"""
자동매매 실행기 — 판단 엔진의 매수/유지/매도/관망 결정을 실제 주문으로 바꾼다.

단계
  decide  (평일 17:00, daily.yml 뒤)  지표 계산 → 4상태 판단 → 주문 계획 data/autotrade/plan.json
          · paper  : 어제 낸 모의 주문을 오늘 종가로 체결 → 오늘 판단 → 새 모의 주문 접수
          · kis_*  : 계획만 만든다 (장이 닫혀 있다)
  execute (다음 거래일 08:40, autotrade.yml)  계획을 읽어 증권사에 주문 (kis_* 전용)
          매도 먼저(시장가) → 매수(전일종가+1% 지정가). 수량은 실시간 잔고로 다시 계산.
  all     decide + execute 를 한 번에 (장중 로컬 실행용)

안전장치 (하나라도 걸리면 주문을 보내지 않고 이유를 기록)
  · enabled:false 또는 config/KILL 파일 → 킬스위치
  · 점수 데이터가 max_data_lag_days 보다 오래됨 → 낡은 정보로 매매 금지
  · 계획 날짜가 최신 거래일이 아님 → 어제 계획을 오늘 또 실행하는 사고 방지
  · 같은 날 같은 종목·방향 주문은 한 번만 (orders.csv 로 중복 차단)
  · 1회 주문/하루 매수/종목 비중 한도 · 현금 버퍼 · 계좌 서킷브레이커
  · kis_real 은 live_trading:true + AUTOTRADE_CONFIRM=I_UNDERSTAND 둘 다 있어야 전송

사용
  python src/autotrader.py --phase decide            # 설정의 broker로
  python src/autotrader.py --phase execute --broker kis_demo
  python src/autotrader.py --phase all --dry-run     # 무엇을 할지 보기만
"""
import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
import features as F                                     # noqa: E402
from decision_engine import decide, load_news, BUY, SELL, HOLD, WATCH   # noqa: E402
from brokers import make_broker, round_tick, PaperBroker  # noqa: E402

CFG = "config/autotrade.yaml"
KILL = "config/KILL"
DIR = "data/autotrade"
LEDGER = f"{DIR}/ledger.json"
PLAN = f"{DIR}/plan.json"
DECISIONS = f"{DIR}/decisions.csv"
ORDERS = f"{DIR}/orders.csv"
EQUITY = f"{DIR}/equity.csv"
STATUS = "docs/autotrade.json"
KST = timezone(timedelta(hours=9))


def now_kst():
    return datetime.now(KST)


def jload(p, default):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default


def jsave(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))


def append_csv(path, rows, fields):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


ORDER_FIELDS = ["ts", "plan_date", "broker", "side", "code", "name", "qty", "kind", "price",
                "ok", "order_no", "msg", "reason"]


def already_sent(plan_date, broker):
    """이 계획 날짜로 이미 보낸 (종목, 방향). 같은 계획을 두 번 실행해도 중복 주문이 안 나간다."""
    if not os.path.exists(ORDERS):
        return set()
    o = pd.read_csv(ORDERS, dtype=str)
    o = o[(o["plan_date"] == plan_date) & (o["broker"] == broker) & (o["ok"] == "True")]
    return set(zip(o["code"], o["side"]))


# ══════════════════════════════════════════════════════════════
# 원장 — 증권사 잔고에 없는 정보(진입일·부분익절 여부)를 보관
# ══════════════════════════════════════════════════════════════
def reconcile(ledger, acct, asof):
    """증권사 잔고가 진실. 원장은 잔고에 맞춰 고친다."""
    new = {}
    for c, p in acct["positions"].items():
        e = ledger.get(c)
        if e is None:
            e = {"entry_date": asof, "partial_done": False, "last_shares": p["shares"]}
        elif p["shares"] < e.get("last_shares", p["shares"]) - 1e-9:
            e["partial_done"] = True            # 줄었으면 부분 매도가 있었다
        elif p["shares"] > e.get("last_shares", 0) + 1e-9 and e.get("last_shares", 0) == 0:
            e["entry_date"] = asof
        e["last_shares"] = p["shares"]
        new[c] = e
    return new


def engine_positions(acct, ledger, panel):
    dates, close = panel["dates"], panel["close"]
    pos = {}
    for c, p in acct["positions"].items():
        if c not in close.columns:
            continue                             # 유니버스 밖 종목 — 엔진이 다루지 않는다
        e = ledger.get(c, {})
        ed = pd.Timestamp(e.get("entry_date") or dates[-1])
        et = int(min(np.searchsorted(dates.values, np.datetime64(ed)), len(dates) - 1))
        seg = close[c].iloc[et:].dropna()
        peak = float(seg.max()) if len(seg) else p["avg"]
        pos[c] = {"shares": p["shares"], "avg": p["avg"], "entry_t": et,
                  "peak": max(peak, p["avg"]), "partial_done": bool(e.get("partial_done"))}
    return pos


# ══════════════════════════════════════════════════════════════
# 안전 점검
# ══════════════════════════════════════════════════════════════
def safety(cfg, panel, feats):
    rk = cfg.get("risk", {})
    issues, flags = [], {}
    if not cfg.get("enabled", True) or os.path.exists(KILL):
        issues.append("킬스위치 ON (enabled:false 또는 config/KILL) — 주문 중단")
    asof = panel["dates"][-1]
    lag = (now_kst().date() - asof.date()).days
    if lag > rk.get("max_data_lag_days", 4):
        issues.append(f"점수 데이터가 {lag}일 묵음 ({asof.date()}) — 주문 중단")
    fl = feats.get("flow")
    if fl is not None:
        valid = fl.notna().sum(axis=1)
        last_ok = valid[valid > len(panel["codes"]) * 0.5].index.max()
        gap = int((panel["dates"] > last_ok).sum()) if pd.notna(last_ok) else 999
        flags["flow_last"] = str(last_ok.date()) if pd.notna(last_ok) else None
        flags["flow_stale"] = gap > rk.get("stale_flow_days", 10)
        flags["flow_gap"] = gap
    return issues, flags


def account_drawdown(equity_now, cfg):
    hist = pd.read_csv(EQUITY) if os.path.exists(EQUITY) else pd.DataFrame(columns=["equity"])
    peak = max([equity_now] + list(hist["equity"].astype(float)))
    dd = equity_now / peak - 1 if peak else 0
    lim = cfg.get("risk", {}).get("max_account_drawdown", 0.30)
    return dd, (f"계좌 고점대비 {dd*100:.1f}% (한도 -{lim*100:.0f}%) — 신규매수 중단" if dd <= -lim else None)


# ══════════════════════════════════════════════════════════════
# 주문 계획
# ══════════════════════════════════════════════════════════════
def build_plan(decisions, acct, equity, cfg, asof):
    st, rk, ex = cfg["strategy"], cfg.get("risk", {}), cfg.get("execution", {})
    orders = []
    for d in decisions:
        if d["state"] != SELL:
            continue
        sh = acct["positions"].get(d["code"], {}).get("shares", 0)
        qty = sh if d["fraction"] >= 1 else max(1, math.floor(sh * d["fraction"]))
        if d["fraction"] < 1 and sh < 2:
            d["state"], d["reason"] = HOLD, d["reason"] + " (1주라 부분익절 불가 → 유지)"
            continue
        orders.append({"side": "sell", "code": d["code"], "name": d["name"], "qty": int(qty),
                       "kind": ex.get("sell_order", "market"), "price": None,
                       "ref_price": d["price"], "reason": d["reason"]})
    # 매도 대금은 체결 후에야 들어오지만, 계획은 '매도 후 현금' 기준으로 짠다 (백테스트와 동일).
    cash = acct["cash"] + sum(o["qty"] * o["ref_price"] for o in orders) * (1 - 0.0015)
    budget_left = min(rk.get("max_daily_buy_krw", 1e18), cash - equity * rk.get("cash_buffer", 0.0))
    slot = equity / st["max_positions"]
    for d in decisions:
        if d["state"] != BUY:
            continue
        limit = round_tick(d["price"] * (1 + ex.get("buy_slippage", 0.01)), up=False) \
            if ex.get("buy_order", "limit") == "limit" else None
        unit = limit or d["price"] * 1.01
        amt = min(slot, rk.get("max_order_krw", 1e18), equity * rk.get("max_position_weight", 1.0),
                  budget_left)
        qty = int(amt // unit) if amt > 0 else 0
        if qty <= 0:
            d["state"] = WATCH
            d["reason"] = (f"매수 조건은 통과했으나 예산 부족 (슬롯 {slot:,.0f}원 · 남은 한도 "
                           f"{max(budget_left, 0):,.0f}원 · 1주 {unit:,.0f}원)")
            d["need"] = ["현금 확보 또는 max_order_krw/계좌 규모 확대"]
            continue
        budget_left -= qty * unit
        d["qty"] = qty
        orders.append({"side": "buy", "code": d["code"], "name": d["name"], "qty": qty,
                       "kind": "limit" if limit else "market", "price": limit,
                       "ref_price": d["price"], "reason": d["reason"]})
    n = rk.get("max_orders_per_run", 99)
    return {"date": str(asof.date()), "created": now_kst().isoformat(timespec="seconds"),
            "broker": cfg.get("broker"), "equity": equity, "cash": acct["cash"],
            "orders": orders[:n], "dropped": len(orders) - min(len(orders), n)}


def send(plan, broker, cfg, dry):
    sent = already_sent(plan["date"], broker.name + ("_live" if broker.is_live else ""))
    rows = []
    for o in sorted(plan["orders"], key=lambda o: o["side"] != "sell"):   # 매도 먼저
        key = (o["code"], o["side"])
        if key in sent:
            rows.append({**o, "ok": False, "order_no": "", "msg": "중복 차단 (이미 전송됨)"})
            continue
        if dry:
            r = {"ok": False, "order_no": "", "msg": "DRY-RUN — 전송 안 함"}
        else:
            r = broker.order(o["side"], o["code"], o["qty"], o["kind"], o["price"], date=plan["date"])
        rows.append({**o, **r})
    ts = now_kst().isoformat(timespec="seconds")
    bname = broker.name + ("_live" if broker.is_live else "")
    append_csv(ORDERS, [{**r, "ts": ts, "plan_date": plan["date"], "broker": bname} for r in rows],
               ORDER_FIELDS)
    return rows


# ══════════════════════════════════════════════════════════════
def phase_decide(cfg, dry):
    panel, feats = F.build_all()
    t = len(panel["dates"]) - 1
    asof = panel["dates"][t]
    asof_s = str(asof.date())
    issues, flags = safety(cfg, panel, feats)
    broker = make_broker(cfg)
    close_t = panel["close"].iloc[t]

    fills = []
    if isinstance(broker, PaperBroker):
        fills = broker.settle(lambda c: float(close_t[c]) if c in close_t and pd.notna(close_t[c])
                              else None, asof_s)
    acct = broker.account()
    ledger = reconcile(jload(LEDGER, {}), acct, asof_s)
    pos = engine_positions(acct, ledger, panel)
    mv = sum(p["shares"] * (float(close_t[c]) if c in close_t and pd.notna(close_t[c]) else p["avg"])
             for c, p in acct["positions"].items())
    equity = acct["cash"] + mv
    dd, halt = account_drawdown(equity, cfg)
    flags["account_halt"] = halt

    decisions, gate = decide(t, panel, feats, pos, cfg, news=load_news(), flags=flags)
    plan = build_plan(decisions, acct, equity, cfg, asof)
    plan["issues"] = issues
    jsave(PLAN, plan)
    if not dry:
        jsave(LEDGER, ledger)

    sent = []
    if isinstance(broker, PaperBroker):
        if issues:
            print("⛔ 안전장치로 모의 주문도 보내지 않음:", *issues, sep="\n  ")
        else:
            sent = send(plan, broker, cfg, dry)
        if not dry:
            broker.save()
    # 평가액 기록 (같은 날 재실행이면 덮어씀)
    eq = pd.read_csv(EQUITY) if os.path.exists(EQUITY) else pd.DataFrame()
    row = {"date": asof_s, "broker": cfg.get("broker"), "equity": round(equity),
           "cash": round(acct["cash"]), "n_positions": len(acct["positions"]),
           "regime": gate["regime"], "gate_open": gate["open"]}
    eq = pd.concat([eq[eq.get("date", pd.Series(dtype=str)) != asof_s] if len(eq) else eq,
                    pd.DataFrame([row])])
    if not dry:
        os.makedirs(DIR, exist_ok=True)
        eq.to_csv(EQUITY, index=False)
    write_decisions(decisions)
    write_status(cfg, decisions, gate, plan, flags, issues, equity, dd, fills, sent, eq)
    report(decisions, gate, plan, flags, issues, equity, fills)
    return plan


def phase_execute(cfg, dry):
    plan = jload(PLAN, None)
    if not plan:
        print("계획 없음 — decide 먼저 실행"); return
    broker = make_broker(cfg)
    if isinstance(broker, PaperBroker):
        print("paper 브로커는 decide 단계에서 주문까지 끝낸다 — execute 불필요"); return
    panel_dates = pd.read_csv("data/scores.csv", usecols=["날짜"])["날짜"]
    latest = str(pd.to_datetime(panel_dates).max().date())
    problems = list(plan.get("issues", []))
    if plan["date"] != latest:
        problems.append(f"계획 날짜 {plan['date']} ≠ 최신 거래일 {latest} — 낡은 계획")
    if not cfg.get("enabled", True) or os.path.exists(KILL):
        problems.append("킬스위치 ON")
    if problems:
        print("⛔ 주문 중단:", *problems, sep="\n  ")
        append_csv(ORDERS, [{"ts": now_kst().isoformat(timespec="seconds"), "plan_date": plan["date"],
                             "broker": broker.name, "side": "-", "code": "-", "ok": False,
                             "msg": " / ".join(problems)}], ORDER_FIELDS)
        return
    # 실시간 잔고로 수량 재계산 — 밤사이 수동 매매가 있었을 수 있다
    acct = broker.account()
    for o in plan["orders"]:
        if o["side"] == "sell":
            have = acct["positions"].get(o["code"], {}).get("shares", 0)
            o["qty"] = min(o["qty"], have)
    plan["orders"] = [o for o in plan["orders"] if o["qty"] > 0]
    rows = send(plan, broker, cfg, dry)
    mode = "실계좌" if broker.is_live else ("모의투자" if getattr(broker, "demo", False) else "DRY-RUN")
    print(f"\n[{mode}] 주문 {len(rows)}건")
    for r in rows:
        print(f"  {'✅' if r['ok'] else '⚠️'} {r['side']:<4} {r['name']}({r['code']}) {r['qty']}주 "
              f"{r['kind']} {r.get('price') or ''}  {r['msg']}")


# ══════════════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════════════
def write_decisions(decisions):
    rows = []
    for d in decisions:
        rows.append({"날짜": d["date"], "상태": d["state"], "행동": d.get("action", ""),
                     "종목코드": d["code"], "종목명": d["name"], "섹터": d.get("sector", ""),
                     "순위": d["rank"], "종가": d["price"], "수량": d.get("qty", d.get("shares", "")),
                     "수익률%": round(d["ret"] * 100, 1) if "ret" in d else "",
                     "감시가": round(d["guard_price"]) if "guard_price" in d else "",
                     "사유": d["reason"], "매수전환조건": " / ".join(d.get("need", [])),
                     "참고(좋음/나쁨)": f"{d['ref']['good']}/{d['ref']['bad']}"})
    os.makedirs(DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(DECISIONS, index=False)


def write_status(cfg, decisions, gate, plan, flags, issues, equity, dd, fills, sent, eq):
    bt = jload("data/engine_backtest.json", {})
    jsave(STATUS, {
        "asof": plan["date"], "generated": now_kst().isoformat(timespec="seconds"),
        "broker": cfg.get("broker"), "live": bool(cfg.get("live_trading")),
        "gate": gate, "flags": flags, "issues": issues,
        "equity": equity, "cash": plan["cash"], "drawdown": dd,
        "counts": {s: sum(1 for d in decisions if d["state"] == s) for s in (BUY, HOLD, SELL, WATCH)},
        "decisions": decisions, "orders": plan["orders"], "fills": fills,
        "sent": [{k: r.get(k) for k in ("side", "code", "name", "qty", "ok", "msg")} for r in sent],
        "equity_curve": eq[["date", "equity"]].to_dict("records") if len(eq) else [],
        "backtest": {k: {p: {m: v.get(m) for m in ("ann", "sharpe", "mdd", "win", "n_trades")}
                         for p, v in bt[k].items()} for k in bt if isinstance(bt.get(k), dict)
                     and "train" in bt[k]},
    })


def report(decisions, gate, plan, flags, issues, equity, fills):
    print(f"\n{'=' * 60}\n  {plan['date']} 자동매매 판단  [{plan['broker']}]\n{'=' * 60}")
    print(f"  시장: {gate['reason']}")
    if flags.get("flow_stale"):
        print(f"  ⚠️ 수급 데이터 {flags.get('flow_last')} 이후 {flags.get('flow_gap')}거래일 비어 있음"
              f" → 매수 보류 (collect_flows.py / KRX 로그인 확인)")
    for i in issues:
        print(f"  ⛔ {i}")
    if fills:
        print(f"\n  ■ 체결 (모의, {plan['date']} 종가)")
        for f in fills:
            print(f"     {f['side']} {f['code']} {f['qty']}주 @ {f.get('fill') or '-'}  {f['status']}")
    for s, icon in [(SELL, "🔴"), (BUY, "🟢"), (HOLD, "🔵"), (WATCH, "⚪")]:
        sub = [d for d in decisions if d["state"] == s]
        print(f"\n  {icon} {s} ({len(sub)})")
        for d in sub[:15]:
            extra = f" → {' / '.join(d['need'])}" if s == WATCH and d.get("need") else ""
            print(f"     {d['name']:<10} {str(d['rank'] or '-'):>4}위  {d['reason']}{extra}")
    print(f"\n  평가액 {equity:,.0f}원 · 주문계획 {len(plan['orders'])}건")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["decide", "execute", "all"], default="decide")
    ap.add_argument("--broker", help="설정 파일의 broker 덮어쓰기")
    ap.add_argument("--dry-run", action="store_true", help="파일 상태를 바꾸지 않고 주문도 보내지 않음")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    if a.broker:
        cfg["broker"] = a.broker
    if a.phase in ("decide", "all"):
        phase_decide(cfg, a.dry_run)
    if a.phase in ("execute", "all"):
        phase_execute(cfg, a.dry_run)


if __name__ == "__main__":
    main()
