"""
자동매매 실행기 — 판단 엔진의 매수/유지/매도/관망 결정을 실제 주문으로 바꾼다.

단계
  decide  (평일 17:00, daily.yml 뒤)  지표 계산 → 4상태 판단 → 주문 계획 data/autotrade/plan.json
          · paper  : 어제 낸 모의 주문을 오늘 종가로 체결 → 오늘 판단 → 새 모의 주문 접수
          · kis_*  : 계획만 만든다 (장이 닫혀 있다)
  execute (다음 거래일 08:40, autotrade.yml)  계획을 읽어 증권사에 주문 (kis_* 전용)
          매도 먼저(시장가) → 매수(전일종가+1% 지정가). 수량은 실시간 잔고로 다시 계산.
  all     decide + execute 를 한 번에 (장중 로컬 실행용)

승인 모드 (execution.approval)
  auto   : 위 흐름 그대로 자동 주문
  manual : 계획만 만들고 '승인 대기' 알림 → 사용자가 GitHub Actions 'autotrade-control'에서
           approve 를 눌러야 주문 (Trade Ideas·Portfolio123 의 '신호 + 확인' 방식).
           --skip 005930,000660 으로 고른 종목만 빼고 승인할 수 있다.

원격 제어 파일 (control.yml 이 커밋한다)
  config/KILL  : 모든 주문 중단 (매도 포함)
  config/PAUSE : 신규 매수만 중단, 손절·익절 매도는 계속 (Freqtrade /pause 와 같은 단계)

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
  python src/autotrader.py --phase execute --approve --skip 005930   # 승인 모드에서 승인
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
import features as F                                     # noqa: E402
from decision_engine import decide, load_news, BUY, SELL, HOLD, WATCH   # noqa: E402
from brokers import make_broker, round_tick, PaperBroker  # noqa: E402
import notify                                             # noqa: E402

CFG = "config/autotrade.yaml"
KILL = "config/KILL"
PAUSE = "config/PAUSE"
DIR = "data/autotrade"
LEDGER = f"{DIR}/ledger.json"
PLAN = f"{DIR}/plan.json"
DECISIONS = f"{DIR}/decisions.csv"
ORDERS = f"{DIR}/orders.csv"
EQUITY = f"{DIR}/equity.csv"
JOURNAL = f"{DIR}/journal.csv"
PLANS = f"{DIR}/plans"                 # 날짜별 계획 보관 — 지난 판단을 고쳐 쓰지 않는다 (Collective2 관행)
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


JOURNAL_FIELDS = ["date", "broker", "side", "code", "name", "qty", "price", "reason", "rank", "regime",
                  "band_p10", "band_p90", "ret_pct", "held_days", "ref_good", "ref_bad", "approved_by"]


def journal(plan, rows, decisions, gate, broker_name, approved_by):
    """자동 매매일지 — 주문마다 '왜 샀나/왜 팔았나'와 그때의 근거 숫자를 남긴다 (Tradervue·Edgewonk 방식)."""
    by = {d["code"]: d for d in decisions or []}
    bands = jload("docs/analytics.json", {}).get("stocks", {})
    out = []
    for r in rows:
        if not r.get("ok"):
            continue
        d = by.get(r["code"], {})
        b = (bands.get(r["code"], {}) or {}).get("band") or {}
        out.append({"date": plan["date"], "broker": broker_name, "side": r["side"], "code": r["code"],
                    "name": r["name"], "qty": r["qty"], "price": r.get("price") or r.get("ref_price"),
                    "reason": r.get("reason", ""), "rank": d.get("rank"), "regime": gate.get("regime") if gate else "",
                    "band_p10": (b.get("ret") or [None])[0], "band_p90": (b.get("ret") or [None] * 5)[-1],
                    "ret_pct": round(d["ret"] * 100, 1) if "ret" in d else "",
                    "held_days": d.get("held_days", ""), "ref_good": (d.get("ref") or {}).get("good", ""),
                    "ref_bad": (d.get("ref") or {}).get("bad", ""), "approved_by": approved_by})
    append_csv(JOURNAL, out, JOURNAL_FIELDS)


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
    """증권사 잔고가 진실. 원장은 잔고에 맞춰 고친다. 사라진 종목은 '_exits'에 청산일을 남긴다."""
    exits = dict(ledger.get("_exits", {}))
    for c in ledger:
        if c != "_exits" and c not in acct["positions"]:
            exits[c] = asof
    new = {"_exits": {c: d for c, d in exits.items()
                      if (pd.Timestamp(asof) - pd.Timestamp(d)).days <= 120}}
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
    if os.path.exists(PAUSE):
        flags["pause_entries"] = True          # 매도는 그대로, 신규 매수만 관망으로
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

    flags["recent_exits"] = {c: int(np.searchsorted(panel["dates"].values, np.datetime64(d)))
                             for c, d in ledger.get("_exits", {}).items()}
    decisions, gate = decide(t, panel, feats, pos, cfg, news=load_news(), flags=flags)
    plan = build_plan(decisions, acct, equity, cfg, asof)
    plan["issues"] = issues
    approval = cfg.get("execution", {}).get("approval", "auto")
    plan["approval"] = approval
    plan["status"] = "승인 대기" if (approval == "manual" and plan["orders"]) else "자동"
    jsave(PLAN, plan)
    if not dry:
        jsave(LEDGER, ledger)
        os.makedirs(PLANS, exist_ok=True)
        shutil.copyfile(PLAN, f"{PLANS}/{asof_s}.json")

    sent = []
    if isinstance(broker, PaperBroker):
        if issues:
            print("⛔ 안전장치로 모의 주문도 보내지 않음:", *issues, sep="\n  ")
        elif approval == "manual":
            print(f"✋ 승인 모드 — 주문 {len(plan['orders'])}건은 승인 후 전송 (autotrade-control → approve)")
        else:
            sent = send(plan, broker, cfg, dry)
            if not dry:
                journal(plan, sent, decisions, gate, "paper", "auto")
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
    if not dry:
        alert_decide(cfg, plan, decisions, gate, issues, fills)
    return plan


def alert_decide(cfg, plan, decisions, gate, issues, fills):
    """알림 — 어떤 사건을 받을지는 alerts.on 에서 고른다."""
    url = (cfg.get("alerts") or {}).get("page_url") or None
    if issues:
        notify.send(cfg, "safety", "\n".join(issues), "⛔ 자동매매 안전장치", urgent=True)
    if fills:
        notify.send(cfg, "fill", notify.fmt_fills(fills), "체결 (모의)")
    sells = [d for d in decisions if d["state"] == SELL]
    if sells:
        notify.send(cfg, "sell", "\n".join(f"{d['action']} {d['name']} — {d['reason']}" for d in sells),
                    "🔵 매도 판단", urgent=True)
    buys = [d for d in decisions if d["state"] == BUY]
    if buys:
        notify.send(cfg, "buy", "\n".join(f"{d['name']} {d['rank']}위" for d in buys), "🔴 매수 판단")
    if plan.get("status") == "승인 대기":
        notify.send(cfg, "approval", f"주문 {len(plan['orders'])}건 승인 대기 — GitHub Actions "
                    f"'autotrade-control' 에서 approve (빼고 싶은 종목코드는 skip 칸에)", "✋ 승인 필요",
                    urgent=True)
    notify.send(cfg, "daily", notify.fmt_decisions(plan, decisions, gate, url), "오늘의 판단")


def phase_execute(cfg, dry, approve=False, skip=()):
    plan = jload(PLAN, None)
    if not plan:
        print("계획 없음 — decide 먼저 실행"); return
    manual = cfg.get("execution", {}).get("approval", "auto") == "manual"
    if manual and not approve:
        print(f"✋ 승인 모드 — 계획 {plan['date']} 주문 {len(plan['orders'])}건이 승인을 기다립니다. "
              "autotrade-control 워크플로우에서 approve 를 실행하세요.")
        return
    broker = make_broker(cfg)
    if isinstance(broker, PaperBroker) and not manual:
        print("paper + 자동 모드는 decide 단계에서 주문까지 끝낸다 — execute 불필요"); return
    if skip:
        dropped = [o for o in plan["orders"] if o["code"] in skip]
        plan["orders"] = [o for o in plan["orders"] if o["code"] not in skip]
        for o in dropped:
            print(f"  ⤫ 사용자가 제외: {o['side']} {o['name']}({o['code']})")
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
    if isinstance(broker, PaperBroker):          # 승인 모드의 모의 계좌 — 다음 decide 때 종가 체결
        rows = send(plan, broker, cfg, dry)
        if not dry:
            broker.save()
            journal(plan, rows, jload("docs/autotrade.json", {}).get("decisions"), None, "paper",
                    "approve" if manual else "auto")
        print(f"[모의] 승인된 주문 {sum(1 for r in rows if r.get('ok'))}건 접수")
        notify.send(cfg, "fill", f"승인된 모의 주문 {len(rows)}건 접수", "✅ 승인 처리")
        return
    # 실시간 잔고로 수량 재계산 — 밤사이 수동 매매가 있었을 수 있다
    acct = broker.account()
    for o in plan["orders"]:
        if o["side"] == "sell":
            have = acct["positions"].get(o["code"], {}).get("shares", 0)
            o["qty"] = min(o["qty"], have)
    plan["orders"] = [o for o in plan["orders"] if o["qty"] > 0]
    rows = send(plan, broker, cfg, dry)
    if not dry:
        journal(plan, rows, jload("docs/autotrade.json", {}).get("decisions"), None,
                broker.name + ("_live" if broker.is_live else ""), "approve" if manual else "auto")
        bad = [r for r in rows if not r.get("ok")]
        notify.send(cfg, "fill", "\n".join(f"{'✅' if r.get('ok') else '⚠️'} {r['side']} {r['name']} {r['qty']}주 "
                                           f"{r.get('msg', '')}" for r in rows), "주문 전송", urgent=bool(bad))
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
        "approval": plan.get("approval", "auto"), "plan_status": plan.get("status", ""),
        "paused": bool(flags.get("pause_entries")),
        "features": (yaml.safe_load(open("config/features.yaml", encoding="utf-8"))
                     if os.path.exists("config/features.yaml") else {}),
        "journal": (pd.read_csv(JOURNAL).tail(40).iloc[::-1].fillna("").to_dict("records")
                    if os.path.exists(JOURNAL) else []),
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
    for s, icon in [(SELL, "🔵"), (BUY, "🔴"), (HOLD, "⚫"), (WATCH, "⚪")]:
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
    ap.add_argument("--approve", action="store_true", help="승인 모드에서 오늘 계획을 승인")
    ap.add_argument("--skip", default="", help="승인할 때 뺄 종목코드 (쉼표 구분)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    if a.broker:
        cfg["broker"] = a.broker
    if a.phase in ("decide", "all"):
        phase_decide(cfg, a.dry_run)
    if a.phase in ("execute", "all"):
        skip = tuple(x.strip().zfill(6) for x in a.skip.split(",") if x.strip())
        phase_execute(cfg, a.dry_run, a.approve, skip)


if __name__ == "__main__":
    main()
