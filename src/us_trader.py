"""
미국 모의 계좌 — S&P 100 추세 모델 (6개월 추세 + 52주 고가 근접 상위 10종목, 순위 밀리면 교체). 실제 돈은 움직이지 않는다.

왜 이 모델인가 (us_lab, 2016~2026, 수수료·미끄러짐 포함)
  국내에서 쓰는 단기 반전형 갈아타기는 미국 대형주에선 손실(연 -4.8%, SPY 대비 P=1.0) → 쓰지 않음.
  270개 모델 중 2015~2022 성적 1등(미리 정한 기준)인 이 추세 모델만 두 기간 모두 SPY 를 앞섰다
  (연 +19.6% · 샤프 1.06 / 2023~ 샤프 1.05, SPY 연 +15.1% · 0.89).
  단 유니버스가 '지금의' S&P 100 이라 그동안 오른 종목이 미리 들어 있다(생존편향) → 실제보다 좋게 나왔을 가능성이 크다.
  그래서 국내 계좌와 섞지 않고 따로 굴려 '앞으로의 성적'으로 확인한다.

흐름 (daily.yml, 한국 저녁 — 지난밤 끝난 미국 장 자료까지)
  1) 체결: 어제 세운 계획을 그다음 미국 장 시가로 (소수점 주식 — 국내 증권사 해외주식 소수점 매매와 같게)
  2) 판단: 마지막 미국 장 마감 기준 순위 → 내일 밤 미국 시가 주문 계획
  비용: 수수료 한쪽 0.1% + 미끄러짐 0.03% + 매도 SEC 0.003%. 세금(연 250만원 넘는 양도차익 22%)은 예상액만 표시.
  평가: 달러 · 원화(그날 환율) 둘 다. 시작할 때 원화 1,000만원을 그날 환율로 달러로 바꿨다고 본다.

  산출: data/us/{account.json, plan.json, ledger.csv, equity.csv} · docs/us.json
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import short_lab as L  # noqa: E402
import us_lab as U  # noqa: E402

DIR = "data/us"
ACC = os.path.join(DIR, "account.json")
PLAN = os.path.join(DIR, "plan.json")
LEDGER = os.path.join(DIR, "ledger.csv")
EQUITY = os.path.join(DIR, "equity.csv")
UI = "docs/us.json"
MODEL = {"id": "S:MOM:10:BAND:none", "kind": "stock", "sel": "MOM", "K": 10, "hold": "BAND", "gate": "none"}
INITIAL_KRW = 10_000_000
LEDGER_COLS = ["날짜", "종목", "구분", "수량", "가격", "금액", "비용", "손익"]
TAX_FREE_KRW = 2_500_000


def jload(p, default):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_csv(path, cols, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def settle(D, acc, plan):
    """계획(made 날 저녁 판단)을 그다음 미국 장 시가로 체결. 반환 (체결 목록, 체결일 인덱스)."""
    if not plan or not plan.get("orders"):
        return [], None
    later = np.where(D.dates > pd.Timestamp(plan["made"]))[0]
    if not len(later):
        return [], None
    t = int(later[0])
    day = str(D.dates[t].date())
    cb, cs = L.costs("us")
    codes = D.P["codes"]
    fills = []
    for o in sorted(plan["orders"], key=lambda o: o["side"] != "sell"):
        j = codes.index(o["code"]) if o["code"] in codes else None
        px = None if j is None or not np.isfinite(D.O[t, j]) else float(D.O[t, j])
        if px is None:
            fills.append({"날짜": day, "종목": o["code"], "구분": f"{o['side']}·미체결", "수량": 0})
            continue
        pos = acc["positions"]
        if o["side"] == "sell":
            p = pos.get(o["code"])
            if not p:
                continue
            q = p["qty"]
            amt = q * px
            fee = amt * cs
            pnl = amt - fee - q * p["avg"]
            acc["cash"] += amt - fee
            acc.setdefault("realized", {}).setdefault(day[:4], 0.0)
            acc["realized"][day[:4]] += pnl
            del pos[o["code"]]
            fills.append({"날짜": day, "종목": o["code"], "구분": "매도", "수량": round(q, 4), "가격": round(px, 2),
                          "금액": round(amt, 2), "비용": round(fee, 2), "손익": round(pnl, 2)})
        else:
            amt = min(o["usd"], acc["cash"] / (1 + cb))
            if amt <= 1:
                continue
            q = math.floor(amt / px * 10000) / 10000              # 소수점 4자리
            amt = q * px
            fee = amt * cb
            acc["cash"] -= amt + fee
            p = pos.get(o["code"]) or {"qty": 0.0, "avg": 0.0, "since": day}
            tot = p["qty"] + q
            p["avg"] = (p["qty"] * p["avg"] + amt + fee) / tot
            p["qty"] = tot
            pos[o["code"]] = p
            fills.append({"날짜": day, "종목": o["code"], "구분": "매수", "수량": q, "가격": round(px, 2),
                          "금액": round(amt, 2), "비용": round(fee, 2), "손익": ""})
    return fills, t


def equity_usd(D, acc, t):
    v = acc["cash"]
    for c, p in acc["positions"].items():
        j = D.P["codes"].index(c) if c in D.P["codes"] else None
        px = D.Cf[t, j] if j is not None else p["avg"]
        v += p["qty"] * (px if np.isfinite(px) else p["avg"])
    return float(v)


def make_plan(D, acc, res, T):
    target = [D.P["codes"][j] for j in L.stock_next(D, MODEL, res)]
    eq = equity_usd(D, acc, T)
    orders = [{"side": "sell", "code": c, "qty": round(p["qty"], 4), "est_price": round(float(D.Cf[T, D.P["codes"].index(c)]), 2)}
              for c, p in acc["positions"].items() if c not in target]
    slot = eq / MODEL["K"]
    for c in target:
        if c in acc["positions"]:
            continue
        orders.append({"side": "buy", "code": c, "usd": round(slot, 2),
                       "est_price": round(float(D.Cf[T, D.P["codes"].index(c)]), 2)})
    return {"made": str(D.dates[T].date()), "model": MODEL["id"], "orders": orders, "equity_usd": round(eq, 2), "target": target}


def run():
    os.makedirs(DIR, exist_ok=True)
    t0 = time.time()
    D = U.USData(U.load_panel())
    fx = U.fx_series(D.dates)
    T = D.T - 1
    today = str(D.dates[T].date())
    acc = jload(ACC, None)
    if acc is None:
        rate = float(fx[T]) if fx is not None else 1300.0
        acc = {"start": today, "initial_krw": INITIAL_KRW, "fx_start": rate, "cash": INITIAL_KRW / rate,
               "positions": {}, "last_date": None, "realized": {}}
        print(f"미국 모의 계좌를 새로 엽니다: {INITIAL_KRW:,}원 → ${acc['cash']:,.2f} (환율 {rate:,.1f}) · 시작 {today}")
    fills, tf = settle(D, acc, jload(PLAN, None))
    append_csv(LEDGER, LEDGER_COLS, fills)
    if fills:
        print(f"체결 {D.dates[tf].date()}: " + ", ".join(f"{f['종목']} {f['구분']}" for f in fills))
    s0 = int(np.searchsorted(D.dates, pd.Timestamp(acc["start"])))
    spy = D.P["bench"]["SPY"]
    rows = []
    for t in range(s0, D.T):
        d = str(D.dates[t].date())
        if acc.get("last_date") and d <= acc["last_date"]:
            continue
        e = equity_usd(D, acc, t)
        rate = float(fx[t]) if fx is not None else acc["fx_start"]
        rows.append({"날짜": d, "평가액_달러": round(e, 2), "평가액_원": round(e * rate), "환율": round(rate, 2),
                     "SPY_원": round(INITIAL_KRW * spy[t] / spy[s0] * rate / acc["fx_start"])})
    append_csv(EQUITY, ["날짜", "평가액_달러", "평가액_원", "환율", "SPY_원"], rows)
    acc["last_date"] = today
    res = L.run_stock(D, MODEL)
    plan = make_plan(D, acc, res, T)
    json.dump(plan, open(PLAN, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(acc, open(ACC, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    eq = equity_usd(D, acc, T)
    rate = float(fx[T]) if fx is not None else acc["fx_start"]
    gain_krw = {y: v * rate for y, v in (acc.get("realized") or {}).items()}
    lab = jload("docs/us_lab.json", {})
    ui = {"updated": time.strftime("%Y-%m-%d %H:%M"), "data_date": today, "start": acc["start"], "model": MODEL["id"],
          "model_desc": "S&P 100 중 6개월 추세 + 52주 고가 근접 상위 10종목 · 순위가 30위 밖으로 밀리거나, 20거래일이 지났는데 10위 밖이면 교체",
          "account": {"equity_usd": round(eq, 2), "equity_krw": round(eq * rate), "cash_usd": round(acc["cash"], 2), "fx": rate,
                      "ret_krw": round(eq * rate / INITIAL_KRW - 1, 4), "ret_usd": round(eq / (INITIAL_KRW / acc["fx_start"]) - 1, 4),
                      "spy_ret_krw": round(spy[T] / spy[s0] * rate / acc["fx_start"] - 1, 4),
                      "tax_est_krw": {y: round(max(0.0, g - TAX_FREE_KRW) * 0.22) for y, g in gain_krw.items()},
                      "positions": [{"code": c, "qty": round(p["qty"], 4), "avg": round(p["avg"], 2), "since": p.get("since"),
                                     "last": round(float(D.Cf[T, D.P["codes"].index(c)]), 2) if c in D.P["codes"] else None}
                                    for c, p in acc["positions"].items()]},
          "plan": plan, "fills_today": fills,
          "research": {k: lab.get(k) for k in ("period", "start", "by_hold", "bench", "costs")} | {
              "rules": {k: {kk: v.get(kk) for kk in ("name", "usd", "test", "usd_after_tax", "krw", "p_vs_spy", "switches")}
                        for k, v in (lab.get("rules") or {}).items()},
              "top_models": [{k: m.get(k) for k in ("id", "desc", "full", "dev", "test")} for m in (lab.get("top_models") or [])[:6]]}}
    try:
        ui["equity"] = pd.read_csv(EQUITY).tail(400).to_dict("records")
        ui["ledger"] = pd.read_csv(LEDGER).tail(40).iloc[::-1].fillna("").to_dict("records")
    except (OSError, ValueError):
        ui.setdefault("equity", [])
        ui.setdefault("ledger", [])
    from short_trader import clean
    json.dump(clean(ui), open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"미국 모의 {today}: ${eq:,.2f} = {eq * rate:,.0f}원 · 보유 {len(acc['positions'])} · 내일 밤 주문 {len(plan['orders'])}건 "
          f"({', '.join(o['side'] + ' ' + o['code'] for o in plan['orders'][:12])}) · {time.time() - t0:.0f}s")
    return ui


if __name__ == "__main__":
    run()
