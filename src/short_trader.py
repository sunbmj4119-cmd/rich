"""
단기 모의투자 — 매일 저녁 (daily.yml, 자료 갱신 뒤) 실행. 실제 돈은 움직이지 않는다.

  1) 체결: 어제 저녁에 세운 계획을 오늘 실제 가격으로 체결한다
           (시가 주문 = 오늘 시가, 종가 주문 = 오늘 종가 · 주식 수는 정수 · 수수료·세금·미끄러짐 포함 = 연구와 같은 비용)
  2) 매일 시뮬레이션: 후보 모델 전부(short_lab)를 2018년부터 오늘까지 다시 돌려 순위표를 만든다
  3) 진화(선택): 채택 규칙(기본: 주 1회 · 최근 250일 샤프 1등 · 느린 후보군)으로 내일 따라갈 모델을 정한다
           — 후보에는 short_evolve.py 가 2시간마다 새로 만든 진화 모델도 들어간다 (등록 뒤 성적만 센다 → 약 6개월 뒤부터 선택 가능)
           — 검증 결과 '매일 1등으로 갈아타기'는 288가지 방식 중 19~29%만 지수 보유를 이겼다 (docs/STRATEGY.md 3-6-5)
           — 다른 선택 규칙들도 모의투자 시작일부터 나란히 기록해(그림자) 앞으로의 성적으로 비교한다
  4) 계획: 내일 주문 → data/short/plan.json (나중에 토스 연결 시 서버가 이 파일을 받아 보낼 수 있는 형식)

산출: data/short/{account.json, plan.json, ledger.csv, equity.csv, selection.csv} · docs/short.json
설정: config/autotrade.yaml 의 short: (enabled · initial_cash · buy_buffer · rule)
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
import settings  # noqa: E402
import short_evolve as SE  # noqa: E402
import short_lab as L  # noqa: E402
import short_panel as SP  # noqa: E402

DIR = "data/short"
ACC = os.path.join(DIR, "account.json")
PLAN = os.path.join(DIR, "plan.json")
LEDGER = os.path.join(DIR, "ledger.csv")
EQUITY = os.path.join(DIR, "equity.csv")
SELECTION = os.path.join(DIR, "selection.csv")
UI = "docs/short.json"
LAB_UI = "docs/short_lab.json"
LEDGER_COLS = ["날짜", "시점", "종목코드", "종목명", "구분", "수량", "가격", "금액", "비용", "손익", "모델"]
ETF_NAME = {"069500": "KODEX 200", "114800": "KODEX 인버스"}
ETF_CODE = [SP.ETF_CODES["IDX"], SP.ETF_CODES["INV"]]


def names():
    p = pd.read_csv(SP.PRICES, dtype={"종목코드": str}, usecols=["종목코드", "종목명"]).drop_duplicates("종목코드", keep="last")
    return dict(zip(p["종목코드"].str.zfill(6), p["종목명"]))


def cfg_short(cfg):
    sc = dict(cfg.get("short") or {})
    sc.setdefault("enabled", True)
    sc.setdefault("initial_cash", 10_000_000)
    sc.setdefault("buy_buffer", 0.02)
    sc["rule"] = {**L.LIVE_RULE, **(sc.get("rule") or {})}
    return sc


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


def clean(o):
    """JSON 에 NaN/inf 가 들어가면 브라우저가 읽지 못한다 → null 로."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


def load_json(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════
# 1) 체결
# ══════════════════════════════════════════════════════════════
def settle(D, acc, plan, nm):
    """계획의 '다음 거래일'이 자료에 들어왔으면 그날 가격으로 체결. 반환: 체결 기록."""
    if not plan or not plan.get("orders"):
        return [], None
    made = pd.Timestamp(plan["made"])
    later = np.where(D.dates > made)[0]
    if not len(later):
        return [], None
    t = int(later[0])
    day = str(D.dates[t].date())
    cbS, csS = L.costs("stock")
    cbE, csE = L.costs("etf")
    etf_ok = D.legs["source"] == "etf"
    fills = []
    pos, etf = acc["positions"], acc["etf"]

    def price(o):
        if o["kind"] == "stock":
            j = D.P["codes"].index(o["code"]) if o["code"] in D.P["codes"] else None
            return None if j is None or np.isnan(D.O[t, j]) else float(D.O[t, j])
        if not etf_ok:
            return None
        j = ETF_CODE.index(o["code"])
        return float(D.Op[t, j] if o["session"] == "open" else D.Cp[t, j])

    order = sorted(plan["orders"], key=lambda o: (o["session"] != "open", o["side"] != "sell"))
    for o in order:
        px = price(o)
        book = pos if o["kind"] == "stock" else etf
        cb, cs = (cbS, csS) if o["kind"] == "stock" else (cbE, csE)
        name = o.get("name") or nm.get(o["code"], o["code"])
        if px is None:
            fills.append({"날짜": day, "시점": o["session"], "종목코드": o["code"], "종목명": name, "구분": f"{o['side']}·미체결",
                          "수량": 0, "가격": "", "금액": 0, "비용": 0, "손익": "", "모델": plan.get("model")})
            continue
        if o["side"] == "sell":
            have = int((book.get(o["code"]) or {}).get("qty", 0))
            q = min(int(o["qty"]), have)
            if q <= 0:
                continue
            amt = q * px
            fee = amt * cs
            avg = book[o["code"]]["avg"]
            pnl = amt - fee - q * avg
            acc["cash"] += amt - fee
            book[o["code"]]["qty"] = have - q
            if book[o["code"]]["qty"] <= 0:
                del book[o["code"]]
            fills.append({"날짜": day, "시점": o["session"], "종목코드": o["code"], "종목명": name, "구분": "매도", "수량": q,
                          "가격": round(px, 2), "금액": round(amt), "비용": round(fee), "손익": round(pnl), "모델": plan.get("model")})
        else:
            q = int(o["qty"])
            q = min(q, int(acc["cash"] // (px * (1 + cb))))
            if q <= 0:
                continue
            amt = q * px
            fee = amt * cb
            acc["cash"] -= amt + fee
            cur = book.get(o["code"]) or {"qty": 0, "avg": 0.0, "since": day, "name": name}
            tot = cur["qty"] + q
            cur["avg"] = (cur["qty"] * cur["avg"] + amt + fee) / tot        # 평단에 매수 비용 포함
            cur["qty"] = tot
            cur.setdefault("since", day)
            book[o["code"]] = cur
            fills.append({"날짜": day, "시점": o["session"], "종목코드": o["code"], "종목명": name, "구분": "매수", "수량": q,
                          "가격": round(px, 2), "금액": round(amt), "비용": round(fee), "손익": "", "모델": plan.get("model")})
    return fills, t


def equity_at(D, acc, t):
    v = acc["cash"]
    for c, p in acc["positions"].items():
        if c in D.P["codes"]:
            v += p["qty"] * D.Cf[t, D.P["codes"].index(c)]
        else:
            v += p["qty"] * p["avg"]
    for c, p in acc["etf"].items():
        j = ETF_CODE.index(c)
        v += p["qty"] * (D.Cp[t, j] if D.legs["source"] == "etf" else p["avg"])
    return float(v)


# ══════════════════════════════════════════════════════════════
# 3) 계획
# ══════════════════════════════════════════════════════════════
def make_plan(D, m, res, acc, sc, nm, t_last):
    T = t_last
    eq = equity_at(D, acc, T)
    cbS, _ = L.costs("stock")
    cbE, _ = L.costs("etf")
    buf = sc["buy_buffer"]
    orders, notes = [], []
    codes = D.P["codes"]
    if m["kind"] == "stock":
        target = [codes[j] for j in L.stock_next(D, m, res)]
        for c, p in acc["positions"].items():
            if c not in target:
                orders.append({"session": "open", "kind": "stock", "code": c, "name": p.get("name") or nm.get(c, c),
                               "side": "sell", "qty": int(p["qty"]), "est_price": round(float(D.Cf[T, codes.index(c)]) if c in codes else p["avg"], 2)})
        for c, p in acc["etf"].items():
            orders.append({"session": "open", "kind": "etf", "code": c, "name": ETF_NAME[c], "side": "sell", "qty": int(p["qty"]),
                           "est_price": round(float(D.Cp[T, ETF_CODE.index(c)]), 2)})
        slot = eq / m["K"]
        for c in target:
            if c in acc["positions"]:
                continue
            px = float(D.Cf[T, codes.index(c)])
            q = int(math.floor(slot / (px * (1 + buf) * (1 + cbS)))) if px > 0 else 0
            if q <= 0:
                notes.append(f"{nm.get(c, c)}: 1주 가격({px:,.0f}원)이 한 종목 몫({slot:,.0f}원)보다 커서 건너뜀")
                continue
            orders.append({"session": "open", "kind": "stock", "code": c, "name": nm.get(c, c), "side": "buy", "qty": q,
                           "est_price": round(px, 2)})
    else:
        wo, wc = L.etf_targets(D, m["rule"], pad=1)
        wo, wc = wo[T + 1], wc[T + 1]
        for c, p in acc["positions"].items():
            orders.append({"session": "open", "kind": "stock", "code": c, "name": p.get("name") or nm.get(c, c), "side": "sell",
                           "qty": int(p["qty"]), "est_price": round(float(D.Cf[T, codes.index(c)]) if c in codes else p["avg"], 2)})
        held = {c: int(p["qty"]) for c, p in acc["etf"].items()}
        after = dict(held)
        for session, w in (("open", wo), ("close", wc)):
            if np.isnan(w[0]):
                continue
            for j, c in enumerate(ETF_CODE):
                px = float(D.Cp[T, j])
                tgt = int(math.floor(w[j] * eq / (px * (1 + buf) * (1 + cbE)))) if w[j] > 0 else 0
                cur = after.get(c, 0)
                cur_w = cur * px / eq
                if tgt == cur or (tgt > 0 and cur > 0 and abs(w[j] - cur_w) <= 0.05):
                    continue
                side = "buy" if tgt > cur else "sell"
                orders.append({"session": session, "kind": "etf", "code": c, "name": ETF_NAME[c], "side": side,
                               "qty": abs(tgt - cur), "est_price": round(px, 2)})
                after[c] = tgt
    return {"made": str(D.dates[T].date()), "model": m["id"], "model_desc": L.describe(m), "orders": orders, "notes": notes,
            "equity_est": round(eq)}


# ══════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════
def run(cfg=None, now=None):
    cfg = cfg or settings.load()
    sc = cfg_short(cfg)
    if not sc["enabled"]:
        print("단기 모의투자 꺼짐 (config short.enabled: false)")
        return None
    os.makedirs(DIR, exist_ok=True)
    t0 = time.time()
    D = L.Data()
    nm = names()
    T = D.T - 1
    today = str(D.dates[T].date())
    acc = load_json(ACC, None)
    if acc is None:
        acc = {"start": today, "cash": float(sc["initial_cash"]), "initial_cash": float(sc["initial_cash"]),
               "positions": {}, "etf": {}, "last_date": None, "model": None, "selected_at": None}
        print(f"단기 모의 계좌를 새로 엽니다: {sc['initial_cash']:,.0f}원 · 시작 {today}")
    plan = load_json(PLAN, None)
    # 1) 체결
    fills, t_fill = settle(D, acc, plan, nm)
    append_csv(LEDGER, LEDGER_COLS, fills)
    if fills:
        print(f"체결 {D.dates[t_fill].date()}: " + ", ".join(f"{f['종목명']} {f['구분']} {f['수량']}주" for f in fills))
    # 하루씩 평가액 기록 (빠진 날도 채움)
    eq_rows = []
    last = acc.get("last_date")
    s0 = int(np.searchsorted(D.dates, pd.Timestamp(acc["start"])))
    b0 = D.Cp[s0, 0]
    for t in range(s0, D.T):
        d = str(D.dates[t].date())
        if last and d <= last:
            continue
        eq_rows.append({"날짜": d, "평가액": round(equity_at(D, acc, t)), "현금": round(acc["cash"]),
                        "지수ETF": round(float(D.Cp[t, 0] / b0 * acc["initial_cash"])), "모델": acc.get("model") or ""})
    append_csv(EQUITY, ["날짜", "평가액", "현금", "지수ETF", "모델"], eq_rows)
    acc["last_date"] = today
    # 2) 매일 시뮬레이션
    models = L.stock_models() + L.etf_models() + SE.models_from_registry()   # + 쉬지 않고 진화한 후보
    recs = [L.run(D, m, record=True) for m in models]
    R = np.c_[[L.rets(r["E"]) for r in recs]].T
    for i, mm in enumerate(models):                    # 진화 후보는 등록일 다음 날부터의 성적만 센다
        if mm.get("born"):
            R[:int(np.searchsorted(D.dates, pd.Timestamp(mm["born"]), side="right")), i] = np.nan
    rule = sc["rule"]
    pool = L.pool_index(models, rule["pool"])
    if D.legs["source"] != "etf":                      # 실제 ETF 가격이 없으면 ETF 모델은 체결할 수 없어 뺀다
        pool = [i for i in pool if models[i]["kind"] == "stock"]
    start = max(1, s0)
    ch = L.choose(R[:, pool], start, **rule)
    pick = ch[T]
    if pick is None:                                   # 자료가 모자라면 최근 250일 샤프 1등
        sc_ = L.trailing_score(R[:, pool], T, 250, "sharpe")
        pick = int(np.argmax(np.nan_to_num(sc_, nan=-np.inf)))
    if isinstance(pick, list):
        pick = pick[0]
    mi = pool[pick]
    m = models[mi]
    changed = m["id"] != acc.get("model")
    if changed:
        acc["model"], acc["selected_at"] = m["id"], today
    changed = changed or acc.get("selected_at") == today          # 같은 날 다시 돌려도 '오늘 바뀜' 유지
    # 순위표 (후보군 안, 최근 250일 샤프 순)
    sh250 = L.trailing_score(R, T, 250, "sharpe")
    sh60 = L.trailing_score(R, T, 60, "sharpe")
    r20 = L.trailing_score(R, T, 20, "ret")
    r60 = L.trailing_score(R, T, 60, "ret")
    board = []
    for i in pool:
        if not np.isfinite(sh250[i]):                  # 아직 시험 기간인 진화 후보는 docs/short_evo.json 에서 보여 준다
            continue
        st = L.stats(recs[i]["E"], D.dates)
        board.append({"id": models[i]["id"], "desc": L.describe(models[i]), "kind": models[i]["kind"],
                      "evolved": bool(models[i].get("born")),
                      "sh250": round(float(sh250[i]), 3), "sh60": round(float(sh60[i]), 3),
                      "r20": round(float(r20[i]), 4), "r60": round(float(r60[i]), 4),
                      "cagr": st["cagr"], "mdd": st["mdd"], "chosen": i == mi})
    board.sort(key=lambda x: -x["sh250"])
    sel_row = {"날짜": today, "모델": m["id"], "설명": L.describe(m), "최근250일샤프": round(float(sh250[mi]), 3), "바뀜": int(changed)}
    try:                                               # 같은 날 다시 돌리면 그날 줄을 바꿔 쓴다
        old = pd.read_csv(SELECTION)
        old = old[old["날짜"].astype(str) != today]
        pd.concat([old, pd.DataFrame([sel_row])]).to_csv(SELECTION, index=False)
    except (OSError, ValueError):
        append_csv(SELECTION, list(sel_row), [sel_row])
    # 그림자 규칙: 모의투자 시작일부터 나란히 (같은 비용, 소수 주식 수)
    shadow = []
    for key, (name, r_) in L.SHADOW_RULES.items():
        if start >= T:
            shadow.append({"key": key, "name": name, "ret": 0.0, "days": 0})
            continue
        E, sw, _ = L.apply_rule(D, models, recs, R, r_, start)
        shadow.append({"key": key, "name": name, "ret": round(float(E[T] / E[start - 1] - 1), 4), "switches": sw,
                       "days": T - start + 1})
    # 3) 계획
    res = recs[mi]
    plan = make_plan(D, m, res, acc, sc, nm, T)
    json.dump(clean(plan), open(PLAN, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(acc, open(ACC, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    eq = equity_at(D, acc, T)
    every = rule["every"]
    days_in = (T - start) % every if T >= start else 0
    ui = {"updated": time.strftime("%Y-%m-%d %H:%M"), "data_date": today, "start": acc["start"],
          "etf_source": D.legs["source"], "rule": rule, "rule_name": L.rule_name(rule),
          "next_reselect_days": int(every - days_in) if every > 1 else 1,
          "account": {"equity": round(eq), "cash": round(acc["cash"]), "initial": acc["initial_cash"],
                      "ret": round(eq / acc["initial_cash"] - 1, 4),
                      "bench_ret": round(float(D.Cp[T, 0] / b0 - 1), 4),
                      "positions": [{"code": c, "name": p.get("name") or nm.get(c, c), "qty": p["qty"], "avg": round(p["avg"], 2),
                                     "last": round(float(D.Cf[T, D.P["codes"].index(c)]), 2) if c in D.P["codes"] else None,
                                     "since": p.get("since")} for c, p in acc["positions"].items()] +
                                   [{"code": c, "name": ETF_NAME[c], "qty": p["qty"], "avg": round(p["avg"], 2),
                                     "last": round(float(D.Cp[T, ETF_CODE.index(c)]), 2), "since": p.get("since")}
                                    for c, p in acc["etf"].items()]},
          "model": {"id": m["id"], "desc": L.describe(m), "selected_at": acc["selected_at"], "changed_today": changed,
                    "sh250": round(float(sh250[mi]), 3)},
          "plan": plan, "leaderboard": board[:20], "pool_size": len(pool), "shadow": shadow,
          "fills_today": fills}
    try:
        e = pd.read_csv(EQUITY)
        ui["equity"] = e.tail(400).to_dict("records")
    except (OSError, ValueError):
        ui["equity"] = []
    try:
        lg = pd.read_csv(LEDGER, dtype={"종목코드": str})
        ui["ledger"] = lg.tail(40).iloc[::-1].fillna("").to_dict("records")
    except (OSError, ValueError):
        ui["ledger"] = []
    json.dump(clean(ui), open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    # 알림
    try:
        import notify
        lines = [f"단기 모의 · {today} · 평가액 {eq:,.0f}원 ({ui['account']['ret'] * 100:+.2f}%, 지수 {ui['account']['bench_ret'] * 100:+.2f}%)",
                 f"모델: {L.describe(m)}" + (" (오늘 바뀜)" if changed else "")]
        for o in plan["orders"]:
            lines.append(f"{'🔴 매수' if o['side'] == 'buy' else '🔵 매도'} {o['name']} {o['qty']}주 ({'시가' if o['session'] == 'open' else '종가'})")
        notify.send(cfg, "daily", "\n".join(lines), title="단기 모의투자")
    except Exception as ex:                                # 알림 실패가 기록을 막지 않게
        print(f"  (알림 생략: {ex})")
    print(f"단기 모의투자 {today}: 평가액 {eq:,.0f}원 · 모델 {m['id']}{' (바뀜)' if changed else ''} · 내일 주문 {len(plan['orders'])}건 "
          f"· 후보 {len(pool)}개 · {time.time() - t0:.0f}s")
    for o in plan["orders"]:
        print(f"  {o['session']:5s} {o['side']:4s} {o['name']} {o['qty']}주 (어제 종가 {o['est_price']:,})")
    for n in plan.get("notes", []):
        print(f"  참고: {n}")
    return ui


if __name__ == "__main__":
    run()
