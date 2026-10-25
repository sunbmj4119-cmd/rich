"""
판단 엔진 백테스트 — decision_engine.decide() 를 **그대로** 매일 호출해 계좌를 굴린다.
-> data/engine_backtest.json

strategy_lab.simulate 와 다른 점
  · 현금·주식수 회계 (비중 근사가 아님). 매수 금액 = 그날 평가액 ÷ max_positions
  · 판단 코드가 라이브 실행기와 동일 → 검증한 규칙과 주문 내는 규칙이 어긋날 수 없다
체결 가정: t일 장 마감 후 판단 → t+1일 종가 체결, 편도 비용 0.15% (왕복 0.3%)
비교: 급락장 게이트 OFF(= 기존 signal.py 규칙) / ON(= 채택안) / 시장(100종목 등가중)

사용: python src/engine_backtest.py
"""
import copy
import json
import sys

import numpy as np

sys.path.insert(0, "src")
import features as F                    # noqa: E402
import settings                                # noqa: E402
from decision_engine import decide, SELL, BUY   # noqa: E402

CFG = "config/autotrade.yaml"
OUT = "data/engine_backtest.json"
SPLIT = "2024-01-01"
WARMUP = 260
COST = 0.0015


def run(panel, feats, cfg, d0, d1, detail=False):
    """detail=True 면 일별 평가액·노출도·거래 상세(종목·날짜·MAE/MFE)를 raw 로 함께 돌려준다 (tearsheet.py 용)."""
    close = panel["close"].values
    codes = panel["codes"]
    ci = {c: i for i, c in enumerate(codes)}
    cash, pos, pending = 1.0, {}, []
    curve, expo, trades, states = [], [], [], {"매수": 0, "유지": 0, "매도": 0, "관망": 0}
    gate_closed = 0
    exits = {}                                   # 종목 → 마지막 전량청산일 (재진입 대기 규칙용)
    for t in range(d0, d1):
        # 1) 어제 판단한 주문을 오늘 종가로 체결
        eq_now = cash + sum(q["shares"] * close[t, ci[c]] for c, q in pos.items()
                            if np.isfinite(close[t, ci[c]]))
        for o in pending:
            px = close[t, ci[o["code"]]]
            if not np.isfinite(px):
                continue
            if o["side"] == "sell" and o["code"] in pos:
                q = pos[o["code"]]
                sh = q["shares"] * o["fraction"]
                cash += sh * px * (1 - COST)
                trades.append({"ret": px * (1 - COST) / (q["avg"] * (1 + COST)) - 1,
                               "held": t - q["entry_t"], "why": o["why"], "frac": o["fraction"],
                               "code": o["code"], "t0": q["entry_t"], "t1": t, "w": q["w0"],
                               "mae": q["low"] / q["avg"] - 1, "mfe": q["peak"] / q["avg"] - 1})
                q["shares"] -= sh
                if o["fraction"] < 1:
                    q["partial_done"] = True
                if q["shares"] <= 1e-12 or o["fraction"] >= 1:
                    pos.pop(o["code"])
                    exits[o["code"]] = t
        for o in pending:
            px = close[t, ci[o["code"]]]
            if o["side"] != "buy" or not np.isfinite(px) or o["code"] in pos:
                continue
            amt = min(eq_now / cfg["strategy"]["max_positions"], cash)
            if amt <= 1e-9:
                continue
            sh = amt / (px * (1 + COST))
            cash -= amt
            pos[o["code"]] = {"shares": sh, "avg": px, "entry_t": t, "peak": px, "low": px,
                              "partial_done": False, "w0": amt / eq_now if eq_now else 0}
        pending = []
        # 2) 고점 갱신 & 평가
        for c, q in pos.items():
            px = close[t, ci[c]]
            if np.isfinite(px):
                q["peak"] = max(q["peak"], px)
                q["low"] = min(q["low"], px)
        eq = cash + sum(q["shares"] * (close[t, ci[c]] if np.isfinite(close[t, ci[c]]) else q["avg"])
                        for c, q in pos.items())
        curve.append(eq)
        expo.append(1 - cash / eq if eq else 0)
        if t >= d1 - 1:
            break
        # 3) 오늘 장 마감 정보로 판단 → 내일 체결
        dec, gate = decide(t, panel, feats, pos, cfg, flags={"recent_exits": exits})
        gate_closed += not gate["open"]
        for d in dec:
            states[d["state"]] += 1
            if d["state"] == SELL:
                pending.append({"side": "sell", "code": d["code"], "fraction": d["fraction"],
                                "why": ("부분익절" if d["action"] == "부분익절"
                                        else d["reason"].split(" ")[0])})
            elif d["state"] == BUY:
                pending.append({"side": "buy", "code": d["code"]})
    c = np.array(curve)
    dr = np.diff(c) / c[:-1]
    yrs = len(c) / 252
    tr = np.array([x["ret"] for x in trades]) if trades else np.array([0.0])
    why = {}
    for x in trades:
        why[x["why"]] = why.get(x["why"], 0) + 1
    return {
        "ann": round((c[-1] ** (1 / yrs) - 1) * 100, 1),
        "total": round((c[-1] - 1) * 100, 1),
        "sharpe": round(float(dr.mean() / (dr.std() + 1e-12) * np.sqrt(252)), 2),
        "mdd": round(float((c / np.maximum.accumulate(c) - 1).min() * 100), 1),
        "n_trades": len(trades), "win": round(float((tr > 0).mean() * 100), 1),
        "avg": round(float(tr.mean() * 100), 2),
        "exits": why, "states": states,
        "gate_closed_days": int(gate_closed),
        "curve": [round(float(x), 4) for x in c[::5]],
        **({"raw": {"curve": c, "expo": np.array(expo), "trades": trades, "d0": d0}} if detail else {}),
    }


def market(panel, d0, d1):
    idx = panel["market"]["idx"].values[d0:d1]
    c = idx / idx[0]
    dr = np.diff(c) / c[:-1]
    return {"ann": round((c[-1] ** (252 / len(c)) - 1) * 100, 1),
            "sharpe": round(float(dr.mean() / dr.std() * np.sqrt(252)), 2),
            "mdd": round(float((c / np.maximum.accumulate(c) - 1).min() * 100), 1)}


def main():
    cfg = settings.load(CFG)
    panel, feats = F.build_all()
    feats = F.apply_strategy(panel, feats, cfg["strategy"])
    dates = panel["dates"]
    nD = len(dates)
    split = int(np.searchsorted(dates.values, np.datetime64(SPLIT)))
    off = copy.deepcopy(cfg)
    off["strategy"]["crash_gate"] = False
    out = {"train": [str(dates[WARMUP].date()), str(dates[split - 1].date())],
           "test": [str(dates[split].date()), str(dates[-1].date())], "cost_oneway": COST}
    print(f"train {out['train']} · test {out['test']}")
    for nm, c in [("기존 규칙 (게이트 OFF)", off), ("판단 엔진 (게이트 ON)", cfg)]:
        tr = run(panel, feats, c, WARMUP, split)
        te = run(panel, feats, c, split, nD)
        out[nm] = {"train": tr, "test": te}
        for k, r in [("train", tr), ("test", te)]:
            print(f"  {nm:<18} {k:<5} 연{r['ann']:+6.1f}% Sh{r['sharpe']:5.2f} MDD{r['mdd']:6.1f}% "
                  f"거래{r['n_trades']:4d} 승{r['win']:4.1f}% 평균{r['avg']:+5.2f}% "
                  f"게이트닫힘 {r['gate_closed_days']}일")
    for k, (a, b) in [("train", (WARMUP, split)), ("test", (split, nD))]:
        m = market(panel, a, b)
        out.setdefault("시장(등가중)", {})[k] = m
        print(f"  {'시장(100종목 등가중)':<18} {k:<5} 연{m['ann']:+6.1f}% Sh{m['sharpe']:5.2f} MDD{m['mdd']:6.1f}%")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
