"""
규칙을 지키는 값 — 같은 모델이 고른 같은 종목을 두고, 규칙대로 사고팔 때와 흔한 뇌동매매 습관대로 할 때를 비교한다.
(wf_lab 의 나무 모델 · 5일 · 3종목 예측을 그대로 쓴다 — 새로 고르지 않음. 20만 원이 아닌 비율로, 3자리 계좌)

  기계적      5거래일마다 상위 3종목을 다음 날 시가에 같은 금액 → 5거래일 뒤 시가에 무조건 판다
  습관 ①     수익은 빨리, 손실은 버티기 (처분 효과): +3% 닿으면 바로 팔고, 내리면 손절 없이 최대 60거래일 버틴다
  습관 ②     무서워서 못 사기: 코스피가 최근 5일 −3% 넘게 빠졌으면 그번엔 사지 않는다
  습관 ③     오른 것만 따라 사기: 고른 종목 중 최근 5일 수익이 상위 20% 안인 것만 산다 (이미 오른 걸 보고 안심하고 산다)
  습관 ①+②

  python src/discipline_lab.py → data/discipline_lab.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exit_lab as XL  # noqa: E402
import setup_lab as SL  # noqa: E402
import wf_lab as W  # noqa: E402

OUT = "data/discipline_lab.json"
PERIODS = {"개발 2008~2018": ("2008-01-01", "2018-12-31"), "최종 2019~": ("2019-01-01", None)}


def picks(D, p, a, b, k=3, h=5):
    sel = (D["date"] >= pd.Timestamp(a)) & ((D["date"] <= pd.Timestamp(b)) if b else True) & np.isfinite(D[f"y{h}"]) & np.isfinite(p)
    d = D.loc[sel, ["date", "era", "t", "j", "r5", f"y{h}"]].assign(p=p[sel.to_numpy()])
    days = np.sort(d["date"].unique())[::h]
    d = d[d["date"].isin(days)].sort_values(["date", "p"], ascending=[True, False])
    return d.groupby("date").head(k)


def book_slots(rows, nets, exit_dates, slots=3):
    """후보를 날짜 순으로 (같은 날은 고른 순), 빈 자리만큼 산다 · 한 자리 = 그때 자산 ÷ slots · 실현 손익으로 자산곡선."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    held, curve, n = [], [], 0
    for (d, _), net, xd in zip(rows, nets, exit_dates):
        still = []
        for x in held:
            if x[0] < d:
                eq += x[1] * x[2]
                curve.append((x[0], eq))
                peak = max(peak, eq)
                mdd = min(mdd, eq / peak - 1)
            else:
                still.append(x)
        held = still
        if len(held) >= slots:
            continue
        held.append((xd, eq / slots, net))
        n += 1
    for x in sorted(held, key=lambda z: z[0]):
        eq += x[1] * x[2]
        curve.append((x[0], eq))
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    if not curve:
        return None
    yrs = max((curve[-1][0] - rows[0][0]).days / 365.25, 1e-9)
    return {"cagr": round(eq ** (1 / yrs) - 1, 4) if eq > 0 else -1.0, "mdd": round(mdd, 4), "trades": n, "total": round(eq - 1, 4)}


def run():
    t0 = time.time()
    D, M = W.build(lambda m: None, hs=(5,), sector=False)
    p = np.load(os.path.join(W.CACHE, "wf_gbm_h5_base.npy"))
    assert len(p) == len(D)
    panels = {"A": SL.panel("A")[0], "B": SL.panel("B")[0]}
    mk5 = M["m5"]
    out = {}
    for pn, (a, b) in PERIODS.items():
        pk = picks(D, p, a, b)
        res = {}
        for era in ("A", "B"):
            sub = pk[pk["era"] == era]
            if sub.empty:
                continue
            P = panels[era]
            dates = pd.DatetimeIndex(P["dates"])
            PA = XL.Paths60(P, sub["t"].to_numpy(), sub["j"].to_numpy())
            mech = (1 + sub["y5"].to_numpy()) * (1 - W.CS) / (1 + W.CB) - 1
            mech_x = dates[np.minimum(sub["t"].to_numpy() + 6, len(dates) - 1)]
            g, xd = XL.simulate(PA, (0.99, 0.03, 1.0, 9.0, 0, 0.0, 60), detail=True)
            emo = XL.nets(g)
            emo_x = dates[np.minimum(sub["t"].to_numpy() + 1 + xd, len(dates) - 1)]
            fear = (mk5.reindex(sub["date"]).to_numpy() < -0.03)
            chase = sub["r5"].to_numpy() >= 0.8
            res[era] = {"rows": list(zip(sub["date"], sub["j"])), "mech": (mech, mech_x), "emo": (emo, emo_x), "fear": fear, "chase": chase}
        rows = sum((res[e]["rows"] for e in res), [])
        cat = lambda key, i: np.concatenate([res[e][key][i] for e in res])          # noqa: E731
        mech, mech_x = cat("mech", 0), cat("mech", 1)
        emo, emo_x = cat("emo", 0), cat("emo", 1)
        fear = np.concatenate([res[e]["fear"] for e in res])
        chase = np.concatenate([res[e]["chase"] for e in res])
        order = np.argsort([r[0] for r in rows], kind="stable")
        rows = [rows[i] for i in order]
        mech, mech_x, emo, emo_x, fear, chase = mech[order], mech_x[order], emo[order], emo_x[order], fear[order], chase[order]

        def keep(mask):
            return [r for r, m in zip(rows, mask) if m]
        cases = {
            "기계적 (규칙대로)": (rows, mech, mech_x),
            "습관 ① 수익은 빨리 · 손실은 버티기": (rows, emo, emo_x),
            "습관 ② 무서워서 못 사기": (keep(~fear), mech[~fear], mech_x[~fear]),
            "습관 ③ 오른 것만 따라 사기": (keep(chase), mech[chase], mech_x[chase]),
            "습관 ①+②": (keep(~fear), emo[~fear], emo_x[~fear]),
        }
        out[pn] = {}
        for nm, (r_, n_, x_) in cases.items():
            st = book_slots(r_, n_, list(x_))
            st["win"] = round(float((np.asarray(n_) > 0).mean()), 3) if len(n_) else None
            st["mean"] = round(float(np.mean(n_)), 5) if len(n_) else None
            out[pn][nm] = st
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "periods": PERIODS, "results": out, "secs": round(time.time() - t0)}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for pn, v in out.items():
        print(f"■ {pn}")
        for nm, st in v.items():
            print(f"  {nm:22s} 연 {st['cagr']:+.1%} · 최대낙폭 {st['mdd']:.0%} · 거래 {st['trades']} · 승률 {st['win']:.0%} · 건당 {st['mean']:+.2%}")
    return res


if __name__ == "__main__":
    run()
