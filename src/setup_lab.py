"""
조건 매매 실험실 — '확실한 조건에서만 들어가기'를 검증한다. 종목은 그때의 시총 상위 100 (상장폐지 포함).

사전 등록 (결과를 보기 전에 정함 — 조건·보유 기간·합격선 모두 고정. 바꾸면 시도로 센다)
  조건 8개 (t일 장 마감 정보로 판단 → t+1일 시가에 사서 → 정해진 날 수 뒤 시가에 판다)
    S1 하루 -7% 이상 급락 + 시장 200일선 위                      → 5일
    S2 RSI(2) < 5 + 종목 200일선 위 (Connors)                    → 5일
    S3 볼린저 하단(20일·2σ) 아래 + 200일선 위                    → 5일
    S4 52주 신고가 + 거래량 20일 평균의 2배                       → 20일
    S5 외국인 5일 연속 순매수 + 5일 수익 플러스                    → 10일
    S6 거래량 3배 + 하루 -5% 이상 (투매)                          → 10일
    S7 시장 1년 고점 대비 -20% + 종목 20일 -15% (공포에 사기)      → 20일
    S8 20일선이 60일선을 뚫음 + 200일선 위 (골든크로스)            → 20일
  같은 종목이 보유 중에 다시 신호를 내면 세지 않는다 (겹침 없음). 기업 행위로 끊긴 구간은 뺀다.
  비용: 매수 0.0186%+미끄러짐 0.05% · 매도 같고 + 거래세 0.20% (왕복 약 0.34%)
  합격 (두 시대 A 2002~2018 · B 2019~ 모두): 건당 순수익 > 0 이고 달별로 묶은 t ≥ 3 · 같은 기간 시장보다 나음(t ≥ 2) ·
        100건 이상 · 비용 2배에서도 건당 순수익 > 0
  통과한 조건만: 20만 원 '한 번에 한 종목' 계좌로 돌려 본다 (신호가 여럿이면 시총 큰 것).

  python src/setup_lab.py → data/setup_lab.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/setup_lab.json"
ERA = {"A": ("2002-01-02", "2018-12-28"), "B": ("2019-01-02", None)}
COMM, SLIP, TAX = 0.000186, 0.0005, 0.0020
CRIT = {"t_net": 3.0, "t_excess": 2.0, "min_n": 100}
SETUPS = {
    "S1": ("하루 -7% 이상 급락 + 시장 200일선 위", 5),
    "S2": ("RSI(2) < 5 + 종목 200일선 위 (Connors)", 5),
    "S3": ("볼린저 하단 아래 + 200일선 위", 5),
    "S4": ("52주 신고가 + 거래량 2배", 20),
    "S5": ("외국인 5일 연속 순매수 + 5일 수익 플러스", 10),
    "S6": ("거래량 3배 + 하루 -5% 이상 (투매)", 10),
    "S7": ("시장 1년 고점 -20% + 종목 20일 -15% (공포에 사기)", 20),
    "S8": ("20일선이 60일선 돌파 + 200일선 위 (골든크로스)", 20),
}


def panel(era):
    if era == "A":
        import history_lab as H
        P = H.krx_panel()
    else:
        import short_panel as SP
        P = SP.load()
    import short_panel as SP
    mk = SP.market(P)
    return P, np.asarray(mk["idx"], float)


def roll(x, n, fn="mean", mp=None):
    df = pd.DataFrame(x)
    r = df.rolling(n, min_periods=mp or int(n * 0.8))
    return getattr(r, fn)().to_numpy()


def signals(P, idx):
    C = P["Cf"] if "Cf" in P else pd.DataFrame(P["C"]).ffill().to_numpy()
    raw = P["C"]
    V = P["V"]
    r1 = raw / pd.DataFrame(raw).shift(1).to_numpy() - 1
    ma20, ma60, ma200 = roll(C, 20), roll(C, 60), roll(C, 200)
    sd20 = roll(C, 20, "std")
    v20 = pd.DataFrame(V).rolling(20, min_periods=15).mean().shift(1).to_numpy()
    hi252 = pd.DataFrame(C).rolling(252, min_periods=200).max().to_numpy()
    up = np.clip(np.nan_to_num(r1), 0, None)
    dn = np.clip(-np.nan_to_num(r1), 0, None)
    rsi2 = 100 * roll(up, 2, mp=2) / (roll(up, 2, mp=2) + roll(dn, 2, mp=2) + 1e-12)
    m = pd.Series(idx)
    m200 = (m > m.rolling(200, min_periods=150).mean()).to_numpy()[:, None]
    mdd = (m / m.rolling(252, min_periods=120).max() - 1).to_numpy()[:, None]
    r5 = C / pd.DataFrame(C).shift(5).to_numpy() - 1
    r20 = C / pd.DataFrame(C).shift(20).to_numpy() - 1
    F = np.nan_to_num(P["FOR"])
    for5 = pd.DataFrame(F > 0).rolling(5).sum().to_numpy() == 5
    ma20p, ma60p = pd.DataFrame(ma20).shift(1).to_numpy(), pd.DataFrame(ma60).shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        S = {
            "S1": (r1 <= -0.07) & m200,
            "S2": (rsi2 < 5) & (C > ma200),
            "S3": (C < ma20 - 2 * sd20) & (C > ma200),
            "S4": (C >= hi252) & (V >= 2 * v20),
            "S5": for5 & (r5 > 0) & np.isfinite(P["FOR"]),
            "S6": (V >= 3 * v20) & (r1 <= -0.05),
            "S7": (mdd <= -0.20) & (r20 <= -0.15),
            "S8": (ma20 > ma60) & (ma20p <= ma60p) & (C > ma200),
        }
    return {k: np.nan_to_num(v, nan=False).astype(bool) & P["U"] for k, v in S.items()}


def trades(P, idx, S, h, a, b, cost_mult=1.0):
    """신호 → 거래 목록 (겹침 없음). 반환 DataFrame: 진입일, 종목, 순수익, 시장수익."""
    O = P["O"]
    T, N = O.shape
    dates = P["dates"]
    brk = P.get("breaks")
    cb = COMM + SLIP * cost_mult
    cs = COMM + SLIP * cost_mult + TAX
    i0 = int(np.searchsorted(dates, pd.Timestamp(a)))
    i1 = len(dates) if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
    rows = []
    busy = np.full(N, -1)
    cumbrk = np.cumsum(brk, axis=0) if brk is not None else None
    for t in range(i0, min(i1, T - h - 1)):
        js = np.where(S[t] & (busy < t))[0]
        for j in js:
            e, x = t + 1, t + 1 + h
            po, px = O[e, j], O[x, j]
            if not (np.isfinite(po) and np.isfinite(px) and po > 0):
                continue
            if cumbrk is not None and cumbrk[x, j] - cumbrk[t, j] > 0:
                continue
            net = px * (1 - cs) / (po * (1 + cb)) - 1
            mret = idx[min(x, T - 1)] / idx[t] - 1 if np.isfinite(idx[t]) and idx[t] > 0 else np.nan
            rows.append((dates[e], j, net, mret, px / po - 1))
            busy[j] = x
    return pd.DataFrame(rows, columns=["date", "j", "net", "mkt", "gross"])


def tstat_month(x, dates):
    s = pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(dates)).dropna()
    if len(s) < 2:
        return None, 0
    m = s.groupby([s.index.year, s.index.month]).mean()
    if len(m) < 6 or m.std() == 0:
        return None, len(m)
    return round(float(m.mean() / m.std() * np.sqrt(len(m))), 2), len(m)


def one_slot(P, S, h, a, b, cash0=200_000, cost_mult=1.0):
    """20만 원 '한 번에 한 종목': 비어 있으면 그날 신호 중 시총 큰 종목을 다음 날 시가에 전부, h일 뒤 시가에 판다."""
    O, C = P["O"], P["Cf"] if "Cf" in P else pd.DataFrame(P["C"]).ffill().to_numpy()
    MC = np.nan_to_num(P["MC"])
    dates = P["dates"]
    T = len(dates)
    i0 = int(np.searchsorted(dates, pd.Timestamp(a)))
    i1 = len(dates) if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
    cb, cs = COMM + SLIP * cost_mult, COMM + SLIP * cost_mult + TAX
    eq, pos, j, units, until = cash0, False, -1, 0.0, -1
    E = np.full(T, np.nan)
    n = 0
    t = i0
    while t < min(i1, T - 1):
        if pos and t == until:
            eq = units * O[t, j] * (1 - cs) if np.isfinite(O[t, j]) else units * C[t - 1, j] * (1 - cs)
            pos = False
        if not pos and t + h + 1 < T:
            js = np.where(S[t])[0]
            js = [k for k in js if np.isfinite(O[t + 1, k]) and O[t + 1, k] > 0]
            if js:
                j = max(js, key=lambda k: MC[t, k])
                units = eq / (O[t + 1, j] * (1 + cb))
                pos, until, n = True, t + 1 + h, n + 1
                E[t] = eq
                t += 1
                E[t] = units * C[t, j]
                t += 1
                continue
        E[t] = units * C[t, j] if pos else eq
        t += 1
    s = pd.Series(E[i0:min(i1, T - 1)], index=dates[i0:min(i1, T - 1)]).ffill().dropna()
    r = s.pct_change().dropna()
    if len(r) < 60:
        return None
    yrs = len(r) / 252
    return {"cagr": round(float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1), 4), "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3) if r.std() > 0 else 0,
            "mdd": round(float((s / s.cummax() - 1).min()), 4), "trades": n, "final": round(float(s.iloc[-1]))}


def main():
    t0 = time.time()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "criteria": CRIT, "era": ERA, "setups": {k: {"name": v[0], "hold": v[1]} for k, v in SETUPS.items()},
           "rows": {}, "n_trials": len(SETUPS)}
    cache = {}
    for era in ("A", "B"):
        P, idx = panel(era)
        S = signals(P, idx)
        cache[era] = (P, idx, S)
        a, b = ERA[era]
        for k, (name, h) in SETUPS.items():
            tr = trades(P, idx, S[k], h, a, b)
            tr2 = trades(P, idx, S[k], h, a, b, cost_mult=2.0)
            row = {"n": int(len(tr))}
            if len(tr):
                tn, nm = tstat_month(tr["net"], tr["date"])
                te, _ = tstat_month(tr["net"] - tr["mkt"], tr["date"])
                row.update({"win": round(float((tr["net"] > 0).mean()), 3), "mean": round(float(tr["net"].mean()), 4),
                            "median": round(float(tr["net"].median()), 4), "excess": round(float((tr["net"] - tr["mkt"]).mean()), 4),
                            "t_net": tn, "t_excess": te, "months": nm, "worst": round(float(tr["net"].min()), 3),
                            "mean_x2": round(float(tr2["net"].mean()), 4) if len(tr2) else None,
                            "per_year": round(len(tr) / max(1, len(set(pd.DatetimeIndex(tr["date"]).year))), 1)})
            res["rows"].setdefault(k, {})[era] = row
    for k in SETUPS:
        ok, why = True, []
        for era in ("A", "B"):
            r = res["rows"][k][era]
            if r["n"] < CRIT["min_n"]:
                ok = False
                why.append(f"{era} {r['n']}건")
                continue
            if not (r["mean"] > 0 and (r["t_net"] or 0) >= CRIT["t_net"]):
                ok = False
                why.append(f"{era} 건당 {r['mean']:+.2%} t={r['t_net']}")
            if not ((r["t_excess"] or 0) >= CRIT["t_excess"] and r["excess"] > 0):
                ok = False
                why.append(f"{era} 시장 대비 {r['excess']:+.2%} t={r['t_excess']}")
            if not ((r["mean_x2"] or -1) > 0):
                ok = False
                why.append(f"{era} 비용 2배 {r['mean_x2']}")
        res["rows"][k]["pass"] = ok
        res["rows"][k]["why"] = "통과" if ok else "; ".join(why[:3])
    passed = [k for k in SETUPS if res["rows"][k]["pass"]]
    res["passed"] = passed
    res["one_slot"] = {}
    for k in passed or []:
        for era in ("A", "B"):
            P, idx, S = cache[era]
            a, b = ERA[era]
            res["one_slot"].setdefault(k, {})[era] = one_slot(P, S[k], SETUPS[k][1], a, b)
    print("■ 조건 매매 (건당 순수익 · 승률 · 달별 t · 시장 대비 · 비용 2배)")
    for k, (name, h) in SETUPS.items():
        r = res["rows"][k]
        print(f"  {k} {name[:30]:30s} {h:>2}일 | " + " | ".join(
            f"{era} {r[era]['n']:5d}건 승률 {r[era].get('win', 0):.0%} 건당 {r[era].get('mean', 0):+.2%} (t {r[era].get('t_net')}) 시장대비 {r[era].get('excess', 0):+.2%} (t {r[era].get('t_excess')}) 2배 {r[era].get('mean_x2', 0) or 0:+.2%}"
            for era in ("A", "B")) + f"  {'✓' if r['pass'] else '✗'}")
    for k, v in res["one_slot"].items():
        print(f"  20만 원 한 종목씩 {k}: " + " | ".join(f"{e} 연 {x['cagr']:+.1%} 샤프 {x['sharpe']} 낙폭 {x['mdd']:.0%} {x['trades']}번" for e, x in v.items() if x))
    res["secs"] = round(time.time() - t0)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"끝 ({res['secs']}초) · 통과 {passed or '없음'} → {OUT}")


if __name__ == "__main__":
    main()
