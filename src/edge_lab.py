"""
새 자리 찾기 실험실 — 흔한 차트 자리 말고, 지금까지 안 써 본 재료(수급·밤/낮 수익·종가 위치·급등 이력·회전율·시장 폭·달력)를
하나·둘씩 묶어 전부 시험하고, 확률 모델(로지스틱)로 '익절선에 먼저 닿을 확률'이 높은 자리만 고른다. 종목은 그때의 상위 100 (상장폐지 포함).

사전 등록 (2026-11-09 · 결과를 보기 전에 정함)
  승률을 실력으로 보려고 손절 = 익절 폭: E3 ±3%·5일 · E5 ±5%·10일 · E7 ±7%·20일 (안 닿으면 그날 종가) — chart_lab 과 같은 판정·비용
  기준선 = 같은 날 상위 100 전체를 같은 손절·익절로 샀을 때의 승률·순수익 (그날 시장의 몫을 뺀다)
  재료 36개 (t일 장 마감에 아는 값) — 종목 재료는 그날 상위 100 안의 순위(0~1), 시장 재료는 A 시대 분위, 달력은 참/거짓
  방법 1 조합 탐색: 하나씩(아래·위 10%) + 둘씩(아래·위 20% 끼리) = 2,382 조건 × 3 = 7,146 칸
         A(2002~2018)에서만 고른다: 500행 이상 중 '승률 − 그날 기준 승률'의 달별 t 상위 30
         B(2019~)에서 확인: 100행 이상 · 승률 초과의 달별 t ≥ 3 (30개를 봤으니 엄격하게) · 건당 순수익 > 0 (t ≥ 2) · 그날 기준보다 나음 · 비용 2배에서도 > 0
  방법 2 확률 모델: A 로 로지스틱 회귀(재료 36개, L2) 학습 → B 에서 날마다 확률 상위 1·3·10 종목 → 같은 확인 기준 (9칸)
  통과하면: 겹침 없이 다시 세고, 20만 원 '한 번에 한 종목' · '상위 3종목 나눠 사기'로 돌려 본다

  python src/edge_lab.py → data/edge_lab.json
"""
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart_lab as CL  # noqa: E402
import setup_lab as SL  # noqa: E402

OUT = "data/edge_lab.json"
ERA = SL.ERA
EXITS = {"E3": (0.03, 0.03, 5), "E5": (0.05, 0.05, 10), "E7": (0.07, 0.07, 20)}
CRIT = {"min_rows_A": 500, "top_k": 30, "min_n_B": 100, "t_win_B": 3.0, "t_net_B": 2.0}
FEAT = {
    "r1": "오늘 수익", "r5": "5일 수익", "r20": "20일 수익", "r60": "60일 수익",
    "ovn1": "오늘 갭(밤사이)", "intra1": "오늘 장중 수익", "ovn20": "20일 밤사이 수익 합", "intra20": "20일 장중 수익 합",
    "clv1": "오늘 종가 위치(범위 안)", "clv5": "5일 종가 위치 평균", "max20": "20일 중 최대 상승일", "min20": "20일 중 최대 하락일",
    "rng": "오늘 변동 폭 ÷ 20일 평균", "volr": "5일 변동성 ÷ 60일 변동성", "vr": "오늘 거래량 ÷ 20일 평균", "vtr": "5일 거래량 ÷ 60일 거래량",
    "turn": "회전율 ÷ 60일 중앙값", "hi52": "52주 고점 대비", "lo52": "52주 저점 대비", "ma120": "120일선 대비",
    "for1": "오늘 외국인 순매수 ÷ 거래대금", "for5": "5일 외국인 순매수 강도", "for20": "20일 외국인 순매수 강도",
    "ins5": "5일 기관 순매수 강도", "ind5": "5일 개인 순매수 강도", "forstk": "외국인 연속 순매수(−면 연속 순매도)",
    "smart": "외국인+기관은 사는데 값은 내림 (수급 순위 − 5일 수익 순위)", "size": "시가총액", "illiq": "비유동성(20일 |수익|÷거래대금)",
    "up20": "20일 중 오른 날 비율",
    "breadth": "시장 폭 (20일선 위 종목 비율)", "mkt5": "시장 5일 수익", "mktdd": "시장 1년 고점 대비",
    "tom": "월말·월초 (마지막 1일 + 처음 3일)", "expiry": "옵션 만기일 (둘째 목요일)", "friday": "금요일",
}
MARKET = ("breadth", "mkt5", "mktdd")
BINARY = ("tom", "expiry", "friday")


def _df(x):
    return pd.DataFrame(x)


def features(P, idx):
    """{이름: T×N} — 종목 재료는 그날 상위 100 안의 순위(0~1, 밖은 NaN) · 시장 재료는 원래 값 · 달력은 0/1."""
    U = P["U"]
    C = P["Cf"] if "Cf" in P else _df(P["C"]).ffill().to_numpy()
    O, H, Lo, V = P["O"], P["H"], P["L"], P["V"]
    VAL, MC = P["VAL"], P["MC"]
    sh = lambda x, k=1: _df(x).shift(k).to_numpy()                       # noqa: E731
    roll = lambda x, n, f="mean", mp=None: getattr(_df(x).rolling(n, min_periods=mp or max(2, int(n * 0.8))), f)().to_numpy()   # noqa: E731
    with np.errstate(all="ignore"):
        r1 = C / sh(C) - 1
        lov = np.log(O / sh(C))
        lin = np.log(C / O)
        clv = (C - Lo) / (H - Lo)
        f = {"r1": r1, "r5": C / sh(C, 5) - 1, "r20": C / sh(C, 20) - 1, "r60": C / sh(C, 60) - 1,
             "ovn1": O / sh(C) - 1, "intra1": C / O - 1, "ovn20": roll(lov, 20, "sum"), "intra20": roll(lin, 20, "sum"),
             "clv1": clv, "clv5": roll(clv, 5), "max20": roll(r1, 20, "max"), "min20": roll(r1, 20, "min"),
             "rng": ((H - Lo) / C) / roll((H - Lo) / C, 20), "volr": roll(r1, 5, "std", 4) / roll(r1, 60, "std"),
             "vr": V / sh(roll(V, 20)), "vtr": roll(V, 5) / roll(V, 60),
             "turn": (VAL / MC) / roll(VAL / MC, 60, "median"), "hi52": C / roll(H, 252, "max", 200) - 1, "lo52": C / roll(Lo, 252, "min", 200) - 1,
             "ma120": C / roll(C, 120) - 1}
        sv5, sv20 = roll(VAL, 5, "sum"), roll(VAL, 20, "sum")
        FOR, INS, IND = (np.nan_to_num(P[k]) for k in ("FOR", "INS", "IND"))
        f["for1"] = FOR / VAL
        f["for5"] = roll(FOR, 5, "sum") / sv5
        f["for20"] = roll(FOR, 20, "sum") / sv20
        f["ins5"] = roll(INS, 5, "sum") / sv5
        f["ind5"] = roll(IND, 5, "sum") / sv5
        pos, neg = (FOR > 0).astype(int), (FOR < 0).astype(int)
        stk = np.zeros_like(FOR)
        run_p = np.zeros(FOR.shape[1])
        run_n = np.zeros(FOR.shape[1])
        for t in range(FOR.shape[0]):
            run_p = (run_p + 1) * pos[t]
            run_n = (run_n + 1) * neg[t]
            stk[t] = run_p - run_n
        f["forstk"] = stk
        f["size"] = MC.astype(float)
        f["illiq"] = roll(np.abs(r1) / VAL, 20)
        f["up20"] = roll((r1 > 0).astype(float), 20)
    R = {}
    for k, x in f.items():
        x = np.where(U & np.isfinite(x), x, np.nan)
        R[k] = _df(x).rank(axis=1, pct=True).to_numpy(np.float32)
    R["smart"] = _df(np.where(U, ((R["for5"] + R["ins5"]) / 2) - R["r5"], np.nan)).rank(axis=1, pct=True).to_numpy(np.float32)
    # 시장 재료 (모든 종목 같은 값)
    T, N = U.shape
    above = np.where(U, C > roll(C, 20), np.nan)
    m = pd.Series(np.asarray(idx, float))
    mk = {"breadth": np.nanmean(above, axis=1), "mkt5": (m / m.shift(5) - 1).to_numpy(),
          "mktdd": (m / m.rolling(252, min_periods=120).max() - 1).to_numpy()}
    for k, v in mk.items():
        R[k] = np.repeat(np.asarray(v, np.float32)[:, None], N, axis=1)
    d = pd.DatetimeIndex(P["dates"])
    mon = d.year * 12 + d.month
    last = np.r_[mon[1:] != mon[:-1], True]
    kday = pd.Series(np.arange(len(d))).groupby(mon).cumcount().to_numpy()                 # 그달 몇 번째 거래일
    tom = last | (kday <= 2)
    thu = d.weekday == 3
    nth_thu = pd.Series(thu.astype(int)).groupby(mon).cumsum().to_numpy()
    cal = {"tom": tom, "expiry": thu & (nth_thu == 2), "friday": d.weekday == 4}
    for k, v in cal.items():
        R[k] = np.repeat(np.asarray(v, np.float32)[:, None], N, axis=1)
    return R


def outcomes(P, a_rows):
    """상위 100 안의 모든 (t, j) 에 대해 3가지 청산의 순수익 · 비용 2배 순수익 (T×N, 밖은 NaN)."""
    T, N = P["U"].shape
    ts, js = a_rows
    PA = CL.Paths(P, ts, js)
    out = {}
    for name, (s, g, h) in EXITS.items():
        net = np.full((T, N), np.nan, np.float32)
        net2 = np.full((T, N), np.nan, np.float32)
        net[ts, js] = PA.exit(s, g, h)
        net2[ts, js] = PA.exit(s, g, h, cost_mult=2.0)
        out[name] = (net, net2)
    return out


def valid_rows(P, a, b):
    U = P["U"]
    T = U.shape[0]
    dates = P["dates"]
    i0 = int(np.searchsorted(dates, pd.Timestamp(a)))
    i1 = T if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
    ok = np.zeros_like(U)
    nxt = np.isfinite(P["O"][1:]) & (P["O"][1:] > 0)
    ok[:-1] = U[:-1] & nxt
    ok[: i0] = False
    ok[min(i1, T - max(EXITS.values(), key=lambda x: x[2])[2] - 2):] = False
    return np.nonzero(ok)


class Era:
    """한 시대의 재료 · 결과 · 그날 기준선 · 달 번호."""

    def __init__(self, era, log=print):
        a, b = ERA[era]
        t0 = time.time()
        self.P, idx = SL.panel(era)
        self.F = features(self.P, idx)
        self.rows = valid_rows(self.P, a, b)
        self.out = outcomes(self.P, self.rows)
        T, N = self.P["U"].shape
        self.valid = np.zeros((T, N), bool)
        self.valid[self.rows] = True
        d = pd.DatetimeIndex(self.P["dates"])
        self.month = (d.year * 12 + d.month).to_numpy()
        self.base = {}
        for k, (net, net2) in self.out.items():
            w = np.where(self.valid, (net > 0).astype(np.float32), np.nan)
            self.base[k] = (np.nanmean(w, axis=1), np.nanmean(np.where(self.valid, net, np.nan), axis=1))
        log(f"  {era}: {len(self.rows[0]):,}행 · 재료 {len(self.F)}개 ({time.time() - t0:.0f}초)")

    def stats(self, mask, ex):
        net, net2 = self.out[ex]
        t, j = np.nonzero(mask & self.valid)
        if len(t) == 0:
            return {"n": 0}
        x = net[t, j].astype(float)
        w = (x > 0).astype(float)
        bw, bn = self.base[ex][0][t], self.base[ex][1][t]
        mo = self.month[t]
        out = {"n": int(len(t)), "win": round(float(w.mean()), 4), "base_win": round(float(bw.mean()), 4),
               "dwin": round(float((w - bw).mean()), 4), "t_dwin": _tm(w - bw, mo), "mean": round(float(x.mean()), 5), "t_net": _tm(x, mo),
               "edge": round(float((x - bn).mean()), 5), "t_edge": _tm(x - bn, mo), "mean_x2": round(float(net2[t, j].astype(float).mean()), 5),
               "days": int(len(np.unique(t)))}
        win, loss = x[x > 0], x[x <= 0]
        out["avg_win"] = round(float(win.mean()), 4) if len(win) else 0.0
        out["avg_loss"] = round(float(loss.mean()), 4) if len(loss) else 0.0
        return out


def _tm(x, mo):
    """달별로 묶은 t (같은 달 안의 신호는 서로 닮았으니 달 평균끼리 비교)."""
    u, inv = np.unique(mo, return_inverse=True)
    if len(u) < 6:
        return None
    s = np.bincount(inv, weights=x)
    c = np.bincount(inv)
    m = s / c
    if m.std() == 0:
        return None
    return round(float(m.mean() / m.std(ddof=1) * np.sqrt(len(m))), 2)


def conditions(F, thr_mk):
    """조건 목록 [(이름, 재료, 함수)] — 하나씩(10%)·둘씩(20%) 에 쓰는 꼬리 정의."""
    out = {}
    for k in FEAT:
        if k in BINARY:
            out[k] = {"one": [(f"{k}", lambda F, k=k: F[k] > 0.5)], "pair": [(f"{k}", lambda F, k=k: F[k] > 0.5)]}
        elif k in MARKET:
            q = thr_mk[k]
            out[k] = {"one": [(f"{k}↓10", lambda F, k=k, v=q[0.1]: F[k] <= v), (f"{k}↑10", lambda F, k=k, v=q[0.9]: F[k] >= v)],
                      "pair": [(f"{k}↓20", lambda F, k=k, v=q[0.2]: F[k] <= v), (f"{k}↑20", lambda F, k=k, v=q[0.8]: F[k] >= v)]}
        else:
            out[k] = {"one": [(f"{k}↓10", lambda F, k=k: F[k] <= 0.1), (f"{k}↑10", lambda F, k=k: F[k] >= 0.9)],
                      "pair": [(f"{k}↓20", lambda F, k=k: F[k] <= 0.2), (f"{k}↑20", lambda F, k=k: F[k] >= 0.8)]}
    rules = []
    for k, v in out.items():
        rules += [(nm, fn) for nm, fn in v["one"]]
    for k1, k2 in itertools.combinations(list(out), 2):
        for (n1, f1), (n2, f2) in itertools.product(out[k1]["pair"], out[k2]["pair"]):
            rules.append((f"{n1} & {n2}", lambda F, f1=f1, f2=f2: f1(F) & f2(F)))
    return rules


def market_thresholds(E):
    """시장 재료의 분위 — A 시대 날들로만 (B 에 그대로 쓴다)."""
    rows_t = np.unique(E.rows[0])
    return {k: {q: float(np.nanquantile(E.F[k][rows_t, 0], q)) for q in (0.1, 0.2, 0.8, 0.9)} for k in MARKET}


# ── 확률 모델 ────────────────────────────────────────────────────────────────

def design(E, keys):
    t, j = E.rows
    X = np.c_[[np.nan_to_num(E.F[k][t, j], nan=0.5) for k in keys]].T.astype(np.float64)
    return X


def fit_logit(X, y, l2=1.0, iters=300, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = np.log(y.mean() / (1 - y.mean()))
    n = len(y)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = Z.T @ (p - y) / n + l2 * w / n
        w -= lr * g
        b -= lr * float((p - y).mean())
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def predict(m, X):
    return 1 / (1 + np.exp(-(((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"])))


def top_mask(E, p, k):
    """날마다 확률 상위 k 종목."""
    T, N = E.P["U"].shape
    Pm = np.full((T, N), -np.inf)
    Pm[E.rows] = p
    rk = (-Pm).argsort(axis=1).argsort(axis=1)
    return (rk < k) & np.isfinite(Pm) & (Pm > -np.inf)


def nonoverlap_stats(E, mask, ex, a, b):
    s, g, h = EXITS[ex]
    ts, js = CL.entries(E.P, mask & E.valid, a, b, h)
    if len(ts) < 20:
        return {"n": int(len(ts))}
    PA = CL.Paths(E.P, ts, js)
    net = PA.exit(s, g, h)
    bw, bn = E.base[ex][0][ts], E.base[ex][1][ts]
    w = (net > 0).astype(float)
    mo = E.month[ts]
    return {"n": int(len(ts)), "win": round(float(w.mean()), 4), "base_win": round(float(bw.mean()), 4), "t_dwin": _tm(w - bw, mo),
            "mean": round(float(net.mean()), 5), "t_net": _tm(net, mo), "edge": round(float((net - bn).mean()), 5)}


def pass_B(s):
    return (s.get("n", 0) >= CRIT["min_n_B"] and (s.get("t_dwin") or 0) >= CRIT["t_win_B"] and s.get("mean", -1) > 0
            and (s.get("t_net") or 0) >= CRIT["t_net_B"] and s.get("edge", -1) > 0 and s.get("mean_x2", -1) > 0)


def main():
    t0 = time.time()
    print("■ 자료·재료·결과 계산")
    EA, EB = Era("A"), Era("B")
    thr = market_thresholds(EA)
    rules = conditions(EA.F, thr)
    print(f"■ 방법 1 — 조건 {len(rules):,} × 청산 {len(EXITS)} = {len(rules) * len(EXITS):,}칸 (A 에서만 고름)")
    scored = []
    for i, (nm, fn) in enumerate(rules):
        mA = fn(EA.F)
        for ex in EXITS:
            s = EA.stats(mA, ex)
            if s.get("n", 0) >= CRIT["min_rows_A"] and s.get("t_dwin") is not None:
                scored.append((s["t_dwin"], nm, ex, s, i))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,}/{len(rules):,} · {(time.time() - t0) / 60:.1f}분", flush=True)
    scored.sort(key=lambda x: -x[0])
    tA = np.array([x[0] for x in scored])
    top = scored[: CRIT["top_k"]]
    picks = []
    for tval, nm, ex, sA, i in top:
        mB = rules[i][1](EB.F)
        sB = EB.stats(mB, ex)
        picks.append({"rule": nm, "exit": ex, "A": sA, "B": sB, "pass": pass_B(sB)})
    # 방법 2 — 확률 모델
    keys = list(FEAT)
    XA, XB = design(EA, keys), design(EB, keys)
    model_rows = []
    coefs = {}
    for ex in EXITS:
        yA = (EA.out[ex][0][EA.rows] > 0).astype(float)
        m = fit_logit(XA, yA)
        coefs[ex] = {k: round(float(w), 4) for k, w in sorted(zip(keys, m["w"]), key=lambda z: -abs(z[1]))[:10]}
        pB = predict(m, XB)
        pA = predict(m, XA)
        yB = (EB.out[ex][0][EB.rows] > 0).astype(float)
        dec = pd.qcut(pB, 10, labels=False, duplicates="drop")
        coefs[ex + "_deciles_B"] = [round(float(yB[dec == q].mean()), 4) for q in range(int(dec.max()) + 1)]
        coefs[ex + "_deciles_A"] = [round(float(((EA.out[ex][0][EA.rows] > 0)[pd.qcut(pA, 10, labels=False, duplicates="drop") == q]).mean()), 4) for q in range(10)]
        for k in (1, 3, 10):
            sA = EA.stats(top_mask(EA, pA, k), ex)
            sB = EB.stats(top_mask(EB, pB, k), ex)
            model_rows.append({"rule": f"확률 모델 상위 {k}", "exit": ex, "A": sA, "B": sB, "pass": pass_B(sB)})
    good = [r for r in picks + model_rows if r["pass"]]
    by_name = dict(rules)
    for r in good:
        if r["rule"] in by_name:
            r["B_nonoverlap"] = nonoverlap_stats(EB, by_name[r["rule"]](EB.F), r["exit"], *ERA["B"])
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "era": ERA, "exits": EXITS, "features": FEAT, "criteria": CRIT,
           "n_trials": len(rules) * len(EXITS) + len(model_rows), "n_scored_A": len(scored),
           "A_t_dwin_dist": {"p50": round(float(np.median(tA)), 2), "p99": round(float(np.quantile(tA, 0.99)), 2), "max": round(float(tA.max()), 2)},
           "picks": picks, "model": model_rows, "model_info": coefs, "passed": [f"{r['rule']} · {r['exit']}" for r in good],
           "base_win": {ex: {"A": round(float(np.nanmean(EA.base[ex][0][np.unique(EA.rows[0])])), 4),
                             "B": round(float(np.nanmean(EB.base[ex][0][np.unique(EB.rows[0])])), 4)} for ex in EXITS},
           "secs": round(time.time() - t0)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("■ 그날 전체 승률(기준): " + " · ".join(f"{ex} A {v['A']:.1%} B {v['B']:.1%}" for ex, v in res["base_win"].items()))
    print("■ A 에서 고른 30개 → B 확인")
    for r in picks:
        A, B = r["A"], r["B"]
        print(f"  {'✓' if r['pass'] else '✗'} {r['rule'][:40]:40s} {r['exit']} | A {A['n']:6,}행 승률 {A['win']:.1%} (기준 {A['base_win']:.1%}, t {A['t_dwin']}) 건당 {A['mean']:+.2%} "
              f"| B {B.get('n', 0):6,}행 승률 {B.get('win', 0):.1%} (기준 {B.get('base_win', 0):.1%}, t {B.get('t_dwin')}) 건당 {B.get('mean', 0):+.2%} (t {B.get('t_net')})")
    print("■ 확률 모델")
    for r in model_rows:
        A, B = r["A"], r["B"]
        print(f"  {'✓' if r['pass'] else '✗'} {r['rule']:12s} {r['exit']} | A 승률 {A['win']:.1%} (기준 {A['base_win']:.1%}) 건당 {A['mean']:+.2%} | "
              f"B {B['n']:,}행 승률 {B['win']:.1%} (기준 {B['base_win']:.1%}, t {B['t_dwin']}) 건당 {B['mean']:+.2%} (t {B['t_net']}) 비용2배 {B['mean_x2']:+.2%}")
    for ex in EXITS:
        print(f"  {ex} B 확률 10분위별 실제 승률: {res['model_info'][ex + '_deciles_B']}")
    print(f"통과 {len(good)} · 끝 ({res['secs']}초) → {OUT}")


if __name__ == "__main__":
    main()
