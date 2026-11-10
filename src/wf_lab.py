"""
해마다 다시 배우는 종목 순위 실험실 — 모든 재료를 한꺼번에 넣고, 해마다 그때까지의 자료로만 다시 배워 다음 해를 맞힌다.
2002~2018 = KRX 그때의 상위 100 (상장폐지 포함) · 2019~ = 단기 패널. 재료는 edge_lab 의 36개 (그날 순위 · 시장 · 달력).

사전 등록 (2026-11-10 · 결과를 보기 전에 정함 — 반복해도 속지 않게 셋으로 나눔)
  배우기     해마다(Y) Y년 1월 1일 전 h+2 거래일까지의 자료로만 학습 (확장 창) → Y년 한 해 예측. 2008~ 부터.
  개발 기간  2008~2018: 여기서는 모델·재료·보유 기간을 바꿔 가며 마음껏 고친다 (모두 '처음 보는 해'에 대한 성적이지만, 고친 횟수만큼 운이 낀다)
  최종 확인  2019~: 개발 기준을 넘은 후보만, 후보마다 한 번 연다 (연 횟수는 기록). 넘지 못한 후보의 최종 성적은 계산하지도 저장하지도 않는다.
  전략       h 거래일마다 예측 상위 k 종목을 같은 비중으로 다음 날 시가에 사서 h 일 뒤 시가에 판다 (매번 전부 갈아탄다고 보고 비용 — 보수적)
  비교       같은 기간 상위 100 전부를 같은 비중으로 (비용 없이 — 대형주 시장) · 코스피 지수
  합격       개발·최종 모두: 대형주 시장 대비 초과수익 > 0 이고 달별 t ≥ 2 · 샤프가 코스피와 대형주 시장보다 높음 ·
             최대낙폭이 코스피보다 나쁘지 않음 · 비용 2배에서도 초과수익 > 0
  시장 타이밍 합격 (지수 ETF ↔ 현금): 샤프가 코스피보다 0.1 이상 높음 · 최대낙폭이 나쁘지 않음 · 비용 2배에서도 샤프가 코스피보다 높음
  1회차 후보 모델 2 (능선 회귀 · 나무 부스팅) × 보유 5·10·20일 × 종목 3·10 = 12 + 시장 타이밍(지수 ETF ↔ 현금) 3 = 15

  python src/wf_lab.py [--round 1] → data/wf_lab.json
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_lab as EL  # noqa: E402
import setup_lab as SL  # noqa: E402

OUT = "data/wf_lab.json"
DEV = ("2008-01-01", "2018-12-31")
HOLD = ("2019-01-01", None)
HS = (5, 10, 20)
KS = (3, 10)
CB, CS = SL.COMM + SL.SLIP, SL.COMM + SL.SLIP + SL.TAX
CRIT = {"t_excess": 2.0}
IDX_COST = 0.0003                                   # 지수 ETF 한쪽 (거래세 없음)


# ── 자료 묶기 ────────────────────────────────────────────────────────────────

SEC_KEYS = ("r5", "r20", "r60", "ma120", "for5", "for20", "ins5", "turn", "ovn20", "intra20", "hi52", "max20")
SEC_FEAT = {**{f"s_{k}": f"업종 안 순위: {EL.FEAT[k]}" for k in SEC_KEYS},
            "sec_r20": "업종 전체의 20일 수익 순위 평균 (업종의 힘)", "sec_r60": "업종 전체의 60일 수익 순위 평균", "sec_for20": "업종 전체의 20일 외국인 순매수 강도 평균"}


def sector_features(P, F):
    """업종 상대 재료 — sectors.csv (지금 종목 212개의 업종) 를 종목코드로 붙인다 · 업종이 없는 종목은 NaN (모델에선 0.5)."""
    sec = pd.read_csv("data/sectors.csv", dtype=str).set_index("종목코드")["섹터"]
    codes = [str(c).zfill(6) for c in P["codes"]]
    sid = np.array([sec.get(c, "") for c in codes])
    T, N = P["U"].shape
    out = {k: np.full((T, N), np.nan, np.float32) for k in SEC_FEAT}
    for s in sorted(set(sid) - {""}):
        cols = np.flatnonzero(sid == s)
        for k in SEC_KEYS:
            sub = F[k][:, cols]
            out[f"s_{k}"][:, cols] = pd.DataFrame(sub).rank(axis=1, pct=True).to_numpy(np.float32)
        for k, src in (("sec_r20", "r20"), ("sec_r60", "r60"), ("sec_for20", "for20")):
            with np.errstate(all="ignore"):
                m = np.nanmean(F[src][:, cols], axis=1)
            out[k][:, cols] = m[:, None]
    for k in ("sec_r20", "sec_r60", "sec_for20"):                                 # 업종 평균도 그날 순위로
        out[k] = pd.DataFrame(np.where(P["U"], out[k], np.nan)).rank(axis=1, pct=True).to_numpy(np.float32)
    return out


def build(log=print, hs=HS, sector=False):
    """두 시대를 이어 붙인 행 자료: 재료 X · 날짜 · 시대 · 종목 · h일 시가→시가 수익 (빈 시가는 그 전 종가)."""
    parts = []
    mk_rows = []
    for era, (a, b) in (("A", ("2002-01-01", "2018-12-31")), ("B", ("2019-01-01", None))):
        P, idx = SL.panel(era)
        F = EL.features(P, idx)
        if sector:
            F.update(sector_features(P, F))
        dates = pd.DatetimeIndex(P["dates"])
        T, N = P["U"].shape
        i0 = int(np.searchsorted(dates, pd.Timestamp(a)))
        i1 = T if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
        O = P["O"]
        Cf = P["Cf"] if "Cf" in P else pd.DataFrame(P["C"]).ffill().to_numpy()
        ok = np.zeros((T, N), bool)
        ok[:-1] = P["U"][:-1] & np.isfinite(O[1:]) & (O[1:] > 0)
        ok[:i0] = False
        ok[i1:] = False
        t, j = np.nonzero(ok)
        keys = list(EL.FEAT) + (list(SEC_FEAT) if sector else [])
        X = np.c_[[F[k][t, j] for k in keys]].T.astype(np.float32)
        lab = {}
        for h in hs:
            e, x = t + 1, np.minimum(t + 1 + h, T - 1)
            px = np.where(np.isfinite(O[x, j]) & (O[x, j] > 0), O[x, j], Cf[np.minimum(x - 1, T - 1), j])
            r = px / O[e, j] - 1
            r[t + 1 + h >= T] = np.nan
            lab[h] = r.astype(np.float32)
        parts.append(pd.DataFrame({"date": dates[t], "era": era, "j": j, "t": t, **{f"r{h}": lab[h] for h in hs}}).assign(**{k: X[:, i] for i, k in enumerate(keys)}))
        # 시장 행 (날마다 하나): 지수 수익 · 집계 재료
        m = pd.Series(np.asarray(idx, float), index=dates)
        agg = pd.DataFrame({"date": dates, "era": era, "idx": m.to_numpy()})
        U = P["U"]
        with np.errstate(all="ignore"):
            val = np.where(U, P["VAL"], np.nan)
            for nm in ("FOR", "INS", "IND"):
                flow = np.where(U, np.nan_to_num(P[nm]), np.nan)
                for n in (5, 20):
                    agg[f"{nm.lower()}{n}"] = (pd.Series(np.nansum(flow, 1)).rolling(n).sum() / pd.Series(np.nansum(val, 1)).rolling(n).sum()).to_numpy()
            r1 = np.where(U, Cf / np.r_[np.full((1, N), np.nan), Cf[:-1]] - 1, np.nan)
            agg["disp"] = np.nanstd(r1, 1)
            agg["upshare"] = np.nanmean(r1 > 0, 1)
            agg["breadth"] = F["breadth"][:, 0]
            hi252 = pd.DataFrame(P["H"]).rolling(252, min_periods=200).max().to_numpy()
            agg["hi52share"] = np.nanmean(np.where(U, Cf >= hi252 * 0.98, np.nan), 1)          # 52주 고점 2% 안 종목 비율
        for n in (1, 5, 20, 60, 120):
            agg[f"m{n}"] = (m / m.shift(n) - 1).to_numpy()
        agg["mvol20"] = m.pct_change().rolling(20).std().to_numpy() * np.sqrt(252)
        agg["mdd"] = (m / m.rolling(252, min_periods=120).max() - 1).to_numpy()
        agg["ma200"] = (m / m.rolling(200, min_periods=150).mean() - 1).to_numpy()
        agg["tom"] = F["tom"][:, 0]
        agg["expiry"] = F["expiry"][:, 0]
        for h in sorted(set(hs) | set(HS)):
            agg[f"fr{h}"] = (m.shift(-(h + 1)) / m.shift(-1) - 1).to_numpy()       # 내일 종가에 사서 h일 뒤 종가
        agg = agg.iloc[i0:i1]
        mk_rows.append(agg)
        log(f"  {era}: 종목 행 {len(t):,} · 날 {i1 - i0:,}")
    D = pd.concat(parts, ignore_index=True)
    M = pd.concat(mk_rows, ignore_index=True).set_index("date")
    return D, M


# ── 모델 ─────────────────────────────────────────────────────────────────────

def fit_ridge(X, y, lam=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * len(y) / 1000 * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return {"kind": "ridge", "w": w, "b": float(y.mean()), "mu": mu, "sd": sd}


def bins_of(X, edges):
    B = np.empty(X.shape, np.uint8)
    for f in range(X.shape[1]):
        B[:, f] = np.searchsorted(edges[f], X[:, f], side="right")
    return B


def fit_gbm(X, y, n_trees=120, depth=3, lr=0.05, nb=32, min_leaf=2000, sub=0.5, seed=0):
    """작은 나무 부스팅 (제곱 손실 · 분위 칸 32 · 깊이 3 · 행 절반씩) — 라이브러리 없이 numpy 로."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    samp = rng.choice(n, size=min(n, 200_000), replace=False)
    edges = [np.unique(np.nanquantile(X[samp, f], np.linspace(0, 1, nb + 1)[1:-1])) for f in range(p)]
    B = bins_of(X, edges)
    F = np.full(n, y.mean())
    trees = []
    for m in range(n_trees):
        rows = rng.random(n) < sub
        g = (y - F)[rows]
        Bs = B[rows]
        node = np.zeros(rows.sum(), np.int64)
        splits = []
        for d in range(depth):
            nn = 2 ** d
            best_gain = np.full(nn, 0.0)
            best = [(-1, 0)] * nn
            cnt_tot = np.bincount(node, minlength=nn).astype(float)
            sum_tot = np.bincount(node, weights=g, minlength=nn)
            for f in range(p):
                k = node * (nb + 1) + Bs[:, f]
                s = np.bincount(k, weights=g, minlength=nn * (nb + 1)).reshape(nn, nb + 1)
                c = np.bincount(k, minlength=nn * (nb + 1)).reshape(nn, nb + 1).astype(float)
                sl, cl = np.cumsum(s, 1)[:, :-1], np.cumsum(c, 1)[:, :-1]
                sr, cr = sum_tot[:, None] - sl, cnt_tot[:, None] - cl
                with np.errstate(all="ignore"):
                    gain = sl ** 2 / cl + sr ** 2 / cr - (sum_tot ** 2 / cnt_tot)[:, None]
                gain[(cl < min_leaf * sub) | (cr < min_leaf * sub)] = -np.inf
                bi = np.argmax(gain, 1)
                bg = gain[np.arange(nn), bi]
                for q in np.flatnonzero(bg > best_gain):
                    best_gain[q] = bg[q]
                    best[q] = (f, int(bi[q]))
            splits.append(best)
            fsel = np.array([b[0] for b in best])[node]
            tsel = np.array([b[1] for b in best])[node]
            go = np.where(fsel >= 0, Bs[np.arange(len(node)), np.maximum(fsel, 0)] > tsel, False)
            node = node * 2 + go
        nl = 2 ** depth
        leaf = np.bincount(node, weights=g, minlength=nl) / np.maximum(np.bincount(node, minlength=nl), 1)
        trees.append((splits, lr * leaf))
        F += lr * leaf[_route(B, splits)]
    return {"kind": "gbm", "edges": edges, "trees": trees, "b": float(y.mean())}


def _route(B, splits):
    node = np.zeros(len(B), np.int64)
    for best in splits:
        fsel = np.array([b[0] for b in best])[node]
        tsel = np.array([b[1] for b in best])[node]
        go = np.where(fsel >= 0, B[np.arange(len(node)), np.maximum(fsel, 0)] > tsel, False)
        node = node * 2 + go
    return node


def predict(m, X):
    if m["kind"] == "ridge":
        return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]
    B = bins_of(X, m["edges"])
    out = np.full(len(X), m["b"])
    for splits, leaf in m["trees"]:
        out += leaf[_route(B, splits)]
    return out


def walk_forward(D, keys, h, kind, every=1, log=print):
    """해마다(또는 every 년마다) 그때까지의 자료로 다시 배워 다음 해(들)를 예측 → 예측값 열."""
    pred = np.full(len(D), np.nan)
    years = range(2008, int(D["date"].dt.year.max()) + 1)
    X = np.nan_to_num(D[keys].to_numpy(np.float32), nan=0.5)
    y = D[f"r{h}"].to_numpy(float)
    dm = D.groupby("date")[f"r{h}"].transform("mean").to_numpy()
    ex = np.clip(y - dm, -0.3, 0.3)                                              # 그날 대형주 평균 대비 (상대 성적)
    dates = D["date"].to_numpy()
    udates = np.unique(dates)
    for y0 in list(years)[::every]:
        start = np.datetime64(f"{y0}-01-01")
        cut_i = np.searchsorted(udates, start) - (h + 2)
        cut = udates[max(cut_i, 0)]
        tr = (dates < cut) & np.isfinite(ex)
        te = (dates >= start) & (dates < np.datetime64(f"{y0 + every}-01-01"))
        if tr.sum() < 10_000 or te.sum() == 0:
            continue
        m = fit_ridge(X[tr], ex[tr]) if kind == "ridge" else fit_gbm(X[tr], ex[tr], seed=y0)
        pred[te] = predict(m, X[te])
        log(f"    {kind} h{h} {y0}: 학습 {tr.sum():,}행 → 예측 {te.sum():,}행")
    return pred


# ── 전략 성적 ────────────────────────────────────────────────────────────────

def book(D, pred, h, k, a, b, cost_mult=1.0):
    """h 거래일마다 예측 상위 k (같은 비중) → 기간 수익 (비용 뺌) · 같은 기간 대형주 전체 평균 (비용 없음)."""
    cb, cs = SL.COMM + SL.SLIP * cost_mult, SL.COMM + SL.SLIP * cost_mult + SL.TAX
    sel = (D["date"] >= pd.Timestamp(a)) & ((D["date"] <= pd.Timestamp(b)) if b else True) & np.isfinite(pred) & np.isfinite(D[f"r{h}"])
    d = D.loc[sel, ["date", f"r{h}"]].assign(p=pred[sel.to_numpy()])
    days = np.sort(d["date"].unique())
    reb = days[::h]
    d = d[d["date"].isin(reb)]
    g = d.groupby("date")
    top = d.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(k)
    strat = top.groupby("date")[f"r{h}"].apply(lambda r: float(((1 + r) * (1 - cs) / (1 + cb) - 1).mean()))
    ew = g[f"r{h}"].mean()
    return pd.DataFrame({"s": strat, "ew": ew}).dropna()


def perf(r, h):
    """기간 수익 → 연수익·샤프·최대낙폭 (기간 = h 거래일)."""
    if len(r) < 10:
        return None
    E = (1 + r).cumprod()
    yrs = len(r) * h / 252
    ann = np.sqrt(252 / h)
    return {"cagr": round(float(E.iloc[-1] ** (1 / yrs) - 1), 4), "sharpe": round(float(r.mean() / r.std() * ann), 3) if r.std() > 0 else 0.0,
            "mdd": round(float((E / E.cummax() - 1).min()), 4)}


def tmonth(x):
    m = x.groupby([x.index.year, x.index.month]).mean()
    if len(m) < 12 or m.std() == 0:
        return None
    return round(float(m.mean() / m.std() * np.sqrt(len(m))), 2)


def kospi(M, a, b, h, dates):
    """코스피를 같은 날들에 h일씩 (종가 기준)."""
    m = M["idx"]
    m = m[(m.index >= pd.Timestamp(a)) & ((m.index <= pd.Timestamp(b)) if b else True)]
    r = m.iloc[::h].pct_change().dropna()
    return r


def evaluate(D, M, pred, h, k, period):
    a, b = period
    bk = book(D, pred, h, k, a, b)
    bk2 = book(D, pred, h, k, a, b, cost_mult=2.0)
    if len(bk) < 10:
        return None
    exc, exc2 = bk["s"] - bk["ew"], bk2["s"] - bk2["ew"]
    ks = kospi(M, a, b, h, bk.index)
    return {"strategy": perf(bk["s"], h), "ew": perf(bk["ew"], h), "kospi": perf(ks, h), "excess_ann": round(float(exc.mean() * 252 / h), 4),
            "t_excess": tmonth(exc), "excess_x2_ann": round(float(exc2.mean() * 252 / h), 4), "periods": int(len(bk)),
            "hit": round(float((exc > 0).mean()), 3)}


def passes(e):
    if not e or not e["strategy"] or not e["kospi"]:
        return False
    s, ew, ks = e["strategy"], e["ew"], e["kospi"]
    return (e["excess_ann"] > 0 and (e["t_excess"] or 0) >= CRIT["t_excess"] and s["sharpe"] > ks["sharpe"] and s["sharpe"] > ew["sharpe"]
            and s["mdd"] >= ks["mdd"] and e["excess_x2_ann"] > 0)


# ── 시장 타이밍 (지수 ETF ↔ 현금) ────────────────────────────────────────────

MKEYS = ["for5", "for20", "ins5", "ins20", "ind5", "ind20", "disp", "upshare", "breadth", "hi52share", "m1", "m5", "m20", "m60", "m120",
         "mvol20", "mdd", "ma200", "tom", "expiry"]


def timing(M, h, log=print):
    """해마다 다시 배우는 능선 회귀: 시장 집계 재료 → 지수 h일 앞 수익 · 예측 > 0 이면 지수, 아니면 현금 (h일마다 판단)."""
    X = np.nan_to_num(M[MKEYS].to_numpy(float))
    y = M[f"fr{h}"].to_numpy(float)
    dates = M.index.to_numpy()
    pred = np.full(len(M), np.nan)
    for y0 in range(2008, int(M.index.year.max()) + 1):
        start = np.datetime64(f"{y0}-01-01")
        cut = dates[max(np.searchsorted(dates, start) - (h + 2), 0)]
        tr = (dates < cut) & np.isfinite(y)
        te = (dates >= start) & (dates < np.datetime64(f"{y0 + 1}-01-01"))
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = fit_ridge(X[tr], y[tr], lam=50.0)
        pred[te] = predict(m, X[te])
    return pred


def timing_eval(M, pred, h, period):
    a, b = period
    sel = (M.index >= pd.Timestamp(a)) & ((M.index <= pd.Timestamp(b)) if b else True)
    mm = M[sel].iloc[::h]
    p = pred[sel][::h]
    r = mm[f"fr{h}"].to_numpy()
    on = p > 0
    sw = np.r_[1, np.abs(np.diff(on.astype(int)))]
    strat = pd.Series(np.where(on, r, 0.0) - sw * IDX_COST * 2, index=mm.index).dropna()
    strat2 = pd.Series(np.where(on, r, 0.0) - sw * IDX_COST * 4, index=mm.index).dropna()
    hold = pd.Series(r, index=mm.index).dropna()
    exc, exc2 = strat - hold, strat2 - hold
    return {"strategy": perf(strat, h), "strategy_x2": perf(strat2, h), "kospi": perf(hold, h), "ew": perf(hold, h), "in_market": round(float(on.mean()), 3),
            "excess_ann": round(float(exc.mean() * 252 / h), 4), "t_excess": tmonth(exc), "excess_x2_ann": round(float(exc2.mean() * 252 / h), 4),
            "periods": int(len(strat)), "hit": round(float((exc > 0).mean()), 3)}


def timing_passes(e):
    """시장 타이밍 합격: 샤프가 코스피보다 0.1 이상 높고 · 최대낙폭이 나쁘지 않고 · 비용 2배에서도 샤프가 코스피보다 높음."""
    s, s2, ks = e["strategy"], e["strategy_x2"], e["kospi"]
    return bool(s and ks and s2 and s["sharpe"] >= ks["sharpe"] + 0.1 and s["mdd"] >= ks["mdd"] and s2["sharpe"] > ks["sharpe"])


# ── 회차 설정 (회차마다 결과를 보기 전에 적는다 — 고친 이유도 함께) ───────────────────

ROUNDS = {
    1: {"why": "모든 재료를 한꺼번에 · 선형과 나무 모델 · 보유 5·10·20일 · 3·10종목 + 시장 타이밍",
        "hs": (5, 10, 20), "sector": False,
        "cands": [{"model": m, "h": h, "k": k} for m in ("ridge", "gbm") for h in (5, 10, 20) for k in (3, 10)],
        "timing": (5, 10, 20)},
}
CACHE = "data/cache"


def stock_pred(D, keys, spec, cache_tag, log):
    """모델 예측 (해마다 다시 배움) — 캐시에 있으면 다시 쓴다. ens = 능선·나무 예측을 그날 순위로 평균."""
    if spec["model"] == "ens":
        ps = [stock_pred(D, keys, {**spec, "model": m}, cache_tag, log) for m in ("ridge", "gbm")]
        rk = [pd.Series(p).groupby(D["date"].to_numpy()).rank(pct=True).to_numpy() for p in ps]
        return (rk[0] + rk[1]) / 2
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"wf_{spec['model']}_h{spec['h']}_{cache_tag}.npy")
    if os.path.exists(fn):
        p = np.load(fn)
        if len(p) == len(D):
            return p
    p = walk_forward(D, keys, spec["h"], spec["model"], every=1 if spec["model"] == "ridge" else 2, log=log)
    np.save(fn, p)
    return p


def overlay_timing(D, M, p, h):
    """시장 타이밍이 '아님'인 날엔 종목을 사지 않는다 (예측을 지움 → 그 기간은 현금)."""
    tp = pd.Series(timing(M, h, lambda m: None), index=M.index)
    on = tp.reindex(D["date"]).to_numpy() > 0
    return np.where(on, p, np.nan), on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    cfg = ROUNDS[a.round]
    t0 = time.time()
    log = lambda m: print(m, flush=True)                            # noqa: E731
    log(f"■ {a.round}회차: {cfg['why']}")
    log("■ 자료 묶기")
    hs = tuple(sorted(set(cfg["hs"])))
    D, M = build(log, hs=hs, sector=cfg["sector"])
    keys = list(EL.FEAT) + (list(SEC_FEAT) if cfg["sector"] else [])
    tag = ("sec" if cfg["sector"] else "base") + "_" + "-".join(map(str, hs))
    prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    rounds = prev.get("rounds", {})
    opened = prev.get("holdout_opened", [])
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "why": cfg["why"], "dev": DEV, "holdout": HOLD, "criteria": CRIT, "candidates": {}}
    log("■ 해마다 다시 배우기 (2008~)")
    for spec in cfg["cands"]:
        p = stock_pred(D, keys, spec, tag, log)
        name = f"{spec['model']}·{spec['h']}일·{spec['k']}종목" + ("·업종" if cfg["sector"] else "") + ("·타이밍" if spec.get("overlay") else "")
        if spec.get("overlay") == "timing":
            p, _ = overlay_timing(D, M, p, spec["h"])
        dev = evaluate(D, M, p, spec["h"], spec["k"], DEV)
        row = {"spec": spec, "dev": dev, "dev_pass": passes(dev)}
        if row["dev_pass"]:
            row["holdout"] = evaluate(D, M, p, spec["h"], spec["k"], HOLD)
            row["holdout_pass"] = passes(row["holdout"])
            opened.append(f"{a.round}:{name}")
        res["candidates"][name] = row
    for h in cfg.get("timing", ()):
        p = timing(M, h, log)
        name = f"시장 타이밍·{h}일"
        dev = timing_eval(M, p, h, DEV)
        row = {"dev": dev, "dev_pass": bool(timing_passes(dev))}
        if row["dev_pass"]:
            row["holdout"] = timing_eval(M, p, h, HOLD)
            row["holdout_pass"] = bool(timing_passes(row["holdout"]))
            opened.append(f"{a.round}:{name}")
        res["candidates"][name] = row
    res["secs"] = round(time.time() - t0)
    rounds[str(a.round)] = res
    out = {"generated": res["generated"], "rounds": rounds, "holdout_opened": opened,
           "n_trials": sum(len(r["candidates"]) for r in rounds.values())}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"■ {a.round}회차 개발 기간 (2008~2018) — 처음 보는 해에 대한 성적")
    for name, r in res["candidates"].items():
        d = r["dev"]
        if not d:
            log(f"  {name}: 자료 부족")
            continue
        s, ew, ks = d["strategy"], d["ew"], d["kospi"]
        log(f"  {'✓' if r['dev_pass'] else '✗'} {name:20s} 연 {s['cagr']:+.1%} 샤프 {s['sharpe']:.2f} 낙폭 {s['mdd']:.0%} | 대형주 연 {ew['cagr']:+.1%} 샤프 {ew['sharpe']:.2f} | "
            f"코스피 {ks['cagr']:+.1%} 샤프 {ks['sharpe']:.2f} 낙폭 {ks['mdd']:.0%} | 초과 연 {d['excess_ann']:+.1%} (t {d['t_excess']}) 비용2배 {d['excess_x2_ann']:+.1%}"
            + (f" | 시장 안 {d['in_market']:.0%}" if "in_market" in d else ""))
        if "holdout" in r:
            hd = r["holdout"]
            s = hd["strategy"]
            log(f"      → 최종 확인 2019~: 연 {s['cagr']:+.1%} 샤프 {s['sharpe']:.2f} 낙폭 {s['mdd']:.0%} | 코스피 샤프 {hd['kospi']['sharpe']:.2f} | "
                f"초과 연 {hd['excess_ann']:+.1%} (t {hd['t_excess']}) {'✓ 통과' if r['holdout_pass'] else '✗'}")
    log(f"끝 ({res['secs']}초) · 최종 확인을 연 횟수 누계 {len(opened)} → {OUT}")


if __name__ == "__main__":
    main()
