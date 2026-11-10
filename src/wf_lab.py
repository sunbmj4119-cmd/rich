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
        assert not ({f"y{h}" for h in hs} | {"date", "era", "j", "t"}) & set(keys), "정답 열 이름이 재료 이름과 겹침"
        X = np.c_[[F[k][t, j] for k in keys]].T.astype(np.float32)
        lab = {}
        for h in hs:
            e, x = t + 1, np.minimum(t + 1 + h, T - 1)
            px = np.where(np.isfinite(O[x, j]) & (O[x, j] > 0), O[x, j], Cf[np.minimum(x - 1, T - 1), j])
            r = px / O[e, j] - 1
            r[t + 1 + h >= T] = np.nan
            lab[h] = r.astype(np.float32)
        parts.append(pd.DataFrame({"date": dates[t], "era": era, "j": j, "t": t, **{f"y{h}": lab[h] for h in hs}}).assign(**{k: X[:, i] for i, k in enumerate(keys)}))
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
    y = D[f"y{h}"].to_numpy(float)
    dm = D.groupby("date")[f"y{h}"].transform("mean").to_numpy()
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
    sel = (D["date"] >= pd.Timestamp(a)) & ((D["date"] <= pd.Timestamp(b)) if b else True) & np.isfinite(D[f"y{h}"])
    d = D.loc[sel, ["date", f"y{h}"]].assign(p=pred[sel.to_numpy()])
    days = np.sort(d["date"].unique())
    reb = days[::h]
    d = d[d["date"].isin(reb)]
    ew = d.groupby("date")[f"y{h}"].mean()
    dp = d[np.isfinite(d["p"])]
    if dp.empty:
        return pd.DataFrame(columns=["s", "ew"])
    top = dp.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(k)
    strat = top.groupby("date")[f"y{h}"].apply(lambda r: float(((1 + r) * (1 - cs) / (1 + cb) - 1).mean()))
    strat = strat.reindex(ew.index).fillna(0.0)                                    # 예측이 없는 기간(덮개로 쉼) = 현금 0%
    return pd.DataFrame({"s": strat, "ew": ew}).dropna()


def book_buffer(D, pred, h, k, keep_n, a, b, cost_mult=1.0):
    """book 과 같지만 '들고 있는 종목이 아직 예측 상위 keep_n 안이면 팔지 않고 둔다' — 빈 자리만 새 상위 종목으로 (거래를 줄인다).
    비용: 새로 산 종목에만 매수 비용, 판 종목에만 매도 비용. 반환: 기간 수익 · 대형주 평균 · 기간마다 바꾼 비율."""
    cb, cs = SL.COMM + SL.SLIP * cost_mult, SL.COMM + SL.SLIP * cost_mult + SL.TAX
    sel = (D["date"] >= pd.Timestamp(a)) & ((D["date"] <= pd.Timestamp(b)) if b else True) & np.isfinite(D[f"y{h}"])
    d = D.loc[sel, ["date", "era", "j", f"y{h}"]].assign(p=pred[sel.to_numpy()])
    days = np.sort(d["date"].unique())[::h]
    d = d[d["date"].isin(days)]
    held, rows = [], []
    for day, g in d.groupby("date", sort=True):
        ew = float(g[f"y{h}"].mean())
        g = g[np.isfinite(g["p"])].sort_values("p", ascending=False)
        if g.empty:
            if held:
                rows.append((day, -cs * 1.0, ew, 1.0))                              # 쉬는 기간 시작: 다 판다 (그 기간 현금)
            else:
                rows.append((day, 0.0, ew, 0.0))
            held = []
            continue
        key = list(zip(g["era"], g["j"]))
        rank = {kk: i for i, kk in enumerate(key)}
        ret = dict(zip(key, g[f"y{h}"]))
        stay = [x for x in held if x in rank and rank[x] < keep_n][:k]
        new = [x for x in key if x not in stay][: k - len(stay)]
        sold = [x for x in held if x not in stay]
        port = stay + new
        if not port:
            rows.append((day, 0.0, ew, 0.0))
            held = []
            continue
        w = 1.0 / len(port)
        r = sum(w * ret[x] for x in port)
        cost = (len(new) * w * cb + len(sold) * (1.0 / max(len(held), 1)) * cs) if held else len(new) * w * cb
        rows.append((day, r - cost, ew, (len(new) + len(sold)) / (2 * k)))
        held = port
    out = pd.DataFrame(rows, columns=["date", "s", "ew", "turn"]).set_index("date")
    return out


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


# ── 3회차: 낙폭 줄이기 — 검증된 안정 엔진(전천후 16장)과 섞기 · 변동성 목표 ───────────────

BLEND_CRIT = {"mdd_floor": -0.25}


def aw_index():
    """전천후 16장 일별 자산 (A = 대용 자산 2005~2018 · B = 실제 ETF 2016~) → 하나의 지수 (2019 부터 B)."""
    import optimize_lab as O
    a, b = O.aw_stream("A"), O.aw_stream("B")
    r = pd.concat([a[a.index < "2019-01-01"], b[b.index >= "2019-01-01"]]).fillna(0.0)
    return (1 + r).cumprod()


def blend_eval(bk, aw, w, h, period, vt=None):
    """종목 몫 기간 수익 bk['s'] 와 같은 기간 전천후 수익을 w : (1-w) 로 (기간마다 다시 맞춤) · vt = 종목 몫 변동성 목표."""
    a, b = period
    x = bk[(bk.index >= pd.Timestamp(a)) & ((bk.index <= pd.Timestamp(b)) if b else True)].copy()
    ix = aw.reindex(aw.index.union(x.index)).ffill()
    nxt = list(x.index[1:]) + [x.index[-1] + pd.Timedelta(days=h * 1.5)]
    x["aw"] = [float(ix.asof(n) / ix.asof(d) - 1) for d, n in zip(x.index, nxt)]
    s = x["s"]
    if vt:
        vol = s.rolling(26, min_periods=10).std().shift(1) * np.sqrt(252 / h)
        s = s * np.minimum(1.0, vt / vol).fillna(1.0)
    r = w * s + (1 - w) * x["aw"]
    return r, x["aw"]


def round3(D, M, log):
    p = np.load(os.path.join(CACHE, "wf_gbm_h5_base.npy"))
    assert len(p) == len(D)
    aw = aw_index()
    out = {}
    for k in (3, 10):
        bk = W_book_cache(D, p, k)
        for nm, w, vt in ((f"종목 30% + 전천후 70% (나무·5일·{k}종목)", 0.3, None), (f"종목 50% + 전천후 50% (나무·5일·{k}종목)", 0.5, None),
                          (f"종목 단독 + 변동성 목표 20% (나무·5일·{k}종목)", 1.0, 0.20)):
            row = {}
            for per_name, per in (("dev", DEV), ("holdout", HOLD)):
                r, raw = blend_eval(bk[1], aw, w, 5, per, vt)
                r2, _ = blend_eval(bk[2], aw, w, 5, per, vt)
                ks = kospi(M, per[0], per[1], 5, r.index)
                row[per_name] = {"strategy": perf(r, 5), "strategy_x2": perf(r2, 5), "aw16": perf(raw, 5), "kospi": perf(ks, 5)}
                if per_name == "dev":
                    e = row["dev"]
                    ok = blend_pass(e) if w < 1 else vt_pass(e, bk[1], per)
                    row["dev_pass"] = ok
                    if not ok:
                        break
                else:
                    row["holdout_pass"] = blend_pass(row["holdout"]) if w < 1 else vt_pass(row["holdout"], bk[1], per)
            out[nm] = row
    return out


def W_book_cache(D, p, k):
    return None, book(D, p, 5, k, DEV[0], None), book(D, p, 5, k, DEV[0], None, cost_mult=2.0)


def blend_pass(e):
    s, s2, a, ks = e["strategy"], e["strategy_x2"], e["aw16"], e["kospi"]
    return bool(s["sharpe"] > ks["sharpe"] and s["sharpe"] > a["sharpe"] and s["cagr"] > a["cagr"] and s["mdd"] >= BLEND_CRIT["mdd_floor"]
                and s2["sharpe"] > a["sharpe"])


def vt_pass(e, bk, per):
    s, s2, ks = e["strategy"], e["strategy_x2"], e["kospi"]
    return bool(s["sharpe"] > ks["sharpe"] and s["mdd"] >= ks["mdd"] and s2["sharpe"] > ks["sharpe"])


# ── 4회차: 가장 나은 후보(종목 30% + 전천후 70%)를 깊게 — 공격 엔진 교체 기준 · 무작위 씨앗 · 계좌 크기별 실제 주식 수 ─────────

def card_index(card, stress=1.0):
    """전천후 카드(예: DM:2) 하나의 일별 자산 — A 대용 2005~2018 · B 실제 2019~ 를 이어 붙임."""
    import optimize_lab as O
    a, b = O.card_streams("A", stress)[card], O.card_streams("B", stress)[card]
    r = pd.concat([a[a.index < "2019-01-01"], b[b.index >= "2019-01-01"]]).fillna(0.0)
    return (1 + r).cumprod()


def aw_index_x(stress):
    import optimize_lab as O
    a, b = O.aw_stream("A", stress), O.aw_stream("B", stress)
    r = pd.concat([a[a.index < "2019-01-01"], b[b.index >= "2019-01-01"]]).fillna(0.0)
    return (1 + r).cumprod()


def period_ret(ix, dates, h):
    ix = ix.reindex(ix.index.union(dates)).ffill()
    nxt = list(dates[1:]) + [dates[-1] + pd.Timedelta(days=h * 1.5)]
    return pd.Series([float(ix.asof(n) / ix.asof(d) - 1) for d, n in zip(dates, nxt)], index=dates)


def integer_sim(D, p, w, k, cash0, a="2019-01-01"):
    """실제 ETF·주식 시세로 2019~ : 5거래일마다 자산의 w 를 예측 상위 k 종목에 똑같이 (정수 주식, 못 사는 종목 몫은 현금) · 나머지는 전천후(분수)."""
    import setup_lab as SL_
    P, _ = SL_.panel("B")
    O = P["O"]
    aw = aw_index()
    sel = (D["era"] == "B") & (D["date"] >= pd.Timestamp(a)) & np.isfinite(D["y5"])
    d = D.loc[sel, ["date", "t", "j", "y5"]].assign(p=p[sel.to_numpy()])
    days = np.sort(d["date"].unique())[::5]
    d = d[d["date"].isin(days)].sort_values(["date", "p"], ascending=[True, False])
    awr = period_ret(aw, pd.DatetimeIndex(days), 5)
    E = cash0
    curve, used = [], []
    for day, g in d.groupby("date", sort=True):
        stock_amt = w * E
        per = stock_amt / k
        got = 0.0
        spent = 0.0
        for _, r in g.head(k).iterrows():
            px = O[int(r["t"]) + 1, int(r["j"])]
            q = int(per // (px * (1 + CB))) if np.isfinite(px) and px > 0 else 0
            if q <= 0:
                continue
            cost = q * px * (1 + CB)
            spent += cost
            got += q * px * (1 + r["y5"]) * (1 - CS)
        used.append(spent / max(stock_amt, 1e-9))
        E = (stock_amt - spent + got) + (1 - w) * E * (1 + awr.get(day, 0.0))
        curve.append((day, E))
    s = pd.Series(dict(curve))
    r = s.pct_change().fillna(s.iloc[0] / cash0 - 1)
    return {"cash0": cash0, **perf(r, 5), "final": round(float(s.iloc[-1])), "invested_share": round(float(np.mean(used)), 3)}


def round4(D, M, log):
    out = {}
    p0 = np.load(os.path.join(CACHE, "wf_gbm_h5_base.npy"))
    # ① 공격 엔진 교체 기준 (optimize_lab 4·5회차와 같은 잣대): 두 시기 × 비용 1·3배 모두 DM:2 보다 샤프 ↑ · 낙폭 나쁘지 않음
    rows = {}
    for tag, cm in (("x1", 1.0), ("x3", 3.0)):
        bk = book(D, p0, 5, 3, DEV[0], None, cost_mult=cm)
        aw = aw_index_x(cm)
        dm = card_index("DM:2", cm)
        x = bk.copy()
        x["aw"] = period_ret(aw, x.index, 5)
        x["dm"] = period_ret(dm, x.index, 5)
        x["blend"] = 0.3 * x["s"] + 0.7 * x["aw"]
        for per_name, (a, b) in (("dev", DEV), ("holdout", HOLD)):
            y = x[(x.index >= pd.Timestamp(a)) & ((x.index <= pd.Timestamp(b)) if b else True)]
            rows.setdefault(per_name, {})[tag] = {"blend": perf(y["blend"], 5), "dm2": perf(y["dm"], 5), "aw16": perf(y["aw"], 5)}
    ok = all(rows[pn][tg]["blend"]["sharpe"] > rows[pn][tg]["dm2"]["sharpe"] and rows[pn][tg]["blend"]["mdd"] >= rows[pn][tg]["dm2"]["mdd"]
             for pn in rows for tg in ("x1", "x3"))
    out["vs_dm2"] = {"rows": rows, "pass": bool(ok)}
    log(f"  ① 공격 엔진 교체 기준 (DM:2 대비): {'통과' if ok else '탈락'}")
    for pn in rows:
        for tg in ("x1", "x3"):
            r_ = rows[pn][tg]
            log(f"     {pn:7s} {tg} 섞기 {r_['blend']['cagr']:+.1%} 샤프 {r_['blend']['sharpe']:.2f} 낙폭 {r_['blend']['mdd']:.0%} | "
                f"DM:2 {r_['dm2']['cagr']:+.1%} {r_['dm2']['sharpe']:.2f} {r_['dm2']['mdd']:.0%} | 전천후 {r_['aw16']['cagr']:+.1%} {r_['aw16']['sharpe']:.2f} {r_['aw16']['mdd']:.0%}")
    # ② 무작위 씨앗을 바꿔도 버티나 (나무 모델의 행 고르기·칸 나누기 표본)
    X = D[list(EL.FEAT)]
    seeds = {}
    for off in (1000, 2000):
        fn = os.path.join(CACHE, f"wf_gbm_h5_base_seed{off}.npy")
        if os.path.exists(fn) and len(np.load(fn)) == len(D):
            ps = np.load(fn)
        else:
            ps = walk_forward_seed(D, list(EL.FEAT), 5, off, log)
            np.save(fn, ps)
        e_dev, e_hold = evaluate(D, M, ps, 5, 3, DEV), evaluate(D, M, ps, 5, 3, HOLD)
        seeds[off] = {"dev": e_dev, "holdout": e_hold}
        log(f"  ② 씨앗 +{off}: 개발 연 {e_dev['strategy']['cagr']:+.1%} 샤프 {e_dev['strategy']['sharpe']:.2f} 초과 t {e_dev['t_excess']} | "
            f"최종 연 {e_hold['strategy']['cagr']:+.1%} 샤프 {e_hold['strategy']['sharpe']:.2f} 초과 {e_hold['excess_ann']:+.1%} (t {e_hold['t_excess']})")
    out["seeds"] = seeds
    _ = X
    # ③ 계좌 크기별 실제 주식 수 (2019~) — 30:70 · 50:50
    ints = {}
    for w in (0.3, 0.5):
        for cash0 in (200_000, 1_000_000, 3_000_000, 10_000_000):
            r_ = integer_sim(D, p0, w, 3, cash0)
            ints[f"{int(w * 100)}:{cash0}"] = r_
            log(f"  ③ 종목 {int(w * 100)}% · {cash0:>11,}원: 연 {r_['cagr']:+.1%} 샤프 {r_['sharpe']:.2f} 낙폭 {r_['mdd']:.0%} → {r_['final']:,}원 · 종목 몫 중 실제로 산 비율 {r_['invested_share']:.0%}")
    out["integer"] = ints
    return out


def walk_forward_seed(D, keys, h, off, log):
    """walk_forward 와 같고 나무 모델의 씨앗만 (y0 + off)."""
    pred = np.full(len(D), np.nan)
    X = np.nan_to_num(D[keys].to_numpy(np.float32), nan=0.5)
    y = D[f"y{h}"].to_numpy(float)
    dm = D.groupby("date")[f"y{h}"].transform("mean").to_numpy()
    ex = np.clip(y - dm, -0.3, 0.3)
    dates = D["date"].to_numpy()
    udates = np.unique(dates)
    for y0 in range(2008, int(D["date"].dt.year.max()) + 1, 2):
        start = np.datetime64(f"{y0}-01-01")
        cut = udates[max(np.searchsorted(udates, start) - (h + 2), 0)]
        tr = (dates < cut) & np.isfinite(ex)
        te = (dates >= start) & (dates < np.datetime64(f"{y0 + 2}-01-01"))
        if tr.sum() < 10_000 or te.sum() == 0:
            continue
        m = fit_gbm(X[tr], ex[tr], seed=y0 + off)
        pred[te] = predict(m, X[te])
        log(f"    씨앗 +{off} {y0}")
    return pred


# ── 5회차: 거래 줄이기 (상위권에 남아 있으면 계속 보유) ───────────────────────────────

def eval_bk(D, M, bk, bk2, h, period):
    a, b = period
    sl = lambda x: x[(x.index >= pd.Timestamp(a)) & ((x.index <= pd.Timestamp(b)) if b else True)]      # noqa: E731
    x, x2 = sl(bk), sl(bk2)
    exc, exc2 = x["s"] - x["ew"], x2["s"] - x2["ew"]
    return {"strategy": perf(x["s"], h), "ew": perf(x["ew"], h), "kospi": perf(kospi(M, a, b, h, x.index), h),
            "excess_ann": round(float(exc.mean() * 252 / h), 4), "t_excess": tmonth(exc), "excess_x2_ann": round(float(exc2.mean() * 252 / h), 4),
            "periods": int(len(x)), "turnover": round(float(x["turn"].mean()), 3) if "turn" in x else None}


def vs_dm2(bks, w):
    """종목 몫(bks = {x1, x3}) w : 전천후 (1-w) 를 DM:2 와 — 두 시기 × 비용 1·3배."""
    rows = {}
    for tag, cm in (("x1", 1.0), ("x3", 3.0)):
        x = bks[tag].copy()
        x["aw"] = period_ret(aw_index_x(cm), x.index, 5)
        x["dm"] = period_ret(card_index("DM:2", cm), x.index, 5)
        x["blend"] = w * x["s"] + (1 - w) * x["aw"]
        for pn, (a, b) in (("dev", DEV), ("holdout", HOLD)):
            y = x[(x.index >= pd.Timestamp(a)) & ((x.index <= pd.Timestamp(b)) if b else True)]
            rows.setdefault(pn, {})[tag] = {"blend": perf(y["blend"], 5), "dm2": perf(y["dm"], 5), "aw16": perf(y["aw"], 5)}
    ok = {pn: all(rows[pn][t]["blend"]["sharpe"] > rows[pn][t]["dm2"]["sharpe"] and rows[pn][t]["blend"]["mdd"] >= rows[pn][t]["dm2"]["mdd"]
                  for t in ("x1", "x3")) for pn in rows}
    return rows, ok


def round5(D, M, log):
    p = np.load(os.path.join(CACHE, "wf_gbm_h5_base.npy"))
    out = {}
    bks = {}
    for keep in (6, 10):
        b1 = book_buffer(D, p, 5, 3, keep, DEV[0], None)
        b2 = book_buffer(D, p, 5, 3, keep, DEV[0], None, cost_mult=2.0)
        b3 = book_buffer(D, p, 5, 3, keep, DEV[0], None, cost_mult=3.0)
        bks[keep] = {"x1": b1, "x2": b2, "x3": b3}
        nm = f"나무·5일·3종목 · 상위 {keep} 안이면 계속 보유"
        dev = eval_bk(D, M, b1, b2, 5, DEV)
        row = {"dev": dev, "dev_pass": passes(dev)}
        if row["dev_pass"]:
            row["holdout"] = eval_bk(D, M, b1, b2, 5, HOLD)
            row["holdout_pass"] = passes(row["holdout"])
        out[nm] = row
    for w in (0.3, 0.5):
        nm = f"종목 {int(w * 100)}% (상위 10 안 계속 보유) + 전천후 {int((1 - w) * 100)}% — DM:2 대비"
        rows, ok = vs_dm2(bks[10], w)
        out[nm] = {"rows": rows, "dev_pass": ok["dev"], "holdout_pass": ok["holdout"] if ok["dev"] else None,
                   "holdout_rows_shown": bool(ok["dev"])}
        if not ok["dev"]:
            out[nm]["rows"] = {"dev": rows["dev"]}                                   # 개발 기준을 못 넘으면 최종 성적은 남기지 않는다
    return out


# ── 6회차: 통과한 섞기(종목 30% · 상위 10 안 계속 보유 + 전천후 70%)가 씨앗을 바꿔도 통과하나 · 씨앗 평균 · DM:2 대비 확실성 ─────────

def boot_sharpe_diff(a, b, n=2000, L=10, seed=0):
    """P[샤프(a) − 샤프(b) ≤ 0] — 기간 수익의 블록 부트스트랩."""
    rng = np.random.default_rng(seed)
    d = np.c_[a, b]
    d = d[np.isfinite(d).all(1)]
    T = len(d)
    if T < 50:
        return None
    nb = int(np.ceil(T / L))
    cnt = 0
    for _ in range(n):
        st = rng.integers(0, T - L, nb)
        x = d[(st[:, None] + np.arange(L)).ravel()[:T]]
        cnt += (x[:, 0].mean() / (x[:, 0].std() + 1e-12)) <= (x[:, 1].mean() / (x[:, 1].std() + 1e-12))
    return round(cnt / n, 4)


def round6(D, M, log):
    ps = {"원래": np.load(os.path.join(CACHE, "wf_gbm_h5_base.npy"))}
    for off in (1000, 2000):
        ps[f"씨앗 +{off}"] = np.load(os.path.join(CACHE, f"wf_gbm_h5_base_seed{off}.npy"))
    rk = [pd.Series(v).groupby(D["date"].to_numpy()).rank(pct=True).to_numpy() for v in ps.values()]
    ps["세 씨앗 평균"] = np.nanmean(np.c_[rk], axis=0)
    out = {}
    for nm, p in ps.items():
        bks = {tag: book_buffer(D, p, 5, 3, 10, DEV[0], None, cost_mult=cm) for tag, cm in (("x1", 1.0), ("x3", 3.0))}
        rows, ok = vs_dm2(bks, 0.3)
        x = bks["x1"].copy()
        x["aw"] = period_ret(aw_index_x(1.0), x.index, 5)
        x["dm"] = period_ret(card_index("DM:2", 1.0), x.index, 5)
        bl = 0.3 * x["s"] + 0.7 * x["aw"]
        pv = {}
        for pn, (a, b) in (("dev", DEV), ("holdout", HOLD)):
            m_ = (x.index >= pd.Timestamp(a)) & ((x.index <= pd.Timestamp(b)) if b else True)
            pv[pn] = boot_sharpe_diff(bl[m_].to_numpy(), x["dm"][m_].to_numpy())
        out[nm] = {"rows": rows, "pass": ok, "p_vs_dm2": pv}
        for pn in ("dev", "holdout"):
            q1, q3 = rows[pn]["x1"], rows[pn]["x3"]
            log(f"  {'✓' if ok[pn] else '✗'} {nm:8s} {pn:7s} 섞기 {q1['blend']['cagr']:+.1%} 샤프 {q1['blend']['sharpe']:.2f} 낙폭 {q1['blend']['mdd']:.0%} "
                f"(비용 3배 샤프 {q3['blend']['sharpe']:.2f}) | DM:2 {q1['dm2']['cagr']:+.1%} {q1['dm2']['sharpe']:.2f} {q1['dm2']['mdd']:.0%} (3배 {q3['dm2']['sharpe']:.2f}) "
                f"| p(샤프가 DM:2 이하) {pv[pn]}")
    return out


# ── 회차 설정 (회차마다 결과를 보기 전에 적는다 — 고친 이유도 함께) ───────────────────

ROUNDS = {
    1: {"why": "모든 재료를 한꺼번에 · 선형과 나무 모델 · 보유 5·10·20일 · 3·10종목 + 시장 타이밍",
        "hs": (5, 10, 20), "sector": False,
        "cands": [{"model": m, "h": h, "k": k} for m in ("ridge", "gbm") for h in (5, 10, 20) for k in (3, 10)],
        "timing": (5, 10, 20)},
    2: {"why": "1회차(10일 칸 유효)에서 나무 모델이 대형주보다 연 +13~17% 앞섰지만 낙폭이 코스피보다 3~4%p 나빴다 → "
               "업종 재료 더하기 · 두 모델 합치기 · 코스피 200일선 아래면 현금 · 40일 보유(비용 줄이기)",
        "hs": (10, 40), "sector": True,
        "cands": [{"model": "gbm", "h": 10, "k": k, "feat": "sec"} for k in (3, 10)]
                 + [{"model": "ens", "h": 10, "k": k} for k in (3, 10)]
                 + [{"model": "gbm", "h": 10, "k": k, "overlay": "ma200"} for k in (3, 10)]
                 + [{"model": "gbm", "h": 40, "k": k} for k in (3, 10)]},
}
ROUNDS[3] = {"why": "나무·5일·3종목이 개발·최종 모두 대형주를 크게 이겼지만(초과 t 4.6 · 2.0) 최종 낙폭 −50% 로 코스피(−45%)보다 나빠 탈락 → "
                    "검증된 안정 엔진(전천후 16장)과 30:70 · 50:50 으로 섞기, 또는 종목 몫 변동성 목표 20% (예측은 1회차 그대로 — 새로 고치지 않음)",
             "blend": True}
ROUNDS[4] = {"why": "3회차의 종목 30% + 전천후 70% 가 최종 기간에 연 +15% · 샤프 1.05 · 낙폭 −20% (코스피 +16% · 0.69 · −45%) — "
                    "전천후 단독 샤프(1.25)를 못 넘어 탈락했지만 지금 공격 엔진 DM:2 보다 나아 보임 → 공격 엔진 교체 기준(optimize_lab 4·5회차와 같은 잣대: "
                    "두 시기 × 비용 1·3배 모두 DM:2 보다 샤프 ↑·낙폭 나쁘지 않음)으로 정식 비교 · 나무 모델 씨앗 2개 더 · 계좌 크기별 실제 주식 수",
             "deep": True}
ROUNDS[5] = {"why": "4회차: 섞기(30:70)가 DM:2 를 비용 1배에선 두 시기 모두 이겼지만 최종 기간 비용 3배에서 졌다 (5일마다 3종목을 다 갈아타 비용에 약함) → "
                    "신호는 그대로 두고 '들고 있는 종목이 아직 상위 6·10 안이면 팔지 않음'으로 거래 줄이기 · 종목 기준 / 공격 엔진 교체 기준 그대로",
             "buffer": True}
ROUNDS[6] = {"why": "5회차 섞기(종목 30% · 상위 10 안 계속 보유 + 전천후 70%)가 공격 엔진 교체 기준을 두 시기 × 비용 1·3배 모두 통과 — "
                    "4회차에서 씨앗에 따라 최종 성적이 흔들렸으므로, 씨앗 2개 더와 세 씨앗 평균(예측 순위 평균)으로도 통과하는지 · DM:2 대비 블록 부트스트랩 p",
             "robust": True}
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
    if cfg.get("robust"):
        D, M = build(log, hs=(5,), sector=False)
        c6 = round6(D, M, log)
        prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
        rounds, opened = prev.get("rounds", {}), prev.get("holdout_opened", [])
        opened += [f"{a.round}:{n}" for n in c6]
        rounds[str(a.round)] = {"generated": time.strftime("%Y-%m-%d %H:%M"), "why": cfg["why"], "candidates": c6, "secs": round(time.time() - t0)}
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "rounds": rounds, "holdout_opened": opened,
                   "n_trials": sum(len(r["candidates"]) for r in rounds.values())}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        log(f"끝 ({round(time.time() - t0)}초) · 최종 확인을 연 횟수 누계 {len(opened)}")
        return
    if cfg.get("buffer"):
        D, M = build(log, hs=(5,), sector=False)
        c5 = round5(D, M, log)
        prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
        rounds, opened = prev.get("rounds", {}), prev.get("holdout_opened", [])
        opened += [f"{a.round}:{n}" for n, r in c5.items() if r.get("dev_pass")]
        rounds[str(a.round)] = {"generated": time.strftime("%Y-%m-%d %H:%M"), "why": cfg["why"], "candidates": c5, "secs": round(time.time() - t0)}
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "rounds": rounds, "holdout_opened": opened,
                   "n_trials": sum(len(r["candidates"]) for r in rounds.values())}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        for n, r in c5.items():
            if "dev" in r:
                for pn in ("dev", "holdout"):
                    if pn in r:
                        e = r[pn]
                        s_ = e["strategy"]
                        log(f"  {'✓' if r.get(pn + '_pass') else '✗'} {n:36s} {pn:7s} 연 {s_['cagr']:+.1%} 샤프 {s_['sharpe']:.2f} 낙폭 {s_['mdd']:.0%} | "
                            f"코스피 {e['kospi']['cagr']:+.1%} {e['kospi']['sharpe']:.2f} {e['kospi']['mdd']:.0%} | 초과 연 {e['excess_ann']:+.1%} (t {e['t_excess']}) "
                            f"비용2배 {e['excess_x2_ann']:+.1%} · 바꾼 비율 {e['turnover']}")
            else:
                for pn, v in r["rows"].items():
                    for tg in ("x1", "x3"):
                        q = v[tg]
                        log(f"  {'✓' if r.get(pn + '_pass') else '✗'} {n:44s} {pn:7s} {tg} 섞기 {q['blend']['cagr']:+.1%} 샤프 {q['blend']['sharpe']:.2f} "
                            f"낙폭 {q['blend']['mdd']:.0%} | DM:2 {q['dm2']['cagr']:+.1%} {q['dm2']['sharpe']:.2f} {q['dm2']['mdd']:.0%}")
        log(f"끝 ({round(time.time() - t0)}초) · 최종 확인을 연 횟수 누계 {len(opened)}")
        return
    if cfg.get("deep"):
        D, M = build(log, hs=(5,), sector=False)
        res4 = round4(D, M, log)
        prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
        rounds, opened = prev.get("rounds", {}), prev.get("holdout_opened", [])
        opened += [f"{a.round}:DM:2 대비 (최종 기간 포함)", f"{a.round}:씨앗 +1000", f"{a.round}:씨앗 +2000", f"{a.round}:실제 주식 수"]
        rounds[str(a.round)] = {"generated": time.strftime("%Y-%m-%d %H:%M"), "why": cfg["why"], "candidates": {"섞기 30:70 vs DM:2": res4["vs_dm2"]},
                                "seeds": res4["seeds"], "integer": res4["integer"], "secs": round(time.time() - t0)}
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "rounds": rounds, "holdout_opened": opened,
                   "n_trials": sum(len(r["candidates"]) for r in rounds.values())}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        log(f"끝 ({round(time.time() - t0)}초) · 최종 확인을 연 횟수 누계 {len(opened)}")
        return
    if cfg.get("blend"):
        D, M = build(log, hs=(5,), sector=False)
        cands = round3(D, M, log)
        prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
        rounds, opened = prev.get("rounds", {}), prev.get("holdout_opened", [])
        opened += [f"{a.round}:{n}" for n, r in cands.items() if r.get("dev_pass")]
        rounds[str(a.round)] = {"generated": time.strftime("%Y-%m-%d %H:%M"), "why": cfg["why"], "criteria": BLEND_CRIT, "candidates": cands,
                                "secs": round(time.time() - t0)}
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "rounds": rounds, "holdout_opened": opened,
                   "n_trials": sum(len(r["candidates"]) for r in rounds.values())}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        for n, r in cands.items():
            for per in ("dev", "holdout"):
                if per not in r:
                    continue
                e = r[per]
                s, a_, ks = e["strategy"], e["aw16"], e["kospi"]
                ok = r.get("dev_pass") if per == "dev" else r.get("holdout_pass")
                log(f"  {'✓' if ok else '✗'} {n:34s} {per:7s} 연 {s['cagr']:+.1%} 샤프 {s['sharpe']:.2f} 낙폭 {s['mdd']:.0%} | 전천후 {a_['cagr']:+.1%} {a_['sharpe']:.2f} {a_['mdd']:.0%} | "
                    f"코스피 {ks['cagr']:+.1%} {ks['sharpe']:.2f} {ks['mdd']:.0%} | 비용2배 샤프 {e['strategy_x2']['sharpe']:.2f}")
        log(f"끝 · 최종 확인을 연 횟수 누계 {len(opened)}")
        return
    log("■ 자료 묶기")
    hs = tuple(sorted(set(cfg["hs"])))
    D, M = build(log, hs=hs, sector=cfg["sector"])
    base_keys = list(EL.FEAT)
    sec_keys = base_keys + (list(SEC_FEAT) if cfg["sector"] else [])
    prev = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    rounds = prev.get("rounds", {})
    opened = prev.get("holdout_opened", [])
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "why": cfg["why"], "dev": DEV, "holdout": HOLD, "criteria": CRIT, "candidates": {}}
    log("■ 해마다 다시 배우기 (2008~)")
    for spec in cfg["cands"]:
        sec = spec.get("feat") == "sec"
        p = stock_pred(D, sec_keys if sec else base_keys, spec, "sec" if sec else "base", log)
        name = (f"{spec['model']}·{spec['h']}일·{spec['k']}종목" + ("·업종" if sec else "")
                + {None: "", "timing": "·타이밍", "ma200": "·200일선"}[spec.get("overlay")])
        if spec.get("overlay") == "timing":
            p, _ = overlay_timing(D, M, p, spec["h"])
        elif spec.get("overlay") == "ma200":                                       # 그날 저녁 코스피가 200일선 아래면 그 기간은 현금
            on = M["ma200"].reindex(D["date"]).to_numpy() > 0
            p = np.where(on, p, np.nan)
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
