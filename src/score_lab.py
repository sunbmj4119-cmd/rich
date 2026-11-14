"""
점수 매매 실험실 — 종목마다 '점수'를 숫자로 매기고, 그 점수로 사고파는 정확한 규칙을 만든다.

  점수 = Σ 가중치 × (요소 순위 − 0.5)        요소 순위: 그날 시총 상위 100 안에서 0~1 (높을수록 그 값이 큼)
  매매 = 매일 장 마감 점수로 판단 → 다음 날 시가에 주문 (5일 주기 같은 '시작 날의 운'이 없음)

단계 (기준은 결과를 보기 전에 여기 적어 둔다 · 2008~2018 개발 기간에서만 고르고, 2019~ 는 마지막에 한 번)
  1  요소 성적표   30개 요소 × 1·5·20일: 날마다 순위 상관(IC) · 해마다 플러스 비율 · 상위 10% 의 초과수익
                   (2002~2007 사전 · 2008~2018 개발 · 2019~ 는 표시만, 고르는 데 안 씀)
  2  점수 공식     요소와 가중치를 바꾼 후보들을 같은 기본 규칙 R0 로 비교
                   고정 공식 5개 (반전 · 수급 · 추세 · 저위험 · 사전 기간 부호 같은 비중)
                   해마다 과거로만 다시 정하는 공식 (IC 가중 · 선형 회귀 · 정수 가중치 진화 탐색) × 5·20일 목표
                   나무 모델 (참고: 블랙박스)
                   R0 = 매일 판단 · 5자리 · 점수 상위 5% 안이면 빈자리에 삼 · 상위 20% 밖으로 밀리면 팜
                   통과: 개발 기간 대형주 평균 대비 초과 t ≥ 2 · 샤프 > 대형주 평균 샤프 · 비용 2배에도 초과 > 0
                   → 통과한 것 중 샤프 상위 3개를 3단계로
  3  매매 규칙     자리 수 · 매수 문턱 · 매도 문턱 · 최소 보유일 · 손절 · 최대 보유일 격자 (개발 기간)
                   통과(2단계와 같음)한 규칙 중 '이웃 평균 샤프'(격자 한 칸씩 옆 규칙들과의 평균)가 가장 높은 것 — 홀로 튄 값 대신 고원
  4  최종 확인     고른 점수 + 규칙을 2019~ 에 한 번
                   통과: 초과 > 0 · 샤프 ≥ max(대형주 평균, 코스피) · 비용 2배에도 초과 > 0

  python src/score_lab.py [--stage 1|2|3|4|all] → data/score_lab.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_lab as EL  # noqa: E402
import setup_lab as SL  # noqa: E402

OUT = "data/score_lab.json"
DOCS = "docs/score_lab.json"
FACTORS = [k for k in EL.FEAT if k not in EL.MARKET and k not in EL.BINARY]
PRE = ("2002-01-01", "2007-12-31")
DEV = ("2008-01-01", "2018-12-31")
HOLD = ("2019-01-01", None)
HS = (1, 5, 20)
CB, CS = SL.COMM + SL.SLIP, SL.COMM + SL.SLIP + SL.TAX
R0 = {"slots": 5, "q_in": 0.05, "q_out": 0.20, "min_hold": 1, "stop": 0.0, "max_hold": 0}
CRIT = {"t_excess": 2.0}

# 고정 공식 (경제적 이유로 미리 정함 — 가중치는 정수)
FIXED = {
    "반전형": {"r5": -2, "r1": -1, "clv5": -1},                       # 최근 많이 내리고 종가가 저가 근처 → 되돌림
    "수급형": {"for5": 1, "ins5": 1, "ind5": -1, "smart": 1},          # 외국인·기관이 사고 개인이 팖
    "추세형": {"r60": 1, "ma120": 1, "hi52": 1},                       # 오르는 종목이 계속 오름
    "저위험형": {"volr": -1, "illiq": -1, "size": 1, "max20": -1},     # 덜 흔들리고 큰 종목
}


# ── 자료 ──────────────────────────────────────────────────────────────────────

class Era:
    """한 시대(A 2002~2018 · B 2019~)의 행렬: 요소 X[K,T,N] · 다음 날 시가→그다음 날 시가 수익 G · h일 라벨 · 거래 가능 표시."""

    def __init__(self, era, log=print):
        P, idx = SL.panel(era)
        F = EL.features(P, idx)
        self.era = era
        self.dates = pd.DatetimeIndex(P["dates"])
        self.codes = [str(c).zfill(6) for c in P["codes"]]
        O = P["O"]
        Cf = P["Cf"] if "Cf" in P else pd.DataFrame(P["C"]).ffill().to_numpy()
        T, N = O.shape
        prevc = np.r_[np.full((1, N), np.nan), Cf[:-1]]
        px = np.where(np.isfinite(O) & (O > 0), O, prevc)                   # 시가가 없으면 전날 종가 (거래정지·상장폐지)
        self.px, self.Cf = px, Cf
        G = np.full((T, N), np.nan)
        with np.errstate(all="ignore"):
            G[:-2] = px[2:] / px[1:-1] - 1
        self.G = G
        U = P["U"].astype(bool)
        opn = np.zeros((T, N), bool)
        opn[:-1] = np.isfinite(O[1:]) & (O[1:] > 0)
        self.U = U
        self.buyable = U & opn                                              # 내일 시가에 살 수 있는 오늘의 상위 100
        self.X = np.stack([np.where(U, np.nan_to_num(F[k], nan=0.5) - 0.5, np.nan) for k in FACTORS]).astype(np.float32)
        a, b = ("2002-01-01", "2018-12-31") if era == "A" else ("2019-01-01", None)
        self.i0 = int(np.searchsorted(self.dates, pd.Timestamp(a)))
        self.i1 = T if b is None else int(np.searchsorted(self.dates, pd.Timestamp(b), side="right"))
        self.Y, self.Yx = {}, {}
        for h in HS:
            y = np.full((T, N), np.nan)
            with np.errstate(all="ignore"):
                y[: T - 1 - h] = px[1 + h:] / px[1: T - h] - 1
            y[~self.buyable] = np.nan
            self.Y[h] = y
            dm = np.nanmean(np.where(U, y, np.nan), axis=1, keepdims=True)
            self.Yx[h] = np.clip(y - dm, -0.3, 0.3)
        with np.errstate(all="ignore"):
            self.ew = np.nanmean(np.where(U, G, np.nan), axis=1)
        log(f"  {era}: {T}일 × {N}종목 · 쓰는 날 {self.dates[self.i0].date()}~{self.dates[self.i1 - 1].date()}")

    def rows(self, a, b):
        """날짜 구간 [a, b] 의 행 번호 (이 시대가 맡는 범위 안)."""
        d = self.dates
        lo = max(self.i0, int(np.searchsorted(d, pd.Timestamp(a))))
        hi = min(self.i1, int(np.searchsorted(d, pd.Timestamp(b), side="right")) if b else self.i1)
        return lo, hi

    def score(self, w):
        """가중치 사전 {요소: 값} 또는 길이 K 배열 → 점수 T×N (상위 100 밖은 NaN)."""
        if isinstance(w, dict):
            w = np.array([w.get(k, 0.0) for k in FACTORS], np.float32)
        return np.tensordot(np.asarray(w, np.float32), self.X, axes=1)


def eras(log=print):
    return {"A": Era("A", log), "B": Era("B", log)}


def pct_rank(S):
    """그날 점수 순위 (1 등 = 가장 작은 값, 0~1) · NaN 은 NaN."""
    return pd.DataFrame(S).rank(axis=1, ascending=False, pct=True).to_numpy()


# ── 1단계: 요소 성적표 ───────────────────────────────────────────────────────

def daily_ic(E, x, h):
    """날마다 요소 순위와 h일 수익 순위의 상관 (그날 거래 가능한 상위 100 안)."""
    y = E.Y[h]
    ok = np.isfinite(y) & np.isfinite(x)
    ry = pd.DataFrame(np.where(ok, y, np.nan)).rank(axis=1, pct=True).to_numpy()
    xx = np.where(ok, x, np.nan)
    with np.errstate(all="ignore"):
        xm = xx - np.nanmean(xx, axis=1, keepdims=True)
        ym = ry - np.nanmean(ry, axis=1, keepdims=True)
        ic = np.nansum(xm * ym, axis=1) / np.sqrt(np.nansum(xm ** 2, axis=1) * np.nansum(ym ** 2, axis=1))
    ic[ok.sum(1) < 30] = np.nan
    return pd.Series(ic, index=E.dates)


def ic_series(ER, h, k=None, S=None):
    """두 시대를 이은 날마다 IC (요소 k 또는 점수 S={era: T×N})."""
    parts = []
    for e, E in ER.items():
        x = E.X[FACTORS.index(k)] if k is not None else S[e]
        s = daily_ic(E, x, h)
        parts.append(s.iloc[E.i0:E.i1])
    return pd.concat(parts).sort_index()


def tstat_month(s):
    m = s.dropna().groupby(s.dropna().index.to_period("M")).mean()
    return float(m.mean() / m.std() * np.sqrt(len(m))) if len(m) > 3 and m.std() > 0 else 0.0


def cut(s, a, b):
    s = s[s.index >= pd.Timestamp(a)]
    return s[s.index <= pd.Timestamp(b)] if b else s


def top_spread(ER, h, k, q=0.1):
    """요소 상위 10% 종목의 h일 초과수익(그날 평균 대비, 비용 전) — 날마다."""
    parts = []
    for E in ER.values():
        x = E.X[FACTORS.index(k)]
        y = E.Yx[h]
        ok = np.isfinite(y) & np.isfinite(x)
        r = pd.DataFrame(np.where(ok, x, np.nan)).rank(axis=1, ascending=False, pct=True).to_numpy()
        top = ok & (r <= q)
        with np.errstate(all="ignore"):
            v = np.nansum(np.where(top, y, 0), 1) / top.sum(1)
        parts.append(pd.Series(v, index=E.dates).iloc[E.i0:E.i1])
    return pd.concat(parts).sort_index()


def stage1(ER, log=print):
    out = []
    for k in FACTORS:
        row = {"factor": k, "name": EL.FEAT[k]}
        for h in HS:
            ic = ic_series(ER, h, k=k)
            sp = top_spread(ER, h, k)
            for pn, (a, b) in (("pre", PRE), ("dev", DEV), ("hold", HOLD)):
                s = cut(ic, a, b)
                row[f"ic{h}_{pn}"] = round(float(s.mean()), 4)
                row[f"t{h}_{pn}"] = round(tstat_month(s), 2)
            dv = cut(ic, *DEV)
            yr = dv.groupby(dv.index.year).mean()
            row[f"yrpos{h}_dev"] = round(float((np.sign(yr) == np.sign(dv.mean())).mean()), 3)
            row[f"top{h}_dev"] = round(float(cut(sp, *DEV).mean()), 5)
            row[f"top{h}_hold"] = round(float(cut(sp, *HOLD).mean()), 5)
        out.append(row)
        log(f"  {k:7s} {EL.FEAT[k][:22]:22s} | IC5 사전 {row['ic5_pre']:+.3f} 개발 {row['ic5_dev']:+.3f} (t {row['t5_dev']:+.1f}, 같은 방향 해 {row['yrpos5_dev']:.0%}) "
            f"최종 {row['ic5_hold']:+.3f} | IC20 개발 {row['ic20_dev']:+.3f} (t {row['t20_dev']:+.1f})")
    return out


# ── 매매 흉내 (매일 판단) ────────────────────────────────────────────────────

_PREP = {}


def _prep(E, S):
    """점수 행렬마다 한 번: 그날 순위(0~1) · 점수 높은 순서."""
    key = (E.era, id(S))
    hit = _PREP.get(key)
    if hit is not None and hit[0] is S:
        return hit[1], hit[2]
    Sm = np.where(E.U, S, np.nan)
    PR = pct_rank(Sm)
    order = np.argsort(-np.where(np.isfinite(Sm), Sm, -np.inf), axis=1, kind="stable")
    if len(_PREP) > 8:
        _PREP.clear()
    _PREP[key] = (S, PR, order)
    return PR, order


def simulate(E, S, rule, a, b, cost_mult=1.0, cash0=None):
    """점수 S(T×N)로 매일 장 마감에 판단 → 다음 날 시가에 주문.
    팜: 점수 순위가 q_out 밖(또는 상위 100 밖) 이고 min_hold 일 지남 · 손절(종가가 산 값 대비 −stop 이하) · 최대 보유일.
    삼: 빈자리만큼 점수 순위 q_in 안의 종목을 높은 순으로, 한 자리 = 그때 자산 ÷ slots.
    cash0 가 있으면 정수 주식 (1주도 못 사면 그 자리는 현금)."""
    cb, cs = SL.COMM + SL.SLIP * cost_mult, SL.COMM + SL.SLIP * cost_mult + SL.TAX
    lo, hi = E.rows(a, b)
    hi = min(hi, E.G.shape[0] - 2)
    PR, order = _prep(E, S)
    slots, q_in, q_out = rule["slots"], rule["q_in"], rule["q_out"]
    min_hold, stop, max_hold = rule.get("min_hold", 1), rule.get("stop", 0.0), rule.get("max_hold", 0)
    cash = float(cash0) if cash0 else 1.0
    pos = {}                                                    # j → [값, 산 값(가격), 보유일, 주식 수]
    eq, ew, trades, holds, invested = [], [], [], [], []
    for t in range(lo, hi):
        # ① 팔 것 (오늘 장 마감 정보로)
        for j in list(pos):
            v, ep, d, q, v0 = pos[j]
            pr = PR[t, j]
            out_band = not np.isfinite(pr) or pr > q_out
            stopped = stop > 0 and np.isfinite(E.Cf[t, j]) and E.Cf[t, j] / ep - 1 <= -stop
            if (out_band and d >= min_hold) or stopped or (max_hold and d >= max_hold):
                cash += v * (1 - cs)
                trades.append(v * (1 - cs) / (v0 * (1 + cb)) - 1)
                holds.append(d)
                del pos[j]
        # ② 살 것
        free = slots - len(pos)
        if free > 0:
            total = cash + sum(p[0] for p in pos.values())
            slot = total / slots
            for j in order[t]:
                if free <= 0:
                    break
                if not np.isfinite(PR[t, j]) or PR[t, j] > q_in:
                    break
                if j in pos or not E.buyable[t, j]:
                    continue
                price = E.px[t + 1, j]
                amt = min(slot, cash)
                if cash0:
                    q = int(amt // (price * (1 + cb)))
                    if q <= 0:
                        continue                                        # 1주도 못 사면 다음 순위로 (살 수 있는 것 중 높은 순)
                    spend = q * price * (1 + cb)
                    val = q * price
                else:
                    q, spend, val = 0, amt, amt / (1 + cb)
                if amt <= 1e-12:
                    break
                cash -= spend
                pos[j] = [val, price, 0, q, val]
                free -= 1
        # ③ 다음 날 시가 → 그다음 날 시가
        tot_before = cash + sum(p[0] for p in pos.values())
        invested.append(1 - cash / tot_before if tot_before > 0 else 0.0)
        for j, p in pos.items():
            g = E.G[t, j]
            p[0] *= 1 + (g if np.isfinite(g) else 0.0)
            p[2] += 1
        eq.append(cash + sum(p[0] for p in pos.values()))
        ew.append(E.ew[t])
    idx = E.dates[lo:hi]
    e = pd.Series(eq, index=idx)
    base = float(cash0) if cash0 else 1.0
    r = e.pct_change().fillna(e.iloc[0] / base - 1)
    res = {"r": r, "ew": pd.Series(ew, index=idx).fillna(0.0), "trades": np.array(trades, float), "holds": np.array(holds, float),
           "invested": float(np.mean(invested)) if invested else 0.0}
    return res


def stitch(ER, S, rule, a, b, **kw):
    """두 시대에 걸친 기간이면 이어 붙임 (B 는 2019~)."""
    parts = []
    for e, E in ER.items():
        lo, hi = E.rows(a, b)
        if hi - lo < 5:
            continue
        parts.append(simulate(E, S[e], rule, a, b, **kw))
    if len(parts) == 1:
        return parts[0]
    return {"r": pd.concat([p["r"] for p in parts]), "ew": pd.concat([p["ew"] for p in parts]),
            "trades": np.concatenate([p["trades"] for p in parts]), "holds": np.concatenate([p["holds"] for p in parts]),
            "invested": float(np.mean([p["invested"] for p in parts]))}


def perf(res):
    r, ew = res["r"], res["ew"]
    if len(r) < 60:
        return None
    E = (1 + r).cumprod()
    yrs = len(r) / 252
    ex = r - ew
    em = ex.groupby(ex.index.to_period("M")).sum()
    ewE = (1 + ew).cumprod()
    tr = res["trades"]
    tr = tr[np.isfinite(tr)]
    return {"cagr": round(float(E.iloc[-1] ** (1 / yrs) - 1), 4), "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3) if r.std() > 0 else 0.0,
            "mdd": round(float((E / E.cummax() - 1).min()), 4), "excess_ann": round(float(ex.mean() * 252), 4),
            "t_excess": round(float(em.mean() / em.std() * np.sqrt(len(em))), 2) if em.std() > 0 else 0.0,
            "ew_cagr": round(float(ewE.iloc[-1] ** (1 / yrs) - 1), 4), "ew_sharpe": round(float(ew.mean() / ew.std() * np.sqrt(252)), 3),
            "trades_per_year": round(float(len(res["trades"]) / yrs), 1), "hold_days": round(float(np.mean(res["holds"])), 1) if len(res["holds"]) else None,
            "win": round(float((tr > 0).mean()), 3) if len(tr) else None, "invested": round(res["invested"], 3)}


def passes(p, p2):
    return bool(p and p2 and p["t_excess"] >= CRIT["t_excess"] and p["sharpe"] > p["ew_sharpe"] and p2["excess_ann"] > 0)


# ── 2단계: 점수 공식 ─────────────────────────────────────────────────────────

def train_rows(ER, h, cut_date, every=1):
    """cut_date 전의 행 (두 시대) — 요소 X · 상대 라벨."""
    Xs, ys, ds = [], [], []
    for E in ER.values():
        lo, hi = E.i0, min(E.i1, int(np.searchsorted(E.dates, cut_date)))
        if hi - lo < 50:
            continue
        sl = slice(lo, hi, every)
        y = E.Yx[h][sl]
        X = E.X[:, sl]
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=0)
        Xs.append(X[:, ok].T)
        ys.append(y[ok])
        ds.append(np.repeat(E.dates[sl].to_numpy()[:, None], y.shape[1], axis=1)[ok])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ds)


def ridge_w(X, y, lam=1e4):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def fitness(ER, w, h, cut_date, k=5, every=3, start=PRE[0]):
    """학습 구간(start ~ cut_date 전)에서 날마다 점수 상위 k 의 h일 상대 수익 평균 → 샤프 비슷한 값 (가중치 수가 많으면 조금 깎음)."""
    vals = []
    for E in ER.values():
        lo = max(E.i0, int(np.searchsorted(E.dates, pd.Timestamp(start))))
        hi = min(E.i1, int(np.searchsorted(E.dates, cut_date)))
        if hi - lo < 50:
            continue
        sl = slice(lo, hi, every)
        S = np.tensordot(np.asarray(w, np.float32), E.X[:, sl], axes=1)
        y = E.Yx[h][sl]
        S = np.where(np.isfinite(y) & np.isfinite(S), S, -np.inf)
        top = np.argpartition(-S, k, axis=1)[:, :k]
        v = np.take_along_axis(y, top, axis=1)
        good = np.isfinite(v).all(1)
        vals.append(v[good].mean(1))
    v = np.concatenate(vals) if vals else np.array([0.0])
    return float(v.mean() / (v.std() + 1e-9) * np.sqrt(252 / h)) - 0.02 * int(np.count_nonzero(w))


def evolve(ER, h, cut_date, start_w, rng, passes_n=3, log=None):
    """정수 가중치(−2~+2) 좌표 탐색: 요소를 하나씩 바꿔 보고 학습 구간 성적이 오르면 남긴다 (과거 자료만)."""
    w = np.array(start_w, np.float32)
    best = fitness(ER, w, h, cut_date)
    for _ in range(passes_n):
        changed = False
        for i in rng.permutation(len(FACTORS)):
            for v in (-2, -1, 0, 1, 2):
                if v == w[i]:
                    continue
                w2 = w.copy()
                w2[i] = v
                f = fitness(ER, w2, h, cut_date)
                if f > best + 1e-6:
                    best, w, changed = f, w2, True
        if not changed:
            break
    return w, best


def wf_scores(ER, method, h, log=print, ic_cache=None):
    """해마다(1월 1일) 그 전까지(h+2 거래일 여유)의 자료로 가중치를 다시 정해 그해 점수를 매김 → {era: T×N} · 해마다 가중치."""
    S = {e: np.full(E.G.shape, np.nan, np.float32) for e, E in ER.items()}
    weights = {}
    rng = np.random.default_rng(7)
    years = range(2008, max(E.dates.max().year for E in ER.values()) + 1)
    prev = None
    for y0 in years:
        start = pd.Timestamp(f"{y0}-01-01")
        cut_date = start - pd.Timedelta(days=int((h + 2) * 1.5) + 2)
        if method == "ic":
            w = np.zeros(len(FACTORS), np.float32)
            for i, k in enumerate(FACTORS):
                s = ic_cache[(k, h)]
                s = s[(s.index >= pd.Timestamp(PRE[0])) & (s.index < cut_date)]
                t = tstat_month(s)
                if abs(t) >= 2:
                    w[i] = float(s.mean())
            w = w / (np.abs(w).max() + 1e-12)
        elif method == "ridge":
            X, y, _ = train_rows(ER, h, cut_date, every=2)
            w = ridge_w(X.astype(float), y).astype(np.float32)
            w = w / (np.abs(w).max() + 1e-12)
        elif method == "evo":
            if prev is None:
                start_w = np.zeros(len(FACTORS), np.float32)
                for i, k in enumerate(FACTORS):
                    s = ic_cache[(k, h)]
                    s = s[(s.index >= pd.Timestamp(PRE[0])) & (s.index < cut_date)]
                    if abs(tstat_month(s)) >= 2:
                        start_w[i] = np.sign(s.mean())
            else:
                start_w = prev
            w, f = evolve(ER, h, cut_date, start_w, rng)
            prev = w
        else:
            raise ValueError(method)
        weights[y0] = {k: round(float(v), 3) for k, v in zip(FACTORS, w) if abs(v) > 1e-6}
        for e, E in ER.items():
            lo = int(np.searchsorted(E.dates, start))
            hi = int(np.searchsorted(E.dates, pd.Timestamp(f"{y0 + 1}-01-01")))
            lo, hi = max(lo, E.i0), min(hi, E.i1)
            if hi > lo:
                S[e][lo:hi] = np.tensordot(w, E.X[:, lo:hi], axes=1)
        log(f"    {method} h{h} {y0}: 요소 {len(weights[y0])}개 · " + ", ".join(f"{k}{v:+.2g}" for k, v in sorted(weights[y0].items(), key=lambda x: -abs(x[1]))[:6]))
    return S, weights


def gbm_scores(ER):
    """참고: 나무 모델(세 씨앗 순위 평균, shadow_wf 캐시) → 점수 행렬."""
    import shadow_wf as SW
    import wf_lab as W
    D, _ = W.build(lambda m: None, hs=(5,), sector=False)
    p = SW.load_pred(D)
    S = {e: np.full(E.G.shape, np.nan, np.float32) for e, E in ER.items()}
    for e in ER:
        m = (D["era"] == e).to_numpy() & np.isfinite(p)
        S[e][D["t"].to_numpy()[m], D["j"].to_numpy()[m]] = p[m]
    return S


def pre_sign_weights(ER, ic_cache, h=5):
    """사전 기간(2002~2007)에 IC 가 |t| ≥ 2 인 요소를 그 부호로 1씩 (같은 비중)."""
    w = {}
    for k in FACTORS:
        s = cut(ic_cache[(k, h)], *PRE)
        t = tstat_month(s)
        if abs(t) >= 2:
            w[k] = float(np.sign(s.mean()))
    return w


def stage2(ER, log=print):
    ic_cache = {(k, h): ic_series(ER, h, k=k) for k in FACTORS for h in (5, 20)}
    cands = {}
    for nm, w in FIXED.items():
        cands[nm] = ({e: E.score(w) for e, E in ER.items()}, {"고정": w})
    w = pre_sign_weights(ER, ic_cache)
    cands["사전 부호 같은 비중"] = ({e: E.score(w) for e, E in ER.items()}, {"고정": w})
    for method, label in (("ic", "IC 가중"), ("ridge", "선형 회귀"), ("evo", "진화 탐색")):
        for h in (5, 20):
            log(f"  ▸ {label} · {h}일 목표 (해마다 과거로만)")
            S, ws = wf_scores(ER, method, h, log, ic_cache)
            cands[f"{label} · {h}일"] = (S, ws)
    try:
        cands["나무 모델 (참고)"] = (gbm_scores(ER), {"블랙박스": "wf_lab 나무 부스팅 · 세 씨앗"})
    except (OSError, AssertionError, ValueError) as ex:
        log(f"  나무 모델 건너뜀: {ex}")
    res = {}
    for nm, (S, ws) in cands.items():
        p = perf(stitch(ER, S, R0, *DEV))
        p2 = perf(stitch(ER, S, R0, *DEV, cost_mult=2.0))
        ic5 = ic_series(ER, 5, S=S)
        res[nm] = {"dev": p, "dev_x2": p2, "pass": passes(p, p2), "ic5_dev": round(float(cut(ic5, *DEV).mean()), 4),
                   "weights": ws if "고정" in ws or "블랙박스" in ws else {str(y): v for y, v in ws.items()}}
        log(f"  {nm:18s} 개발: 연 {p['cagr']:+.1%} · 샤프 {p['sharpe']:.2f} (대형주 {p['ew_sharpe']:.2f}) · 낙폭 {p['mdd']:.0%} · 초과 {p['excess_ann']:+.1%}/년 "
            f"(t {p['t_excess']:.1f}) · 비용2배 초과 {p2['excess_ann']:+.1%} · 거래 {p['trades_per_year']:.0f}/년 · 보유 {p['hold_days']}일 · 승률 {p['win'] or 0:.0%} "
            f"{'✓' if res[nm]['pass'] else '✗'}")
    return res, cands


# ── 3단계: 매매 규칙 격자 ────────────────────────────────────────────────────

GRID = {"slots": (3, 5, 10), "q_in": (0.02, 0.05, 0.10), "q_out": (0.10, 0.20, 0.40, 0.60), "min_hold": (1, 5),
        "stop": (0.0, 0.08, 0.15), "max_hold": (0, 20)}


def grid_rules():
    import itertools
    keys = list(GRID)
    for vals in itertools.product(*[GRID[k] for k in keys]):
        r = dict(zip(keys, vals))
        if r["q_out"] <= r["q_in"]:
            continue
        yield r


def neighbors(rule):
    for k, vals in GRID.items():
        i = vals.index(rule[k])
        for di in (-1, 1):
            if 0 <= i + di < len(vals):
                r = dict(rule)
                r[k] = vals[i + di]
                if r["q_out"] > r["q_in"]:
                    yield r


def rkey(r):
    return "|".join(f"{k}={r[k]}" for k in GRID)


def stage3(ER, S, log=print):
    table = {}
    t0 = time.time()
    for i, r in enumerate(grid_rules()):
        p = perf(stitch(ER, S, r, *DEV))
        table[rkey(r)] = {"rule": r, "dev": p}
        if i % 50 == 0:
            log(f"    규칙 {i}개 ({time.time() - t0:.0f}초)")
    for k, v in table.items():
        nb = [table[rkey(n)]["dev"]["sharpe"] for n in neighbors(v["rule"]) if rkey(n) in table]
        v["plateau"] = round(float(np.mean([v["dev"]["sharpe"]] + nb)), 3)
    ranked = sorted(table.values(), key=lambda v: -v["plateau"])
    chosen = None
    for v in ranked[:40]:
        p2 = perf(stitch(ER, S, v["rule"], *DEV, cost_mult=2.0))
        v["dev_x2"] = p2
        v["pass"] = passes(v["dev"], p2)
        if v["pass"] and chosen is None:
            chosen = v
    return table, ranked[:40], chosen


# ── 4단계: 최종 확인 ─────────────────────────────────────────────────────────

def kospi_daily(idx):
    import shadow_wf as SW
    k = SW.kospi_index(None)
    k = k.reindex(k.index.union(idx)).ffill()
    return k.reindex(idx).pct_change().fillna(0.0)


def card_daily(card, idx):
    import wf_lab as W
    c = W.card_index(card)
    c = c.reindex(c.index.union(idx)).ffill()
    return c.reindex(idx).pct_change().fillna(0.0)


def stats_r(r):
    E = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"cagr": round(float(E.iloc[-1] ** (1 / yrs) - 1), 4), "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3),
            "mdd": round(float((E / E.cummax() - 1).min()), 4)}


def stage4(ER, S, rule, log=print):
    out = {}
    res = stitch(ER, S, rule, *HOLD)
    p = perf(res)
    p2 = perf(stitch(ER, S, rule, *HOLD, cost_mult=2.0))
    idx = res["r"].index
    ks = stats_r(kospi_daily(idx))
    dm = stats_r(card_daily("DM:2", idx))
    out["hold"] = p
    out["hold_x2"] = p2
    out["kospi"] = ks
    out["dm2"] = dm
    out["pass"] = bool(p and p2 and p["excess_ann"] > 0 and p["sharpe"] >= max(p["ew_sharpe"], ks["sharpe"]) and p2["excess_ann"] > 0)
    yr = (1 + res["r"]).groupby(res["r"].index.year).prod() - 1
    yew = (1 + res["ew"]).groupby(res["ew"].index.year).prod() - 1
    out["years"] = {str(y): {"strategy": round(float(yr[y]), 4), "ew": round(float(yew[y]), 4)} for y in yr.index}
    out["accounts"] = {}
    for c0 in (200_000, 1_000_000, 3_000_000, 10_000_000):
        q = perf(stitch(ER, S, rule, *HOLD, cash0=c0))
        out["accounts"][str(c0)] = q
    out["curve"] = {"date": [str(d.date()) for d in idx], "strategy": [round(float(v), 4) for v in (1 + res["r"]).cumprod()],
                    "ew": [round(float(v), 4) for v in (1 + res["ew"]).cumprod()]}
    return out


# ── 실행 ─────────────────────────────────────────────────────────────────────

def save(res):
    for fn in (OUT, DOCS):
        json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


def load():
    try:
        return json.load(open(OUT, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)                     # noqa: E731
    t0 = time.time()
    ER = eras(log)
    res = load()
    res.update({"generated": time.strftime("%Y-%m-%d %H:%M"), "factors": {k: EL.FEAT[k] for k in FACTORS}, "R0": R0, "grid": GRID,
                "periods": {"pre": PRE, "dev": DEV, "hold": HOLD}, "crit": CRIT})
    st = a.stage
    if st in ("1", "all"):
        log("■ 1단계: 요소 성적표")
        res["stage1"] = stage1(ER, log)
        save(res)
    if st in ("2", "3", "4", "all"):
        log("■ 2단계: 점수 공식")
        s2, cands = stage2(ER, log)
        res["stage2"] = s2
        save(res)
        if st in ("3", "4", "all"):
            ok = sorted([k for k, v in s2.items() if v["pass"] and "참고" not in k], key=lambda k: -s2[k]["dev"]["sharpe"])[:3]
            log(f"■ 3단계: 매매 규칙 격자 — 점수 {ok}")
            res["stage3"] = {}
            best = None
            for nm in ok:
                table, top, chosen = stage3(ER, cands[nm][0], log)
                res["stage3"][nm] = {"top": top, "chosen": chosen, "n_rules": len(table)}
                if chosen:
                    log(f"  {nm}: 고른 규칙 {chosen['rule']} · 개발 샤프 {chosen['dev']['sharpe']:.2f} (이웃 평균 {chosen['plateau']:.2f}) · 연 {chosen['dev']['cagr']:+.1%} · 낙폭 {chosen['dev']['mdd']:.0%}")
                    if best is None or chosen["plateau"] > best[1]["plateau"]:
                        best = (nm, chosen)
            save(res)
            if best and st in ("4", "all"):
                nm, ch = best
                log(f"■ 4단계: 최종 확인 2019~ (한 번) — {nm} · {ch['rule']}")
                res["final"] = {"score": nm, "rule": ch["rule"], **stage4(ER, cands[nm][0], ch["rule"], log)}
                f = res["final"]
                log(f"  최종: 연 {f['hold']['cagr']:+.1%} · 샤프 {f['hold']['sharpe']:.2f} (대형주 {f['hold']['ew_sharpe']:.2f} · 코스피 {f['kospi']['sharpe']:.2f} · DM:2 {f['dm2']['sharpe']:.2f}) "
                    f"· 초과 {f['hold']['excess_ann']:+.1%} (t {f['hold']['t_excess']:.1f}) · 비용2배 초과 {f['hold_x2']['excess_ann']:+.1%} → {'통과' if f['pass'] else '탈락'}")
    res["secs"] = round(time.time() - t0)
    save(res)
    log(f"끝 ({res['secs']}초) → {OUT}")


if __name__ == "__main__":
    main()
