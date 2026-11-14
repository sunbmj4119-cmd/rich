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

  ── 1차 결과: 2~3단계 통과 → 4단계(2019~) 탈락. 진단: 비용 전엔 대부분 아직 시장보다 낫지만(+5~36%/년) 거래가 150~450번/년이라
     비용이 다 먹음 · 2019 뒤 시장 성격이 바뀜 (반전 약해지고 추세 강해짐) · 2002 년부터 쌓은 가중치는 바뀐 시장을 못 따라감.

  2차 (2026-11-14, 결과 보기 전에 적음) — '해마다 과거로만 다시 고르는 체계' 자체의 실적을 잰다 (중첩 워크포워드)
     요소 5개 추가: 120일 수익 · 1년 수익(최근 20일 빼고) · 20일선 대비 · 60일 변동성 · 60일 고점 대비
     점수 후보 17개: 고정 5 (반전 · 수급 · 추세 · 저위험 · 장기추세) + (IC 가중 · 선형 회귀 · 진화 탐색) × (5 · 20일) × (2002~ 전체 · 최근 3년)
     규칙 후보 108개: 자리 3·5·10 × 매수 상위 2·5·10% × 매도 상위 20·40·60% 밖 × 최소 보유 1·10일 × 손절 없음·8%
     → 1,836개를 2008~ 매일 흉내 (비용 1배 · 2배)
     해마다 1월 1일: 직전 3년 '비용 뺀 샤프'가 대형주 평균보다 높은 후보 중 상위 5개를 골라 그해 5등분해 담음
                     (없으면 코스피 지수 ETF) · 바꾼 몫만큼 왕복 비용
     통과: 2011~2018 과 2019~ 각각 샤프 > max(대형주 평균, 코스피) · 대형주 대비 초과 > 0 · 비용 2배 체계도 초과 > 0
     ⚠ 2019~ 는 1차에서 이미 봤고, 추가 요소·거래 줄이는 규칙은 그 진단에서 나왔다 → 깨끗한 시험지는 아님.
     민감도(고르는 데 안 씀): 직전 1·3·5년 × 상위 1·5개

  python src/score_lab.py [--stage 1|2|3|4|all|r2] → data/score_lab.json
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
BASE = [k for k in EL.FEAT if k not in EL.MARKET and k not in EL.BINARY]
EXTRA = {"r120": "120일 수익", "r250x": "1년 수익 (최근 20일 빼고)", "ma20": "20일선 대비", "vol60": "60일 변동성", "dd60": "60일 고점 대비"}
FACTORS = list(BASE)


def set_factors(extra):
    """1차 = 30개 · 2차 = 30 + 5개 (전역 목록을 바꾼다 — 시대(Era)를 만들기 전에 부를 것)."""
    global FACTORS
    FACTORS = BASE + (list(EXTRA) if extra else [])


def factor_name(k):
    return EL.FEAT.get(k) or EXTRA.get(k, k)
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
        if any(k in EXTRA for k in FACTORS):
            F.update(extra_features(Cf, U))
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


def extra_features(Cf, U):
    """2차에 더한 요소 (종가만으로) — 그날 상위 100 안의 순위 0~1."""
    c = pd.DataFrame(Cf)
    r = c.pct_change()
    with np.errstate(all="ignore"):
        raw = {"r120": c / c.shift(120) - 1, "r250x": c.shift(20) / c.shift(250) - 1, "ma20": c / c.rolling(20, min_periods=15).mean() - 1,
               "vol60": r.rolling(60, min_periods=40).std(), "dd60": c / c.rolling(60, min_periods=40).max() - 1}
    return {k: v.where(U).rank(axis=1, pct=True).to_numpy(np.float32) for k, v in raw.items()}


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
           "invested": float(np.mean(invested)) if invested else 0.0,
           "pos": {int(j): {"value": p[0] / max(eq[-1], 1e-12) if eq else 0.0, "days": p[2], "entry": p[1]} for j, p in pos.items()}}
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

def train_rows(ER, h, cut_date, every=1, start=None):
    """[start,] cut_date 전의 행 (두 시대) — 요소 X · 상대 라벨."""
    Xs, ys, ds = [], [], []
    for E in ER.values():
        lo = E.i0 if start is None else max(E.i0, int(np.searchsorted(E.dates, pd.Timestamp(start))))
        hi = min(E.i1, int(np.searchsorted(E.dates, cut_date)))
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


def evolve(ER, h, cut_date, start_w, rng, passes_n=3, log=None, start=PRE[0]):
    """정수 가중치(−2~+2) 좌표 탐색: 요소를 하나씩 바꿔 보고 학습 구간 성적이 오르면 남긴다 (과거 자료만)."""
    w = np.array(start_w, np.float32)
    best = fitness(ER, w, h, cut_date, start=start)
    for _ in range(passes_n):
        changed = False
        for i in rng.permutation(len(FACTORS)):
            for v in (-2, -1, 0, 1, 2):
                if v == w[i]:
                    continue
                w2 = w.copy()
                w2[i] = v
                f = fitness(ER, w2, h, cut_date, start=start)
                if f > best + 1e-6:
                    best, w, changed = f, w2, True
        if not changed:
            break
    return w, best


def wf_scores(ER, method, h, log=print, ic_cache=None, window=None):
    """해마다(1월 1일) 그 전까지(h+2 거래일 여유)의 자료로 가중치를 다시 정해 그해 점수를 매김 → {era: T×N} · 해마다 가중치.
    window = None 이면 2002 년부터 전부, 숫자면 최근 그 햇수만."""
    S = {e: np.full(E.G.shape, np.nan, np.float32) for e, E in ER.items()}
    weights = {}
    rng = np.random.default_rng(7)
    years = range(2008, max(E.dates.max().year for E in ER.values()) + 1)
    prev = None
    for y0 in years:
        start = pd.Timestamp(f"{y0}-01-01")
        cut_date = start - pd.Timedelta(days=int((h + 2) * 1.5) + 2)
        tr0 = pd.Timestamp(PRE[0]) if window is None else max(pd.Timestamp(PRE[0]), start - pd.DateOffset(years=window))
        if method == "ic":
            w = np.zeros(len(FACTORS), np.float32)
            for i, k in enumerate(FACTORS):
                s = ic_cache[(k, h)]
                s = s[(s.index >= tr0) & (s.index < cut_date)]
                t = tstat_month(s)
                if abs(t) >= 2:
                    w[i] = float(s.mean())
            w = w / (np.abs(w).max() + 1e-12)
        elif method == "ridge":
            X, y, _ = train_rows(ER, h, cut_date, every=2, start=tr0)
            w = ridge_w(X.astype(float), y).astype(np.float32)
            w = w / (np.abs(w).max() + 1e-12)
        elif method == "evo":
            if prev is None:
                start_w = np.zeros(len(FACTORS), np.float32)
                for i, k in enumerate(FACTORS):
                    s = ic_cache[(k, h)]
                    s = s[(s.index >= tr0) & (s.index < cut_date)]
                    if abs(tstat_month(s)) >= 2:
                        start_w[i] = np.sign(s.mean())
            else:
                start_w = prev
            w, f = evolve(ER, h, cut_date, start_w, rng, start=tr0)
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


def rebuild(ER, w):
    """저장된 가중치(고정 또는 해마다) → 점수 행렬."""
    if "고정" in w:
        return {e: E.score(w["고정"]) for e, E in ER.items()}
    out = {e: np.full(E.G.shape, np.nan, np.float32) for e, E in ER.items()}
    for y, ww in w.items():
        y = int(y)
        vec = np.array([ww.get(k, 0.0) for k in FACTORS], np.float32)
        for e, E in ER.items():
            lo = max(E.i0, int(np.searchsorted(E.dates, pd.Timestamp(f"{y}-01-01"))))
            hi = min(E.i1, int(np.searchsorted(E.dates, pd.Timestamp(f"{y + 1}-01-01"))))
            if hi > lo:
                out[e][lo:hi] = np.tensordot(vec, E.X[:, lo:hi], axes=1)
    return out


def no_cost(fn):
    c, t, sl = SL.COMM, SL.TAX, SL.SLIP
    SL.COMM, SL.TAX, SL.SLIP = 0.0, 0.0, 0.0
    try:
        return fn()
    finally:
        SL.COMM, SL.TAX, SL.SLIP = c, t, sl


def diagnose(ER, res, log=print):
    """1차 탈락 원인 (고르는 데 안 씀): 후보마다 비용 뺀 것 · 비용 0 일 때의 대형주 대비 초과 — 개발 · 2019~."""
    rules = {"R0": R0}
    for nm, v in res.get("stage3", {}).items():
        if v.get("chosen"):
            rules[nm] = v["chosen"]["rule"]
    out = []
    for nm, v in res["stage2"].items():
        if "참고" in nm:
            continue
        Sx = rebuild(ER, v["weights"])
        for rn in ("R0",) + ((nm,) if nm in rules else ()):
            r = rules[rn]
            row = {"score": nm, "rule": "기본 규칙 R0" if rn == "R0" else "3단계에서 고른 규칙", "rule_text": rule_label(r)}
            for pn, per in (("dev", DEV), ("hold", HOLD)):
                p = perf(stitch(ER, Sx, r, *per))
                g = no_cost(lambda: perf(stitch(ER, Sx, r, *per)))
                row.update({f"{pn}_net": p["excess_ann"], f"{pn}_gross": g["excess_ann"], f"{pn}_sharpe": p["sharpe"], f"{pn}_cagr": p["cagr"],
                            f"{pn}_trades": p["trades_per_year"], f"{pn}_hold": p["hold_days"]})
            out.append(row)
            log(f"  {nm:14s} {row['rule']}: 개발 초과 순 {row['dev_net']:+.0%} · 비용0 {row['dev_gross']:+.0%} | 2019~ 순 {row['hold_net']:+.0%} · 비용0 {row['hold_gross']:+.0%} · 거래 {row['hold_trades']:.0f}/년")
    return out


# ── 2차: 해마다 과거로만 다시 고르는 체계 ───────────────────────────────────

RULES2 = [{"slots": sl, "q_in": qi, "q_out": qo, "min_hold": mh, "stop": st, "max_hold": 0}
          for sl in (3, 5, 10) for qi in (0.02, 0.05, 0.10) for qo in (0.2, 0.4, 0.6) for mh in (1, 10) for st in (0.0, 0.08)]
FIXED2 = {**FIXED, "장기추세형": {"r250x": 1, "r120": 1}}
LOOKBACK, TOPN, SEL_START = 3, 5, 2011


def rule_label(r):
    s = f"{r['slots']}자리 · 상위 {r['q_in']:.0%} 안이면 삼 · 상위 {r['q_out']:.0%} 밖이면 팜"
    if r.get("min_hold", 1) > 1:
        s += f" · 최소 {r['min_hold']}일 보유"
    if r.get("stop"):
        s += f" · 손절 {r['stop']:.0%}"
    if r.get("max_hold"):
        s += f" · 최대 {r['max_hold']}일"
    return s


def candidate_scores(ER, log=print):
    ic_cache = {(k, h): ic_series(ER, h, k=k) for k in FACTORS for h in (5, 20)}
    out = {}
    for nm, w in FIXED2.items():
        out[nm] = ({e: E.score(w) for e, E in ER.items()}, {"고정": w})
    for method, label in (("ic", "IC 가중"), ("ridge", "선형 회귀"), ("evo", "진화 탐색")):
        for h in (5, 20):
            for win in (None, 3):
                nm = f"{label} · {h}일 · {'2002~ 전체' if win is None else '최근 3년'}"
                log(f"  ▸ {nm}")
                S, ws = wf_scores(ER, method, h, lambda m: None, ic_cache, window=win)
                out[nm] = (S, {str(y): v for y, v in ws.items()})
    return out


def run_candidates(ER, scores, log=print):
    """모든 (점수, 규칙) 을 2008~ 매일 흉내 → 비용 1배·2배 일별 수익 표."""
    R1, R2, meta = {}, {}, {}
    ew = None
    t0 = time.time()
    for si, (nm, (S, _)) in enumerate(scores.items()):
        for ri, r in enumerate(RULES2):
            key = f"{si}:{ri}"
            a = stitch(ER, S, r, DEV[0], None)
            b = stitch(ER, S, r, DEV[0], None, cost_mult=2.0)
            R1[key], R2[key] = a["r"], b["r"]
            meta[key] = {"score": nm, "rule": r, "trades_per_year": round(len(a["trades"]) / (len(a["r"]) / 252), 1)}
            ew = a["ew"]
        log(f"    점수 {si + 1}/{len(scores)} {nm} ({time.time() - t0:.0f}초)")
    return pd.DataFrame(R1), pd.DataFrame(R2), ew, meta


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else -9.0


def yearly_system(R, ew, ks, lookback=LOOKBACK, topn=TOPN):
    """해마다 1월 1일: 직전 lookback 년 샤프가 대형주 평균보다 높은 후보 중 상위 topn → 그해 똑같이 나눠 담음 (없으면 코스피).
    바꾼 몫만큼 왕복 비용. 반환: 일별 수익 · 해마다 고른 것."""
    years = range(SEL_START, R.index.max().year + 1)
    out, picks, prev = [], {}, set()
    rt = CB + CS
    for y in years:
        a, b = pd.Timestamp(f"{y - lookback}-01-01"), pd.Timestamp(f"{y}-01-01")
        past = R[(R.index >= a) & (R.index < b)]
        if len(past) < 120:
            continue
        sh = past.apply(sharpe)
        bar = sharpe(ew[(ew.index >= a) & (ew.index < b)])
        good = sh[sh > bar].sort_values(ascending=False)
        chosen = list(good.index[:topn])
        cur = R[(R.index >= b) & (R.index < pd.Timestamp(f"{y + 1}-01-01"))]
        if cur.empty:
            continue
        if chosen:
            r = cur[chosen].mean(axis=1)
        else:
            r = ks.reindex(cur.index).fillna(0.0)
        changed = len(set(chosen) - prev) / max(len(chosen), 1) if chosen else (1.0 if prev else 0.0)
        r = r.copy()
        r.iloc[0] -= changed * rt
        out.append(r)
        picks[y] = {"chosen": chosen, "past_sharpe": [round(float(sh[c]), 3) for c in chosen], "bar": round(bar, 3)}
        prev = set(chosen)
    return pd.concat(out), picks


def stats_full(r, ew=None):
    if len(r) < 60:
        return None
    E = (1 + r).cumprod()
    yrs = len(r) / 252
    st = {"cagr": round(float(E.iloc[-1] ** (1 / yrs) - 1), 4), "sharpe": round(sharpe(r), 3), "mdd": round(float((E / E.cummax() - 1).min()), 4)}
    if ew is not None:
        ex = r - ew.reindex(r.index).fillna(0.0)
        em = ex.groupby(ex.index.to_period("M")).sum()
        st["excess_ann"] = round(float(ex.mean() * 252), 4)
        st["t_excess"] = round(float(em.mean() / em.std() * np.sqrt(len(em))), 2) if em.std() > 0 else 0.0
    return st


def round2(ER, log=print):
    t0 = time.time()
    log("  점수 후보 만드는 중 (해마다 과거로만)")
    scores = candidate_scores(ER, log)
    log(f"  후보 {len(scores)} 점수 × {len(RULES2)} 규칙 = {len(scores) * len(RULES2)}개 매일 흉내")
    R1, R2, ew, meta = run_candidates(ER, scores, log)
    idx = R1.index
    ks = kospi_daily(idx)
    dm = card_daily("DM:2", idx)
    sysr, picks = yearly_system(R1, ew, ks)
    sysr2, _ = yearly_system(R2, ew, ks)
    periods = {"2011~2018": ("2011-01-01", "2018-12-31"), "2019~": ("2019-01-01", None),
               "최근 1년": ((idx.max() - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), None), "2011~ 전체": ("2011-01-01", None)}
    table = {}
    for pn, (a, b) in periods.items():
        c = lambda x: cut(x, a, b)                                      # noqa: E731
        table[pn] = {"system": stats_full(c(sysr), ew), "system_x2": stats_full(c(sysr2), ew), "ew": stats_full(c(ew)),
                     "kospi": stats_full(c(ks)), "dm2": stats_full(c(dm))}
    ok = {}
    for pn in ("2011~2018", "2019~"):
        t = table[pn]
        ok[pn] = bool(t["system"]["sharpe"] > max(t["ew"]["sharpe"], t["kospi"]["sharpe"]) and t["system"]["excess_ann"] > 0 and t["system_x2"]["excess_ann"] > 0)
    sens = {}
    for lb in (1, 3, 5):
        for tn in (1, 5):
            r, _ = yearly_system(R1, ew, ks, lb, tn)
            sens[f"직전 {lb}년 · 상위 {tn}"] = {pn: stats_full(cut(r, *periods[pn]), ew) for pn in ("2011~2018", "2019~")}
    # 후보 하나에 계속 머물렀다면 (비교용): 2011~2018 샤프 1등 후보를 끝까지
    dev_best = R1[(R1.index >= "2011-01-01") & (R1.index <= "2018-12-31")].apply(sharpe).idxmax()
    fixed_best = {pn: stats_full(cut(R1[dev_best], *periods[pn]), ew) for pn in ("2011~2018", "2019~")}
    # 해마다 고른 것 풀어 쓰기
    pick_rows = []
    for y, v in picks.items():
        for c, ps in zip(v["chosen"], v["past_sharpe"]):
            m = meta[c]
            yr = cut(R1[c], f"{y}-01-01", f"{y}-12-31")
            pick_rows.append({"year": y, "key": c, "score": m["score"], "rule": rule_label(m["rule"]), "past_sharpe": ps, "bar": v["bar"],
                              "year_ret": round(float((1 + yr).prod() - 1), 4), "trades_per_year": m["trades_per_year"]})
        if not v["chosen"]:
            pick_rows.append({"year": y, "key": "", "score": "코스피 지수 (통과 후보 없음)", "rule": "", "past_sharpe": None, "bar": v["bar"],
                              "year_ret": round(float((1 + cut(ks, f"{y}-01-01", f"{y}-12-31")).prod() - 1), 4), "trades_per_year": 0})
    yrs = sorted({d.year for d in sysr.index})
    year_tab = {}
    for y in yrs:
        g = lambda x: float((1 + cut(x, f"{y}-01-01", f"{y}-12-31")).prod() - 1)  # noqa: E731
        year_tab[str(y)] = {"system": round(g(sysr), 4), "ew": round(g(ew), 4), "kospi": round(g(ks), 4), "dm2": round(g(dm), 4)}
    curve_idx = sysr.index
    curve = {"date": [str(d.date()) for d in curve_idx]}
    for k, x in (("system", sysr), ("ew", ew), ("kospi", ks), ("dm2", dm)):
        x = x.reindex(curve_idx).fillna(0.0)
        curve[k] = [round(float(v), 4) for v in (1 + x).cumprod()]
    # 오늘의 규칙 카드: 올해 고른 후보마다 점수 공식 · 규칙 · 지금 들고 있을 종목 · 오늘 살 후보
    card = today_card(ER, scores, meta, picks)
    res = {"plan": {"lookback": LOOKBACK, "topn": TOPN, "sel_start": SEL_START, "n_scores": len(scores), "n_rules": len(RULES2),
                    "extra_factors": EXTRA, "rules": RULES2},
           "table": table, "pass": ok, "sensitivity": sens, "fixed_best": {"key": dev_best, "score": meta[dev_best]["score"],
                                                                            "rule": rule_label(meta[dev_best]["rule"]), **fixed_best},
           "picks": pick_rows, "years": year_tab, "curve": curve, "card": card,
           "score_weights": {nm: w for nm, (_, w) in scores.items()}, "secs": round(time.time() - t0)}
    for pn, t in table.items():
        s_, e_, k_, d_ = t["system"], t["ew"], t["kospi"], t["dm2"]
        log(f"  {pn:10s} 체계: 연 {s_['cagr']:+.1%} · 샤프 {s_['sharpe']:.2f} · 낙폭 {s_['mdd']:.0%} · 초과 {s_['excess_ann']:+.1%} (t {s_['t_excess']:.1f}) · 비용2배 {t['system_x2']['cagr']:+.1%} "
            f"| 대형주 {e_['cagr']:+.1%}·{e_['sharpe']:.2f} · 코스피 {k_['cagr']:+.1%}·{k_['sharpe']:.2f} · DM:2 {d_['cagr']:+.1%}·{d_['sharpe']:.2f}")
    log(f"  통과: {ok}")
    for k, v in sens.items():
        log(f"  민감도 {k}: " + " / ".join(f"{pn} 연 {x['cagr']:+.1%} 샤프 {x['sharpe']:.2f}" for pn, x in v.items()))
    log(f"  한 후보에 계속 (2011~2018 1등: {meta[dev_best]['score']} · {rule_label(meta[dev_best]['rule'])}): " +
        " / ".join(f"{pn} 연 {x['cagr']:+.1%} 샤프 {x['sharpe']:.2f}" for pn, x in fixed_best.items()))
    return res


def today_card(ER, scores, meta, picks):
    """올해(마지막 해) 고른 후보들: 점수 공식(올해 가중치) · 규칙 · 지금 들고 있을 종목 · 오늘 점수 상위."""
    import shadow_wf as SW
    nm = SW.names()
    E = ER["B"]
    y = max(picks)
    out = []
    for c in picks[y]["chosen"]:
        m = meta[c]
        S, ws = scores[m["score"]]
        w = ws.get("고정") if "고정" in ws else ws.get(str(y))
        res = simulate(E, S["B"], m["rule"], "2019-01-01", None)
        last = int(np.searchsorted(E.dates, E.dates[E.i1 - 1]))
        Sm = np.where(E.U[last], S["B"][last], np.nan)
        order = np.argsort(-np.where(np.isfinite(Sm), Sm, -np.inf))[:10]
        out.append({"key": c, "score": m["score"], "weights": {k: v for k, v in sorted((w or {}).items(), key=lambda x: -abs(x[1]))},
                    "weight_names": {k: factor_name(k) for k in (w or {})}, "rule": m["rule"], "rule_text": rule_label(m["rule"]),
                    "holding": [{"code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j]), "share": round(v["value"], 3), "days": v["days"]}
                                for j, v in sorted(res["pos"].items(), key=lambda x: -x[1]["value"])],
                    "top_today": [{"rank": i + 1, "code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j])} for i, j in enumerate(order) if np.isfinite(Sm[j])],
                    "date": str(E.dates[last].date())})
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


def export(res, outdir):
    """화면(대시보드)용 자료 파일 — {rows, meta} · meta 에 만든 코드."""
    os.makedirs(outdir, exist_ok=True)
    code = open(os.path.abspath(__file__), encoding="utf-8").read()
    meta = [{"label": "score_lab.py", "value": code, "type": "code", "lang": "python"},
            {"label": "입력", "value": "그때의 시총 상위 100 일봉·수급 (2002~2018 KRX 상장폐지 포함 · 2019~ 단기 패널), 코스피 지수, DM:2 (allweather)"},
            {"label": "비용", "value": "수수료 0.0186% + 미끄러짐 0.05% (사고팔 때 각각) + 거래세 0.20% (팔 때) = 왕복 약 0.34%"}]
    f = {}
    f["factors"] = [{"factor": r["factor"], "name": r["name"], **{k: r[k] for k in r if k[:2] in ("ic", "t1", "t5", "t2", "yr", "to")}} for r in res.get("stage1", [])]
    kind = lambda nm: "참고" if "참고" in nm else ("고정" if nm in FIXED or nm == "사전 부호 같은 비중" else "해마다 다시")   # noqa: E731
    f["formulas"] = [{"formula": nm, "kind": kind(nm), "pass": bool(v["pass"]), **{k: v["dev"][k] for k in ("cagr", "sharpe", "mdd", "excess_ann", "t_excess", "ew_sharpe", "trades_per_year", "hold_days", "win")},
                      "excess_x2": v["dev_x2"]["excess_ann"]} for nm, v in res.get("stage2", {}).items()]
    f["rules1"] = [{"formula": nm, "rule": rule_label(v["chosen"]["rule"]), "plateau": v["chosen"]["plateau"], **{k: v["chosen"]["dev"][k] for k in ("cagr", "sharpe", "mdd", "trades_per_year", "hold_days")},
                    "chosen": nm == res.get("final", {}).get("score")} for nm, v in res.get("stage3", {}).items() if v.get("chosen")]
    fin = res.get("final")
    if fin:
        f["final1"] = [{"who": "고른 점수 + 규칙", **{k: fin["hold"][k] for k in ("cagr", "sharpe", "mdd")}, "excess_ann": fin["hold"]["excess_ann"]},
                       {"who": "대형주 100 평균", "cagr": fin["hold"]["ew_cagr"], "sharpe": fin["hold"]["ew_sharpe"], "mdd": None, "excess_ann": 0.0},
                       {"who": "코스피", **fin["kospi"], "excess_ann": None}, {"who": "DM:2", **fin["dm2"], "excess_ann": None}]
    f["diagnosis"] = res.get("diagnosis", [])
    r2 = res.get("round2")
    if r2:
        c = r2["curve"]
        f["system_curve"] = [{"date": d, **{k: c[k][i] for k in ("system", "ew", "kospi", "dm2")}} for i, d in enumerate(c["date"])]
        names = {"system": "해마다 다시 고르는 체계", "system_x2": "같은 체계 · 비용 2배", "ew": "대형주 100 평균", "kospi": "코스피", "dm2": "DM:2"}
        f["system_table"] = [{"period": pn, "key": k, "who": names[k], **(st or {})} for pn, t in r2["table"].items() for k, st in t.items() if st]
        f["system_years"] = [{"year": int(y), **v} for y, v in r2["years"].items()]
        f["picks"] = r2["picks"]
        f["sensitivity"] = [{"setting": k, "period": pn, **st} for k, v in r2["sensitivity"].items() for pn, st in v.items()]
        card = r2["card"]
        if card:
            c0 = card[0]
            f["card_weights"] = [{"factor": k, "name": c0["weight_names"].get(k, k), "weight": w} for k, w in c0["weights"].items()]
            f["card_rules"] = [{"n": i + 1, "formula": x["score"], "rule": x["rule_text"], "slots": x["rule"]["slots"]} for i, x in enumerate(card)]
            f["card_hold"] = [{"n": i + 1, "rule": x["rule_text"], **h} for i, x in enumerate(card) for h in x["holding"]]
            f["card_top"] = [{**t, "date": c0["date"]} for t in c0["top_today"]]
    for k, rows in f.items():
        json.dump({"rows": rows, "meta": meta}, open(os.path.join(outdir, f"{k}.json"), "w", encoding="utf-8"), ensure_ascii=False, default=float)
    return list(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--export", default=None, help="화면용 자료 파일을 쓸 폴더 (계산 없이)")
    a = ap.parse_args()
    if a.export:
        print(export(load(), a.export))
        return
    log = lambda m: print(m, flush=True)                     # noqa: E731
    t0 = time.time()
    set_factors(a.stage == "r2")
    ER = eras(log)
    res = load()
    res.update({"generated": time.strftime("%Y-%m-%d %H:%M"), "factors": {k: factor_name(k) for k in BASE}, "factors2": {k: factor_name(k) for k in BASE + list(EXTRA)},
                "R0": R0, "grid": GRID,
                "periods": {"pre": PRE, "dev": DEV, "hold": HOLD}, "crit": CRIT})
    st = a.stage
    if st == "diag":
        log("■ 1차 탈락 원인 진단 (고르는 데 안 씀)")
        res["diagnosis"] = diagnose(ER, res, log)
        save(res)
        return
    if st == "r2":
        log("■ 2차: 해마다 과거로만 다시 고르는 체계")
        res["round2"] = round2(ER, log)
        res["secs"] = round(time.time() - t0)
        save(res)
        log(f"끝 ({res['secs']}초) → {OUT}")
        return
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
