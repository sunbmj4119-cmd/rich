"""
다분야 가설 검증실 — 논문·책·강연에서 나온 아이디어를 '우리 자료'로 같은 잣대에 올린다.

  잣대 (모든 가설 공통)
    · 미래 정보 없음: t일 신호는 t일 장 마감까지의 값, 체결은 그다음 거래일.
    · 비용 포함: ETF 호가 반 칸+수수료(allweather.costs), 주식 왕복 0.34%(국내)·0.26%(미국).
    · 기준과 비교: 같은 기간 '아무것도 안 하고 들고 있기'와의 샤프 차이를 블록 부트스트랩 P 로.
    · 여러 가설을 한꺼번에 봤으니 Holm 보정 P(가족 오류율 5%)로 최종 판정 — 하나쯤 우연히 좋게 나오는 것을 걸러낸다.
    · 개발(~2020/2022)·확인(이후) 두 기간 모두 같은 방향이어야 '통과'.
  과최적화 판정 (통계학)
    · Deflated Sharpe Ratio (Bailey & López de Prado 2014): 시도한 모델 수·분포·꼬리를 반영한 '진짜일 확률'.
    · PBO (Bailey, Borwein, López de Prado, Zhu 2017, CSCV): 과거 1등을 골랐을 때 앞으로 중간 아래로 떨어질 확률.

  python src/science_lab.py  → data/science/lab.json · docs/science.json
"""
import itertools
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allweather as AW  # noqa: E402
import short_lab as L  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = "data/science/lab.json"
REG = "data/science/registry.json"          # 가설마다 처음 검증한 날 — 그 뒤 성적(앞으로의 성적)을 따로 센다
UI = "docs/science.json"
EULER = 0.5772156649


# ══════════════════════════════════════════════════════════════
# 통계 도구
# ══════════════════════════════════════════════════════════════
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _nppf(p):
    """표준정규 분위수 (Acklam 근사 — scipy 없이)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549671373533484e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def psr(r, sr0=0.0):
    """Probabilistic Sharpe Ratio: 일별 수익 r 의 (하루 단위) 샤프가 sr0 보다 클 확률 — 꼬리·비대칭 반영."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 30 or r.std() == 0:
        return None
    sr = r.mean() / r.std()
    z = (r - r.mean()) / r.std()
    g3, g4 = float((z ** 3).mean()), float((z ** 4).mean())
    den = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr * sr))
    return _ncdf((sr - sr0) * math.sqrt(T - 1) / den)


def dsr(r_best, all_sr_daily):
    """Deflated Sharpe Ratio: N 개를 시도해 1등을 골랐다는 사실을 반영한 기준선 SR0 로 PSR.
    SR0 = sqrt(V[SR]) · ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))."""
    s = np.asarray([x for x in all_sr_daily if np.isfinite(x)])
    N = len(s)
    if N < 2:
        return None, None
    sr0 = math.sqrt(s.var()) * ((1 - EULER) * _nppf(1 - 1 / N) + EULER * _nppf(1 - 1 / (N * math.e)))
    return psr(r_best, sr0), sr0 * math.sqrt(252)


def pbo(R, S=16, max_combos=4000, seed=0):
    """Probability of Backtest Overfitting (CSCV). R: T×N 일별 수익 (NaN 은 0 취급).
    시간을 S 토막으로 나눠 절반 조합마다 '앞 절반 1등'이 '뒤 절반'에서 몇 등인지 → 중간 아래로 떨어진 비율."""
    X = np.nan_to_num(np.asarray(R, float))
    T, N = X.shape
    edges = np.linspace(0, T, S + 1).astype(int)
    s1 = np.array([X[a:b].sum(0) for a, b in zip(edges[:-1], edges[1:])])
    s2 = np.array([(X[a:b] ** 2).sum(0) for a, b in zip(edges[:-1], edges[1:])])
    n = np.array([b - a for a, b in zip(edges[:-1], edges[1:])])
    combos = list(itertools.combinations(range(S), S // 2))
    if len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]

    def sharpe(idx):
        m = s1[idx].sum(0) / n[idx].sum()
        v = s2[idx].sum(0) / n[idx].sum() - m * m
        return m / np.sqrt(np.maximum(v, 1e-18))
    lam, deg = [], []
    allb = set(range(S))
    for J in combos:
        Jc = sorted(allb - set(J))
        si, so = sharpe(list(J)), sharpe(Jc)
        b = int(np.argmax(si))
        w = (np.sum(so < so[b]) + 0.5 * (np.sum(so == so[b]) - 1) + 1) / (N + 1)
        lam.append(math.log(w / (1 - w)))
        deg.append((si[b], so[b]))
    lam = np.array(lam)
    d = np.array(deg)
    slope = float(np.polyfit(d[:, 0], d[:, 1], 1)[0]) if len(d) > 2 else None
    return {"pbo": round(float((lam <= 0).mean()), 3), "combos": len(combos), "n": int(N),
            "oos_sharpe_of_is_best_med": round(float(np.median(d[:, 1]) * math.sqrt(252)), 3),
            "is_sharpe_best_med": round(float(np.median(d[:, 0]) * math.sqrt(252)), 3), "slope": round(slope, 3) if slope is not None else None}


def holm(ps):
    """Holm-Bonferroni 보정 P (None 은 건너뜀)."""
    idx = [i for i, p in enumerate(ps) if p is not None]
    order = sorted(idx, key=lambda i: ps[i])
    m = len(order)
    out = [None] * len(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = round(run, 4)
    return out


def sh(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 20 and r.std() > 0 else float("nan")


def stats_r(r, dates, a=None, b=None):
    E = np.cumprod(1 + np.nan_to_num(r))
    return L.stats(E, dates, a, b)


def compare(name, field, source, rule, r_new, r_base, dates, split, base_name, extra=None):
    """가설 전략 r_new 와 기준 r_base 비교 (같은 날짜축). 두 기간 모두 샤프가 나아야 '같은 방향'."""
    r_new, r_base = np.nan_to_num(r_new), np.nan_to_num(r_base)
    s = int(np.searchsorted(dates, pd.Timestamp(split)))
    dev_new, dev_base = sh(r_new[:s]), sh(r_base[:s])
    tst_new, tst_base = sh(r_new[s:]), sh(r_base[s:])
    p = L.boot_p(r_new, r_base)
    return {"_rn": r_new, "_rb": r_base, "_d": dates,
            "name": name, "field": field, "source": source, "rule": rule, "base": base_name,
            "new": stats_r(r_new, dates), "baseline": stats_r(r_base, dates),
            "dev": [round(dev_new, 3), round(dev_base, 3)], "test": [round(tst_new, 3), round(tst_base, 3)],
            "split": split, "both_periods": bool(dev_new > dev_base and tst_new > tst_base), "p": p,
            "period": [str(dates[0].date()), str(dates[-1].date())], **(extra or {})}


def cond_means(r, mask, dates):
    """조건이 참인 날 vs 아닌 날의 평균 일수익(bp)과 t 값."""
    r = np.asarray(r, float)
    ok = np.isfinite(r) & np.isfinite(mask.astype(float))
    a, b = r[ok & mask], r[ok & ~mask]
    if len(a) < 20 or len(b) < 20:
        return None
    se = math.sqrt(a.var() / len(a) + b.var() / len(b))
    return {"on_bp": round(float(a.mean() * 1e4), 2), "off_bp": round(float(b.mean() * 1e4), 2), "n_on": int(len(a)),
            "n_off": int(len(b)), "t": round(float((a.mean() - b.mean()) / se), 2) if se > 0 else None}


# ══════════════════════════════════════════════════════════════
# 자료
# ══════════════════════════════════════════════════════════════
class Etf:
    """KODEX 200 등 ETF (2014-08~). r_cc: 종가→종가, r_oc: 시가→종가, r_co: 전일 종가→시가."""
    def __init__(self):
        self.dates, self.Op, self.Cp = AW.load_prices()
        C = AW.ffill(self.Cp)
        self.C = C
        j = AW.IX["IDX"]
        c, o = C[:, j], self.Op[:, j]
        self.r_cc = np.r_[0, c[1:] / c[:-1] - 1]
        self.r_oc = np.where(np.isfinite(o), c / o - 1, 0.0)
        self.r_co = np.r_[0, o[1:] / c[:-1] - 1]
        self.cost = float(sum(AW.costs(np.array([c[-1]]))[i][0] for i in (0, 1)))   # 왕복


def overlay(r, w, cost_rt):
    """비중 w[t] (t-1일 저녁 결정, t일 수익에 적용)로 r 을 들고 있을 때의 일수익. 비중 바뀔 때 왕복 비용의 절반 × 변화량."""
    w = np.nan_to_num(np.asarray(w, float))
    dw = np.abs(np.diff(np.r_[0.0, w]))
    return w * np.nan_to_num(r) - dw * cost_rt / 2


def lag(x, k=1):
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan)
    y[k:] = x[:-k]
    return y


# ══════════════════════════════════════════════════════════════
# 가설들
# ══════════════════════════════════════════════════════════════
def h_vol_managed(name, field, source, r, dates, cost_rt, split, base_name, look=21, cap=1.0):
    """변동성 관리 (Moreira & Muir 2017; Barroso & Santa-Clara 2015): 비중 = min(1, 목표 변동성 / 최근 변동성).
    목표 = 그때까지의 변동성 중앙값 (미래 정보 없음). 비중 변화가 0.1 넘을 때만 바꿈."""
    s = pd.Series(np.nan_to_num(r))
    vol = s.rolling(look, min_periods=look).std().to_numpy()
    tgt = pd.Series(vol).expanding(min_periods=look * 3).median().to_numpy()
    raw = np.minimum(cap, tgt / vol)
    w = np.full(len(r), np.nan)
    cur = 1.0
    for t in range(len(r)):
        x = raw[t]
        if np.isfinite(x) and abs(x - cur) > 0.1:
            cur = x
        w[t] = cur
    rn = overlay(r, lag(w), cost_rt)
    rn[0] = 0
    return compare(name, field, source, f"비중 = min(1, 그때까지 변동성 중앙값 ÷ 최근 {look}일 변동성), 0.1 넘게 바뀔 때만 조정",
                   rn, np.nan_to_num(r), dates, split, base_name, {"avg_weight": round(float(np.nanmean(w)), 3)})


def absorption_ratio(D, win=250, frac=0.2):
    """흡수비율 (Kritzman, Li, Page, Rigobon 2011): 최근 win일 상위 100종목 수익 공분산에서 상위 20% 고유값이 설명하는 몫."""
    C = D.P["C"]
    r = C[1:] / C[:-1] - 1
    r = np.vstack([np.full((1, C.shape[1]), np.nan), r])
    r[D.P["breaks"]] = np.nan
    T = len(r)
    ar = np.full(T, np.nan)
    for t in range(win, T, 1):
        X = r[t - win + 1: t + 1]
        ok = (np.isfinite(X).mean(0) > 0.9) & D.P["U"][t]
        X = np.nan_to_num(X[:, ok])
        if X.shape[1] < 30:
            continue
        X = X - X.mean(0)
        ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
        k = max(1, int(round(frac * len(ev))))
        ar[t] = ev[:k].sum() / ev.sum()
    return ar


def h_absorption(D, E, split):
    ar = absorption_ratio(D)
    s = pd.Series(ar)
    dar = ((s.rolling(15, min_periods=10).mean() - s.rolling(250, min_periods=120).mean()) / s.rolling(250, min_periods=120).std()).to_numpy()
    # KR 패널 날짜 → ETF 날짜로
    sig = pd.Series(dar, index=D.dates).reindex(E.dates).ffill().to_numpy()
    on = np.where(np.isfinite(sig), sig > 1.0, False)
    w = np.where(on, 0.0, 1.0)
    m = E.dates >= D.dates[300]
    rn = overlay(E.r_cc, lag(w), E.cost)
    res = compare("흡수비율 급등 때 현금 (위험 쏠림 경보)", "물리학 · 랜덤행렬이론", "Kritzman et al. 2011 · Laloux et al. 1999",
                  "상위 100종목 250일 공분산의 상위 20% 고유값 몫(흡수비율)이 15일 평균으로 1년 평균보다 1 표준편차 넘게 오르면 다음 날 KODEX 200 → 현금",
                  rn[m], E.r_cc[m], E.dates[m], split, "KODEX 200 보유",
                  {"cond": cond_means(E.r_cc[m], lag(on.astype(float))[m] > 0.5, E.dates[m]), "days_off": round(float(on[m].mean()), 3),
                   "fwd20_corr": round(float(pd.Series(sig[m]).corr(pd.Series(E.C[:, AW.IX["IDX"]][m]).pct_change(20).shift(-20))), 3)})
    return res, ar


def h_slowing_down(E, split, win=60):
    """임계 둔화 (Scheffer et al. 2009): 체제 전환 직전 자기상관·분산이 함께 오른다 → 둘 다 1년 대비 높으면 현금."""
    r = pd.Series(E.r_cc)
    ac = r.rolling(win).apply(lambda x: pd.Series(x).autocorr(1), raw=True)
    var = r.rolling(win).var()
    z = lambda x: (x - x.rolling(250, min_periods=120).mean()) / x.rolling(250, min_periods=120).std()  # noqa: E731
    on = ((z(ac) > 1) & (z(var) > 1)).to_numpy()
    w = np.where(on, 0.0, 1.0)
    rn = overlay(E.r_cc, lag(w), E.cost)
    m = np.arange(len(E.dates)) > 300
    return compare("자기상관·분산 동반 상승 때 현금 (임계 둔화 경보)", "생태학·물리학 · 임계 전이", "Scheffer et al. 2009 (Nature) · Dakos et al. 2012",
                   f"KODEX 200 일수익의 {win}일 1차 자기상관과 분산이 둘 다 1년 대비 +1 표준편차 넘으면 다음 날 현금",
                   rn[m], E.r_cc[m], E.dates[m], split, "KODEX 200 보유",
                   {"cond": cond_means(E.r_cc[m], lag(on.astype(float))[m] > 0.5, E.dates[m]), "days_off": round(float(on[m].mean()), 3)})


def family_returns(D, models, recs, fam):
    idx = [i for i, m in enumerate(models) if m["kind"] == "stock" and m["sel"] in fam and m["hold"] in ("H5", "H10")]
    R = np.c_[[L.rets(recs[i]["E"]) for i in idx]].T
    return np.nanmean(R, 1)


def variance_ratio(D, win=120, q=5):
    """종목별 분산비 VR(q) = Var(q일 수익)/(q·Var(1일 수익)) 의 단면 중앙값 (Lo & MacKinlay 1988). <1 = 되돌림 우세."""
    C = AW.ffill(D.P["C"])
    lr = np.log(C)
    r1 = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(lr, axis=0)])
    rq = np.vstack([np.full((q, C.shape[1]), np.nan), lr[q:] - lr[:-q]])
    v1 = pd.DataFrame(r1).rolling(win, min_periods=win // 2).var().to_numpy()
    vq = pd.DataFrame(rq).rolling(win, min_periods=win // 2).var().to_numpy()
    vr = np.where(D.P["U"], vq / (q * v1), np.nan)
    return np.nanmedian(vr, 1)


def h_vr_switch(D, rev, mom, split):
    vr = variance_ratio(D)
    on_rev = lag(vr < 1.0)
    w_rev = np.where(np.isfinite(on_rev), on_rev, 0.5)
    rn = w_rev * np.nan_to_num(rev) + (1 - w_rev) * np.nan_to_num(mom)
    base = 0.5 * np.nan_to_num(rev) + 0.5 * np.nan_to_num(mom)
    m = np.arange(D.T) > 250
    return compare("되돌림·추세 갈아타기 (분산비)", "통계학 · 분산비 검정 / 물리학 · 허스트 지수", "Lo & MacKinlay 1988 · Mandelbrot (허스트)",
                   "상위 100종목 120일 분산비 VR(5) 중앙값이 1 미만(되돌림 우세)이면 반전 모델 묶음, 1 이상이면 추세 모델 묶음 (5·10일 보유 모델 평균)",
                   rn[m], base[m], D.dates[m], "2022-12-31", "반전·추세 반반",
                   {"share_rev": round(float(np.nanmean(w_rev[m])), 3), "vr_now": round(float(vr[-1]), 3)})


def h_nagel(D, rev, split):
    """조건부 반전 (Nagel 2012): 시장 변동성이 높을 때 단기 반전 수익이 크다 → 변동성 높은 날에만 반전 묶음."""
    idx = np.cumprod(1 + np.nan_to_num(L.SP.market(D.P)["ret_cw"]))
    vol = pd.Series(idx).pct_change().rolling(20, min_periods=15).std().to_numpy()
    med = pd.Series(vol).expanding(min_periods=120).median().to_numpy()
    hi = vol > med
    w = np.where(lag(hi.astype(float)) > 0.5, 1.0, 0.0)
    rn = w * np.nan_to_num(rev) - np.abs(np.diff(np.r_[0, w])) * sum(L.costs("stock")) / 2
    m = np.arange(D.T) > 250
    return compare("변동성 높을 때만 반전 매매", "경제학 · 유동성 공급 보상", "Nagel 2012 (RFS) 'Evaporating Liquidity'",
                   "시장 20일 변동성이 그때까지 중앙값보다 높은 다음 날만 반전 모델 묶음 보유, 아니면 현금",
                   rn[m], np.nan_to_num(rev)[m], D.dates[m], "2022-12-31", "반전 묶음 항상",
                   {"cond": cond_means(np.nan_to_num(rev)[m], lag(hi.astype(float))[m] > 0.5, D.dates[m])})


def calendar_tests(E, split, spy=None):
    out = []
    d = E.dates
    ym = d.year * 12 + d.month
    first = np.r_[True, ym[1:] != ym[:-1]]
    last = np.r_[ym[1:] != ym[:-1], False]
    # 월말·월초 (Lakonishok & Smidt 1988; Ariel 1987): 마지막 거래일 + 첫 3거래일
    pos = np.zeros(len(d), int)
    k = 0
    for t in range(len(d)):
        k = 0 if first[t] else k + 1
        pos[t] = k
    tom = last | (pos <= 2)
    for nm, r, dd, base in (("KODEX 200", E.r_cc, d, "KODEX 200 보유"),):
        w = tom.astype(float)
        rn = overlay(r, w, E.cost)
        out.append(compare(f"월말·월초에만 보유 ({nm})", "경제학 · 달력 효과 (급여·자금 유입)", "Lakonishok & Smidt 1988 · Ariel 1987 · McConnell & Xu 2008",
                           "매달 마지막 거래일과 첫 3거래일에만 KODEX 200 보유, 나머지 날 현금", rn, r, dd, split, base,
                           {"cond": cond_means(r, tom, dd)}))
    # 명절·연휴 전날 (Ariel 1990)
    gap = np.r_[(d[1:] - d[:-1]).days, 1]
    pre = (gap > 3) | ((gap > 1) & (d.dayofweek < 4))
    out.append(compare("연휴 전날만 보유", "행동경제학 · 연휴 전 낙관", "Ariel 1990 · Kim & Park 1994",
                       "다음 거래일까지 쉬는 날이 끼는(주말 제외 휴장) 전날 종가 매수 → 연휴 뒤 첫날 종가 매도", overlay(E.r_cc, lag(pre.astype(float)), E.cost),
                       E.r_cc, d, split, "KODEX 200 보유", {"cond": cond_means(E.r_cc, lag(pre.astype(float)) > 0.5, d), "n_events": int(pre.sum())}))
    # 달의 위상 (Yuan, Zheng, Zhu 2006; Dichev & Janes 2003)
    jd = (d.values.astype("datetime64[s]").astype(np.int64) / 86400.0) + 2440587.5
    phase = ((jd - 2451550.1) / 29.530588853) % 1.0
    newm = (phase < 0.1) | (phase > 0.9)
    out.append(compare("초승달 무렵만 보유 (달의 위상)", "천문학 · 생체 리듬 가설", "Yuan, Zheng & Zhu 2006 · Dichev & Janes 2003",
                       "새달 ±3일 안의 날만 KODEX 200 보유", overlay(E.r_cc, lag(newm.astype(float)), E.cost), E.r_cc, d, split, "KODEX 200 보유",
                       {"cond": cond_means(E.r_cc, lag(newm.astype(float)) > 0.5, d)}))
    # 핼러윈 (Bouman & Jacobsen 2002): 11~4월만 보유
    hw = np.isin(d.month, [11, 12, 1, 2, 3, 4])
    out.append(compare("11~4월만 보유 (Sell in May)", "경제학 · 계절성", "Bouman & Jacobsen 2002 · Jacobsen & Zhang 2012",
                       "11월~4월에는 KODEX 200, 5월~10월에는 현금", overlay(E.r_cc, lag(hw.astype(float)), E.cost), E.r_cc, d, split, "KODEX 200 보유",
                       {"cond": cond_means(E.r_cc, lag(hw.astype(float)) > 0.5, d)}))
    return out


def h_foreign(D, E, split):
    """외국인 수급 (Choe, Kho & Stulz 1999; Richards 2005): 상위 100종목 외국인 5일 순매수 합(시총 대비) > 0 이면 보유."""
    F = np.where(D.P["U"], D.P["FOR"], np.nan)
    mc = np.where(D.P["U"], D.P["MC"], np.nan)
    agg = np.nansum(F, 1) / np.nansum(mc, 1)
    s5 = pd.Series(agg).rolling(5, min_periods=4).sum()
    sig = pd.Series(s5.to_numpy(), index=D.dates).reindex(E.dates).ffill().to_numpy()
    on = np.where(np.isfinite(sig), sig > 0, True)
    m = E.dates >= D.dates[20]
    rn = overlay(E.r_cc, lag(on.astype(float)), E.cost)
    return compare("외국인 5일 순매수일 때만 보유", "경제학 · 정보 비대칭·가격 압력 (한국 시장)", "Choe, Kho & Stulz 1999 · Richards 2005 · Kim & Wei 2002",
                   "상위 100종목 외국인 5일 순매수 합(시총 대비)이 플러스인 다음 날만 KODEX 200 보유", rn[m], E.r_cc[m], E.dates[m], split, "KODEX 200 보유",
                   {"cond": cond_means(E.r_cc[m], lag(on.astype(float))[m] > 0.5, E.dates[m])})


def h_us_lead(E, split):
    """미국 → 한국 (전이 엔트로피·선행 관계): 전날 밤 SPY 가 크게 오르면 한국은 시가에 이미 반영 → 장중에는 되돌림?"""
    import us_lab as UL
    P = UL.load_panel(min_date="2014-01-01")
    spy = pd.Series(P["bench"]["SPY"], index=P["dates"]).pct_change()
    # 한국 t일 장중(시가→종가) ↔ 그 전 미국 거래일 수익 (미국 날짜 < 한국 날짜)
    s = spy.reindex(spy.index.union(E.dates)).ffill().shift(1).reindex(E.dates).to_numpy()
    big = np.where(np.isfinite(s), s > 0.01, False)
    inv = E.Op[:, AW.IX["INV"]]
    inv_oc = np.where(np.isfinite(inv), AW.ffill(E.Cp)[:, AW.IX["INV"]] / inv - 1, 0)
    cost = float(sum(AW.costs(np.array([AW.ffill(E.Cp)[-1, AW.IX["INV"]]]))[i][0] for i in (0, 1)))
    rn = np.where(big, inv_oc - cost, 0.0)
    m = np.isfinite(s)
    res = compare("미국 급등 다음 날 장중 인버스", "정보이론 · 선행·전이 (미국→한국)", "Eun & Shim 1989 · Schreiber 2000 (전이 엔트로피)",
                  "전날 밤 SPY 가 +1% 넘게 오르면 한국 시가에 KODEX 인버스를 사서 그날 종가에 판다 (아니면 현금)",
                  rn[m], np.zeros(m.sum()), E.dates[m], split, "현금",
                  {"cond_oc": cond_means(E.r_oc[m], big[m], E.dates[m]), "cond_gap": cond_means(E.r_co[m], big[m], E.dates[m]),
                   "n_events": int(big[m].sum())})
    res["p"] = L.boot_p(rn[m], np.zeros(m.sum()) + 1e-12 * np.random.default_rng(0).standard_normal(m.sum()))
    return res


def h_pairs(D, split):
    """우선주·보통주 괴리 되돌림 (화학의 평형 / Ornstein–Uhlenbeck 반감기, Gatev et al. 2006).
    공매도 없이: 우선주가 보통주 대비 60일 평균보다 2σ 싸지면 우선주 매수, 괴리가 평균으로 돌아오면 매도."""
    pairs = [("005935", "005930"), ("005385", "005380"), ("009155", "009150"), ("000155", "000150")]
    codes = D.P["codes"]
    C = AW.ffill(D.P["C"])
    O = D.P["O"]
    cb, cs = L.costs("stock")
    rets, base, info = [], [], []
    for p, c in pairs:
        if p not in codes or c not in codes:
            continue
        jp, jc = codes.index(p), codes.index(c)
        sp = np.log(C[:, jp]) - np.log(C[:, jc])
        s = pd.Series(sp)
        z = ((s - s.rolling(60, min_periods=40).mean()) / s.rolling(60, min_periods=40).std()).to_numpy()
        hl = None
        ds = s.diff().dropna()
        lagv = s.shift(1).dropna().loc[ds.index]
        if len(ds) > 100:
            beta = np.polyfit(lagv - lagv.mean(), ds, 1)[0]
            hl = round(float(-math.log(2) / beta), 1) if beta < 0 else None
        pos = 0
        r = np.zeros(D.T)
        n_tr = 0
        for t in range(61, D.T):
            po, pc_prev = O[t, jp], C[t - 1, jp]
            if pos and np.isfinite(po):
                r[t] = C[t, jp] / C[t - 1, jp] - 1
            if not pos and np.isfinite(z[t - 1]) and z[t - 1] < -2 and np.isfinite(po):
                pos = 1
                n_tr += 1
                r[t] = C[t, jp] / po - 1 - cb
            elif pos and np.isfinite(z[t - 1]) and z[t - 1] > 0 and np.isfinite(po):
                pos = 0
                r[t] = po / pc_prev - 1 - cs
        rets.append(r)
        base.append(np.r_[0, C[1:, jp] / C[:-1, jp] - 1])
        info.append({"pair": f"{p}/{c}", "trades": n_tr, "half_life_days": hl})
    rn = np.nanmean(np.c_[rets].T, 1) if rets else np.zeros(D.T)
    rb = np.nanmean(np.c_[base].T, 1) if base else np.zeros(D.T)
    m = np.arange(D.T) > 61
    return compare("우선주 괴리 되돌림", "화학 · 평형과 반감기 / 통계학 · 공적분", "Gatev, Goetzmann & Rouwenhorst 2006 · Uhlenbeck & Ornstein 1930",
                   "우선주/보통주 가격비(로그)가 60일 평균보다 2σ 낮아지면 다음 날 시가에 우선주 매수, 평균 위로 돌아오면 매도 (4쌍 균등)",
                   rn[m], rb[m], D.dates[m], "2022-12-31", "우선주 4종 보유", {"pairs": info})


def h_herding(D, E, split):
    """군집 (Chang, Cheng & Khorana 2000 CSAD): 시장이 크게 움직이는데 종목 간 흩어짐이 평소보다 작으면 '떼 지어 움직임'
    → 그다음 5일 지수가 되돌아가나? 군집 후 다음 날부터 5일 현금."""
    C = D.P["C"]
    r = np.vstack([np.full((1, C.shape[1]), np.nan), C[1:] / C[:-1] - 1])
    r[D.P["breaks"]] = np.nan
    r = np.where(D.P["U"], r, np.nan)
    rm = np.nanmean(r, 1)
    csad = np.nanmean(np.abs(r - rm[:, None]), 1)
    X = pd.DataFrame({"csad": csad, "a": np.abs(rm), "q": rm ** 2})
    resid = np.full(len(csad), np.nan)
    for t in range(250, len(csad)):            # 그때까지 250일로 회귀 → 오늘 잔차 (미래 정보 없음)
        w = X.iloc[t - 250:t].dropna()
        A = np.c_[np.ones(len(w)), w["a"], w["q"]]
        coef = np.linalg.lstsq(A, w["csad"].to_numpy(), rcond=None)[0]
        resid[t] = csad[t] - np.array([1, abs(rm[t]), rm[t] ** 2]) @ coef
    sd = pd.Series(resid).rolling(250, min_periods=120).std().to_numpy()
    herd = (resid < -1.5 * sd) & (np.abs(rm) > pd.Series(np.abs(rm)).rolling(250, min_periods=120).quantile(0.8).to_numpy())
    sig = pd.Series(herd.astype(float), index=D.dates).reindex(E.dates).fillna(0).to_numpy()
    off = pd.Series(sig).rolling(5, min_periods=1).max().to_numpy() > 0
    m = E.dates >= D.dates[300]
    rn = overlay(E.r_cc, lag(np.where(off, 0.0, 1.0)), E.cost)
    return compare("군집 뒤 5일 현금", "생물학 · 무리 행동 / 행동재무", "Chang, Cheng & Khorana 2000 · Christie & Huang 1995",
                   "종목 간 흩어짐(CSAD)이 시장 움직임에 비해 1.5σ 넘게 작은(떼 지어 움직인) 큰 움직임의 날 뒤 5거래일 KODEX 200 → 현금",
                   rn[m], E.r_cc[m], E.dates[m], split, "KODEX 200 보유", {"events": int(herd[300:].sum())})


def fip_signal(P, look=250, skip=5):
    """정보 이산성 (Da, Gurun & Warachka 2014 'Frog in the Pan'): ID = sign(누적수익) × (하락일 비율 − 상승일 비율).
    값이 클수록(= -ID) 조금씩 꾸준히 오른 추세 → 덜 알아챈 정보라 오래 간다."""
    C = P["C"]
    r = np.vstack([np.full((1, C.shape[1]), np.nan), C[1:] / C[:-1] - 1])
    pos = pd.DataFrame((r > 0).astype(float)).rolling(look, min_periods=look // 2).mean().shift(skip).to_numpy()
    neg = pd.DataFrame((r < 0).astype(float)).rolling(look, min_periods=look // 2).mean().shift(skip).to_numpy()
    pret = pd.DataFrame(C).shift(skip).to_numpy() / pd.DataFrame(C).shift(look).to_numpy() - 1
    return -(np.sign(pret) * (neg - pos))


def h_fip(Dx, label, cost_kind, split, start, base_k=10):
    Dx.S["fip"] = fip_signal(Dx.P)
    base = {"id": "base", "kind": "stock", "sel": "MOM", "K": base_k, "hold": "BAND", "gate": "none"}
    new = {**base, "id": "fip", "sel": "MOM_FIP", "combo": [("mom120", 1.0), ("hi52", 1.0), ("fip", 1.0)]}
    rb = L.rets(L.run_stock(Dx, base)["E"])[start:]
    rn = L.rets(L.run_stock(Dx, new)["E"])[start:]
    return compare(f"추세 + 꾸준함 (Frog in the Pan) — {label}", "행동재무·심리학 · 점진적 정보 반영", "Da, Gurun & Warachka 2014 (RFS) · Hong & Stein 1999",
                   "6개월 추세·52주 고가에 '조금씩 꾸준히 오른 정도'(상승일 비율 − 하락일 비율)를 같은 무게로 더해 상위 10종목",
                   rn, rb, Dx.dates[start:], split, f"추세 상위 10 ({label})")


def h_panic(DU, r_mom, s0, split):
    """모멘텀 붕괴 대비 (Daniel & Moskowitz 2016): 시장이 2년 전보다 낮고 변동성이 상위 20%인 '공포 국면'에는 추세 몫 절반."""
    spy = DU.P["bench"]["SPY"]
    s = pd.Series(spy)
    bear = (s < s.shift(504)).to_numpy()
    vol = s.pct_change().rolling(126, min_periods=60).std()
    hi = (vol > vol.expanding(min_periods=250).quantile(0.8)).to_numpy()
    w = np.where(bear & hi, 0.5, 1.0)
    rn = overlay(r_mom, lag(w), sum(L.costs("us")))
    return compare("공포 국면에 추세 절반 (미국)", "경제학 · 모멘텀 붕괴", "Daniel & Moskowitz 2016 (JFE) 'Momentum Crashes'",
                   "SPY 가 2년 전보다 낮고 126일 변동성이 그때까지 상위 20%면 미국 추세 상위 10 비중 50%, 아니면 100%",
                   rn[s0:], np.nan_to_num(r_mom)[s0:], DU.dates[s0:], split, "미국 추세 상위 10 그대로",
                   {"days_half": round(float((bear & hi)[s0:].mean()), 3)})


def h_canary(E, split):
    """카나리아 (Keller 2022 BAA): 여러 위험자산 중 하나라도 13612W 모멘텀이 마이너스면 방어(단기채)."""
    C = E.C
    keys = ["IDX", "KQ", "SPX", "B10"]
    mom = []
    for k in keys:
        s = pd.Series(C[:, AW.IX[k]])
        m = (12 * s.pct_change(21) + 4 * s.pct_change(63) + 2 * s.pct_change(126) + s.pct_change(252)) / 4
        mom.append(m.to_numpy())
    M = np.c_[mom].T
    ok = np.isfinite(M).all(1)
    risk_on = np.where(ok, (M > 0).all(1), True)
    cash = np.r_[0, C[1:, AW.IX["CASH"]] / C[:-1, AW.IX["CASH"]] - 1]
    w = lag(risk_on.astype(float))
    rn = np.nan_to_num(w) * E.r_cc + (1 - np.nan_to_num(w, nan=1)) * np.nan_to_num(cash) - np.abs(np.diff(np.r_[1, np.nan_to_num(w, nan=1)])) * E.cost
    m = np.arange(len(E.dates)) > 260
    return compare("카나리아 경보 때 단기채 (국내·코스닥·미국·장기채 중 하나라도 꺾이면)", "실무 · 자산배분", "Keller 2022 'Bold Asset Allocation' (SSRN 4166845)",
                   "KODEX 200·코스닥150·S&P500·국고채10년의 13612W 모멘텀이 모두 플러스일 때만 KODEX 200, 하나라도 마이너스면 단기채",
                   rn[m], E.r_cc[m], E.dates[m], split, "KODEX 200 보유", {"days_def": round(float((~risk_on[m]).mean()), 3)})


def hrp_weights(X):
    """계층 위험균형 (López de Prado 2016): 상관 거리로 군집 → 준대각 순서 → 둘로 나누며 분산 반비례 배분."""
    X = np.nan_to_num(X)
    cov = np.cov(X, rowvar=False)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    corr = cov / np.outer(sd, sd)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    n = len(sd)
    clusters = {i: [i] for i in range(n)}
    order_tree = {i: [i] for i in range(n)}
    active = list(range(n))
    nid = n
    dmat = {(i, j): dist[i, j] for i in range(n) for j in range(n) if i < j}
    while len(active) > 1:
        a, b = min(((i, j) for i in active for j in active if i < j), key=lambda p: dmat[p])
        clusters[nid] = clusters[a] + clusters[b]
        order_tree[nid] = order_tree[a] + order_tree[b]
        for k in active:
            if k in (a, b):
                continue
            dmat[(min(k, nid), max(k, nid))] = min(dmat[(min(k, a), max(k, a))], dmat[(min(k, b), max(k, b))])
        active = [k for k in active if k not in (a, b)] + [nid]
        nid += 1
    order = order_tree[active[0]]
    w = np.ones(n)
    stack = [order]
    while stack:
        c = stack.pop()
        if len(c) < 2:
            continue
        h = len(c) // 2
        c1, c2 = c[:h], c[h:]

        def cvar(cc):
            sub = cov[np.ix_(cc, cc)]
            iv = 1 / np.maximum(np.diag(sub), 1e-18)
            iv /= iv.sum()
            return iv @ sub @ iv
        v1, v2 = cvar(c1), cvar(c2)
        a1 = 1 - v1 / (v1 + v2)
        w[c1] *= a1
        w[c2] *= 1 - a1
        stack += [c1, c2]
    return w / w.sum()


def card_mix(R, rules, dates, how, extra=None):
    """전천후 혼합을 실제 운용과 같은 방식으로 재현: 단기채 초과수익 기준 변동성, 거의 안 움직이는 카드는 제외, 매달 첫 거래일.
    how = rp(지금) · hrp · eq · rp_tsm(12개월 성적 마이너스면 절반). extra: 추가 카드 수익 열."""
    if extra is not None:
        R = np.c_[R, extra]
    cash = rules.index("H:CASH")
    Rx = R - np.nan_to_num(R[:, cash])[:, None]
    keep = [i for i in range(R.shape[1]) if i != cash]
    s = int(np.searchsorted(dates, pd.Timestamp(AW.MIX_START)))
    newm = AW.month_start(dates)
    out = np.zeros(len(dates))
    w = None
    for t in range(s, len(dates)):
        if w is None or newm[t]:
            X = Rx[max(1, t - 120): t][:, keep]
            v = np.nanstd(X, 0)
            idx = [keep[i] for i in range(len(keep)) if np.isfinite(v[i]) and v[i] > 1e-6 and np.isfinite(X[:, i]).mean() > 0.9]
            Xs = np.nan_to_num(Rx[max(1, t - 120): t][:, idx])
            if how == "hrp":
                wv = hrp_weights(Xs)
            elif how == "eq":
                wv = np.ones(len(idx)) / len(idx)
            else:
                wv = 1 / Xs.std(0)
                if how == "rp_tsm":
                    m12 = np.array([np.prod(1 + np.nan_to_num(R[max(1, t - 252): t, i])) - 1 for i in idx])
                    wv = wv * np.where(m12 > 0, 1.0, 0.5)
                wv = wv / wv.sum()
            w = dict(zip(idx, wv))
        out[t] = sum(wt * np.nan_to_num(R[t, i]) for i, wt in w.items())
    return out, s


def h_hrp(split):
    """전천후 몫: 카드 변동성 반비례(지금) vs 계층 위험균형(HRP) — 매달, 그때까지 120일 자료로."""
    dates, Op, Cp = AW.load_prices()
    F = AW.features(Cp, dates)
    rules, Ws, recs, R = AW.card_returns(dates, Op, Cp, F)
    rp, st = card_mix(R, rules, dates, "rp")
    hrp, _ = card_mix(R, rules, dates, "hrp")
    eq, _ = card_mix(R, rules, dates, "eq")
    res = compare("전천후 몫을 계층 위험균형(HRP)으로", "수학 · 계층 군집·그래프 이론", "López de Prado 2016 (JPM) 'Building Diversified Portfolios that Outperform Out of Sample'",
                  "전략 카드의 120일 초과수익 상관으로 군집을 만들고 군집끼리 분산 반비례로 나눔 (매달)",
                  hrp[st:], rp[st:], dates[st:], split, "카드 변동성 반비례 (지금)")
    res["eq"] = stats_r(eq[st:], dates[st:])
    return res


def h_blend(D, models, recs, R, E_live, start, split):
    """예측 합치기 (Carver 2015; Bates & Granger 1969): 1등 하나 대신 최근 1년 성적이 플러스인 후보 전부를 같은 비중으로."""
    pool = L.pool_index(models, "SLOW")
    Rp = R[:, pool]
    r = np.zeros(D.T)
    w = None
    for t in range(start, D.T):
        if w is None or (t - start) % 21 == 0:
            sc = L.trailing_score(Rp, t - 1, 250, "sharpe")
            good = np.where(np.nan_to_num(sc, nan=-9) > 0)[0]
            w = good if len(good) else np.arange(Rp.shape[1])
        r[t] = np.nanmean(Rp[t, w])
    r_live = L.rets(E_live)
    return compare("1등 하나 대신 '좋은 후보 전부' 섞기 (단기 몫)", "통계학 · 예측 결합 / 실무", "Bates & Granger 1969 · Carver 2015 'Systematic Trading' · Timmermann 2006",
                   "매달, 최근 250일 샤프가 플러스인 단기 후보 모델 전부를 같은 비중으로 (모델별 비용 포함, 서로 상쇄되는 매매는 반영 안 함 = 보수적)",
                   r[start:], r_live[start:], D.dates[start:], split, "매주 1등 따라가기 (지금)")


def h_custom_stock(Dx, combo, K, hold, label, field, source, rule, start, split, base_model, base_name, prep=None):
    """새 신호 조합의 종목 모델 vs 기준 모델 (같은 엔진·같은 비용)."""
    if prep:
        prep(Dx)
    m = {"id": "x", "kind": "stock", "sel": "X", "K": K, "hold": hold, "gate": "none", "combo": combo}
    rn = L.rets(L.run_stock(Dx, m)["E"])[start:]
    rb = L.rets(L.run_stock(Dx, base_model)["E"])[start:]
    return compare(label, field, source, rule, rn, rb, Dx.dates[start:], split, base_name)


def h_lowvol(Dx, label, start, split):
    """저변동성 (Frazzini & Pedersen 2014; 한국: Pyo & Lee 2018): 60일 변동성 가장 낮은 10종목, 20일 보유."""
    def prep(D):
        C = D.P["C"]
        r = np.vstack([np.full((1, C.shape[1]), np.nan), C[1:] / C[:-1] - 1])
        D.S["lowvol60"] = -pd.DataFrame(r).rolling(60, min_periods=40).std().to_numpy()
    ew = {"id": "ew", "kind": "stock", "sel": "MOM", "K": 10, "hold": "BAND", "gate": "none"}
    return h_custom_stock(Dx, [("lowvol60", 1.0)], 10, "BAND", f"저변동성 10종목 ({label})", "경제학 · 저위험 이상현상 (레버리지 제약)",
                          "Frazzini & Pedersen 2014 (JFE) 'Betting Against Beta' · Pyo & Lee 2018 (PBFJ, KOSPI200)",
                          "60일 일변동성이 가장 낮은 10종목, 순위가 30위 밖으로 밀리면 교체 (최대 20일마다 점검)", start, split, ew, f"추세 상위 10 ({label})", prep)


def h_flow_cs(D, start, split, who="FOR", label="외국인"):
    """수급 단면 (Choe, Kho & Stulz 1999/2005; Kim, Kang & Roh 2025; 개인은 Kaniel, Saar & Titman 2008):
    5일 순매수 / 시총 상위 10종목을 10일 보유."""
    key = f"{who.lower()}5"

    def prep(Dx):
        Dx.S[key] = pd.DataFrame(Dx.P[who]).rolling(5, min_periods=4).sum().to_numpy() / Dx.P["MC"]
    base = {"id": "b", "kind": "stock", "sel": "REVC", "K": 10, "hold": "H10", "gate": "none"}
    src = ("Choe, Kho & Stulz 1999/2005 · Richards 2005 · Kim, Kang & Roh 2025 (PBFJ)" if who == "FOR"
           else "Kaniel, Saar & Titman 2008 (JF) · Kim & Park 2015 · Kwak 2024 (코로나 뒤 약화)")
    return h_custom_stock(D, [(key, 1.0)], 10, "H10", f"{label} 5일 순매수 상위 10 (10일 보유)", "경제학 · 정보거래·유동성 공급 (한국 수급)", src,
                          f"상위 100종목 중 {label} 5일 순매수(시총 대비) 상위 10종목을 다음 날 시가에 사서 10거래일 보유", start, split, base,
                          "반전 종합 10종목 10일 보유", prep)


def h_card_tsmom(split):
    """요인 모멘텀 (Ehsani & Linnainmaa 2022): 전략 카드도 자기 12개월 성적이 플러스면 제 몫, 마이너스면 절반."""
    dates, Op, Cp = AW.load_prices()
    F = AW.features(Cp, dates)
    rules, Ws, recs, R = AW.card_returns(dates, Op, Cp, F)
    rp, st = card_mix(R, rules, dates, "rp")
    ts, _ = card_mix(R, rules, dates, "rp_tsm")
    return compare("전천후 카드에 '자기 추세' 적용", "경제학 · 요인 모멘텀", "Ehsani & Linnainmaa 2022 (JF) 'Factor Momentum and the Momentum Factor'",
                   "매달 카드별 변동성 반비례 비중에, 그 카드의 최근 12개월 성적이 마이너스면 절반만",
                   ts[st:], rp[st:], dates[st:], split, "카드 변동성 반비례 (지금)")


def h_extra_cards(E, split):
    """새 카드 2장(난류 회피 KODEX 200 · 11~4월 KODEX 200)을 전천후 혼합에 더하면?"""
    dates, Op, Cp = AW.load_prices()
    F = AW.features(Cp, dates)
    rules, Ws, recs, R = AW.card_returns(dates, Op, Cp, F)
    C = E.C
    cash = np.r_[0, C[1:, AW.IX["CASH"]] / C[:-1, AW.IX["CASH"]] - 1]
    on = turbulence_flag(E)
    w_tb = np.nan_to_num(lag(np.where(on, 0.0, 1.0)), nan=1.0)
    hw = np.nan_to_num(lag(np.isin(dates.month, [11, 12, 1, 2, 3, 4]).astype(float)), nan=1.0)
    cards = []
    for w in (w_tb, hw):
        cards.append(w * E.r_cc + (1 - w) * np.nan_to_num(cash) - np.abs(np.diff(np.r_[1, w])) * E.cost / 2)
    base, st = card_mix(R, rules, dates, "rp")
    plus, _ = card_mix(R, rules, dates, "rp", extra=np.c_[cards].T)
    return compare("전천후 혼합에 '난류 회피'·'11~4월' 카드 추가", "통계물리 + 계절성", "Kritzman & Li 2010 · Bouman & Jacobsen 2002",
                   "전략 카드 21장에 'KODEX 200, 금융 난류 상위 10%면 단기채'와 '11~4월만 KODEX 200, 나머지 단기채' 2장을 더해 같은 방식으로 섞음",
                   plus[st:], base[st:], dates[st:], split, "지금 카드 21장")


def h_close_exec(Dx, label, model, start, split, cost_kind):
    """체결 시점 (Lou, Polk & Skouras 2019): 추세 수익은 밤사이에 난다 → 다음 날 시가 대신 다음 날 종가에 사고팔면?"""
    import copy
    Dc = copy.copy(Dx)
    Dc.O = Dx.Cf.copy()
    Dc.O[np.isnan(Dx.P["C"])] = np.nan
    rn = L.rets(L.run_stock(Dc, model)["E"])[start:]
    rb = L.rets(L.run_stock(Dx, model)["E"])[start:]
    return compare(f"추세 모델을 종가에 체결 ({label})", "경제학 · 밤사이 vs 장중 수익의 줄다리기", "Lou, Polk & Skouras 2019 (JFE) 'A Tug of War'",
                   "같은 추세 상위 10 모델을 다음 날 시가 대신 다음 날 종가에 사고팖", rn, rb, Dx.dates[start:], split, f"다음 날 시가 체결 ({label})")


def turbulence_flag(E):
    keys = ["IDX", "KQ", "B10", "B3", "USD", "GOLD", "SPX", "NDX"]
    C = E.C[:, [AW.IX[k] for k in keys]]
    r = np.vstack([np.full((1, len(keys)), np.nan), C[1:] / C[:-1] - 1])
    T = len(r)
    d = np.full(T, np.nan)
    for t in range(500, T):
        X = r[t - 500:t]
        X = X[np.isfinite(X).all(1)]
        if len(X) < 250 or not np.isfinite(r[t]).all():
            continue
        mu = X.mean(0)
        S = np.cov(X, rowvar=False) + 1e-10 * np.eye(len(keys))
        x = r[t] - mu
        d[t] = float(x @ np.linalg.solve(S, x))
    sm = pd.Series(d).rolling(10, min_periods=5).mean()
    return (sm > sm.expanding(min_periods=250).quantile(0.9)).to_numpy()


def h_turbulence(E, split):
    """금융 난류 (Kritzman & Li 2010): 자산들의 오늘 움직임이 평소 공분산에서 얼마나 '이상한가'(마할라노비스 거리)."""
    keys = ["IDX", "KQ", "B10", "B3", "USD", "GOLD", "SPX", "NDX"]
    C = E.C[:, [AW.IX[k] for k in keys]]
    r = np.vstack([np.full((1, len(keys)), np.nan), C[1:] / C[:-1] - 1])
    T = len(r)
    d = np.full(T, np.nan)
    for t in range(500, T):
        X = r[t - 500:t]
        ok = np.isfinite(X).all(1)
        X = X[ok]
        if len(X) < 250 or not np.isfinite(r[t]).all():
            continue
        mu = X.mean(0)
        S = np.cov(X, rowvar=False) + 1e-10 * np.eye(len(keys))
        x = r[t] - mu
        d[t] = float(x @ np.linalg.solve(S, x))
    sm = pd.Series(d).rolling(10, min_periods=5).mean()
    on = (sm > sm.expanding(min_periods=250).quantile(0.9)).to_numpy()
    w = np.where(on, 0.0, 1.0)
    rn = overlay(E.r_cc, lag(w), E.cost)
    m = np.arange(T) > 760
    return compare("금융 난류 때 현금", "통계물리 · 마할라노비스 거리", "Kritzman & Li 2010 (FAJ) 'Skulls, Financial Turbulence, and Risk Management'",
                   "국내주식·코스닥·장기/단기채·달러·금·미국 2종 ETF 의 오늘 움직임이 500일 공분산 기준으로 얼마나 이례적인지(10일 평균)가 그때까지 상위 10%면 다음 날 KODEX 200 → 현금",
                   rn[m], E.r_cc[m], E.dates[m], split, "KODEX 200 보유", {"days_off": round(float(on[m].mean()), 3)})


def h_us_underreact(E, split):
    """점진적 정보 확산 (Hong & Stein 1999): 미국이 크게 움직였는데 한국 시가가 덜 반영했으면 장중에 마저 따라간다?"""
    import us_lab as UL
    P = UL.load_panel(min_date="2014-01-01")
    spy = pd.Series(P["bench"]["SPY"], index=P["dates"]).pct_change()
    s = spy.reindex(spy.index.union(E.dates)).ffill().shift(1).reindex(E.dates).to_numpy()
    gap = E.r_co
    beta = np.full(len(gap), np.nan)
    for t in range(260, len(gap)):
        x, y = s[t - 250:t], gap[t - 250:t]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 150 and x[ok].var() > 0:
            beta[t] = np.cov(x[ok], y[ok])[0, 1] / x[ok].var()
    big = np.isfinite(s) & (np.abs(s) > 0.015) & np.isfinite(beta)
    under = big & (np.abs(gap) < 0.5 * np.abs(beta * s)) & (np.sign(gap) == np.sign(s))
    under |= big & (np.sign(gap) != np.sign(s))
    inv_o, inv_c = E.Op[:, AW.IX["INV"]], AW.ffill(E.Cp)[:, AW.IX["INV"]]
    inv_oc = np.where(np.isfinite(inv_o), inv_c / inv_o - 1, 0)
    cost = E.cost
    rn = np.where(under & (s > 0), E.r_oc - cost, np.where(under & (s < 0), inv_oc - cost, 0.0))
    m = np.arange(len(E.dates)) > 260
    res = compare("미국 큰 움직임을 한국 시가가 덜 반영하면 장중 따라가기", "정보이론·행동재무 · 점진적 정보 확산", "Hong & Stein 1999 (JF) · Schreiber 2000 (전이 엔트로피)",
                  "전날 SPY 가 ±1.5% 넘게 움직였는데 KODEX 200 시가 갭이 (250일 회귀 기울기 × SPY)의 절반도 안 되면 같은 방향(오르면 KODEX 200, 내리면 인버스)으로 시가→종가",
                  rn[m], np.zeros(m.sum()), E.dates[m], split, "현금", {"n_events": int(under[m].sum()),
                  "cond": cond_means(np.where(s > 0, E.r_oc, -E.r_oc)[m], under[m], E.dates[m])})
    res["p"] = L.boot_p(rn[m], 1e-12 * np.random.default_rng(1).standard_normal(m.sum()))
    return res


def h_capitulation(D, E, split):
    """투매 군집 (Christie & Huang 1995; Choi & Yoon 2020 — 한국은 하락장 군집): 큰 하락일에 종목이 한꺼번에 빠지면(흩어짐이 작으면)
    다음 날 시가에 사서 5일 보유 — 그냥 큰 하락일 다음 매수와 비교."""
    C = D.P["C"]
    r = np.vstack([np.full((1, C.shape[1]), np.nan), C[1:] / C[:-1] - 1])
    r[D.P["breaks"]] = np.nan
    r = np.where(D.P["U"], r, np.nan)
    rm = np.nanmean(r, 1)
    csad = np.nanmean(np.abs(r - rm[:, None]), 1)
    resid = np.full(len(csad), np.nan)
    for t in range(250, len(csad)):
        a, q, y = np.abs(rm[t - 250:t]), rm[t - 250:t] ** 2, csad[t - 250:t]
        ok = np.isfinite(a) & np.isfinite(y)
        A = np.c_[np.ones(ok.sum()), a[ok], q[ok]]
        coef = np.linalg.lstsq(A, y[ok], rcond=None)[0]
        resid[t] = csad[t] - np.array([1, abs(rm[t]), rm[t] ** 2]) @ coef
    sdr = pd.Series(rm).rolling(250, min_periods=120).std().to_numpy()
    sde = pd.Series(resid).rolling(250, min_periods=120).std().to_numpy()
    down = rm < -2 * sdr
    herd = down & (resid < -1.5 * sde)
    def hold5(ev):
        e = pd.Series(ev.astype(float), index=D.dates).reindex(E.dates).fillna(0).to_numpy()
        return pd.Series(lag(e)).rolling(5, min_periods=1).max().fillna(0).to_numpy()
    w_h, w_d = hold5(herd), hold5(down & ~herd)
    rn = overlay(E.r_cc, w_h, E.cost)
    rb = overlay(E.r_cc, w_d, E.cost)
    m = E.dates >= D.dates[260]
    return compare("한꺼번에 빠진 큰 하락일 뒤 5일 매수", "생물학 · 무리 행동 (투매)", "Christie & Huang 1995 · Chang, Cheng & Khorana 2000 · Choi & Yoon 2020",
                   "100종목 평균이 -2σ 넘게 빠졌는데 종목 간 흩어짐이 평소보다 1.5σ 작으면(한꺼번에 투매) 다음 날부터 5일 KODEX 200 보유",
                   rn[m], rb[m], E.dates[m], split, "흩어진 큰 하락일 뒤 5일 보유", {"n_herd": int(herd[260:].sum()), "n_down": int((down & ~herd)[260:].sum())})


def h_inst_market(D, E, split):
    """기관 수급 (Hong & Lee 2011; Yang et al. 2021 — 기관 순매수는 다음 날 플러스, 개인은 마이너스): 시장 전체 기관 순매수 z>1.5 면 3일 보유."""
    I = np.where(D.P["U"], D.P["INS"], np.nan)
    mc = np.where(D.P["U"], D.P["MC"], np.nan)
    agg = pd.Series(np.nansum(I, 1) / np.nansum(mc, 1))
    z = ((agg - agg.rolling(250, min_periods=120).mean()) / agg.rolling(250, min_periods=120).std()).to_numpy()
    ev = pd.Series((z > 1.5).astype(float), index=D.dates).reindex(E.dates).fillna(0).to_numpy()
    w = pd.Series(lag(ev)).rolling(3, min_periods=1).max().fillna(0).to_numpy()
    rn = overlay(E.r_cc, w, E.cost)
    m = E.dates >= D.dates[260]
    res = compare("기관이 시장 전체로 크게 산 다음 3일 보유", "경제학 · 정보를 가진 투자자 (한국 기관)", "Hong & Lee 2011 · Yang, Samitas & Kampouris 2021 (IJMF)",
                  "상위 100종목 기관 순매수 합(시총 대비)이 1년 기준 z>1.5 인 날 다음부터 3일 KODEX 200, 아니면 현금",
                  rn[m], E.r_cc[m] * 0, E.dates[m], split, "현금", {"cond": cond_means(E.r_cc[m], w[m] > 0.5, E.dates[m]), "days_in": round(float((w[m] > 0.5).mean()), 3)})
    res["p"] = L.boot_p(rn[m], 1e-12 * np.random.default_rng(2).standard_normal(m.sum()))
    return res


def h_comomentum(D, start, split):
    """혼잡 (Lou & Polk 2022 Comomentum; 생태학의 포식자–피식자): 추세 상위 종목끼리 너무 같이 움직이면(붐빔) 추세 몫 끔."""
    mom = {"id": "m", "kind": "stock", "sel": "MOM", "K": 10, "hold": "H10", "gate": "none"}
    rb = L.rets(L.run_stock(D, mom)["E"])
    C = AW.ffill(D.P["C"])
    r = np.vstack([np.full((1, C.shape[1]), np.nan), C[1:] / C[:-1] - 1])
    rm = np.nanmean(np.where(D.P["U"], r, np.nan), 1)
    ra = r - rm[:, None]
    rk = D.ranks("MOM")
    como = np.full(D.T, np.nan)
    for t in range(260, D.T, 5):
        top = np.where(rk[t] <= 10)[0]
        X = ra[t - 250:t][:, top]
        X = X[np.isfinite(X).all(1)]
        if len(X) < 100:
            continue
        cm = np.corrcoef(X, rowvar=False)
        como[t] = (cm.sum() - len(top)) / (len(top) * (len(top) - 1))
    cs = pd.Series(como).ffill()
    hi = (cs > cs.expanding(min_periods=250).quantile(0.8)).to_numpy()
    w = np.where(lag(hi.astype(float)) > 0.5, 0.0, 1.0)
    rn = overlay(rb, w, 0.0) - np.abs(np.diff(np.r_[1, w])) * sum(L.costs("stock")) / 2
    return compare("추세 종목이 붐빌 때 추세 몫 끄기", "생태학 · 포식자–피식자 / 경제학 · 전략 혼잡", "Lou & Polk 2022 (RFS) 'Comomentum' · Farmer 2002",
                   "추세 상위 10종목의 250일 시장 대비 수익 평균 상관이 그때까지 상위 20%면 추세 모델(10종목·10일 보유)을 현금으로",
                   rn[start:], rb[start:], D.dates[start:], split, "추세 10종목 10일 보유 그대로", {"days_off": round(float(np.nanmean(1 - w[start:])), 3)})


def h_max_filter(D, start, split):
    """복권 주식 (Bali, Cakici & Whitelaw 2011; 한국 개인의 복권 선호): 최근 21일 '가장 크게 오른 5일 평균'이 상위 20%인 종목은 반전 매수에서 뺀다."""
    def prep(Dx):
        C = Dx.P["C"]
        r = np.vstack([np.full((1, C.shape[1]), np.nan), C[1:] / C[:-1] - 1])
        mx = pd.DataFrame(r).rolling(21, min_periods=15).apply(lambda x: np.sort(x[np.isfinite(x)])[-5:].mean() if np.isfinite(x).sum() >= 5 else np.nan, raw=True).to_numpy()
        pr = pd.DataFrame(np.where(Dx.P["U"], mx, np.nan)).rank(axis=1, pct=True).to_numpy()
        Dx.S["nomax"] = np.where(pr > 0.8, -10.0, 0.0)
    base = {"id": "b", "kind": "stock", "sel": "REVC", "K": 10, "hold": "H10", "gate": "none"}
    combo = [(k, float(w)) for k, w in L.SELECTORS["REVC"]] + [("nomax", 1.0)]
    return h_custom_stock(D, combo, 10, "H10", "반전 매수에서 '복권 주식' 빼기", "행동재무·심리학 · 복권 선호", "Bali, Cakici & Whitelaw 2011 (JFE) · Barber & Odean 2008",
                          "반전 종합 순위는 그대로 두되, 최근 21일 상위 5일 평균 상승률이 상위 20%인 종목은 후보에서 제외 (10종목·10일 보유)",
                          start, split, base, "반전 종합 10종목 10일 보유", prep)


def h_pref_switch(D, split):
    """우선주↔보통주 갈아타기 (Yang, Ryu & De Luca 2021): 보통주를 들고 있다가 우선주가 250일 기준 2σ 싸지면 우선주로, z>0 이면 다시 보통주."""
    pairs = [("005935", "005930"), ("005385", "005380")]
    codes = D.P["codes"]
    C = AW.ffill(D.P["C"])
    cb, cs = L.costs("stock")
    outs, bases = [], []
    for p, c in pairs:
        if p not in codes or c not in codes:
            continue
        jp, jc = codes.index(p), codes.index(c)
        sp = pd.Series(np.log(C[:, jp]) - np.log(C[:, jc]))
        z = ((sp - sp.rolling(250, min_periods=150).mean()) / sp.rolling(250, min_periods=150).std()).to_numpy()
        hold = jc
        r = np.zeros(D.T)
        for t in range(1, D.T):
            r[t] = C[t, hold] / C[t - 1, hold] - 1
            if hold == jc and np.isfinite(z[t]) and z[t] < -2:
                hold = jp
                r[t] -= cs + cb
            elif hold == jp and np.isfinite(z[t]) and z[t] > 0:
                hold = jc
                r[t] -= cs + cb
        outs.append(r)
        bases.append(np.r_[0, C[1:, jc] / C[:-1, jc] - 1])
    rn, rb = np.mean(outs, 0), np.mean(bases, 0)
    m = np.arange(D.T) > 260
    return compare("보통주 대신 싸진 우선주로 갈아타기", "화학 · 평형으로의 복귀 / 한국 시장 구조", "Yang, Ryu & De Luca 2021 (Sustainability) · Gatev et al. 2006",
                   "삼성전자·현대차 보통주를 들고 있다가 우선주/보통주 가격비가 250일 평균보다 2σ 낮아지면 그날 종가에 우선주로 바꾸고, 평균 위로 돌아오면 다시 보통주",
                   rn[m], rb[m], D.dates[m], split, "보통주 그대로 보유")


# ══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "hypotheses": [], "overfit": {}}
    E = Etf()
    D = L.Data()
    print(f"자료: ETF {E.dates[0].date()}~{E.dates[-1].date()} · 국내 {D.dates[0].date()}~{D.dates[-1].date()} ({time.time() - t0:.0f}s)")
    models = L.stock_models() + L.etf_models()
    recs = [L.run(D, m, record=True) for m in models]
    R = np.c_[[L.rets(r["E"]) for r in recs]].T
    print(f"국내 모델 {len(models)}개 재계산 ({time.time() - t0:.0f}s)")
    H = res["hypotheses"]
    # 1) 변동성 관리
    start = int(np.searchsorted(D.dates, pd.Timestamp("2019-01-02")))
    E_live, _, _ = L.apply_rule(D, models, recs, R, L.LIVE_RULE, start)
    r_live = L.rets(E_live)
    H.append(h_vol_managed("단기 몫에 변동성 관리", "통계학·경제학 · 변동성 관리 포트폴리오", "Moreira & Muir 2017 (JF) · Barroso & Santa-Clara 2015",
                           r_live[start:], D.dates[start:], sum(L.costs("stock")), "2022-12-31", "단기 갈아타기 그대로"))
    H.append(h_vol_managed("KODEX 200 에 변동성 관리", "통계학·경제학 · 변동성 관리 포트폴리오", "Moreira & Muir 2017 (JF)",
                           E.r_cc, E.dates, E.cost, "2020-12-31", "KODEX 200 보유"))
    try:
        import us_lab as UL
        DU = UL.USData(UL.load_panel())
        mom = {"id": "S:MOM:10:BAND:none", "kind": "stock", "sel": "MOM", "K": 10, "hold": "BAND", "gate": "none"}
        r_mom = L.rets(L.run_stock(DU, mom)["E"])
        s0 = int(np.searchsorted(DU.dates, pd.Timestamp("2016-01-04")))
        H.append(h_vol_managed("미국 추세 상위 10 에 변동성 관리", "통계학·경제학 · 모멘텀 붕괴 방지", "Barroso & Santa-Clara 2015 (JFE) · Daniel & Moskowitz 2016",
                               r_mom[s0:], DU.dates[s0:], sum(L.costs("us")), "2022-12-31", "미국 추세 상위 10 그대로", look=126))
        # 미국 모델 과최적화 판정
        ums = UL.models()
        urecs = [L.run(DU, m) for m in ums]
        UR = np.c_[[L.rets(r["E"]) for r in urecs]].T[s0:]
        usr = [np.nanmean(UR[:, i]) / np.nanstd(UR[:, i]) for i in range(UR.shape[1])]
        bi = ums.index(next(m for m in ums if m["id"] == mom["id"]))
        p_d, sr0 = dsr(UR[:, bi], usr)
        res["overfit"]["us"] = {"what": f"미국 모델 {len(ums)}개 중 추세 상위 10", "dsr": round(p_d, 3) if p_d is not None else None,
                                "sr0_annual": round(sr0, 3) if sr0 else None, "sharpe": round(sh(UR[:, bi]), 3), **pbo(UR)}
        print(f"  미국 과최적화: DSR {res['overfit']['us']['dsr']} · PBO {res['overfit']['us']['pbo']}")
        H.append(h_panic(DU, r_mom, s0, "2022-12-31"))
        H.append(h_fip(DU, "미국", "us", "2022-12-31", s0))
        H.append(h_lowvol(DU, "미국", s0, "2022-12-31"))
        H.append(h_close_exec(DU, "미국", mom, s0, "2022-12-31", "us"))
    except Exception as ex:
        print("  미국 단계 건너뜀:", type(ex).__name__, ex)
    # 2) 물리학·복잡계
    h, ar = h_absorption(D, E, "2022-12-31")
    H.append(h)
    H.append(h_slowing_down(E, "2020-12-31"))
    # 3) 통계학: 분산비 갈아타기 · 경제학: 조건부 반전
    rev = family_returns(D, models, recs, ("REV1", "REVC", "REVC_LV", "REV3_LV", "VSD", "GAP", "REVID", "MA20_LV"))
    mom_k = family_returns(D, models, recs, ("MOM",))
    H.append(h_vr_switch(D, rev, mom_k, "2022-12-31"))
    H.append(h_nagel(D, rev, "2022-12-31"))
    # 4) 달력·천문
    H.extend(calendar_tests(E, "2020-12-31"))
    # 5) 한국 수급 · 미국→한국 · 우선주 · 군집
    H.append(h_foreign(D, E, "2022-12-31"))
    try:
        H.append(h_us_lead(E, "2020-12-31"))
    except Exception as ex:
        print("  미국→한국 건너뜀:", type(ex).__name__, ex)
    H.append(h_pairs(D, "2022-12-31"))
    H.append(h_herding(D, E, "2022-12-31"))
    # 6) 실무서·책: 카나리아 · HRP · 예측 결합 · 꾸준한 추세(국내)
    H.append(h_canary(E, "2020-12-31"))
    H.append(h_hrp("2020-12-31"))
    H.append(h_blend(D, models, recs, R, E_live, start, "2022-12-31"))
    H.append(h_fip(D, "국내", "stock", "2022-12-31", start))
    H.append(h_lowvol(D, "국내", start, "2022-12-31"))
    H.append(h_flow_cs(D, start, "2022-12-31", "FOR", "외국인"))
    H.append(h_flow_cs(D, start, "2022-12-31", "IND", "개인"))
    H.append(h_card_tsmom("2020-12-31"))
    H.append(h_close_exec(D, "국내", {"id": "S:MOM:10:H10:none", "kind": "stock", "sel": "MOM", "K": 10, "hold": "H10", "gate": "none"},
                          start, "2022-12-31", "stock"))
    # 7) 물리·생물·행동 추가 (조사 결과에서)
    H.append(h_turbulence(E, "2020-12-31"))
    H.append(h_extra_cards(E, "2020-12-31"))
    try:
        H.append(h_us_underreact(E, "2020-12-31"))
    except Exception as ex:
        print("  미국 과소반응 건너뜀:", type(ex).__name__, ex)
    H.append(h_capitulation(D, E, "2022-12-31"))
    H.append(h_inst_market(D, E, "2022-12-31"))
    H.append(h_comomentum(D, start, "2022-12-31"))
    H.append(h_max_filter(D, start, "2022-12-31"))
    H.append(h_pref_switch(D, "2022-12-31"))
    # 등록 뒤 성적 (Adaptive Markets · McLean & Pontiff: 발표 뒤 줄어드는지 앞으로 지켜본다)
    reg = json.load(open(REG, encoding="utf-8")) if os.path.exists(REG) else {}
    today = str(D.dates[-1].date())
    for h in H:
        born = reg.setdefault(h["name"], today)
        d = h.pop("_d")
        rn, rb = h.pop("_rn"), h.pop("_rb")
        k = d > pd.Timestamp(born)
        h["registered"] = born
        h["forward"] = {"days": int(k.sum()), "new": round(float(np.prod(1 + rn[k]) - 1), 4) if k.any() else 0.0,
                        "base": round(float(np.prod(1 + rb[k]) - 1), 4) if k.any() else 0.0}
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    json.dump(reg, open(REG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 다중검정 보정 · 판정
    adj = holm([h.get("p") for h in H])
    for h, a in zip(H, adj):
        h["p_holm"] = a
        better = h["new"] and h["baseline"] and h["new"]["sharpe"] > h["baseline"]["sharpe"]
        h["verdict"] = ("통과" if (better and h["both_periods"] and a is not None and a < 0.05) else
                        "유망 (확신 부족)" if (better and h["both_periods"]) else "기각")
        print(f"  [{h['verdict']:8s}] {h['name']:28s} 샤프 {h['new']['sharpe']:.2f} vs {h['baseline']['sharpe']:.2f} · "
              f"두 기간 {h['both_periods']} · P {h['p']} → 보정 {a}")
    # 과최적화: 국내 단기 모델 전체
    pool = L.pool_index(models, "SLOW")
    Rs = R[start:][:, pool]
    srs = [np.nanmean(Rs[:, i]) / np.nanstd(Rs[:, i]) for i in range(Rs.shape[1])]
    p_d, sr0 = dsr(r_live[start:], srs)
    res["overfit"]["kr_short"] = {"what": f"국내 단기 후보 {len(pool)}개 + 고르는 방식 288가지 중 채택 규칙",
                                  "dsr": round(p_d, 3) if p_d is not None else None, "sr0_annual": round(sr0, 3) if sr0 else None,
                                  "sharpe": round(sh(r_live[start:]), 3), **pbo(Rs)}
    print(f"  국내 단기 과최적화: DSR {res['overfit']['kr_short']['dsr']} · PBO {res['overfit']['kr_short']['pbo']}")
    res["absorption_now"] = round(float(ar[-1]), 4) if np.isfinite(ar[-1]) else None
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    from short_trader import clean
    res = clean(res)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(res, open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"끝 ({time.time() - t0:.0f}s) → {OUT}")
    return res


if __name__ == "__main__":
    main()
