"""
단기매매 연구실 — '보유 1~20일 단기매매가 비용을 내고도 이기는가' 와 '매일 가장 잘 버는 모델로 갈아타기가 실제로 통하는가'를 잰다.

체결 가정 (실제 주문 방식과 같게)
  t일 장 마감 뒤 판단 → t+1일 시가(09:00 동시호가)에 사고판다. 밤사이 ETF 모델만 t+1일 종가(15:30 동시호가)에 사서 t+2일 시가에 판다.
  판단은 언제나 t일 저녁까지의 자료로만 한다 (장중 정보 사용 안 함).
비용 (2026년 기준, 한쪽씩)
  수수료 0.015% + 유관기관 제비용 ≈0.0036% → 0.0186%
  증권거래세+농특세 0.20% (코스피, 매도 때만) — 국내 주식형 ETF 는 면제
  미끄러짐: 주식 0.05% · ETF 는 호가 반 칸 + 0.005% (allweather.costs — 2,000원 미만 1원·이상 5원 호가, 2026-10-28부터)
  → 주식 왕복 ≈0.34%, KODEX 200 왕복 ≈0.06%, KODEX 인버스(1천원대) 왕복 ≈0.13%.  stress=3 이면 미끄러짐 3배.
유니버스: 날짜마다 시가총액 상위 100 (short_panel). 지수 ETF 는 실제 가격이 없으면 시총가중 지수로 근사.

명령
  python src/short_lab.py            # 전체 연구 → data/short/lab.json, docs/short_lab.json
  python src/short_lab.py --quick    # 모델 수를 줄인 빠른 점검
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allweather as AW  # noqa: E402
import short_panel as SP  # noqa: E402

OUT_DIR = "data/short"
COMM = 0.000186
TAX = 0.0020
SLIP = {"stock": 0.0005, "etf": 0.0001, "us": 0.0003}
COMM_US = 0.001                 # 미국 주식 수수료 한쪽 0.1% (국내 증권사 해외주식 보통 0.07~0.25%) · 매도 SEC 수수료 ≈0.003%
SEC_US = 0.00003
DEV_END = "2022-12-31"          # 개발 구간 끝 (이후 = 확인 구간)


def costs(kind, stress=1.0):
    s = SLIP[kind] * stress
    if kind == "us":
        return COMM_US + s, COMM_US + s + SEC_US
    return COMM + s, COMM + s + (TAX if kind == "stock" else 0.0)


# ══════════════════════════════════════════════════════════════
# 모델 정의
# ══════════════════════════════════════════════════════════════
SELECTORS = {
    "REV1": [("rev1", 1)],
    "REVID": [("rev_id", 1)],
    "REVC": [("rev1", 1), ("rev_id", 1), ("clv_lo", 1)],
    "REVC_LV": [("rev1", 1), ("rev_id", 1), ("clv_lo", 1), ("lowvol", 1.5)],
    "REVC_FLOW": [("rev1", 1), ("rev_id", 1), ("clv_lo", 1), ("fi1_rev", 1)],
    "REV3_LV": [("rev3", 1), ("lowvol", 1)],
    "VSD": [("vsurge_dn", 1), ("rev1", 1)],
    "MA20_LV": [("ma20rev", 1), ("lowvol", 1)],
    "GAP": [("gap_rev", 1), ("rev1", 1)],
    "MOM": [("mom120", 1), ("hi52", 1)],
    "FOR20": [("for20", 1), ("mom120", 1)],
}
SEL_NAME = {"REV1": "1일 반전", "REVID": "장중 반전", "REVC": "반전 종합", "REVC_LV": "반전+저변동",
            "REVC_FLOW": "반전+수급역행", "REV3_LV": "3일반전+저변동", "VSD": "투매 반전", "MA20_LV": "20일선 이격+저변동",
            "GAP": "갭하락 반전", "MOM": "6개월 추세(대조군)", "FOR20": "외국인 20일+추세"}
# (최소 보유, 최대 보유, 이탈 순위 배수)
HOLDS = {"H1": (1, 1, 1), "H3": (3, 3, 2), "H5": (5, 5, 2), "H10": (10, 10, 3), "BAND": (1, 20, 3)}
HOLD_NAME = {"H1": "매일 교체", "H3": "3일 보유", "H5": "5일 보유", "H10": "10일 보유", "BAND": "순위 유지 시 보유(최대 20일)"}
GATES = {"none": "항상", "trend": "지수 120일선 위", "crash": "급락장 제외"}

ETF_RULES = {
    "hold": "지수 ETF 계속 보유 (기준)",
    "overnight": "밤사이만 보유 (종가 매수 → 다음날 시가 매도)",
    "overnight_trend": "밤사이 보유, 지수 120일선 위일 때만",
    "overnight_rev": "밤사이 보유, 전날 지수가 내렸을 때만",
    "overnight_for": "밤사이 보유, 전날 외국인 순매수일 때만",
    "day_rev": "장중만 보유 (시가→종가), 전날 지수가 내렸을 때만",
    "oo_rev": "전날 지수가 내렸으면 지수 ETF 보유, 오르면 다음 날 시가에 팔고 현금",
    "oo_revid": "전날 장중에 지수가 내렸으면 지수 ETF 보유, 아니면 현금 (시가에 사고팖)",
    "ls_rev": "전날 내렸으면 지수 ETF, 올랐으면 인버스 보유 (시가에 바꿈)",
    "trend_hold": "지수 120일선 위면 보유, 아래면 현금",
    "trend_overnight": "120일선 위면 계속 보유, 아래면 밤사이만",
}


def stock_models(quick=False):
    out = []
    sels = ["REV1", "REVC", "REVC_LV", "MOM"] if quick else list(SELECTORS)
    for s in sels:
        for K in (5, 10):
            for h in HOLDS:
                for g in (("none", "trend") if quick else GATES):
                    out.append({"id": f"S:{s}:{K}:{h}:{g}", "kind": "stock", "sel": s, "K": K, "hold": h, "gate": g})
    return out


def etf_models():
    return [{"id": f"E:{r}", "kind": "etf", "rule": r} for r in ETF_RULES]


def describe(m):
    if m["kind"] == "etf":
        return ETF_RULES[m["rule"]]
    if m.get("combo"):                                   # short_evolve 가 만든 모델
        sig = "+".join(SP.SIGNAL_INFO.get(k, k).split(" (")[0] for k, _ in m["combo"])
        return f"진화 {m['id']} · {sig} · {m['K']}종목 · {HOLD_NAME[m['hold']]} · {GATES[m['gate']]}"
    return f"{SEL_NAME[m['sel']]} · {m['K']}종목 · {HOLD_NAME[m['hold']]} · {GATES[m['gate']]}"


# ══════════════════════════════════════════════════════════════
# 데이터
# ══════════════════════════════════════════════════════════════
class Data:
    def __init__(self, P=None):
        self.P = P or SP.load()
        P = self.P
        self.dates = P["dates"]
        self.T, self.N = P["C"].shape
        self.O = P["O"]
        self.Cf = np.nan_to_num(P["C"].copy())
        self.Cf = pd.DataFrame(P["C"]).ffill().fillna(0).to_numpy(copy=True)
        self.S = SP.signals(P)
        self.Op, self.Cp, self.legs = SP.etf_prices(P)
        self.M = SP.market_signals(P, self.legs)
        mk = SP.market(P)
        self.gate = {"none": np.ones(self.T, bool),
                     "trend": np.nan_to_num(self.M["m_trend120"], nan=1) > 0,
                     "crash": ~mk["crash_off"]}
        self._rk = {}
        self._pr = {}

    def pr(self, name):
        """신호 하나의 유니버스 안 백분위 (값이 없으면 0.5) — 조합 순위의 재료. 한 번만 계산."""
        if name not in self._pr:
            x = np.where(self.P["U"], self.S[name], np.nan)
            p = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy(copy=True)
            self._pr[name] = np.nan_to_num(p, nan=0.5)
        return self._pr[name]

    def ranks(self, sel, combo=None):
        """순위(1=최고, 유니버스 밖 inf). sel = SELECTORS 이름, 또는 combo=[(신호, 가중치)…] (진화 모델)."""
        combo = combo or SELECTORS[sel]
        key = tuple((k, float(w)) for k, w in combo)
        if key not in self._rk:
            tot = sum(w * self.pr(k) for k, w in combo)
            tot = np.where(self.P["U"], tot, np.nan)
            rk = pd.DataFrame(-tot).rank(axis=1, method="first").to_numpy(copy=True)
            rk[np.isnan(rk)] = np.inf
            self._rk[key] = rk
        return self._rk[key]


# ══════════════════════════════════════════════════════════════
# 시뮬레이터 — 연구·매일 모의투자가 같은 함수를 쓴다
# ══════════════════════════════════════════════════════════════
def stock_decide(units, since, rk, ok, t, K, X, mn, mx, trad):
    """종목 모델의 하루 판단 (t-1일 저녁 순위 rk 로 t일 시가에 할 일). 반환: 팔 종목(bool), 살 종목(순서대로)."""
    held = units > 0
    sell = np.zeros(len(units), bool)
    if held.any():
        age = t - since
        sell = held & trad & (((rk > X) & (age >= mn)) | ((age >= mx) & (rk > K)))
    buys = np.array([], int)
    if ok:
        free = K - int((held & ~sell).sum())
        if free > 0:
            cand = np.where((rk <= K) & ~held & trad)[0]
            buys = cand[np.argsort(rk[cand])][:free]
    return sell, buys


def stock_params(m):
    mn, mx, xm = HOLDS[m["hold"]]
    return m["K"], m["K"] * xm, mn, mx


def run_stock(D, m, stress=1.0, record=False, t0=1, log=None):
    """종목 모델: 순위 상위 K를 시가에 사고, 보유 규칙대로 시가에 판다. 반환: 종가 기준 자산곡선 등.
    log 에 목록을 넘기면 매매마다 {t, j, side, px, rank, why} 를 남긴다 (해설 화면용)."""
    R = D.ranks(m["sel"], m.get("combo"))
    K, X, mn, mx = stock_params(m)
    ok = D.gate[m["gate"]]
    cb, cs = costs(getattr(D, "cost_kind", "stock"), stress)
    T, N = D.T, D.N
    O, Cf = D.O, D.Cf
    units = np.zeros(N)
    since = np.full(N, -1)
    cash = 1.0
    E = np.ones(T)
    traded = 0.0
    nb = 0
    hold = [None] * T if record else None
    for t in range(t0, T):
        po = O[t]
        trad = ~np.isnan(po)
        sell, buys = stock_decide(units, since, R[t - 1], ok[t - 1], t, K, X, mn, mx, trad)
        if log is not None:
            for j in np.where(sell)[0]:
                rk_j = R[t - 1, j]
                log.append({"t": t, "j": int(j), "side": "sell", "px": float(po[j]), "entry_t": int(since[j]),
                            "entry_px": None, "rank": None if not np.isfinite(rk_j) else int(rk_j),
                            "why": "rank" if rk_j > X and t - since[j] >= mn else "max_hold"})
        if sell.any():
            v = (units[sell] * po[sell]).sum()
            cash += v * (1 - cs)
            traded += v
            units[sell] = 0
            since[sell] = -1
        if len(buys):
            eq = cash + (units * np.where(trad, po, Cf[t - 1])).sum()
            amt = min(eq / K, cash / (1 + cb) / len(buys))
            if amt > 0:
                units[buys] = amt / po[buys]
                since[buys] = t
                if log is not None:
                    for j in buys:
                        log.append({"t": t, "j": int(j), "side": "buy", "px": float(po[j]), "rank": int(R[t - 1, j]), "why": "rank"})
                cash -= amt * len(buys) * (1 + cb)
                traded += amt * len(buys)
                nb += len(buys)
        E[t] = cash + (units * Cf[t]).sum()
        if record:
            hold[t] = np.where(units > 0)[0]
    return {"E": E, "turnover": traded, "buys": nb, "hold": hold, "units": units, "cash": cash, "since": since}


def stock_next(D, m, res):
    """마지막 날 저녁 기준, 이 모델이 내일 시가 거래 뒤 들고 있을 종목 (모두 거래 가능하다고 가정)."""
    K, X, mn, mx = stock_params(m)
    T = D.T
    sell, buys = stock_decide(res["units"], res["since"], D.ranks(m["sel"], m.get("combo"))[T - 1], D.gate[m["gate"]][T - 1],
                              T, K, X, mn, mx, np.ones(D.N, bool))
    keep = np.where((res["units"] > 0) & ~sell)[0]
    return sorted(set(keep.tolist()) | set(buys.tolist()))


def etf_targets(D, rule, pad=0):
    """t행 = t일 시가 직후 목표 비중 / t일 종가 직후 목표 비중 (NaN = 그때 거래 안 함). 판단은 t-1일 저녁.
    pad=1 이면 한 행 더 = 내일(마지막 날 저녁 판단)."""
    T = D.T + pad
    wo = np.full((T, 2), np.nan)
    wc = np.full((T, 2), np.nan)
    M = D.M
    prev = (lambda x: np.r_[np.nan, x[:-1]]) if not pad else (lambda x: np.r_[np.nan, x])   # noqa: E731 — t-1일 값
    rev = prev(M["m_rev1"]) > 0
    revid = prev(M["m_rev_id"]) > 0
    fo = prev(M["m_for1"]) > 0
    tr = prev(np.nan_to_num(M["m_trend120"], nan=1)) > 0
    one, zero = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    if rule == "hold":
        wo[:] = one
    elif rule == "overnight":
        wo[:] = zero
        wc[:] = one
    elif rule in ("overnight_trend", "overnight_rev", "overnight_for"):
        cond = {"overnight_trend": tr, "overnight_rev": rev, "overnight_for": fo}[rule]
        wo[:] = zero
        wc[cond] = one
        wc[~cond] = zero
    elif rule == "day_rev":
        wo[rev] = one
        wo[~rev] = zero
        wc[:] = zero
    elif rule in ("oo_rev", "oo_revid"):
        cond = rev if rule == "oo_rev" else revid
        wo[cond] = one
        wo[~cond] = zero
    elif rule == "ls_rev":
        wo[rev] = one
        wo[~rev] = np.array([0.0, 1.0])
    elif rule == "trend_hold":
        wo[tr] = one
        wo[~tr] = zero
    elif rule == "trend_overnight":
        wo[tr] = one
        wo[~tr] = zero
        wc[tr] = one
        wc[~tr] = one
    return wo, wc


def pad_assets(w2):
    """2자산(지수·인버스) 목표 → 전체 자산 목표 (나머지 0, 거래 안 하는 날은 NaN 그대로)."""
    W = np.full((w2.shape[0], AW.A), np.nan)
    W[:, :2] = w2
    W[~np.isnan(w2[:, 0]), 2:] = 0.0
    return W


def run_etf(D, m, stress=1.0, record=False, t0=1, band=0.05, log=None):
    """ETF 모델 — 전천후 모델과 같은 시뮬레이터 (자산별 호가 비용·이익 과세)."""
    wo, wc = etf_targets(D, m["rule"])
    return AW.simulate(D.Op, D.Cp, pad_assets(wo), pad_assets(wc), t0=t0, band=band, stress=stress, record=record, log=log)


def run(D, m, **kw):
    return run_stock(D, m, **kw) if m["kind"] == "stock" else run_etf(D, m, **kw)


# ══════════════════════════════════════════════════════════════
# 성적
# ══════════════════════════════════════════════════════════════
def stats(E, dates, a=None, b=None):
    s = pd.Series(E, index=dates)
    if a:
        s = s[s.index >= a]
    if b:
        s = s[s.index <= b]
    r = s.pct_change().dropna()
    if len(r) < 20:
        return None
    yrs = len(r) / 252
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    mdd = (s / s.cummax() - 1).min()
    return {"cagr": round(float(cagr), 4), "vol": round(float(vol), 4),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0, 3),
            "mdd": round(float(mdd), 4)}


def yearly(E, dates):
    s = pd.Series(E, index=dates)
    y = s.groupby(s.index.year).last()
    first = s.iloc[0]
    out = {}
    prev = first
    for yr, v in y.items():
        out[str(yr)] = round(float(v / prev - 1), 4)
        prev = v
    return out


def boot_p(ra, rb, n=2000, L=20, seed=0):
    """P[Sharpe(a) - Sharpe(b) ≤ 0] — 블록 부트스트랩 (블록 길이 L일)."""
    rng = np.random.default_rng(seed)
    d = np.c_[ra, rb]
    d = d[~np.isnan(d).any(1)]
    T = len(d)
    if T < 100:
        return None
    nb = int(np.ceil(T / L))
    cnt = 0
    for _ in range(n):
        st = rng.integers(0, T - L, nb)
        ix = (st[:, None] + np.arange(L)).ravel()[:T]
        x = d[ix]
        sa = x[:, 0].mean() / (x[:, 0].std() + 1e-12)
        sb = x[:, 1].mean() / (x[:, 1].std() + 1e-12)
        cnt += sa - sb <= 0
    return round(cnt / n, 4)


def rets(E):
    return np.r_[0.0, E[1:] / E[:-1] - 1]


# ══════════════════════════════════════════════════════════════
# 1부 — 모델별 성적 (개발 2018~2022 / 확인 2023~)
# ══════════════════════════════════════════════════════════════
def evaluate(D, models, stress=1.0, verbose=True):
    rows, curves = [], {}
    t0 = time.time()
    for i, m in enumerate(models):
        res = run(D, m, stress=stress)
        E = res["E"]
        curves[m["id"]] = E
        yrs = D.T / 252
        row = {"id": m["id"], "kind": m["kind"], "desc": describe(m),
               "full": stats(E, D.dates), "dev": stats(E, D.dates, b=DEV_END),
               "test": stats(E, D.dates, a="2023-01-01"),
               "turnover_x": round(res["turnover"] / max(E.mean(), 1e-9) / yrs, 1),
               "buys_per_year": round(res["buys"] / yrs, 1), "years": yearly(E, D.dates)}
        rows.append(row)
        if verbose and (i + 1) % 50 == 0:
            print(f"  … {i + 1}/{len(models)} ({time.time() - t0:.0f}s)", flush=True)
    return rows, curves


# ══════════════════════════════════════════════════════════════
# 2부 — '매일 최고 모델로 갈아타기' 검증 (미래 정보 없이 매일 고르고, 교체 비용까지 실제로 낸다)
# ══════════════════════════════════════════════════════════════
def follow(D, models, choice, records, stress=1.0, t0=1):
    """choice[t] = t일 저녁에 고른 모델 번호(또는 번호 목록=균등 혼합) → t+1일에 그 모델의 보유를 따라간다.
    종목: 모델이 t+1일 시가 거래 뒤 보유한 종목을 같은 비중으로 / ETF: 모델의 시가·종가 목표 비중."""
    T, N = D.T, D.N
    cbS, csS = costs(getattr(D, "cost_kind", "stock"), stress)
    O, Cf = D.O, D.Cf
    Op, Cp = D.Op[:, :2], AW.ffill(D.Cp[:, :2])
    us = np.zeros(N)
    ue = np.zeros(2)
    cash = 1.0
    E = np.ones(T)
    switches = 0
    targets_o = {}
    for i, m in enumerate(models):
        if m["kind"] == "etf":
            targets_o[i] = etf_targets(D, m["rule"])
    prev = None
    for t in range(t0, T):
        ch = choice[t - 1]
        if ch is None:
            E[t] = cash + (us * Cf[t]).sum() + (ue * Cp[t]).sum()
            continue
        chs = ch if isinstance(ch, (list, tuple)) else [ch]
        if prev is not None and set(chs) != set(prev):
            switches += 1
        prev = chs
        w = 1.0 / len(chs)
        po = O[t]
        trad = ~np.isnan(po)
        cbE, csE = AW.costs(Op[t], stress)
        eq = cash + (us * np.where(trad, po, Cf[t - 1])).sum() + (ue * Op[t]).sum()
        # 시가 목표
        ts = np.zeros(N)
        te = np.zeros(2)
        etf_open_set = False
        tc = np.zeros(2)
        etf_close_set = False
        for c in chs:
            m = models[c]
            if m["kind"] == "stock":
                h = records[c]["hold"][t]
                if h is not None and len(h):
                    ts[h] += w / m["K"]
            else:
                wo, wc = targets_o[c]
                if not np.isnan(wo[t, 0]):
                    te += w * wo[t]
                    etf_open_set = True
                tc += w * (wc[t] if not np.isnan(wc[t, 0]) else (wo[t] if not np.isnan(wo[t, 0]) else 0))
                etf_close_set |= not np.isnan(wc[t, 0])
        if not any(models[c]["kind"] == "etf" for c in chs) and ue.any():
            etf_open_set = True                      # 종목 모델로 갈아탔으면 남은 ETF 는 시가에 판다 (모의투자와 같게)
        # 종목: 목표에 없는 것 매도, 목표에 새로 들어온 것 매수 (겹치는 종목은 그대로 — 실제 매매와 같게)
        sell = (us > 0) & trad & (ts == 0)
        if sell.any():
            v = (us[sell] * po[sell]).sum()
            cash += v * (1 - csS)
            us[sell] = 0
        # ETF 시가 거래
        if etf_open_set:
            for j in range(2):
                d = te[j] - ue[j] * Op[t, j] / max(eq, 1e-12)
                if d < -0.05 or (te[j] == 0 and ue[j] > 0):
                    v = min(-d * eq, ue[j] * Op[t, j])
                    ue[j] -= v / Op[t, j]
                    cash += v * (1 - csE[j])
        buy = (ts > 0) & (us == 0) & trad
        if buy.any():
            need = (ts[buy] * eq).sum()
            scale = min(1.0, cash / (1 + cbS) / max(need, 1e-12))
            amt = ts[buy] * eq * scale
            us[buy] = amt / po[buy]
            cash -= amt.sum() * (1 + cbS)
        if etf_open_set:
            for j in range(2):
                d = te[j] - ue[j] * Op[t, j] / max(eq, 1e-12)
                if d > 0.05 or (te[j] > 0 and ue[j] == 0 and d > 1e-9):
                    v = min(d * eq, cash / (1 + cbE[j]))
                    if v > 0:
                        ue[j] += v / Op[t, j]
                        cash -= v * (1 + cbE[j])
        # 종가 거래 (ETF 밤사이 모델)
        if etf_close_set:
            cbE, csE = AW.costs(Cp[t], stress)
            eqc = cash + (us * Cf[t]).sum() + (ue * Cp[t]).sum()
            for j in range(2):
                d = tc[j] - ue[j] * Cp[t, j] / eqc
                if d < -0.05 or (tc[j] == 0 and ue[j] > 0):
                    v = min(-d * eqc, ue[j] * Cp[t, j])
                    ue[j] -= v / Cp[t, j]
                    cash += v * (1 - csE[j])
            for j in range(2):
                d = tc[j] - ue[j] * Cp[t, j] / eqc
                if d > 0.05 or (tc[j] > 0 and ue[j] == 0 and d > 1e-9):
                    v = min(d * eqc, cash / (1 + cbE[j]))
                    if v > 0:
                        ue[j] += v / Cp[t, j]
                        cash -= v * (1 + cbE[j])
        E[t] = cash + (us * Cf[t]).sum() + (ue * Cp[t]).sum()
    return E, switches


def trailing_score(R, t, L, how):
    """R: (T×S) 일별 수익률. t일까지 최근 L일 성적.
    NaN = 아직 성적으로 치지 않는 날 (진화 모델은 태어나기 전 날들이 NaN) — 유효한 날이 L/2(최소 20)보다 적으면 NaN."""
    x = R[max(1, t - L + 1): t + 1]
    need = max(20, L // 2)
    if len(x) < need:
        return None
    n = (~np.isnan(x)).sum(0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if how == "ret":
            out = np.prod(1 + np.nan_to_num(x), 0) - 1
        else:
            mu, sd = np.nanmean(x, 0), np.nanstd(x, 0)
            out = mu / np.where(sd > 0, sd, np.inf) * np.sqrt(252)
    return np.where(n >= need, out, np.nan)


# 선택 규칙 = (몇 거래일마다 다시 고르나, 최근 며칠 성적으로, 무엇으로(ret=수익·sharpe=위험조정), 몇 개를 섞나, 끈기)
#   끈기: 새 1등이 지금 모델보다 이만큼(샤프 단위) 낫지 않으면 바꾸지 않는다
LIVE_RULE = {"pool": "SLOW", "every": 5, "L": 250, "how": "sharpe", "k": 1, "sticky": 0.0}
SHADOW_RULES = {
    "live": ("채택 · 매주 · 최근 250일 샤프 1등 (느린 후보군)", LIVE_RULE),
    "daily_ret20": ("매일 · 최근 20일 수익 1등 (모든 후보)", {"pool": "ALL", "every": 1, "L": 20, "how": "ret", "k": 1, "sticky": 0.0}),
    "daily_ret60": ("매일 · 최근 60일 수익 1등 (모든 후보)", {"pool": "ALL", "every": 1, "L": 60, "how": "ret", "k": 1, "sticky": 0.0}),
    "daily_sh60": ("매일 · 최근 60일 샤프 1등 (모든 후보)", {"pool": "ALL", "every": 1, "L": 60, "how": "sharpe", "k": 1, "sticky": 0.0}),
    "daily_sh250": ("매일 · 최근 250일 샤프 1등 (느린 후보군)", {"pool": "SLOW", "every": 1, "L": 250, "how": "sharpe", "k": 1, "sticky": 0.0}),
    "monthly_sh250": ("매달 · 최근 250일 샤프 1등 (느린 후보군)", {"pool": "SLOW", "every": 21, "L": 250, "how": "sharpe", "k": 1, "sticky": 0.0}),
}
POOL_NAME = {"ALL": "모든 후보", "SLOW": "느린 후보군 (5·10일 보유 종목 모델 + ETF 모델)"}


def pool_index(models, pool):
    if pool == "ALL":
        return list(range(len(models)))
    return [i for i, m in enumerate(models) if m["kind"] == "etf" or m["hold"] in ("H5", "H10")]


def rule_name(r):
    every = {1: "매일", 5: "매주", 21: "매달"}.get(r["every"], f"{r['every']}일마다")
    how = "수익" if r["how"] == "ret" else "샤프"
    k = "1등" if r["k"] == 1 else f"상위 {r['k']}개 혼합"
    st = f" · 끈기 {r['sticky']}" if r.get("sticky") else ""
    return f"{every} · 최근 {r['L']}일 {how} {k}{st} ({'모든 후보' if r['pool'] == 'ALL' else '느린 후보군'})"


def choose(R, start, every, L, how, k, sticky=0.0, **_):
    """t일 저녁에 t일까지의 모델 성적만 보고 고른다. 반환: 날짜별 모델 번호(또는 번호 목록)."""
    T = R.shape[0]
    ch = [None] * T
    cur = None
    for t in range(start, T):
        if cur is None or (t - start) % every == 0:
            sc = trailing_score(R, t, L, how)
            if sc is None:
                continue
            sc = np.nan_to_num(sc, nan=-np.inf)
            order = np.argsort(-sc)
            if k == 1:
                b = int(order[0])
                if cur is not None and sticky > 0 and sc[b] < sc[cur] + (sticky if how == "sharpe" else sticky * 0.1):
                    b = cur
                cur = b
            else:
                new = [int(i) for i in order[:k]]
                if cur is not None and sticky > 0:
                    top2 = set(order[:k * 2].tolist())
                    keep = [c for c in cur if c in top2]
                    new = (keep + [i for i in new if i not in keep])[:k]
                cur = new
        ch[t] = cur
    return ch


def apply_rule(D, models, recs, R, rule, start, stress=1.0):
    pool = pool_index(models, rule["pool"])
    ch = choose(R[:, pool], start, **rule)
    ch = [None if c is None else (pool[c] if isinstance(c, int) else [pool[i] for i in c]) for c in ch]
    E, sw = follow(D, models, ch, recs, stress=stress, t0=start)
    return E, sw, ch


def rule_grid(D, models, recs, R, start, stress=1.0):
    rows = []
    for pool in ("ALL", "SLOW"):
        for every in (1, 5, 21):
            for L in (20, 40, 60, 120, 250, 500):
                for how in ("ret", "sharpe"):
                    for k in (1, 3):
                        for sticky in (0.0, 0.3):
                            r = {"pool": pool, "every": every, "L": L, "how": how, "k": k, "sticky": sticky}
                            E, sw, _ = apply_rule(D, models, recs, R, r, start, stress)
                            rows.append({**r, "switches": sw, "all": stats(E[start:], D.dates[start:]),
                                         "dev": stats(E, D.dates, a=str(D.dates[start].date()), b=DEV_END),
                                         "test": stats(E, D.dates, a="2023-01-01")})
    return rows


def ic_table(D, horizons=(1, 5, 10)):
    """신호별 순위 IC (유니버스 안 횡단면 순위상관, t일 신호 ↔ t+1 시가→t+1+h 시가 수익률). 겹치지 않는 날만으로 t값."""
    O, U = D.O, D.P["U"]
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = []
    for name in SP.SIGNAL_INFO:
        x = np.where(U, D.S[name], np.nan)
        rx = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy(copy=True)
        row = {"signal": name, "desc": SP.SIGNAL_INFO[name]}
        for h in horizons:
            fr = np.full_like(O, np.nan)
            fr[:-(1 + h)] = O[1 + h:] / O[1:-h] - 1
            fr = np.where(U, fr, np.nan)
            rf = pd.DataFrame(fr).rank(axis=1, pct=True).to_numpy(copy=True)
            ok = ~np.isnan(rx) & ~np.isnan(rf)
            a, b = np.where(ok, rx, np.nan), np.where(ok, rf, np.nan)
            with np.errstate(all="ignore"):
                am, bm = a - np.nanmean(a, 1, keepdims=True), b - np.nanmean(b, 1, keepdims=True)
                ic = np.nansum(am * bm, 1) / np.sqrt(np.nansum(am ** 2, 1) * np.nansum(bm ** 2, 1))
                top = np.nanmean(np.where(a >= 0.9, fr, np.nan), 1) - np.nanmean(np.where(ok, fr, np.nan), 1)
            ic[ok.sum(1) < 30] = np.nan
            sub = np.arange(0, D.T, h)
            v = ic[sub][~np.isnan(ic[sub])]
            sp = top[sub][~np.isnan(top[sub])]
            yrs = pd.Series(ic, index=D.dates).groupby(D.dates.year).mean()
            row[f"h{h}"] = {"ic": round(float(v.mean()), 4), "t": round(float(v.mean() / v.std() * np.sqrt(len(v))), 2),
                            "top10_bp": round(float(sp.mean() * 1e4), 1), "pos_years": int((yrs > 0).sum()), "years": int(len(yrs))}
        out.append(row)
    return out


def bench_curves(D):
    out = {}
    out["지수 ETF 보유"] = run_etf(D, {"rule": "hold"}, stress=0)["E"]
    ew = np.r_[1.0, np.cumprod(1 + np.nan_to_num(SP.market(D.P)["ret_ew"][1:]))]
    out["100종목 등가중"] = ew[:D.T]
    return out


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    D = Data()
    print(f"데이터: {D.dates[0].date()} ~ {D.dates[-1].date()} · {D.N}종목 · 지수ETF={D.legs['source']} ({time.time() - t0:.0f}s)")
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "period": [str(D.dates[0].date()), str(D.dates[-1].date())],
           "etf_source": D.legs["source"], "dev_end": DEV_END,
           "costs": {"stock_roundtrip": round(sum(costs("stock")), 5), "etf_roundtrip": round(sum(costs("etf")), 5),
                     "commission": COMM, "tax": TAX, "slip": SLIP}}
    print("0부: 신호 IC")
    res["ic"] = ic_table(D)
    models = stock_models(quick) + etf_models()
    print(f"1부: 모델 {len(models)}개 성적 (기본 비용 · 미끄러짐 3배)")
    rows, curves = evaluate(D, models)
    rows_s, curves_s = evaluate(D, models, stress=3.0, verbose=False)
    sm = {r["id"]: r for r in rows_s}
    for r in rows:
        r["stress"] = sm[r["id"]]["full"]
    res["models"] = rows
    df = pd.DataFrame([{"id": r["id"], "kind": r["kind"], "hold": r["id"].split(":")[3] if r["kind"] == "stock" else "ETF",
                        "full": r["full"]["sharpe"], "dev": r["dev"]["sharpe"], "test": r["test"]["sharpe"],
                        "cagr": r["full"]["cagr"], "turn": r["turnover_x"]} for r in rows])
    res["by_hold"] = {h: {"n": int(len(g)), "sharpe_med": round(float(g.full.median()), 3),
                          "dev_med": round(float(g.dev.median()), 3), "test_med": round(float(g.test.median()), 3),
                          "positive": int((g.cagr > 0).sum()), "turnover_med": round(float(g.turn.median()), 1)}
                      for h, g in df.groupby("hold")}
    bench = bench_curves(D)
    start = int(np.searchsorted(D.dates, pd.Timestamp("2019-01-02")))
    res["meta_start"] = str(D.dates[start].date())
    res["bench"] = {k: {"full": stats(v, D.dates), "dev": stats(v, D.dates, b=DEV_END), "test": stats(v, D.dates, a="2023-01-01"),
                        "meta": stats(v[start:], D.dates[start:]), "years": yearly(v, D.dates)} for k, v in bench.items()}
    print("2부: 매일 갈아타기 검증 (선택 규칙 격자)")
    recs = [run(D, m, record=True) for m in models]
    R = np.c_[[rets(r["E"]) for r in recs]].T
    hold_r = rets(bench["지수 ETF 보유"])
    if not quick:
        grid = rule_grid(D, models, recs, R, start)
        g = pd.DataFrame([{"pool": x["pool"], "all": x["all"]["sharpe"], "dev": x["dev"]["sharpe"], "test": x["test"]["sharpe"]} for x in grid])
        bm = res["bench"]["지수 ETF 보유"]
        res["grid"] = {"n": len(grid), "rows": grid, "summary": {
            p: {"n": int(len(x)), "q": [round(float(v), 3) for v in x["all"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])],
                "beat_bench": round(float((x["all"] > bm["meta"]["sharpe"]).mean()), 3),
                "dev_test_spearman": round(float(x[["dev", "test"]].corr("spearman").iloc[0, 1]), 3)}
            for p, x in g.groupby("pool")}}
    recs_s = [run(D, m, record=True, stress=3.0) for m in models]
    R_s = np.c_[[rets(r["E"]) for r in recs_s]].T
    te = int(np.searchsorted(D.dates, pd.Timestamp("2023-01-02")))
    ids = [m["id"] for m in models]
    out = {}
    for key, (name, rule) in SHADOW_RULES.items():
        E, sw, ch = apply_rule(D, models, recs, R, rule, start)
        r = rets(E)
        picks = pd.Series([ids[c] for c in ch[start:] if isinstance(c, int)]).value_counts().head(6)
        out[key] = {"name": name, "rule": rule, "switches": sw, "all": stats(E[start:], D.dates[start:]),
                    "dev": stats(E, D.dates, a=str(D.dates[start].date()), b=DEV_END), "test": stats(E, D.dates, a="2023-01-01"),
                    "p_vs_hold": boot_p(r[start:], hold_r[start:]), "p_vs_hold_test": boot_p(r[te:], hold_r[te:]),
                    "stress": stats(apply_rule(D, models, recs_s, R_s, rule, start, 3.0)[0][start:], D.dates[start:]),
                    "years": yearly(E[start:], D.dates[start:]),
                    "picks": {k2: int(v) for k2, v in picks.items()},
                    "curve": [round(float(x), 4) for x in E[start::5]]}
        print(f"  {name:40s} 교체 {sw:4d} · " + json.dumps(out[key]["all"], ensure_ascii=False) + f" · P={out[key]['p_vs_hold']}", flush=True)
    res["rules"] = out
    res["curve_dates"] = [str(d.date()) for d in D.dates[start::5]]
    res["bench_curve"] = [round(float(x / bench["지수 ETF 보유"][start]), 4) for x in bench["지수 ETF 보유"][start::5]]
    os.makedirs(OUT_DIR, exist_ok=True)
    from short_trader import clean
    res = clean(res)
    json.dump(res, open(os.path.join(OUT_DIR, "lab.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    slim = {k: v for k, v in res.items() if k not in ("models", "grid")}
    slim["top_models"] = sorted([{k: r[k] for k in ("id", "desc", "full", "dev", "test", "stress", "turnover_x", "buys_per_year")}
                                 for r in rows], key=lambda r: -(r["dev"] or {}).get("sharpe", -9))[:15]
    slim["etf_models"] = [{k: r[k] for k in ("id", "desc", "full", "dev", "test", "stress", "turnover_x")} for r in rows if r["kind"] == "etf"]
    if "grid" in res:
        slim["grid_summary"] = res["grid"]["summary"]
    json.dump(slim, open("docs/short_lab.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"끝 ({time.time() - t0:.0f}s) → {OUT_DIR}/lab.json · docs/short_lab.json")
    return res


if __name__ == "__main__":
    main()
