"""
단기매매용 데이터 패널 — 연구(short_lab)와 매일 모의투자(short_trader)가 **같은 함수**를 쓴다.

  가격: data/ohlcv.csv (시가·고가·저가·종가·거래량·거래대금, KRX 원본 = 분할 미보정)
        → 하루에 ±30%를 넘는 변화(가격제한폭 밖 = 분할·병합·무상증자)는 data/prices.csv(수정주가) 또는
          시가총액 연속성으로 비율을 구해 과거 가격을 소급 보정한다. 둘 다 안 되면 그날을 '끊김'으로 표시.
  수급: data/flows.csv (외국인·기관·개인 순매수 금액, 원)
  시총: data/marketcap.csv
  유니버스: 날짜마다 시가총액 상위 100 (그날 종가가 있는 종목 중). 현재 상장 종목만 있어 생존편향이 남는다.

모든 행렬은 (거래일 T × 종목 N) numpy 배열. t행 값은 t일 장 마감 뒤에 알 수 있는 정보만 쓴다.
"""
import os

import numpy as np
import pandas as pd

OHLCV = "data/ohlcv.csv"
PRICES = "data/prices.csv"
FLOWS = "data/flows.csv"
MCAP = "data/marketcap.csv"
UNIVERSE_N = 100
LIMIT = 0.305          # 가격제한폭 ±30% (+여유) — 이보다 큰 하루 변화는 거래로 생길 수 없다


def _read(path, cols):
    df = pd.read_csv(path, dtype={"종목코드": str}, usecols=["날짜", "종목코드"] + cols)
    df["종목코드"] = df["종목코드"].str.zfill(6)
    return df


def _pivot(df, col, dates, codes):
    return (df.pivot_table(index="날짜", columns="종목코드", values=col, aggfunc="last")
              .reindex(index=dates, columns=codes).to_numpy(dtype=float, copy=True))


def adjust_factors(close, adj_close, mcap):
    """분할 보정 계수 (T×N). 보정가 = 원본가 × 계수. 끊김 위치는 breaks(T×N, bool)."""
    T, N = close.shape
    fac = np.ones((T, N))
    breaks = np.zeros((T, N), bool)
    for j in range(N):
        c = close[:, j]
        idx = np.where(~np.isnan(c))[0]
        if len(idx) < 2:
            continue
        a, b = idx[:-1], idx[1:]                     # 직전 거래일 → 다음 거래일 (휴장·정지 건너뜀)
        raw = c[b] / c[a]
        ev = np.where((raw > 1 + LIMIT) | (raw < 1 - LIMIT))[0]
        for e in ev:
            p, q = a[e], b[e]
            true = np.nan
            if adj_close is not None and not np.isnan(adj_close[p, j]) and not np.isnan(adj_close[q, j]):
                t = adj_close[q, j] / adj_close[p, j]
                if 1 - LIMIT <= t <= 1 + LIMIT:
                    true = t
            if np.isnan(true) and not np.isnan(mcap[p, j]) and not np.isnan(mcap[q, j]) and mcap[p, j] > 0:
                t = mcap[q, j] / mcap[p, j]
                if 1 - LIMIT <= t <= 1 + LIMIT:
                    true = t
            if np.isnan(true):
                breaks[q, j] = True
                continue
            fac[:q, j] *= raw[e] / true              # q 이전 가격 전부에 곱한다
    return fac, breaks


def load(min_date=None):
    o = _read(OHLCV, ["시가", "고가", "저가", "종가", "거래량", "거래대금"])
    if min_date:
        o = o[o["날짜"] >= min_date]
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(o["날짜"].unique())))
    o["날짜"] = pd.to_datetime(o["날짜"])
    codes = sorted(o["종목코드"].unique())
    P = {"dates": dates, "codes": codes}
    for k, col in [("O", "시가"), ("H", "고가"), ("L", "저가"), ("C", "종가"), ("V", "거래량"), ("VAL", "거래대금")]:
        P[k] = _pivot(o, col, dates, codes)
    for k in ("O", "H", "L", "C"):
        P[k][P[k] <= 0] = np.nan                     # 거래정지일 0 표기
    m = _read(MCAP, ["시가총액"])
    m["날짜"] = pd.to_datetime(m["날짜"])
    P["MC"] = _pivot(m, "시가총액", dates, codes)
    adj = None
    if os.path.exists(PRICES):
        a = _read(PRICES, ["종가"])
        a["날짜"] = pd.to_datetime(a["날짜"])
        adj = _pivot(a, "종가", dates, codes)
    fac, breaks = adjust_factors(P["C"], adj, P["MC"])
    for k in ("O", "H", "L", "C"):
        P[k] = P[k] * fac
    P["V"] = P["V"] / fac
    P["breaks"] = breaks
    f = _read(FLOWS, ["외국인순매수", "기관순매수", "개인순매수"])
    f["날짜"] = pd.to_datetime(f["날짜"])
    P["FOR"] = _pivot(f, "외국인순매수", dates, codes)
    P["INS"] = _pivot(f, "기관순매수", dates, codes)
    P["IND"] = _pivot(f, "개인순매수", dates, codes)
    # 시점별 유니버스: 그날 시총 상위 100 (그날 종가·시총이 있는 종목)
    mc = np.where(np.isnan(P["C"]), np.nan, P["MC"])
    rank = np.argsort(np.argsort(-np.nan_to_num(mc, nan=-1), axis=1), axis=1)
    P["U"] = (rank < UNIVERSE_N) & ~np.isnan(mc)
    # 마지막 알려진 가격 (평가용, 정지 종목은 직전가로)
    P["Cf"] = pd.DataFrame(P["C"]).ffill().to_numpy(copy=True)
    return P


def market(P):
    """시장 지수(유니버스 시총가중·등가중, 종가 기준)와 국면 표시."""
    C, U, MC = P["C"], P["U"], P["MC"]
    r = C[1:] / C[:-1] - 1
    r[P["breaks"][1:]] = np.nan
    u = U[:-1]
    w = np.where(u & ~np.isnan(r), MC[:-1], 0.0)
    cw = np.nansum(np.where(w > 0, r, 0) * w, 1) / np.maximum(w.sum(1), 1)
    ew = np.nanmean(np.where(u, r, np.nan), 1)
    out = {"ret_cw": np.r_[0, np.nan_to_num(cw)], "ret_ew": np.r_[0, np.nan_to_num(ew)]}
    idx = np.cumprod(1 + out["ret_cw"])
    out["idx"] = idx
    s = pd.Series(idx)
    out["ma60"] = s.rolling(60, min_periods=40).mean().to_numpy()
    out["ma120"] = s.rolling(120, min_periods=80).mean().to_numpy()
    out["dd"] = idx / s.cummax().to_numpy() - 1
    out["rebound"] = idx / s.rolling(20, min_periods=10).min().to_numpy() - 1
    # 급락장 게이트 (자동매매 엔진과 같은 정의): 고점대비 -15%↓ 이고 20일 저점 대비 +5% 반등 전이면 신규매수 중단
    out["crash_off"] = (out["dd"] <= -0.15) & (out["rebound"] < 0.05)
    out["trend_off"] = idx < out["ma120"]
    return out


# ══════════════════════════════════════════════════════════════
# 지수 ETF (KODEX 200 · KODEX 인버스) — 국내 주식형 ETF는 매도 시 증권거래세가 없다
#   data/etf_ohlcv.csv 가 있으면 실제 ETF 시가·종가, 없으면 유니버스 시총가중 지수로 근사
# ══════════════════════════════════════════════════════════════
ETF = "data/etf_ohlcv.csv"
ETF_CODES = {"IDX": "069500", "INV": "114800"}


def index_legs(P):
    """시총가중 지수의 밤사이(전일 종가→시가)·장중(시가→종가)·하루(종가→종가) 수익률."""
    O, C, MC, U = P["O"], P["C"], P["MC"], P["U"]
    T = C.shape[0]
    co, oc, cc = np.zeros(T), np.zeros(T), np.zeros(T)
    w = np.where(U & ~np.isnan(C) & ~np.isnan(MC), MC, 0.0)
    for t in range(1, T):
        m = (w[t - 1] > 0) & ~np.isnan(O[t]) & ~np.isnan(C[t]) & ~np.isnan(C[t - 1]) & ~P["breaks"][t]
        if m.sum() < 20:
            continue
        ww = w[t - 1, m] / w[t - 1, m].sum()
        g = O[t, m] / C[t - 1, m]
        co[t] = (ww * g).sum() - 1
        wo = ww * g / (ww * g).sum()
        oc[t] = (wo * C[t, m] / O[t, m]).sum() - 1
        cc[t] = (1 + co[t]) * (1 + oc[t]) - 1
    return co, oc, cc


def etf_prices(P):
    """(시가, 종가) 두 행렬 (T×2): 0열 = 지수 ETF, 1열 = 인버스 ETF. source 는 'etf' 또는 'proxy'."""
    dates = P["dates"]
    co, oc, cc = index_legs(P)
    pc = np.cumprod(1 + cc)
    po = np.r_[1.0, pc[:-1]] * (1 + co)
    ic = np.cumprod(1 - cc)                       # 인버스: 매일 종가 기준 -1배
    io = np.r_[1.0, ic[:-1]] * (1 - co)
    Op = np.c_[po, io] * 10000
    Cp = np.c_[pc, ic] * 10000
    src = "proxy"
    if os.path.exists(ETF):
        e = pd.read_csv(ETF, dtype={"종목코드": str})
        e["종목코드"] = e["종목코드"].str.zfill(6)
        e["날짜"] = pd.to_datetime(e["날짜"])
        ok = True
        for j, k in enumerate(("IDX", "INV")):
            x = e[e["종목코드"] == ETF_CODES[k]].set_index("날짜").reindex(dates)
            if x["종가"].notna().mean() < 0.9:
                ok = False
                break
            o_, c_ = x["시가"].to_numpy(float, copy=True), x["종가"].to_numpy(float, copy=True)
            o_[o_ <= 0] = np.nan
            c_[c_ <= 0] = np.nan
            fac = np.ones(len(c_))                      # 분할·병합(하루 ±30% 넘는 변화)은 그날 수익 0으로 보고 과거를 소급 보정
            ix = np.where(~np.isnan(c_))[0]
            for p, q in zip(ix[:-1], ix[1:]):
                r = c_[q] / c_[p]
                if r > 1 + LIMIT or r < 1 - LIMIT:
                    fac[:q] *= r
            o_, c_ = o_ * fac, c_ * fac
            have = ~np.isnan(o_) & ~np.isnan(c_)
            # 실제 값이 있는 날은 실제 ETF, 빠진 날은 지수 근사로 이어 붙인다
            r_c = np.where(have, c_, np.nan)
            Op[:, j] = np.where(have, o_, Op[:, j] * np.nanmedian(r_c / Cp[:, j]))
            Cp[:, j] = np.where(have, c_, Cp[:, j] * np.nanmedian(r_c / Cp[:, j]))
        if ok:
            src = "etf"
    return Op, Cp, {"co": co, "oc": oc, "cc": cc, "source": src}


# ══════════════════════════════════════════════════════════════
# 신호 — 값이 클수록 '산다'. t행은 t일 장 마감까지의 정보만.
# ══════════════════════════════════════════════════════════════
def _lag(x, k):
    y = np.full_like(x, np.nan)
    y[k:] = x[:-k]
    return y


def _roll(x, n, f="mean"):
    r = pd.DataFrame(x).rolling(n, min_periods=max(2, int(n * 0.7)))
    return getattr(r, f)().to_numpy(copy=True)


SIGNAL_INFO = {
    "rev1": "오늘 하락폭 (1일 반전)",
    "rev3": "3일 하락폭",
    "rev_id": "오늘 장중 하락폭 (시가→종가)",
    "clv_lo": "종가가 당일 저가 근처",
    "vsurge_dn": "거래량 급증 + 하락",
    "gap_rev": "오늘 시가 갭 하락",
    "ma20rev": "20일선 아래 이격",
    "lowvol": "변동성 낮음",
    "lomax": "최근 급등일 없음",
    "fi1_rev": "외국인+기관 오늘 순매도",
    "mom120": "6개월 추세",
    "hi52": "52주 고가 근접",
    "for20": "외국인 20일 순매수",
}


def signals(P):
    O, H, L, C, V, MC = P["O"], P["H"], P["L"], P["C"], P["V"], P["MC"]
    with np.errstate(all="ignore"):
        r1 = C / _lag(C, 1) - 1
        r1[P["breaks"]] = np.nan
        S = {
            "rev1": -r1,
            "rev3": -(C / _lag(C, 3) - 1),
            "rev_id": -(C / O - 1),
            "clv_lo": -((C - L) / (H - L)),
            "vsurge_dn": (V / _lag(_roll(V, 20), 1)) * (r1 < 0),
            "gap_rev": -(O / _lag(C, 1) - 1),
            "ma20rev": -(C / _roll(C, 20) - 1),
            "lowvol": -_roll(r1, 20, "std"),
            "lomax": -_roll(r1, 21, "max"),
            "fi1_rev": -(P["FOR"] + P["INS"]) / MC,
            "mom120": _lag(C, 5) / _lag(C, 120) - 1,
            "hi52": C / _roll(C, 252, "max"),
            "for20": _roll(P["FOR"], 20, "sum") / MC,
        }
    for k in S:
        S[k][~np.isfinite(S[k])] = np.nan
    return S


def rank_score(S, U, combo):
    """조합 점수 = Σ 가중치 × (유니버스 안 백분위). 반환: 순위(1=최고, 유니버스 밖 = inf)."""
    tot = None
    for name, w in combo:
        x = np.where(U, S[name], np.nan)
        pr = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy(copy=True)
        pr = np.nan_to_num(pr, nan=0.5)            # 값이 없으면 중간
        tot = w * pr if tot is None else tot + w * pr
    tot = np.where(U, tot, np.nan)
    rk = pd.DataFrame(-tot).rank(axis=1, method="first").to_numpy(copy=True)
    rk[np.isnan(rk)] = np.inf
    return rk


def market_signals(P, legs):
    """지수 단위 신호 (t일 마감 기준)."""
    oc, cc = legs["oc"], legs["cc"]
    idx = np.cumprod(1 + cc)
    s = pd.Series(idx)
    w = np.where(P["U"] & ~np.isnan(P["MC"]), P["MC"], 0.0)
    tot = np.maximum(w.sum(1), 1)
    mfor = np.nansum(np.where(w > 0, P["FOR"], 0), 1) / tot
    return {
        "m_rev1": -cc,
        "m_rev_id": -oc,
        "m_for1": mfor,
        "m_trend120": idx / s.rolling(120, min_periods=80).mean().to_numpy() - 1,
        "m_trend20": idx / s.rolling(20, min_periods=15).mean().to_numpy() - 1,
        "m_dd": idx / s.cummax().to_numpy() - 1,
    }
