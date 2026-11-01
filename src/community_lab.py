"""
남들이 해 본 전략 검증실 — 국내외 개인 투자자·블로그·책에서 가장 많이 쓰는 자동매매·자산배분 전략을 '우리 잣대'로 다시 잰다.

  잣대 (다른 연구와 같음)
    · 그날 저녁까지의 자료로 판단 → 다음 거래일 시가 체결 · 수수료·미끄러짐·세금 포함
    · '발표 뒤 성적' = 진짜 표본 외 (저자는 발표 전 자료만 보고 규칙을 만들었다. McLean·Pontiff 2016: 발표 뒤 평균 58% 줄어듦)
    · 공통 기간(2007-03~)에 60/40 · 우리 전천후 혼합과 비교, 여러 전략을 한꺼번에 봤으니 Holm 보정
    · 원화·세후(해외 ETF 양도세 22% 근사)로도 — 한국 사람이 실제로 손에 쥐는 숫자
  전략
    미국 ETF 자산배분: 60/40 · 영구 포트폴리오(Browne) · 올웨더(Dalio/Robbins) · 듀얼 모멘텀 GEM(Antonacci) · VAA-G4 · DAA-G12 · HAA(Keller)
                       · GTAA5 10개월선(Faber) · LAA 근사(강환국, 실업률 조건 빠짐)
    국내: 변동성 돌파(Larry Williams, k=0.5) — 실제 KODEX 200 2014-08~ (그대로 · 5일선 위 · 노이즈 k) · 대형주 2001~ (생존편향)
          코스피 지수 1997~ 판은 참고만 (지수 시가가 장 시작 직후 가격을 덜 반영 → 같은 기간 실제 ETF 의 두 배로 부풀려짐)
  자료가 늦게 생긴 ETF 는 비슷한 ETF 로 앞을 잇는다 (PROXY). 대용을 쓴 구간은 결과에 표시.

  python src/community_lab.py → data/community/lab.json · docs/community.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import history_lab as HL  # noqa: E402
import science_lab as S  # noqa: E402
import short_lab as L  # noqa: E402

warnings.filterwarnings("ignore")
OUT = "data/community/lab.json"
UI = "docs/community.json"
COMMON = "2007-03-01"
US_TAX = 0.22
PROXY = {"AGG": "IEF", "BND": "AGG", "LQD": "IEF", "HYG": "LQD", "TIP": "IEF", "BIL": "SHY", "VEA": "EFA", "VWO": "EEM",
         "VGK": "EFA", "EWJ": "EFA", "GSG": "DBC", "VTI": "SPY"}
COST_US = sum(L.costs("us")) / 2          # 한쪽 (수수료 0.1% + 미끄러짐 0.03% + SEC)


# ══════════════════════════════════════════════════════════════
# 자료
# ══════════════════════════════════════════════════════════════
def us_panel(h, syms):
    """미국 ETF (시가·종가) — SPY 거래일 축. 늦게 생긴 ETF 는 PROXY 수익률로 앞을 잇는다."""
    dates = HL.series(h, "SPY").index
    dates = dates[dates >= "1999-01-01"]
    O, C, spliced = {}, {}, {}

    def get(sym, depth=0):
        if sym in C:
            return O[sym], C[sym]
        c = HL.series(h, sym).reindex(dates)
        o = HL.series(h, sym, "시가").reindex(dates)
        p = PROXY.get(sym)
        if p and depth < 3:
            po, pc = get(p, depth + 1)
            first = c.first_valid_index()
            if first is None:
                c, o = pc.copy(), po.copy()
                spliced[sym] = (p, "전 기간")
            else:
                before = pc.index < first
                if before.any() and pc[before].notna().any():
                    k = c[first] / pc[first] if pd.notna(pc[first]) else np.nan
                    c[before], o[before] = pc[before] * k, po[before] * k
                    spliced[sym] = (p, f"~{first.date()}")
        o = o.where(o > 0, c.shift(1))
        O[sym], C[sym] = o, c
        return o, c
    for s in syms:
        get(s)
    Op = pd.DataFrame({s: O[s] for s in syms})
    Cp = pd.DataFrame({s: C[s] for s in syms})
    return dates, Op, Cp, {s: spliced[s] for s in syms if s in spliced}


def month_ends(dates):
    d = pd.DatetimeIndex(dates)
    return np.r_[d.month[1:] != d.month[:-1], True]


def run_weights(dec, Op, Cp, cost=COST_US):
    """dec[t] = t일 종가 뒤 정한 목표 비중 (NaN 줄 = 그대로). t+1일 시가에 맞춘다. 남는 비중은 현금(0%).
    반환: 일별 수익 (그날 시가 전 '밤사이'는 옛 비중, 시가 뒤는 새 비중)."""
    O, C = Op.to_numpy(float), Cp.to_numpy(float)
    T, A = C.shape
    r = np.zeros(T)
    w = np.zeros(A)
    turn_tot = 0.0
    for t in range(1, T):
        g = np.nan_to_num(O[t] / C[t - 1] - 1)
        d = np.nan_to_num(C[t] / O[t] - 1)
        rg = float((w * g).sum())
        w = w * (1 + g) / (1 + rg) if abs(1 + rg) > 1e-12 else w
        c = 0.0
        tgt = dec[t - 1]
        if not np.isnan(tgt).all():
            tgt = np.nan_to_num(tgt)
            tv = float(np.abs(tgt - w).sum())
            c = tv * cost
            turn_tot += tv
            w = tgt.copy()
        rd = float((w * d).sum())
        r[t] = (1 + rg) * (1 - c) * (1 + rd) - 1
        w = w * (1 + d) / (1 + rd) if abs(1 + rd) > 1e-12 else w
    return r, turn_tot


# ══════════════════════════════════════════════════════════════
# 미국 ETF 자산배분 전략 — 각 함수: (월말 판단 행렬 dec, 쓰는 종목)
# ══════════════════════════════════════════════════════════════
def mom(Cp, me_idx, k):
    """월말 i 에서 k개월 수익 (월말 종가 기준)."""
    m = Cp.iloc[me_idx]
    return (m / m.shift(k) - 1)


def w13612(Cp, me):
    return 12 * mom(Cp, me, 1) + 4 * mom(Cp, me, 3) + 2 * mom(Cp, me, 6) + mom(Cp, me, 12)


def avg_mom(Cp, me):
    return (mom(Cp, me, 1) + mom(Cp, me, 3) + mom(Cp, me, 6) + mom(Cp, me, 12)) / 4


def strategies():
    """이름 → (설명, 출처, 발표, 종목, 판단 함수(Cp, me, cols) → 월말별 비중 dict 목록, 재조정 주기)."""
    def fixed(wts):
        return lambda Cp, me, cols: [dict(wts) for _ in me]

    def gem(Cp, me, cols):
        r12 = mom(Cp, me, 12)
        out = []
        for i in range(len(me)):
            x = r12.iloc[i]
            if x[["SPY", "EFA", "BIL"]].isna().any():
                out.append(None)
                continue
            if x["SPY"] > x["BIL"]:
                out.append({"SPY" if x["SPY"] >= x["EFA"] else "EFA": 1.0})
            else:
                out.append({"AGG": 1.0})
        return out

    def vaa(Cp, me, cols):
        sc = w13612(Cp, me)
        off, de = ["SPY", "EFA", "EEM", "AGG"], ["LQD", "IEF", "SHY"]
        out = []
        for i in range(len(me)):
            x = sc.iloc[i]
            if x[off + de].isna().any():
                out.append(None)
                continue
            out.append({x[off].idxmax(): 1.0} if (x[off] > 0).all() else {x[de].idxmax(): 1.0})
        return out

    def daa(Cp, me, cols):
        sc = w13612(Cp, me)
        risky = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "EEM", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD"]
        can, cash = ["VWO", "BND"], ["SHY", "IEF", "LQD"]
        out = []
        for i in range(len(me)):
            x = sc.iloc[i]
            if x[risky + can + cash].isna().any():
                out.append(None)
                continue
            b = int((x[can] <= 0).sum())
            cf = {0: 0.0, 1: 0.5, 2: 1.0}[b]
            w = {}
            if cf < 1:
                for s in x[risky].sort_values(ascending=False).index[:6]:
                    w[s] = w.get(s, 0) + (1 - cf) / 6
            if cf > 0:
                c = x[cash].idxmax()
                w[c] = w.get(c, 0) + cf
            out.append(w)
        return out

    def haa(Cp, me, cols):
        sc = avg_mom(Cp, me)
        off = ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"]
        out = []
        for i in range(len(me)):
            x = sc.iloc[i]
            if x[off + ["TIP", "BIL", "IEF"]].isna().any():
                out.append(None)
                continue
            cash = "BIL" if x["BIL"] >= x["IEF"] else "IEF"
            if x["TIP"] <= 0:
                out.append({cash: 1.0})
                continue
            w = {}
            for s in x[off].sort_values(ascending=False).index[:4]:
                k = s if x[s] > 0 else cash
                w[k] = w.get(k, 0) + 0.25
            out.append(w)
        return out

    def gtaa(Cp, me, cols):
        assets = ["SPY", "EFA", "IEF", "VNQ", "DBC"]
        m = Cp.iloc[me]
        sma = Cp.rolling(210, min_periods=200).mean().iloc[me]
        out = []
        for i in range(len(me)):
            x, s = m.iloc[i], sma.iloc[i]
            if x[assets].isna().any() or s[assets].isna().any():
                out.append(None)
                continue
            w = {}
            for a in assets:
                k = a if x[a] > s[a] else "SHY"
                w[k] = w.get(k, 0) + 0.2
            out.append(w)
        return out

    def laa(Cp, me, cols):
        ma = Cp["SPY"].rolling(200, min_periods=190).mean().iloc[me]
        px = Cp["SPY"].iloc[me]
        out = []
        for i in range(len(me)):
            if pd.isna(ma.iloc[i]):
                out.append(None)
                continue
            last = "QQQ" if px.iloc[i] >= ma.iloc[i] else "SHY"
            out.append({"IWM": 0.25, "GLD": 0.25, "IEF": 0.25, last: 0.25})
        return out

    return {
        "60_40": ("60/40 (주식 60 · 채권 40)", "전통적 기준", "1950", ["SPY", "AGG"], fixed({"SPY": 0.6, "AGG": 0.4}), "Q"),
        "permanent": ("영구 포트폴리오 (주식·장기채·금·현금 25%씩)", "Harry Browne 1981", "1981-01-01",
                      ["SPY", "TLT", "GLD", "BIL"], fixed({"SPY": .25, "TLT": .25, "GLD": .25, "BIL": .25}), "Y"),
        "allweather": ("올웨더 (주식 30 · 장기채 40 · 중기채 15 · 금 7.5 · 원자재 7.5)", "Ray Dalio / Tony Robbins 2014", "2014-11-18",
                       ["SPY", "TLT", "IEF", "GLD", "DBC"], fixed({"SPY": .3, "TLT": .4, "IEF": .15, "GLD": .075, "DBC": .075}), "Q"),
        "gem": ("듀얼 모멘텀 GEM: 미국>단기채면 미국·선진국 중 12개월 1등, 아니면 채권", "Gary Antonacci 2014", "2014-11-01",
                ["SPY", "EFA", "AGG", "BIL"], gem, "M"),
        "vaa_g4": ("VAA-G4: 공격 4종 점수 모두 +면 공격 1등, 하나라도 −면 방어 1등 (13612W 점수)", "Keller·Keuning 2017", "2017-07-14",
                   ["SPY", "EFA", "EEM", "AGG", "LQD", "IEF", "SHY"], vaa, "M"),
        "daa_g12": ("DAA-G12: 카나리아(신흥국·채권) 나빠진 수만큼 현금, 나머지 위험자산 12종 중 상위 6", "Keller·Keuning 2018", "2018-07-23",
                    ["SPY", "IWM", "QQQ", "VGK", "EWJ", "EEM", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD", "VWO", "BND", "SHY", "IEF"], daa, "M"),
        "haa": ("HAA: 물가연동채(TIP) 추세가 +면 공격 8종 중 상위 4, 아니면 현금", "Keller·Keuning 2023", "2023-02-01",
                ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT", "TIP", "BIL"], haa, "M"),
        "gtaa5": ("GTAA5: 5개 자산 20%씩, 10개월선 아래면 그 몫은 단기채", "Mebane Faber 2007", "2007-01-01",
                  ["SPY", "EFA", "IEF", "VNQ", "DBC", "SHY"], gtaa, "M"),
        "laa_approx": ("LAA 근사: 소형주·금·중기채 25%씩 + 나스닥(S&P 200일선 위)/단기채 25% — 실업률 조건 빠짐", "강환국 2019 (근사)", "2019-01-01",
                       ["IWM", "GLD", "IEF", "QQQ", "SHY", "SPY"], laa, "M"),
    }


def run_us(h):
    st = strategies()
    syms = sorted({s for v in st.values() for s in v[3]} | {"SPY"})
    dates, Op, Cp, spliced = us_panel(h, syms)
    me = np.where(month_ends(dates))[0]
    out = {}
    for key, (name, src, pub, cols, fn, freq) in st.items():
        ws = fn(Cp, me, cols)
        dec = np.full((len(dates), len(syms)), np.nan)
        first = None
        for i, w in zip(me, ws):
            if w is None:
                continue
            d = dates[i]
            if first is not None and freq == "Q" and d.month not in (3, 6, 9, 12):
                continue
            if first is not None and freq == "Y" and d.month != 12:
                continue
            row = np.zeros(len(syms))
            for s, x in w.items():
                row[syms.index(s)] += x
            dec[i] = row
            first = first if first is not None else i
        if first is None:
            continue
        r, turn = run_weights(dec, Op, Cp)
        r[:first + 1] = np.nan
        yrs = (len(dates) - first) / 252
        out[key] = {"name": name, "source": src, "pub": pub, "tickers": cols, "freq": freq, "start": str(dates[first + 1].date()),
                    "turnover_yr": round(turn / max(yrs, 1e-9), 2),
                    "proxies": {s: list(spliced[s]) for s in cols if s in spliced}, "r": pd.Series(r, index=dates)}
    out["spy"] = {"name": "S&P 500 (SPY) 보유", "source": "기준", "pub": "1993", "tickers": ["SPY"], "freq": "-", "start": str(dates[1].date()),
                  "turnover_yr": 0, "proxies": {}, "r": pd.Series(np.r_[np.nan, Cp["SPY"].pct_change().to_numpy()[1:]], index=dates)}
    return out


# ══════════════════════════════════════════════════════════════
# 국내: 변동성 돌파 (Larry Williams)
# ══════════════════════════════════════════════════════════════
def breakout_index(h, k=0.5, ma=None, noise=False):
    """코스피 지수(=지수 ETF) 변동성 돌파: 오늘 고가가 시가+k×(어제 고가−저가)를 넘으면 그 값에 사고 다음 날 시가에 판다."""
    ks = h[h["종목코드"] == "^KS11"].drop_duplicates("날짜").set_index("날짜").sort_index()
    ks = ks[(ks["시가"] > 0) & (ks["고가"] > 0) & (ks["저가"] > 0)]
    O, Hh, Lw, C = (ks[c].to_numpy(float) for c in ("시가", "고가", "저가", "종가"))
    dates = ks.index
    rng = np.r_[np.nan, (Hh - Lw)[:-1]]
    kk = np.full(len(C), k)
    if noise:
        nr = 1 - np.abs(O - C) / np.where(Hh - Lw > 0, Hh - Lw, np.nan)
        kk = np.r_[np.nan, pd.Series(nr).rolling(20, min_periods=15).mean().to_numpy()[:-1]]
    tgt = O + kk * rng
    ok = (Hh >= tgt) & np.isfinite(tgt)
    if ma:
        m = pd.Series(C).rolling(ma).mean().to_numpy()
        ok &= np.r_[False, (C > m)[:-1]]
    side = HL.SLIP["069500"] + 0.000186
    r = np.zeros(len(C))
    for t in np.where(ok[:-1])[0]:
        r[t] += C[t] / tgt[t] - 1 - side
        r[t + 1] += O[t + 1] / C[t] - 1 - side
    return pd.Series(r, index=dates), float(ok.mean())


def breakout_etf(k=0.5, ma=None, noise=False):
    """실제 KODEX 200 시가·고가·저가·종가(2014-08~)로 같은 규칙 — 판정은 이것으로 (지수 시가는 장 시작 직후 덜 반영돼 결과가 부풀려진다)."""
    import allweather as AW
    e = pd.read_csv("data/etf_ohlcv.csv", dtype={"종목코드": str})
    e["종목코드"] = e["종목코드"].str.zfill(6)
    k_ = e[e["종목코드"] == "069500"].copy()
    k_["날짜"] = pd.to_datetime(k_["날짜"])
    k_ = k_.drop_duplicates("날짜").set_index("날짜").sort_index()
    k_ = k_[(k_["시가"] > 0) & (k_["고가"] > 0) & (k_["저가"] > 0)]
    O, Hh, Lw, C = (k_[c].to_numpy(float) for c in ("시가", "고가", "저가", "종가"))
    rng = np.r_[np.nan, (Hh - Lw)[:-1]]
    kk = np.full(len(C), k)
    if noise:
        nr = 1 - np.abs(O - C) / np.where(Hh - Lw > 0, Hh - Lw, np.nan)
        kk = np.r_[np.nan, pd.Series(nr).rolling(20, min_periods=15).mean().to_numpy()[:-1]]
    tgt = O + kk * rng
    ok = (Hh >= tgt) & np.isfinite(tgt)
    if ma:
        m = pd.Series(C).rolling(ma).mean().to_numpy()
        ok &= np.r_[False, (C > m)[:-1]]
    r = np.zeros(len(C))
    parts = {"intraday": 0.0, "overnight": 0.0, "cost": 0.0}
    for t in np.where(ok[:-1])[0]:
        cb = float(AW.costs(np.array([tgt[t]]))[0][0])
        cs = float(AW.costs(np.array([O[t + 1]]))[1][0])
        a, b = C[t] / tgt[t] - 1, O[t + 1] / C[t] - 1
        r[t] += a - cb
        r[t + 1] += b - cs
        parts["intraday"] += a
        parts["overnight"] += b
        parts["cost"] += cb + cs
    yrs = len(C) / 252
    return pd.Series(r, index=k_.index), float(ok.mean()), {k2: round(v / yrs, 4) for k2, v in parts.items()}


def breakout_stocks(k=0.5, slots=10):
    """국내 대형주(거래대금 상위 100) 변동성 돌파 — 돌파한 종목 중 노이즈 낮은 순 최대 10개, 한 자리 10%. 주식 왕복 비용(세금 포함)."""
    P = HL.stock_panel(HL.KR_STOCKS, ("^KS11",), "kr")
    O, Hh, Lw, C, U = P["O"], P["H"], P["L"], P["C"], P["U"]
    T = len(P["dates"])
    rng = np.r_[np.full((1, O.shape[1]), np.nan), (Hh - Lw)[:-1]]
    tgt = O + k * rng
    nr = 1 - np.abs(O - C) / np.where(Hh - Lw > 0, Hh - Lw, np.nan)
    nz = np.r_[np.full((1, O.shape[1]), np.nan), pd.DataFrame(nr).rolling(20, min_periods=15).mean().to_numpy()[:-1]]
    cb, cs = L.costs("stock")
    r = np.zeros(T)
    n_tr = 0
    uprev = np.r_[np.zeros((1, O.shape[1]), bool), U[:-1]]
    for t in range(1, T - 1):
        ok = uprev[t] & np.isfinite(tgt[t]) & (Hh[t] >= tgt[t]) & np.isfinite(O[t + 1]) & np.isfinite(C[t]) & ~P["breaks"][t] & ~P["breaks"][t + 1]
        idx = np.where(ok)[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(np.nan_to_num(nz[t, idx], nan=1))][:slots]
        n_tr += len(idx)
        r[t] += np.mean(C[t, idx] / tgt[t, idx] - 1) * len(idx) / slots - cb * len(idx) / slots
        r[t + 1] += np.mean(O[t + 1, idx] / C[t, idx] - 1) * len(idx) / slots - cs * len(idx) / slots
    return pd.Series(r, index=P["dates"]), n_tr


# ══════════════════════════════════════════════════════════════
def to_krw(r, fx, tax=True):
    """달러 일수익 → 원화(환율 변화 포함) → 세후(해마다 이익의 22%)."""
    x = r.dropna()
    f = fx.reindex(x.index).ffill().bfill()
    rk = (1 + x) * (f / f.shift(1)).fillna(1) - 1
    if not tax:
        return rk
    import us_lab as UL
    E = np.cumprod(1 + rk.to_numpy())
    return pd.Series(L.rets(UL.after_tax(E, x.index, US_TAX)), index=x.index)


def summarize(r, pub=None):
    r = r.dropna()
    d = r.index
    o = {"full": HL.stats_r(r.to_numpy(), d), "common": HL.stats_r(r[d >= COMMON].to_numpy(), d[d >= COMMON]),
         "years": L.yearly(np.cumprod(1 + r.to_numpy()), d), "rolling": HL.rolling_origin(r)}
    if pub:
        p = pd.Timestamp(pub)
        o["pre"] = HL.stats_r(r[d < p].to_numpy(), d[d < p]) if (d < p).sum() > 250 else None
        o["post"] = HL.stats_r(r[d >= p].to_numpy(), d[d >= p]) if (d >= p).sum() > 250 else None
    o["crises"] = {}
    for name, a, b in HL.CRISES:
        x = r[(d >= a) & (d <= b)]
        if len(x) >= 15 and d[0] <= pd.Timestamp(a) + pd.Timedelta(days=10):
            o["crises"][name] = round(float(np.prod(1 + x.to_numpy()) - 1), 4)
    return o


def main():
    t0 = time.time()
    h = HL.load_hist()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "common_from": COMMON, "cost_us_side": round(COST_US, 5), "us_tax": US_TAX}
    US = run_us(h)
    fx = h[h["종목코드"] == "KRW=X"].drop_duplicates("날짜").set_index("날짜")["종가"].sort_index()
    # 우리 전천후 혼합 (원화·세금 포함 시뮬레이터, 대용 자산) — 비교 기준
    dates, Op, Cp = HL.proxy_prices(h)
    _, r_aw = HL.aw_long(dates, Op, Cp)
    r_aw = r_aw.dropna()
    rows = {}
    base = US["60_40"]["r"]
    for key, v in US.items():
        r = v["r"]
        rk = to_krw(r, fx)
        o = {k: v[k] for k in ("name", "source", "pub", "tickers", "freq", "start", "turnover_yr", "proxies")}
        o["usd"] = summarize(r, v["pub"])
        o["krw_after_tax"] = summarize(rk, v["pub"])
        both = pd.concat([r, base], axis=1).dropna()
        both = both[both.index >= COMMON]
        o["p_vs_6040"] = L.boot_p(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()) if key != "60_40" else None
        ab = pd.concat([rk, r_aw], axis=1).dropna()
        ab = ab[ab.index >= COMMON]
        o["p_vs_aw"] = L.boot_p(ab.iloc[:, 0].to_numpy(), ab.iloc[:, 1].to_numpy())
        o["group"] = "us"
        rows[key] = o
    # 국내 변동성 돌파
    kospi = pd.Series(np.r_[0, Cp[1:, 0] / Cp[:-1, 0] - 1], index=dates)
    for key, (name, kw) in {"vb_etf": ("변동성 돌파 k=0.5 — 실제 KODEX 200", {}),
                            "vb_etf_ma5": ("변동성 돌파 k=0.5 + 5일선 위일 때만 — 실제 KODEX 200", {"ma": 5}),
                            "vb_etf_noise": ("변동성 돌파 k=최근 20일 노이즈 평균 — 실제 KODEX 200", {"noise": True}),
                            "vb_idx": ("(참고) 변동성 돌파 k=0.5 — 코스피 지수 1997~ (지수 시가 탓에 부풀려짐)", {"index": True})}.items():
        if kw.pop("index", False):
            r, freq = breakout_index(h, **kw)
            parts = None
        else:
            r, freq, parts = breakout_etf(**kw)
        o = {"name": name, "source": "Larry Williams (1980년대) · 국내 유튜브·블로그 2017~", "pub": "2017-01-01", "tickers": ["KODEX 200"],
             "freq": f"매수일 {freq * 100:.0f}%", "start": str(r.index[1].date()), "turnover_yr": round(freq * 2 * 252, 1), "proxies": {},
             "krw_after_tax": summarize(r, "2017-01-01"), "group": "kr", "parts": parts, "reference": key == "vb_idx"}
        o["usd"] = None
        b = pd.concat([r, kospi], axis=1).dropna()
        o["p_vs_kospi"] = L.boot_p(b.iloc[130:, 0].to_numpy(), b.iloc[130:, 1].to_numpy())
        ab = pd.concat([r, r_aw], axis=1).dropna()
        ab = ab[ab.index >= COMMON]
        o["p_vs_aw"] = L.boot_p(ab.iloc[:, 0].to_numpy(), ab.iloc[:, 1].to_numpy())
        rows[key] = o
    if os.path.exists(HL.KR_STOCKS):
        r, n = breakout_stocks()
        o = {"name": "변동성 돌파 k=0.5 — 국내 대형주 최대 10종목 (노이즈 낮은 순)", "source": "국내 블로그·책 (조코딩·systrader79 등)",
             "pub": "2017-01-01", "tickers": ["코스피100 대형주"], "freq": f"연 {n / max(len(r) / 252, 1):.0f}회 매매", "start": str(r.index[1].date()),
             "turnover_yr": round(n / max(len(r) / 252, 1) / 10 * 2, 1), "proxies": {}, "krw_after_tax": summarize(r, "2017-01-01"),
             "usd": None, "group": "kr"}
        ab = pd.concat([r, r_aw], axis=1).dropna()
        ab = ab[ab.index >= COMMON]
        o["p_vs_aw"] = L.boot_p(ab.iloc[:, 0].to_numpy(), ab.iloc[:, 1].to_numpy())
        b = pd.concat([r, kospi], axis=1).dropna()
        o["p_vs_kospi"] = L.boot_p(b.iloc[130:, 0].to_numpy(), b.iloc[130:, 1].to_numpy())
        rows["vb_stocks"] = o
    keys = [k for k in rows if k not in ("spy", "vb_idx")]
    for k, a in zip(keys, S.holm([rows[k]["p_vs_aw"] for k in keys])):
        rows[k]["p_vs_aw_holm"] = a
    rows["our_aw"] = {"name": "우리 전천후 혼합 (국내 ETF 대용 · 원화 · 세금 포함)", "source": "이 시스템", "pub": "2026-10-01", "tickers": ["국내 ETF 12종"],
                      "freq": "매달", "start": str(r_aw.index[0].date()), "turnover_yr": None, "proxies": {},
                      "krw_after_tax": summarize(r_aw), "usd": None, "group": "ours"}
    rows["kospi"] = {"name": "코스피 보유 (배당 1.5% 근사)", "source": "기준", "pub": "1997", "tickers": ["KODEX 200"], "freq": "-",
                     "start": str(dates[1].date()), "turnover_yr": 0, "proxies": {}, "krw_after_tax": summarize(kospi), "usd": None, "group": "base"}
    res["rows"] = rows
    print(f"공통 기간 {COMMON}~ · 원화·세후 기준 (우리 전천후 대비 P → Holm)")
    order = sorted(rows, key=lambda k: -((rows[k]["krw_after_tax"]["common"] or {}).get("sharpe") or -9))
    for k in order:
        v = rows[k]
        c = v["krw_after_tax"]["common"] or {}
        post = v["krw_after_tax"].get("post") or {}
        print(f"  {k:12s} 샤프 {c.get('sharpe')} · 연 {c.get('cagr')} · 낙폭 {c.get('mdd')} | 발표 뒤 샤프 {post.get('sharpe')} "
              f"| P {v.get('p_vs_aw')}→{v.get('p_vs_aw_holm')} | 3년 손실확률 {(v['krw_after_tax']['rolling'].get('3y') or {}).get('p_loss')}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    from short_trader import clean
    res = clean(res)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(res, open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"끝 ({time.time() - t0:.0f}s) → {OUT}")
    return res


if __name__ == "__main__":
    main()
