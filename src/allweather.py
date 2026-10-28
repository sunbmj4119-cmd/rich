"""
전천후(국면 스위칭) 모델의 부품 — 연구(allweather_lab)·매일 모의투자(short_trader)가 같은 함수를 쓴다.

  '어떤 시장에서도 할 일이 있게': 주식이 안 될 때 돈을 벌 수 있는 자산(인버스·달러·금·채권·미국주식)을 후보에 넣고,
  '지금이 어떤 시장인가'(국면)를 숫자로 판정해, 과거에 같은 국면에서 가장 잘 된 전략으로 갈아탄다.

국면 (그날 장 마감 뒤 KODEX 200 종가로만 판정 — 미래 정보 없음)
  급락: 최근 1년 고점 대비 -15% 이하
  약세: 200일 이동평균 아래
  강세: 200일선 위 + 최근 60일 수익 > 0
  횡보: 200일선 위지만 60일 수익 ≤ 0

전략 카드 (모두 t-1일 저녁 판단 → t일 시가 체결)
  보유      지수·코스닥·미국 S&P·나스닥·금·달러·국고채10년·3년·단기채·인버스·레버리지 그대로 들고 있기
  추세      지수가 120일선 위면 지수, 아래면 인버스(T:ls) / 단기채(T:cash)
            200일선 아래면 방어 자산(달러·금·장기채·단기채 중 60일 수익 1등)으로 (T:def)
  모멘텀    매달, 지수·미국 S&P·나스닥·금·달러·장기채 중 120일 수익 1등(DM:1)·상위 2개(DM:2), 단기채보다 못하면 단기채
  위험균형  매달, 지수·S&P·금·장기채·달러를 변동성 반비례 비중으로 (RP)
  변동성    지수 비중 = 15% ÷ 최근 20일 변동성 (최대 100%), 나머지 단기채 (VT)
  급락반등  1년 고점 대비 -15% 아래에서 5일 -6% 넘게 빠지면 레버리지 10일 (CR:lev), 아니면 단기채
  하락추세  60일선이 200일선 아래 + 지수가 60일선 아래면 인버스, 아니면 단기채 (BR:inv)
  달러피신  약세·급락이면 달러, 아니면 지수 (BR:usd)

비용: 수수료+제비용 0.0186%(한쪽) + 미끄러짐 = 호가 반 칸(가격 2,000원 미만 1원·이상 5원) + 0.005%.
세금: ETF 는 증권거래세 없음. 기타 ETF(채권·달러·금·미국) 는 이익 난 매도마다 이익의 15.4% (실제 과세표준보다 보수적).
      국내 주식형·주가지수 파생형(레버리지·인버스)은 매매차익 과세가 사실상 없어 0 으로 둔다.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_etf import ASSETS  # noqa: E402

ETF = "data/etf_ohlcv.csv"
CODES = list(ASSETS)                     # 0 = 069500 KODEX 200, 1 = 114800 KODEX 인버스 (예전 2자산 모델과 같은 자리)
KEYS = {"IDX": "069500", "INV": "114800", "LEV": "122630", "INV2": "252670", "KQ": "229200", "CASH": "153130",
        "B3": "114260", "B10": "148070", "USD": "261240", "GOLD": "132030", "SPX": "143850", "NDX": "133690"}
IX = {k: CODES.index(c) for k, c in KEYS.items()}
A = len(CODES)
NAME = {c: ASSETS[c][0] for c in CODES}
TAXR = np.array([0.154 if ASSETS[c][1] == "other" else 0.0 for c in CODES])
COMM = 0.000186
LIMIT = 0.305
REGIMES = ["강세", "횡보", "약세", "급락"]
REG_DESC = {"강세": "200일선 위 + 60일 수익 플러스", "횡보": "200일선 위지만 60일 수익 마이너스",
            "약세": "200일선 아래", "급락": "1년 고점 대비 -15% 이하"}


def slip(px, stress=1.0):
    """호가 반 칸 + 0.005% (가격이 없으면 0.01%)."""
    with np.errstate(all="ignore"):
        t = np.where(px < 2000, 1.0, 5.0)
        s = 0.5 * t / px + 0.00005
    return np.where(np.isfinite(s), s, 0.0001) * stress


def costs(px, stress=1.0):
    s = slip(px, stress)
    return COMM + s, COMM + s


# ══════════════════════════════════════════════════════════════
# 가격
# ══════════════════════════════════════════════════════════════
def _guard(o_, c_):
    """하루 ±30% 넘는 변화(분할·병합)는 그날 수익 0으로 보고 과거를 소급 보정."""
    fac = np.ones(len(c_))
    ix = np.where(~np.isnan(c_))[0]
    for p, q in zip(ix[:-1], ix[1:]):
        r = c_[q] / c_[p]
        if r > 1 + LIMIT or r < 1 - LIMIT:
            fac[:q] *= r
    return o_ * fac, c_ * fac


def load_prices(dates=None):
    """(dates, Op, Cp) — T×A. dates 를 안 주면 KODEX 200 거래일 전체. 상장 전·자료 없는 날은 NaN."""
    e = pd.read_csv(ETF, dtype={"종목코드": str})
    e["종목코드"] = e["종목코드"].str.zfill(6)
    e["날짜"] = pd.to_datetime(e["날짜"])
    if dates is None:
        dates = pd.DatetimeIndex(sorted(e.loc[e["종목코드"] == CODES[0], "날짜"].unique()))
    Op = np.full((len(dates), A), np.nan)
    Cp = np.full((len(dates), A), np.nan)
    for j, c in enumerate(CODES):
        x = e[e["종목코드"] == c].drop_duplicates("날짜", keep="last").set_index("날짜").reindex(dates)
        if x["종가"].notna().sum() == 0:
            continue
        o_, c_ = x["시가"].to_numpy(float, copy=True), x["종가"].to_numpy(float, copy=True)
        o_[~(o_ > 0)] = np.nan
        c_[~(c_ > 0)] = np.nan
        o_ = np.where(np.isnan(o_) & ~np.isnan(c_), c_, o_)       # 시가가 빠진 날은 종가로
        Op[:, j], Cp[:, j] = _guard(o_, c_)
    return dates, Op, Cp


def ffill(x):
    return pd.DataFrame(x).ffill().to_numpy(copy=True)


# ══════════════════════════════════════════════════════════════
# 국면·특징 (t행 = t일 장 마감 뒤에 아는 값)
# ══════════════════════════════════════════════════════════════
def month_start(dates, pad=0):
    """t행 = t일이 그달 첫 거래일인가 (매달 다시 맞추는 카드의 날짜). pad=1 이면 내일(다음 평일)까지."""
    d = pd.DatetimeIndex(dates)
    m = np.r_[False, d.month[1:] != d.month[:-1]]
    if pad:
        nxt = d[-1] + pd.offsets.BDay(1)
        m = np.r_[m, nxt.month != d[-1].month]
    return m


def features(Cp, dates=None):
    C = ffill(Cp)
    s = pd.Series(C[:, IX["IDX"]])
    F = {"c": s.to_numpy()}
    F["dates"] = None if dates is None else pd.DatetimeIndex(dates)
    for n in (20, 60, 120, 200):
        F[f"ma{n}"] = s.rolling(n, min_periods=int(n * 0.8)).mean().to_numpy()
    F["hi252"] = s.rolling(252, min_periods=120).max().to_numpy()
    F["dd"] = F["c"] / F["hi252"] - 1
    for n in (5, 20, 60, 120):
        F[f"r{n}"] = (s / s.shift(n) - 1).to_numpy()
    r = s.pct_change()
    F["vol20"] = (r.rolling(20, min_periods=15).std() * np.sqrt(252)).to_numpy()
    df = pd.DataFrame(C)
    F["am60"] = (df / df.shift(60) - 1).to_numpy()
    F["am120"] = (df / df.shift(120) - 1).to_numpy()
    F["avol60"] = (df.pct_change().rolling(60, min_periods=40).std() * np.sqrt(252)).to_numpy()
    F["reg"] = regimes(F)
    return F


def regimes(F):
    c, ma200, dd, r60 = F["c"], F["ma200"], F["dd"], F["r60"]
    reg = np.full(len(c), -1)
    ok = ~np.isnan(ma200) & ~np.isnan(dd) & ~np.isnan(r60)
    reg[ok] = 1
    reg[ok & (c >= ma200) & (r60 > 0)] = 0
    reg[ok & (c < ma200)] = 2
    reg[ok & (dd <= -0.15)] = 3
    return reg


# ══════════════════════════════════════════════════════════════
# 전략 카드 → 목표 비중 (t행 = t일 시가에 맞출 비중, NaN 행 = 그날 거래 안 함)
# ══════════════════════════════════════════════════════════════
HOLD_KEYS = ["IDX", "KQ", "SPX", "NDX", "GOLD", "USD", "B10", "B3", "CASH", "INV", "LEV"]
RULES = {f"H:{k}": f"{NAME[KEYS[k]]} 계속 보유" for k in HOLD_KEYS}
RULES.update({
    "T:ls": "지수가 120일선 위면 KODEX 200, 아래면 KODEX 인버스",
    "T:cash": "지수가 120일선 위면 KODEX 200, 아래면 단기채권",
    "T:def": "지수가 200일선 위면 KODEX 200, 아래면 달러·금·장기채·단기채 중 60일 수익 1등",
    "DM:1": "매달 지수·미국 S&P·나스닥·금·달러·장기채 중 120일 수익 1등 (단기채보다 못하면 단기채)",
    "DM:2": "매달 위 6개 중 120일 수익 상위 2개 반반 (단기채보다 못한 자리는 단기채)",
    "RP": "매달 지수·S&P·금·장기채·달러를 변동성 반비례 비중으로 (위험 균형)",
    "VT": "지수 비중 = 15% ÷ 최근 20일 변동성 (최대 100%), 나머지 단기채",
    "CR:lev": "1년 고점 대비 -15% 아래에서 5일간 -6% 넘게 빠지면 레버리지 10일, 아니면 단기채 (급락 반등)",
    "BR:inv": "60일선이 200일선 아래이고 지수가 60일선 아래면 인버스, 아니면 단기채 (하락 추세)",
    "BR:usd": "약세·급락이면 달러, 아니면 KODEX 200 (달러 피신)",
})
GROUP = {"H": "보유", "T": "추세", "DM": "모멘텀", "RP": "위험균형", "VT": "변동성", "CR": "급락반등", "BR": "하락대비"}


def _lag(x):
    """t행 = t-1일 값 (t일 시가 판단에 쓰는 어제 저녁 정보)."""
    y = np.full_like(x, np.nan, dtype=float)
    y[1:] = x[:-1]
    return y


def _onehot(k):
    w = np.zeros(A)
    w[IX[k]] = 1.0
    return w


def targets(F, Cp, rule, pad=0):
    """규칙 → (T+pad)×A 목표 비중. pad=1 이면 마지막 줄 = 내일(마지막 날 저녁 판단)."""
    T = len(F["c"]) + pad
    avail = ~np.isnan(ffill(Cp))
    if pad:
        avail = np.r_[avail, avail[-1:]]

    def L(x):
        x = np.asarray(x, float)
        return _lag(np.r_[x, x[-1:]] if pad else x)

    def LL(x):                                   # 2차원
        y = np.full((T, x.shape[1]), np.nan)
        y[1:] = x[:T - 1] if not pad else x
        return y

    W = np.full((T, A), np.nan)
    av = np.r_[np.zeros((1, A), bool), avail[:-1]] if not pad else np.r_[np.zeros((1, A), bool), avail[:T - 1]]

    def put(t, w):
        w = np.asarray(w, float).copy()
        miss = (w > 0) & ~av[t]
        if miss.any():                            # 아직 상장 전인 자산 → 단기채(없으면 현금)
            extra = w[miss].sum()
            w[miss] = 0
            if av[t, IX["CASH"]]:
                w[IX["CASH"]] += extra
        W[t] = w

    head, kind = rule.split(":")[0], rule
    c, ma60, ma120, ma200 = L(F["c"]), L(F["ma60"]), L(F["ma120"]), L(F["ma200"])
    if head == "H":
        for t in range(1, T):
            put(t, _onehot(rule.split(":")[1]))
        return W
    if kind in ("T:ls", "T:cash"):
        other = "INV" if kind == "T:ls" else "CASH"
        for t in range(1, T):
            if np.isnan(ma120[t]):
                continue
            put(t, _onehot("IDX" if c[t] > ma120[t] else other))
        return W
    if kind == "T:def":
        am60 = LL(F["am60"])
        cand = [IX[k] for k in ("USD", "GOLD", "B10", "CASH")]
        for t in range(1, T):
            if np.isnan(ma200[t]):
                continue
            if c[t] > ma200[t]:
                put(t, _onehot("IDX"))
            else:
                m = [(am60[t, j] if av[t, j] and np.isfinite(am60[t, j]) else -9) for j in cand]
                w = np.zeros(A)
                w[cand[int(np.argmax(m))]] = 1
                put(t, w)
        return W
    newm = month_start(F["dates"], pad) if F.get("dates") is not None else (np.arange(T) % 21 == 1)
    if kind in ("DM:1", "DM:2"):
        am = LL(F["am120"])
        cand = [IX[k] for k in ("IDX", "SPX", "NDX", "GOLD", "USD", "B10")]
        k = 1 if kind == "DM:1" else 2
        first = True
        for t in range(121, T):
            if not (newm[t] or first) or not np.isfinite(am[t]).any():
                continue
            first = False
            cash_m = am[t, IX["CASH"]] if np.isfinite(am[t, IX["CASH"]]) else 0.0
            sc = [(am[t, j] if av[t, j] and np.isfinite(am[t, j]) else -9) for j in cand]
            order = np.argsort(sc)[::-1][:k]
            w = np.zeros(A)
            for o in order:
                w[cand[o] if sc[o] > cash_m else IX["CASH"]] += 1.0 / k
            put(t, w)
        return W
    if kind == "RP":
        vol = LL(F["avol60"])
        cand = [IX[k] for k in ("IDX", "SPX", "GOLD", "B10", "USD")]
        first = True
        for t in range(61, T):
            if not (newm[t] or first):
                continue
            iv = np.array([1 / vol[t, j] if av[t, j] and np.isfinite(vol[t, j]) and vol[t, j] > 0 else 0 for j in cand])
            if iv.sum() <= 0:
                continue
            first = False
            w = np.zeros(A)
            w[cand] = iv / iv.sum()
            put(t, w)
        return W
    if kind == "VT":
        v = L(F["vol20"])
        last = None
        for t in range(1, T):
            if not np.isfinite(v[t]) or v[t] <= 0:
                continue
            x = min(1.0, 0.15 / v[t])
            if last is not None and abs(x - last) < 0.10:
                continue
            last = x
            w = np.zeros(A)
            w[IX["IDX"]], w[IX["CASH"]] = x, 1 - x
            put(t, w)
        return W
    if kind == "CR:lev":
        dd, r5 = L(F["dd"]), L(F["r5"])
        until = -1
        for t in range(1, T):
            if not np.isfinite(dd[t]):
                continue
            if dd[t] <= -0.15 and np.isfinite(r5[t]) and r5[t] <= -0.06 and t > until:
                until = t + 9
            put(t, _onehot("LEV" if t <= until else "CASH"))
        return W
    if kind == "BR:inv":
        for t in range(1, T):
            if np.isnan(ma200[t]) or np.isnan(ma60[t]):
                continue
            put(t, _onehot("INV" if (ma60[t] < ma200[t] and c[t] < ma60[t]) else "CASH"))
        return W
    if kind == "BR:usd":
        rg = L(F["reg"])
        for t in range(1, T):
            if rg[t] < 0:
                continue
            put(t, _onehot("USD" if rg[t] >= 2 else "IDX"))
        return W
    raise KeyError(rule)


# ══════════════════════════════════════════════════════════════
# 시뮬레이터 (여러 자산 · 시가/종가 거래 · 자산별 비용 · 이익 과세)
# ══════════════════════════════════════════════════════════════
def simulate(Op, Cp, wo, wc=None, t0=1, band=0.05, stress=1.0, record=False, log=None):
    T = Op.shape[0]
    Cf = ffill(Cp)
    units = np.zeros(A)
    avg = np.zeros(A)
    cash = 1.0
    E = np.ones(T)
    traded = 0.0
    nb = 0
    hold = [None] * T if record else None

    def reb(px, tgt, t, session):
        nonlocal cash, traded, nb
        ok = np.isfinite(px)
        pv = np.where(ok, px, Cf[t - 1] if t > 0 else 0)
        eq = cash + np.nansum(units * pv)
        if eq <= 0:
            return
        cb, cs = costs(pv, stress)
        cur = units * pv / eq
        for j in range(A):
            if not ok[j]:
                continue
            d = tgt[j] - cur[j]
            if abs(d) <= band and tgt[j] > 0 and cur[j] > 0:
                continue
            if d < -1e-9 and units[j] > 0:
                v = min(-d * eq, units[j] * px[j])
                q = v / px[j]
                gain = v * (1 - cs[j]) - q * avg[j]
                tax = TAXR[j] * max(gain, 0.0)
                units[j] -= q
                cash += v * (1 - cs[j]) - tax
                traded += v
                if units[j] < 1e-12:
                    units[j], avg[j] = 0.0, 0.0
                if log is not None:
                    log.append({"t": t, "j": j, "side": "sell", "px": float(px[j]), "session": session})
        eq = cash + np.nansum(units * pv)
        cur = units * pv / eq
        for j in range(A):
            if not ok[j]:
                continue
            d = tgt[j] - cur[j]
            if d > 1e-9 and not (abs(d) <= band and cur[j] > 0):
                v = min(d * eq, cash / (1 + cb[j]))
                if v <= 0:
                    continue
                q = v / px[j]
                avg[j] = (avg[j] * units[j] + v * (1 + cb[j])) / (units[j] + q)
                units[j] += q
                cash -= v * (1 + cb[j])
                traded += v
                nb += 1
                if log is not None:
                    log.append({"t": t, "j": j, "side": "buy", "px": float(px[j]), "session": session})

    for t in range(t0, T):
        if not np.isnan(wo[t]).all():
            reb(Op[t], np.nan_to_num(wo[t]), t, "open")
        if wc is not None and not np.isnan(wc[t]).all():
            reb(Cp[t], np.nan_to_num(wc[t]), t, "close")
        E[t] = cash + np.nansum(units * Cf[t])
        if record:
            hold[t] = units * Cf[t] / max(E[t], 1e-12)
    return {"E": E, "turnover": traded, "buys": nb, "hold": hold, "units": units, "cash": cash, "avg": avg}


def desired(wo):
    """그날까지 마지막으로 정한 목표 비중 (갈아탈 때 '지금 이 전략이라면 들고 있을 것')."""
    P = ffill(wo)
    return np.nan_to_num(P)


# ══════════════════════════════════════════════════════════════
# 국면 조건부 고르기 — 같은 국면이었던 과거 날들의 성적으로
# ══════════════════════════════════════════════════════════════
class RegimeScore:
    """R (T×S) 일별 수익률, reg (T) 국면. t일 저녁: 'reg[t] 국면이었던 날(전날 저녁 국면 기준)들의 수익'으로 샤프.
    누적합으로 미리 계산해 매일 O(S)."""

    def __init__(self, R, reg):
        T, S = R.shape
        self.R = R
        valid = ~np.isnan(R)
        X = np.nan_to_num(R)
        prev = np.r_[-1, reg[:-1]]                # d일 수익은 d-1일 저녁 국면에서 들고 있던 결과
        self.cs = {}
        for r in range(4):
            m = (prev == r)[:, None] & valid
            self.cs[r] = (np.cumsum(X * m, 0), np.cumsum(X * X * m, 0), np.cumsum(m, 0))

    def at(self, t, r):
        s1, s2, n = (x[t] for x in self.cs[r])
        with np.errstate(all="ignore"):
            mu = s1 / n
            sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0))
            sh = mu / np.where(sd > 0, sd, np.inf) * np.sqrt(252)
        return np.where(n > 0, sh, np.nan), n


def trailing_sharpe(R, t, L=250):
    x = R[max(1, t - L + 1): t + 1]
    need = max(20, L // 2)
    if len(x) < need:
        return None
    n = (~np.isnan(x)).sum(0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(x, 0), np.nanstd(x, 0)
        out = mu / np.where(sd > 0, sd, np.inf) * np.sqrt(252)
    return np.where(n >= need, out, np.nan)


def choose_regime(R, reg, start, every=5, L=250, mode="blend", shrink=120, min_n=40, on_change=True, scorer=None):
    """반환: 날짜별 고른 번호. mode: 'uncond'(최근 L일 샤프) · 'cond'(같은 국면 샤프) · 'blend'(표본 수로 섞음).
    다시 고르는 때: every 거래일마다 + (on_change) 국면이 바뀐 날 저녁."""
    T = R.shape[0]
    rs = scorer or (RegimeScore(R, reg) if mode != "uncond" else None)
    ch = [None] * T
    cur = None
    for t in range(start, T):
        changed = on_change and t > 0 and reg[t] != reg[t - 1]
        if cur is None or (t - start) % every == 0 or changed:
            u = trailing_sharpe(R, t, L)
            if u is None:
                continue
            if mode == "uncond" or reg[t] < 0:
                s = u
            else:
                c, n = rs.at(t, reg[t])
                if mode == "cond":
                    s = np.where(n >= min_n, c, u)
                else:
                    w = n / (n + shrink)
                    s = np.where(np.isfinite(c), w * c + (1 - w) * np.nan_to_num(u, nan=-9), u)
            s = np.nan_to_num(s, nan=-np.inf)
            cur = int(np.argmax(s))
        ch[t] = cur
    return ch


def regime_now(F, t=None):
    t = len(F["c"]) - 1 if t is None else t
    r = int(F["reg"][t])
    return {"code": r, "name": REGIMES[r] if r >= 0 else "판정 불가", "desc": REG_DESC.get(REGIMES[r]) if r >= 0 else "",
            "dd": float(F["dd"][t]), "trend200": float(F["c"][t] / F["ma200"][t] - 1) if np.isfinite(F["ma200"][t]) else None,
            "r60": float(F["r60"][t]) if np.isfinite(F["r60"][t]) else None,
            "vol20": float(F["vol20"][t]) if np.isfinite(F["vol20"][t]) else None}


def choose_mix(Rx, reg, start, every=21, L=250, mode="topk", k=5, shrink=250, on_change=True, scorer=None, cash=None, resched=None):
    """여러 카드를 섞어 들기 (한 장만 고르면 운에 크게 흔들린다 — 연구에서 확인).
    반환: 날짜별 [(카드 번호, 비중), …]. 비중 합 = 1 (모자라면 cash 카드로).
      all   점수 있는 카드 전부 같은 비중 (고르지 않음)
      rp    전부, 최근 120일 변동성 반비례
      topk  점수(같은 국면 + 최근 L일, 표본 수로 섞음) 상위 k장 같은 비중, 점수 0 이하 자리는 cash"""
    T, S = Rx.shape
    rs = scorer or (RegimeScore(Rx, reg) if mode == "topk" else None)
    ch = [None] * T
    cur = None
    for t in range(start, T):
        changed = on_change and t > 0 and reg[t] != reg[t - 1]
        due = resched[t + 1] if resched is not None and t + 1 < len(resched) else (t - start) % every == 0
        if cur is None or due or changed:
            u = trailing_sharpe(Rx, t, L)
            if u is None:
                continue
            ok = [i for i in range(S) if np.isfinite(u[i]) and i != cash]
            if not ok:
                continue
            if mode == "all":
                cur = [(i, 1.0 / len(ok)) for i in ok]
            elif mode == "rp":
                x = Rx[max(1, t - 119): t + 1]
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    v = np.nanstd(x, 0)
                iv = np.array([1 / v[i] if np.isfinite(v[i]) and v[i] > 1e-6 else 0.0 for i in ok])
                if iv.sum() <= 0:
                    continue
                cur = [(i, float(w)) for i, w in zip(ok, iv / iv.sum())]
            else:
                if reg[t] >= 0:
                    c, n = rs.at(t, reg[t])
                    w = n / (n + shrink)
                    s = np.where(np.isfinite(c), w * c + (1 - w) * np.nan_to_num(u, nan=-9), u)
                else:
                    s = u
                s = np.nan_to_num(s, nan=-np.inf)
                if cash is not None:
                    s[cash] = -np.inf
                order = [int(i) for i in np.argsort(-s)[:k]]
                cur = [(i, 1.0 / k) for i in order if s[i] > 0]
                rest = 1.0 - sum(w for _, w in cur)
                if rest > 1e-9 and cash is not None:
                    cur.append((cash, rest))
        ch[t] = cur
    return ch


def mix_weights(Ps, Ws, ch, t0, T):
    """[(카드, 비중)] 고르기 → t일 시가 목표 자산 비중. 구성·비중이 바뀌었거나 구성 카드가 거래하는 날만 줄을 채운다."""
    wo = np.full((T, A), np.nan)
    prev, sw = None, 0
    for t in range(t0, T):
        c = ch[t - 1]
        if c is None:
            continue
        key = tuple((i, round(w, 4)) for i, w in c)
        trade = key != prev or any(not np.isnan(Ws[i][t]).all() for i, _ in c)
        if trade:
            wo[t] = sum(w * Ps[i][t] for i, w in c)
            if key != prev:
                sw += prev is not None
            prev = key
    return wo, sw


# ══════════════════════════════════════════════════════════════
# 매일 모의투자용 — 전천후 혼합(카드 전부, 변동성 반비례, 매달 1일 다시 맞춤)의 오늘 상태와 내일 목표
# ══════════════════════════════════════════════════════════════
LIVE_MIX = {"mode": "rp", "on_change": False, "L": 250}
MIX_START = "2016-01-04"


def card_returns(dates, Op, Cp, F, rules=None, stress=1.0):
    """카드별 (목표 비중 W, 기록, 일별 수익 R). 보유 카드는 자산 상장 전 날들이 NaN."""
    rules = rules or list(RULES)
    Ws, recs = [], []
    for k in rules:
        W = targets(F, Cp, k)
        Ws.append(W)
        recs.append(simulate(Op, Cp, W, stress=stress))
    R = np.c_[[np.r_[0.0, x["E"][1:] / x["E"][:-1] - 1] for x in recs]].T
    for i, k in enumerate(rules):
        if k.startswith("H:"):
            j = IX[k.split(":")[1]]
            first = np.where(~np.isnan(Cp[:, j]))[0]
            R[: (first[0] + 2 if len(first) else len(dates)), i] = np.nan
    return rules, Ws, recs, R


def excess_over_cash(R, rules):
    rc = np.nan_to_num(R[:, rules.index("H:CASH")])
    return R - rc[:, None]


def live_mix(upto=None, stress=1.0):
    """반환 dict: 날짜·가격·특징, 카드, 혼합 고르기(날짜별), 혼합 자산곡선, 내일 목표 비중, 오늘 국면."""
    dates, Op, Cp = load_prices()
    if upto is not None:
        keep = dates <= pd.Timestamp(upto)
        dates, Op, Cp = dates[keep], Op[keep], Cp[keep]
    F = features(Cp, dates)
    rules, Ws, recs, R = card_returns(dates, Op, Cp, F, stress=stress)
    Rx = excess_over_cash(R, rules)
    cash = rules.index("H:CASH")
    s = int(np.searchsorted(dates, pd.Timestamp(MIX_START)))
    resched = month_start(dates, pad=1)
    ch = choose_mix(Rx, F["reg"], s, cash=cash, resched=resched, **LIVE_MIX)
    P = [desired(W) for W in Ws]
    wo, sw = mix_weights(P, Ws, ch, s, len(dates))
    res = simulate(Op, Cp, wo, t0=s, stress=stress)
    # 내일 목표 = 오늘 저녁 고른 혼합 × 각 카드의 내일 목표
    Wn = [targets(F, Cp, k, pad=1) for k in rules]
    Pn = [desired(W)[-1] for W in Wn]
    mix = ch[-1] or []
    tgt = np.zeros(A)
    for i, w in mix:
        tgt += w * Pn[i]
    trade_tomorrow = bool(mix) and (any(not np.isnan(Wn[i][-1]).all() for i, _ in mix) or
                                    tuple((i, round(w, 4)) for i, w in mix) != tuple((i, round(w, 4)) for i, w in (ch[-2] or [])))
    return {"dates": dates, "Op": Op, "Cp": Cp, "F": F, "rules": rules, "R": R, "Ws": Ws, "ch": ch, "start": s,
            "E": res["E"], "switches": sw, "target": tgt, "mix": mix, "trade_tomorrow": trade_tomorrow,
            "card_now": {rules[i]: Pn[i] for i in range(len(rules))}, "regime": regime_now(F)}
