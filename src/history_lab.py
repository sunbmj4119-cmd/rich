"""
긴 역사 검증 — '설계에 한 번도 안 쓴 옛 자료'로 지금 규칙을 그대로 돌려 본다 (진짜 표본 외 시험).

  왜: 규칙은 2014~2026(종목 모델은 2019~) 자료로 만들고 골랐다. 그 안의 '뒤 기간'도 이미 본 자료라 반쯤은 시험지를 본 시험이다.
      그 전 자료(1997~2013: IMF·닷컴·2008 금융위기·유럽 재정위기)는 한 번도 안 봤다 → 규칙을 바꾸지 않고 돌리면 진짜 시험.
  무엇을 (모두 지금 쓰는 시뮬레이터·비용·세금 그대로)
    1) 국내 ETF 규칙 10개(전날 내렸으면 보유 등)를 코스피 1997~ 로 — 1997~2013(표본 외) vs 2014~(설계 기간)
    2) 전천후 혼합(같은 카드 21장·같은 섞기)을 대용 자산으로 2005~
    3) 종목 모델 전부(국내 2000~ · 미국 2000~): 지금 '좋다'고 본 모델이 옛날에도 좋았나 (기간 사이 순위 상관)
    4) 언제 시작했어도: 모든 날을 시작일로 1년·3년 들고 갔을 때 손실 확률·중앙값·최악
    5) 위기별 성적: 1997 외환위기 ~ 2026 급락까지 9개 구간
    6) 업종: 국내 업종 ETF·미국 업종 SPDR 회전 (Holm 보정) · 국내 단기 모델이 어느 업종에서 벌었나
  대용 자산(2014년 전 국내 ETF 가 없는 자리 — 정확한 재현이 아니라 '같은 방법'의 시험):
    지수 ← 코스피 지수 + 배당 연 1.5% · 인버스/레버리지 ← 지수 일수익 −1/2/−2배 · 코스닥 ← 코스닥 지수 · 단기채 ← 연 3% ·
    국고채3/10년 ← 미국 IEF/TLT(환헤지 가정) · 달러 ← 원/달러 · 금 ← GLD · S&P(H) ← SPY · 나스닥 ← QQQ×환율
    미국 자산은 '한국 날짜 전날까지 끝난 미국 장' 값만 쓴다(미래 정보 없음). 호가 비용은 가격 수준과 무관하게 한쪽 0.03% 로.
  한계: 종목은 '지금 남아 있는 대형주'만 → 생존편향(옛날 성적이 실제보다 좋게 나옴). 세율·수수료는 지금 기준.

  python src/history_lab.py → data/history/lab.json · docs/history.json
"""
import contextlib
import json
import os
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allweather as AW  # noqa: E402
import science_lab as S  # noqa: E402
import short_lab as L  # noqa: E402

warnings.filterwarnings("ignore")
HIST = "data/history/hist.csv"
US_STOCKS = "data/history/us_stocks.csv"
KR_STOCKS = "data/history/kr_stocks.csv"
OUT = "data/history/lab.json"
UI = "docs/history.json"
GATE = "data/history/gate.json"   # 긴 역사 관문: 옛 시대(국내 2001~2018)에 샤프 < 0 인 모델 → 단기 몫 후보에서 뺀다 (short_lab.long_gate)
OOS_END = "2013-12-31"            # 이날까지 = 설계에 안 쓴 자료 (ETF·전천후)
KR_DESIGN = "2019-01-01"          # 국내 종목 모델을 만든 자료의 시작
US_DESIGN = "2015-01-01"          # 미국 종목 모델 연구 자료의 시작
DIV = 0.015                       # 코스피 지수(가격지수)에 더할 배당 (연)
CASH = 0.03                       # 단기채 대용 (연)
# 대용 자산 호가 비용 한쪽: 실제 ETF 의 '흔한 가격대'에서 호가 반 칸 + 0.005% (지수류 3만원대·인버스 5천원대·곱버스 3천원대·코스닥 1만원대)
SLIP = {"069500": 0.00013, "114800": 0.00055, "122630": 0.00015, "252670": 0.00088, "229200": 0.0003, "153130": 0.0001,
        "114260": 0.0001, "148070": 0.0001, "261240": 0.0003, "132030": 0.0002, "143850": 0.00015, "133690": 0.00015}
SCALE = 10000.0
CRISES = [("1997 외환위기", "1997-07-01", "1998-06-30"), ("닷컴 붕괴", "2000-03-01", "2002-10-31"),
          ("2008 금융위기", "2007-10-01", "2009-03-31"), ("유럽 재정위기", "2011-07-01", "2011-10-31"),
          ("중국 쇼크", "2015-06-01", "2016-02-29"), ("2018 미중 무역분쟁", "2018-10-01", "2018-12-31"),
          ("코로나", "2020-02-01", "2020-03-31"), ("2022 금리 급등", "2022-01-01", "2022-10-31"), ("2026 급락", "2026-06-22", "2026-10-31")]


def load_hist():
    h = pd.read_csv(HIST)
    h["날짜"] = pd.to_datetime(h["날짜"])
    return h


def series(h, sym, col="종가"):
    x = h[h["종목코드"] == sym].drop_duplicates("날짜").set_index("날짜")[col].astype(float).sort_index()
    return x[x > 0]


def stats_r(r, dates, a=None, b=None):
    return L.stats(np.cumprod(1 + np.nan_to_num(np.asarray(r, float))), dates, a, b)


def spearman(a, b):
    """순위 상관 (scipy 없이)."""
    x = pd.concat([pd.Series(a, dtype=float), pd.Series(b, dtype=float)], axis=1).dropna()
    if len(x) < 3:
        return None
    return round(float(x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank())), 3)


def cut(dates, day):
    return int(np.searchsorted(dates, pd.Timestamp(day)))


@contextlib.contextmanager
def flat_slip(x=None):
    """대용 자산은 가격 수준이 실제 ETF 와 달라 호가 칸 비용이 왜곡된다 → 자산별 고정값으로."""
    vec = np.array([(x or SLIP)[c] for c in AW.CODES])
    old = AW.slip

    def fixed(px, stress=1.0):
        sh = np.shape(px)
        if len(sh) == 1 and sh[0] <= len(vec):
            return vec[:sh[0]] * stress                    # follow() 는 앞 두 자산(지수·인버스)만 넘긴다
        return np.full(sh, float(np.median(vec))) * stress
    AW.slip = fixed
    try:
        yield
    finally:
        AW.slip = old


# ══════════════════════════════════════════════════════════════
# 대용 자산 (12개, 코스피 거래일 축 1996-12~)
# ══════════════════════════════════════════════════════════════
def before(x, dates):
    """한국 날짜 t 마다 't 보다 앞선 날'의 마지막 값 (미국 장·환율 → 다음 한국 거래일에 반영, 미래 정보 없음)."""
    x = x.dropna()
    a = pd.DataFrame({"d": dates})
    b = pd.DataFrame({"d": x.index, "v": x.to_numpy(float)})
    m = pd.merge_asof(a, b, on="d", allow_exact_matches=False)
    return pd.Series(m["v"].to_numpy(float), index=dates)


def proxy_prices(h):
    ks_c, ks_o = series(h, "^KS11"), series(h, "^KS11", "시가")
    dates = ks_c.index
    ks_o = ks_o.reindex(dates)
    rk = ks_c.pct_change().fillna(0).to_numpy()
    gap = (ks_o / ks_c.shift(1) - 1).to_numpy()
    gap = np.where(np.isfinite(gap) & (np.abs(gap) < 0.2), gap, 0.0)
    d = DIV / 252

    def lever(k, extra=0.0):
        c = SCALE * np.cumprod(np.maximum(1 + k * rk + extra, 0.01))
        o = np.r_[np.nan, c[:-1]] * (1 + k * gap + extra)
        return o, c
    P = {"069500": lever(1, d), "114800": lever(-1), "122630": lever(2), "252670": lever(-2)}
    kq_c, kq_o = series(h, "^KQ11").reindex(dates), series(h, "^KQ11", "시가").reindex(dates)
    f = SCALE / kq_c.dropna().iloc[0]
    P["229200"] = ((kq_o * f).to_numpy(), (kq_c * f).to_numpy())
    cash = SCALE * np.cumprod(np.r_[1.0, np.full(len(dates) - 1, (1 + CASH) ** (1 / 252))])
    P["153130"] = (np.r_[np.nan, cash[:-1]], cash)
    fx = before(series(h, "KRW=X"), dates)

    def us(sym, fxadj=False):
        x = before(series(h, sym), dates)
        if fxadj:
            x = x * fx
        x = x * SCALE / x.dropna().iloc[0]
        return x.to_numpy(), x.to_numpy()             # 한국 장 동안 미국 장은 닫혀 있다 → 시가 = 종가 (둘 다 전날 밤 미국 종가)
    P["114260"], P["148070"], P["132030"], P["143850"] = us("IEF"), us("TLT"), us("GLD"), us("SPY")
    P["133690"] = us("QQQ", fxadj=True)
    fxs = (fx * SCALE / fx.dropna().iloc[0]).to_numpy()
    P["261240"] = (fxs, fxs)
    Op = np.c_[[P[c][0] for c in AW.CODES]].T.astype(float)
    Cp = np.c_[[P[c][1] for c in AW.CODES]].T.astype(float)
    Op[~np.isfinite(Op)] = np.nan
    Cp[~np.isfinite(Cp)] = np.nan
    Op[np.isnan(Cp)] = np.nan
    return dates, Op, Cp


def market_D(dates, Op, Cp):
    """short_lab.etf_targets 가 쓰는 모양 (지수 신호는 대용 지수에서)."""
    c, o = Cp[:, 0], Op[:, 0]
    cc = np.r_[0, c[1:] / c[:-1] - 1]
    oc = np.nan_to_num(c / o - 1)
    s = pd.Series(np.cumprod(1 + cc))
    M = {"m_rev1": -cc, "m_rev_id": -oc, "m_for1": np.full(len(c), np.nan),
         "m_trend120": (s / s.rolling(120, min_periods=80).mean() - 1).to_numpy()}
    return types.SimpleNamespace(T=len(dates), M=M, Op=Op, Cp=Cp, dates=dates)


# ══════════════════════════════════════════════════════════════
# 1) 국내 ETF 규칙 1997~
# ══════════════════════════════════════════════════════════════
def etf_rules_long(dates, Op, Cp):
    D = market_D(dates, Op, Cp)
    e = cut(dates, OOS_END) + 1
    with flat_slip():
        runs = {m["rule"]: L.run_etf(D, m)["E"] for m in L.etf_models() if m["rule"] != "overnight_for"}
    hold = runs["hold"]
    rh = L.rets(hold)
    out = {}
    for k, E in runs.items():
        r = L.rets(E)
        out[k] = {"name": L.ETF_RULES[k], "oos": L.stats(E[:e], dates[:e]), "design": L.stats(E[e - 1:], dates[e - 1:]),
                  "p_oos": L.boot_p(r[130:e], rh[130:e]), "p_design": L.boot_p(r[e:], rh[e:]),
                  "years": L.yearly(E, dates)}
    keys = [k for k in out if k != "hold"]
    for k, a in zip(keys, S.holm([out[k]["p_oos"] for k in keys])):
        out[k]["p_oos_holm"] = a
    sh_o = pd.Series({k: (out[k]["oos"] or {}).get("sharpe") for k in out})
    sh_d = pd.Series({k: (out[k]["design"] or {}).get("sharpe") for k in out})
    return out, {k: pd.Series(L.rets(E), index=dates) for k, E in runs.items()}, spearman(sh_o, sh_d)


# ══════════════════════════════════════════════════════════════
# 2) 전천후 혼합 2005~
# ══════════════════════════════════════════════════════════════
def aw_long(dates, Op, Cp, start="2005-01-03"):
    with flat_slip():
        m = AW.live_mix(prices=(dates, Op, Cp), start=start)
    s = m["start"]
    E = m["E"] / m["E"][s]
    rE = L.rets(E)
    ri = np.r_[0, Cp[1:, 0] / Cp[:-1, 0] - 1]
    e = cut(dates, OOS_END) + 1
    nmix = [len(x) for x in m["ch"][s:] if x]
    out = {"what": "전천후 혼합 (같은 카드 21장 · 같은 변동성 반비례 섞기 · 대용 자산)", "from": str(dates[s].date()),
           "oos": L.stats(E[s:e], dates[s:e]), "design": L.stats(E[e - 1:], dates[e - 1:]),
           "idx_oos": stats_r(ri[s:e], dates[s:e]), "idx_design": stats_r(ri[e:], dates[e:]),
           "p_oos": L.boot_p(rE[s:e], ri[s:e]), "years": L.yearly(E[s:], dates[s:]),
           "cards_avg": round(float(np.mean(nmix)), 1) if nmix else 0, "switches": m["switches"]}
    return out, pd.Series(np.where(np.arange(len(rE)) >= s, rE, np.nan), index=dates)


# ══════════════════════════════════════════════════════════════
# 3) 종목 모델 전부 — 옛날에도 좋았나
# ══════════════════════════════════════════════════════════════
def stock_panel(path, bench, market):
    import us_lab as UL
    P = UL.load_panel(min_date="2000-01-01", path=path, bench=bench, market=market)
    if market == "kr":                                     # 야후 국내 자료의 이상한 날(가격제한폭 밖) = 그날 수익 무시
        r = P["C"][1:] / P["C"][:-1] - 1
        lim = np.where(P["dates"][1:] < pd.Timestamp("2015-06-15"), 0.16, 0.31)[:, None]
        bad = np.abs(r) > lim
        P["breaks"][1:] |= np.nan_to_num(bad, nan=False).astype(bool)
        P["n_bad"] = int(np.nansum(bad))
    return P


def models_long(path, bench, market, design_from, periods, meta_from="2002-01-02"):
    """모든 고정 종목 모델(수급 신호 제외, 국내는 ETF 규칙 포함)을 2000~ 로 돌리고,
    지금 단기 몫이 쓰는 고르기(매주 · 최근 250일 샤프 1등)도 그대로 돌린다.
    반환: 기간별 분포 · 설계 기간과의 순위 상관 · 지금 1등 모델들의 옛 성적 · 고르기 성적."""
    import us_lab as UL
    P = stock_panel(path, bench, market)
    D = UL.USData(P) if market == "us" else L.Data(P)
    dates = D.dates
    ms = UL.models()
    if market == "kr":
        ms = ms + [m for m in L.etf_models() if m["rule"] != "overnight_for"]
    with flat_slip():
        recs = [L.run(D, m, record=True) for m in ms]
    E = {m["id"]: r["E"] for m, r in zip(ms, recs)}
    b = P["bench"][bench[0]]
    rb = np.r_[0, b[1:] / b[:-1] - 1]
    per = dict(periods)
    per["설계 기간"] = (design_from, None)
    tab = {}
    for name, (a, z) in per.items():
        st = {k: L.stats(v, dates, a, z) or {} for k, v in E.items()}
        tab[name] = (pd.Series({k: x.get("sharpe") for k, x in st.items()}, dtype=float),
                     pd.Series({k: x.get("cagr") for k, x in st.items()}, dtype=float))
    holds = {}
    for m in ms:
        holds.setdefault(m["hold"] if m["kind"] == "stock" else "ETF", []).append(m["id"])
    by = {}
    for name, (sh, cg) in tab.items():
        a, z = per[name]
        i, j = cut(dates, a), (cut(dates, z) + 1 if z else None)
        bs = stats_r(rb[i:j], dates[i:j])
        by[name] = {"n": int(sh.notna().sum()), "positive": int((cg > 0).sum()), "sharpe_med": round(float(sh.median()), 3),
                    "beat_bench": int((sh > (bs or {}).get("sharpe", 0)).sum()), "bench": bs,
                    "by_hold": {h: round(float(sh[ids].median()), 3) for h, ids in holds.items()}}
    d_sh = tab["설계 기간"][0]
    corr = {name: spearman(sh, d_sh) for name, (sh, _) in tab.items() if name != "설계 기간"}
    desc = {m["id"]: L.describe(m) for m in ms}
    best = [{"id": k, "desc": desc[k], "design": round(float(d_sh[k]), 3),
             **{name: (None if pd.isna(tab[name][0][k]) else round(float(tab[name][0][k]), 3)) for name in periods}}
            for k in d_sh.sort_values(ascending=False).index[:10]]
    q = pd.qcut(d_sh.rank(method="first"), 5, labels=False)
    quint = {name: [round(float(tab[name][0][q == i].median()), 3) for i in range(5)] for name in periods}
    # 지금 단기 몫의 고르기 규칙 그대로
    R = np.c_[[L.rets(r["E"]) for r in recs]].T
    st = cut(dates, meta_from)
    with flat_slip():
        Em, sw, ch = L.apply_rule(D, ms, recs, R, L.LIVE_RULE, st, gate=set())     # 관문 없이 (관문 자체를 여기서 만든다)
    Em = Em / Em[st]
    rm = L.rets(Em)
    meta = {"name": "매주 · 최근 250일 샤프 1등 (지금 단기 몫 규칙)", "from": str(dates[st].date()), "switches": sw, "periods": {}}
    for name, (a, z) in {**per, "전체": (meta_from, None)}.items():
        a = max(pd.Timestamp(a), dates[st])
        i, j = cut(dates, a), (cut(dates, z) + 1 if z else None)
        picks = pd.Series([ms[c]["id"] for c in ch[i:j] if isinstance(c, int)]).value_counts()
        meta["periods"][name] = {"stats": L.stats(Em[i:j], dates[i:j]), "bench": stats_r(rb[i:j], dates[i:j]),
                                 "p": L.boot_p(rm[i:j], rb[i:j]),
                                 "picks": {k: round(float(v / max(picks.sum(), 1)), 3) for k, v in picks.head(5).items()}}
    # 옛 자료로만 고르고 → 새 자료로 채점 (사전 등록과 같은 방식)
    old_a, old_z = min(v[0] for v in periods.values()), max(v[1] for v in periods.values())
    o_sh = pd.Series({k: (L.stats(v, dates, old_a, old_z) or {}).get("sharpe") for k, v in E.items()}, dtype=float)
    top_old = list(o_sh.sort_values(ascending=False).index[:10])
    pos = [i for i, m in enumerate(ms) if o_sh.get(m["id"], np.nan) > 0]
    ds = cut(dates, design_from)
    with flat_slip():
        Ef, swf, chf = L.apply_rule(D, [ms[i] for i in pos], [recs[i] for i in pos], R[:, pos], L.LIVE_RULE, ds, gate=set())
    Eu = Em[ds:] / Em[ds]
    Ef = Ef[ds:] / Ef[ds]
    picks = pd.Series([ms[pos[c]]["id"] for c in chf[ds:] if isinstance(c, int)]).value_counts()
    old_new = {"old": [old_a, old_z], "new_from": design_from,
               "top10_old": [{"id": k, "desc": desc[k], "old": round(float(o_sh[k]), 3),
                              "new": None if pd.isna(d_sh[k]) else round(float(d_sh[k]), 3)} for k in top_old],
               "top10_new_median": round(float(d_sh[top_old].median()), 3), "all_new_median": round(float(d_sh.median()), 3),
               "pos_old": len(pos), "pos_old_new_median": round(float(d_sh.iloc[[d_sh.index.get_loc(ms[i]["id"]) for i in pos]].median()), 3),
               "neg_old_new_median": round(float(d_sh.drop([ms[i]["id"] for i in pos]).median()), 3),
               "meta_all": L.stats(Eu, dates[ds:]), "meta_filtered": L.stats(Ef, dates[ds:]), "meta_filtered_switches": swf,
               "meta_filtered_picks": {k: round(float(v / max(picks.sum(), 1)), 3) for k, v in picks.head(5).items()},
               "bench_new": stats_r(rb[ds:], dates[ds:])}
    alive = int(np.isfinite(P["C"][cut(dates, "2001-01-02")]).sum())
    old_new["old_sharpe"] = {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in o_sh.items()}
    return ({"old_new": old_new, "n_models": len(ms), "n_stocks": len(P["codes"]), "alive_2001": alive, "n_bad_days": P.get("n_bad", 0),
             "periods": {k: list(v) for k, v in per.items()}, "by_period": by, "rank_corr": corr, "design_top": best,
             "quintiles": quint, "meta": meta},
            {k: pd.Series(L.rets(v), index=dates) for k, v in E.items()}, pd.Series(rb, index=dates),
            pd.Series(np.where(np.arange(len(rm)) > st, rm, np.nan), index=dates))


KRX = {k: os.path.join("data/history", f"krx_{k}.csv.gz") for k in ("universe", "ohlcv", "flows")}


def krx_panel():
    """KRX 긴 역사(collect_krx_history.py)로 '그때 실제 대형주' 패널: 해마다 시총 상위 100 = 유니버스(상장폐지 포함),
    외국인·기관·개인 순매수, 시가총액 ≈ 연말 시총 × 종가 비율. 파일이 없으면 None."""
    if not all(os.path.exists(v) for v in KRX.values()):
        return None
    u = pd.read_csv(KRX["universe"], dtype={"종목코드": str})
    o = pd.read_csv(KRX["ohlcv"], dtype={"종목코드": str})
    f = pd.read_csv(KRX["flows"], dtype={"종목코드": str})
    o = o[o["종가"] > 0]
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(o["날짜"]).unique()))
    codes = sorted(o["종목코드"].unique())
    P = {"dates": dates, "codes": codes, "market": "kr"}

    def piv(df, col):
        x = df.assign(날짜=pd.to_datetime(df["날짜"])).pivot_table(index="날짜", columns="종목코드", values=col, aggfunc="last")
        return x.reindex(index=dates, columns=codes).to_numpy(dtype=float, copy=True)
    for k, col in [("O", "시가"), ("H", "고가"), ("L", "저가"), ("C", "종가"), ("V", "거래량")]:
        P[k] = piv(o, col)
    for k in ("O", "H", "L", "C"):
        P[k][~(P[k] > 0)] = np.nan
    P["VAL"] = piv(o, "거래대금") if "거래대금" in o.columns else P["C"] * P["V"]   # 실제 거래대금 (수정주가 × 거래량은 분할 전후로 어긋남)
    # 시가총액: 해마다 기준일 시총 × (종가 / 기준일 종가) — 그 해 안에선 주식 수가 같다고 근사
    MC = np.full(P["C"].shape, np.nan)
    U = np.zeros(P["C"].shape, bool)
    ci = {c: j for j, c in enumerate(codes)}
    u["기준일"] = pd.to_datetime(u["기준일"])
    for d0, g in u.groupby("기준일"):
        t0 = int(np.searchsorted(dates, d0, side="right")) - 1
        nxt = u["기준일"][u["기준일"] > d0]
        t1 = int(np.searchsorted(dates, nxt.min(), side="right")) if len(nxt) else len(dates)
        if t0 < 0:
            continue
        for _, r in g.iterrows():
            j = ci.get(r["종목코드"])
            if j is None or not np.isfinite(P["C"][t0, j]):
                continue
            MC[t0 + 1:t1, j] = r["시가총액"] * P["C"][t0 + 1:t1, j] / P["C"][t0, j]
            if r["순위"] <= 100:
                U[t0 + 1:t1, j] = True                       # 기준일 다음 날부터 다음 기준일까지 (그날 이후 정보 없음)
    P["MC"], P["U"] = MC, U & ~np.isnan(MC)
    r = P["C"][1:] / P["C"][:-1] - 1
    lim = np.where(dates[1:] < pd.Timestamp("2015-06-15"), 0.16, 0.31)[:, None]
    P["breaks"] = np.r_[np.zeros((1, len(codes)), bool), np.nan_to_num(np.abs(r) > lim, nan=False).astype(bool)]
    for k, col in (("FOR", "외국인순매수"), ("INS", "기관순매수"), ("IND", "개인순매수")):
        P[k] = piv(f, col)
    P["Cf"] = pd.DataFrame(P["C"]).ffill().to_numpy(copy=True)
    ks = series(load_hist(), "^KS11").reindex(dates).ffill().to_numpy(float)
    P["bench"] = {"^KS11": ks}
    return P


def pit_report(meta_from="2002-01-02", end="2018-12-31"):
    """상장폐지 포함·수급 포함 긴 역사로: 모든 모델(수급 모델 포함)의 옛 시대 성적 · 지금 단기 몫 고르기 규칙(수급 포함/제외)."""
    P = krx_panel()
    if P is None:
        return None
    D = L.Data(P)
    ms = L.stock_models() + [m for m in L.etf_models() if m["rule"] != "overnight_for"]
    with flat_slip():
        recs = [L.run(D, m, record=True) for m in ms]
    dates = D.dates
    e = cut(dates, end) + 1
    sh = {m["id"]: (L.stats(r["E"][:e], dates[:e], "2001-01-02") or {}).get("sharpe") for m, r in zip(ms, recs)}
    R = np.c_[[L.rets(r["E"]) for r in recs]].T
    st = cut(dates, meta_from)
    flow = {m["id"] for m in ms if m.get("sel") in ("REVC_FLOW", "FOR20")}
    out = {"from": str(dates[0].date()), "to": str(dates[min(e, len(dates)) - 1].date()), "n_codes": len(P["codes"]),
           "old_sharpe": {k: (None if v is None else round(float(v), 3)) for k, v in sh.items()},
           "flow_models": {k: sh[k] for k in sorted(flow) if sh.get(k) is not None}}
    b = P["bench"]["^KS11"]
    rb = np.r_[0, b[1:] / b[:-1] - 1]
    for name, gate in (("수급 포함 (지금과 같음)", set()), ("가격 신호만", flow)):
        with flat_slip():
            E, sw, ch = L.apply_rule(D, ms, recs, R, L.LIVE_RULE, st, gate=gate)
        E = E[:e] / E[st]
        out.setdefault("meta", {})[name] = {"stats": L.stats(E[st:], dates[st:e]), "bench": stats_r(rb[st:e], dates[st:e]),
                                           "p": L.boot_p(L.rets(E)[st:], rb[st:e]), "switches": sw}
    return out


def combine(r_short, r_aw, start, share=0.5, cost=0.0025):
    """지금 계좌 방식: 단기 몫 + 전천후 몫, 분기 첫 거래일에 목표 비율로 다시 맞춤 (allweather_lab.rebalanced 와 같은 방식)."""
    import allweather_lab as AL
    x = pd.concat([r_short, r_aw], axis=1).dropna()
    x = x[x.index >= start]
    E = AL.rebalanced(x.index, x.iloc[:, 0].to_numpy(), x.iloc[:, 1].to_numpy(), share, 0, cost)
    return pd.Series(L.rets(E), index=x.index), E


# ══════════════════════════════════════════════════════════════
# 4) 언제 시작했어도 · 5) 위기별
# ══════════════════════════════════════════════════════════════
def rolling_origin(r, horizons=(252, 756)):
    """모든 날을 시작일로 h 거래일 들고 갔을 때 수익 분포."""
    r = pd.Series(r).dropna()
    E = np.cumprod(1 + r.to_numpy())
    out = {}
    for hz in horizons:
        if len(E) <= hz + 20:
            continue
        g = E[hz:] / E[:-hz] - 1
        out[f"{hz // 252}y"] = {"n": int(len(g)), "p_loss": round(float((g < 0).mean()), 3), "median": round(float(np.median(g)), 4),
                                "p5": round(float(np.percentile(g, 5)), 4), "worst": round(float(g.min()), 4), "best": round(float(g.max()), 4)}
    return out


def crisis_table(series_map):
    rows = []
    for name, a, b in CRISES:
        row = {"name": name, "from": a, "to": b, "vals": {}}
        for k, r in series_map.items():
            r = r.dropna()
            x = r[(r.index >= a) & (r.index <= b)]
            if len(x) < 15 or r.index.min() > pd.Timestamp(a) + pd.Timedelta(days=10):
                row["vals"][k] = None
                continue
            E = np.cumprod(1 + x.to_numpy())
            row["vals"][k] = {"ret": round(float(E[-1] - 1), 4), "mdd": round(float((E / np.maximum.accumulate(E) - 1).min()), 4)}
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════
# 6) 업종
# ══════════════════════════════════════════════════════════════
def sector_rotation(h, syms, bench_sym, label, start, cost_rt):
    """업종 회전 (Moskowitz & Grinblatt 1999 업종 모멘텀): 매달 첫 거래일, 최근 L개월 수익 상위 k 업종 같은 비중."""
    px = pd.DataFrame({s: series(h, s) for s in syms}).sort_index()
    px = px[px.index >= start].ffill()
    b = series(h, bench_sym).reindex(px.index).ffill()
    R = px.pct_change().to_numpy()
    rb = b.pct_change().fillna(0).to_numpy()
    d = px.index
    cols = list(px.columns)
    newm = np.r_[True, d.month[1:] != d.month[:-1]]
    res = {}
    for look in (63, 126, 252):
        for k in (2, 3):
            r = np.zeros(len(d))
            w = {}
            for t in range(1, len(d)):
                if newm[t] and t > look:
                    m = (px.iloc[t - 1] / px.iloc[t - 1 - look] - 1).dropna()
                    nw = {s: 1 / k for s in m.sort_values(ascending=False).index[:k]}
                    turn = sum(abs(nw.get(s, 0) - w.get(s, 0)) for s in set(nw) | set(w))
                    w = nw
                    r[t] -= turn * cost_rt / 2
                r[t] += sum(wt * np.nan_to_num(R[t, cols.index(s)]) for s, wt in w.items())
            res[f"{look // 21}개월·상위{k}"] = {"stats": stats_r(r, d), "p": L.boot_p(r[260:], rb[260:]),
                                              "oos": stats_r(r, d, b=OOS_END) if d[0] < pd.Timestamp("2010-01-01") else None,
                                              "design": stats_r(r, d, a="2014-01-01")}
    for (k, v), a in zip(res.items(), S.holm([v["p"] for v in res.values()])):
        v["p_holm"] = a
    bench = {"stats": stats_r(rb, d), "oos": stats_r(rb, d, b=OOS_END) if d[0] < pd.Timestamp("2010-01-01") else None,
             "design": stats_r(rb, d, a="2014-01-01")}
    each = {s: stats_r(px[s].pct_change().fillna(0).to_numpy(), d) for s in cols}
    return {"label": label, "from": str(d[0].date()), "variants": res, "bench": bench, "each": each}


def kr_sector_attribution():
    """국내 단기 종목 모델(5·10일 보유, 걸러내기 없음)이 어느 업종에서 벌고 잃었나 — 매매를 업종별로 합산 (2019~ 실제 자료)."""
    D = L.Data()
    sec = pd.read_csv("data/sectors.csv", dtype={"종목코드": str}).set_index("종목코드")["섹터"].to_dict()
    codes = D.P["codes"]
    rt = sum(L.costs("stock"))
    agg = {}
    for m in L.stock_models():
        if m["hold"] not in ("H5", "H10") or m["gate"] != "none":
            continue
        log = []
        L.run_stock(D, m, log=log)
        pend = {}
        for e in log:
            if e["side"] == "buy":
                pend[e["j"]] = e
            elif e["j"] in pend:
                b = pend.pop(e["j"])
                r = e["px"] / b["px"] - 1 - rt
                a = agg.setdefault(sec.get(codes[e["j"]], "기타"), {"n": 0, "sum": 0.0, "win": 0})
                a["n"] += 1
                a["sum"] += r
                a["win"] += r > 0
    rows = [{"sector": s, "trades": v["n"], "avg_bp": round(v["sum"] / v["n"] * 1e4, 1), "win": round(v["win"] / v["n"], 3)}
            for s, v in agg.items() if v["n"] >= 30]
    return sorted(rows, key=lambda x: -x["avg_bp"])


# ══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    h = load_hist()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "oos_end": OOS_END,
           "assumptions": {"div": DIV, "cash": CASH, "slip": SLIP, "kr_design": KR_DESIGN, "us_design": US_DESIGN},
           "range": {s: [str(g["날짜"].min().date()), str(g["날짜"].max().date())] for s, g in h.groupby("종목코드")}}
    dates, Op, Cp = proxy_prices(h)
    etf, etf_r, etf_corr = etf_rules_long(dates, Op, Cp)
    res["etf_rules"], res["etf_rank_corr"] = etf, etf_corr
    print(f"■ 국내 ETF 규칙 {dates[0].date()}~ (표본 외 ~{OOS_END}) · 두 기간 순위 상관 {etf_corr}")
    for k, v in etf.items():
        print(f"  {k:16s} 표본 외 {v['oos']} P {v['p_oos']}→{v.get('p_oos_holm')} | 설계 {v['design']}")
    aw, r_aw = aw_long(dates, Op, Cp)
    res["aw"] = aw
    print(f"■ 전천후(대용) {aw['from']}~2013: {aw['oos']} · 지수 {aw['idx_oos']} · P {aw['p_oos']}")
    print(f"             2014~: {aw['design']} · 지수 {aw['idx_design']}")
    smap = {"코스피(배당 포함 근사)": etf_r["hold"], "전날 내림→보유 (단기 몫 규칙)": etf_r["oo_rev"], "전천후 혼합(대용)": r_aw,
            "S&P 500 (SPY, 달러)": series(h, "SPY").pct_change()}
    pers = {"2001~2008": ("2001-01-02", "2008-12-31"), "2009~2013": ("2009-01-02", "2013-12-31")}
    try:
        us, us_r, spy_r, us_meta = models_long(US_STOCKS, ("SPY", "QQQ"), "us", US_DESIGN,
                                               {"2001~2008": ("2001-01-02", "2008-12-31"), "2009~2014": ("2009-01-02", "2014-12-31")})
        res["us_models"] = us
        smap["미국 추세 상위 10 (달러)"] = us_r["S:MOM:10:BAND:none"]
        print("  미국 옛→새:", {k: v for k, v in us["old_new"].items() if k != "top10_old"})
        print("  미국 고르기 규칙:", {k: (v["stats"], v["bench"] and v["bench"]["sharpe"], v["p"]) for k, v in us["meta"]["periods"].items()})
        print(f"■ 미국 종목 모델 {us['n_models']}개 · {us['n_stocks']}종목(2001년 살아 있던 {us['alive_2001']}) · 순위 상관 {us['rank_corr']}")
        for k, v in us["by_period"].items():
            print(f"  {k}: 수익 {v['positive']}/{v['n']} · 지수보다 샤프 높음 {v['beat_bench']} · 샤프 중앙값 {v['sharpe_med']} · 기준 {v['bench']}")
        print("  설계 기간 5분위(아래→위)별 옛 샤프 중앙값:", us["quintiles"])
    except Exception as ex:
        print("미국 종목 모델 건너뜀:", type(ex).__name__, ex)
    if os.path.exists(KR_STOCKS):
        try:
            kr, kr_r, ks_r, kr_meta = models_long(KR_STOCKS, ("^KS11",), "kr", KR_DESIGN,
                                                  {**pers, "2014~2018": ("2014-01-02", "2018-12-31")})
            res["kr_models"] = kr
            smap["단기 몫 고르기 규칙 (국내, 2002~)"] = kr_meta
            try:
                import settings
                share = float((settings.load().get("short") or {}).get("aw_share", 0.5))
            except Exception:
                share = 0.5
            res["combined_by_share"] = {}
            for sh_ in sorted({0.5, 0.65, share, 0.9, 1.0}):
                rc_, Ec_ = combine(kr_meta, r_aw, "2005-01-03", share=sh_)
                e_ = cut(rc_.index, OOS_END) + 1
                res["combined_by_share"][f"{sh_:.2f}"] = {"all": L.stats(Ec_, rc_.index), "oos": L.stats(Ec_[:e_], rc_.index[:e_]),
                                                         "design": L.stats(Ec_[e_ - 1:], rc_.index[e_ - 1:]),
                                                         "p_loss_3y": (rolling_origin(rc_).get("3y") or {}).get("p_loss")}
            rc, Ec = combine(kr_meta, r_aw, "2005-01-03", share=share)
            smap[f"지금 계좌 방식 (단기{round((1 - share) * 100)}+전천후{round(share * 100)}, 2005~)"] = rc
            e = cut(rc.index, OOS_END) + 1
            res["combined"] = {"from": str(rc.index[0].date()), "share": share, "oos": L.stats(Ec[:e], rc.index[:e]),
                               "design": L.stats(Ec[e - 1:], rc.index[e - 1:]), "years": L.yearly(Ec, rc.index)}
            print("  국내 옛→새:", {k: v for k, v in kr["old_new"].items() if k != "top10_old"})
            for b in kr["old_new"]["top10_old"]:
                print("     ", b)
            print("  국내 고르기 규칙:", {k: (v["stats"], v["bench"] and v["bench"]["sharpe"], v["p"], v["picks"]) for k, v in kr["meta"]["periods"].items()})
            print("  지금 계좌 방식(대용) 2005~2013:", res["combined"]["oos"], "· 2014~:", res["combined"]["design"])
            print(f"■ 국내 종목 모델 {kr['n_models']}개 · {kr['n_stocks']}종목(2001년 {kr['alive_2001']}) · 이상한 날 {kr['n_bad_days']} · 순위 상관 {kr['rank_corr']}")
            for k, v in kr["by_period"].items():
                print(f"  {k}: 수익 {v['positive']}/{v['n']} · 지수보다 샤프 높음 {v['beat_bench']} · 샤프 중앙값 {v['sharpe_med']} · 기준 {v['bench']}")
            print("  설계 기간 5분위(아래→위)별 옛 샤프 중앙값:", kr["quintiles"])
            for b in kr["design_top"][:5]:
                print("   ", b)
        except Exception as ex:
            print("국내 종목 모델 건너뜀:", type(ex).__name__, ex)
    res["rolling"] = {k: rolling_origin(v) for k, v in smap.items()}
    res["crises"] = crisis_table(smap)
    for k, v in res["rolling"].items():
        print(f"  언제 시작했어도 {k}: {v}")
    for row in res["crises"]:
        print(f"  {row['name']:12s}", {k: (v and v['ret']) for k, v in row["vals"].items()})
    kr_secs = [s for s in ("091160.KS", "091170.KS", "091180.KS", "102970.KS", "117460.KS", "117700.KS", "117680.KS", "140710.KS")
               if (h["종목코드"] == s).any()]
    res["sector_kr"] = sector_rotation(h, kr_secs, "069500.KS", "국내 업종 ETF 회전 (8개 업종)", "2011-05-02", 0.0012)
    res["sector_us"] = sector_rotation(h, ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"], "SPY",
                                       "미국 업종 SPDR 회전 (9개 업종)", "1999-01-04", 0.0005)
    for key in ("sector_kr", "sector_us"):
        x = res[key]
        print(f"■ {x['label']} ({x['from']}~) 기준 {x['bench']['stats']}")
        for k, v in x["variants"].items():
            print(f"   {k}: {v['stats']} · P {v['p']}→{v['p_holm']} · 표본 외 {v['oos'] and v['oos']['sharpe']} 설계 {v['design'] and v['design']['sharpe']}")
    try:
        res["kr_attr"] = kr_sector_attribution()
        print("■ 국내 단기 모델 업종별:", res["kr_attr"][:4], "…", res["kr_attr"][-3:])
    except Exception as ex:
        print("업종별 기여 건너뜀:", type(ex).__name__, ex)
    try:
        pit = pit_report()
        if pit:
            res["kr_pit"] = {k: v for k, v in pit.items() if k != "old_sharpe"}
            print("■ 상장폐지·수급 포함 긴 역사:", {k: (v["stats"], v["bench"] and v["bench"]["sharpe"], v["p"]) for k, v in pit["meta"].items()})
    except Exception as ex:
        pit = None
        print("상장폐지·수급 포함 검증 건너뜀:", type(ex).__name__, ex)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if "kr_models" in res:
        on = res["kr_models"]["old_new"]
        osh = on.pop("old_sharpe")
        if pit:                                        # 생존편향 없는 자료가 있으면 그것으로 판정 (수급 모델도 관문 대상이 된다)
            osh = {**osh, **{k: v for k, v in pit["old_sharpe"].items() if v is not None}}
        for k, v in res["etf_rules"].items():           # 지수 ETF 규칙은 생존편향 없는 코스피 지수(1997~2013)로 판정
            osh[f"E:{k}"] = (v["oos"] or {}).get("sharpe")
        fail = sorted(k for k, v in osh.items() if v is not None and v < 0)
        gate = {"generated": res["generated"], "era": {"stock": on["old"], "etf": [str(dates[0].date()), OOS_END]},
                "rule": "설계에 안 쓴 옛 시대 샤프 < 0 이면 단기 몫 후보에서 뺀다",
                "source": {"stock": ("KRX 그때 실제 대형주 2001~2018 (상장폐지·수급 포함)" if pit else
                                     "국내 대형주 2001~2018 (야후 수정가 · 지금 쓰는 시뮬레이터·비용 · 생존편향 있음)"),
                           "etf": "코스피 지수 1997~2013 + 배당 근사 (지금 쓰는 시뮬레이터·비용)"},
                "fail": fail, "sharpe": osh}
        json.dump(gate, open(GATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        res["gate"] = {"n": len(osh), "n_fail": len(fail), "etf_fail": [k for k in fail if k.startswith("E:")], "era": gate["era"],
                       "by_hold": {h: int(sum(1 for k in fail if k.startswith("S:") and k.split(":")[3] == h)) for h in ("H1", "H3", "H5", "H10", "BAND")}}
        print(f"긴 역사 관문: 옛 시대 실패 {len(fail)}/{len(osh)}개 → {GATE}")
    if "us_models" in res:
        res["us_models"]["old_new"].pop("old_sharpe", None)
    from short_trader import clean
    res = clean(res)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(res, open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"끝 ({time.time() - t0:.0f}s) → {OUT}")
    return res


if __name__ == "__main__":
    main()
