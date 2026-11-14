"""
고배당 전략 실험실 — 고배당주를 사서 배당을 받으며 가면 무엇이 달라지나, 이익이 어디서 나오나 (가격 vs 배당).

배당 자료: 따로 받지 않는다. data/history/kr_stocks.csv(Yahoo 수정가 = 배당까지 반영) 와 시뮬레이터 패널(배당 빠진 실제 가격)의
  비율 log(수정가/실제가) 가 배당락일에 계단처럼 오르고 그대로 남는다 → 그 계단 = 그날 배당 ÷ 전날 종가.
  하루만 튀고 돌아오는 차이(두 자료원의 종가 어긋남)는 버린다: 오른 폭의 80% 이상이 다음 날과 사흘 뒤에도 남아 있어야 배당으로 센다.
  배당자료가 있는 종목만 고배당 후보 (최근 코스피 100 에 든 적 있는 208개 → 2018 전에는 생존편향: 나중에 망한 고배당주가 빠져 있음).

계산 (결과 보기 전에 정함, 2026-11-14)
  · 배당락일 전날 장 마감에 들고 있으면 배당을 받는다 · 배당소득세 15.4% 를 뗀다 (금융소득 2천만 원 이하 가정)
  · 매일 장 마감 판단 → 다음 날 시가 주문, 왕복 비용 약 0.34% (score_lab 과 같은 흉내 장치)
  · 수익 = 가격 + 배당 으로 나눠 본다 (같은 매매를 배당 없이 한 번 더 돌린 차이 = 배당 몫)
  · 기준: 같은 '배당자료 있는' 대형주를 똑같이 산 평균 (배당 포함, 세후) · KODEX 200 (Yahoo 수정가 = 배당 포함) · DM:2

전략 8개
  D1 고배당 꾸준히      점수 = 최근 1년 배당수익률 순위 · 10자리 · 상위 10% 안이면 삼 · 상위 30% 밖이면 팜 · 최소 60일
  D2 고배당 1년 보유     같은 점수 · 10자리 · 상위 10% 안이면 삼 · 상위 50% 밖이면 팜 · 최소 250일 (거의 해마다 한 번)
  D3 고배당 + 추세·눌림  점수 = 배당수익률 순위 + 장기 추세·단기 눌림 점수(3차) 반반 · 10자리 · 10% / 50% / 최소 20일
  D4 고배당 + 덜 흔들림  점수 = 배당수익률 순위 + (−60일 변동성 순위) 반반 · D1 규칙
  D5 배당철 받기         해마다 10월 첫 거래일에 배당수익률 상위 10개를 사서 다음 해 1월 말에 판다 (연말 배당을 받음) · 그 밖에는 현금
  D6 배당락 전 팔기      10월 첫 거래일에 사서 12월 20일에 판다 (배당락 전 오름만 먹고 배당은 안 받음) · 그 밖에는 현금
  C  장기 추세 + 단기 눌림 (3차에서 찾은 것) — 배당을 넣어도 대형주를 이기나 (비교용)
  E  배당자료 있는 대형주 똑같이 (기준)

  결과를 본 뒤 더한 것 (탐색): 작은 계좌용 3자리 (D1·D4, 상위 5% 안이면 삼) · C 와 D4 반반 섞기
  한계: 수정가는 주식배당·유상증자도 '배당'처럼 반영 → 그런 종목(예: 셀트리온)이 고배당으로 잘못 잡힐 수 있음 (수익 계산은 실제가의 빠짐을 메우는 쪽이라 크게 안 틀림)
        2024 뒤 배당일을 옮긴 회사의 배당 일부가 빠짐(보수적) · 2018 연말 배당은 두 자료 이음매라 빠짐

  python src/div_lab.py → data/div_lab.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_lab as SC  # noqa: E402

OUT = "data/div_lab.json"
DOCS = "docs/div_lab.json"
TAX = 0.154
START = "2003-01-01"
ERAS3 = {"2003~2007": ("2003-01-01", "2007-12-31"), "2008~2018": ("2008-01-01", "2018-12-31"), "2019~": ("2019-01-01", None), "전체 2003~": ("2003-01-01", None)}
COMPOSITE = {"r250x": 1, "r120": 1, "ma120": 1, "hi52": 1, "r60": 1, "r5": -1, "r1": -1, "ma20": -1, "clv5": -1, "clv1": -1, "r20": -1}
RULES = {"D1": {"slots": 10, "q_in": 0.10, "q_out": 0.30, "min_hold": 60, "stop": 0.0, "max_hold": 0},
         "D2": {"slots": 10, "q_in": 0.10, "q_out": 0.50, "min_hold": 250, "stop": 0.0, "max_hold": 0},
         "D3": {"slots": 10, "q_in": 0.10, "q_out": 0.50, "min_hold": 20, "stop": 0.0, "max_hold": 0},
         "D4": {"slots": 10, "q_in": 0.10, "q_out": 0.30, "min_hold": 60, "stop": 0.0, "max_hold": 0},
         "SEASON": {"slots": 10, "q_in": 0.10, "q_out": 1.01, "min_hold": 1, "stop": 0.0, "max_hold": 0},
         "C": {"slots": 10, "q_in": 0.10, "q_out": 0.50, "min_hold": 20, "stop": 0.0, "max_hold": 0}}
NAMES = {"D1": "고배당 꾸준히", "D2": "고배당 1년 보유", "D3": "고배당 + 추세·눌림", "D4": "고배당 + 덜 흔들림", "D5": "배당철 받기 (10월~1월)",
         "D6": "배당락 전 팔기 (10월~12월 20일)", "C": "장기 추세 + 단기 눌림 (3차)", "E": "배당자료 있는 대형주 똑같이 (기준)"}


def adjusted_prices():
    h = pd.read_csv("data/history/kr_stocks.csv", dtype={"종목코드": str}, usecols=["날짜", "종가", "종목코드"])
    h["날짜"] = pd.to_datetime(h["날짜"])
    return h.pivot_table(index="날짜", columns="종목코드", values="종가")


def dividend_events(E, adj, raw_close=None):
    """시대 E 의 T×N 배당 (배당락일에 '배당 ÷ 전날 종가', 아니면 0) · 배당자료가 있는 칸 표시.
    배당으로 세는 조건: 비율의 오른 폭이 0.2%~15% · 1·3·5·10·20 거래일 뒤에도 오른 폭의 80% 이상이 남음
    (자료가 며칠 비어 앞 값으로 채운 구간에서 생기는 '올랐다 돌아오는' 어긋남을 버림) · 그날과 전날 실제 종가 기록이 있음."""
    T, N = E.G.shape
    raw = pd.DataFrame(E.Cf, index=E.dates, columns=E.codes)
    A = adj.reindex(E.dates)
    Dv = np.zeros((T, N), np.float32)
    cov = np.zeros((T, N), bool)
    has = np.isfinite(raw_close) if raw_close is not None else np.ones((T, N), bool)
    for j, c in enumerate(E.codes):
        if c not in A.columns:
            continue
        a = A[c].to_numpy(float)
        r = raw[c].to_numpy(float)
        with np.errstate(all="ignore"):
            q = np.log(a / r)
        ok = np.isfinite(q)
        cov[:, j] = ok
        qf = pd.Series(q).ffill().to_numpy()
        d = np.r_[np.nan, np.diff(qf)]
        for t in np.where((d > 0.002) & (d < 0.15))[0]:
            if t < 1 or t + 20 >= T or not (has[t, j] and has[t - 1, j]):
                continue
            base = qf[t - 1]
            if all((qf[t + k] - base) >= 0.8 * d[t] for k in (1, 3, 5, 10, 20)):
                Dv[t, j] = np.expm1(d[t])
    # 배당 일정 거르기: 3·6·9·12월 마지막 7거래일 (2023 배당 절차 개편 뒤로는 2~4월 전부) 밖의 계단은
    #   그날 여러 종목(4개 이상)에서 함께 생겼거나 1% 미만이면 자료 어긋남으로 보고 버린다.
    win = dividend_windows(E.dates)
    cnt = (Dv > 0).sum(1)
    off = ~win
    Dv[off & (cnt > 3)] = 0.0
    small = (Dv < 0.01) & off[:, None]
    Dv[small] = 0.0
    return Dv, cov


def dividend_windows(dates):
    d = pd.DatetimeIndex(dates)
    win = np.zeros(len(d), bool)
    s = pd.Series(np.arange(len(d)), index=d)
    for (y, m), g in s.groupby([d.year, d.month]):
        if m in (3, 6, 9, 12):
            win[g.to_numpy()[-7:]] = True
        if y >= 2023 and m in (2, 3, 4):
            win[g.to_numpy()] = True
    return win


def attach_dividends(ER, log=print):
    import setup_lab as SLB
    adj = adjusted_prices()
    for e, E in ER.items():
        Dv, cov = dividend_events(E, adj, raw_close=SLB.panel(e)[0]["C"])
        T, N = Dv.shape
        E.Dv, E.cov = Dv, cov
        ttm = pd.DataFrame(Dv).rolling(250, min_periods=240).sum().to_numpy()
        ok_hist = pd.DataFrame(cov.astype(float)).rolling(250, min_periods=240).sum().to_numpy() >= 240
        E.ttm = np.where(ok_hist & E.U, ttm, np.nan)
        inc = np.zeros((T, N))
        inc[:-2] = Dv[2:]                                        # 다음 날 시가 → 그다음 날 시가 사이에 배당락이 오면 그 구간에 받음
        E.inc = (1 - TAX) * inc
        E.G_price = E.G.copy()
        E.G_div = np.where(np.isfinite(E.G), E.G + E.inc, np.nan)
        Ucov = E.U & cov
        with np.errstate(all="ignore"):
            E.ew_cov_div = np.nanmean(np.where(Ucov, E.G_div, np.nan), axis=1)
            E.ew_cov_price = np.nanmean(np.where(Ucov, E.G_price, np.nan), axis=1)
            E.ew_all_div = np.nanmean(np.where(E.U, np.where(cov, E.G_div, E.G), np.nan), axis=1)
        y = np.nansum(np.where(Ucov, Dv, 0), axis=1)
        log(f"  {e}: 배당 계단 {int((Dv > 0).sum()):,}번 · 배당자료 덮는 비율(상위 100 일수) {Ucov.sum() / E.U.sum():.0%} · "
            f"덮는 종목 평균 배당수익률/년 {float(np.nansum(y) / max(1, Ucov.sum()) * 252):.1%}")
    return ER


def rank01(x, U):
    return pd.DataFrame(np.where(U, x, np.nan)).rank(axis=1, pct=True).to_numpy() - 0.5


def scores(ER):
    out = {}
    for e, E in ER.items():
        Uc = E.U & np.isfinite(E.ttm)
        y = rank01(E.ttm, Uc)
        comp = E.score(COMPOSITE)
        comp_r = rank01(comp, Uc)
        vol = E.X[SC.FACTORS.index("vol60")]
        T = E.G.shape[0]
        season5, season6 = np.zeros(T, bool), np.zeros(T, bool)
        d = E.dates
        for yr in sorted(set(d.year)):
            a = np.searchsorted(d, pd.Timestamp(f"{yr}-10-01"))
            b5 = np.searchsorted(d, pd.Timestamp(f"{yr + 1}-02-01"))
            b6 = np.searchsorted(d, pd.Timestamp(f"{yr}-12-21"))
            if a < T:
                season5[a:min(b5, T)] = True
                season6[a:min(b6, T)] = True
        y_entry5 = np.full_like(y, np.nan)
        y_entry6 = np.full_like(y, np.nan)
        for s, tgt in ((season5, y_entry5), (season6, y_entry6)):
            tgt[s] = y[s]
        out[e] = {"D1": y, "D2": y, "D3": y + comp_r, "D4": y + rank01(-vol, Uc), "D5": y_entry5, "D6": y_entry6,
                  "C": np.where(E.U, comp, np.nan)}
    return out


def run_one(ER, S, rule, with_div=True, cost_mult=1.0, a=START, b=None, cash0=None):
    """배당 포함/제외로 흉내 (E.G 를 잠시 바꿔 끼움)."""
    saved = {e: (E.G, E.ew) for e, E in ER.items()}
    try:
        for E in ER.values():
            E.G = E.G_div if with_div else E.G_price
            E.ew = E.ew_cov_div if with_div else E.ew_cov_price
        return SC.stitch(ER, S, rule, a, b, cost_mult=cost_mult, cash0=cash0)
    finally:
        for e, E in ER.items():
            E.G, E.ew = saved[e]


def stats(r, ew=None):
    return SC.stats_full(r, ew)


def kodex_tr(idx):
    h = pd.read_csv("data/history/hist.csv", dtype={"종목코드": str})
    k = h[h["종목코드"] == "069500.KS"].assign(날짜=lambda x: pd.to_datetime(x["날짜"])).set_index("날짜")["종가"].sort_index()
    k = k.reindex(k.index.union(idx)).ffill().reindex(idx)
    return k.pct_change().fillna(0.0)


def run(log=print):
    t0 = time.time()
    SC.set_factors(True)
    ER = SC.eras(log)
    attach_dividends(ER, log)
    S = scores(ER)
    Sx = {k: {e: S[e][k] for e in ER} for k in ("D1", "D2", "D3", "D4", "D5", "D6", "C")}
    rule_of = {"D1": "D1", "D2": "D2", "D3": "D3", "D4": "D4", "D5": "SEASON", "D6": "SEASON", "C": "C"}
    res = {"strategies": {}, "names": NAMES, "tax": TAX}
    series = {}
    for k in ("D1", "D2", "D3", "D4", "D5", "D6", "C"):
        r = RULES[rule_of[k]]
        a = run_one(ER, Sx[k], r, True)
        p = run_one(ER, Sx[k], r, False)
        x2 = run_one(ER, Sx[k], r, True, cost_mult=2.0)
        idx = a["r"].index
        series[k] = a["r"]
        ew = a["ew"]
        row = {"name": NAMES[k], "rule": SC.rule_label(r) if k not in ("D5", "D6") else NAMES[k], "trades_per_year": round(len(a["trades"]) / (len(idx) / 252), 1),
               "invested": round(a["invested"], 3)}
        for pn, (x0, x1) in ERAS3.items():
            c = lambda s: SC.cut(s, x0, x1)                                  # noqa: E731
            st, sp = stats(c(a["r"]), c(ew)), stats(c(p["r"]))
            row[pn] = {**st, "price_cagr": sp["cagr"], "div_part": round(st["cagr"] - sp["cagr"], 4), "x2_cagr": stats(c(x2["r"]))["cagr"]}
        # 배당 수입 (세후, 연 평균 · 자산 대비) — 같은 매매에서 배당 몫만
        row["div_income_ann"] = round(float((a["r"] - p["r"]).mean() * 252), 4)
        res["strategies"][k] = row
        h = row["2019~"]
        log(f"  {k} {NAMES[k]:22s}: 전체 연 {row['전체 2003~']['cagr']:+.1%} (가격 {row['전체 2003~']['price_cagr']:+.1%} + 배당 {row['전체 2003~']['div_part']:+.1%}) 샤프 {row['전체 2003~']['sharpe']:.2f} "
            f"낙폭 {row['전체 2003~']['mdd']:.0%} | 시기별 초과 {row['2003~2007']['excess_ann']:+.0%}/{row['2008~2018']['excess_ann']:+.0%}/{h['excess_ann']:+.0%} "
            f"| 2019~ 연 {h['cagr']:+.1%} 샤프 {h['sharpe']:.2f} 낙폭 {h['mdd']:.0%} · 비용2배 {h['x2_cagr']:+.1%} · 거래 {row['trades_per_year']:.0f}/년")
        SC._PREP.clear()
    # 기준들
    idx = series["D1"].index
    ew_div = pd.Series(np.concatenate([ER[e].ew_cov_div[SC.Era.rows(ER[e], START, None)[0]:min(SC.Era.rows(ER[e], START, None)[1], ER[e].G.shape[0] - 2)] for e in ER]), index=idx).fillna(0.0)
    ew_px = pd.Series(np.concatenate([ER[e].ew_cov_price[SC.Era.rows(ER[e], START, None)[0]:min(SC.Era.rows(ER[e], START, None)[1], ER[e].G.shape[0] - 2)] for e in ER]), index=idx).fillna(0.0)
    kx = kodex_tr(idx)
    ks = SC.kospi_daily(idx)
    dm = SC.card_daily("DM:2", idx)
    bench = {}
    for pn, (x0, x1) in ERAS3.items():
        c = lambda s: SC.cut(s, x0, x1)                                      # noqa: E731
        bench[pn] = {"ew_div": stats(c(ew_div)), "ew_price": stats(c(ew_px)), "kodex200_tr": stats(c(kx)) if x0 >= "2019" else None, "kospi_price": stats(c(ks)),
                     "dm2": stats(c(dm)) if x0 >= "2005" else None}
        bench[pn]["ew_div"]["div_part"] = round(bench[pn]["ew_div"]["cagr"] - bench[pn]["ew_price"]["cagr"], 4)
    res["bench"] = bench
    for pn, v in bench.items():
        log(f"  기준 {pn}: 대형주 똑같이 연 {v['ew_div']['cagr']:+.1%} (배당 {v['ew_div']['div_part']:+.1%}) 샤프 {v['ew_div']['sharpe']:.2f} · "
            f"코스피(가격) {v['kospi_price']['cagr']:+.1%}" + (f" · KODEX 200(배당 포함) {v['kodex200_tr']['cagr']:+.1%}" if v["kodex200_tr"] else "")
            + (f" · DM:2 {v['dm2']['cagr']:+.1%} 샤프 {v['dm2']['sharpe']:.2f}" if v["dm2"] else ""))
    # 해마다 · 곡선
    yrs = {}
    for k, s in {**series, "E": ew_div}.items():
        g = (1 + s).groupby(s.index.year).prod() - 1
        for y, v in g.items():
            yrs.setdefault(str(y), {})[k] = round(float(v), 4)
    res["years"] = yrs
    res["curve"] = {"date": [str(d.date()) for d in idx[::5]], **{k: [round(float(v), 4) for v in (1 + s).cumprod()[::5]] for k, s in {**series, "E": ew_div, "kodex": kx}.items()}}
    # 실제 주식 수 · 20만 원 등 (2019~)
    acc = {}
    for k in ("D1", "D2", "D3", "D5"):
        r = RULES[rule_of[k]]
        acc[k] = {}
        for c0 in (200_000, 1_000_000, 10_000_000):
            q = run_one(ER, Sx[k], r, True, a="2019-01-01", cash0=c0)
            st = stats(q["r"], q["ew"])
            q0 = run_one(ER, Sx[k], r, False, a="2019-01-01", cash0=c0)
            div_won = float(c0 * ((1 + q["r"]).prod() - (1 + q0["r"]).prod()))
            acc[k][str(c0)] = {**st, "invested": round(q["invested"], 3), "final": round(float(c0 * (1 + q["r"]).prod())), "div_won_total": round(div_won)}
        SC._PREP.clear()
    for k in ("D1", "D4"):
        r = {**RULES[k], "slots": 3, "q_in": 0.05}
        acc[k + "·3자리"] = {}
        for c0 in (200_000, 1_000_000):
            q = run_one(ER, Sx[k], r, True, a="2019-01-01", cash0=c0)
            q0 = run_one(ER, Sx[k], r, False, a="2019-01-01", cash0=c0)
            st = stats(q["r"], q["ew"])
            acc[k + "·3자리"][str(c0)] = {**st, "invested": round(q["invested"], 3), "final": round(float(c0 * (1 + q["r"]).prod())),
                                          "div_won_total": round(float(c0 * ((1 + q["r"]).prod() - (1 + q0["r"]).prod())))}
        SC._PREP.clear()
    res["accounts"] = acc
    for k, v in acc.items():
        for c0, a_ in v.items():
            log(f"  계좌 {k} {int(c0):>10,}원 (2019~): 연 {a_['cagr']:+.1%} 샤프 {a_['sharpe']:.2f} 낙폭 {a_['mdd']:.0%} 산 비율 {a_['invested']:.0%} · 배당(세후) 합계 {a_['div_won_total']:,}원")
    mix = 0.5 * series["C"] + 0.5 * series["D4"]
    res["mix"] = {"what": "장기 추세 + 단기 눌림 50% + 고배당·덜 흔들림 50% (날마다 반반, 탐색)",
                  "corr_C_D4": round(float(np.corrcoef(series["C"], series["D4"])[0, 1]), 3),
                  **{pn: stats(SC.cut(mix, x0, x1), SC.cut(ew_div, x0, x1)) for pn, (x0, x1) in ERAS3.items()}}
    series["MIX"] = mix
    log("  반반 섞기 (C+D4): " + " · ".join(f"{pn} 연 {res['mix'][pn]['cagr']:+.1%} 샤프 {res['mix'][pn]['sharpe']:.2f} 낙폭 {res['mix'][pn]['mdd']:.0%}" for pn in ERAS3))
    for k in ("MIX",):
        g = (1 + series[k]).groupby(series[k].index.year).prod() - 1
        for y, v in g.items():
            res["years"].setdefault(str(y), {})[k] = round(float(v), 4)
        res["curve"][k] = [round(float(v), 4) for v in (1 + series[k]).cumprod()[::5]]
    # 오늘 기준 고배당 상위 (B 마지막 날)
    import shadow_wf as SW
    nm = SW.names()
    E = ER["B"]
    t = E.i1 - 1
    yv = E.ttm[t]
    order = [j for j in np.argsort(-np.where(np.isfinite(yv), yv, -np.inf))[:15] if np.isfinite(yv[j])]
    d3 = S["B"]["D3"][t]
    order3 = [j for j in np.argsort(-np.where(np.isfinite(d3), d3, -np.inf))[:15] if np.isfinite(d3[j])]
    res["today"] = {"date": str(E.dates[t].date()),
                    "high_yield": [{"rank": i + 1, "code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j]), "ttm_yield": round(float(yv[j]), 4)} for i, j in enumerate(order)],
                    "yield_plus_trend": [{"rank": i + 1, "code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j]), "ttm_yield": round(float(yv[j]), 4) if np.isfinite(yv[j]) else None}
                                         for i, j in enumerate(order3)]}
    log("  오늘 배당수익률 상위: " + ", ".join(f"{x['name']} {x['ttm_yield']:.1%}" for x in res["today"]["high_yield"][:10]))
    log("  오늘 고배당 + 추세·눌림 상위: " + ", ".join(x["name"] for x in res["today"]["yield_plus_trend"][:10]))
    # 배당 자료 점검: 잘 알려진 종목의 해마다 배당수익률
    chk = {}
    for code in ("005930", "105560", "017670", "033780", "055550", "030200"):
        for e, E in ER.items():
            if code in E.codes:
                j = E.codes.index(code)
                s = pd.Series(E.Dv[:, j], index=E.dates)
                lo, hi = E.i0, E.i1
                g = s.iloc[lo:hi].groupby(E.dates[lo:hi].year).sum()
                for y, v in g.items():
                    chk.setdefault(nm.get(code, code), {})[str(y)] = round(float(v), 4)
    res["check"] = chk
    res["secs"] = round(time.time() - t0)
    res["generated"] = time.strftime("%Y-%m-%d %H:%M")
    for fn in (OUT, DOCS):
        json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"끝 ({res['secs']}초) → {OUT}")
    return res




def export(outdir):
    """화면(대시보드)용 자료 파일 — {rows, meta}."""
    res = json.load(open(OUT, encoding="utf-8"))
    os.makedirs(outdir, exist_ok=True)
    code = open(os.path.abspath(__file__), encoding="utf-8").read()
    meta = [{"label": "div_lab.py", "value": code, "type": "code", "lang": "python"},
            {"label": "배당 자료", "value": "data/history/kr_stocks.csv (Yahoo 수정가) 와 실제가 비율의 계단 → 배당락일 배당 (세후 15.4% 뺌)"},
            {"label": "한계", "value": "주식배당·유상증자도 배당처럼 잡힐 수 있음 · 2024 뒤 일부 배당 빠짐 · 2018 전 배당자료는 지금 살아 있는 종목만(생존편향)"}]
    f = {}
    order = ["D1", "D2", "D3", "D4", "D5", "D6", "C"]
    f["div_structure"] = [{"key": k, "name": res["strategies"][k]["name"], "period": pn,
                           **{m: res["strategies"][k][pn][m] for m in ("cagr", "price_cagr", "div_part", "sharpe", "mdd", "excess_ann", "x2_cagr")},
                           "trades_per_year": res["strategies"][k]["trades_per_year"], "invested": res["strategies"][k]["invested"]}
                          for k in order for pn in ERAS3]
    for pn in ERAS3:
        b = res["bench"][pn]
        f["div_structure"].append({"key": "E", "name": NAMES["E"], "period": pn, "cagr": b["ew_div"]["cagr"], "price_cagr": b["ew_price"]["cagr"],
                                   "div_part": b["ew_div"]["div_part"], "sharpe": b["ew_div"]["sharpe"], "mdd": b["ew_div"]["mdd"], "excess_ann": 0.0,
                                   "x2_cagr": None, "trades_per_year": None, "invested": None})
        m = res["mix"][pn]
        f["div_structure"].append({"key": "MIX", "name": "추세·눌림 50% + 고배당·덜 흔들림 50% (탐색)", "period": pn, "cagr": m["cagr"], "price_cagr": None,
                                   "div_part": None, "sharpe": m["sharpe"], "mdd": m["mdd"], "excess_ann": m.get("excess_ann"), "x2_cagr": None,
                                   "trades_per_year": None, "invested": None})
    f["div_bench"] = [{"period": pn, "who": who, **st} for pn, b in res["bench"].items()
                      for who, st in (("코스피 (가격만)", b["kospi_price"]), ("KODEX 200 (배당 포함)", b["kodex200_tr"]), ("DM:2", b["dm2"])) if st]
    f["div_accounts"] = [{"key": k, "name": NAMES.get(k.split("·")[0], k) + (" · 3자리" if "3자리" in k else ""), "cash0": int(c0), **v}
                         for k, d in res["accounts"].items() for c0, v in d.items()]
    f["div_today"] = [{**x, "date": res["today"]["date"]} for x in res["today"]["high_yield"]]
    f["div_check"] = [{"stock": s, "year": int(y), "yield": v} for s, d in res["check"].items() for y, v in d.items() if int(y) >= 2005]
    for k, rows in f.items():
        json.dump({"rows": rows, "meta": meta}, open(os.path.join(outdir, f"{k}.json"), "w", encoding="utf-8"), ensure_ascii=False, default=float)
    return list(f)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--export":
        print(export(sys.argv[2]))
    else:
        run(lambda m: print(m, flush=True))
