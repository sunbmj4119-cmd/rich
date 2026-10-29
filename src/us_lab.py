"""
미국 주식 단기 모델 연구 — 국내와 같은 엔진(short_lab)으로 S&P 100 에서 단기 모델이 비용·세금을 내고도 버는지 잰다.

  자료: data/us_ohlcv.csv (collect_us.py, 배당·분할 수정가) · data/fx_usdkrw.csv
  체결: t일 장 마감 뒤 판단 → t+1일 시가 (미국 장 = 한국 밤 23:30 / 22:30). 판단은 한국 저녁 daily 때 이미 끝난 미국 장 자료로.
  비용: 수수료 한쪽 0.1% + 미끄러짐 0.03% + 매도 SEC 0.003%. 거래세 없음.
  세금: 해외주식 양도차익 연 250만원 공제 뒤 22% — 연구에선 공제 없이 '해마다 이익의 22%'를 떼는 보수적 근사를 따로 보여 준다.
  환율: 원화로 본 성적 = 달러 성적 × 원/달러 변화 (달러를 들고 있는 동안의 환차손익 포함).
  유니버스: 날짜마다 최근 60일 평균 거래대금 상위 100 (S&P 100 구성 = 지금 살아남은 대형주 → 생존편향).
  수급 신호(외국인·기관)는 미국 자료가 없어 그 신호를 쓰는 모델은 뺀다.

  python src/us_lab.py   → data/short/us_lab.json · docs/us_lab.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import short_lab as L  # noqa: E402

US = "data/us_ohlcv.csv"
FX = "data/fx_usdkrw.csv"
OUT = "data/short/us_lab.json"
UI = "docs/us_lab.json"
BENCH = ("SPY", "QQQ")
US_TAX = 0.22
NO_FLOW = ("REVC_FLOW", "FOR20")          # 미국엔 수급 자료가 없다


def load_panel(min_date="2015-01-01"):
    """short_panel.load() 와 같은 모양의 P (미국판). 수급 행렬은 NaN, 시총 대신 60일 평균 거래대금."""
    o = pd.read_csv(US)
    bench = o[o["종목코드"].isin(BENCH)]
    o = o[~o["종목코드"].isin(BENCH)]
    o = o[o["날짜"] >= min_date]
    o["날짜"] = pd.to_datetime(o["날짜"])
    dates = pd.DatetimeIndex(sorted(o["날짜"].unique()))
    codes = sorted(o["종목코드"].unique())
    P = {"dates": dates, "codes": codes, "market": "us"}

    def piv(col):
        return (o.pivot_table(index="날짜", columns="종목코드", values=col, aggfunc="last")
                 .reindex(index=dates, columns=codes).to_numpy(dtype=float, copy=True))
    for k, col in [("O", "시가"), ("H", "고가"), ("L", "저가"), ("C", "종가"), ("V", "거래량")]:
        P[k] = piv(col)
    for k in ("O", "H", "L", "C"):
        P[k][~(P[k] > 0)] = np.nan
    P["VAL"] = P["C"] * P["V"]
    P["MC"] = pd.DataFrame(P["VAL"]).rolling(60, min_periods=20).mean().to_numpy(copy=True)
    P["breaks"] = np.zeros(P["C"].shape, bool)              # 수정가라 분할 끊김 없음 · 가격제한폭도 없음(±30% 넘는 날도 진짜)
    nan = np.full(P["C"].shape, np.nan)
    P["FOR"], P["INS"], P["IND"] = nan, nan.copy(), nan.copy()
    mc = np.where(np.isnan(P["C"]), np.nan, P["MC"])
    rank = np.argsort(np.argsort(-np.nan_to_num(mc, nan=-1), axis=1), axis=1)
    P["U"] = (rank < 100) & ~np.isnan(mc)
    P["Cf"] = pd.DataFrame(P["C"]).ffill().to_numpy(copy=True)
    bench["날짜"] = pd.to_datetime(bench["날짜"])
    P["bench"] = {b: bench[bench["종목코드"] == b].set_index("날짜")["종가"].reindex(dates).ffill().to_numpy(float)
                  for b in BENCH}
    return P


class USData(L.Data):
    cost_kind = "us"


def fx_series(dates):
    if not os.path.exists(FX):
        return None
    f = pd.read_csv(FX)
    f["날짜"] = pd.to_datetime(f["날짜"])
    return f.set_index("날짜")["환율"].reindex(dates).ffill().bfill().to_numpy(float)


def after_tax(E, dates, rate=US_TAX):
    """해마다 이익의 22% 를 그해 마지막 날 뗀다 (손실 난 해는 0, 이월 공제 없음 — 보수적)."""
    s = pd.Series(E, index=dates)
    out = s.copy()
    scale, prev_end = 1.0, s.iloc[0]
    for yr, g in s.groupby(s.index.year):
        gain = g.iloc[-1] / prev_end - 1
        out.loc[g.index] = g * scale
        if gain > 0:
            scale *= (1 + gain * (1 - rate)) / (1 + gain)
        prev_end = g.iloc[-1]
    return out.to_numpy()


def models():
    return [m for m in L.stock_models() if m["sel"] not in NO_FLOW]


def main():
    t0 = time.time()
    D = USData(load_panel())
    fx = fx_series(D.dates)
    print(f"미국 자료: {D.dates[0].date()} ~ {D.dates[-1].date()} · {D.N}종목 · 비용 왕복 {sum(L.costs('us')) * 100:.2f}%"
          + (f" · 환율 {fx[-1]:.0f}" if fx is not None else " · 환율 없음"))
    ms = models()
    recs = [L.run(D, m, record=True) for m in ms]
    R = np.c_[[L.rets(r["E"]) for r in recs]].T
    rows = []
    for m, r in zip(ms, recs):
        E = r["E"]
        rows.append({"id": m["id"], "desc": L.describe(m), "hold": m["hold"], "full": L.stats(E, D.dates),
                     "dev": L.stats(E, D.dates, b=L.DEV_END), "test": L.stats(E, D.dates, a="2023-01-01"),
                     "years": L.yearly(E, D.dates)})
    df = pd.DataFrame([{"hold": r["hold"], "sh": r["full"]["sharpe"], "cagr": r["full"]["cagr"],
                        "dev": r["dev"]["sharpe"], "test": r["test"]["sharpe"]} for r in rows])
    by_hold = {h: {"n": int(len(g)), "sharpe_med": round(float(g.sh.median()), 3), "positive": int((g.cagr > 0).sum()),
                   "dev_med": round(float(g.dev.median()), 3), "test_med": round(float(g.test.median()), 3)}
               for h, g in df.groupby("hold")}
    for h, x in by_hold.items():
        print(f"  {h:5s} 모델 {x['n']:3d}개 · 수익 {x['positive']:3d} · 샤프 중앙값 {x['sharpe_med']:.2f} (~2022 {x['dev_med']:.2f} / 2023~ {x['test_med']:.2f})")
    start = int(np.searchsorted(D.dates, pd.Timestamp("2016-01-04")))
    te = int(np.searchsorted(D.dates, pd.Timestamp("2023-01-03")))
    spy = D.P["bench"]["SPY"]
    spy_E = spy / spy[start]
    out = {}
    for key, (name, rule) in {"live": ("매주 · 최근 250일 샤프 1등 (느린 후보군) — 국내와 같은 규칙", L.LIVE_RULE),
                              "monthly": ("매달 · 최근 250일 샤프 1등", {**L.LIVE_RULE, "every": 21}),
                              "top3": ("매주 · 최근 250일 샤프 상위 3개 섞기", {**L.LIVE_RULE, "k": 3})}.items():
        E, sw, ch = L.apply_rule(D, ms, recs, R, rule, start)
        e = E / E[start]
        res = {"name": name, "switches": sw, "usd": L.stats(e[start:], D.dates[start:]),
               "test": L.stats(e, D.dates, a="2023-01-01"), "years": L.yearly(e[start:], D.dates[start:]),
               "p_vs_spy": L.boot_p(L.rets(e)[start:], L.rets(spy_E)[start:]),
               "p_vs_spy_test": L.boot_p(L.rets(e)[te:], L.rets(spy_E)[te:])}
        res["usd_after_tax"] = L.stats(after_tax(e, D.dates)[start:], D.dates[start:])
        if fx is not None:
            k = e * fx / fx[start]
            res["krw"] = L.stats(k[start:], D.dates[start:])
            res["krw_after_tax"] = L.stats(after_tax(k, D.dates)[start:], D.dates[start:])
        picks = pd.Series([ms[c]["id"] for c in ch[start:] if isinstance(c, int)]).value_counts().head(5)
        res["picks"] = {a: int(b) for a, b in picks.items()}
        res["curve"] = [round(float(x), 4) for x in e[start::5]]
        out[key] = res
        print(f"■ {name:44s} 교체 {sw:3d} · 달러 {json.dumps(res['usd'], ensure_ascii=False)} · 세후 연 {res['usd_after_tax']['cagr'] * 100:+.1f}%"
              + (f" · 원화 연 {res['krw']['cagr'] * 100:+.1f}%" if fx is not None else "") + f" · SPY 대비 P={res['p_vs_spy']}")
    b = {"SPY 보유": {"usd": L.stats(spy_E[start:], D.dates[start:]), "test": L.stats(spy_E, D.dates, a="2023-01-01"),
                      "years": L.yearly(spy_E[start:], D.dates[start:]), "usd_after_tax": L.stats(after_tax(spy_E, D.dates)[start:], D.dates[start:]),
                      "curve": [round(float(x), 4) for x in spy_E[start::5]]}}
    if fx is not None:
        k = spy_E * fx / fx[start]
        b["SPY 보유"]["krw"] = L.stats(k[start:], D.dates[start:])
    print(f"□ SPY 보유 {json.dumps(b['SPY 보유']['usd'], ensure_ascii=False)}")
    top = sorted(rows, key=lambda r: -(r["dev"] or {}).get("sharpe", -9))[:12]
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "period": [str(D.dates[0].date()), str(D.dates[-1].date())],
           "start": str(D.dates[start].date()), "n_stocks": D.N, "n_models": len(ms),
           "costs": {"roundtrip": round(sum(L.costs("us")), 5), "commission": L.COMM_US, "slip": L.SLIP["us"], "tax": US_TAX},
           "by_hold": by_hold, "rules": out, "bench": b, "top_models": top, "curve_dates": [str(d.date()) for d in D.dates[start::5]]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    from short_trader import clean
    res = clean(res)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(res, open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"끝 ({time.time() - t0:.0f}s) → {OUT}")
    return res


if __name__ == "__main__":
    main()
