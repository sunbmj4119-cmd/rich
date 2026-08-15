"""
팩터 연구소 — 새 팩터 후보의 예측력을 '연도별로' 재고, 안정적인 것만 걸러낸다.
-> data/factor_lab.json

왜 연도별인가 (이 저장소가 한 번 크게 데인 부분)
  전체 구간 백테스트만 보면 모멘텀이 최고로 보였다(test 연 +84.8%).
  그런데 연도별 IC를 내보니 -0.010으로 **음수**였다. 2024~25년에만 통한 것이었고
  그 구간이 test를 지배했을 뿐이다. 전체 평균 하나로는 이걸 절대 못 잡는다.
  그래서 여기서는 전체 IC와 함께 **양수였던 해의 수**를 항상 같이 본다.

채택 기준 (셋 다 만족해야 후보)
  1) 전체 IC > 0.015           — 실전에서 의미 있는 최소 수준
  2) 양수 연도 ≥ 전체의 2/3    — 특정 구간에서만 통한 게 아님
  3) 기존 종합점수와 |상관| < 0.6 — 이미 있는 정보의 재탕이 아님

미래참조 차단
  · 재무는 '사용가능일' 기준 as-of(backward) 조인 — 그날 알 수 없는 값은 안 쓴다.
  · 가격·수급 지표는 t일까지의 값만으로 만든다.
"""
import os
import json

import numpy as np
import pandas as pd

PRICES = "data/prices.csv"
SCORES = "data/scores.csv"
FIN = "data/financials.csv"
FLOWS = "data/flows.csv"
MCAP = "data/marketcap.csv"
OHLCV = "data/ohlcv.csv"
OUT = "data/factor_lab.json"

FWD = 30          # 예측 지평(거래일)
STEP = 5          # 며칠마다 IC를 잴지
MIN_IC = 0.015
MIN_POS_RATIO = 2 / 3
MAX_CORR = 0.6


def xs_rank(df):
    """그날 종목 간 백분위 0~100 (score.py와 같은 표준화)"""
    return df.rank(axis=1, pct=True) * 100


def build_candidates(dates, px, fin_wide, flow, inst, indiv, cap, amt):
    """후보 팩터들. 값이 클수록 '좋다'는 방향으로 통일한다."""
    F = {}
    ret = px.pct_change()

    # ── 가격·변동성 계열 ─────────────────────────────────────
    vol60 = ret.rolling(60, min_periods=30).std()
    F["저변동성"] = -vol60                                   # 저변동성 이상현상
    F["52주고가근접"] = px / px.rolling(252, min_periods=120).max()   # George·Hwang(2004)
    F["단기반전"] = -px.pct_change(20)                        # 대형주 1개월 역방향
    mom12_1 = px.shift(21) / px.shift(252) - 1
    F["위험조정모멘텀"] = mom12_1 / (vol60 * np.sqrt(252) + 1e-9)
    F["가속모멘텀"] = px.pct_change(60) - px.pct_change(120)   # 추세가 빨라지는가
    F["최대낙폭회복"] = px / px.rolling(60, min_periods=30).max()

    # ── 이격률 (이동평균에서 얼마나 벌어졌나) ─────────────────
    # 한국에서 오래 쓰이는 지표다. 가설은 평균회귀 — 이평선 위로 많이 뜬 종목은
    # 되돌아오고, 아래로 처진 종목은 붙으러 올라온다. 그래서 부호를 뒤집어
    # '이평선 아래일수록 높은 점수'로 넣고, IC 부호로 가설이 맞는지 본다.
    # (부호가 음수로 나오면 평균회귀가 아니라 추세추종이 맞다는 뜻이다.)
    for n in (20, 60, 120):
        ma = px.rolling(n, min_periods=max(10, n // 2)).mean()
        disp = px / ma - 1
        F[f"이격도{n}역"] = -disp
        # 종목마다 평소 벌어지는 폭이 다르다. 변동성 큰 종목은 늘 ±10%씩 벌어지고
        # 우량주는 ±3%도 크다. 그래서 '그 종목 기준으로 지금 유별난가'를 따로 본다.
        z = (disp - disp.rolling(252, min_periods=120).mean()) \
            / (disp.rolling(252, min_periods=120).std() + 1e-9)
        F[f"이격도{n}역Z"] = -z

    # ── 수급 계열 (기관·개인은 지금 아예 안 쓰고 있다) ─────────
    if flow is not None and cap is not None:
        F["기관수급20"] = flow_norm(inst, cap, 20)
        F["개인역방향"] = -flow_norm(indiv, cap, 20)          # 개인 순매수는 역지표 가설
        F["외국인수급60"] = flow_norm(flow, cap, 60)          # 기존은 20일 — 더 긴 창
        F["수급합치"] = (flow_norm(flow, cap, 20).rank(axis=1, pct=True)
                        + flow_norm(inst, cap, 20).rank(axis=1, pct=True))
    if amt is not None:
        F["거래대금증가"] = (amt.rolling(20, min_periods=10).mean()
                          / (amt.rolling(120, min_periods=60).mean() + 1e-9))

    # ── 재무 계열 (사용가능일 as-of 조인된 값) ────────────────
    if fin_wide:
        g = fin_wide
        if "영업이익" in g and "자산총계" in g:
            F["총이익성"] = g["영업이익"] / g["자산총계"]      # Novy-Marx 근사
        if "부채비율" in g:
            F["저부채"] = -g["부채비율"]
        if "매출액" in g and cap is not None:
            F["저PSR"] = -(cap / (g["매출액"] * 4 + 1e-9))     # 분기매출 연율화
        if "ROE" in g:
            F["ROE개선"] = g["ROE"] - g["ROE"].shift(60)      # 60거래일 전 대비
        if "EPS" in g:
            F["EPS성장"] = g["EPS"] / g["EPS"].shift(252) - 1
    return F


def flow_norm(f, cap, win):
    return f.rolling(win, min_periods=max(5, win // 2)).sum() / (cap + 1e-9)


def asof_financials(dates, codes):
    """재무를 '사용가능일' 기준으로 각 거래일에 backward 조인 → {항목: DataFrame}"""
    if not os.path.exists(FIN):
        return {}
    f = pd.read_csv(FIN, dtype={"종목코드": str})
    f["종목코드"] = f["종목코드"].str.zfill(6)
    f["사용가능일"] = pd.to_datetime(f["사용가능일"], errors="coerce")
    f = f.dropna(subset=["사용가능일"]).sort_values("사용가능일")
    cols = ["매출액", "영업이익", "당기순이익", "자산총계", "부채비율", "ROE", "EPS"]
    cols = [c for c in cols if c in f.columns]
    out = {c: pd.DataFrame(index=dates, columns=codes, dtype=float) for c in cols}
    base = pd.DataFrame({"날짜": dates})
    for code in codes:
        sub = f[f["종목코드"] == code]
        if not len(sub):
            continue
        m = pd.merge_asof(base, sub[["사용가능일"] + cols],
                          left_on="날짜", right_on="사용가능일", direction="backward")
        for c in cols:
            out[c][code] = m[c].values
    return out


def yearly_ic(fac, fwd, dates):
    """(전체 IC, 연도별 IC, 양수 연도수, 총 연도수)"""
    ics, ds = [], []
    for t in range(0, len(dates) - FWD, STEP):
        a, b = fac.iloc[t], fwd.iloc[t]
        m = a.notna() & b.notna()
        if m.sum() < 30:
            continue
        ics.append(a[m].rank().corr(b[m].rank()))
        ds.append(dates[t])
    if len(ics) < 50:
        return None
    s = pd.Series(ics, index=pd.DatetimeIndex(ds)).dropna()
    yr = s.groupby(s.index.year).mean()
    return float(s.mean()), {int(k): round(float(v), 4) for k, v in yr.items()}, \
        int((yr > 0).sum()), int(len(yr))


def main():
    sc = pd.read_csv(SCORES, dtype={"종목코드": str})
    sc["종목코드"] = sc["종목코드"].str.zfill(6)
    sc["날짜"] = pd.to_datetime(sc["날짜"])
    dates = np.array(sorted(sc["날짜"].unique()))
    px = sc.pivot_table(index="날짜", columns="종목코드", values="종가").reindex(dates)
    codes = list(px.columns)
    fwd = px.shift(-FWD) / px - 1
    total = sc.pivot_table(index="날짜", columns="종목코드", values="종합점수").reindex(dates)

    def load(path, val):
        if not os.path.exists(path):
            return None
        d = pd.read_csv(path, dtype={"종목코드": str})
        d["종목코드"] = d["종목코드"].str.zfill(6)
        d["날짜"] = pd.to_datetime(d["날짜"])
        return d.pivot_table(index="날짜", columns="종목코드", values=val)\
                .reindex(dates).reindex(columns=codes)

    flow = load(FLOWS, "외국인순매수")
    inst = load(FLOWS, "기관순매수")
    indiv = load(FLOWS, "개인순매수")
    cap = load(MCAP, "시가총액")
    if cap is not None:
        cap = cap.ffill()
    amt = load(OHLCV, "거래대금")

    print("재무 as-of 조인 중...")
    finw = asof_financials(dates, codes)
    print(f"  {len(finw)}개 항목")

    cands = build_candidates(dates, px, finw, flow, inst, indiv, cap, amt)
    # 기존 팩터도 같이 재서 비교 기준으로 삼는다
    for k, nm in [("s_value", "[기존]가치"), ("s_profit", "[기존]수익성"),
                  ("s_flow", "[기존]수급"), ("s_grow", "[기존]성장"),
                  ("s_mom", "[기존]모멘텀")]:
        if k in sc.columns:
            cands[nm] = sc.pivot_table(index="날짜", columns="종목코드", values=k)\
                          .reindex(dates).reindex(columns=codes)
    cands["[기존]종합점수"] = total

    rows = []
    for nm, f in cands.items():
        f = f.reindex(columns=codes)
        r = yearly_ic(f, fwd, dates)
        if r is None:
            print(f"  {nm}: 표본부족")
            continue
        ic, yr, pos, ny = r
        # 기존 종합점수와의 독립성 (같은 정보의 재탕인지)
        fr = f.rank(axis=1, pct=True)
        tr = total.rank(axis=1, pct=True)
        corr = float(fr.corrwith(tr, axis=1).mean())
        rows.append({"name": nm, "ic": round(ic, 4), "pos": pos, "ny": ny,
                     "pos_ratio": round(pos / ny, 2), "corr": round(corr, 3),
                     "yearly": yr,
                     "pass": bool(ic > MIN_IC and pos / ny >= MIN_POS_RATIO
                                  and abs(corr) < MAX_CORR and not nm.startswith("[기존]"))})
    rows.sort(key=lambda x: -x["ic"])

    years = sorted({y for r in rows for y in r["yearly"]})
    print(f"\n{'팩터':<16}{'전체IC':>8}{'양수년':>8}{'종합상관':>9}  " +
          " ".join(f"{y%100:>5}" for y in years) + "  채택")
    print("-" * (42 + 6 * len(years)))
    for r in rows:
        line = (f"{r['name']:<16}{r['ic']:+8.4f}{r['pos']:>4}/{r['ny']:<3}{r['corr']:+9.2f}  " +
                " ".join(f"{r['yearly'].get(y, float('nan')):+5.2f}".replace("+0.", " .").replace("-0.", "-.")
                         for y in years))
        print(line + ("  ✅" if r["pass"] else ""))

    passed = [r for r in rows if r["pass"]]
    print(f"\n채택 후보 {len(passed)}개 "
          f"(기준: 전체IC>{MIN_IC} · 양수년≥{MIN_POS_RATIO:.0%} · |종합상관|<{MAX_CORR})")
    for r in passed:
        print(f"  ✅ {r['name']:<16} IC {r['ic']:+.4f} · 양수 {r['pos']}/{r['ny']}년 · 상관 {r['corr']:+.2f}")
    if not passed:
        print("  없음 — 기존 팩터 조합을 이길 새 정보가 이 데이터엔 없다는 뜻.")

    os.makedirs("data", exist_ok=True)
    json.dump({"fwd": FWD, "min_ic": MIN_IC, "rows": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, allow_nan=False)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
