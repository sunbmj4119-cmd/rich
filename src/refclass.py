"""
기준집단(reference class) 통계 -> data/refclass.json

왜 이걸 만드는가 — verify_thesis.py가 밝힌 사실 때문이다.
  기존 방식(그 종목의 과거 '점수 비슷했던 날'을 모아 승률을 내는 것)은
  34,221건 검증에서 **예측력이 0이었다**(예측승률↔실제 상관 -0.003).
  "승률 75%"라고 표시된 구간의 실제 승률이 48.9%였다. 즉 숫자가 노이즈였다.

  원인은 표본이 얇아서가 아니라 **기준집단이 잘못돼서**다.
  한 종목의 과거 15~25개 에피소드는 그 시기 시장이 무엇을 했는지를 잴 뿐,
  '점수 65점인 종목'에 대해 아무것도 말해주지 않는다.

그래서 기준집단을 바꾼다.
  (종합순위 밴드 × 시장국면) 칸마다 **100종목 전체·전 기간**을 모은다.
  이건 백테스트로 검증된 신호(순위)를 그대로 쓰는 것이고, 칸마다 표본이
  수천 건이라 통계가 안정적이다.

독립표본 계산 (중요)
  같은 날 서로 다른 종목의 30일 수익은 **독립이 아니다**(시장이 같이 움직인다).
  그래서 표본 수가 아니라 **겹치지 않는 '날짜 묶음' 수**를 독립 관측치로 센다.
  2000거래일이면 최대 66개다. 신뢰구간이 넓어 보이는 게 정상이고 정직하다.
"""
import os
import json

import numpy as np
import pandas as pd

import thesis
from regime import build_index, classify

SCORES = "data/scores.csv"
PRICES = "data/prices.csv"
OUT = "data/refclass.json"

FWD = 30
BANDS = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
MIN_CELL = 200        # 이보다 표본이 적은 칸은 국면을 무시한 밴드 통계로 폴백


def band_of(r):
    for lo, hi in BANDS:
        if lo <= r <= hi:
            return f"{lo}-{hi}"
    return None


def indep_dates(day_idx):
    """겹치지 않는 날짜 묶음 수. 같은 날의 여러 종목은 1개로 센다."""
    cnt, nxt = 0, -1
    for p in sorted(set(day_idx)):
        if p >= nxt:
            cnt += 1
            nxt = p + FWD
    return cnt


def stats(rets, days, base_win):
    if len(rets) < 30:
        return None
    st = thesis.probability(list(rets), indep_dates(days), base_win)
    if st:
        st["n_days"] = int(len(set(days)))
    return st


def main():
    s = pd.read_csv(SCORES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종가", "종합점수"])
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])
    dates = np.array(sorted(s["날짜"].unique()))
    dord = {d: i for i, d in enumerate(dates)}

    px = s.pivot_table(index="날짜", columns="종목코드", values="종가").reindex(dates)
    sc = s.pivot_table(index="날짜", columns="종목코드", values="종합점수").reindex(dates)
    fwd = px.shift(-FWD) / px - 1
    rank = sc.rank(axis=1, ascending=False, method="first")

    # 시장국면 (그날 정보만)
    pxl = pd.read_csv(PRICES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종가"])
    pxl["종목코드"] = pxl["종목코드"].str.zfill(6)
    pxl["날짜"] = pd.to_datetime(pxl["날짜"])
    rdf, _, _, _ = build_index(pxl)
    reg = rdf.apply(classify, axis=1).reindex(dates)

    # 롱포맷으로 펼쳐 칸을 채운다
    fv, rv = fwd.values, rank.values
    rows = []
    for j in range(len(dates)):
        rg = reg.iloc[j]
        if not isinstance(rg, str):
            continue
        fr, rr = fv[j], rv[j]
        ok = ~np.isnan(fr) & ~np.isnan(rr)
        for ci in np.flatnonzero(ok):
            b = band_of(int(rr[ci]))
            if b:
                rows.append((j, b, rg, float(fr[ci])))
    d = pd.DataFrame(rows, columns=["j", "band", "regime", "fwd"])
    base_win = round(float((d["fwd"] > 0).mean() * 100), 1)

    cells, by_band, by_regime = {}, {}, {}
    for b, g in d.groupby("band"):
        by_band[b] = stats(g["fwd"].values, g["j"].values, base_win)
    for rg, g in d.groupby("regime"):
        by_regime[rg] = stats(g["fwd"].values, g["j"].values, base_win)
    for (b, rg), g in d.groupby(["band", "regime"]):
        if len(g) < MIN_CELL:
            continue
        st = stats(g["fwd"].values, g["j"].values, base_win)
        if st:
            cells[f"{b}|{rg}"] = st

    overall = stats(d["fwd"].values, d["j"].values, base_win)
    out = {"asof": str(pd.Timestamp(dates[-1]).date()), "fwd": FWD,
           "bands": [f"{lo}-{hi}" for lo, hi in BANDS],
           "base_win": base_win, "overall": overall,
           "cells": cells, "by_band": by_band, "by_regime": by_regime,
           "n": int(len(d)), "n_days": int(d["j"].nunique())}
    os.makedirs("data", exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, allow_nan=False)

    print(f"기준집단: 표본 {out['n']:,}건 / {out['n_days']:,}거래일 · 전체 승률 {base_win}%")
    print(f"\n{'순위밴드':>10} {'승률':>7} {'95%구간':>15} {'평균':>7} {'손절률':>7} {'표본':>8} {'독립':>5}")
    for b in out["bands"]:
        st = by_band.get(b)
        if not st:
            continue
        print(f"{b+'위':>10} {st['win']:6.1f}% {st['win_lo']:5.1f}~{st['win_hi']:5.1f}% "
              f"{st['ev_raw']:+6.2f}% {st['p_stop']:6.1f}% {st['n']:8,} {st['eff_n']:5.0f}")
    print(f"\n{'국면':>10} {'승률':>7} {'평균':>7} {'손절률':>7} {'표본':>8}")
    for rg, st in sorted(by_regime.items(), key=lambda x: -(x[1]["win"] if x[1] else 0)):
        if st:
            print(f"{rg:>10} {st['win']:6.1f}% {st['ev_raw']:+6.2f}% {st['p_stop']:6.1f}% {st['n']:8,}")
    print(f"\n칸(밴드×국면) {len(cells)}개 생성 (표본 {MIN_CELL}건 미만 칸은 밴드 통계로 폴백)")


if __name__ == "__main__":
    main()
