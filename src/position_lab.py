"""
포지션 상태별 결과 표 -> data/position_lab.json

무엇에 답하나
  "지금 +18% 수익인데, 고점 대비로는 -4% 밀렸다. 팔까 들고갈까?"
  이 질문에 감이 아니라 **과거 같은 상태에서 실제로 무슨 일이 있었는지**로 답한다.

만드는 법 (전략과 무관하게, 가격 경로만으로)
  모든 (종목, 진입일) 쌍을 가정하고 보유 1~60거래일 동안의 상태를 기록한다.
    상태 = (현재 수익률 구간, 고점 대비 낙폭 구간)
  그 상태에서 **앞으로 30거래일** 어떻게 됐는지를 모은다.
    · 오를 확률 / 기대수익 (지금 팔았을 때 대비)
    · 가는 길에 -10% 손절선을 **밟을** 확률
    · 가는 길에 +25% 익절선을 **밟을** 확률
    · 더 올랐다면 최고 얼마까지 갔나 / 더 밀렸다면 최저 얼마까지 갔나

  '밟을 확률'은 30일 뒤 종가가 아니라 경로 전체로 센다.
  손절과 익절은 도중에 닿기만 해도 체결되기 때문이다.

왜 전략 시뮬레이션이 아니라 전수 조사인가
  전략이 실제로 잡은 포지션만 쓰면 표본이 1만 건 남짓이라 칸이 비어버린다.
  가격 경로만으로 만들면 수백만 건이 되어 칸마다 통계가 안정적이다.
  대신 '이 전략이 고른 종목'이라는 조건은 빠지므로, 순위밴드를 따로 붙여 보정한다.

한계
  · 겹치는 관측이라 독립표본은 훨씬 적다 → 날짜 묶음 수로 따로 센다.
  · 생존 종목만 있는 표본이다(상장폐지 미포함). 하락 꼬리가 실제보다 얇다.
"""
import os
import json

import numpy as np
import pandas as pd

SCORES = "data/scores.csv"
OUT = "data/position_lab.json"

FWD = 30            # 앞으로 볼 거래일
MAX_HOLD = 60       # 진입 후 이만큼까지의 상태를 기록
ENTRY_STEP = 3      # 진입일 샘플링 간격
STOP = -0.10        # 손절선 (signal.py와 동일)
TP = 0.25           # 부분익절선 (signal.py와 동일)
MIN_CELL = 300      # 이보다 적은 칸은 통계를 내지 않는다

# 현재 수익률 구간 — 익절·손절 판단이 갈리는 지점으로 끊는다
RET_BINS = [-1.0, -0.10, -0.05, 0.0, 0.05, 0.15, 0.25, 0.40, 9.0]
RET_LABS = ["-10%↓", "-10~-5%", "-5~0%", "0~+5%", "+5~15%", "+15~25%", "+25~40%", "+40%↑"]
# 고점 대비 낙폭 — '얼마나 뱉었나'
DD_BINS = [-1.0, -0.10, -0.05, -0.02, 0.0]
DD_LABS = ["고점-10%↓", "고점-10~-5%", "고점-5~-2%", "고점근처(-2~0%)"]


def main():
    s = pd.read_csv(SCORES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종가", "종합점수"])
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])
    dates = np.array(sorted(s["날짜"].unique()))
    px = s.pivot_table(index="날짜", columns="종목코드", values="종가").reindex(dates).values
    rank = (s.pivot_table(index="날짜", columns="종목코드", values="종합점수")
              .reindex(dates).rank(axis=1, ascending=False, method="first").values)
    nD, nS = px.shape
    print(f"가격 {nD}일 × {nS}종목 — 포지션 상태 전수 조사")

    # 앞으로 30일 동안의 최저가·최고가를 미리 굴려둔다 (경로로 손절/익절 체결을 판정하려고)
    fmin = np.full_like(px, np.nan)
    fmax = np.full_like(px, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        for t in range(nD - 1):
            w = px[t + 1:t + 1 + FWD]
            fmin[t] = np.nanmin(w, axis=0)
            fmax[t] = np.nanmax(w, axis=0)

    cols = ["ret", "dd", "fwd", "lo", "hi", "day", "rank", "hold"]
    recs = {k: [] for k in cols}
    for e in range(0, nD - FWD - 5, ENTRY_STEP):
        ent = px[e]                                   # 진입가
        top = min(e + MAX_HOLD, nD - FWD)
        if top <= e + 1:
            continue
        seg = px[e:top]                               # 보유 구간 가격
        peak = np.maximum.accumulate(seg, axis=0)     # 진입 후 고점
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = seg / ent - 1                       # 현재 수익률
            dd = seg / peak - 1                       # 고점 대비
            fwd = px[e + FWD:top + FWD] / seg - 1     # 여기서 앞으로 30일 (종가 기준)
            lo = fmin[e:top] / ent - 1                # 앞으로 30일 최저 (진입가 기준)
            hi = fmax[e:top] / ent - 1                # 앞으로 30일 최고 (진입가 기준)
        n = min(len(ret), len(fwd))
        for h in range(1, n):                         # h=보유일
            ok = np.isfinite(ret[h]) & np.isfinite(dd[h]) & np.isfinite(fwd[h]) & np.isfinite(lo[h])
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            for k, v in (("ret", ret), ("dd", dd), ("fwd", fwd), ("lo", lo), ("hi", hi)):
                recs[k].append(v[h][idx].astype(np.float32))
            recs["rank"].append(rank[e + h][idx].astype(np.float32))
            recs["day"].append(np.full(idx.size, e + h, np.int32))
            recs["hold"].append(np.full(idx.size, h, np.int16))
    d = pd.DataFrame({k: np.concatenate(v) for k, v in recs.items()})
    print(f"  포지션-일 표본 {len(d):,}건")

    d["rb"] = pd.cut(d["ret"], RET_BINS, labels=RET_LABS)
    d["db"] = pd.cut(d["dd"], DD_BINS, labels=DD_LABS)
    # 경로로 판정한다 — 손절도 익절도 도중에 닿기만 하면 체결된다
    d["hit_stop"] = d["lo"] <= STOP
    d["hit_tp"] = d["hi"] >= TP
    # 지금 팔았을 때 대비 — 더 밀리면 얼마까지, 더 오르면 얼마까지
    d["give"] = (1 + d["lo"]) / (1 + d["ret"]) - 1
    d["gain"] = (1 + d["hi"]) / (1 + d["ret"]) - 1
    # 진입가가 아니라 '지금 가격' 기준 — 팔지 말지는 지금 가격에서 갈린다
    d["give5"], d["give10"] = d["give"] <= -0.05, d["give"] <= -0.10
    d["gain5"], d["gain10"] = d["gain"] >= 0.05, d["gain"] >= 0.10

    def indep(days):
        """겹치지 않는 날짜 묶음 수 — 신뢰구간용"""
        cnt, nxt = 0, -1
        for p in sorted(set(days)):
            if p >= nxt:
                cnt += 1
                nxt = p + FWD
        return cnt

    def stats(g):
        if len(g) < MIN_CELL:
            return None
        f = g["fwd"].values
        return {
            "n": int(len(g)), "indep": indep(g["day"].values),
            "up": round(float((f > 0).mean() * 100), 1),
            "ev": round(float(f.mean() * 100), 2),
            "med": round(float(np.median(f) * 100), 2),
            "p25": round(float(np.percentile(f, 25) * 100), 1),
            "p75": round(float(np.percentile(f, 75) * 100), 1),
            "hit_stop": round(float(g["hit_stop"].mean() * 100), 1),
            "hit_tp": round(float(g["hit_tp"].mean() * 100), 1),
            "give": round(float(g["give"].mean() * 100), 2),
            "gain": round(float(g["gain"].mean() * 100), 2),
            "give5": round(float(g["give5"].mean() * 100), 1),
            "give10": round(float(g["give10"].mean() * 100), 1),
            "gain5": round(float(g["gain5"].mean() * 100), 1),
            "gain10": round(float(g["gain10"].mean() * 100), 1),
        }

    def table(key):
        out = {}
        for k, g in d.groupby(key, observed=True):
            st = stats(g)
            if st:
                out[str(k)] = st
        return out

    cells = {}
    for (rb, db), g in d.groupby(["rb", "db"], observed=True):
        st = stats(g)
        if st:
            cells[f"{rb}|{db}"] = st
    by_ret, by_dd = table("rb"), table("db")
    # 순위밴드 보정 — 이 전략이 사는 상위권과 전체가 다른지
    d["rkb"] = pd.cut(d["rank"], [0, 10, 20, 100], labels=["1-10", "11-20", "21-100"])
    by_rank = table("rkb")
    # 순위밴드 × 수익률구간 — 신규매수 자리(갓 산 0~+5%)를 순위별로 갈라 보려고
    rcells = {}
    for (rk, rb), g in d.groupby(["rkb", "rb"], observed=True):
        st = stats(g)
        if st:
            rcells[f"{rk}|{rb}"] = st
    # 보유기간 — '얼마나 들고 있었나'가 결과를 바꾸는지
    d["hb"] = pd.cut(d["hold"], [0, 10, 20, 40, 60], labels=["1-10일", "11-20일", "21-40일", "41-60일"])
    by_hold = table("hb")
    overall = stats(d)

    out = {"fwd": FWD, "stop": STOP, "tp": TP, "max_hold": MAX_HOLD,
           "ret_labels": RET_LABS, "dd_labels": DD_LABS,
           "ret_bins": RET_BINS, "dd_bins": DD_BINS,
           "overall": overall, "cells": cells, "rank_cells": rcells,
           "by_ret": by_ret, "by_dd": by_dd, "by_rank": by_rank, "by_hold": by_hold}
    os.makedirs("data", exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, allow_nan=False)

    def dump(title, tab, labs=None):
        print(f"\n{title:<16}{'상승':>6}{'기대':>8}{'5%더밀림':>10}{'5%더오름':>10}"
              f"{'평균밀림':>9}{'평균오름':>9}{'표본':>10}")
        for lab in (labs or tab):
            st = tab.get(lab)
            if st:
                print(f"{lab:<16}{st['up']:5.1f}%{st['ev']:+7.2f}%{st['give5']:9.1f}%"
                      f"{st['gain5']:9.1f}%{st['give']:+8.2f}%{st['gain']:+8.2f}%{st['n']:10,}")

    print(f"\n전체: 상승 {overall['up']}% · 기대 {overall['ev']:+.2f}% · "
          f"손절밟음 {overall['hit_stop']}% · 익절밟음 {overall['hit_tp']}%")
    dump("현재 수익률", by_ret, RET_LABS)
    dump("고점 대비", by_dd, DD_LABS)
    dump("종합순위", by_rank)
    dump("보유기간", by_hold)
    print(f"\n{'갓 산 자리(0~+5%)':<16}{'상승':>6}{'기대':>8}{'손절밟음':>10}{'익절밟음':>10}{'표본':>10}")
    for rk in ("1-10", "11-20", "21-100"):
        st = rcells.get(f"{rk}|0~+5%")
        if st:
            print(f"{'추천 ' + rk + '위':<16}{st['up']:5.1f}%{st['ev']:+7.2f}%"
                  f"{st['hit_stop']:9.1f}%{st['hit_tp']:9.1f}%{st['n']:10,}")
    print(f"\n칸(수익률×고점대비) {len(cells)}개 · 저장 {OUT}")


if __name__ == "__main__":
    main()
