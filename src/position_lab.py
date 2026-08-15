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

사라진 종목도 넣는다
  data/prices_delisted.csv가 있으면 상장폐지 종목의 가격 경로도 표본에 넣는다.
  이게 없으면 "많이 빠진 뒤 반등확률이 높다"는 통계가 낙관 쪽으로 치우친다 —
  끝내 회복 못 한 종목은 살아있는 표본에 남아있지 않기 때문이다.

  주의할 함정: 상장폐지 뒤에는 가격이 없어서 그냥 두면 **마지막 30일이 NaN으로
  빠져버린다**. 생존편향을 고치려고 받은 데이터가 정작 가장 중요한 구간에서
  사라지는 것이다. 그래서 마지막 체결가를 30거래일까지 끌고 간다 —
  정리매매까지의 급락은 데이터에 들어 있고, 그 뒤는 '마지막 가격에 팔았다'로 본다.

  **다만 섞지는 않는다.** 사라진 76종목은 대부분 소형주와 우선주라, 표본 수대로
  합치면 대형주 판단에 소형주의 부도율을 밀어넣는 꼴이 된다. 편향 하나를
  다른 편향으로 바꾸는 것뿐이다. 그래서 본표는 생존 종목으로 내고,
  사라진 종목은 **따로 재서 그 차이를 화면에 그대로 보여준다**.
  실제 값은 둘 사이 어딘가다 — 어느 쪽 하나를 진실이라고 우기지 않는다.

한계
  · 겹치는 관측이라 독립표본은 훨씬 적다 → 날짜 묶음 수로 따로 센다.
  · 상장폐지 종목을 넣어도 코스피 주권만이고, 거래정지 중 장기 방치된 구간은
    가격이 없어 반영되지 않는다. 하락 꼬리는 여전히 실제보다 얇다.
"""
import os
import json

import numpy as np
import pandas as pd

SCORES = "data/scores.csv"
# 이격률 구간 — "20일선에서 얼마나 벌어졌나". 되밀릴 확률을 가르는 힘이 확인된 값이다.
DSP_BINS = [-99, -5, -2, 2, 5, 10, 99]
DSP_LABS = ["20일선 -5%↓", "-5~-2%", "-2~+2%", "+2~+5%", "+5~+10%", "+10%↑"]
PRICES = "data/prices.csv"
DPRICES = "data/prices_delisted.csv"
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


def load_matrix():
    """가격 행렬과 순위 행렬. 상장폐지 종목이 있으면 가격에만 합친다.

    순위는 scores.csv에만 있으므로 사라진 종목은 순위밴드 표에서 자동으로 빠진다.
    가격 경로 통계(수익률·고점대비 칸)에는 들어간다 — 그게 넣는 이유다.
    """
    want = ["날짜", "종목코드", "종가", "종합점수", "이격도20"]
    head = pd.read_csv(SCORES, nrows=0).columns
    s = pd.read_csv(SCORES, dtype={"종목코드": str}, usecols=[c for c in want if c in head])
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])
    live = s.pivot_table(index="날짜", columns="종목코드", values="종가")
    rk = (s.pivot_table(index="날짜", columns="종목코드", values="종합점수")
            .rank(axis=1, ascending=False, method="first"))
    dsp = (s.pivot_table(index="날짜", columns="종목코드", values="이격도20")
           if "이격도20" in s.columns else None)

    dead = None
    if os.path.exists(DPRICES):
        d = pd.read_csv(DPRICES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종가"])
        d["종목코드"] = d["종목코드"].str.zfill(6)
        d["날짜"] = pd.to_datetime(d["날짜"])
        d = d[~d["종목코드"].isin(live.columns)]        # 살아있는 쪽이 우선
        if len(d):
            dead = d.pivot_table(index="날짜", columns="종목코드", values="종가")

    if dead is None:
        px = live
        print(f"가격 {len(px)}일 × {px.shape[1]}종목 (생존 종목만) — 포지션 상태 전수 조사")
    else:
        px = live.join(dead, how="outer").sort_index()
        # 상장폐지 뒤 30거래일까지 마지막 체결가를 끌고 간다.
        # 안 그러면 사라진 종목의 마지막 30일이 NaN이 되어 통째로 빠진다 —
        # 하필 가장 크게 빠진 구간이 빠지므로 편향을 고치려다 되레 키우게 된다.
        px[dead.columns] = px[dead.columns].ffill(limit=FWD)
        px = px.loc[live.index.min():]                 # 점수 기간 밖은 버린다
        print(f"가격 {len(px)}일 × {px.shape[1]}종목 "
              f"(생존 {live.shape[1]} + 상장폐지 {dead.shape[1]}) — 포지션 상태 전수 조사")

    rk = rk.reindex(index=px.index, columns=px.columns)
    dsp = (dsp.reindex(index=px.index, columns=px.columns).values
           if dsp is not None else np.full(px.shape, np.nan))
    n_dead = 0 if dead is None else dead.shape[1]
    return px.index.values, px.values, rk.values, dsp, live.shape[1], n_dead


def main():
    dates, px, rank, dspm, n_live, n_dead = load_matrix()
    nD, nS = px.shape

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

    # 순위가 통째로 비어 있는 열 = 상장폐지 종목 (scores.csv에 없으므로)
    dead_col = np.isnan(rank).all(axis=0)
    cols = ["ret", "dd", "fwd", "lo", "hi", "day", "rank", "hold", "dead", "dsp"]
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
            recs["dead"].append(dead_col[idx])
            recs["dsp"].append(dspm[e + h][idx].astype(np.float32))
    full = pd.DataFrame({k: np.concatenate(v) for k, v in recs.items()})
    full["rb"] = pd.cut(full["ret"], RET_BINS, labels=RET_LABS)
    full["db"] = pd.cut(full["dd"], DD_BINS, labels=DD_LABS)
    # 경로로 판정한다 — 손절도 익절도 도중에 닿기만 하면 체결된다
    full["hit_stop"] = full["lo"] <= STOP
    full["hit_tp"] = full["hi"] >= TP
    # 지금 팔았을 때 대비 — 더 밀리면 얼마까지, 더 오르면 얼마까지
    full["give"] = (1 + full["lo"]) / (1 + full["ret"]) - 1
    full["gain"] = (1 + full["hi"]) / (1 + full["ret"]) - 1
    # 진입가가 아니라 '지금 가격' 기준 — 팔지 말지는 지금 가격에서 갈린다
    full["sb"] = pd.cut(full["dsp"], DSP_BINS, labels=DSP_LABS)
    full["give5"], full["give10"] = full["give"] <= -0.05, full["give"] <= -0.10
    full["gain5"], full["gain10"] = full["gain"] >= 0.05, full["gain"] >= 0.10

    # 본표는 생존 종목만. 사라진 종목은 따로 재서 차이를 보여준다 (이유는 문서 앞부분에)
    d = full[~full["dead"]].reset_index(drop=True)
    dead = full[full["dead"]].reset_index(drop=True)
    print(f"  포지션-일 표본 {len(d):,}건 (생존)"
          + (f" · {len(dead):,}건 (상장폐지, 별도 집계)" if len(dead) else ""))

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

    def table(key, src=None):
        out = {}
        for k, g in (d if src is None else src).groupby(key, observed=True):
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
    # 이격률 — 종목 선택에는 값어치가 없었지만(strategy_lab 6-b) '지금 되밀릴 확률'은 가른다
    by_disp = table("sb")
    overall = stats(d)
    # 사라진 종목 — 본표에는 안 섞고, 얼마나 다른지만 잰다.
    # 이 차이가 곧 '생존편향의 크기'다.
    dead_block = None
    if len(dead) >= MIN_CELL:
        do = stats(dead)
        dead_block = {"overall": do, "by_ret": table("rb", dead),
                      "n_stocks": n_dead,
                      "up_gap": round(do["up"] - overall["up"], 1),
                      "ev_gap": round(do["ev"] - overall["ev"], 2)}

    out = {"fwd": FWD, "stop": STOP, "tp": TP, "max_hold": MAX_HOLD,
           "n_live": n_live, "n_dead": n_dead,
           "ret_labels": RET_LABS, "dd_labels": DD_LABS,
           "ret_bins": RET_BINS, "dd_bins": DD_BINS,
           "overall": overall, "cells": cells, "rank_cells": rcells,
           "by_ret": by_ret, "by_dd": by_dd, "by_rank": by_rank,
           "by_hold": by_hold, "by_disp": by_disp,
           "dsp_labels": DSP_LABS, "dsp_bins": DSP_BINS, "dead": dead_block}
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
    dump("20일선 이격률", by_disp, DSP_LABS)
    print(f"\n{'갓 산 자리(0~+5%)':<16}{'상승':>6}{'기대':>8}{'손절밟음':>10}{'익절밟음':>10}{'표본':>10}")
    for rk in ("1-10", "11-20", "21-100"):
        st = rcells.get(f"{rk}|0~+5%")
        if st:
            print(f"{'추천 ' + rk + '위':<16}{st['up']:5.1f}%{st['ev']:+7.2f}%"
                  f"{st['hit_stop']:9.1f}%{st['hit_tp']:9.1f}%{st['n']:10,}")
    if dead_block:
        do = dead_block["overall"]
        print(f"\n■ 생존편향의 크기 — 사라진 {n_dead}종목만 따로 (본표에는 섞지 않음)")
        print(f"  전체:   생존 상승 {overall['up']}% / 상폐 {do['up']}% "
              f"({dead_block['up_gap']:+.1f}%p) · "
              f"기대 {overall['ev']:+.2f}% / {do['ev']:+.2f}% ({dead_block['ev_gap']:+.2f}%p)")
        print(f"  {'구간':<12}{'생존 상승':>10}{'상폐 상승':>10}{'차이':>8}{'상폐 표본':>11}")
        for lab in RET_LABS:
            a, b = by_ret.get(lab), dead_block["by_ret"].get(lab)
            if a and b:
                print(f"  {lab:<12}{a['up']:9.1f}%{b['up']:9.1f}%"
                      f"{b['up'] - a['up']:+7.1f}%p{b['n']:11,}")
    print(f"\n칸(수익률×고점대비) {len(cells)}개 · 저장 {OUT}")


if __name__ == "__main__":
    main()
