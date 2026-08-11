"""
판단 등급·확률의 정직성 검증 -> data/verify_thesis.json

대시보드가 "승률 62%, A등급"이라고 말할 때, 그 말이 과거에 실제로 맞았는지 확인한다.
셋을 답한다.

  Q1. 캘리브레이션 — "승률 60%"라 했을 때 실제로 60% 올랐나?
      (예측확률을 구간별로 묶어 실제 상승 비율과 비교. 어긋나면 숫자를 믿으면 안 된다.)
  Q2. 변별력 — A등급이 D등급보다 실제로 나았나?
      (등급별 30일 수익 분포. 순서가 뒤집히면 등급은 무의미하다.)
  Q3. 실전 — 이 등급대로 샀다면 어땠나?
      (비중첩 30일 리밸런싱. 기존 상위10 전략·시장과 비교.)

미래참조 차단 (이게 핵심)
  · 유사사례 표본은 t시점에 **결과가 이미 나온 것만** 쓴다 (p <= t-30).
  · 기준승률(base_win)도 t시점까지 확정된 표본으로만 계산한다.
  · 국면 판정은 원래 그날 정보만 쓰고, 국면별 승률은 t 이전 확정 표본만 쓴다.
  · 변동성 백분위는 t시점 과거 250일로만 계산한다.

라이브 등급과의 차이 (정직 고지)
  백테스트에 넣지 못한 반대논리 체크: 공매도 집중(2026-06부터만 존재),
  뉴스 악재(과거 데이터 없음), 하락장 취약(하락베타 계산 생략), 포트폴리오 쏠림(개인별).
  이들은 **점수를 깎기만** 하므로 백테스트 확신도는 라이브보다 약간 높게 나온다
  = 백테스트의 A등급은 라이브 A등급보다 **느슨한 기준**이다(결과는 보수적으로 해석).
"""
import os
import json

import numpy as np
import pandas as pd
import yaml

import thesis
from regime import build_index, classify

SCORES = "data/scores.csv"
PRICES = "data/prices.csv"
WEIGHTS = "config/weights.yaml"
OUT = "data/verify_thesis.json"

FWD = 30          # 예측 지평(거래일)
EVERY = 5         # 몇 거래일마다 평가할지 (전 종목 × 전 거래일은 과하다)
MIN_POOL = 10     # 유사사례 최소 표본 (build_data와 동일)
FINE_MIN = 15     # 타이밍분위까지 맞춘 정밀표본 최소치 (build_data와 동일)
WARMUP = 250      # 최소 이 정도 과거가 쌓인 뒤부터 평가
REF_BANDS = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100)]   # refclass.py와 동일
REF_MIN = 200     # 칸 최소 표본 (refclass.py와 동일)
FACTORS = [("s_value", "가치"), ("s_profit", "수익성"), ("s_grow", "성장"),
           ("s_flow", "수급"), ("s_mom", "모멘텀")]
WKEY = {"s_value": "value", "s_profit": "profit", "s_grow": "growth",
        "s_flow": "flow", "s_mom": "momentum"}


def indep_dates(day_idx):
    """겹치지 않는 '날짜 묶음' 수. 같은 날 여러 종목은 1개로 센다
    (같은 날 종목들의 30일 수익은 시장 요인 때문에 독립이 아니다)."""
    cnt, nxt = 0, -1
    for p in sorted(set(day_idx)):
        if p >= nxt:
            cnt += 1
            nxt = p + FWD
    return cnt


def greedy_independent(pos):
    """겹치지 않는 관측치 수 (build_data.independent_n과 같은 규칙)"""
    cnt = 0
    nxt = -1
    for p in pos:
        if p >= nxt:
            cnt += 1
            nxt = p + FWD
    return cnt


def main():
    w = yaml.safe_load(open(WEIGHTS, encoding="utf-8")).get("logic", {})
    weights = {k: w.get(v, 0) for k, v in WKEY.items()}

    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])
    s = s.sort_values(["종목코드", "날짜"])

    dates = np.array(sorted(s["날짜"].unique()))
    dord = {d: i for i, d in enumerate(dates)}
    nD = len(dates)

    # ── 종목별 배열 (거래일 = 위치) ─────────────────────────────
    piv = {}
    for col in ["종합점수", "종가"] + [k for k, _ in FACTORS]:
        piv[col] = s.pivot_table(index="날짜", columns="종목코드", values=col).reindex(dates)
    px = piv["종가"]
    codes = list(px.columns)

    fwd = px.shift(-FWD) / px - 1
    ret20 = px.pct_change(20)
    roll60 = px.rolling(60, min_periods=20).max()
    dd = px / roll60 - 1
    # 타이밍 분위 (당일 횡단면 5분위) — build_data와 동일 정의
    ddq = dd.rank(axis=1, method="first").apply(
        lambda r: pd.qcut(r, 5, labels=False, duplicates="drop") if r.notna().sum() >= 5 else r * np.nan,
        axis=1)

    # 횡단면 z → 타이밍 100점 환산 (build_data와 동일)
    def zrow(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0)
    timing_z = 0.6 * (-zrow(dd)) + 0.4 * (-zrow(ret20))
    tmin = timing_z.min(axis=1)
    timing100 = timing_z.sub(tmin, axis=0).div((timing_z.max(axis=1) - tmin) + 1e-9, axis=0) * 100

    score = piv["종합점수"]
    rank = score.rank(axis=1, ascending=False, method="first")

    # ── 시장 국면 (그날 정보만 사용) ────────────────────────────
    pxl = pd.read_csv(PRICES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종가"])
    pxl["종목코드"] = pxl["종목코드"].str.zfill(6)
    pxl["날짜"] = pd.to_datetime(pxl["날짜"])
    rdf, _, ret, mret = build_index(pxl)
    rdf["regime"] = rdf.apply(classify, axis=1)
    regime_at = rdf["regime"].reindex(dates)

    # 변동성 백분위 (t시점 과거 250일)
    vol = ret.reindex(dates).rolling(250, min_periods=120).std() * np.sqrt(252)
    vol_rank = vol.rank(axis=1, pct=True) * 100

    # ── 기준승률(base_win)의 시점별 값 ─────────────────────────
    # 표본 (종목, p)의 결과는 p+FWD에 확정된다 → 그 시점 이후로만 집계에 넣는다.
    known = np.zeros(nD)      # 각 날짜까지 확정된 표본 수
    wins = np.zeros(nD)
    fv = fwd.values
    for j in range(nD - FWD):
        row = fv[j]
        ok = ~np.isnan(row)
        known[j + FWD] = ok.sum()
        wins[j + FWD] = (row[ok] > 0).sum()
    cum_known = np.cumsum(known)
    cum_wins = np.cumsum(wins)
    base_win_at = np.where(cum_known > 0, cum_wins / np.maximum(cum_known, 1) * 100, 50.0)

    # ── 국면별 전략 승률 (t 이전 확정분만) ──────────────────────
    # 전략 = 그날 상위10 종목 동일가중의 30일 수익
    strat = pd.Series(np.nanmean(np.where(rank.values <= 10, fv, np.nan), axis=1), index=dates)
    reg_hist = {}     # 날짜인덱스 → {국면: 승률}
    acc = {}          # 국면 → [n, wins]
    reg_win_at = []
    for j in range(nD):
        if j >= FWD:
            k = j - FWD
            rg = regime_at.iloc[k]
            v = strat.iloc[k]
            if isinstance(rg, str) and not np.isnan(v):
                a = acc.setdefault(rg, [0, 0])
                a[0] += 1
                a[1] += 1 if v > 0 else 0
        cur = regime_at.iloc[j]
        a = acc.get(cur) if isinstance(cur, str) else None
        reg_win_at.append(a[1] / a[0] * 100 if a and a[0] >= FWD else None)

    # ── BuyFit (시스템의 실제 추천 순서) ────────────────────────
    zs = zrow(score)
    buyfit = zs + 0.25 * timing_z
    buyfit_pct = buyfit.rank(axis=1, pct=True) * 100     # 100 = 가장 추천

    # ── 기준집단(순위밴드 × 국면) 누적 통계 — t시점까지 확정분만 ──
    # 라이브(refclass.py)와 같은 칸 정의. 백테스트에선 시점마다 확장한다.
    cell_acc = {}      # (band, regime) -> [rets..., day_idx...]

    def band_of(r):
        for lo, hi in REF_BANDS:
            if lo <= r <= hi:
                return f"{lo}-{hi}"
        return None

    # ── 평가 루프 ───────────────────────────────────────────────
    recs = []
    sv = score.values
    ddqv = ddq.values
    t100 = timing100.values
    ddv = dd.values
    rkv = rank.values
    vrv = vol_rank.values
    bpv = buyfit_pct.values
    fac = {k: piv[k].values for k, _ in FACTORS}

    eval_idx = sorted(j for j in range(WARMUP, nD - FWD) if j % EVERY == 0)
    eval_set = set(eval_idx)
    cell_cache = {}
    overall_acc = [[], []]

    for j in range(nD - FWD):
        # (1) t시점에 결과가 확정된 표본(j-FWD 시점의 관측)을 누적에 넣는다
        k = j - FWD
        if k >= 0:
            rgk = regime_at.iloc[k]
            if isinstance(rgk, str):
                fr, rr = fv[k], rkv[k]
                ok = ~np.isnan(fr) & ~np.isnan(rr)
                for ci in np.flatnonzero(ok):
                    b = band_of(int(rr[ci]))
                    if b:
                        a = cell_acc.setdefault((b, rgk), [[], []])
                        a[0].append(float(fr[ci]))
                        a[1].append(k)
                        overall_acc[0].append(float(fr[ci]))
                        overall_acc[1].append(k)
            cell_cache.clear()

        if j not in eval_set:
            continue
        rg = regime_at.iloc[j]
        if not isinstance(rg, str) or len(overall_acc[0]) < 2000:
            continue
        ov_win = float(np.mean(np.array(overall_acc[0]) > 0) * 100)

        for ci, code in enumerate(codes):
            if np.isnan(sv[j, ci]) or np.isnan(fv[j, ci]) or np.isnan(rkv[j, ci]):
                continue
            b = band_of(int(rkv[j, ci]))
            if not b:
                continue
            key = (b, rg)
            if key not in cell_cache:
                a = cell_acc.get(key)
                if not a or len(a[0]) < REF_MIN:
                    # 국면별 표본이 얇으면 밴드 전체로 폴백
                    merged = [[], []]
                    for (bb, _rr), aa in cell_acc.items():
                        if bb == b:
                            merged[0].extend(aa[0])
                            merged[1].extend(aa[1])
                    a = merged
                if len(a[0]) < REF_MIN:
                    cell_cache[key] = None
                else:
                    cell_cache[key] = thesis.probability(
                        a[0], indep_dates(a[1]), round(ov_win, 1))
            pr = cell_cache[key]
            if pr is None:
                continue

            factors = [{"key": kk, "name": nm,
                        "val": float(fac[kk][j, ci]) if not np.isnan(fac[kk][j, ci]) else 50.0,
                        "w": weights.get(kk, 0)} for kk, nm in FACTORS]
            for f in factors:
                f["contrib"] = f["val"] * f["w"]
            ctx = {"name": code, "code": code, "score": float(sv[j, ci]), "rank": int(rkv[j, ci]),
                   "price": 0, "factors": factors,
                   "buyfit_pct": float(bpv[j, ci]) if not np.isnan(bpv[j, ci]) else None,
                   "timing": float(t100[j, ci]) if not np.isnan(t100[j, ci]) else 50.0,
                   "dd": round(float(ddv[j, ci]) * 100, 1) if not np.isnan(ddv[j, ci]) else None,
                   "analog": {"med": 0, "worst": 0}, "exp": None,
                   "tgt_lo": None, "tgt_hi": None, "stop_buy": None,
                   "short_pct": None, "short_rank": None,
                   "beta": None, "down_beta": None,
                   "vol": None, "vol_rank": (float(vrv[j, ci]) if not np.isnan(vrv[j, ci]) else None),
                   "news_flags": [], "sector_conc": None, "signal": "", "prob": pr}
            bear = thesis.bear_case(ctx)
            bull = thesis.bull_case(ctx)
            rw = reg_win_at[j]
            fake_regime = ({"current": {"name": rg}, "history": [{"name": rg, "win": rw}]}
                           if rw is not None else None)
            vd = thesis.verdict(ctx, bull, bear, fake_regime, round(ov_win, 1))

            recs.append((j, ci, vd["conf"], vd["grade"], pr["win"], pr["ev"], pr["eff_n"],
                         pr["edge_pp"], bear["bear_score"], float(fv[j, ci]), int(rkv[j, ci]),
                         rg, float(bpv[j, ci]) if not np.isnan(bpv[j, ci]) else np.nan))

    df = pd.DataFrame(recs, columns=["j", "ci", "conf", "grade", "pwin", "pev", "eff_n",
                                     "edge", "bear", "fwd", "rank", "regime", "bpct"])
    if not len(df):
        print("평가 표본 없음 — 데이터 부족")
        return
    df["hit"] = (df["fwd"] > 0).astype(int)
    df["date"] = [dates[j] for j in df["j"]]

    out = {"n_eval": int(len(df)), "every": EVERY, "fwd": FWD,
           "period": [str(pd.Timestamp(dates[WARMUP]).date()), str(pd.Timestamp(dates[-1]).date())],
           "base_win": round(float(df["hit"].mean() * 100), 1)}

    # ── Q1. 캘리브레이션 ────────────────────────────────────────
    bins = [0, 40, 45, 50, 55, 60, 65, 100]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = df[(df["pwin"] >= lo) & (df["pwin"] < hi)]
        if len(g) < 30:
            continue
        # 겹침 보정: 같은 종목의 인접 평가는 독립이 아니다 → 종목별 비중첩 표본만으로 오차범위
        indep = sum(greedy_independent(sorted(gg["j"].tolist()))
                    for _, gg in g.groupby("ci"))
        act = float(g["hit"].mean() * 100)
        ci_lo, ci_hi = thesis.wilson(act / 100, max(1, indep))
        cal.append({"bin": f"{lo}~{hi}%", "pred": round(float(g["pwin"].mean()), 1),
                    "act": round(act, 1), "n": int(len(g)), "indep": int(indep),
                    "lo": round(ci_lo * 100, 1), "hi": round(ci_hi * 100, 1),
                    "ok": bool(ci_lo * 100 <= g["pwin"].mean() <= ci_hi * 100)})
    out["calibration"] = cal
    # 예측-실현 상관 (예측이 방향이라도 맞나)
    out["cal_corr"] = round(float(df["pwin"].corr(df["hit"])), 4)
    out["cal_bias"] = round(float(df["pwin"].mean() - df["hit"].mean() * 100), 1)
    out["ev_pred"] = round(float(df["pev"].mean()), 2)
    out["ev_act"] = round(float(df["fwd"].mean() * 100), 2)

    # ── Q2. 등급별 변별력 ───────────────────────────────────────
    gr = []
    for g0 in ["A", "B", "C", "D"]:
        sub = df[df["grade"] == g0]
        if len(sub) < 30:
            continue
        indep = sum(greedy_independent(sorted(gg["j"].tolist()))
                    for _, gg in sub.groupby("ci"))
        win = float(sub["hit"].mean() * 100)
        lo, hi = thesis.wilson(win / 100, max(1, indep))
        gr.append({"grade": g0, "n": int(len(sub)), "indep": int(indep),
                   "win": round(win, 1), "lo": round(lo * 100, 1), "hi": round(hi * 100, 1),
                   "avg": round(float(sub["fwd"].mean() * 100), 2),
                   "med": round(float(sub["fwd"].median() * 100), 2),
                   "p_stop": round(float((sub["fwd"] <= -0.10).mean() * 100), 1)})
    out["grades"] = gr
    if len(gr) >= 2:
        out["grade_spread"] = round(gr[0]["avg"] - gr[-1]["avg"], 2)
    out["conf_corr"] = round(float(df["conf"].corr(df["fwd"])), 4)

    # ── Q3. 이 등급대로 샀다면 (비중첩 30일 리밸런싱) ────────────
    def simulate(pick, label):
        """pick(day_df) -> 선택된 행들. 30거래일마다 갈아타며 동일가중."""
        eq = 1.0
        curve, rets_, npick, skipped = [], [], [], 0
        j = WARMUP
        while j < nD - FWD:
            day = df[df["j"] == j]
            if len(day):
                sel = pick(day)
                if len(sel):
                    r = float(sel["fwd"].mean()) - 0.003     # 왕복비용 0.3%
                    npick.append(len(sel))
                else:
                    r = 0.0                                  # 살 게 없으면 현금(수익 0)
                    skipped += 1
                eq *= (1 + r)
                rets_.append(r)
                curve.append({"date": str(pd.Timestamp(dates[j]).date()), "v": round(eq, 4)})
            j += FWD
        if not rets_:
            return None
        a = np.array(rets_)
        yrs = len(a) * FWD / 252
        return {"name": label, "n_rebal": len(a), "cash": skipped,
                "avg_n": round(float(np.mean(npick)), 1) if npick else 0,
                "total": round((eq - 1) * 100, 1),
                "ann": round((eq ** (1 / max(yrs, 0.5)) - 1) * 100, 1),
                "win": round(float((a > 0).mean() * 100), 1),
                "avg": round(float(a.mean() * 100), 2),
                "worst": round(float(a.min() * 100), 1),
                "sharpe": round(float(a.mean() / (a.std() + 1e-9) * np.sqrt(252 / FWD)), 2),
                "curve": curve}

    # 대조군이 핵심이다. 'A등급 상위5'가 좋아 보여도, 그게 등급 덕인지
    # 그냥 5종목으로 집중했기 때문인지 구분하지 못하면 아무것도 증명 못 한다.
    sims = []
    for label, fn in [
        ("A등급 중 BuyFit 상위5", lambda d: d[d["grade"] == "A"].nlargest(5, "bpct")),
        ("BuyFit 상위5 (등급 무시·대조군)", lambda d: d.nlargest(5, "bpct")),
        ("A등급 전체 동일가중", lambda d: d[d["grade"] == "A"]),
        ("A+B등급 중 BuyFit 상위5", lambda d: d[d["grade"].isin(["A", "B"])].nlargest(5, "bpct")),
        ("BuyFit 상위10 (추천순)", lambda d: d.nlargest(10, "bpct")),
        ("기존 전략 (종합점수 상위10)", lambda d: d.nsmallest(10, "rank")),
        ("D등급 전체 (반대 검증)", lambda d: d[d["grade"] == "D"]),
        ("전 종목 동일가중 (시장)", lambda d: d),
    ]:
        r = simulate(fn, label)
        if r:
            sims.append(r)
    out["sims"] = sims

    # ── 국면별 등급 성적 ────────────────────────────────────────
    rg_tab = []
    for rgname, g in df.groupby("regime"):
        if not rgname or len(g) < 50:
            continue
        row = {"regime": rgname, "n": int(len(g))}
        for g0 in ["A", "D"]:
            sub = g[g["grade"] == g0]
            row[g0] = (round(float(sub["fwd"].mean() * 100), 2) if len(sub) >= 20 else None)
        rg_tab.append(row)
    out["by_regime"] = rg_tab

    os.makedirs("data", exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, allow_nan=False)

    # ── 콘솔 리포트 ─────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"  판단 등급·확률 검증  ({out['period'][0]} ~ {out['period'][1]})")
    print(f"  평가 {out['n_eval']:,}건 (매 {EVERY}거래일 × 100종목, 미래참조 차단)")
    print(f"{'='*66}")

    print(f"\n■ Q1. 캘리브레이션 — 예측 승률이 실제와 맞나")
    print(f"   {'예측구간':>10} {'예측':>6} {'실제':>6} {'95%구간':>16} {'표본':>7} {'독립':>6}  판정")
    for c in cal:
        mark = "✅ 일치" if c["ok"] else "❌ 벗어남"
        print(f"   {c['bin']:>10} {c['pred']:5.1f}% {c['act']:5.1f}% "
              f"{c['lo']:6.1f}~{c['hi']:5.1f}% {c['n']:7,} {c['indep']:6}  {mark}")
    print(f"   전체 편향(예측−실제): {out['cal_bias']:+.1f}%p · 예측-결과 상관 {out['cal_corr']:+.4f}")
    print(f"   기대수익 예측 {out['ev_pred']:+.2f}% vs 실제 {out['ev_act']:+.2f}%")

    print(f"\n■ Q2. 변별력 — 등급이 실제 결과와 이어지나")
    print(f"   {'등급':>4} {'표본':>7} {'독립':>6} {'승률':>7} {'95%구간':>16} {'평균':>7} {'중앙':>7} {'손절률':>7}")
    for g0 in gr:
        print(f"   {g0['grade']:>4} {g0['n']:7,} {g0['indep']:6} {g0['win']:6.1f}% "
              f"{g0['lo']:6.1f}~{g0['hi']:5.1f}% {g0['avg']:+6.2f}% {g0['med']:+6.2f}% {g0['p_stop']:6.1f}%")
    if "grade_spread" in out:
        print(f"   A−D 평균수익 격차: {out['grade_spread']:+.2f}%p · 확신도-수익 상관 {out['conf_corr']:+.4f}")

    print(f"\n■ Q3. 이 방법으로 투자했다면 (30거래일 비중첩 리밸런싱, 비용 0.3%)")
    print(f"   {'전략':>26} {'종목':>5} {'현금':>5} {'누적':>9} {'연율':>8} {'승률':>7} {'평균':>7} {'최악':>8} {'Sharpe':>7}")
    for r in sims:
        print(f"   {r['name']:>26} {r['avg_n']:5} {r['cash']:5} {r['total']:+8.1f}% {r['ann']:+7.1f}% "
              f"{r['win']:6.1f}% {r['avg']:+6.2f}% {r['worst']:+7.1f}% {r['sharpe']:6.2f}")

    if rg_tab:
        print(f"\n■ 국면별 A등급 vs D등급 평균 30일 수익")
        for r in rg_tab:
            a = f"{r['A']:+.2f}%" if r.get("A") is not None else "표본부족"
            dd_ = f"{r['D']:+.2f}%" if r.get("D") is not None else "표본부족"
            print(f"   {r['regime']:>6}  A {a:>9}   D {dd_:>9}   (평가 {r['n']:,}건)")
    print()


if __name__ == "__main__":
    main()
