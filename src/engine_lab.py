"""
판단 엔진 실험대 — 논문·서적·차트분석에서 가져온 후보를 **이 데이터로** 검증한다.
-> data/engine_lab.json

후보 (출처)
  [차트]  200일선·추세정렬 (Faber 2007, Minervini 'Trend Template')
          52주 고가 근접 (George & Hwang 2004)
          RSI 과열 회피 (Wilder 1978) · 거래량 증가 (Lee & Swaminathan 2000)
          60일 낙폭 큰 '떨어지는 칼' 회피
  [재무]  수정 F-score (Piotroski 2000) · 이익 서프라이즈 SUE (Bernard & Thomas 1989)
  [동종]  섹터 내 상대가치 (Asness·Porter·Stevens 2000)
          섹터 가격 모멘텀 (Moskowitz & Grinblatt 1999) · 섹터 이익 모멘텀
  [시장]  국면별 노출 조절 (Faber 2007 · Moreira & Muir 2017 변동성 관리)

검증 절차 (strategy_lab.py와 같은 기준 — 과적합 방지)
  1) IC: 지표 vs 30거래일 뒤 수익의 순위상관, 연도별 부호 일관성
  2) 기존 전략(signal.py 규칙) 위에 '매수 조건'으로 얹었을 때 train/test 성과
  3) 둘 다 개선된 후보만 7개 다른 설정에서 교차 확인
  채택: train·test Sharpe 동시 개선 + 교차확인 train/test 각각 5/7 이상
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import features as F                         # noqa: E402
from strategy_lab import simulate, WARMUP, SPLIT   # noqa: E402

OUT = "data/engine_lab.json"
FWD = 30


def ic_by_year(feat, fwd, dates, step=5):
    rows = {}
    for t in range(WARMUP, len(dates) - FWD, step):
        x, y = feat[t], fwd[t]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30 or np.nanstd(x[m]) == 0:
            continue
        ic = pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())
        rows.setdefault(pd.Timestamp(dates[t]).year, []).append(ic)
    yr = {k: float(np.mean(v)) for k, v in rows.items()}
    allv = [i for v in rows.values() for i in v]
    return {"ic": round(float(np.mean(allv)), 4) if allv else None,
            "pos_years": f"{sum(1 for v in yr.values() if v > 0)}/{len(yr)}",
            "by_year": {k: round(v, 3) for k, v in yr.items()}}


def main():
    p, f = F.build_all()
    dates = p["dates"]
    nD = len(dates)
    split = int(np.searchsorted(dates.values, np.datetime64(SPLIT)))
    PX = p["close"].values
    VOL = f["vol20"].values
    RANK = f["rank"].values
    FOK = ((f["flow"] >= 0) | f["flow"].isna()).values if "flow" in f else np.ones_like(PX, bool)
    fwd = (p["close"].shift(-FWD) / p["close"] - 1).values
    mk = p["market"]
    REG = mk["regime"].values
    out = {"split": SPLIT, "period": [str(dates[WARMUP].date()), str(dates[-1].date())]}

    # ── 1) IC ───────────────────────────────────────────────
    print("■ 1. 예측력(IC) — 30거래일 뒤 수익과의 순위상관")
    cands = {
        "52주고가근접 hi52": f["hi52"], "추세정렬 trend_up": f["trend_up"].astype(float),
        "RSI14 (낮을수록↑)": -f["rsi14"], "거래량증가 vol_ratio": f["vol_ratio"],
        "60일낙폭 (깊을수록↑)": -f["dd60"], "200일선 기울기": f["ma200_slope"],
        "F-score": f["fscore"], "이익서프라이즈 SUE": f["sue"], "영업이익 증가율": f["op_yoy"],
        "섹터내 가치순위": f["sec_value_rank"], "섹터 가격모멘텀": f["sec_mom_rank"],
        "섹터 이익모멘텀": f["sec_earn"], "종합점수 (기준)": p["score"],
    }
    out["ic"] = {}
    for k, v in cands.items():
        r = ic_by_year(v.values.astype(float), fwd, dates)
        out["ic"][k] = r
        print(f"  {k:<22} IC {r['ic']:+.4f}  양수년 {r['pos_years']}")

    # ── 2) 매수 조건으로 얹기 ─────────────────────────────────
    BASE = dict(top_n=10, entry_rank=10, exit_rank=20, min_hold=30,
                stop_mode="fixed", stop=0.10, stop_k=5.0,
                trail=0.08, tp=None, tp_mode="fixed", tp_k=8.0,
                tiers=[(0.30, 0.03), (0.15, 0.05)], parts=[(0.25, 1 / 3)])
    G = {
        "추세정렬": f["trend_up"].values,
        "200일선 위": (p["close"] > f["ma200"]).values | f["ma200"].isna().values,
        "52주고가 75%↑": (f["hi52"] >= 0.75).values | f["hi52"].isna().values,
        "52주고가 85%↑": (f["hi52"] >= 0.85).values | f["hi52"].isna().values,
        "RSI<70 (과열회피)": (f["rsi14"] < 70).values | f["rsi14"].isna().values,
        "RSI<75": (f["rsi14"] < 75).values | f["rsi14"].isna().values,
        "60일낙폭 -25%↑ (칼 회피)": (f["dd60"] > -0.25).values | f["dd60"].isna().values,
        "F-score≥3": (f["fscore"] >= 3).values | f["fscore"].isna().values,
        "F-score≥4": (f["fscore"] >= 4).values | f["fscore"].isna().values,
        "SUE>0 (이익 개선)": (f["sue"] > 0).values | f["sue"].isna().values,
        "섹터내 가치 하위30% 제외": (f["sec_value_rank"] >= 0.3).values,
        "섹터모멘텀 하위30% 제외": (f["sec_mom_rank"] >= 0.3).values | f["sec_mom_rank"].isna().values,
        "섹터이익 감소 제외": (f["sec_earn"] > 0).values | f["sec_earn"].isna().values,
    }
    G["F≥3 + 칼회피"] = G["F-score≥3"] & G["60일낙폭 -25%↑ (칼 회피)"]
    G["F≥3 + RSI<75"] = G["F-score≥3"] & G["RSI<75"]

    def run(gate=None, params=BASE, **kw):
        fok = FOK & gate if gate is not None else FOK
        tr = simulate(PX, RANK, fok, VOL, params, WARMUP, split, reg=REG, **kw)
        te = simulate(PX, RANK, fok, VOL, params, split, nD, reg=REG, **kw)
        return tr, te

    def line(name, tr, te):
        print(f"  {name:<24} train 연{tr['ann']:+6.1f}% Sh{tr['sharpe']:5.2f} MDD{tr['mdd']:6.1f}%"
              f" 승{tr['win']:4.1f}% │ test 연{te['ann']:+6.1f}% Sh{te['sharpe']:5.2f}"
              f" MDD{te['mdd']:6.1f}% 승{te['win']:4.1f}%")

    def slim(r):
        return {k: r[k] for k in ["ann", "sharpe", "mdd", "win", "n_trades", "avg"]}

    print("\n■ 2. 기존 전략 + 매수 조건 (조건 불충족 = 관망)")
    b_tr, b_te = run()
    line("기준선 (현재 규칙)", b_tr, b_te)
    out["gates"] = {"기준선": {"train": slim(b_tr), "test": slim(b_te)}}
    passed = []
    for k, g in G.items():
        tr, te = run(g)
        line(k, tr, te)
        both = tr["sharpe"] > b_tr["sharpe"] and te["sharpe"] > b_te["sharpe"]
        out["gates"][k] = {"train": slim(tr), "test": slim(te), "both_better": both}
        if both:
            passed.append(k)
    print(f"\n  train·test Sharpe 동시 개선: {passed or '없음'}")

    # ── 3) 국면별 노출 ───────────────────────────────────────
    print("\n■ 3. 시장 국면별 신규매수 비중")
    reb = (mk["rebound"] >= 0.05).values
    REG2 = np.array([("급락장_반등" if (r == "급락장" and reb[t]) else r) for t, r in enumerate(REG)],
                    dtype=object)
    scen = {
        "약세 0.5": dict(scale={"약세장": 0.5}),
        "약세 0.5·급락 0.5": dict(scale={"약세장": 0.5, "급락장": 0.5}),
        "급락장 바닥확인 전 중단": dict(block={"급락장"}, reg_override=REG2),
        "약세 0.5 + 급락 바닥확인 전 중단": dict(scale={"약세장": 0.5}, block={"급락장"},
                                         reg_override=REG2),
    }
    out["regime"] = {}
    for k, kw in scen.items():
        kw = dict(kw)                       # 원본을 건드리면 교차확인이 다른 시나리오를 돌게 된다
        rg = kw.pop("reg_override", REG)
        fok = FOK
        tr = simulate(PX, RANK, fok, VOL, BASE, WARMUP, split, reg=rg, **kw)
        te = simulate(PX, RANK, fok, VOL, BASE, split, nD, reg=rg, **kw)
        line(k, tr, te)
        out["regime"][k] = {"train": slim(tr), "test": slim(te),
                            "both_better": tr["sharpe"] > b_tr["sharpe"] and te["sharpe"] > b_te["sharpe"]}

    # ── 4) 교차 확인 ─────────────────────────────────────────
    print("\n■ 4. 교차 확인 — 다른 설정에서도 통하나")
    alts = {
        "5종목": dict(BASE, top_n=5, entry_rank=5),
        "15종목": dict(BASE, top_n=15, entry_rank=15),
        "이탈30위": dict(BASE, exit_rank=30),
        "최소보유10일": dict(BASE, min_hold=10),
        "최소보유60일": dict(BASE, min_hold=60),
        "트레일12%": dict(BASE, trail=0.12),
        "사다리 없음": dict(BASE, tiers=None, parts=None),
    }
    cands = passed + [k for k, v in out["regime"].items() if v["both_better"]]
    out["cross"] = {}
    for k in cands:
        w = {"tr": 0, "te": 0, "trmdd": 0, "temdd": 0}
        for an, ap in alts.items():
            if k in G:
                a0 = run(None, ap)
                a1 = run(G[k], ap)
            else:
                kw = dict(scen[k])
                rg = kw.pop("reg_override", REG)
                a0 = (simulate(PX, RANK, FOK, VOL, ap, WARMUP, split, reg=REG),
                      simulate(PX, RANK, FOK, VOL, ap, split, nD, reg=REG))
                a1 = (simulate(PX, RANK, FOK, VOL, ap, WARMUP, split, reg=rg, **kw),
                      simulate(PX, RANK, FOK, VOL, ap, split, nD, reg=rg, **kw))
            w["tr"] += a1[0]["sharpe"] > a0[0]["sharpe"]
            w["te"] += a1[1]["sharpe"] > a0[1]["sharpe"]
            w["trmdd"] += a1[0]["mdd"] > a0[0]["mdd"]
            w["temdd"] += a1[1]["mdd"] > a0[1]["mdd"]
        adopt = w["tr"] >= 5 and w["te"] >= 5
        out["cross"][k] = {**{x: int(y) for x, y in w.items()}, "n": len(alts), "adopt": adopt}
        print(f"  {k:<28} Sharpe 개선 train {w['tr']}/{len(alts)} · test {w['te']}/{len(alts)}"
              f" · MDD 개선 train {w['trmdd']}/{len(alts)} · test {w['temdd']}/{len(alts)}"
              f"  → {'✅ 채택' if adopt else '❌ 기각'}")

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
