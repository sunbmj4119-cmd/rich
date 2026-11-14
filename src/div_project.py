"""
고배당 주식·ETF 로 갈 때 — 자산이 얼마나 빨리 불어나나 · 무엇을 바꾸면 나아지나 · 앞으로 어디쯤 갈 수 있나.

포트폴리오 (div_lab 의 배당 복원·세후 15.4%·매일 판단 흉내를 그대로 씀)
  ETF대용      고배당 ETF 대용: 최근 1년 배당수익률 상위 30종목 같은 비중 · 상위 30% 안이면 삼 · 상위 60% 밖이면 팜 · 최소 250일 (≈ 해마다 교체)
               · 운용보수 연 0.3% (실제 고배당 ETF 가격은 이 환경에서 못 받아 같은 방식으로 만든 것)
  ETF대용·저변동  같은 30종목 방식, 점수 = 배당수익률 + 낮은 60일 변동성 반반
  고배당3      고배당 꾸준히를 3자리로 (작은 계좌용) · 상위 5% 안이면 삼 · 상위 30% 밖이면 팜 · 최소 60일
  고배당·저변동  div_lab D4 (10자리)
  섞기         추세·눌림(3차) 50% + 고배당·저변동 50% (탐색)
  대형주       배당자료 있는 대형주 똑같이 (배당 포함 기준)
  DM:2         지금 계획 (ETF 분배금은 빠져 있어 조금 낮게 잡힘)

1. 증폭 속도 (실제 지나온 길): 연 수익 · 두 배 걸린 햇수 · 최악의 1년/3년 · 가장 길게 물려 있던 기간 · 5년 굴린 수익의 범위
2. 매달 넣기: 20만 원으로 시작해 매달 0 · 10만 · 30만 원 — 2003 년부터 시작 날을 바꿔 가며 10년 뒤 금액의 범위 (지나온 길 그대로)
3. 개선 방향 (같은 길에서 하나만 바꿔 보기): 배당을 꺼내 쓰면 · 세금 0 (ISA 등 절세 계좌 가정) · 보수 0.3% vs 0.1% · 섞기
4. 앞으로 (방향 예측): 달마다 수익을 12개월 덩어리로 다시 이어 붙여(부트스트랩) 10년 길 10,000개 → 10·50·90% 자리
   바탕 3가지: 2003~ 전체 · 2019~ 최근 · 2008~2018 (가장 약했던 시기 = 보수적)
   '과거와 비슷한 일이 다시 섞여 일어난다면'의 범위이지 예언이 아니다.

  python src/div_project.py → data/div_project.json
"""
import json
import math
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import div_lab as DL  # noqa: E402
import score_lab as SC  # noqa: E402

OUT = "data/div_project.json"
DOCS = "docs/div_project.json"
FEE = 0.003
START = "2003-01-01"
MONTHLY = (0, 100_000, 300_000)
CASH0 = 200_000
YEARS = 10
NPATH = 10_000
BASES = {"2003~ 전체": ("2003-01-01", None), "2019~ 최근": ("2019-01-01", None), "2008~2018 (보수적)": ("2008-01-01", "2018-12-31")}
NAMES = {"ETF대용": "고배당 ETF 대용 (30종목 · 해마다 교체 · 보수 0.3%)", "ETF대용·저변동": "고배당·저변동 ETF 대용 (30종목)",
         "고배당3": "고배당 꾸준히 · 3자리 (작은 계좌)", "고배당·저변동": "고배당 + 덜 흔들림 (10자리)",
         "섞기": "추세·눌림 50% + 고배당·저변동 50%", "대형주": "대형주 똑같이 (배당 포함 · 기준)", "DM:2": "DM:2 (지금 계획)"}


def build(log=print):
    SC.set_factors(True)
    ER = SC.eras(lambda m: None)
    DL.attach_dividends(ER, lambda m: None)
    S = DL.scores(ER)
    Sx = {k: {e: S[e][k] for e in ER} for k in ("D1", "D4", "C")}
    etf_rule = {"slots": 30, "q_in": 0.30, "q_out": 0.60, "min_hold": 250, "stop": 0.0, "max_hold": 0}
    out, out_notax, out_px = {}, {}, {}

    def run(key, Sk, rule, fee=0.0):
        a = DL.run_one(ER, Sk, rule, True)
        p = DL.run_one(ER, Sk, rule, False)
        saved = DL.TAX
        DL.TAX = 0.0
        try:
            DL.attach_dividends(ER, lambda m: None)
            n = DL.run_one(ER, Sk, rule, True)
        finally:
            DL.TAX = saved
            DL.attach_dividends(ER, lambda m: None)
        out[key] = a["r"] - fee / 252
        out_px[key] = p["r"] - fee / 252
        out_notax[key] = n["r"] - fee / 252
        log(f"  {key}: 거래 {len(a['trades']) / (len(a['r']) / 252):.0f}/년")
        SC._PREP.clear()
        return a
    a = run("ETF대용", Sx["D1"], etf_rule, FEE)
    run("ETF대용·저변동", Sx["D4"], etf_rule, FEE)
    run("고배당3", Sx["D1"], {**DL.RULES["D1"], "slots": 3, "q_in": 0.05})
    run("고배당·저변동", Sx["D4"], DL.RULES["D4"])
    c = run("추세·눌림", Sx["C"], DL.RULES["C"])
    idx = a["r"].index
    for d in (out, out_px, out_notax):
        d["섞기"] = 0.5 * d["추세·눌림"] + 0.5 * d["고배당·저변동"]
    ew = a["ew"].reindex(idx).fillna(0.0)
    out["대형주"] = ew
    out["DM:2"] = SC.card_daily("DM:2", idx)
    out["ETF대용(보수0.1%)"] = out["ETF대용"] + (FEE - 0.001) / 252
    del c
    return idx, out, out_px, out_notax


def speed(r):
    r = r.dropna()
    E = (1 + r).cumprod()
    yrs = len(r) / 252
    g = E.iloc[-1] ** (1 / yrs) - 1
    yr = (1 + r).groupby(r.index.year).prod() - 1
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    roll36 = (1 + m).rolling(36).apply(np.prod, raw=True) - 1
    roll60 = ((1 + m).rolling(60).apply(np.prod, raw=True)) ** (1 / 5) - 1
    peak = E.cummax()
    under = E < peak
    longest, cur = 0, 0
    for u in under.to_numpy():
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return {"cagr": round(float(g), 4), "double_years": round(math.log(2) / math.log(1 + g), 1) if g > 0 else None,
            "worst_year": round(float(yr.min()), 4), "worst_3y": round(float(roll36.min()), 4), "mdd": round(float((E / peak - 1).min()), 4),
            "longest_underwater_years": round(longest / 252, 1), "roll5_min": round(float(roll60.min()), 4), "roll5_med": round(float(roll60.median()), 4),
            "roll5_max": round(float(roll60.max()), 4), "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)), 3)}


def monthly(r):
    return (1 + r).groupby(r.index.to_period("M")).prod() - 1


def dca_hist(m, monthly_add, years=YEARS, cash0=CASH0):
    """지나온 길 그대로: 시작 달을 바꿔 가며 cash0 + 매달 넣기 → years 뒤 금액들."""
    n = years * 12
    vals, paid = [], cash0 + monthly_add * n
    arr = m.to_numpy()
    for s in range(0, len(arr) - n + 1):
        v = cash0
        for k in range(n):
            v = v * (1 + arr[s + k]) + monthly_add
        vals.append(v)
    vals = np.array(vals)
    if not len(vals):
        return None
    return {"n_starts": int(len(vals)), "paid": paid, "p10": round(float(np.percentile(vals, 10))), "p50": round(float(np.percentile(vals, 50))),
            "p90": round(float(np.percentile(vals, 90))), "min": round(float(vals.min())), "loss_prob": round(float((vals < paid).mean()), 3)}


def bootstrap(m, monthly_add, years=YEARS, n_paths=NPATH, block=12, seed=0, cash0=CASH0):
    """달 수익을 12개월 덩어리로 다시 이어 붙인 길 n_paths 개 → 달마다 10·25·50·75·90% 자리 · 끝 금액 분포."""
    rng = np.random.default_rng(seed)
    arr = m.to_numpy()
    L = len(arr)
    n = years * 12
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, L - block + 1, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :n]
    R = arr[idx]
    V = np.empty((n_paths, n + 1))
    V[:, 0] = cash0
    for k in range(n):
        V[:, k + 1] = V[:, k] * (1 + R[:, k]) + monthly_add
    paid = cash0 + monthly_add * np.arange(n + 1)
    pct = {q: np.percentile(V, q, axis=0) for q in (10, 25, 50, 75, 90)}
    fin = V[:, -1]
    return {"paid": float(paid[-1]), "p10": round(float(np.percentile(fin, 10))), "p50": round(float(np.percentile(fin, 50))), "p90": round(float(np.percentile(fin, 90))),
            "loss_prob": round(float((fin < paid[-1]).mean()), 3), "median_cagr_like": round(float(np.median((1 + R).prod(1)) ** (1 / years) - 1), 4),
            "path": {"month": list(range(n + 1)), **{f"p{q}": [round(float(v)) for v in pct[q]] for q in pct}, "paid": [round(float(v)) for v in paid]}}


def run(log=print):
    t0 = time.time()
    idx, R, Rpx, Rnt = build(log)
    keys = ["ETF대용", "ETF대용·저변동", "고배당3", "고배당·저변동", "섞기", "대형주", "DM:2"]
    res = {"names": NAMES, "fee": FEE, "cash0": CASH0, "monthly": list(MONTHLY), "years": YEARS, "bases": BASES}
    # 1. 증폭 속도
    res["speed"] = {}
    for k in keys:
        res["speed"][k] = {pn: speed(SC.cut(R[k], max(a, "2005-01-01") if k == "DM:2" else a, b)) for pn, (a, b) in BASES.items()}
        s = res["speed"][k]["2003~ 전체"]
        log(f"  속도 {k:10s}: 연 {s['cagr']:+.1%} · 두 배 {s['double_years']}년 · 최악 1년 {s['worst_year']:+.0%} · 최악 3년 {s['worst_3y']:+.0%} · 가장 긴 물림 {s['longest_underwater_years']}년 · 5년 굴림 {s['roll5_min']:+.0%}~{s['roll5_max']:+.0%}")
    # 배당 수입: 자산 대비 세후 배당 (연)
    res["div_yield_after_tax"] = {k: round(float((R[k] - Rpx[k]).mean() * 252), 4) for k in ("ETF대용", "ETF대용·저변동", "고배당3", "고배당·저변동", "섞기")}
    # 2. 매달 넣기 (지나온 길)
    res["dca_hist"] = {k: {str(c): dca_hist(monthly(SC.cut(R[k], START if k != "DM:2" else "2005-01-01", None)), c) for c in MONTHLY} for k in keys}
    for k in ("ETF대용", "섞기", "대형주"):
        v = res["dca_hist"][k]["100000"]
        log(f"  지나온 길 {k}: 20만 + 매달 10만 · 10년 → 낸 돈 {v['paid']:,} · 10% {v['p10']:,} · 중앙 {v['p50']:,} · 90% {v['p90']:,} · 원금 밑 {v['loss_prob']:.0%}")
    # 3. 개선 방향 (2003~ 전체, 같은 길에서 하나만 바꿈)
    imp = []
    base = R["ETF대용"]
    for nm, r in (("고배당 ETF 대용 (기본: 배당 재투자 · 세금 15.4% · 보수 0.3%)", base),
                  ("배당을 꺼내 씀 (재투자 안 함)", Rpx["ETF대용"]),
                  ("세금 0 (ISA 등 절세 계좌 가정)", Rnt["ETF대용"]),
                  ("보수 0.1% 짜리", R["ETF대용(보수0.1%)"]),
                  ("종목 직접 · 저변동 섞기 (10자리)", R["고배당·저변동"]),
                  ("추세·눌림 50% 섞기", R["섞기"])):
        s = speed(SC.cut(r, START, None))
        imp.append({"change": nm, **s, "dca_100k_p50": dca_hist(monthly(SC.cut(r, START, None)), 100_000)["p50"]})
        log(f"  개선 {nm:42s}: 연 {s['cagr']:+.1%} · 샤프 {s['sharpe']:.2f} · 낙폭 {s['mdd']:.0%} · 20만+매달 10만 10년 중앙 {imp[-1]['dca_100k_p50']:,}원")
    res["improve"] = imp
    # 4. 앞으로 (부트스트랩)
    proj = {}
    for k in keys:
        proj[k] = {}
        for bn, (a, b) in BASES.items():
            if k == "DM:2" and a < "2005":
                a = "2005-01-01"
            m = monthly(SC.cut(R[k], a, b))
            proj[k][bn] = {str(c): bootstrap(m, c, seed=zlib.crc32(f'{k}|{bn}|{c}'.encode())) for c in MONTHLY}
    res["projection"] = proj
    for k in ("ETF대용", "섞기", "DM:2"):
        for bn in BASES:
            v = proj[k][bn]["100000"]
            log(f"  앞으로 {k:8s} [{bn}] 20만 + 매달 10만 · 10년: 낸 돈 {v['paid']:,.0f} → 10% {v['p10']:,} · 중앙 {v['p50']:,} · 90% {v['p90']:,} · 원금 밑 {v['loss_prob']:.0%}")
    # 10년 뒤 배당 (중앙 금액 × 세후 배당수익률)
    res["div_income_y10"] = {k: {bn: round(proj[k][bn]["100000"]["p50"] * res["div_yield_after_tax"].get(k, 0.0)) for bn in BASES}
                             for k in res["div_yield_after_tax"]}
    # 곡선 (지나온 길, 1원 시작, 5일 간격)
    res["curve"] = {"date": [str(d.date()) for d in idx[::5]], **{k: [round(float(v), 4) for v in (1 + R[k]).cumprod()[::5]] for k in keys}}
    res["secs"] = round(time.time() - t0)
    res["generated"] = time.strftime("%Y-%m-%d %H:%M")
    for fn in (OUT, DOCS):
        json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"끝 ({res['secs']}초) → {OUT}")
    return res


def export(outdir):
    """화면(대시보드)용 자료 파일 — {rows, meta}."""
    r = json.load(open(OUT, encoding="utf-8"))
    os.makedirs(outdir, exist_ok=True)
    code = open(os.path.abspath(__file__), encoding="utf-8").read()
    meta = [{"label": "div_project.py", "value": code, "type": "code", "lang": "python"},
            {"label": "가정", "value": "배당 재투자 · 배당소득세 15.4% · 왕복 비용 약 0.34% · ETF 대용 보수 연 0.3% · 시작 20만 원 · 매달 넣는 돈은 달 끝에"},
            {"label": "앞으로의 범위", "value": "달 수익을 12개월 덩어리로 다시 이어 붙인 10년 길 10,000개의 10·25·50·75·90% 자리 — 과거와 비슷한 일이 섞여 다시 일어난다는 가정"}]
    f = {}
    f["proj_paths"] = [{"key": k, "base": bn, "monthly": int(c), "month": mo, **{q: v["path"][q][i] for q in ("p10", "p25", "p50", "p75", "p90", "paid")}}
                       for k, d in r["projection"].items() for bn, dd in d.items() for c, v in dd.items() for i, mo in enumerate(v["path"]["month"])]
    f["proj_summary"] = [{"key": k, "name": r["names"][k], "base": bn, "monthly": int(c), **{m: v[m] for m in ("paid", "p10", "p50", "p90", "loss_prob")}}
                         for k, d in r["projection"].items() for bn, dd in d.items() for c, v in dd.items()]
    f["proj_speed"] = [{"key": k, "name": r["names"][k], "base": bn, **v} for k, d in r["speed"].items() for bn, v in d.items()]
    f["proj_improve"] = r["improve"]
    f["proj_dca"] = [{"key": k, "name": r["names"][k], "monthly": int(c), **v} for k, d in r["dca_hist"].items() for c, v in d.items() if v]
    f["proj_div10"] = [{"key": k, "name": r["names"][k], "base": bn, "income": v, "yield_after_tax": r["div_yield_after_tax"][k]}
                       for k, d in r["div_income_y10"].items() for bn, v in d.items()]
    for k, rows in f.items():
        json.dump({"rows": rows, "meta": meta}, open(os.path.join(outdir, f"{k}.json"), "w", encoding="utf-8"), ensure_ascii=False, default=float)
    return list(f)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--export":
        print(export(sys.argv[2]))
    else:
        run(lambda m: print(m, flush=True))
