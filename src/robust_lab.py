"""
견고한 점수 찾기 (3차) — 과거 자료 전부(2003~2026)를 쓰되, 찾은 것이 운인지 실력인지를 통계로 가른다.

왜: 2019~ 는 이미 여러 번 열어 봐서 '처음 보는 시험지'가 아니다. 그래서 시험지를 따로 떼는 대신
    ① 세 시기(2003~2007 · 2008~2018 · 2019~) 모두에서 비용 뒤에도 통해야 하고
    ② 시도한 개수를 반영한 샤프(DSR)와 ③ 기간을 섞어 다시 고르는 과최적화 확률(PBO)로 운을 걸러낸다.

후보 (결과 보기 전에 정함, 2026-11-14)
  점수 공식: 요소 35개(1차 30 + 2차 5) 중 1개 또는 2개, 가중치 ±1 → 70 + 2,380 = 2,450개 (단순해야 덜 속는다)
  매매 규칙 3개 (거래 속도만 다름):
    빠름  5자리 · 상위 5% 안이면 삼 · 상위 20% 밖이면 팜 · 최소 1일
    중간  5자리 · 상위 5% 안이면 삼 · 상위 40% 밖이면 팜 · 최소 5일
    느림 10자리 · 상위 10% 안이면 삼 · 상위 50% 밖이면 팜 · 최소 20일
  → 7,350개를 2003~2026 매일 흉내 (매일 장 마감 판단 → 다음 날 시가, 왕복 비용 약 0.34%)

'유의미'의 기준 (모두)
  a. 비용 뒤 대형주 100 평균 대비 초과수익이 세 시기 각각 플러스, 전체 기간 t ≥ 2
  b. 비용 2배에서도 전체 기간 초과 > 0
  c. 초과수익의 DSR ≥ 0.95 — 시도 수 N = 이번 후보 수(7,350) 그리고 시도 장부 전체로도 따로 보고
  d. 이번 후보 묶음의 PBO(초과수익 기준, 16 토막) < 0.2
  통과한 것이 있으면 그중 전체 기간 초과 샤프가 가장 높은 것을 규칙 카드로, 상위 통과들을 고르게 섞은 것도 함께.

  python src/robust_lab.py [--workers 4] → data/robust_lab.json
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_lab as SC  # noqa: E402

OUT = "data/robust_lab.json"
DOCS = "docs/robust_lab.json"
START = "2003-01-01"
ERAS3 = {"2003~2007": ("2003-01-01", "2007-12-31"), "2008~2018": ("2008-01-01", "2018-12-31"), "2019~": ("2019-01-01", None)}
RULES3 = {"빠름": {"slots": 5, "q_in": 0.05, "q_out": 0.20, "min_hold": 1, "stop": 0.0, "max_hold": 0},
          "중간": {"slots": 5, "q_in": 0.05, "q_out": 0.40, "min_hold": 5, "stop": 0.0, "max_hold": 0},
          "느림": {"slots": 10, "q_in": 0.10, "q_out": 0.50, "min_hold": 20, "stop": 0.0, "max_hold": 0}}
CRIT = {"t_full": 2.0, "dsr": 0.95, "pbo": 0.2}

_ER = None


def formulas():
    ks = list(SC.FACTORS)
    out = [{ks[i]: s} for i in range(len(ks)) for s in (1, -1)]
    for i, j in itertools.combinations(range(len(ks)), 2):
        for si in (1, -1):
            for sj in (1, -1):
                out.append({ks[i]: si, ks[j]: sj})
    return out


def fname(w):
    return " ".join(f"{'+' if v > 0 else '−'}{SC.factor_name(k)}" for k, v in w.items())


def _init():
    global _ER
    SC.set_factors(True)
    _ER = SC.eras(lambda m: None)


def _work(chunk):
    """formula 묶음 → (번호, 규칙, 일별 수익, 비용 2배 일별 수익, 거래/년)."""
    out = []
    for fi, w in chunk:
        S = {e: E.score(w) for e, E in _ER.items()}
        for rn, r in RULES3.items():
            a = SC.stitch(_ER, S, r, START, None)
            out.append((fi, rn, a["r"].to_numpy(np.float32), round(len(a["trades"]) / (len(a["r"]) / 252), 1)))
        SC._PREP.clear()
    return out


def run(workers=4, log=print):
    import multiprocessing as mp
    t0 = time.time()
    _init()
    F = formulas()
    probe = SC.stitch(_ER, {e: E.score(F[0]) for e, E in _ER.items()}, RULES3["빠름"], START, None)
    idx, ew = probe["r"].index, probe["ew"]
    log(f"  후보 {len(F)} 공식 × {len(RULES3)} 규칙 = {len(F) * len(RULES3):,}개 · {len(idx):,}일 ({idx[0].date()}~{idx[-1].date()})")
    chunks = [list(enumerate(F))[i::workers * 8] for i in range(workers * 8)]
    R, meta = {}, {}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for n, res in enumerate(pool.imap_unordered(_work, chunks), 1):
            for fi, rn, r, tpy in res:
                key = f"{fi}|{rn}"
                assert len(r) == len(idx)
                R[key] = r
                meta[key] = {"f": fi, "rule": rn, "tpy": tpy}
            log(f"    {n}/{len(chunks)} 묶음 ({time.time() - t0:.0f}초)")
    keys = sorted(R, key=lambda k: (int(k.split("|")[0]), k.split("|")[1]))
    M = np.stack([R[k] for k in keys], axis=1).astype(np.float64)          # T × N
    EW = ew.reindex(idx).fillna(0.0).to_numpy()
    X = M - EW[:, None]                                                     # 대형주 평균 대비 일별 초과
    return idx, keys, M, X, EW, meta, F, time.time() - t0


def month_t(x, idx):
    s = pd.Series(x, index=idx)
    m = s.groupby(idx.to_period("M")).sum()
    return float(m.mean() / m.std() * math.sqrt(len(m))) if m.std() > 0 else 0.0


def analyze(idx, keys, M, X, EW, meta, F, log=print):
    import science_lab as SL2
    import trials as TR
    T, N = M.shape
    era_mask = {pn: ((idx >= pd.Timestamp(a)) & ((idx <= pd.Timestamp(b)) if b else True)) for pn, (a, b) in ERAS3.items()}
    ex_era = {pn: X[m].mean(0) * 252 for pn, m in era_mask.items()}
    ex_full = X.mean(0) * 252
    ir_daily = X.mean(0) / (X.std(0) + 1e-12)
    sharpe = M.mean(0) / (M.std(0) + 1e-12) * math.sqrt(252)
    consistent = np.all(np.stack([ex_era[pn] > 0 for pn in ERAS3]), axis=0)
    log(f"  세 시기 모두 비용 뒤 초과 > 0: {int(consistent.sum()):,} / {N:,}")
    # 과최적화 확률 (이번 묶음 전체, 초과수익 기준)
    pb = SL2.pbo(X, S=16, max_combos=4000)
    log(f"  PBO (초과수익, 16 토막): {pb['pbo']} · 앞 절반 1등의 뒤 절반 샤프 중앙 {pb['oos_sharpe_of_is_best_med']} (앞 절반 {pb['is_sharpe_best_med']})")
    try:
        ledger = int(TR.count()["total"])
    except Exception:                                                       # noqa: BLE001
        ledger = None
    cand = np.where(consistent)[0]
    tfull = {}
    for i in cand:
        tfull[i] = month_t(X[:, i], idx)
    ok_t = [i for i in cand if tfull[i] >= CRIT["t_full"]]
    log(f"  + 전체 기간 t ≥ 2: {len(ok_t):,}")
    ranked = sorted(ok_t, key=lambda i: -ir_daily[i])[:60]
    # 비용 2배 (상위 60개만 다시 흉내)
    SC.set_factors(True)
    ER = SC.eras(lambda m: None)
    rows = []
    for i in ranked:
        k = keys[i]
        fi, rn = int(k.split("|")[0]), k.split("|")[1]
        w = F[fi]
        S = {e: E.score(w) for e, E in ER.items()}
        b = SC.stitch(ER, S, RULES3[rn], START, None, cost_mult=2.0)
        x2 = b["r"].reindex(idx).fillna(0.0).to_numpy() - EW
        d_n, _ = SL2.dsr(X[:, i], ir_daily, n_trials=N)
        d_l, _ = SL2.dsr(X[:, i], ir_daily, n_trials=ledger) if ledger else (None, None)
        E_ = np.cumprod(1 + M[:, i])
        rows.append({"key": k, "formula": fname(w), "weights": w, "rule": rn, "rule_text": SC.rule_label(RULES3[rn]),
                     "cagr": round(float(E_[-1] ** (252 / T) - 1), 4), "sharpe": round(float(sharpe[i]), 3),
                     "mdd": round(float((E_ / np.maximum.accumulate(E_) - 1).min()), 4),
                     "excess_full": round(float(ex_full[i]), 4), "t_full": round(tfull[i], 2), "ir": round(float(ir_daily[i] * math.sqrt(252)), 3),
                     **{f"excess_{pn}": round(float(ex_era[pn][i]), 4) for pn in ERAS3},
                     "excess_x2": round(float(x2.mean() * 252), 4), "dsr_n": None if d_n is None else round(float(d_n), 4),
                     "dsr_ledger": None if d_l is None else round(float(d_l), 4), "trades_per_year": meta[k]["tpy"]})
        SC._PREP.clear()
    passed = [r for r in rows if r["excess_x2"] > 0 and (r["dsr_n"] or 0) >= CRIT["dsr"] and pb["pbo"] < CRIT["pbo"]]
    log(f"  + 비용 2배 > 0 · DSR(N={N:,}) ≥ 0.95 · PBO < 0.2: {len(passed)}")
    for r in rows[:15]:
        log(f"    {r['formula'][:40]:40s} {r['rule']} 연 {r['cagr']:+.1%} 샤프 {r['sharpe']:.2f} 낙폭 {r['mdd']:.0%} | 초과 {r['excess_full']:+.1%} (t {r['t_full']:.1f}) "
            f"시기별 {r['excess_2003~2007']:+.0%}/{r['excess_2008~2018']:+.0%}/{r['excess_2019~']:+.0%} · 비용2배 {r['excess_x2']:+.1%} · DSR {r['dsr_n']} · 장부 DSR {r['dsr_ledger']} · 거래 {r['trades_per_year']:.0f}/년")
    # 통과한 것 고르게 섞기 (상위 10개)
    blend = None
    if passed:
        top = [keys.index(r["key"]) for r in passed[:10]]
        rb = M[:, top].mean(1)
        xb = rb - EW
        E_ = np.cumprod(1 + rb)
        blend = {"n": len(top), "cagr": round(float(E_[-1] ** (252 / T) - 1), 4), "sharpe": round(float(rb.mean() / rb.std() * math.sqrt(252)), 3),
                 "mdd": round(float((E_ / np.maximum.accumulate(E_) - 1).min()), 4), "excess_full": round(float(xb.mean() * 252), 4),
                 **{f"excess_{pn}": round(float(xb[m].mean() * 252), 4) for pn, m in era_mask.items()}}
    # 요소 하나짜리만 따로 (가장 단순)
    singles = []
    for i, k in enumerate(keys):
        fi = int(k.split("|")[0])
        if len(F[fi]) == 1:
            singles.append({"key": k, "formula": fname(F[fi]), "rule": k.split("|")[1], "excess_full": round(float(ex_full[i]), 4),
                            **{f"excess_{pn}": round(float(ex_era[pn][i]), 4) for pn in ERAS3}, "consistent": bool(consistent[i])})
    singles.sort(key=lambda r: -r["excess_full"])
    ew_E = np.cumprod(1 + EW)
    bench = {"ew_cagr": round(float(ew_E[-1] ** (252 / T) - 1), 4), "ew_sharpe": round(float(EW.mean() / EW.std() * math.sqrt(252)), 3)}
    return {"n_candidates": N, "n_consistent": int(consistent.sum()), "n_t2": len(ok_t), "pbo": pb, "ledger_trials": ledger,
            "top": rows, "passed": [r["key"] for r in passed], "blend": blend, "singles": singles[:40], "bench": bench,
            "dist": {"excess_full_pct": [round(float(np.percentile(ex_full, q)), 4) for q in (5, 25, 50, 75, 95)]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)                                   # noqa: E731
    idx, keys, M, X, EW, meta, F, secs = run(a.workers, log)
    res = analyze(idx, keys, M, X, EW, meta, F, log)
    res.update({"generated": time.strftime("%Y-%m-%d %H:%M"), "crit": CRIT, "rules": RULES3, "eras": ERAS3, "secs": round(secs)})
    os.makedirs("data/cache", exist_ok=True)
    np.save("data/cache/robust_M.npy", M.astype(np.float32))
    json.dump({"keys": keys, "idx": [str(d.date()) for d in idx]}, open("data/cache/robust_keys.json", "w"))
    for fn in (OUT, DOCS):
        json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"끝 → {OUT}")


if __name__ == "__main__":
    main()
