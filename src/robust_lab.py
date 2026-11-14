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


# ── 보조 검정 (DSR 이 0 으로 나온 뒤 추가 — 사전 등록 아님, 결과와 함께 그렇게 적는다) ──────────────
#   DSR 의 기준선은 '후보들 샤프의 퍼짐'으로 정하는데, 이번 후보에는 같은 공식의 반대 부호가 짝으로 들어 있어
#   진짜 신호와 그 거울상이 퍼짐을 키운다 → 기준선이 연 2.7 정도로 부풀려짐. 이런 '서로 얽힌 많은 후보 중 1등' 문제에
#   쓰는 표준 검정이 Hansen SPA (White Reality Check 의 개선판): 후보 전체를 같은 날짜 덩어리로 다시 뽑아(정상 부트스트랩)
#   '아무 후보도 대형주를 못 이긴다'는 가정 아래 1등의 성적이 이만큼 나올 확률을 직접 잰다.

def spa(X, B=500, mean_block=20, seed=0):
    """Hansen SPA (consistent, 표준화) — X: T×N 초과수익. 반환 p 값 (작을수록 '운만으로는 어렵다')."""
    rng = np.random.default_rng(seed)
    T, N = X.shape
    mu = X.mean(0)
    # 날짜 덩어리 길이가 평균 mean_block 인 정상 부트스트랩 → 덩어리 평균의 분산으로 표준오차
    sd = X.std(0) / math.sqrt(T)
    lag = int(mean_block)
    adj = np.zeros(N)
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        adj += 2 * w * ((X[k:] - mu) * (X[:-k] - mu)).mean(0)
    omega = np.sqrt(np.maximum((X.var(0) + adj) / T, 1e-18))
    stat = max(0.0, float(np.max(mu / omega)))
    thr = -np.sqrt(2 * math.log(math.log(T))) * omega
    mu_c = np.where(mu >= thr, mu, 0.0)
    p_q = 1.0 / mean_block
    cnt = 0
    for _ in range(B):
        idx = np.empty(T, int)
        i = rng.integers(T)
        for t in range(T):
            if t == 0 or rng.random() < p_q:
                i = rng.integers(T)
            else:
                i = (i + 1) % T
            idx[t] = i
        mb = X[idx].mean(0)
        z = max(0.0, float(np.max((mb - mu_c) / omega)))                  # Hansen: 쓸 만한 후보는 평균을 0 으로 옮기고, 아주 나쁜 후보는 그대로
        cnt += z >= stat
    del sd
    return {"stat": round(stat, 3), "p": round(cnt / B, 4), "B": B}


def placebo(ER, idx, EW, n=30, seed=3):
    """무작위 점수(아무 정보 없음)로 같은 규칙 → 대형주 대비 초과의 분포 (비용 뒤)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        S = {e: np.where(E.U, rng.random(E.U.shape), np.nan).astype(np.float32) for e, E in ER.items()}
        for rn, r in RULES3.items():
            a = SC.stitch(ER, S, r, START, None)
            x = a["r"].reindex(idx).fillna(0.0).to_numpy() - EW
            out.append(float(x.mean() * 252))
        SC._PREP.clear()
    a = np.array(out)
    return {"n": len(a), "mean": round(float(a.mean()), 4), "p95": round(float(np.percentile(a, 95)), 4), "max": round(float(a.max()), 4)}


def yearly_hits(x, idx):
    s = pd.Series(x, index=idx)
    y = s.groupby(idx.year).sum()
    return round(float((y > 0).mean()), 3), {str(k): round(float(v), 4) for k, v in y.items()}


FAMILIES = {   # 결과를 본 뒤 정한 묶음 (탐색용) — 1등 하나가 아니라 같은 생각의 공식 전체가 통하는지
    "장기 추세 + 단기 눌림": ({"r250x": 1, "r120": 1, "ma120": 1, "hi52": 1, "r60": 1}, {"r5": -1, "r1": -1, "ma20": -1, "clv5": -1, "clv1": -1, "r20": -1}),
    "외국인 반대 + 눌림·수급": ({"for1": -1, "for5": -1, "for20": -1, "forstk": -1}, {"smart": 1, "ma20": -1, "r5": -1, "clv5": -1, "ind5": 1}),
}


def families(X, keys, F, idx):
    out = {}
    era_mask = {pn: ((idx >= pd.Timestamp(a)) & ((idx <= pd.Timestamp(b)) if b else True)) for pn, (a, b) in ERAS3.items()}
    for nm, (A, Bs) in FAMILIES.items():
        for rn in RULES3:
            cols = []
            for ka, sa in A.items():
                for kb, sb in Bs.items():
                    if ka == kb:
                        continue
                    for fi, w in enumerate(F):
                        if len(w) == 2 and w.get(ka) == sa and w.get(kb) == sb:
                            cols.append(keys.index(f"{fi}|{rn}"))
            if not cols:
                continue
            ex = {pn: X[m][:, cols].mean(0) * 252 for pn, m in era_mask.items()}
            allpos = np.all(np.stack([ex[pn] > 0 for pn in ERAS3]), axis=0)
            xb = X[:, cols].mean(1)
            out[f"{nm} · {rn}"] = {"members": len(cols), "all_eras_positive": round(float(allpos.mean()), 3),
                                   **{f"median_{pn}": round(float(np.median(ex[pn])), 4) for pn in ERAS3},
                                   "equal_mix_excess": {pn: round(float(xb[m].mean() * 252), 4) for pn, m in era_mask.items()},
                                   "equal_mix_t": round(month_t(xb, idx), 2)}
    return out


def snoop(log=print):
    """저장된 7,350개 일별 수익으로 SPA · 무작위 기준선 · 해마다 이긴 비율 (상위 후보)."""
    M = np.load("data/cache/robust_M.npy").astype(np.float64)
    kj = json.load(open("data/cache/robust_keys.json"))
    keys, idx = kj["keys"], pd.DatetimeIndex(kj["idx"])
    SC.set_factors(True)
    ER = SC.eras(lambda m: None)
    F = formulas()
    probe = SC.stitch(ER, {e: E.score(F[0]) for e, E in ER.items()}, RULES3["빠름"], START, None)
    EW = probe["ew"].reindex(idx).fillna(0.0).to_numpy()
    X = M - EW[:, None]
    t0 = time.time()
    out = {"spa_all": spa(X)}
    log(f"  SPA (7,350개 전체 중 1등이 운인가): p = {out['spa_all']['p']} ({time.time() - t0:.0f}초)")
    m19 = idx >= pd.Timestamp("2019-01-01")
    out["spa_2019"] = spa(X[m19], B=500)
    log(f"  SPA 2019~ 만: p = {out['spa_2019']['p']}")
    out["placebo"] = placebo(ER, idx, EW)
    log(f"  무작위 점수 (아무 정보 없음) 대형주 대비: 평균 {out['placebo']['mean']:+.1%} · 상위 5% {out['placebo']['p95']:+.1%} · 최고 {out['placebo']['max']:+.1%}")
    out["families"] = families(X, keys, F, idx)
    for k, v in out["families"].items():
        log(f"  묶음 {k}: {v['members']}개 · 세 시기 모두 플러스 {v['all_eras_positive']:.0%} · 중앙 {v['median_2003~2007']:+.0%}/{v['median_2008~2018']:+.0%}/{v['median_2019~']:+.0%} "
            f"· 고르게 섞기 {v['equal_mix_excess']['2003~2007']:+.0%}/{v['equal_mix_excess']['2008~2018']:+.0%}/{v['equal_mix_excess']['2019~']:+.0%} (t {v['equal_mix_t']:.1f})")
    res = json.load(open(OUT, encoding="utf-8"))
    for r in res["top"]:
        i = keys.index(r["key"])
        r["year_hit"], r["years"] = yearly_hits(X[:, i], idx)
    res["snoop"] = out
    res["snoop_note"] = "SPA 와 무작위 기준선은 DSR 이 0 으로 나온 뒤 추가한 보조 검정 (사전 등록 아님)."
    for fn in (OUT, DOCS):
        json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for r in sorted(res["top"], key=lambda r: -r["year_hit"])[:12]:
        log(f"    이긴 해 {r['year_hit']:.0%} · {r['formula'][:38]:38s} {r['rule']} · 초과 {r['excess_full']:+.1%} · 시기별 {r['excess_2003~2007']:+.0%}/{r['excess_2008~2018']:+.0%}/{r['excess_2019~']:+.0%} · 거래 {r['trades_per_year']:.0f}/년")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--snoop", action="store_true", help="저장된 결과로 보조 검정만")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)                                   # noqa: E731
    if a.snoop:
        snoop(log)
        return
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
