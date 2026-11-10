"""
매도 규칙 실험실 — 손절·익절을 넓히고, 나눠 팔고, 1차 익절 뒤 손절을 본전으로, 고점 대비 따라 올라가는 손절(트레일링)까지.
매수 자리는 지난 실험에서 근거가 있던 것만 (새로 찾지 않음). 일봉 고가·저가, 그때의 상위 100 (상장폐지 포함).

사전 등록 (2026-11-09 · 결과를 보기 전에 정함)
  매수 자리  N1 확률 모델 상위 3 (edge_lab 의 ±7% 모델 — A 로 학습, B 는 처음 보는 자료) · N2 확률 모델 상위 1 ·
             N3 상승 추세 속 20일선 −5% 눌림 (chart_lab C03) — 다음 날 시가에 산다
  매도 규칙  손절 −5·−10·−15·−20% × 1차 익절 +5·+10·+15·+20·+30% × 1차에서 1/3·1/2·전부 ×
             나머지 2차 익절 +30·+50%·없음 × 1차 뒤 손절을 본전으로 함·안 함 × 고점(종가) 대비 손절 없음·−10·−20% × 최대 20·60거래일
             = 1,464 가지 (1차에서 전부 팔면 2차·본전은 뜻이 없어 한 번만 셈)
  판정 순서  한 봉 안: 손절(시가가 손절선 아래면 시가) → 1차 익절 → 2차 익절 · 기한 날 종가에 남은 것 모두 · 트레일링은 '어제까지의 최고 종가' 기준
  비용       왕복 약 0.34% (거래세 0.20%) — chart_lab 과 같음
  기준선     같은 날 상위 100 에서 무작위 10종목을 같은 규칙으로 (그날 시장의 몫을 뺀다)
  고르기     A(2002~2018)에서 매수 자리마다 '이웃 칸(손잡이 하나만 한 칸 옮김) 건당 순수익 중앙값'이 가장 큰 규칙 — 100건 이상
  확인       B(2019~)에서 숫자 그대로: 100건 이상 · 건당 순수익 > 0 (달별 t ≥ 2) · 같은 날 무작위보다 나음 (t ≥ 2) · 비용 2배에서도 > 0
  (N1·N2 는 A 가 모델을 배운 시대라 A 성적은 부풀어 있다 — 판정은 B 로만)

  python src/exit_lab.py → data/exit_lab.json
"""
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart_lab as CL  # noqa: E402
import edge_lab as EL  # noqa: E402
import intraday_sweep as IS  # noqa: E402

OUT = "data/exit_lab.json"
HMAX = 60
GRID = {"stop": [0.05, 0.10, 0.15, 0.20], "tp1": [0.05, 0.10, 0.15, 0.20, 0.30], "frac": [1 / 3, 0.5, 1.0],
        "tp2": [0.30, 0.50, 9.0], "be": [1, 0], "trail": [0.0, 0.10, 0.20], "hold": [20, 60]}
KEYS = list(GRID)
CRIT = {"min_n": 100, "t_net": 2.0, "t_edge": 2.0, "n_random": 10}
ENTRIES = {"N1": "확률 모델 상위 3", "N2": "확률 모델 상위 1", "N3": "상승 추세 속 20일선 −5% 눌림"}


def combo_key(c):
    """같은 규칙이면 같은 열쇠 · 2차 ≤ 1차면 None."""
    s, a, f, b, be, tr, h = c
    if f >= 1:
        return (s, a, 1.0, tr, h)
    return None if b <= a else c


class Paths60:
    """진입 다음 날부터 60거래일 (진입 시가 대비). 빈 고가·저가는 '닿지 않음', 빈 종가는 그 전 종가."""

    def __init__(self, P, ts, js):
        O, H, L = P["O"], P["H"], P["L"]
        C = P["Cf"] if "Cf" in P else pd.DataFrame(P["C"]).ffill().to_numpy()
        T = O.shape[0]
        ts, js = np.asarray(ts, int), np.asarray(js, int)
        e = ts + 1
        d = np.minimum(e[:, None] + np.arange(HMAX)[None, :], T - 1)
        po = O[e, js][:, None]
        jj = js[:, None]
        self.n = len(ts)
        self.ts = ts
        self.o = O[d, jj] / po
        self.h = np.where(np.isfinite(H[d, jj]), H[d, jj] / po, 0.0)
        self.l = np.where(np.isfinite(L[d, jj]), L[d, jj] / po, np.inf)
        self.c = pd.DataFrame(C[d, jj] / po).ffill(axis=1).fillna(1.0).to_numpy()


def simulate(PA, c, detail=False):
    """규칙 하나를 모든 행에 동시에 (날마다 한 걸음). 반환: 매도 금액 합(진입 시가 대비, 비용 전) [· 청산 날].
    한 봉 안 순서: 손절(갭이면 시가) → 1차 익절 → 2차 익절 · 기한 날 종가 · 트레일링 = 어제까지 최고 종가 × (1 − 폭)."""
    s, a, f, b, be, tr, h = c
    n = PA.n
    alive = np.ones(n, bool)
    half = np.zeros(n, bool)
    left = np.ones(n)
    got = np.zeros(n)
    runmax = np.ones(n)
    xd = np.full(n, h - 1)
    base_stop = 1 - s
    for d in range(h):
        if not alive.any():
            break
        lvl = np.where(half & (be > 0), np.maximum(base_stop, 1.0), base_stop)
        if tr > 0:
            lvl = np.maximum(lvl, runmax * (1 - tr))
        o, hi, lo, cl = PA.o[:, d], PA.h[:, d], PA.l[:, d], PA.c[:, d]
        if d > 0:
            gap = alive & np.isfinite(o) & (o <= lvl)
            got[gap] += left[gap] * o[gap]
            alive &= ~gap
            xd[gap] = d
        hit = alive & (lo <= lvl)
        got[hit] += left[hit] * lvl[hit]
        alive &= ~hit
        xd[hit] = d
        t1 = alive & ~half & (hi >= 1 + a)
        got[t1] += f * (1 + a)
        left[t1] -= f
        half |= t1
        done = alive & (left <= 1e-9)
        alive &= ~done
        xd[done] = d
        if b < 5:
            t2 = alive & half & (hi >= 1 + b)
            got[t2] += left[t2] * (1 + b)
            alive &= ~t2
            xd[t2] = d
        runmax = np.where(alive, np.maximum(runmax, np.nan_to_num(cl, nan=0.0)), runmax)
    last = PA.c[:, h - 1]
    got[alive] += left[alive] * last[alive]
    return (got, xd) if detail else got


def nets(got, mult=1.0):
    """매도 금액 합 → 순수익 (미끄러짐을 mult 배)."""
    cb, cs = CL.COMM + CL.SLIP * mult, CL.COMM + CL.SLIP * mult + CL.TAX
    return got * (1 - cs) / (1 + cb) - 1


def random_rows(E, k, seed):
    """날마다 그날 상위 100 (60일 길이가 있는 행) 중 무작위 k 종목."""
    rng = np.random.default_rng(seed)
    t, j = E.rows
    T = E.P["U"].shape[0]
    keep = t + HMAX + 1 < T
    t, j = t[keep], j[keep]
    order = np.argsort(t, kind="stable")
    t, j = t[order], j[order]
    starts = np.r_[0, np.flatnonzero(np.diff(t)) + 1, len(t)]
    rt, rj = [], []
    for a_, b_ in zip(starts[:-1], starts[1:]):
        pick = rng.choice(np.arange(a_, b_), size=min(k, b_ - a_), replace=False)
        rt.append(t[pick])
        rj.append(j[pick])
    return np.concatenate(rt), np.concatenate(rj)


def entry_rows(E, mask):
    t, j = np.nonzero(mask & E.valid)
    keep = t + HMAX + 1 < E.P["U"].shape[0]
    return t[keep], j[keep]


def baseline(E, rnd, log=None, tag=""):
    """같은 날 무작위 종목들의 규칙별 건당 순수익 평균 — {규칙 열쇠: (날짜 번호 배열, 날마다 평균)} (매수 자리마다 다시 계산하지 않게)."""
    PR = Paths60(E.P, *rnd)
    uday, inv = np.unique(rnd[0], return_inverse=True)
    cnt = np.bincount(inv)
    out = {}
    for c in itertools.product(*GRID.values()):
        k = combo_key(c)
        if k is None or k in out:
            continue
        out[k] = np.bincount(inv, weights=nets(simulate(PR, c))) / cnt
        if log and len(out) % 300 == 0:
            log(f"    {tag} 기준선 {len(out)}칸")
    return uday, out


def grid_eval(E, rows, base):
    """모든 규칙 × 매수 자리 → 칸마다 (거래 수 · 승률 · 건당 · 달별 t · 같은 날 무작위 대비 · 그 t · 비용 2배 건당)."""
    combos = list(itertools.product(*GRID.values()))
    PA = Paths60(E.P, *rows)
    mo = E.month[rows[0]]
    uday, bt = base
    pos = np.minimum(np.searchsorted(uday, rows[0]), len(uday) - 1)
    okday = uday[pos] == rows[0]
    out = np.full((len(combos), 7), np.nan)
    done = {}
    for ci, c in enumerate(combos):
        k = combo_key(c)
        if k is None:
            continue
        if k in done:
            out[ci] = out[done[k]]
            continue
        done[k] = ci
        g = simulate(PA, c)
        x = nets(g)
        ed = np.where(okday, x - bt[k][pos], np.nan)
        ok = np.isfinite(ed)
        out[ci] = (len(x), (x > 0).mean(), x.mean(), EL._tm(x, mo) or np.nan, np.nanmean(ed),
                   EL._tm(ed[ok], mo[ok]) or np.nan, nets(g, 2.0).mean())
    return out.reshape(tuple(len(v) for v in GRID.values()) + (7,))


def combo_of(ix):
    return {k: GRID[k][i] for k, i in zip(KEYS, ix)}


def describe(c):
    f = c["frac"]
    s = f"손절 −{c['stop']:.0%} · 1차 +{c['tp1']:.0%}"
    if f >= 1:
        s += " 전부"
    else:
        s += f" {'1/3' if f < 0.4 else '1/2'} · 2차 " + ("없음" if c["tp2"] > 5 else f"+{c['tp2']:.0%}") + (" · 본전 손절" if c["be"] else "")
    if c["trail"]:
        s += f" · 고점 −{c['trail']:.0%} 따라 손절"
    return s + f" · 최대 {c['hold']}일"


def row(R, ix):
    v = R[ix]
    f = lambda x, r=5: None if not np.isfinite(x) else round(float(x), r)       # noqa: E731
    return {"n": int(v[0]) if np.isfinite(v[0]) else 0, "win": f(v[1], 3), "mean": f(v[2]), "t_net": f(v[3], 2), "edge": f(v[4]),
            "t_edge": f(v[5], 2), "mean_x2": f(v[6])}


def trades_of(E, rows, c):
    """행마다 (순수익, 청산 날 번호) — 계좌 흉내에 쓴다."""
    t, j = rows
    g, xd = simulate(Paths60(E.P, t, j), tuple(c[k] for k in KEYS), detail=True)
    return nets(g), t + 1 + xd


def portfolio(E, rows, score, c, slots, cash0=200_000, pre=None):
    """매수 자리 후보를 날마다 점수 순으로, 빈 자리(slots)만큼 산다 (같은 종목 동시 1번) · 한 자리 = 그때 자산 ÷ slots."""
    t, j = rows
    net, exit_t = pre if pre is not None else trades_of(E, rows, c)
    order = np.lexsort((-score, t))
    eq = 1.0
    peak, mdd = 1.0, 0.0
    held = []                                                    # (exit_t, j, amount, net)
    n = wins = 0
    dates = E.P["dates"]
    curve = []
    for q in order:
        tt = t[q]
        still = []
        for p in held:
            if p[0] < tt + 1:
                eq += p[2] * p[3]
                curve.append((p[0], eq))
                peak = max(peak, eq)
                mdd = min(mdd, eq / peak - 1)
            else:
                still.append(p)
        held = still
        if len(held) >= slots or any(p[1] == j[q] for p in held):
            continue
        held.append((exit_t[q], j[q], eq / slots, net[q]))
        n += 1
        wins += net[q] > 0
    for p in sorted(held):
        eq += p[2] * p[3]
        curve.append((p[0], eq))
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    if not curve:
        return None
    d0, d1 = dates[t.min()], dates[min(max(x[0] for x in curve), len(dates) - 1)]
    yrs = max((d1 - d0).days / 365.25, 1e-9)
    return {"trades": n, "win": round(wins / max(n, 1), 3), "total": round(eq - 1, 4), "cagr": round(eq ** (1 / yrs) - 1, 4) if eq > 0 else -1,
            "mdd": round(mdd, 4), "final_200k": round(cash0 * eq)}


def main():
    t0 = time.time()
    log = lambda m: print(m, flush=True)                          # noqa: E731
    log("■ 자료·확률 모델 (edge_lab 과 같은 ±7% 모델)")
    EA, EB = EL.Era("A", log), EL.Era("B", log)
    keys = list(EL.FEAT)
    XA, XB = EL.design(EA, keys), EL.design(EB, keys)
    m = EL.fit_logit(XA, (EA.out["E7"][0][EA.rows] > 0).astype(float))
    pA, pB = EL.predict(m, XA), EL.predict(m, XB)
    score = {}
    masks = {}
    for era, E, p in (("A", EA, pA), ("B", EB, pB)):
        S = CL.setup_signals(E.P)["C03"]
        masks[era] = {"N1": EL.top_mask(E, p, 3), "N2": EL.top_mask(E, p, 1), "N3": S}
        T, N = E.P["U"].shape
        Pm = np.full((T, N), np.nan)
        Pm[E.rows] = p
        MC = np.nan_to_num(E.P["MC"])
        score[era] = {"N1": Pm, "N2": Pm, "N3": MC}
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "grid": GRID, "criteria": CRIT, "entries": ENTRIES, "eras": EL.ERA,
           "n_rules": len({combo_key(c) for c in itertools.product(*GRID.values())} - {None}), "rows": {}, "picks": {}, "marginals": {}}
    R = {}
    for era, E in (("A", EA), ("B", EB)):
        rnd = random_rows(E, CRIT["n_random"], seed=11 if era == "A" else 22)
        log(f"  {era} 같은 날 무작위 {len(rnd[0]):,}행 — 규칙별 기준선 계산")
        base = baseline(E, rnd, log, era)
        for nm in ENTRIES:
            rows = entry_rows(E, masks[era][nm])
            log(f"  {era} {nm} {ENTRIES[nm]}: {len(rows[0]):,}건 ({(time.time() - t0) / 60:.1f}분)")
            R[(era, nm)] = grid_eval(E, rows, base)
            res["rows"].setdefault(nm, {})[era] = len(rows[0])
    for nm in ENTRIES:
        RA, RB = R[("A", nm)], R[("B", nm)]
        M = RA[..., 2]
        elig = (RA[..., 0] >= CRIT["min_n"]) & np.isfinite(M)
        sm = IS.neighbor_median(np.where(np.isfinite(M), M, np.nan))
        sc = np.where(elig, sm, -np.inf)
        best = np.flatnonzero(sc.ravel() == sc.max())
        ix = np.unravel_index(best[np.argmax(M.ravel()[best])], M.shape)
        c = combo_of(ix)
        b = row(RB, ix)
        ok = (b["n"] >= CRIT["min_n"] and (b["mean"] or -1) > 0 and (b["t_net"] or 0) >= CRIT["t_net"]
              and (b["edge"] or -1) > 0 and (b["t_edge"] or 0) >= CRIT["t_edge"] and (b["mean_x2"] or -1) > 0)
        peakA = np.unravel_index(np.nanargmax(np.where(elig, M, -np.inf)), M.shape)
        bestB = np.unravel_index(np.nanargmax(np.where(RB[..., 0] >= CRIT["min_n"], RB[..., 2], -np.inf)), M.shape)
        res["picks"][nm] = {"rule": c, "desc": describe(c), "A": row(RA, ix), "A_neighbor_median": round(float(sm[ix]), 5), "B": b, "pass": bool(ok),
                            "peak_A": {"desc": describe(combo_of(peakA)), "A": row(RA, peakA), "B": row(RB, peakA)},
                            "best_B_hindsight": {"desc": describe(combo_of(bestB)), "B": row(RB, bestB), "A": row(RA, bestB)},
                            "share_pos_both": round(float(np.nanmean(((RA[..., 2] > 0) & (RB[..., 2] > 0))[np.isfinite(RA[..., 2]) & np.isfinite(RB[..., 2])])), 4)}
        mg = {}
        for ax, k in enumerate(KEYS):
            mg[k] = {}
            for i, v in enumerate(GRID[k]):
                a_ = np.take(RA[..., 2], i, axis=ax).ravel()
                b_ = np.take(RB[..., 2], i, axis=ax).ravel()
                mg[k][str(round(v, 3))] = {"A": round(float(np.nanmedian(a_)), 5), "B": round(float(np.nanmedian(b_)), 5)}
        res["marginals"][nm] = mg
        pf = {}
        for era, E in (("A", EA), ("B", EB)):
            rows = entry_rows(E, masks[era][nm])
            sco = score[era][nm][rows]
            pre = trades_of(E, rows, c)
            rr = random_rows(E, CRIT["n_random"], seed=33 if era == "A" else 44)       # 같은 날 아무 종목 계좌 (같은 규칙) 30번
            rpre = trades_of(E, rr, c)
            pf[era] = {}
            for slots in (1, 3):
                me = portfolio(E, rows, sco, c, slots, pre=pre)
                rs = [portfolio(E, rr, np.random.default_rng(sd).random(len(rr[0])), c, slots, pre=rpre) for sd in range(30)]
                cg = np.array([x["cagr"] for x in rs])
                md = np.array([x["mdd"] for x in rs])
                pf[era][f"{slots}자리"] = {**me, "random_cagr_median": round(float(np.median(cg)), 4), "random_cagr_p90": round(float(np.quantile(cg, 0.9)), 4),
                                         "random_mdd_median": round(float(np.median(md)), 4), "beats_random": round(float((cg < me["cagr"]).mean()), 3)}
        res["picks"][nm]["portfolio"] = pf
    res["passed"] = [nm for nm, p in res["picks"].items() if p["pass"]]
    res["n_trials"] = res["n_rules"] * len(ENTRIES)
    res["secs"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for nm, p in res["picks"].items():
        A, B = p["A"], p["B"]
        log(f"■ {nm} {ENTRIES[nm]} — A 로 고른 규칙: {p['desc']}")
        log(f"   A 승률 {A['win']:.0%} 건당 {A['mean']:+.2%} (t {A['t_net']}) 무작위 대비 {A['edge']:+.2%} | "
            f"B {B['n']:,}건 승률 {B['win']:.0%} 건당 {B['mean']:+.2%} (t {B['t_net']}) 무작위 대비 {B['edge']:+.2%} (t {B['t_edge']}) 비용2배 {B['mean_x2']:+.2%} {'✓' if p['pass'] else '✗'}")
        pk, hb = p["peak_A"], p["best_B_hindsight"]
        log(f"   (A 1등 '{pk['desc']}': A {pk['A']['mean']:+.2%} → B {pk['B']['mean']:+.2%}) · (B 를 보고 고른 1등 '{hb['desc']}': B {hb['B']['mean']:+.2%}, A {hb['A']['mean']:+.2%})")
        for era in ("A", "B"):
            for k, q in p["portfolio"][era].items():
                log(f"   20만 원 {era} {k}: 연 {q['cagr']:+.1%} 낙폭 {q['mdd']:.0%} 거래 {q['trades']} 승률 {q['win']:.0%} → {q['final_200k']:,}원 | "
                    f"아무 종목 30번: 연 중앙 {q['random_cagr_median']:+.1%} (상위 10% {q['random_cagr_p90']:+.1%}) 낙폭 중앙 {q['random_mdd_median']:.0%} · 무작위보다 나은 비율 {q['beats_random']:.0%}")
    log(f"통과 {res['passed'] or '없음'} · 규칙 {res['n_rules']:,} × 자리 {len(ENTRIES)} · {res['secs']}초 → {OUT}")


if __name__ == "__main__":
    main()
