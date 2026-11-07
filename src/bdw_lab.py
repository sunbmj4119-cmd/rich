"""
불장단타왕(불단왕) 규칙 검증 — 그의 규칙 중 국내 대형주 일봉으로 옮길 수 있는 것만, 그가 말한 숫자 그대로 (최적화 없음).

  자료: 사용자가 준 '불장단타왕 전체 규칙집'(영상 109·쇼츠 307·라이브 1편 자막에서 뽑은 규칙 약 1,580개)과 '매매 수칙 전집'의
  'rich 적용: 시험할 가설 12개'. 그의 주 무대는 코인·미국 주식 1~5분봉 단타라 대부분은 일봉으로 옮길 수 없다.

사전 등록 (결과를 보기 전에 정함 — 조건·기간·합격선 고정)
  B1  일봉 VWMA100 돌파 뒤 5일 안 되밟기(저가 ≤ 선×1.01) + 종가 선 위 → 다음 날 시가 매수 · 종가가 선 아래면 다음 날 시가 매도 (최대 20일)
  B1s 같은 조건을 단순 100일선으로 (거래량 가중이 의미가 있나)
  B3  VWMA100 돌파 당일(거래량 20일 평균 2배) 바로 매수 — 같은 청산 (돌파 즉시 vs 되밟기)
  B4  정배열(50일선 > 200일선) + 50일선 눌림(저가 ≤ 50선×1.01, 종가 > 50선) → 종가 50선 이탈 시 매도 (최대 20일)
  B5  하루 −5% 이하 + 거래대금 2배 + 긴 아래꼬리(꼬리 ≥ 몸통 2배, 종가가 그날 범위 위쪽 절반) → 10일
  B6  B1 + 시간 손절 (5일째 손실이면 정리)
  B9  지수 −2% 이하인 날 + 그 종목 개인 순매도·외국인 순매수 (둘 다 20일 평균 크기의 2배 이상) → 5일
  B10 역배열(50 < 200) 종목이 아래에서 25일선에 닿음(고가 ≥ 25선, 종가 < 25선) → 10일 — '저항'이면 손실이어야 함 (매수 금지 필터 후보)
  B8  6월 중순(6/18 뒤 첫 거래일) 코스피 매수 → 9월 말 매도, 같은 길이 무작위 구간과 비교 (지수, 1997~)
  비용·합격선은 setup_lab 과 같다 (왕복 0.34% · 두 시대 건당 t ≥ 3 · 시장 대비 t ≥ 2 · 100건 · 비용 2배 플러스).
  B10 은 반대로: 두 시대 모두 건당 순수익 < 0 이고 t ≤ −3 이면 '매수 금지 필터'로 통과.
  B8 은: 두 시대 모두 무작위 구간보다 평균이 높고 순열 p < 0.05.

  python src/bdw_lab.py → data/bdw_lab.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_lab as SL  # noqa: E402

OUT = "data/bdw_lab.json"
TESTS = {
    "B1": "VWMA100 돌파 뒤 되밟기 매수 · 종가 이탈 매도",
    "B1s": "같은 조건, 단순 100일선",
    "B3": "VWMA100 돌파 당일(거래량 2배) 매수 · 종가 이탈 매도",
    "B4": "정배열 + 50일선 눌림 매수 · 50선 이탈 매도",
    "B5": "−5% 투매 + 거래대금 2배 + 긴 아래꼬리 → 10일",
    "B6": "B1 + 시간 손절(5일째 손실이면 정리)",
    "B9": "지수 −2% 날 개인 투매·외국인 받기 → 5일",
    "B10": "역배열 25일선 저항 (매수 금지 필터 후보) → 10일",
}


def lines(P):
    C = P["Cf"] if "Cf" in P else pd.DataFrame(P["C"]).ffill().to_numpy()
    V = np.nan_to_num(P["V"])
    cv = pd.DataFrame(C * V).rolling(100, min_periods=80).sum().to_numpy()
    vv = pd.DataFrame(V).rolling(100, min_periods=80).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwma = np.where(vv > 0, cv / vv, np.nan)
    return C, {"vwma": vwma, "sma100": SL.roll(C, 100), "ma25": SL.roll(C, 25), "ma50": SL.roll(C, 50), "ma200": SL.roll(C, 200)}


def signals(P, idx):
    C, Ln = lines(P)
    H, Lo, O = P["H"], P["L"], P["O"]
    raw = P["C"]
    V, VAL = P["V"], P["VAL"]
    prevC = pd.DataFrame(C).shift(1).to_numpy()
    r1 = raw / pd.DataFrame(raw).shift(1).to_numpy() - 1
    v20 = pd.DataFrame(V).rolling(20, min_periods=15).mean().shift(1).to_numpy()
    val20 = pd.DataFrame(VAL).rolling(20, min_periods=15).mean().shift(1).to_numpy()
    S, X = {}, {}
    with np.errstate(invalid="ignore"):
        for nm, line in (("B1", Ln["vwma"]), ("B1s", Ln["sma100"])):
            prevL = pd.DataFrame(line).shift(1).to_numpy()
            cross = (C > line) & (prevC <= prevL)
            recent = pd.DataFrame(cross).rolling(5, min_periods=1).max().shift(1).fillna(0).to_numpy() > 0   # 지난 5일 안에 돌파
            S[nm] = recent & ~cross & (Lo <= line * 1.01) & (C > line)
            X[nm] = C < line                                                   # 종가 이탈 = 다음 날 시가 매도
        prevL = pd.DataFrame(Ln["vwma"]).shift(1).to_numpy()
        S["B3"] = (C > Ln["vwma"]) & (prevC <= prevL) & (V >= 2 * v20)
        X["B3"] = C < Ln["vwma"]
        S["B4"] = (Ln["ma50"] > Ln["ma200"]) & (Lo <= Ln["ma50"] * 1.01) & (C > Ln["ma50"])
        X["B4"] = C < Ln["ma50"]
        body = np.abs(raw - O)
        lower = np.minimum(raw, O) - Lo
        upper_half = (raw - Lo) >= 0.5 * (H - Lo)
        S["B5"] = (r1 <= -0.05) & (VAL >= 2 * val20) & (lower >= 2 * body) & upper_half
        S["B6"] = S["B1"]
        X["B6"] = X["B1"]
        m = pd.Series(idx)
        mret = (m / m.shift(1) - 1).to_numpy()[:, None]
        F, I = np.asarray(P["FOR"], float), np.asarray(P["IND"], float)
        fa = pd.DataFrame(np.abs(F)).rolling(20, min_periods=15).mean().shift(1).to_numpy()
        ia = pd.DataFrame(np.abs(I)).rolling(20, min_periods=15).mean().shift(1).to_numpy()
        S["B9"] = (mret <= -0.02) & (I <= -2 * ia) & (F >= 2 * fa) & np.isfinite(F) & np.isfinite(I)
        S["B10"] = (Ln["ma50"] < Ln["ma200"]) & (H >= Ln["ma25"]) & (C < Ln["ma25"]) & (prevC < pd.DataFrame(Ln["ma25"]).shift(1).to_numpy())
    S = {k: np.nan_to_num(v, nan=False).astype(bool) & P["U"] for k, v in S.items()}
    X = {k: np.nan_to_num(v, nan=False).astype(bool) for k, v in X.items()}
    return S, X


HOLD = {"B1": 20, "B1s": 20, "B3": 20, "B4": 20, "B5": 10, "B6": 20, "B9": 5, "B10": 10}


def trades_dyn(P, idx, S, X, max_h, a, b, cost_mult=1.0, time_stop=None):
    """신호 다음 날 시가 진입 → (근거 이탈이면 다음 날 시가) 또는 max_h 일 뒤 시가 청산. time_stop=5 면 5일째 종가가 매입가 아래면 다음 날 시가."""
    O = P["O"]
    C = P["Cf"] if "Cf" in P else pd.DataFrame(P["C"]).ffill().to_numpy()
    T, N = O.shape
    dates = P["dates"]
    brk = P.get("breaks")
    cumbrk = np.cumsum(brk, axis=0) if brk is not None else None
    cb, cs = SL.COMM + SL.SLIP * cost_mult, SL.COMM + SL.SLIP * cost_mult + SL.TAX
    i0 = int(np.searchsorted(dates, pd.Timestamp(a)))
    i1 = len(dates) if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
    rows, busy = [], np.full(N, -1)
    for t in range(i0, min(i1, T - max_h - 1)):
        for j in np.where(S[t] & (busy < t))[0]:
            e = t + 1
            po = O[e, j]
            if not (np.isfinite(po) and po > 0):
                continue
            x = e + max_h
            if X is not None:
                for k in range(e, e + max_h):
                    if X[k, j]:
                        x = k + 1
                        break
                    if time_stop and k == e + time_stop - 1 and C[k, j] < po:
                        x = k + 1
                        break
            x = min(x, T - 1)
            px = O[x, j]
            if not np.isfinite(px):
                continue
            if cumbrk is not None and cumbrk[x, j] - cumbrk[t, j] > 0:
                continue
            net = px * (1 - cs) / (po * (1 + cb)) - 1
            mret = idx[x] / idx[t] - 1 if np.isfinite(idx[t]) and idx[t] > 0 else np.nan
            rows.append((dates[e], j, net, mret, x - e, e, x))
            busy[j] = x
    return pd.DataFrame(rows, columns=["date", "j", "net", "mkt", "days", "e", "x"])


def one_slot_trades(P, tr, a, b, cash0=200_000):
    """20만 원 '한 번에 한 종목': 비어 있으면 그날 진입하는 거래 중 시총이 큰 것, 청산일까지 들고 간다 (매일 평가)."""
    if tr.empty:
        return None
    C = P["Cf"] if "Cf" in P else pd.DataFrame(P["C"]).ffill().to_numpy()
    MC = np.nan_to_num(P["MC"])
    O = P["O"]
    dates = P["dates"]
    by_e = {e: g for e, g in tr.groupby("e")}
    i0 = int(np.searchsorted(dates, pd.Timestamp(a)))
    i1 = len(dates) if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
    eq, E, t, n = float(cash0), [], i0, 0
    cb = SL.COMM + SL.SLIP
    while t < i1 - 1:
        g = by_e.get(t)
        if g is not None:
            r = g.loc[g["j"].map(lambda j: MC[t - 1, j]).idxmax()]
            j, x = int(r["j"]), int(r["x"])
            units = eq / (O[t, j] * (1 + cb))
            for k in range(t, min(x, i1 - 1)):
                E.append((dates[k], units * C[k, j]))
            eq = eq * (1 + float(r["net"]))
            n += 1
            t = x
            continue
        E.append((dates[t], eq))
        t += 1
    s = pd.Series(dict(E)).sort_index()
    r = s.pct_change().dropna()
    if len(r) < 60:
        return None
    yrs = len(r) / 252
    return {"cagr": round(float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1), 4), "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3) if r.std() > 0 else 0,
            "mdd": round(float((s / s.cummax() - 1).min()), 4), "trades": n}


def june_test():
    """6월 중순 매수 → 9월 말 매도 (코스피, 1997~) vs 같은 해 같은 길이 무작위 구간."""
    import history_lab as H
    s = H.series(H.load_hist(), "^KS11").dropna()
    rng = np.random.default_rng(7)
    out = {}
    for era, (a, b) in {"A": (1997, 2018), "B": (2019, 2025)}.items():
        ex, rows = [], []
        for y in range(a, b + 1):
            ys = s[(s.index >= f"{y}-01-01") & (s.index <= f"{y}-12-31")]
            st = ys[ys.index > f"{y}-06-18"]
            en = ys[ys.index <= f"{y}-09-30"]
            if st.empty or en.empty:
                continue
            i, k = ys.index.get_loc(st.index[0]), ys.index.get_loc(en.index[-1])
            n = k - i
            r = ys.iloc[k] / ys.iloc[i] - 1
            starts = rng.integers(0, len(ys) - n - 1, 200)
            rr = np.array([ys.iloc[q + n] / ys.iloc[q] - 1 for q in starts])
            ex.append(r - rr.mean())
            rows.append((y, round(float(r), 4), round(float(rr.mean()), 4)))
        ex = np.array(ex)
        perm = np.array([np.mean(ex * rng.choice([-1, 1], len(ex))) for _ in range(5000)])
        out[era] = {"years": rows, "mean_excess": round(float(ex.mean()), 4), "win": round(float((ex > 0).mean()), 3),
                    "p": round(float((perm >= ex.mean()).mean()), 4), "n": int(len(ex))}
    out["pass"] = all(out[e]["mean_excess"] > 0 and out[e]["p"] < 0.05 for e in ("A", "B"))
    return out


def main():
    t0 = time.time()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "tests": TESTS, "hold": HOLD, "criteria": SL.CRIT, "rows": {},
           "n_trials": len(TESTS) + 1}
    cache = {}
    for era in ("A", "B"):
        P, idx = SL.panel(era)
        S, X = signals(P, idx)
        cache[era] = (P, idx, S, X)
        a, b = SL.ERA[era]
        for k in TESTS:
            ts = 5 if k == "B6" else None
            tr = trades_dyn(P, idx, S[k], X.get(k), HOLD[k], a, b, time_stop=ts)
            tr2 = trades_dyn(P, idx, S[k], X.get(k), HOLD[k], a, b, cost_mult=2.0, time_stop=ts)
            row = {"n": int(len(tr))}
            if len(tr):
                tn, nm = SL.tstat_month(tr["net"], tr["date"])
                te, _ = SL.tstat_month(tr["net"] - tr["mkt"], tr["date"])
                row.update({"win": round(float((tr["net"] > 0).mean()), 3), "mean": round(float(tr["net"].mean()), 4),
                            "median": round(float(tr["net"].median()), 4), "excess": round(float((tr["net"] - tr["mkt"]).mean()), 4),
                            "t_net": tn, "t_excess": te, "months": nm, "worst": round(float(tr["net"].min()), 3),
                            "days": round(float(tr["days"].mean()), 1), "mean_x2": round(float(tr2["net"].mean()), 4) if len(tr2) else None,
                            "per_year": round(len(tr) / max(1, len(set(pd.DatetimeIndex(tr["date"]).year))), 1)})
            res["rows"].setdefault(k, {})[era] = row
            cache[(era, k)] = tr
    for k in TESTS:
        ok, why = True, []
        for era in ("A", "B"):
            r = res["rows"][k][era]
            if r["n"] < SL.CRIT["min_n"]:
                ok = False
                why.append(f"{era} {r['n']}건")
                continue
            if k == "B10":
                if not (r["mean"] < 0 and (r["t_net"] or 0) <= -SL.CRIT["t_net"]):
                    ok = False
                    why.append(f"{era} 건당 {r['mean']:+.2%} t={r['t_net']} (저항이면 음수여야)")
                continue
            if not (r["mean"] > 0 and (r["t_net"] or 0) >= SL.CRIT["t_net"]):
                ok = False
                why.append(f"{era} 건당 {r['mean']:+.2%} t={r['t_net']}")
            if not ((r["t_excess"] or 0) >= SL.CRIT["t_excess"] and r["excess"] > 0):
                ok = False
                why.append(f"{era} 시장 대비 {r['excess']:+.2%} t={r['t_excess']}")
            if not ((r["mean_x2"] or -1) > 0):
                ok = False
                why.append(f"{era} 비용 2배 {r['mean_x2']}")
        res["rows"][k]["pass"] = ok
        res["rows"][k]["why"] = "통과" if ok else "; ".join(why[:3])
    res["june"] = june_test()
    res["passed"] = [k for k in TESTS if res["rows"][k]["pass"]] + (["B8"] if res["june"]["pass"] else [])
    res["one_slot"] = {}
    for k in [k for k in TESTS if k != "B10"]:
        for era in ("A", "B"):
            P = cache[era][0]
            a, b = SL.ERA[era]
            res["one_slot"].setdefault(k, {})[era] = one_slot_trades(P, cache[(era, k)], a, b)
    print("■ 불장단타왕 규칙 (건수 · 승률 · 건당 순수익 (t) · 시장 대비 (t) · 평균 보유일 · 비용 2배)")
    for k, name in TESTS.items():
        r = res["rows"][k]
        print(f"  {k:4s} {name[:30]:30s} | " + " | ".join(
            f"{e} {r[e]['n']:5d}건 {r[e].get('win', 0):.0%} {r[e].get('mean', 0):+.2%} ({r[e].get('t_net')}) 시장대비 {r[e].get('excess', 0):+.2%} ({r[e].get('t_excess')}) {r[e].get('days')}일 2배 {r[e].get('mean_x2') or 0:+.2%}"
            for e in ("A", "B")) + f"  {'✓' if r['pass'] else '✗'}")
    j = res["june"]
    print(f"  B8   6월 중순 매수 → 9월 말 | A 무작위 대비 {j['A']['mean_excess']:+.2%} (p {j['A']['p']}, {j['A']['n']}년) | B {j['B']['mean_excess']:+.2%} (p {j['B']['p']}, {j['B']['n']}년)  {'✓' if j['pass'] else '✗'}")
    print("■ 20만 원 '한 번에 한 종목'")
    for k, v in res["one_slot"].items():
        print(f"  {k:4s} " + " | ".join(f"{e} 연 {x['cagr']:+.1%} 샤프 {x['sharpe']} 낙폭 {x['mdd']:.0%} {x['trades']}번" for e, x in v.items() if x))
    res["secs"] = round(time.time() - t0)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"끝 ({res['secs']}초) · 통과 {res['passed'] or '없음'} → {OUT}")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ══════════════════════════════════════════════════════════════
# 2단계 (1단계 결과를 본 뒤, 시도로 셈) — 그의 '근거가 깨지면 바로 자른다' + '100일 기준선'을 실제 돈 엔진(DM:2, ETF)에
#   매달 DM:2 가 고른 ETF 를 들고 있다가, 종가가 100일선 아래로 마감하면 다음 날 시가에 그 몫을 단기채로 (그달 남은 기간).
#   다음 달 첫 거래일에 다시 DM:2 를 따르되, 고른 ETF 가 이미 100일선 아래면 처음부터 단기채.
#   합격 (4회차와 같음): 두 시대 모두 DM:2 단독보다 샤프 높고 최대낙폭이 나쁘지 않음, 비용 1배·3배 모두.
# ══════════════════════════════════════════════════════════════
def dm2_stop_targets(dates, Cp, W, cash_j, n=100):
    import allweather as AW
    c = AW.ffill(Cp)
    sma = pd.DataFrame(c).rolling(n, min_periods=int(n * 0.8)).mean().to_numpy()
    below = c < sma                                        # t일 저녁에 아는 것
    month = AW.month_start(dates)
    T, A = W.shape
    out = np.full((T, A), np.nan)
    cur = None
    for t in range(1, T):
        if month[t] or cur is None:
            cur = W[t].copy()
            if not np.isfinite(cur).all():
                cur = None
                continue
            for j in np.where(cur > 0)[0]:
                if j != cash_j and below[t - 1, j]:
                    cur[cash_j] += cur[j]
                    cur[j] = 0.0
            out[t] = cur
            continue
        hit = [j for j in np.where(cur > 0)[0] if j != cash_j and below[t - 1, j]]
        if hit:
            for j in hit:
                cur[cash_j] += cur[j]
                cur[j] = 0.0
            out[t] = cur.copy()
    return out


def dm2_stop_test():
    import allweather as AW
    import history_lab as H
    import optimize_lab as O
    import size_policy as SZ
    cash_j = AW.CODES.index("153130")
    port = {}
    for era in ("A", "B"):
        a_, b_ = O.PORT_ERA[era]
        for tag, k in (("x1", 1.0), ("x3", 3.0)):
            if era == "A":
                dates, Op, Cp = H.proxy_prices(H.load_hist())
                ctx = H.flat_slip()
            else:
                dates, Op, Cp = AW.load_prices()
                import contextlib
                ctx = contextlib.nullcontext()
            with ctx:
                W_aw, W_ag, s = SZ.weight_paths(dates, Op, Cp, start="2005-01-03" if era == "A" else None, stress=k)
                base = AW.simulate(Op, Cp, np.where(AW.month_start(dates)[:, None], W_ag, np.nan), t0=s, stress=k)["E"]
                stop = AW.simulate(Op, Cp, dm2_stop_targets(dates, Cp, W_ag, cash_j), t0=s, stress=k)["E"]
            for nm, E in (("DM:2 단독", base), ("DM:2 + 100일선 이탈 손절", stop)):
                x = pd.Series(np.r_[0.0, E[1:] / E[:-1] - 1], index=dates)
                x = x[(x.index >= pd.Timestamp(a_)) & ((x.index <= pd.Timestamp(b_)) if b_ else True)]
                st = O.sstats(x.to_numpy(), x.index)
                if tag == "x1":
                    st.update(O.steadiness(x.to_numpy(), x.index))
                port.setdefault(nm, {}).setdefault(era, {})[tag] = st
    ok, why = True, []
    for era in ("A", "B"):
        for tag in ("x1", "x3"):
            me, b = port["DM:2 + 100일선 이탈 손절"][era][tag], port["DM:2 단독"][era][tag]
            if me["sharpe"] <= b["sharpe"]:
                ok = False
                why.append(f"{era}·{tag} 샤프 {me['sharpe']} ≤ {b['sharpe']}")
            if me["mdd"] < b["mdd"]:
                ok = False
                why.append(f"{era}·{tag} 낙폭 {me['mdd']} < {b['mdd']}")
    return {"rows": port, "pass": ok, "why": "통과" if ok else "; ".join(why[:4])}


def stage2():
    res = json.load(open(OUT, encoding="utf-8"))
    r = dm2_stop_test()
    res["dm2_stop"] = r
    res["n_trials"] = len(TESTS) + 2
    if r["pass"]:
        res["passed"] = sorted(set(res["passed"]) | {"DM2_STOP"})
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for n, v in r["rows"].items():
        print(f"  {n:22s} " + " | ".join(f"{e} 연 {v[e]['x1']['cagr']:+.1%} 샤프 {v[e]['x1']['sharpe']:.2f} 낙폭 {v[e]['x1']['mdd']:.1%} (3배 {v[e]['x3']['sharpe']:.2f}) 오른 달 {v[e]['x1']['month_win']:.0%}" for e in ("A", "B")))
    print(f"  판정: {'✓ 통과' if r['pass'] else '✗ ' + r['why']}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "stage2":
    stage2()
