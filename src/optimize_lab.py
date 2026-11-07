"""
최적화 실험실 — 실제 돈을 넣기 전에 '여러 모델을 합치거나 손실 방지 장치를 덧대면 진짜 나아지는가'를 미리 정한 기준으로 판정한다.

사전 등록 (결과를 보기 전에 정한 규칙 — 바꾸면 기록에 남는다)
  두 시대 교차 확인
    A: 2002~2018 KRX 그때의 시총 상위 100 (상장폐지 포함 · history_lab.krx_panel) · 전천후는 대용 자산 2005~
    B: 2019~지금 실제 자료 (단기 패널 · 실제 ETF)
    → 한 시대에서만 좋은 것은 탈락. 후보는 두 시대의 결과를 보지 않고 정의한다 (앙상블·규칙은 '그때까지의 자료'만 쓴다).
  1단계 — 종목 몫 후보: 두 시대 모두 코스피보다 샤프가 높고, 최대낙폭이 지수보다 5%p 넘게 나쁘지 않고,
          두 시대를 이어 붙인 블록 부트스트랩 p(샤프 차이 ≤ 0) 를 Holm 보정해 < 0.05
  2단계 — 전체 계좌 (전천후 + 1단계 통과 종목 몫): 지금 실제 돈 계획(전천후 16장 단독)과 비교해 두 시대 모두
          샤프가 같거나 높고 · 연수익이 높고 · 최대낙폭이 -15% 안쪽 · 미끄러짐 3배에서도 같은 조건
  반복은 최대 3회. 모든 시도는 data/optimize/lab.json 의 n_trials 로 시도 장부(trials.py)에 들어간다.

  python src/optimize_lab.py [--round 1]  → data/optimize/lab.json · docs/optimize.json
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allweather as AW  # noqa: E402
import short_lab as L  # noqa: E402

OUT = "data/optimize/lab.json"
DOCS = "docs/optimize.json"
ERA = {"A": ("2002-01-02", "2018-12-28"), "B": ("2019-01-02", None)}
PORT_ERA = {"A": ("2005-01-03", "2018-12-28"), "B": ("2019-01-02", None)}
CRIT = {"s1_p": 0.05, "s1_mdd_slack": 0.05, "s2_mdd_floor": -0.15, "s2_sharpe_slack": 0.0, "max_rounds": 3}
FAMILY = {"REV": ["REV1", "REVID", "REVC", "REVC_LV", "REV3_LV", "GAP", "VSD", "MA20_LV"], "MOM": ["MOM"], "FLOW": ["FOR20", "REVC_FLOW"]}
ONE_WAY = 0.0017                       # 주식 한쪽 평균 비용 (매수 0.07% · 매도 0.27% 의 평균) — 덮개가 비중을 바꿀 때


def cut(dates, a, b=None):
    i = int(np.searchsorted(dates, pd.Timestamp(a)))
    j = len(dates) if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
    return i, j


def sstats(r, dates):
    """일별 수익 → 연수익·변동성·샤프·최대낙폭·칼마·최악의 해·하위 5% 하루 평균(CVaR)."""
    r = pd.Series(np.nan_to_num(np.asarray(r, float)), index=dates)
    if len(r) < 60:
        return None
    E = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = E.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    mdd = float((E / E.cummax() - 1).min())
    yr = E.groupby(E.index.year).last()
    yret = yr / yr.shift(1).fillna(1.0) - 1
    q = r.quantile(0.05)
    return {"cagr": round(float(cagr), 4), "vol": round(float(vol), 4), "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0, 3),
            "mdd": round(mdd, 4), "calmar": round(float(cagr / -mdd), 3) if mdd < 0 else None, "worst_year": round(float(yret.min()), 4),
            "cvar5": round(float(r[r <= q].mean()), 4)}


# ══════════════════════════════════════════════════════════════
# 종목 몫 후보 (시대마다 같은 정의)
# ══════════════════════════════════════════════════════════════
def era_data(era):
    if era == "A":
        import history_lab as H
        P = H.krx_panel()
        if P is None:
            raise SystemExit("KRX 긴 역사 자료가 없습니다 (data/history/krx_*.csv.gz)")
        D = L.Data(P)
        idx = P["bench"]["^KS11"]
    else:
        D = L.Data()
        idx = AW.ffill(D.Cp)[:, 0]
    rb = np.r_[0.0, idx[1:] / idx[:-1] - 1]
    return D, np.nan_to_num(rb)


def stock_candidates(models):
    """이름 → (종류, 정의). 종류 ens = 정해진 모델 묶음을 같은 비중 · rule = 그때까지 성적으로 고르기."""
    slow = [i for i, m in enumerate(models) if m["kind"] == "stock" and m["hold"] in ("H5", "H10")]
    c = {"S0 지금 규칙 (250일 샤프 1등·매주)": ("rule", dict(L.LIVE_RULE)),
         "S1 느린 종목 모델 전부 같은 비중": ("ens", slow),
         "S2 급락장 제외 모델만 전부": ("ens", [i for i in slow if models[i]["gate"] == "crash"]),
         "S3 120일선 위 모델만 전부": ("ens", [i for i in slow if models[i]["gate"] == "trend"]),
         "S4 250일 샤프 상위 5개 혼합·매주": ("rule", {**L.LIVE_RULE, "k": 5}),
         "S5 250일 샤프 상위 10개 혼합·매주": ("rule", {**L.LIVE_RULE, "k": 10})}
    for tag, (fam, sels) in zip(("S6", "S7", "S8"), FAMILY.items()):
        c[f"{tag} {fam} 계열 전부"] = ("ens", [i for i in slow if models[i]["sel"] in sels])
    return c


def run_era(era, stress=1.0):
    import history_lab as H
    import contextlib
    with (H.flat_slip() if era == "A" else contextlib.nullcontext()):   # A 시대 ETF 는 대용 가격 → 자산별 고정 미끄러짐 (긴 역사 검증과 같게)
        return _run_era(era, stress)


def _run_era(era, stress=1.0):
    D, rb = era_data(era)
    models = L.stock_models() + [m for m in L.etf_models() if m["rule"] not in ("overnight_for",)]
    recs = [L.run(D, m, record=True, stress=stress) for m in models]
    R = np.c_[[L.rets(r["E"]) for r in recs]].T
    a, b = ERA[era]
    st, e = cut(D.dates, a, b)
    out = {}
    gate = set() if era == "A" else None                              # A 시대는 관문이 그 시대 자료로 만든 것이라 쓰지 않는다
    for name, (kind, spec) in stock_candidates(models).items():
        if kind == "rule":
            pool = L.pool_index(models, spec["pool"], gate)
            ch = L.choose(R[:, pool], st, **spec)
            ch = [None if x is None else (pool[x] if isinstance(x, int) else [pool[i] for i in x]) for x in ch]
        else:
            ch = [None] * D.T
            for t in range(st, D.T):
                ch[t] = list(spec)
        E, sw = L.follow(D, models, ch, recs, stress=stress, t0=st)
        r = L.rets(E)
        out[name] = {"r": r, "switches": sw}
    return {"dates": D.dates, "st": st, "e": e, "rb": rb, "streams": out, "models": models, "recs": recs, "D": D}


# ══════════════════════════════════════════════════════════════
# 손실 방지 덮개 (그날 아침에 아는 정보로만 비중을 정한다)
# ══════════════════════════════════════════════════════════════
def overlay(r, kind, x, idx=None):
    r = np.nan_to_num(np.asarray(r, float))
    T = len(r)
    e = np.ones(T)
    if kind == "vt":                                   # 변동성 목표: 비중 = min(1, 목표 ÷ 최근 20일 변동성)
        v = pd.Series(r).rolling(20, min_periods=15).std().shift(1).to_numpy() * np.sqrt(252)
        e = np.where(np.isfinite(v) & (v > 0), np.minimum(1.0, x / v), 1.0)
    elif kind == "ddb":                                # 낙폭 브레이크: 원래 전략이 고점 대비 -x 면 현금, -x/2 까지 회복하면 복귀
        U = np.cumprod(1 + r)
        peak = np.maximum.accumulate(U)
        dd = U / peak - 1
        on = True
        for t in range(1, T):
            d = dd[t - 1]
            if on and d <= -x:
                on = False
            elif not on and d >= -x / 2:
                on = True
            e[t] = 1.0 if on else 0.0
    elif kind == "mkt":                                # 지수가 200일선 아래면 비중 x (0 이면 현금)
        s = pd.Series(idx)
        ma = s.rolling(200, min_periods=150).mean()
        ok = (s > ma).shift(1).fillna(True).to_numpy()
        e = np.where(ok, 1.0, x)
    turn = np.abs(np.diff(np.r_[1.0, e]))
    return e * r - turn * ONE_WAY, float(e.mean())


# ══════════════════════════════════════════════════════════════
# 전천후 몫 (시대 A = 대용 자산, B = 실제 ETF) · 전체 계좌
# ══════════════════════════════════════════════════════════════
def aw_stream(era, stress=1.0):
    import history_lab as H
    if era == "A":
        h = H.load_hist()
        dates, Op, Cp = H.proxy_prices(h)
        with H.flat_slip():
            m = AW.live_mix(prices=(dates, Op, Cp), start="2005-01-03", stress=stress)
    else:
        m = AW.live_mix(stress=stress)
    return pd.Series(L.rets(m["E"]), index=m["dates"])


def combine(r_s, r_aw, share, start, end=None, cost=0.0025):
    import history_lab as H
    rr, _ = H.combine(r_s, r_aw, start, share=share, cost=cost)
    return rr if end is None else rr[rr.index <= pd.Timestamp(end)]


# ══════════════════════════════════════════════════════════════
# 판정
# ══════════════════════════════════════════════════════════════
def holm(ps):
    import science_lab as S
    return S.holm(ps)


def stage1(eras, cands):
    rows = {}
    for name in cands:
        row = {}
        pooled_a, pooled_b = [], []
        for era, res in eras.items():
            s, e = res["st"], res["e"]
            d = res["dates"][s:e]
            r = res["streams"][name]["r"][s:e]
            rb = res["rb"][s:e]
            row[era] = {"me": sstats(r, d), "idx": sstats(rb, d), "switches": res["streams"][name]["switches"]}
            pooled_a.append(r)
            pooled_b.append(rb)
        row["p"] = L.boot_p(np.concatenate(pooled_a), np.concatenate(pooled_b))
        rows[name] = row
    names = list(rows)
    for n, adj in zip(names, holm([rows[n]["p"] if rows[n]["p"] is not None else 1.0 for n in names])):
        rows[n]["p_holm"] = adj
        ok = all(rows[n][era]["me"]["sharpe"] > rows[n][era]["idx"]["sharpe"] and
                 rows[n][era]["me"]["mdd"] >= rows[n][era]["idx"]["mdd"] - CRIT["s1_mdd_slack"] for era in eras)
        rows[n]["pass"] = bool(ok and adj < CRIT["s1_p"])
        rows[n]["why"] = ("통과" if rows[n]["pass"] else
                          "; ".join([f"{era}: 샤프 {rows[n][era]['me']['sharpe']} ≤ 지수 {rows[n][era]['idx']['sharpe']}"
                                     for era in eras if rows[n][era]["me"]["sharpe"] <= rows[n][era]["idx"]["sharpe"]] +
                                    ([f"p(Holm) {adj} ≥ {CRIT['s1_p']}"] if adj >= CRIT["s1_p"] else [])))
    return rows


def stage2(port, base_key):
    out = {}
    for name, eras in port.items():
        if name == base_key:
            continue
        ok, why = True, []
        for era, v in eras.items():
            b = port[base_key][era]
            for tag in ("x1", "x3"):
                me, bb = v[tag], b[tag]
                if me["sharpe"] < bb["sharpe"] - CRIT["s2_sharpe_slack"]:
                    ok = False
                    why.append(f"{era}·{tag} 샤프 {me['sharpe']} < 기준 {bb['sharpe']}")
                if me["cagr"] <= bb["cagr"]:
                    ok = False
                    why.append(f"{era}·{tag} 연수익 {me['cagr']} ≤ 기준 {bb['cagr']}")
                if me["mdd"] < CRIT["s2_mdd_floor"]:
                    ok = False
                    why.append(f"{era}·{tag} 낙폭 {me['mdd']} < {CRIT['s2_mdd_floor']}")
        out[name] = {"pass": ok, "why": "통과" if ok else "; ".join(why[:4])}
    return out


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in ("r",)}
    return x


# ══════════════════════════════════════════════════════════════
# 2회차 (1회차 결과를 보기 전에 정함) — ETF 만으로: 전천후 + 주식 쪽 위성 카드 1장
#   종목 앙상블은 수십~수백 종목이라 작은 계좌로는 따라 할 수 없다 → ETF 만으로 수익을 올릴 수 있는지
# ══════════════════════════════════════════════════════════════
SATS = ("T:def", "DM:2", "VT", "T:cash")
SAT_SHARES = (0.8, 0.7, 0.6)


def card_streams(era, stress=1.0):
    """전천후 16장 혼합 + 위성 카드 4장의 일별 수익 (시대 A = 대용 자산 2005~, B = 실제 ETF)."""
    import history_lab as H
    if era == "A":
        dates, Op, Cp = H.proxy_prices(H.load_hist())
        with H.flat_slip():
            m = AW.live_mix(prices=(dates, Op, Cp), start="2005-01-03", stress=stress)
            F = AW.features(Cp, dates)
            rules, _, _, R = AW.card_returns(dates, Op, Cp, F, rules=list(SATS), stress=stress)
    else:
        m = AW.live_mix(stress=stress)
        dates, Op, Cp = m["dates"], m["Op"], m["Cp"]
        rules, _, _, R = AW.card_returns(dates, Op, Cp, m["F"], rules=list(SATS), stress=stress)
    out = {"AW": pd.Series(L.rets(m["E"]), index=m["dates"])}
    for i, k in enumerate(rules):
        out[k] = pd.Series(np.nan_to_num(R[:, i]), index=dates)
    return out


def round2():
    port = {}
    base = "P0 전천후 16장 단독"
    for era in ("A", "B"):
        a_, b_ = PORT_ERA[era]
        for tag, k in (("x1", 1.0), ("x3", 3.0)):
            cs = card_streams(era, k)
            aw = cs["AW"]
            sub = aw[(aw.index >= pd.Timestamp(a_)) & ((aw.index <= pd.Timestamp(b_)) if b_ else True)]
            port.setdefault(base, {}).setdefault(era, {})[tag] = sstats(sub.to_numpy(), sub.index)
            for sat in SATS:
                for share in SAT_SHARES:
                    nm = f"전천후 {int(share * 100)} + {sat} {int(round((1 - share) * 100))}"
                    rr = combine(cs[sat], aw, share, a_, b_)
                    port.setdefault(nm, {}).setdefault(era, {})[tag] = sstats(rr.to_numpy(), rr.index)
            for sat in SATS:                                           # 참고: 위성 카드 단독
                x = cs[sat]
                x = x[(x.index >= pd.Timestamp(a_)) & ((x.index <= pd.Timestamp(b_)) if b_ else True)]
                port.setdefault(f"(참고) {sat} 단독", {}).setdefault(era, {})[tag] = sstats(x.to_numpy(), x.index)
    s2 = stage2({k: v for k, v in port.items() if not k.startswith("(참고)")}, base)
    for n, v in port.items():
        print(f"  {n:24s} " + " | ".join(f"{era} 샤프 {v[era]['x1']['sharpe']:.3f} 연 {v[era]['x1']['cagr']:+.1%} 낙폭 {v[era]['x1']['mdd']:.1%}" for era in ("A", "B"))
              + (f"  {'✓' if s2[n]['pass'] else '✗'} {s2[n]['why'][:70]}" if n in s2 else ""))
    return port, s2


# ══════════════════════════════════════════════════════════════
# 3회차 (1·2회차 결과를 보기 전에 정함) — 종목마다 손절·익절을 붙인 '느린 종목 모델 전부 같은 비중'(S1)
# ══════════════════════════════════════════════════════════════
STOPS = {"S1+손절7%": {"sl": 0.07}, "S1+손절10%": {"sl": 0.10}, "S1+익절10%": {"tp": 0.10}, "S1+손절7%·익절15%": {"sl": 0.07, "tp": 0.15}}


def stop_streams(era, stress=1.0):
    import history_lab as H
    import contextlib
    with (H.flat_slip() if era == "A" else contextlib.nullcontext()):
        return _stop_streams(era, stress)


def _stop_streams(era, stress=1.0):
    D, rb = era_data(era)
    slow = [m for m in L.stock_models() if m["hold"] in ("H5", "H10")]
    a, b = ERA[era]
    st, e = cut(D.dates, a, b)
    out = {}
    for name, extra in {"S1 (손절·익절 없음)": {}, **STOPS}.items():
        ms = [{**m, **extra} for m in slow]
        recs = [L.run(D, m, record=True, stress=stress) for m in ms]
        ch = [None] * D.T
        for t in range(st, D.T):
            ch[t] = list(range(len(ms)))
        E, sw = L.follow(D, ms, ch, recs, stress=stress, t0=st)
        out[name] = {"r": L.rets(E), "switches": sw}
        print(f"    {era} {name} 끝", flush=True)
    return {"dates": D.dates, "st": st, "e": e, "rb": rb, "streams": out}


def round3():
    eras = {era: stop_streams(era) for era in ("A", "B")}
    s1 = stage1(eras, list(eras["A"]["streams"]))
    for n, v in s1.items():
        print(f"  {n:22s} A {v['A']['me']['sharpe']:6.3f}/{v['A']['idx']['sharpe']:5.3f} 낙폭 {v['A']['me']['mdd']:7.1%} | "
              f"B {v['B']['me']['sharpe']:6.3f}/{v['B']['idx']['sharpe']:5.3f} 낙폭 {v['B']['me']['mdd']:7.1%} | p→{v['p_holm']} {'✓' if v['pass'] else '✗'}")
    port, s2 = {}, {}
    passed = [n for n, v in s1.items() if v["pass"]]
    if passed:
        eras3 = {era: stop_streams(era, 3.0) for era in ("A", "B")}
        base = "P0 전천후 16장 단독"
        for era in ("A", "B"):
            a_, b_ = PORT_ERA[era]
            for tag, k, src in (("x1", 1, eras), ("x3", 3, eras3)):
                r_aw = aw_stream(era, stress=float(k))
                sub = r_aw[(r_aw.index >= pd.Timestamp(a_)) & ((r_aw.index <= pd.Timestamp(b_)) if b_ else True)]
                port.setdefault(base, {}).setdefault(era, {})[tag] = sstats(sub.to_numpy(), sub.index)
                for n in passed:
                    rs = pd.Series(src[era]["streams"][n]["r"], index=src[era]["dates"])
                    for share in (0.8, 0.6):
                        nm = f"전천후 {int(share * 100)} + {n} {int(round((1 - share) * 100))}"
                        rr = combine(rs, r_aw, share, a_, b_)
                        port.setdefault(nm, {}).setdefault(era, {})[tag] = sstats(rr.to_numpy(), rr.index)
        s2 = stage2(port, base)
        for n, v in s2.items():
            print(f"  {n:32s} {'✓' if v['pass'] else '✗'} {v['why'][:90]}")
    return strip(s1), port, s2


# ══════════════════════════════════════════════════════════════
# 4회차 (결과를 보기 전에 정함) — 매일 하는 단기 매매: 추세 위에서만 밤사이 보유 (KODEX 200)
#   장 마감 단일가(CLS)에 사서 다음 날 장 시작 단일가(OPG)에 판다. 판단 = 전날 저녁 지수가 120일선 위인가.
#   기존 긴 역사 검증에서 두 시대 모두 샤프 0.7 (보유 0.4·0.56) — 공격 엔진(DM:2)에 섞으면 나아지는가?
#   합격: 두 시대 모두 DM:2 단독보다 샤프 높고 최대낙폭이 나쁘지 않음 (비용 1배·3배 모두)
# ══════════════════════════════════════════════════════════════
def overnight_stream(dates, Op, Cp, stress=1.0, j=0, ma=120, slip_fixed=None):
    """밤사이 보유 일별 수익: t-1일 저녁 지수가 120일선 위면 t일 종가 단일가에 사서 t+1일 시가 단일가에 판다."""
    c = AW.ffill(Cp)[:, j]
    o = Op[:, j]
    m = pd.Series(c).rolling(ma, min_periods=int(ma * 0.8)).mean().to_numpy()
    sig = c > m                                         # t일 저녁 신호
    T = len(c)
    r = np.zeros(T)
    for t in range(2, T):
        if not sig[t - 2] or not np.isfinite(o[t]) or not np.isfinite(c[t - 1]):
            continue                                   # t-1일 종가에 살지 = t-2일 저녁 신호
        if slip_fixed is not None:
            cb = cs = AW.COMM + slip_fixed * stress
        else:
            cb = float(AW.costs(np.array([c[t - 1]]), stress)[0][0])
            cs = float(AW.costs(np.array([o[t]]), stress)[1][0])
        r[t] = o[t] * (1 - cs) / (c[t - 1] * (1 + cb)) - 1
    return r


def steadiness(r, dates):
    r = pd.Series(np.nan_to_num(np.asarray(r, float)), index=dates)
    act = r[r != 0]
    mo = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    neg = (mo < 0).astype(int).to_numpy()
    run = best = 0
    for x in neg:
        run = run + 1 if x else 0
        best = max(best, run)
    return {"day_win": round(float((act > 0).mean()), 3) if len(act) else None, "days_in": round(float(len(act) / max(len(r), 1)), 3),
            "month_win": round(float((mo > 0).mean()), 3), "max_losing_months": int(best)}


def round4():
    import history_lab as H
    out, port = {}, {}
    for era in ("A", "B"):
        a_, b_ = PORT_ERA[era]
        for tag, k in (("x1", 1.0), ("x3", 3.0)):
            cs = card_streams(era, k)
            if era == "A":
                dates, Op, Cp = H.proxy_prices(H.load_hist())
                on = pd.Series(overnight_stream(dates, Op, Cp, k, slip_fixed=H.SLIP["069500"]), index=dates)
            else:
                dates, Op, Cp = AW.load_prices()
                on = pd.Series(overnight_stream(dates, Op, Cp, k), index=dates)
            dm = cs["DM:2"]

            def win(x):
                return x[(x.index >= pd.Timestamp(a_)) & ((x.index <= pd.Timestamp(b_)) if b_ else True)]
            rows = {"DM:2 단독 (지금 공격 엔진)": win(dm), "D1 밤사이 보유 (추세 위)": win(on),
                    "D2 DM:2 50 + 밤사이 50": combine(on, dm, 0.5, a_, b_)}
            for n, x in rows.items():
                st = sstats(x.to_numpy(), x.index)
                if tag == "x1":
                    st.update(steadiness(x.to_numpy(), x.index))
                port.setdefault(n, {}).setdefault(era, {})[tag] = st
    base = "DM:2 단독 (지금 공격 엔진)"
    verdict = {}
    for n in port:
        if n == base:
            continue
        ok, why = True, []
        for era in ("A", "B"):
            for tag in ("x1", "x3"):
                me, b = port[n][era][tag], port[base][era][tag]
                if me["sharpe"] <= b["sharpe"]:
                    ok = False
                    why.append(f"{era}·{tag} 샤프 {me['sharpe']} ≤ {b['sharpe']}")
                if me["mdd"] < b["mdd"]:
                    ok = False
                    why.append(f"{era}·{tag} 낙폭 {me['mdd']} < {b['mdd']}")
        verdict[n] = {"pass": ok, "why": "통과" if ok else "; ".join(why[:4])}
    for n, v in port.items():
        print(f"  {n:24s} " + " | ".join(
            f"{era} 연 {v[era]['x1']['cagr']:+.1%} 샤프 {v[era]['x1']['sharpe']:.2f} 낙폭 {v[era]['x1']['mdd']:.1%} "
            f"(3배 {v[era]['x3']['sharpe']:.2f}) 오른 달 {v[era]['x1']['month_win']:.0%} 연속 손실 {v[era]['x1']['max_losing_months']}달"
            for era in ("A", "B")) + (f"  {'✓' if verdict[n]['pass'] else '✗'} {verdict[n]['why'][:60]}" if n in verdict else "  (기준)"))
    return port, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    if a.round == 4:
        print("■ 4회차 — 매일 단기 매매: 추세 위에서만 밤사이 보유 (기준 = DM:2 단독)")
        port, verdict = round4()
        save(a.round, t0, {"portfolio": port, "stage2": verdict}, n_trials=len(verdict),
             passed=[n for n, v in verdict.items() if v["pass"]])
        return
    if a.round == 3:
        print("■ 3회차 — 종목마다 손절·익절 (S1 = 느린 종목 모델 전부 같은 비중)")
        s1, port, s2 = round3()
        save(a.round, t0, {"stage1": s1, "portfolio": port, "stage2": s2, "stops": STOPS},
             n_trials=len(s1) - 1 + len(s2), passed=[n for n, v in s2.items() if v["pass"]])
        return
    if a.round == 2:
        print("■ 2회차 — ETF 만: 전천후 16장 + 주식 위성 카드 (기준 = 전천후 16장 단독)")
        port, s2 = round2()
        save(a.round, t0, {"portfolio": port, "stage2": s2, "sats": list(SATS), "shares": list(SAT_SHARES)},
             n_trials=len(s2), passed=[n for n, v in s2.items() if v["pass"]])
        return
    print("■ 1단계 — 종목 몫 후보 (시대 A 2002~2018 KRX · 시대 B 2019~)")
    eras = {era: run_era(era) for era in ("A", "B")}
    names = list(eras["A"]["streams"])
    s1 = stage1({k: v for k, v in eras.items()}, names)
    for n, v in s1.items():
        print(f"  {n:28s} A {v['A']['me']['sharpe']:6.3f}/{v['A']['idx']['sharpe']:5.3f} 낙폭 {v['A']['me']['mdd']:7.1%} | "
              f"B {v['B']['me']['sharpe']:6.3f}/{v['B']['idx']['sharpe']:5.3f} 낙폭 {v['B']['me']['mdd']:7.1%} | p {v['p']} → {v['p_holm']} {'✓' if v['pass'] else '✗'}")
    # 덮개: 1단계 후보 중 아무것도 고르지 않는 S1 에 미리 정한 3종 × 2값
    ov = {}
    print("■ 덮개 (S1 에 · 그날 아침 정보로만)")
    for kind, x in (("vt", 0.15), ("vt", 0.25), ("ddb", 0.15), ("ddb", 0.25), ("mkt", 0.0), ("mkt", 0.5)):
        nm = f"S1+{kind}{x}"
        row = {}
        for era, res in eras.items():
            s, e = res["st"], res["e"]
            idx = np.cumprod(1 + res["rb"])
            r, expo = overlay(res["streams"]["S1 느린 종목 모델 전부 같은 비중"]["r"], kind, x, idx)
            res["streams"][nm] = {"r": r, "switches": None}
            row[era] = expo
        ov[nm] = row
    s1o = stage1(eras, list(ov))
    for n, v in s1o.items():
        print(f"  {n:28s} A {v['A']['me']['sharpe']:6.3f} 낙폭 {v['A']['me']['mdd']:7.1%} | B {v['B']['me']['sharpe']:6.3f} 낙폭 {v['B']['me']['mdd']:7.1%} | p {v['p_holm']} {'✓' if v['pass'] else '✗'}")
    # 2단계: 전체 계좌
    print("■ 2단계 — 전체 계좌 (기준 = 전천후 16장 단독, 지금 실제 돈 계획)")
    aw = {era: {1: aw_stream(era), 3: aw_stream(era, stress=3.0)} for era in ("A", "B")}
    eras3 = {era: run_era(era, stress=3.0) for era in ("A", "B")}
    for era in eras3:
        for nm, (kind, x) in {f"S1+{k}{v}": (k, v) for k, v in (("vt", 0.15), ("vt", 0.25), ("ddb", 0.15), ("ddb", 0.25), ("mkt", 0.0), ("mkt", 0.5))}.items():
            idx = np.cumprod(1 + eras3[era]["rb"])
            eras3[era]["streams"][nm] = {"r": overlay(eras3[era]["streams"]["S1 느린 종목 모델 전부 같은 비중"]["r"], kind, x, idx)[0]}
    allrows = {**s1, **s1o}
    port = {}
    base = "P0 전천후 16장 단독"
    for era in ("A", "B"):
        a_, b_ = PORT_ERA[era]
        for tag, k in (("x1", 1), ("x3", 3)):
            r_aw = aw[era][k]
            sub = r_aw[(r_aw.index >= pd.Timestamp(a_)) & ((r_aw.index <= pd.Timestamp(b_)) if b_ else True)]
            port.setdefault(base, {}).setdefault(era, {})[tag] = sstats(sub.to_numpy(), sub.index)
            src = eras if k == 1 else eras3
            for n in [base_n for base_n in allrows if allrows[base_n]["pass"]] + ["S0 지금 규칙 (250일 샤프 1등·매주)"]:
                rs = pd.Series(src[era]["streams"][n]["r"], index=src[era]["dates"])
                for share in (0.8, 0.6):
                    nm = f"전천후 {int(share * 100)} + {n.split(' ')[0]} {int(round((1 - share) * 100))}"
                    rr = combine(rs, r_aw, share, a_, b_)
                    port.setdefault(nm, {}).setdefault(era, {})[tag] = sstats(rr.to_numpy(), rr.index)
    s2 = stage2(port, base)
    for n, v in port.items():
        print(f"  {n:32s} " + " | ".join(f"{era} 샤프 {v[era]['x1']['sharpe']:.3f} 연 {v[era]['x1']['cagr']:+.1%} 낙폭 {v[era]['x1']['mdd']:.1%}" for era in ("A", "B"))
              + (f"  {'✓' if s2[n]['pass'] else '✗'} {s2[n]['why'][:80]}" if n in s2 else "  (기준)"))
    n_trials = len(s1) + len(s1o) + len([n for n in port if n != base])
    save(a.round, t0, {"stage1": strip(s1), "overlays": strip(s1o), "overlay_exposure": ov, "portfolio": port, "stage2": s2},
         n_trials=n_trials, passed=[n for n, v in s2.items() if v["pass"]])


def save(rnd, t0, body, n_trials, passed):
    """회차마다 rounds[회차] 에 쌓고, 시도 수는 모든 회차를 더한다 (trials.py 가 읽음)."""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prev = (json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}) or {}
    rounds = prev.get("rounds") or {}
    if not rounds and prev.get("stage1"):                       # 예전 형식(1회차만 있던 파일) → rounds["1"]
        rounds["1"] = {k: prev[k] for k in ("generated", "n_trials", "passed", "secs", "stage1", "overlays", "overlay_exposure",
                                            "portfolio", "stage2") if k in prev}
    rounds[str(rnd)] = {"generated": time.strftime("%Y-%m-%d %H:%M"), "n_trials": n_trials, "passed": passed,
                        "secs": round(time.time() - t0), **body}
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "criteria": CRIT, "era": ERA, "port_era": PORT_ERA, "rounds": rounds,
           "n_trials": sum(r["n_trials"] for r in rounds.values()), "passed": {k: r["passed"] for k, r in rounds.items()}}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    json.dump(res, open(DOCS, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"끝 ({rounds[str(rnd)]['secs']}초) · 이번 시도 {n_trials} · 통과 {passed or '없음'} → {OUT}")


if __name__ == "__main__":
    main()
