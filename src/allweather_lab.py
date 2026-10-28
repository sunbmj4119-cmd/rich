"""
전천후(국면 스위칭) 연구 — '어떤 시장에서도 버는 모델'이 실제로 가능한지, 과거 자료로 정직하게 잰다.

  1) 전략 카드 21장(allweather.RULES)을 2014-08 ~ 오늘 따로 돌려 국면별 성적을 낸다 (비용·세금 포함)
  2) 고르는 방식 비교 — 매주(그리고 국면이 바뀐 날) 저녁, 그때까지의 자료로만:
       uncond  최근 L일 샤프 1등 (지금 단기 모의투자 방식)
       cond    '지금과 같은 국면이었던 과거 날들'의 샤프 1등
       blend   두 점수를 표본 수로 섞음 (같은 국면 표본이 적으면 최근 성적 쪽으로)
  3) 검증: 개발 2016~2020 / 확인 2021~ 나눠 보기, 국면별 성적, 지수 보유 대비 부트스트랩 P,
           뒤늦은 지식 점검(지난 10년 크게 오른 나스닥·금을 빼고도 되나), 비용 3배
  산출: data/short/allweather.json · docs/allweather.json

  python src/allweather_lab.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allweather as AW  # noqa: E402
import short_lab as L  # noqa: E402

OUT = "data/short/allweather.json"
UI = "docs/allweather.json"
META_START = "2016-01-04"
DEV_END = "2020-12-31"
MIXES = {
    "mix_all": ("카드 전부 같은 비중 (고르지 않음)", {"mode": "all", "on_change": False}),
    "mix_rp": ("카드 전부, 변동성 반비례 (매달) — 채택", {"mode": "rp", "on_change": False}),
    "mix_top3": ("같은 국면+최근 250일 점수 상위 3장 (매달 + 국면 바뀐 날)", {"mode": "topk", "k": 3}),
    "mix_top5": ("같은 국면+최근 250일 점수 상위 5장 (매달 + 국면 바뀐 날)", {"mode": "topk", "k": 5}),
    "mix_top5_u": ("최근 250일 점수 상위 5장 (매달, 국면 안 봄)", {"mode": "topk", "k": 5, "shrink": 1e12, "on_change": False}),
}
VARIANTS = {
    "uncond_250": ("최근 250일 샤프 1등 (매주)", {"mode": "uncond", "L": 250, "on_change": False}),
    "uncond_120": ("최근 120일 샤프 1등 (매주)", {"mode": "uncond", "L": 120, "on_change": False}),
    "cond": ("같은 국면 샤프 1등 (매주 + 국면 바뀐 날)", {"mode": "cond", "L": 250}),
    "blend_120": ("같은 국면 + 최근 250일 섞기 (표본 120일 기준)", {"mode": "blend", "L": 250, "shrink": 120}),
    "blend_250": ("같은 국면 + 최근 250일 섞기 (표본 250일 기준)", {"mode": "blend", "L": 250, "shrink": 250}),
    "blend_120_L120": ("같은 국면 + 최근 120일 섞기 (표본 120일 기준)", {"mode": "blend", "L": 120, "shrink": 120}),
}


def born_mask(rule, dates, Cp):
    """보유 카드는 그 자산이 상장되기 전 날들을 '성적 없음'(NaN)으로 — 단기채로 대신 들고 있던 기록이 섞이지 않게."""
    if not rule.startswith("H:"):
        return np.zeros(len(dates), bool)
    j = AW.IX[rule.split(":")[1]]
    first = np.where(~np.isnan(Cp[:, j]))[0]
    m = np.ones(len(dates), bool)
    if len(first):
        m[:first[0] + 2] = True
        m[first[0] + 2:] = False
    return m


def excess(R, rules):
    """점수는 '단기채보다 얼마나 더'로 잰다 — 그냥 샤프로 고르면 거의 안 움직이는 단기채가 늘 1등이 된다(1차 실행에서 확인)."""
    rc = np.nan_to_num(R[:, rules.index("H:CASH")]) if "H:CASH" in rules else 0.0
    return R - (rc[:, None] if np.ndim(rc) else 0.0)


def follow(Op, Cp, Ws, ch, t0, stress=1.0):
    """고른 카드의 비중을 다음 날 시가에 따라간다. 갈아타는 날은 새 카드가 '지금 들고 있을' 비중으로 한 번에."""
    T = Op.shape[0]
    wo = np.full((T, AW.A), np.nan)
    P = [AW.desired(W) for W in Ws]
    prev, sw = None, 0
    for t in range(t0, T):
        c = ch[t - 1]
        if c is None:
            continue
        if c != prev:
            wo[t] = P[c][t]
            sw += prev is not None
            prev = c
        elif not np.isnan(Ws[c][t]).all():
            wo[t] = Ws[c][t]
    res = AW.simulate(Op, Cp, wo, t0=t0, stress=stress)
    res["switches"] = sw
    return res


def by_regime(r, reg_prev):
    out = {}
    for k, name in enumerate(AW.REGIMES):
        x = r[reg_prev == k]
        x = x[np.isfinite(x)]
        if len(x) < 20:
            out[name] = None
            continue
        sd = x.std()
        out[name] = {"days": int(len(x)), "ann": round(float((1 + x).prod() ** (252 / len(x)) - 1), 4),
                     "sharpe": round(float(x.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0, 3),
                     "hit": round(float((x > 0).mean()), 3)}
    return out


def summarize(E, dates, s, reg_prev, bench_r=None, te=None):
    r = L.rets(E)
    out = {"all": L.stats(E[s:], dates[s:]), "dev": L.stats(E, dates, a=str(dates[s].date()), b=DEV_END),
           "test": L.stats(E, dates, a="2021-01-01"), "years": L.yearly(E[s:], dates[s:]),
           "regime": by_regime(r[s + 1:], reg_prev[s + 1:])}
    if bench_r is not None:
        out["p_vs_idx"] = L.boot_p(r[s:], bench_r[s:])
        if te is not None:
            out["p_vs_idx_test"] = L.boot_p(r[te:], bench_r[te:])
    return out


def run_all(dates, Op, Cp, F, rules, s, stress=1.0, variants=VARIANTS, mixes=None):
    Ws, recs = [], []
    for k in rules:
        W = AW.targets(F, Cp, k)
        Ws.append(W)
        recs.append(AW.simulate(Op, Cp, W, stress=stress))
    R = np.c_[[L.rets(x["E"]) for x in recs]].T
    for i, k in enumerate(rules):
        R[born_mask(k, dates, Cp), i] = np.nan
    reg = F["reg"]
    Rx = excess(R, rules)
    rs = AW.RegimeScore(Rx, reg)
    metas = {}
    for key, (name, kw) in variants.items():
        ch = AW.choose_regime(Rx, reg, s, every=5, scorer=rs, **kw)
        res = follow(Op, Cp, Ws, ch, s, stress=stress)
        metas[key] = {"name": name, "E": res["E"], "ch": ch, "switches": res["switches"], "turnover": res["turnover"]}
    if mixes:
        P = [AW.desired(W) for W in Ws]
        cash = rules.index("H:CASH")
        for key, (name, kw) in mixes.items():
            ch = AW.choose_mix(Rx, reg, s, cash=cash, scorer=rs if kw.get("mode") == "topk" else None,
                               resched=AW.month_start(dates, pad=1), **kw)
            wo, sw = AW.mix_weights(P, Ws, ch, s, len(dates))
            res = AW.simulate(Op, Cp, wo, t0=s, stress=stress)
            metas[key] = {"name": name, "E": res["E"], "ch": [None if c is None else max(c, key=lambda x: x[1])[0] for c in ch],
                          "mix": ch, "switches": sw, "turnover": res["turnover"]}
    return Ws, recs, R, metas


def rebalanced(dates, rl, ra, share, s, cost=0.0025):
    """두 몫을 따로 굴리다 분기 첫 거래일에 몫을 다시 맞춘다 (옮기는 금액에 0.25% 비용 — 주식을 팔 때 세금 포함 근사)."""
    d = pd.DatetimeIndex(dates)
    q = np.r_[False, ((d.month[1:] - 1) // 3 != (d.month[:-1] - 1) // 3) | (d.year[1:] != d.year[:-1])]
    E = np.ones(len(rl))
    wl, wa = 1 - share, share
    for t in range(s + 1, len(rl)):
        wl *= 1 + np.nan_to_num(rl[t])
        wa *= 1 + np.nan_to_num(ra[t])
        if q[t]:
            tot = wl + wa
            tot -= abs(wl - (1 - share) * tot) * cost
            wl, wa = (1 - share) * tot, share * tot
        E[t] = wl + wa
    E[:s + 1] = 1.0
    return E


def combined(dates, Op, Cp, F, share=0.5, start="2019-01-02"):
    """채택안: 단기 갈아타기(short_lab.LIVE_RULE, 2018~ 패널) + 전천후 혼합(allweather.live_mix 와 같은 고르기).
    두 몫을 각각 정확히 시뮬레이션하고 분기마다 50:50 으로 다시 맞춘다 (단기 쪽 자료가 시작되는 2019년부터).
    (맞추지 않으면 단기 몫이 커져 위험이 다시 커진다 — 샤프 0.99, 낙폭 -27%. 분기 맞춤: 1.11, -20%)"""
    D = L.Data()
    models = L.stock_models() + L.etf_models()
    recs = [L.run(D, m, record=True) for m in models]
    R = np.c_[[L.rets(r["E"]) for r in recs]].T
    s = int(np.searchsorted(D.dates, pd.Timestamp(start)))
    E_live, sw, _ = L.apply_rule(D, models, recs, R, L.LIVE_RULE, s)
    m = AW.live_mix(upto=D.dates[-1])
    E_aw = pd.Series(m["E"], index=m["dates"]).reindex(D.dates).ffill().to_numpy()
    idx = recs[[x["id"] for x in models].index("E:hold")]["E"]
    a, b, c = E_live / E_live[s], E_aw / E_aw[s], idx / idx[s]
    E = rebalanced(D.dates, L.rets(E_live), L.rets(E_aw), share, s)
    reg = pd.Series(F["reg"], index=dates).reindex(D.dates).ffill().fillna(-1).to_numpy().astype(int)
    reg_prev = np.r_[-1, reg[:-1]]
    te = int(np.searchsorted(D.dates, pd.Timestamp("2023-01-02")))
    out = {"from": str(D.dates[s].date()), "to": str(D.dates[-1].date()), "share_aw": share, "rows": {}}
    for name, e in (("모의 계좌 (단기 50 + 전천후 50, 분기마다 맞춤)", E), ("단기 갈아타기만", a), ("전천후 혼합만", b), ("KODEX 200 보유", c)):
        r = L.rets(e)
        out["rows"][name] = {"all": L.stats(e[s:], D.dates[s:]), "test": L.stats(e, D.dates, a="2023-01-01"),
                             "years": L.yearly(e[s:], D.dates[s:]), "regime": by_regime(r[s + 1:], reg_prev[s + 1:]),
                             "p_vs_idx": L.boot_p(r[s:], L.rets(c)[s:]), "p_vs_idx_test": L.boot_p(r[te:], L.rets(c)[te:]),
                             "curve": [round(float(x), 4) for x in e[s::5]]}
        x = out["rows"][name]
        print(f"◆ {name:24s} " + json.dumps(x["all"], ensure_ascii=False) + f" · P={x['p_vs_idx']} · 손실 연도 "
              f"{sum(v < 0 for v in x['years'].values())}/{len(x['years'])} · "
              + " ".join(f"{k}{v['ann'] * 100:+.0f}%" for k, v in x["regime"].items() if v))
    out["curve_dates"] = [str(d.date()) for d in D.dates[s::5]]
    out["switches_live"] = sw
    return out


def main():
    t0 = time.time()
    dates, Op, Cp = AW.load_prices()
    F = AW.features(Cp, dates)
    reg = F["reg"]
    reg_prev = np.r_[-1, reg[:-1]]
    s = int(np.searchsorted(dates, pd.Timestamp(META_START)))
    te = int(np.searchsorted(dates, pd.Timestamp("2021-01-04")))
    rules = list(AW.RULES)
    print(f"자료 {dates[0].date()} ~ {dates[-1].date()} ({len(dates)}일) · 자산 {AW.A}개 · 전략 카드 {len(rules)}장 · 고르기 시작 {dates[s].date()}")
    Ws, recs, R, metas = run_all(dates, Op, Cp, F, rules, s, mixes=MIXES)
    idx_r = L.rets(recs[rules.index("H:IDX")]["E"])
    # 국면 분포
    cnt = pd.Series(reg[s:]).value_counts()
    reg_days = {AW.REGIMES[k]: int(cnt.get(k, 0)) for k in range(4)}
    episodes = []
    cur, st = None, None
    for t in range(s, len(dates)):
        if reg[t] != cur:
            if cur is not None and cur >= 0:
                episodes.append({"regime": AW.REGIMES[cur], "from": str(dates[st].date()), "to": str(dates[t - 1].date()), "days": t - st})
            cur, st = reg[t], t
    episodes.append({"regime": AW.REGIMES[cur] if cur >= 0 else "-", "from": str(dates[st].date()), "to": str(dates[-1].date()),
                     "days": len(dates) - st})
    # 1) 카드별 성적
    cards = []
    for i, k in enumerate(rules):
        E = recs[i]["E"]
        r = R[:, i]
        ok = np.isfinite(r)
        first = int(np.argmax(ok)) if ok.any() else s
        cards.append({"id": k, "desc": AW.RULES[k], "group": AW.GROUP[k.split(":")[0]],
                      "from": str(dates[max(first, s)].date()),
                      **summarize(E, dates, max(first, s), reg_prev),
                      "turnover_x": round(recs[i]["turnover"] / max(E.mean(), 1e-9) / (len(dates) / 252), 1)})
        print(f"  {k:8s} {AW.RULES[k][:38]:38s} " + json.dumps(cards[-1]["all"], ensure_ascii=False))
    # 2) 고르는 방식
    out = {}
    for key, m in metas.items():
        E = m["E"]
        smr = summarize(E, dates, s, reg_prev, idx_r, te)
        ids = [rules[c] if c is not None else None for c in m["ch"]]
        picks = pd.Series([x for x in ids[s:] if x]).value_counts()
        per_reg = {}
        for k in range(4):
            sel = [ids[t] for t in range(s, len(dates)) if reg[t] == k and ids[t]]
            if sel:
                vc = pd.Series(sel).value_counts(normalize=True).head(4)
                per_reg[AW.REGIMES[k]] = {a: round(float(b), 3) for a, b in vc.items()}
        out[key] = {"name": m["name"], "switches": m["switches"],
                    "turnover_x": round(m["turnover"] / max(E[s:].mean(), 1e-9) / ((len(dates) - s) / 252), 1),
                    **smr, "picks": {a: int(b) for a, b in picks.head(8).items()}, "picks_by_regime": per_reg,
                    "curve": [round(float(x / E[s]), 4) for x in E[s::5]]}
        print(f"■ {m['name']:34s} 교체 {m['switches']:3d} · " + json.dumps(smr["all"], ensure_ascii=False)
              + f" · 시험 {json.dumps(smr['test'], ensure_ascii=False)} · P={smr['p_vs_idx']}")
        print("   국면별: " + " · ".join(f"{a} {b['ann']*100:+.1f}%/샤프{b['sharpe']}" for a, b in smr["regime"].items() if b))
    # 비교 기준
    bench = {}
    W6040 = np.full((len(dates), AW.A), np.nan)
    for t in range(1, len(dates), 21):
        W6040[t] = 0.0
        W6040[t, AW.IX["IDX"]], W6040[t, AW.IX["B10"]] = 0.6, 0.4
    for name, E in (("KODEX 200 보유", recs[rules.index("H:IDX")]["E"]), ("주식 60·장기채 40 (매달)", AW.simulate(Op, Cp, W6040)["E"])):
        bench[name] = summarize(E, dates, s, reg_prev, idx_r, te) | {"curve": [round(float(x / E[s]), 4) for x in E[s::5]]}
    # 모든 카드 균등 혼합 (매일 1/S) — '고르지 않고 다 섞으면'
    mix = np.nanmean(np.where(np.isnan(R), np.nan, R), 1)
    Emix = np.cumprod(1 + np.nan_to_num(mix))
    bench["모든 카드 균등 혼합"] = summarize(Emix, dates, s, reg_prev, idx_r, te) | {"curve": [round(float(x / Emix[s]), 4) for x in Emix[s::5]]}
    # 3) 점검: 뒤늦은 지식(나스닥·금 빼기) · 비용 3배
    rob = {}
    rules_x = [k for k in rules if k not in ("H:NDX", "H:GOLD")]
    sub = [rules.index(k) for k in rules_x]
    Ws_x = [Ws[i] for i in sub]
    R_x = excess(R[:, sub], rules_x)
    rs_x = AW.RegimeScore(R_x, reg)
    for key in ("uncond_250", "blend_120", "cond"):
        ch = AW.choose_regime(R_x, reg, s, every=5, scorer=rs_x, **VARIANTS[key][1])
        E = follow(Op, Cp, Ws_x, ch, s)["E"]
        rob[f"{key}·나스닥금보유카드빼기"] = summarize(E, dates, s, reg_prev, idx_r, te)
    _, _, _, m3 = run_all(dates, Op, Cp, F, rules, s, stress=3.0,
                          variants={k: VARIANTS[k] for k in ("uncond_250", "blend_120", "cond")})
    for k, m in m3.items():
        rob[f"{k}·비용3배"] = summarize(m["E"], dates, s, reg_prev, idx_r, te)
    for k, v in rob.items():
        print(f"  점검 {k:28s} " + json.dumps(v["all"], ensure_ascii=False) + f" P={v.get('p_vs_idx')}")
    # 4) 채택안 정확 재현: 모의 계좌 = 단기 갈아타기(지금 규칙) 50% + 전천후 혼합 50% (따로 굴리고 분기마다 50:50)
    combo = combined(dates, Op, Cp, F)
    now = AW.regime_now(F)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "period": [str(dates[0].date()), str(dates[-1].date())],
           "meta_start": str(dates[s].date()), "dev_end": DEV_END, "assets": {c: AW.ASSETS[c] for c in AW.CODES},
           "regime_rule": AW.REG_DESC, "regime_days": reg_days, "episodes": episodes[-30:], "regime_now": now,
           "cards": cards, "variants": out, "bench": bench, "robust": rob, "combined": combo,
           "live_mix": AW.LIVE_MIX,
           "curve_dates": [str(d.date()) for d in dates[s::5]],
           "costs": {"commission": AW.COMM, "slip": "호가 반 칸 + 0.005%", "tax_other_etf": 0.154}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    from short_trader import clean
    res = clean(res)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    slim = {k: v for k, v in res.items() if k not in ("cards",)}
    slim["cards"] = [{k: c[k] for k in ("id", "desc", "group", "from", "all", "dev", "test", "regime", "turnover_x")} for c in res["cards"]]
    slim["variants"] = {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in res["variants"].items()}
    json.dump(slim, open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"지금 국면: {now['name']} (1년 고점 대비 {now['dd'] * 100:.1f}% · 200일선 대비 {(now['trend200'] or 0) * 100:+.1f}%) · {time.time() - t0:.0f}s → {OUT}")
    return res


if __name__ == "__main__":
    main()
