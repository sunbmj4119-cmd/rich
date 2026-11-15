"""
ISA (미래에셋 중개형) 매일 안내 — 사람이 그대로 따라 하는 ETF 계획. 주문은 사람이 낸다 (토스 오픈API 는 ISA 주문을 못 함).

계획 후보 (2026-11-15, 결과 보기 전에 적음)
  P0  핵심 고배당 50 + KODEX 200 50  (지난 카드, docs/ACCOUNTS.md 2026-11-14)
  P1  핵심 고배당 50 + 회전 DM:2 50
      핵심  PLUS 고배당주(161510) — 사서 오래 들고 간다 · 분배금은 꺼내지 않음(재투자) · 떨어져도 팔지 않음
      회전  DM:2 — 매달 첫 거래일, 6개(KODEX 200 · TIGER 미국S&P500선물(H) · TIGER 미국나스닥100 · KODEX 골드선물(H) ·
            KODEX 미국달러선물 · KOSEF 국고채10년) 중 120일 수익 상위 2개를 반반 · 단기채(KODEX 단기채권)보다 못한 자리는 단기채
            → 순위 밖으로 밀린 ETF 를 파는 것이 '익절·손절' (가격 기준 % 가 아니라 규칙으로 정해짐)
      매달 넣는 돈은 핵심·회전 중 적은 쪽 먼저 · 매년 1월 첫 거래일 한쪽이 60% 를 넘으면 50:50 으로
  고르는 규칙: 세 시기(2005~2007 · 2008~2018 · 2019~) 중 두 시기 이상에서 P1 샤프 ≥ P0 이면 P1, 아니면 P0 (회전 신호는 참고로만)
              (DM:2 기록이 2005 부터라 첫 시기는 2005~2007)
  재료: 고배당 ETF = div_project 'ETF대용' (배당 재투자 · 배당세 15.4% · 보수 0.3% — ISA 는 세금이 더 적어 보수적)
        KODEX 200 = 코스피 지수 + 배당 1.7%/년 · DM:2 = 전천후 카드 기록 (해외 ETF 이익에 15.4% — ISA 에서는 보수적)
        일별 시점이 서로 달라 달 단위로 섞는다

  python src/isa_daily.py --research  → data/isa_lab.json (계획 비교 · 10년 범위)
  python src/isa_daily.py             → docs/isa.json (매일: 다음 거래일 할 일 · 회전 순위 · ETF 지표)  — 개인 잔고는 넣지 않는다
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAB = "data/isa_lab.json"
OUT = "docs/isa.json"
CORE = "161510"
CORE_ALT = "279530"
ROT = ["069500", "143850", "133690", "132030", "261240", "148070"]
SAFE = "153130"
KOSPI_DIV = 0.017
ERAS = {"2005~2007": ("2005-01-01", "2007-12-31"), "2008~2018": ("2008-01-01", "2018-12-31"), "2019~": ("2019-01-01", None), "전체 2005~": ("2005-01-01", None)}
PLANS = {"P0": "고배당 ETF 50 + KODEX 200 50", "P1": "고배당 ETF 50 + 매달 회전(DM:2) 50", "고배당만": "고배당 ETF 100", "KODEX 200": "KODEX 200 100", "DM:2": "매달 회전(DM:2) 100"}
ETF_NOTE = {
    "161510": "국내 고배당주 · 핵심 1순위",
    "279530": "국내 고배당 · 161510 을 못 찾으면",
    "210780": "코스피 고배당 · 다른 후보",
    "446720": "미국 배당 성장주 (SCHD 와 같은 지수) · 환율 따라 움직임",
    "458730": "미국 배당 성장주 (같은 지수, 다른 운용사)",
    "360750": "미국 S&P500 (환율 그대로)",
    "069500": "코스피200 · 회전 후보",
    "143850": "미국 S&P500 (환율 막음) · 회전 후보",
    "133690": "미국 나스닥100 · 회전 후보",
    "132030": "금 (환율 막음) · 회전 후보",
    "261240": "달러 · 회전 후보",
    "148070": "국고채 10년 · 회전 후보",
    "153130": "단기채 · 회전의 피난처",
}


# ── 연구: 계획 비교 ─────────────────────────────────────────
def monthly_from_curve(dates, vals):
    s = pd.Series(vals, index=pd.to_datetime(dates)).astype(float)
    m = s.resample("ME").last().dropna()
    return m.pct_change().dropna()


def kospi_monthly():
    import shadow_wf as SW
    k = SW.kospi_index(None)
    m = k.resample("ME").last().dropna().pct_change().dropna()
    return m + KOSPI_DIV / 12


def mix(a, b, w=0.5, rebalance="annual"):
    """두 달 수익 a·b 를 w : 1−w 로. annual = 해마다 1월에만 맞춤 (그 사이 비중은 흘러감), monthly = 매달 맞춤."""
    idx = a.index.intersection(b.index)
    a, b = a.reindex(idx), b.reindex(idx)
    if rebalance == "monthly":
        return w * a + (1 - w) * b
    va, vb, out = w, 1 - w, []
    for d, ra, rb in zip(idx, a.to_numpy(), b.to_numpy()):
        if d.month == 1:
            t = va + vb
            va, vb = w * t, (1 - w) * t
        t0 = va + vb
        va, vb = va * (1 + ra), vb * (1 + rb)
        out.append((va + vb) / t0 - 1)
    return pd.Series(out, index=idx)


def mstats(m):
    m = m.dropna()
    if len(m) < 12:
        return None
    E = (1 + m).cumprod()
    yrs = len(m) / 12
    yr = (1 + m).groupby(m.index.year).prod() - 1
    return {"cagr": round(float(E.iloc[-1] ** (1 / yrs) - 1), 4), "vol": round(float(m.std() * math.sqrt(12)), 4),
            "sharpe": round(float(m.mean() / m.std() * math.sqrt(12)), 3) if m.std() > 0 else 0.0,
            "mdd": round(float((E / E.cummax() - 1).min()), 4), "worst_year": round(float(yr.min()), 4), "best_year": round(float(yr.max()), 4),
            "months": int(len(m))}


def cut(m, a, b):
    m = m[m.index >= pd.Timestamp(a)]
    return m[m.index <= pd.Timestamp(b)] if b else m


def research(log=print):
    import div_project as DP
    t0 = time.time()
    dp = json.load(open("data/div_project.json", encoding="utf-8"))
    c = dp["curve"]
    core = monthly_from_curve(c["date"], c["ETF대용"])
    dm = monthly_from_curve(c["date"], c["DM:2"])
    dm = dm[dm.index >= pd.Timestamp("2005-01-01")]
    k200 = kospi_monthly()
    series = {"P0": mix(core, k200), "P1": mix(core, dm), "고배당만": core, "KODEX 200": k200, "DM:2": dm,
              "P0(매달 맞춤)": mix(core, k200, rebalance="monthly"), "P1(매달 맞춤)": mix(core, dm, rebalance="monthly")}
    res = {"plans": PLANS, "eras": {}, "rule": "세 시기 중 두 시기 이상 P1 샤프 ≥ P0 이면 P1", "corr": {}}
    for pn, (a, b) in ERAS.items():
        res["eras"][pn] = {k: mstats(cut(v, a, b)) for k, v in series.items()}
        e = res["eras"][pn]
        log(f"  {pn:9s}: P0 연 {e['P0']['cagr']:+.1%} 샤프 {e['P0']['sharpe']:.2f} 낙폭 {e['P0']['mdd']:.0%} | "
            f"P1 연 {e['P1']['cagr']:+.1%} 샤프 {e['P1']['sharpe']:.2f} 낙폭 {e['P1']['mdd']:.0%} | 고배당만 {e['고배당만']['cagr']:+.1%} · DM:2 {e['DM:2']['cagr']:+.1%}")
    j = core.index.intersection(dm.index).intersection(k200.index)
    res["corr"] = {"고배당-KODEX200": round(float(np.corrcoef(core[j], k200[j])[0, 1]), 2), "고배당-DM:2": round(float(np.corrcoef(core[j], dm[j])[0, 1]), 2)}
    wins = sum(res["eras"][pn]["P1"]["sharpe"] >= res["eras"][pn]["P0"]["sharpe"] for pn in ("2005~2007", "2008~2018", "2019~"))
    res["picked"] = "P1" if wins >= 2 else "P0"
    res["p1_wins"] = int(wins)
    # 10년 범위: 20만 원 + 매달 10만 원 · 지나온 길 + 12개월 덩어리 다시 잇기 (2008~2018 약한 시기 바탕도)
    pk = series[res["picked"]]
    res["projection"] = {}
    for bn, (a, b) in {"2005~ 전체": ("2005-01-01", None), "2008~2018 (보수적)": ("2008-01-01", "2018-12-31")}.items():
        m = cut(pk, a, b)
        res["projection"][bn] = {"hist": DP.dca_hist(m, 100_000), "boot": DP.bootstrap(m, 100_000, seed=11)}
        bt = res["projection"][bn]["boot"]
        log(f"  10년 ({bn}): 낸 돈 {bt['paid']:,.0f} → 나쁜 쪽 {bt['p10']:,} · 가운데 {bt['p50']:,} · 좋은 쪽 {bt['p90']:,} · 원금 아래 {bt['loss_prob']:.0%}")
    cv = (1 + pk).cumprod()
    res["curve"] = {"date": [str(d.date()) for d in cv.index], "plan": [round(float(v), 4) for v in cv],
                    "고배당만": [round(float(v), 4) for v in (1 + core.reindex(cv.index).fillna(0)).cumprod()],
                    "KODEX 200": [round(float(v), 4) for v in (1 + k200.reindex(cv.index).fillna(0)).cumprod()]}
    yr = (1 + pk).groupby(pk.index.year).prod() - 1
    res["years"] = {str(y): round(float(v), 4) for y, v in yr.items()}
    res["generated"] = time.strftime("%Y-%m-%d %H:%M")
    res["secs"] = round(time.time() - t0)
    json.dump(res, open(LAB, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"  P1 이 샤프로 이긴 시기 {wins}/3 → {res['picked']} ({PLANS[res['picked']]}) · 상관 {res['corr']} → {LAB}")
    return res


# ── 매일: 다음 거래일 할 일 · 회전 순위 · ETF 지표 ───────────────
def load_etf():
    e = pd.read_csv("data/etf_ohlcv.csv", dtype={"종목코드": str})
    e["종목코드"] = e["종목코드"].str.zfill(6)
    e["날짜"] = pd.to_datetime(e["날짜"])
    return e.sort_values(["종목코드", "날짜"])


def etf_metrics(g):
    """한 ETF 일봉 → 지표 (마지막 종가 기준)."""
    c = g.set_index("날짜")["종가"].astype(float)
    c = c[c > 0]
    if len(c) < 30:
        return {"have": False, "days": int(len(c))}
    last = float(c.iloc[-1])

    def ret(n):
        return round(float(last / c.iloc[-1 - n] - 1), 4) if len(c) > n else None
    y = c.iloc[-250:]
    r = c.pct_change().dropna()
    val = (g["종가"].astype(float) * g["거래량"].astype(float)).iloc[-20:]
    ma200 = float(c.iloc[-200:].mean()) if len(c) >= 200 else None
    return {"have": True, "date": str(c.index[-1].date()), "price": last, "r1m": ret(21), "r3m": ret(63), "r6m": ret(126), "r12m": ret(250),
            "r120": ret(120), "mdd1y": round(float((y / y.cummax() - 1).min()), 4), "vol1y": round(float(r.iloc[-250:].std() * math.sqrt(252)), 4),
            "value20": round(float(val.mean())) if len(val) else None, "above200": (last > ma200) if ma200 else None,
            "since": str(c.index[0].date()), "days": int(len(c))}


def next_bday(d):
    return (pd.Timestamp(d) + pd.offsets.BDay(1)).normalize()


def first_bday(year, month):
    d = pd.Timestamp(year=year, month=month, day=1)
    return d if d.weekday() < 5 else d + pd.offsets.BDay(1)


def dm2_now():
    """전천후 부품으로 DM:2 목표: 이번 달 들고 있을 것 · 다음 거래일 목표 · 지금 120일 수익 순위."""
    import allweather as AW
    dates, Op, Cp = AW.load_prices()
    F = AW.features(Cp, dates)
    W = AW.desired(AW.targets(F, Cp, "DM:2", pad=1))
    ms = AW.month_start(dates, pad=1)
    C = AW.ffill(Cp)
    t = len(dates) - 1
    r120 = {AW.CODES[j]: (float(C[t, j] / C[t - 120, j] - 1) if t >= 120 and np.isfinite(C[t, j]) and np.isfinite(C[t - 120, j]) and C[t - 120, j] > 0 else None)
            for j in range(len(AW.CODES))}
    last_ms = int(np.where(ms[:len(dates)])[0][-1]) if ms[:len(dates)].any() else t
    held = {AW.CODES[j]: round(float(x), 3) for j, x in enumerate(W[last_ms]) if x > 1e-4}
    nxt = {AW.CODES[j]: round(float(x), 3) for j, x in enumerate(W[-1]) if x > 1e-4}
    cash_r = r120.get(SAFE)
    rank = sorted([c for c in ROT if r120.get(c) is not None], key=lambda c: -r120[c])
    would = {}
    for c in rank[:2]:
        k = c if (cash_r is None or r120[c] > cash_r) else SAFE
        would[k] = would.get(k, 0) + 0.5
    return {"dates": dates, "held": held, "held_since": str(dates[last_ms].date()), "next": nxt, "next_is_rebalance": bool(ms[-1]),
            "r120": r120, "rank": rank, "would_now": would, "cash_r120": cash_r, "names": AW.NAME}


def todo_text(now_date, nxt_date, is_reb, held, nxt, nm, january):
    lines = []
    if is_reb:
        lines.append(f"{nxt_date:%m월 %d일}(이번 달 첫 거래일) 10:00~14:30 에 앱을 연다.")
        lines.append("① 입금된 돈 + 남은 돈 + 들어온 분배금 = 예수금. 핵심(고배당)과 회전 중 평가금액이 적은 쪽부터 채운다.")
        sell = [c for c in held if c not in nxt]
        buy = [c for c in nxt if c not in held]
        if sell or buy:
            if sell:
                lines.append("② 회전 몫: " + ", ".join(nm.get(c, c) for c in sell) + " 을(를) 전부 판다 (순위 밖).")
            if buy:
                lines.append("③ 회전 몫: 판 돈 + 회전 몫 예수금으로 " + " · ".join(f"{nm.get(c, c)} {int(round(w * 100))}%" for c, w in nxt.items()) +
                             " 가 되게 정수 주로 산다 (지정가 = 매도 1호가).")
        else:
            lines.append("② 회전 몫: 그대로 (" + " · ".join(nm.get(c, c) for c in nxt) + "). 새 돈이 회전 쪽이면 같은 비중으로 더 산다.")
        if january:
            lines.append("④ 1월: 핵심·회전 중 한쪽이 전체의 60% 를 넘으면 넘는 만큼 그쪽을 팔아 다른 쪽을 산다 (50:50).")
    else:
        lines.append("할 일 없음 — 들고 있기. 떨어져도 팔지 않는다.")
    return lines


def daily(log=print):
    import collect_etf as CE
    lab = {}
    try:
        lab = json.load(open(LAB, encoding="utf-8"))
    except (OSError, ValueError):
        pass
    e = load_etf()
    dm = dm2_now()
    nm = dict(dm["names"])
    nm.update({k: v[0] for k, v in CE.ISA_EXTRA.items()})
    metrics = {}
    codes = list(CE.ISA_EXTRA) + ROT + [SAFE]
    for c in dict.fromkeys(codes):
        g = e[e["종목코드"] == c]
        m = etf_metrics(g) if len(g) else {"have": False, "days": 0}
        real = str(g["종목명"].iloc[-1]) if len(g) else None
        m.update({"code": c, "name": real or nm.get(c, c), "note": ETF_NOTE.get(c, ""),
                  "group": (CE.ISA_EXTRA.get(c) or CE.ASSETS.get(c) or ("", "", ""))[2], "rot": c in ROT, "safe": c == SAFE,
                  "core": c in (CORE, CORE_ALT)})
        if real:
            nm[c] = real
        metrics[c] = m
    last = dm["dates"][-1]
    nxt_date = next_bday(last)
    is_reb = dm["next_is_rebalance"]
    nm2 = {c: metrics.get(c, {}).get("name") or nm.get(c, c) for c in set(nm) | set(metrics)}
    todo = todo_text(last, nxt_date, is_reb, dm["held"], dm["next"], nm2, nxt_date.month == 1)
    nm_first = first_bday(nxt_date.year + (nxt_date.month == 12), nxt_date.month % 12 + 1) if not is_reb else nxt_date
    core_m = metrics.get(CORE, {})
    out = {"what": "ISA (미래에셋 중개형) — 사람이 따라 하는 ETF 계획 · 매일 장 마감 뒤 갱신 · 개인 잔고 없음",
           "data_until": str(last.date()), "next_day": str(nxt_date.date()), "is_action_day": bool(is_reb),
           "next_action_day": str(nm_first.date()), "todo": todo,
           "plan": {"picked": lab.get("picked", "P1"), "name": PLANS.get(lab.get("picked", "P1")),
                    "core": {"code": CORE, "alt": CORE_ALT, "name": nm2.get(CORE, "PLUS 고배당주"), "share": 0.5, "have": bool(core_m.get("have"))},
                    "rotation": {"share": 0.5, "universe": ROT, "safe": SAFE, "rule": "매달 첫 거래일 · 6개 중 120일 수익 상위 2개 반반 · 단기채보다 못하면 단기채"},
                    "rules": ["매달 첫 거래일 한 번만 본다 (그 사이에는 아무것도 안 함)",
                              "새 돈은 핵심·회전 중 평가금액이 적은 쪽부터",
                              "회전 몫은 순위 밖으로 밀린 ETF 를 전부 팔고 새 1·2등을 반반 — 이것이 익절·손절",
                              "핵심(고배당)은 떨어져도 팔지 않는다 · 분배금은 꺼내지 않는다",
                              "매년 1월 첫 거래일: 한쪽이 60% 를 넘으면 50:50 으로",
                              "3년 안에 해지하지 않는다 (세금 혜택을 돌려내야 함)"]},
           "rotation": {"held": [{"code": c, "name": nm2.get(c, c), "w": w} for c, w in dm["held"].items()], "held_since": dm["held_since"],
                        "next": [{"code": c, "name": nm2.get(c, c), "w": w} for c, w in dm["next"].items()],
                        "would_now": [{"code": c, "name": nm2.get(c, c), "w": w} for c, w in dm["would_now"].items()],
                        "rank": [{"code": c, "name": nm2.get(c, c), "r120": round(dm["r120"][c], 4), "beats_cash": dm["cash_r120"] is None or dm["r120"][c] > dm["cash_r120"]}
                                 for c in dm["rank"]],
                        "cash_r120": round(dm["cash_r120"], 4) if dm["cash_r120"] is not None else None},
           "etfs": list(metrics.values()),
           "history": {k: lab.get(k) for k in ("eras", "picked", "p1_wins", "corr", "projection", "curve", "years", "rule")},
           "generated": time.strftime("%Y-%m-%d %H:%M")}
    log(f"  ISA 안내: 자료 {out['data_until']} · 다음 거래일 {out['next_day']} {'(할 일 있음)' if is_reb else '(할 일 없음)'} · "
        f"회전 지금 {[x['name'] for x in out['rotation']['held']]} · 지금 순위로는 {[x['name'] for x in out['rotation']['would_now']]} · "
        f"핵심 자료 {'있음' if core_m.get('have') else '없음'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", action="store_true")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)                                  # noqa: E731
    if a.research:
        research(log)
        return
    out = daily(log)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
