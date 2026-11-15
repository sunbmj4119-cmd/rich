"""
배당으로 매달 50만 원 — 얼마가 필요하고, 몇 년 걸리고, 어떻게 받나 (2026-11-15 · 같은 날 100만 원 → 50만 원으로 목표 고침)

  목표: 세후 배당 매달 500,000원 = 연 6,000,000원
  0) 계좌별 세금: 일반 15.4% · ISA 는 비과세 한도(순이익 200만, 서민형 400만 — 계좌 기간 전체 합) 넘는 몫 9.9% → 9.9% 로 잡음(보수적) ·
     연금계좌 55~69세 5.5% (70~79세 4.4%, 80세~ 3.3%)
  5) 배당 대신 일부를 팔아 꺼내 쓰면: 모은 돈에서 매달 50만 원(물가만큼 늘림)을 꺼낼 때 30년 버틸 확률 (ISA 계획 달 수익) ·
     90% 버티려면 얼마가 필요한가
  1) 필요한 돈 = 연 1,200만 ÷ 세후 배당률. 배당률은 지금 사면 받는 값(최근 1년 배당 ÷ 오늘 가격, 시총 상위 100 중 배당자료 있는 종목)과
     2003~ 지나온 평균 두 가지로 · 고배당 30종목(ETF 대용) / 고배당·저변동 10종목 / 고배당 3종목
  2) 받는 동안의 흔들림: 2003 년(또는 2008 년)에 한 번 사 두고 배당을 꺼내 쓰기만 했다면 해마다 받은 배당이 어떻게 변했나 (배당 줄어든 해 · 최악)
     — 같은 매매를 하되 배당은 다시 사지 않고 따로 셈 (simulate_income). 2018 은 자료 이음매로 연말 배당이 빠져 흔들림 계산에서 뺌
  3) 모으는 기간: ISA 계획(고배당 50 + 매달 회전 50, isa_lab) 의 달 수익을 12개월 덩어리로 다시 이어 붙인 길로
     '지금 모은 돈 + 매달 넣는 돈' 이 필요한 돈에 닿는 해 (가운데 · 나쁜 쪽 10%) · 물가 2%/년 반영 여부
  4) 배당 달력: 배당락이 몇 월에 몰리나 (받는 달은 보통 1~2달 뒤)

  python src/income_goal.py → data/income_goal.json · docs/income_goal.json
"""
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/income_goal.json"
DOCS = "docs/income_goal.json"
GOAL_MONTH = 500_000
GOAL_YEAR = 12 * GOAL_MONTH
TAX = 0.154
ACCT_TAX = {"일반": 0.154, "ISA": 0.099, "연금": 0.055}
ACCT_NAME = {"일반": "일반 계좌 (토스 등) 15.4%", "ISA": "ISA — 비과세 한도 넘는 몫 9.9%", "연금": "연금계좌 55~69세 5.5%"}
DRAW_CAPS = [60_000_000, 80_000_000, 100_000_000, 120_000_000, 150_000_000, 200_000_000]
DRAW_YEARS = 30
INFL = 0.02
CONTRIB = [300_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000]
NPATH = 4000
MAXY = 45
PORTS = {"ETF대용": "고배당 30종목 (ETF 처럼)", "고배당·저변동": "고배당·덜 흔들림 10종목", "고배당3": "고배당 3종목 (집중)"}


def current_yields(E):
    """오늘 사면 받는 배당률 (최근 1년 현금배당 ÷ 오늘 종가, 세전): 배당자료 있는 상위 100 중 배당수익률 상위 30 · 10 · 3 평균. E = 배당 붙인 시대 B."""
    import shadow_wf as SW
    t = E.i1 - 1
    y = np.where(E.U[t] & np.isfinite(E.ttm[t]), E.ttm[t], np.nan)
    order = [int(j) for j in np.argsort(-np.nan_to_num(y, nan=-1)) if np.isfinite(y[j])]
    nm = SW.names()
    out = {"date": str(E.dates[t].date()), "n": len(order)}
    for k, n in (("ETF대용", 30), ("고배당·저변동", 10), ("고배당3", 3)):
        out[k] = round(float(np.mean([y[j] for j in order[:n]])), 4)
    out["top10"] = [{"name": nm.get(E.codes[j], E.codes[j]), "yield": round(float(y[j]), 4)} for j in order[:10]]
    # 배당 달력: 2019~ 배당락 달별 몫 (배당자료 있는 상위 100 전체, 금액 가중)
    D = np.where(E.U, E.Dv, 0.0)
    m = pd.Series(D.sum(1), index=E.dates)
    m = m[m.index >= pd.Timestamp("2019-01-01")]
    bym = m.groupby(m.index.month).sum()
    out["ex_month_share"] = {int(k): round(float(v / bym.sum()), 3) for k, v in bym.items()}
    return out


def simulate_income(E, S, rule, a, b, V0=1.0):
    """score_lab.simulate 와 같은 매매(매일 판단 → 다음 날 시가, 비용 포함) 인데 배당은 다시 사지 않고 꺼내 쓴다.
    → 날마다 받은 배당(세후) · 평가액. 배당 = 그날 들고 있는 값 × E.inc (div_lab, 배당락 구간에 받음)."""
    import score_lab as SC
    import setup_lab as SL
    cb, cs = SL.COMM + SL.SLIP, SL.COMM + SL.SLIP + SL.TAX
    lo, hi = E.rows(a, b)
    hi = min(hi, E.G.shape[0] - 2)
    PR, order = SC._prep(E, S)
    slots, q_in, q_out, min_hold = rule["slots"], rule["q_in"], rule["q_out"], rule.get("min_hold", 1)
    cash, pos = float(V0), {}
    inc, val = [], []
    for t in range(lo, hi):
        for j in list(pos):
            v, d = pos[j]
            pr = PR[t, j]
            if (not np.isfinite(pr) or pr > q_out) and d >= min_hold:
                cash += v * (1 - cs)
                del pos[j]
        free = slots - len(pos)
        if free > 0:
            slot = (cash + sum(p[0] for p in pos.values())) / slots
            for j in order[t]:
                if free <= 0 or not np.isfinite(PR[t, j]) or PR[t, j] > q_in:
                    break
                if j in pos or not E.buyable[t, j]:
                    continue
                amt = min(slot, cash)
                if amt <= 1e-12:
                    break
                cash -= amt
                pos[j] = [amt / (1 + cb), 0]
                free -= 1
        got = 0.0
        for j, p in pos.items():
            x = E.inc[t, j]
            got += p[0] * (x if np.isfinite(x) else 0.0)
            g = E.G_price[t, j]
            p[0] *= 1 + (g if np.isfinite(g) else 0.0)
            p[1] += 1
        inc.append(got)
        val.append(cash + sum(p[0] for p in pos.values()))
    idx = E.dates[lo:hi]
    return pd.Series(inc, index=idx), pd.Series(val, index=idx)


def income_series(ER, S, rule, start):
    """두 시대를 이어서 (2019 첫날 앞 시대의 끝 평가액으로 다시 시작)."""
    parts_i, parts_v, V0 = [], [], 1.0
    for e, E in ER.items():
        lo, hi = E.rows(start, None)
        if hi - lo < 5:
            continue
        i, v = simulate_income(E, S[e], rule, max(start, str(E.dates[E.i0].date())), None, V0)
        parts_i.append(i)
        parts_v.append(v)
        V0 = float(v.iloc[-1])
    return pd.concat(parts_i), pd.concat(parts_v)


GAP_YEARS = (2018,)            # 2018 연말 배당은 두 자료 이음매라 빠짐 (div_lab 한계) → 흔들림 계산에서 뺌


def yearly_income(inc, val, start):
    inc, val = inc[inc.index >= pd.Timestamp(start)], val[val.index >= pd.Timestamp(start)]
    yi = inc.groupby(inc.index.year).sum()
    v0 = val.groupby(val.index.year).first()
    last = inc.index[-1]
    full = [y for y in yi.index if (inc.index.year == y).sum() > 200 and not (y == last.year and last.month < 12)]   # 끝나지 않은 해는 뺌
    yi, v0 = yi.reindex(full), v0.reindex(full)
    ok = [y for y in full if y not in GAP_YEARS and (y - 1) not in GAP_YEARS]
    chg = (yi / yi.shift(1) - 1).reindex(ok).dropna()
    good = yi.drop([y for y in GAP_YEARS if y in yi.index])
    return {"years": [int(y) for y in full], "income": [round(float(v), 4) for v in yi], "value": [round(float(v), 4) for v in v0],
            "yield": [round(float(a / b), 4) for a, b in zip(yi, v0)], "gap_years": list(GAP_YEARS),
            "worst_drop": round(float(chg.min()), 4) if len(chg) else None, "worst_year": int(chg.idxmin()) if len(chg) else None,
            "growth": round(float((good.iloc[-1] / good.iloc[0]) ** (1 / max(1, good.index[-1] - good.index[0])) - 1), 4) if len(good) > 1 else None,
            "down_years": int((chg < 0).sum()), "n": int(len(chg)), "min_vs_first": round(float(good.min() / good.iloc[0]), 3) if len(good) else None}


def years_to_target(m, start_cash, monthly, target, infl=0.0, n_paths=NPATH, block=12, seed=0, max_years=MAXY):
    """달 수익 m 을 12개월 덩어리로 이어 붙인 길마다 '모은 돈 ≥ 목표(물가만큼 커짐)' 가 처음 되는 달 → 해 단위 분포."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(m, float)
    L, n = len(arr), max_years * 12
    nb = math.ceil(n / block)
    starts = rng.integers(0, L - block + 1, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :n]
    R = arr[idx]
    V = np.full(n_paths, float(start_cash))
    hit = np.full(n_paths, np.inf)
    tg = target * (1 + infl) ** (np.arange(1, n + 1) / 12)
    for k in range(n):
        V = V * (1 + R[:, k]) + monthly
        newly = (V >= tg[k]) & ~np.isfinite(hit)
        hit[newly] = (k + 1) / 12
    def q(p):
        v = float(np.percentile(hit, p, method="higher"))
        return round(v, 1) if np.isfinite(v) else None
    return {"p50": q(50), "p90": q(90), "within10": round(float((hit <= 10).mean()), 3), "within20": round(float((hit <= 20).mean()), 3),
            "within30": round(float((hit <= 30).mean()), 3)}


def draw_success(m, capital, draw, years=DRAW_YEARS, infl=INFL, n_paths=NPATH, block=12, seed=0):
    """모은 돈 capital 에서 매달 draw 원(해마다 물가만큼 늘림)을 꺼낼 때: years 년 동안 바닥나지 않을 확률 · 끝에 남은 돈(지금 돈 가치) 가운데."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(m, float)
    L, n = len(arr), years * 12
    nb = math.ceil(n / block)
    starts = rng.integers(0, L - block + 1, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :n]
    R = arr[idx]
    V = np.full(n_paths, float(capital))
    dead = np.zeros(n_paths, bool)
    for k in range(n):
        V = V * (1 + R[:, k]) - draw * (1 + infl) ** (k / 12)
        dead |= V <= 0
        V = np.where(dead, 0.0, V)
    real = V / (1 + infl) ** years
    return {"ok": round(float((~dead).mean()), 3), "end_real_p50": round(float(np.median(real))), "end_real_p10": round(float(np.percentile(real, 10)))}


def capital_for(m, draw, prob=0.9, **kw):
    lo, hi = 1e7, 3e9
    for _ in range(30):
        mid = (lo + hi) / 2
        if draw_success(m, mid, draw, **kw)["ok"] >= prob:
            hi = mid
        else:
            lo = mid
    return round(hi, -6)


def run(log=print):
    import div_lab as DL
    import score_lab as SC
    t0 = time.time()
    SC.set_factors(True)
    res = {"goal_month": GOAL_MONTH, "goal_year": GOAL_YEAR, "tax": TAX, "infl": INFL, "ports": PORTS}
    ER = SC.eras(lambda m: None)
    DL.attach_dividends(ER, lambda m: None)
    Sx = DL.scores(ER)
    specs = {"ETF대용": ("D1", {"slots": 30, "q_in": 0.30, "q_out": 0.60, "min_hold": 250, "stop": 0.0, "max_hold": 0}),
             "고배당·저변동": ("D4", DL.RULES["D4"]),
             "고배당3": ("D1", {**DL.RULES["D1"], "slots": 3, "q_in": 0.05})}
    cy = current_yields(ER["B"])
    res["current"] = cy
    hist, paths = {}, {}
    for k, (sk, rule) in specs.items():
        S = {e: Sx[e][sk] for e in ER}
        paths[k] = {}
        for st in ("2003-01-01", "2008-01-01"):
            i, v = income_series(ER, S, rule, st)
            paths[k][st] = yearly_income(i, v, st)
        y = pd.Series(paths[k]["2003-01-01"]["yield"], index=paths[k]["2003-01-01"]["years"]).drop(list(GAP_YEARS), errors="ignore")
        hist[k] = {"all": round(float(y.mean()), 4), "2019~": round(float(y[y.index >= 2019].mean()), 4)}
        SC._PREP.clear()
    res["hist_yield_after_tax"] = hist
    # 필요한 돈 (세후 배당률 = 세전 × (1 − 15.4%))
    need = {}
    for k in PORTS:
        a_now = cy[k] * (1 - TAX)
        need[k] = {"yield_now_gross": cy[k], "yield_now_after": round(a_now, 4), "need_now": round(GOAL_YEAR / a_now, -5),
                   "yield_hist_after": hist[k]["2019~"], "need_hist": round(GOAL_YEAR / hist[k]["2019~"], -5)}
        log(f"  {PORTS[k]:20s}: 지금 배당률 세전 {cy[k]:.1%} → 세후 {a_now:.1%} → 필요한 돈 {need[k]['need_now'] / 1e8:.2f}억 · "
            f"2019~ 평균 세후 {hist[k]['2019~']:.1%} → {need[k]['need_hist'] / 1e8:.2f}억 · 2003~ 평균 {hist[k]['all']:.1%}")
    res["need"] = need
    # 계좌별 (세전 배당률 × (1 − 계좌 세율))
    res["acct"] = ACCT_NAME
    res["need_by_acct"] = {k: {a: round(GOAL_YEAR / (cy[k] * (1 - t)), -5) for a, t in ACCT_TAX.items()} for k in PORTS}
    for k in PORTS:
        log(f"  {PORTS[k]:20s}: 계좌별 필요한 돈 " + " · ".join(f"{a} {v / 1e8:.2f}억" for a, v in res["need_by_acct"][k].items()))
    res["ten_pct"] = {"capital": 60_000_000, "best_yield_now": max(t["yield"] for t in cy["top10"]),
                      "income_now_by_port": {k: round(60_000_000 * cy[k] * (1 - ACCT_TAX["ISA"]) / 12, -3) for k in PORTS}}
    res["flat_yields"] = [{"after_tax": y, "need": round(GOAL_YEAR / y, -5)} for y in (0.03, 0.035, 0.04, 0.045, 0.05, 0.06)]
    # 받는 동안의 흔들림
    res["income_path"] = paths
    for k in PORTS:
        ip = paths[k]["2003-01-01"]
        log(f"  배당 흐름 {k}: 2003 에 사 두고 꺼내 씀 → 받은 배당 연 {ip['growth']:+.1%} 늘어남 · 줄어든 해 {ip['down_years']}/{ip['n']} · "
            f"최악 {ip['worst_drop']:+.0%} ({ip['worst_year']}) · 가장 적은 해 = 첫해의 {ip['min_vs_first']:.0%}")
    # 모으는 기간 (ISA 계획 달 수익)
    il = json.load(open("data/isa_lab.json", encoding="utf-8"))
    cv = pd.Series(il["curve"]["plan"], index=pd.to_datetime(il["curve"]["date"]))
    m = cv.pct_change().dropna()
    m = m[m.index < pd.Timestamp("2026-11-01")]                      # 끝 달은 덜 찬 달이라 뺌
    res["accum"] = {"what": "ISA 계획 (고배당 ETF 50 + 매달 회전 50) 달 수익 · 2005~", "monthly": [round(float(x), 5) for x in m],
                    "dates": [str(d.date())[:7] for d in m.index], "conservative": ["2008-01", "2018-12"]}
    target = res["need_by_acct"]["고배당·저변동"]["ISA"]
    res["target_main"] = {"port": "고배당·저변동", "acct": "ISA", "capital": target}
    tab = []
    for base, (a, b) in {"2005~ 전체": ("2005-01-01", None), "2008~2018 (보수적)": ("2008-01-01", "2018-12-31")}.items():
        mm = m[(m.index >= pd.Timestamp(a)) & ((m.index <= pd.Timestamp(b)) if b else True)]
        for c in CONTRIB:
            for infl in (0.0, INFL):
                r = years_to_target(mm.to_numpy(), 0, c, target, infl, seed=zlib.crc32(f'{base}|{c}|{infl}'.encode()))
                tab.append({"base": base, "monthly": c, "infl": infl, **r})
                if infl == 0.0:
                    log(f"  [{base}] 0원 + 매달 {c / 1e4:,.0f}만 → {target / 1e8:.2f}억: 가운데 {r['p50']}년 · 나쁜 쪽 {r['p90']}년 · 20년 안 {r['within20']:.0%}")
    res["years_table"] = tab
    # 배당 대신 일부를 팔아 꺼내 쓰면 (매달 GOAL_MONTH, 물가만큼 늘림, 30년)
    dr = {"draw": GOAL_MONTH, "years": DRAW_YEARS, "rows": [], "need90": {}}
    for base, (a, b) in {"2005~ 전체": ("2005-01-01", None), "2008~2018 (보수적)": ("2008-01-01", "2018-12-31")}.items():
        mm = m[(m.index >= pd.Timestamp(a)) & ((m.index <= pd.Timestamp(b)) if b else True)].to_numpy()
        for cap in DRAW_CAPS:
            r = draw_success(mm, cap, GOAL_MONTH, seed=zlib.crc32(f'draw|{base}|{cap}'.encode()))
            dr["rows"].append({"base": base, "capital": cap, **r})
            log(f"  [{base}] {cap / 1e8:.1f}억에서 매달 {GOAL_MONTH / 1e4:.0f}만(물가 2%) 꺼냄 → 30년 버틸 확률 {r['ok']:.0%} · 남은 돈(지금 가치) 가운데 {r['end_real_p50'] / 1e8:.2f}억")
        dr["need90"][base] = capital_for(mm, GOAL_MONTH, seed=zlib.crc32(f'need|{base}'.encode()))
        log(f"  [{base}] 90% 버티려면 {dr['need90'][base] / 1e8:.2f}억")
    res["draw"] = dr
    res["generated"] = time.strftime("%Y-%m-%d %H:%M")
    res["secs"] = round(time.time() - t0)
    for fn in (OUT, DOCS):
        json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"끝 ({res['secs']}초) → {OUT} · {DOCS}")
    return res


if __name__ == "__main__":
    run(lambda m: print(m, flush=True))
