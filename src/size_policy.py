"""
자산 규모별 전략 — 작을 때는 공격적으로(수익 쪽), 커지면 안정적으로(전천후 쪽).

  두 엔진 (둘 다 이미 두 시대에서 검증한 것 — 여기서 새로 찾지 않는다)
    공격: 듀얼 모멘텀 상위 2 (allweather 카드 DM:2) — 매달 지수·S&P·나스닥·금·달러·장기채 중 120일 수익 상위 2개 반반,
          단기채보다 못한 자리는 단기채. ETF 2개만 들어 작은 돈으로도 그대로 따라 할 수 있다.
          시대 A(대용 2005~2018) 샤프 0.69 · 연 9.7% · 최대낙폭 -17.5% / 시대 B(2016~) 0.87 · 15.9% · -28.9%
    안정: 전천후 16장 혼합 (allweather.live_mix) — A 0.89 · 6.5% · -9.1% / B 1.15 · 6.8% · -8.4%

  섞는 비율 (공격 몫 a) — '감당할 최대 손실 금액'(loss_budget) 기준
    a = (loss_budget ÷ 최고 자산 − 안정 최악 낙폭 9%) ÷ (공격 최악 낙폭 29% − 9%), 0~1 로 자르고 10% 단위로 내림
    → 과거 최악 낙폭으로 잃을 수 있는 돈 ≈ loss_budget 을 넘지 않게. 자산이 커질수록 a 가 줄어 안정 쪽으로.
    '최고 자산'(고점) 기준이라 손실이 나도 다시 공격적으로 돌아가지 않는다 (잃은 걸 만회하려 위험을 키우지 않음).

  정수 주식 — 1주 가격이 1만~16만 원이라 작은 돈에선 비중대로 못 산다. 목표와의 차이가 가장 줄어드는 순서로 1주씩 채운다.
  다시 맞추기: 매달 첫 거래일 (두 엔진 모두 매달 바꾸므로) + 처음 시작할 때.
  비용: 수수료 0.0186% + 호가 반 칸 + 0.005% · 채권·달러·금·미국 ETF 는 이익 난 매도마다 15.4% (보수적).

  python src/size_policy.py → data/size_policy.json (20만 원 시작 백테스트 · 입금 시나리오 · 앞으로의 모의 기록)
"""
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allweather as AW  # noqa: E402

OUT = "data/size_policy.json"
DEFAULT = {"enabled": True, "loss_budget": 1_000_000, "aggr_card": "DM:2", "mdd_aggr": 0.29, "mdd_stable": 0.09,
           "max_aggr": 1.0, "step": 0.1, "cash_buffer": 0.01, "paper_start": "2026-11-09", "paper_cash": 200_000}


def conf(cfg=None):
    if cfg is None:
        try:
            import settings
            cfg = settings.load()
        except Exception:
            cfg = {}
    return {**DEFAULT, **(((cfg.get("short") or {}).get("size_policy")) or {})}


def aggr_share(hwm, sp=None):
    """최고 자산(hwm) → 공격 몫 a (0~max_aggr, step 단위 내림)."""
    sp = {**DEFAULT, **(sp or {})}
    if not hwm or hwm <= 0:
        return float(sp["max_aggr"])
    x = (sp["loss_budget"] / hwm - sp["mdd_stable"]) / (sp["mdd_aggr"] - sp["mdd_stable"])
    x = min(float(sp["max_aggr"]), max(0.0, x))
    st = float(sp["step"])
    return round(math.floor(x / st + 1e-9) * st, 4)


def tiers(sp=None, points=(200_000, 1_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000, 20_000_000)):
    """자산 규모별 공격 몫 표 (화면·문서용)."""
    sp = {**DEFAULT, **(sp or {})}
    return [{"equity": p, "aggr": aggr_share(p, sp), "worst_loss": round(p * (aggr_share(p, sp) * sp["mdd_aggr"] + (1 - aggr_share(p, sp)) * sp["mdd_stable"]))}
            for p in points]


def allocate(w, px, equity, cb=None, lam=0.5):
    """목표 비중 w (A) · 1주 가격 px (A) · 쓸 돈 equity → 정수 주식 수 (A).
    먼저 목표 금액 안에서 내림으로 사고, 남은 돈으로 '종목별 목표와의 차이 + lam × 놀고 있는 현금'이 가장 줄어드는 1주씩 더 산다.
    lam=0.5: 놀리는 현금은 절반만 벌점 → 5% 자리에 비싼 1주를 몰아넣지는 않되, 2개짜리 공격 엔진은 현금으로 놀지 않게."""
    w = np.nan_to_num(np.asarray(w, float))
    px = np.asarray(px, float)
    ok = np.isfinite(px) & (px > 0) & (w > 0)
    cb = np.zeros(len(w)) if cb is None else np.nan_to_num(np.asarray(cb, float))
    unit = np.where(ok, px * (1 + cb), np.inf)
    ideal = np.where(ok, w * equity, 0.0)
    q = np.where(ok, np.floor(ideal / unit), 0.0)
    left = equity - float((q * np.where(ok, unit, 0)).sum())
    while True:
        cur = np.abs(ideal - q * np.where(ok, px, 0))
        nxt = np.abs(ideal - (q + 1) * np.where(ok, px, 0))
        gain = np.where(ok & (unit <= left + 1e-9), cur - nxt + lam * np.where(ok, unit, 0), -np.inf)
        j = int(np.argmax(gain))
        if not np.isfinite(gain[j]) or gain[j] <= 0:
            break
        q[j] += 1
        left -= unit[j]
    return q.astype(int)


# ══════════════════════════════════════════════════════════════
# 목표 비중 경로 (백테스트) · 내일 목표 (매일 밤)
# ══════════════════════════════════════════════════════════════
def weight_paths(dates, Op, Cp, sp=None, start=None, stress=1.0):
    """t행 = t일 시가에 맞출 목표 (전천후 16장 W_aw, 공격 카드 W_ag). 반환 (W_aw, W_ag, 시작 행)."""
    sp = {**DEFAULT, **(sp or {})}
    F = AW.features(Cp, dates)
    rules, Ws, recs, R = AW.card_returns(dates, Op, Cp, F, rules=AW.allowed_rules(), stress=stress)
    Rx = AW.excess_over_cash(R, rules)
    s = int(np.searchsorted(dates, pd.Timestamp(start or AW.MIX_START)))
    ch = AW.choose_mix(Rx, F["reg"], s, cash=rules.index("H:CASH"), resched=AW.month_start(dates, pad=1), **AW.LIVE_MIX)
    P = [AW.desired(W) for W in Ws]
    wo, _ = AW.mix_weights(P, Ws, ch, s, len(dates))
    card = sp["aggr_card"]
    W_ag = P[rules.index(card)] if card in rules else AW.desired(AW.targets(F, Cp, card))    # 진화 카드(DM:2t 등)는 혼합 목록 밖
    return AW.desired(wo), W_ag, s


def tomorrow(m, sp=None):
    """allweather.live_mix 결과 → 내일 목표 {코드: 비중} 두 개 + 내일이 다시 맞추는 날인가."""
    sp = {**DEFAULT, **(sp or {})}
    aw = {AW.CODES[j]: round(float(x), 4) for j, x in enumerate(m["target"]) if x > 1e-4}
    card = sp["aggr_card"]
    now = m["card_now"][card] if card in m["card_now"] else AW.desired(AW.targets(m["F"], m["Cp"], card, pad=1))[-1]
    ag = {AW.CODES[j]: round(float(x), 4) for j, x in enumerate(now) if x > 1e-4}
    turn = bool(AW.month_start(m["dates"], pad=1)[-1])
    px = AW.ffill(m["Cp"])[-1]
    return {"aw": aw, "aggr": ag, "rebalance": turn, "card": sp["aggr_card"],
            "px": {AW.CODES[j]: round(float(px[j]), 2) for j in range(len(AW.CODES)) if np.isfinite(px[j])}}


def blend(aw, ag, a):
    codes = set(aw) | set(ag)
    return {c: (1 - a) * aw.get(c, 0.0) + a * ag.get(c, 0.0) for c in codes}


# ══════════════════════════════════════════════════════════════
# 정수 주식 시뮬레이션
# ══════════════════════════════════════════════════════════════
def simulate(dates, Op, Cp, W_aw, W_ag, s, cash0, sp=None, deposits=None, stress=1.0, integer=True, fixed_a=None):
    """매달 첫 거래일 시가에 (1-a)·전천후 + a·공격 으로 다시 맞춘다. a 는 그때까지의 최고 자산으로.
    deposits = {행 번호: 금액} (그날 시가 전에 입금). 반환: 자산곡선(E), 넣은 돈 누계, a 경로, 거래 수, 세금."""
    sp = {**DEFAULT, **(sp or {})}
    T, A = Cp.shape
    Cf = AW.ffill(Cp)
    taxr = AW.TAXR
    q = np.zeros(A)
    avg = np.zeros(A)
    cash = float(cash0)
    paid = float(cash0)
    hwm = float(cash0)
    E = np.full(T, np.nan)
    IN = np.full(T, np.nan)
    a_path = np.full(T, np.nan)
    reb = AW.month_start(dates)
    trades, tax_paid, cur_a = 0, 0.0, None
    for t in range(s, T):
        if deposits and t in deposits:
            cash += deposits[t]
            paid += deposits[t]
        po = Op[t]
        val_px = np.where(np.isfinite(po), po, Cf[t - 1] if t > 0 else Cf[t])
        eq = cash + float(np.nansum(q * val_px))
        hwm = max(hwm, eq)
        if t == s or reb[t] or (deposits and t in deposits):
            a = float(fixed_a) if fixed_a is not None else aggr_share(hwm, sp)
            w = (1 - a) * W_aw[t] + a * W_ag[t]
            cb, cs = AW.costs(np.where(np.isfinite(po), po, np.nan), stress)
            tradable = np.isfinite(po) & (po > 0)
            if integer:
                tq = allocate(np.where(tradable, w, 0), po, eq * (1 - sp["cash_buffer"]), cb)
                tq = np.where(tradable, tq, q)
            else:
                tq = np.where(tradable, w * eq * (1 - sp["cash_buffer"]) / np.where(tradable, po * (1 + cb), np.inf), q)
            for j in np.where(tradable & (tq < q - 1e-9))[0]:             # 매도 먼저
                n = q[j] - tq[j]
                gain = max(0.0, (po[j] - avg[j]) * n)
                tx = taxr[j] * gain
                cash += n * po[j] * (1 - cs[j]) - tx
                tax_paid += tx
                q[j] = tq[j]
                trades += 1
            for j in np.where(tradable & (tq > q + 1e-9))[0]:
                n = tq[j] - q[j]
                cost = n * po[j] * (1 + cb[j])
                if cost > cash + 1e-6:
                    n = math.floor(cash / (po[j] * (1 + cb[j]))) if integer else cash / (po[j] * (1 + cb[j]))
                    cost = n * po[j] * (1 + cb[j])
                if n <= 0:
                    continue
                avg[j] = (avg[j] * q[j] + po[j] * n) / (q[j] + n)
                q[j] += n
                cash -= cost
                trades += 1
            cur_a = a
        E[t] = cash + float(np.nansum(q * Cf[t]))
        IN[t] = paid
        a_path[t] = cur_a if cur_a is not None else np.nan
        hwm = max(hwm, E[t])
    return {"E": E, "IN": IN, "a": a_path, "trades": trades, "tax": tax_paid, "q": q, "cash": cash}


def stats_flow(E, IN, dates, s):
    """입금이 있으면 '돈의 시간가중 수익'(입금일 기준 나눠 이어 붙임)으로 샤프·연수익·낙폭."""
    e = pd.Series(E[s:], index=dates[s:]).astype(float)
    inn = pd.Series(IN[s:], index=dates[s:]).astype(float)
    dep = inn.diff().fillna(0)
    r = (e - dep) / e.shift(1) - 1
    r = r.iloc[1:].replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(r) < 20:
        return None
    U = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"cagr": round(float(U.iloc[-1] ** (1 / yrs) - 1), 4), "vol": round(float(r.std() * np.sqrt(252)), 4),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0, 3),
            "mdd": round(float((U / U.cummax() - 1).min()), 4), "final": round(float(e.iloc[-1])), "paid": round(float(inn.iloc[-1])),
            "worst_won": round(float((e - e.cummax()).min()))}


def forward(sp=None, prices=None):
    """앞으로의 모의 기록: paper_start 부터 paper_cash 로 같은 규칙 (상태 파일 없이 매번 처음부터 — 결과가 늘 같다)."""
    sp = {**DEFAULT, **(sp or {})}
    dates, Op, Cp = prices or AW.load_prices()
    if dates[-1] < pd.Timestamp(sp["paper_start"]):
        return {"start": sp["paper_start"], "days": 0, "equity": sp["paper_cash"], "note": "아직 시작 전"}
    W_aw, W_ag, _ = weight_paths(dates, Op, Cp, sp)
    s = int(np.searchsorted(dates, pd.Timestamp(sp["paper_start"])))
    res = simulate(dates, Op, Cp, W_aw, W_ag, s, sp["paper_cash"], sp)
    E = res["E"][s:]
    return {"start": str(dates[s].date()), "days": int(len(E)), "equity": round(float(E[-1])),
            "ret": round(float(E[-1] / sp["paper_cash"] - 1), 4), "mdd": round(float((E / np.maximum.accumulate(E) - 1).min()), 4),
            "aggr": float(res["a"][-1]), "holdings": {AW.CODES[j]: int(x) for j, x in enumerate(res["q"]) if x > 0},
            "cash": round(res["cash"])}


CHALLENGERS = ("DM:2v", "DM:2t")                 # 5회차 도전자 (실제 돈 아님 — optimize_lab --round 5, STRATEGY 3-6-10-10)


def challenger_forward(sp=None, prices=None, cards=CHALLENGERS):
    """그림자 기록 (실제 돈 아님): 공격 엔진 자리에 도전자 카드를 꽂고, 같은 규모별 규칙·20만 원·정수 주식으로 paper_start 부터.
    지금 엔진(DM:2) 기록과 나란히 — 상태 파일 없이 매번 처음부터 계산해 결과가 늘 같다."""
    sp = {**DEFAULT, **(sp or {})}
    dates, Op, Cp = prices or AW.load_prices()
    st = pd.Timestamp(sp["paper_start"])
    if dates[-1] < st:
        return {"start": sp["paper_start"], "days": 0, "note": "아직 시작 전"}
    W_aw, W_dm, _ = weight_paths(dates, Op, Cp, sp)
    F = AW.features(Cp, dates)
    s = int(np.searchsorted(dates, st))
    out = {"start": str(dates[s].date()), "what": "그림자 기록 — 실제 돈 아님 (공격 엔진 자리만 바꿈)", "cards": {}}
    for card, W_ag in [("DM:2", W_dm)] + [(c, AW.desired(AW.targets(F, Cp, c))) for c in cards]:
        res = simulate(dates, Op, Cp, W_aw, W_ag, s, sp["paper_cash"], sp)
        E = res["E"][s:]
        out["cards"][card] = {"desc": AW.DM_EVOLVED.get(card, "지금 공격 엔진"), "days": int(len(E)), "equity": round(float(E[-1])),
                              "ret": round(float(E[-1] / sp["paper_cash"] - 1), 4),
                              "mdd": round(float((E / np.maximum.accumulate(E) - 1).min()), 4),
                              "holdings": {AW.CODES[j]: int(x) for j, x in enumerate(res["q"]) if x > 0}}
    return out


def overnight_forward(sp=None, prices=None):
    """그림자 기록 (실제 돈 아님): 매일 단기 매매 — 추세 위에서만 KODEX 200 밤사이 보유(D1)와 DM:2 반반 섞기(D2)를
    paper_start 부터 앞으로 쌓는다. 4회차 시험에서 비용 3배에 무너져 실제 돈에는 안 넣었다 — 경매 체결(장 시작·마감 단일가)은
    호가 차이가 없어 실제 비용이 가정보다 낮을 수 있으니, 앞으로의 기록으로 다시 본다."""
    import optimize_lab as O
    sp = {**DEFAULT, **(sp or {})}
    dates, Op, Cp = prices or AW.load_prices()
    st = pd.Timestamp(sp["paper_start"])
    if dates[-1] < st:
        return {"start": sp["paper_start"], "days": 0, "note": "아직 시작 전"}
    on = pd.Series(O.overnight_stream(dates, Op, Cp), index=dates)
    W_aw, W_ag, _ = weight_paths(dates, Op, Cp, sp)
    s = int(np.searchsorted(dates, st))
    dm = simulate(dates, Op, Cp, W_aw, W_ag, s, sp["paper_cash"], sp, fixed_a=1.0)["E"][s:]
    r_on = on.to_numpy()[s:]
    r_dm = np.r_[0.0, dm[1:] / dm[:-1] - 1]
    E1 = sp["paper_cash"] * np.cumprod(1 + r_on)
    E2 = sp["paper_cash"] * np.cumprod(1 + 0.5 * r_on + 0.5 * r_dm)
    return {"start": str(dates[s].date()), "days": int(len(E1)), "what": "그림자 기록 — 실제 돈 아님",
            "D1_overnight": {"equity": round(float(E1[-1])), "ret": round(float(E1[-1] / sp["paper_cash"] - 1), 4),
                             "nights_in": int((r_on != 0).sum())},
            "D2_mix": {"equity": round(float(E2[-1])), "ret": round(float(E2[-1] / sp["paper_cash"] - 1), 4)},
            "DM2": {"equity": round(float(dm[-1])), "ret": round(float(dm[-1] / sp["paper_cash"] - 1), 4)}}


# ══════════════════════════════════════════════════════════════
# 보고서
# ══════════════════════════════════════════════════════════════
def report(sp=None):
    sp = {**DEFAULT, **(sp or {})}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "rule": sp, "tiers": tiers(sp), "real": {}, "proxy": {}}
    dates, Op, Cp = AW.load_prices()
    W_aw, W_ag, s = weight_paths(dates, Op, Cp, sp)
    month = AW.month_start(dates)
    dep = {t: 200_000 for t in range(s + 1, len(dates)) if month[t]}
    runs = {"규모별 (20만 원 시작)": dict(), "전천후 16장 고정 (20만 원)": dict(fixed_a=0.0), "공격(DM:2) 고정 (20만 원)": dict(fixed_a=1.0),
            "규모별 (20만 원 + 매달 20만 원 입금)": dict(deposits=dep), "전천후 고정 (같은 입금)": dict(deposits=dep, fixed_a=0.0),
            "규모별 (분수 주식 — 정수 영향 비교용)": dict(integer=False)}
    for name, kw in runs.items():
        for tag, st in (("x1", 1.0), ("x3", 3.0)):
            r = simulate(dates, Op, Cp, W_aw, W_ag, s, 200_000, sp, stress=st, **kw)
            row = stats_flow(r["E"], r["IN"], dates, s)
            row.update({"trades": r["trades"], "tax": round(r["tax"]),
                        "a_last": None if np.isnan(r["a"][-1]) else float(r["a"][-1]),
                        "a_changes": int(np.nansum(np.abs(np.diff(r["a"][s:])) > 1e-9))})
            yr = pd.Series(r["E"][s:], index=dates[s:])
            if not kw.get("deposits"):
                y = yr.groupby(yr.index.year).last()
                row["years"] = {str(k): round(float(v), 4) for k, v in (y / y.shift(1).fillna(200_000) - 1).items()}
            out["real"].setdefault(name, {})[tag] = row
    # 대용 자산 2005~2018 (가격 수준이 실제와 달라 정수 효과는 못 봄 → 분수 주식)
    import history_lab as H
    with H.flat_slip():
        d2, o2, c2 = H.proxy_prices(H.load_hist())
        Wa, Wg, s2 = weight_paths(d2, o2, c2, sp, start="2005-01-03")
        e2 = int(np.searchsorted(d2, pd.Timestamp("2018-12-28"), side="right"))
        for name, kw in {"규모별 (20만 원)": {}, "전천후 고정": {"fixed_a": 0.0}, "공격 고정": {"fixed_a": 1.0},
                         "규모별 (매달 20만 원 입금)": {"deposits": {t: 200_000 for t in range(s2 + 1, e2) if AW.month_start(d2)[t]}}}.items():
            r = simulate(d2[:e2], o2[:e2], c2[:e2], Wa[:e2], Wg[:e2], s2, 200_000, sp, integer=False, **kw)
            out["proxy"][name] = stats_flow(r["E"], r["IN"], d2[:e2], s2)
    out["forward"] = forward(sp, (dates, Op, Cp))
    return out


def main():
    res = report(conf())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print("■ 자산 규모별 공격 몫 (감당할 최대 손실", f"{res['rule']['loss_budget']:,}원)")
    for t in res["tiers"]:
        print(f"  {t['equity']:>12,}원 → 공격 {t['aggr']:.0%} · 과거 최악 낙폭이면 약 {t['worst_loss']:,}원 손실")
    print("■ 실제 ETF 2016~ (정수 주식)")
    for n, v in res["real"].items():
        x = v["x1"]
        print(f"  {n:30s} 연 {x['cagr']:+.1%} 샤프 {x['sharpe']:.2f} 낙폭 {x['mdd']:.1%} 최종 {x['final']:,}원 (넣은 돈 {x['paid']:,}) 거래 {x['trades']} · 3배 비용 샤프 {v['x3']['sharpe']:.2f}")
    print("■ 대용 자산 2005~2018")
    for n, x in res["proxy"].items():
        print(f"  {n:24s} 연 {x['cagr']:+.1%} 샤프 {x['sharpe']:.2f} 낙폭 {x['mdd']:.1%}")
    print("■ 앞으로의 모의 기록:", res["forward"])


if __name__ == "__main__":
    main()
