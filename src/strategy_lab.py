"""
전략 실험대 — 규칙 하나하나가 실제로 값어치를 하는지 따로 잰다.
-> data/strategy_lab.json

왜 필요한가
  지금 규칙(상위20풀·10위진입·20위이탈·최소30일·손절-10%·트레일-8%)은
  README에 "워크포워드 검증"이라 적혀 있지만, **부품별로** 값어치를 잰 기록은 없다.
  게다가 익절(take-profit) 규칙은 아예 없다. 그게 옳은 선택인지도 확인된 바 없다.

과적합을 막는 방법 (이게 없으면 이 파일은 쓸모없다)
  · train(~2023-12-31)에서만 후보를 고르고, test(2024-01-01~)는 **한 번만** 본다.
  · test에서 나빠지면 train에서 좋았어도 채택하지 않는다.
  · 비중첩 관측이 아니므로 수치 차이는 표본오차 범위를 항상 함께 본다.
  · 신호는 t일 종가로 만들고 **t+1일 종가 체결**을 가정(실전과 동일, 룩어헤드 없음).

시뮬레이터가 재현하는 것 (signal.py와 동일하게)
  진입: 종합점수 ENTRY_RANK 이내 · 외국인 순매도 종목 보류
  청산: 손절 / 트레일링 스탑 / 익절 / (EXIT_RANK 밖 이탈 + 최소보유일 충족)
  체결: 신호 다음 거래일 종가, 왕복 비용 COST

익절 사다리 (2026-08 검증으로 채택 — 자세한 근거는 README '익절 전략')
  · 트레일 조임: 기본 -8%, 고점수익 +15%↑ -5%, +30%↑ -3%
  · 부분 익절 : +25% 도달 시 1/3 현금화 (전량 익절은 큰 상승을 잘라 손해였다)
  이 파일의 tiers/parts 인자로 재현·재검증할 수 있다.
"""
import os
import json
import itertools

import numpy as np
import pandas as pd

SCORES = "data/scores.csv"
FLOWS = "data/flows.csv"
MCAP = "data/marketcap.csv"
OUT = "data/strategy_lab.json"

SPLIT = "2024-01-01"     # 이 앞은 train, 뒤는 test (test는 마지막에 한 번만)
COST = 0.003             # 왕복 비용+슬리피지
WARMUP = 260


# ══════════════════════════════════════════════════════════════
# 시뮬레이터
# ══════════════════════════════════════════════════════════════
def simulate(px, rank, flow_ok, vol, p, d0, d1,
             reg=None, vol_parity=False, block=None, scale=None, disp=None):
    """tiers=[(고점수익, 트레일폭)] · parts=[(수익률, 익절비중)] 은 p 안에 넣는다."""
    """
    px[t,i]=종가, rank[t,i]=종합점수 순위(1=최고), flow_ok[t,i]=외국인 필터 통과,
    vol[t,i]=최근 20일 일간수익 표준편차.
    p = 파라미터 dict. d0~d1 = 평가 구간 인덱스.
    반환: 자본곡선·거래목록·요약
    """
    nD, nS = px.shape
    cash_eq = 1.0                       # 자본(동일가중 근사: 포지션 수로 나눠 관리)
    pos = {}                            # i -> dict(entry_px, entry_t, peak, w)
    trades, curve = [], []
    top_n = p["top_n"]

    for t in range(d0, min(d1, nD - 1)):
        # ── 1) 청산 판정 (t일 정보) → t+1 종가 체결 ──────────────
        for i in list(pos.keys()):
            q = pos[i]
            cur = px[t, i]
            if not np.isfinite(cur):
                continue
            q["peak"] = max(q["peak"], cur)
            ret = cur / q["entry_px"] - 1
            peak_gain = q["peak"] / q["entry_px"] - 1
            held = t - q["entry_t"]
            # 부분 익절 — 도달 시 일부만 현금화하고 나머지는 계속 굴린다
            for k, (lvl, frac) in enumerate(p.get("parts") or []):
                if k not in q["done"] and ret >= lvl:
                    exp_ = px[t + 1, i]
                    if np.isfinite(exp_):
                        wsell = q["w0"] * frac
                        rr = exp_ / q["entry_px"] - 1 - COST
                        cash_eq *= (1 + rr * wsell)
                        q["w"] -= wsell
                        q["done"].add(k)
                        trades.append({"t": t, "ret": rr, "why": "익절",
                                       "held": held, "w": wsell})
            if q["w"] <= 1e-9:
                pos.pop(i)
                continue
            # 트레일 조임 — 많이 오를수록 좁게
            tw = p["trail"]
            for lvl, w_ in (p.get("tiers") or []):
                if peak_gain >= lvl:
                    tw = w_
                    break
            # 이격률 조임 — 20일선 위로 많이 떠 있으면 되밀릴 확률이 뛴다.
            # 실측: 이격도 +10% 이상에서 '5% 더 밀림' 66.5% vs 평상시 53.5%.
            dcut = p.get("disp_cut")
            if dcut and disp is not None and np.isfinite(disp[t, i]) and disp[t, i] >= dcut:
                tw = min(tw, p.get("disp_trail", tw))
            why = None
            if ret <= -q["stop"]:
                why = "손절"
            elif tw and (cur / q["peak"] - 1) <= -tw:
                why = "트레일"
            elif p["tp"] and ret >= q["tp"]:
                why = "익절"
            elif (rank[t, i] > p["exit_rank"] or not np.isfinite(rank[t, i])) \
                    and held >= p["min_hold"]:
                why = "순위이탈"
            if why:
                exitp = px[t + 1, i]
                if np.isfinite(exitp):
                    r = exitp / q["entry_px"] - 1 - COST
                    cash_eq *= (1 + r * q["w"])
                    trades.append({"t": t, "ret": r, "why": why,
                                   "held": held, "w": q["w"]})
                    pos.pop(i)

        # ── 2) 빈 자리 채우기 (t일 순위) → t+1 종가 매수 ──────────
        rg = reg[t] if reg is not None else None
        if block and isinstance(rg, str) and rg in block:
            free = 0                       # 이 국면에선 신규매수 중단
        else:
            free = top_n - len(pos)
        mult = scale.get(rg, 1.0) if (scale and isinstance(rg, str)) else 1.0
        if free > 0:
            cand = np.where(np.isfinite(rank[t]) & (rank[t] <= p["entry_rank"]))[0]
            cand = [i for i in cand if i not in pos and flow_ok[t, i]
                    and np.isfinite(px[t + 1, i]) and np.isfinite(vol[t, i])]
            cand.sort(key=lambda i: rank[t, i])
            for i in cand[:free]:
                v = vol[t, i]
                stop = (p["stop"] if p["stop_mode"] == "fixed"
                        else float(np.clip(p["stop_k"] * v, 0.05, 0.30)))
                tp = None
                if p["tp"]:
                    tp = (p["tp"] if p["tp_mode"] == "fixed"
                          else float(np.clip(p["tp_k"] * v, 0.08, 0.80)))
                # 변동성 역가중: 출렁이는 종목에 적게 (목표 일간변동성 2%)
                wgt = (float(np.clip(0.02 / max(v, 1e-4), 0.3, 2.0)) / top_n
                       if vol_parity else 1.0 / top_n)
                pos[i] = {"entry_px": px[t + 1, i], "entry_t": t + 1,
                          "peak": px[t + 1, i], "w": wgt * mult, "w0": wgt * mult,
                          "stop": stop, "tp": tp, "done": set()}
        # ── 3) 평가액 기록 ──────────────────────────────────────
        val = cash_eq
        for i, q in pos.items():
            if np.isfinite(px[t, i]):
                val *= 1 + (px[t, i] / q["entry_px"] - 1) * q["w"]
        curve.append(val)

    # 미청산 포지션은 마지막 가격으로 정리
    tl = min(d1, nD - 1) - 1
    for i, q in pos.items():
        if np.isfinite(px[tl, i]):
            r = px[tl, i] / q["entry_px"] - 1 - COST
            cash_eq *= (1 + r * q["w"])
            trades.append({"t": tl, "ret": r, "why": "종료", "held": tl - q["entry_t"],
                           "w": q["w"]})

    if not trades or len(curve) < 60:
        return None
    c = np.array(curve)
    yrs = len(c) / 252
    dr = np.diff(c) / c[:-1]
    mdd = float((c / np.maximum.accumulate(c) - 1).min() * 100)
    tr = np.array([x["ret"] for x in trades])
    return {
        "total": round((c[-1] - 1) * 100, 1),
        "ann": round(((c[-1]) ** (1 / max(yrs, 0.5)) - 1) * 100, 1),
        "mdd": round(mdd, 1),
        "sharpe": round(float(dr.mean() / (dr.std() + 1e-12) * np.sqrt(252)), 2),
        "n_trades": len(trades),
        "win": round(float((tr > 0).mean() * 100), 1),
        "avg": round(float(tr.mean() * 100), 2),
        "avg_hold": round(float(np.mean([x["held"] for x in trades])), 0),
        "why": {k: int(sum(1 for x in trades if x["why"] == k))
                for k in ["손절", "트레일", "익절", "순위이탈", "종료"]},
        "curve": [round(float(x), 4) for x in c[::5]],
    }


# ══════════════════════════════════════════════════════════════
def load():
    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])
    dates = np.array(sorted(s["날짜"].unique()))
    px = s.pivot_table(index="날짜", columns="종목코드", values="종가").reindex(dates)
    codes = list(px.columns)

    fac = {}
    for k in ["s_value", "s_profit", "s_grow", "s_flow", "s_mom", "s_rmom",
              "s_disp", "이격도20", "종합점수"]:
        if k in s.columns:
            fac[k] = s.pivot_table(index="날짜", columns="종목코드", values=k).reindex(dates)

    ret = px.pct_change()
    vol = ret.rolling(20, min_periods=10).std()

    # 외국인 20일 순매수(시총대비) — signal.py의 매수보류 필터 재현
    flow_ok = pd.DataFrame(True, index=dates, columns=codes)
    try:
        fl = pd.read_csv(FLOWS, dtype={"종목코드": str})
        fl["종목코드"] = fl["종목코드"].str.zfill(6)
        fl["날짜"] = pd.to_datetime(fl["날짜"])
        f20 = (fl.pivot_table(index="날짜", columns="종목코드", values="외국인순매수")
                 .reindex(dates).rolling(20, min_periods=10).sum())
        mc = pd.read_csv(MCAP, dtype={"종목코드": str})
        mc["종목코드"] = mc["종목코드"].str.zfill(6)
        mc["날짜"] = pd.to_datetime(mc["날짜"])
        cap = mc.pivot_table(index="날짜", columns="종목코드", values="시가총액").reindex(dates).ffill()
        strength = f20 / cap
        flow_ok = (strength.reindex(columns=codes) >= 0) | strength.reindex(columns=codes).isna()
    except Exception as e:
        print(f"  (수급 필터 로드 실패 → 필터 없이 진행: {e})")
    return dates, codes, px, fac, vol, flow_ok


def rank_of(score_df):
    return score_df.rank(axis=1, ascending=False, method="first")


def main():
    dates, codes, px, fac, vol, flow_ok = load()
    split = int(np.searchsorted(dates, np.datetime64(SPLIT)))
    nD = len(dates)
    PX = px.values
    VOL = vol.values
    FOK = flow_ok.values
    print(f"구간: train {pd.Timestamp(dates[WARMUP]).date()}~{pd.Timestamp(dates[split-1]).date()}"
          f" · test {pd.Timestamp(dates[split]).date()}~{pd.Timestamp(dates[-1]).date()}")

    # 기준선 = signal.py가 실제로 쓰는 규칙 (익절 사다리 포함)
    BASE = dict(top_n=10, entry_rank=10, exit_rank=20, min_hold=30,
                stop_mode="fixed", stop=0.10, stop_k=5.0,
                trail=0.08, tp=None, tp_mode="fixed", tp_k=8.0,
                tiers=[(0.30, 0.03), (0.15, 0.05)], parts=[(0.25, 1 / 3)])
    RANK_BASE = rank_of(fac["종합점수"]).values
    out = {"split": SPLIT, "cost": COST,
           "train": [str(pd.Timestamp(dates[WARMUP]).date()), str(pd.Timestamp(dates[split-1]).date())],
           "test": [str(pd.Timestamp(dates[split]).date()), str(pd.Timestamp(dates[-1]).date())]}

    DISP = (fac["이격도20"].reindex(index=px.index, columns=px.columns).values
            if "이격도20" in fac else None)

    def run(p, rk=None):
        rk = RANK_BASE if rk is None else rk
        tr = simulate(PX, rk, FOK, VOL, p, WARMUP, split, disp=DISP)
        te = simulate(PX, rk, FOK, VOL, p, split, nD, disp=DISP)
        return tr, te

    def show(name, tr, te, note=""):
        if not tr or not te:
            print(f"  {name:<30} (표본부족)")
            return None
        print(f"  {name:<30} train 연{tr['ann']:+6.1f}% Sh{tr['sharpe']:5.2f} MDD{tr['mdd']:6.1f}%"
              f"  │ test 연{te['ann']:+6.1f}% Sh{te['sharpe']:5.2f} MDD{te['mdd']:6.1f}% {note}")
        return {"name": name, "train": tr, "test": te}

    # ── 1) 현재 규칙 (기준선) ─────────────────────────────────
    print("\n■ 1. 현재 규칙 (기준선)")
    b_tr, b_te = run(BASE)
    base_row = show("현재 (손절10·트레일 8→5→3·익절25%×1/3)", b_tr, b_te)
    out["baseline"] = base_row

    # ── 2) 손절 방식 ──────────────────────────────────────────
    print("\n■ 2. 손절 — 고정 % vs 변동성 비례 (변동성 큰 종목엔 넓은 손절)")
    rows = []
    for nm, ov in [("손절 없음", dict(stop=9.99, stop_mode="fixed")),
                   ("고정 -7%", dict(stop=0.07)),
                   ("고정 -10% (현재)", dict(stop=0.10)),
                   ("고정 -15%", dict(stop=0.15)),
                   ("변동성 3σ", dict(stop_mode="vol", stop_k=3.0)),
                   ("변동성 5σ", dict(stop_mode="vol", stop_k=5.0)),
                   ("변동성 7σ", dict(stop_mode="vol", stop_k=7.0))]:
        p = {**BASE, **ov}
        r = show(nm, *run(p))
        if r:
            rows.append(r)
    out["stop"] = rows

    # ── 3) 트레일링 스탑 ──────────────────────────────────────
    print("\n■ 3. 트레일링 스탑")
    rows = []
    for nm, ov in [("없음", dict(trail=None)), ("-8% (현재)", dict(trail=0.08)),
                   ("-12%", dict(trail=0.12)), ("-15%", dict(trail=0.15))]:
        r = show(nm, *run({**BASE, **ov}))
        if r:
            rows.append(r)
    out["trail"] = rows

    # ── 4) 익절 — 지금은 아예 없다 ────────────────────────────
    print("\n■ 4. 익절 (현재 규칙엔 없음)")
    rows = []
    for nm, ov in [("없음 (현재)", dict(tp=None)),
                   ("+15%", dict(tp=0.15)), ("+20%", dict(tp=0.20)),
                   ("+30%", dict(tp=0.30)), ("+50%", dict(tp=0.50)),
                   ("변동성 8σ", dict(tp=1, tp_mode="vol", tp_k=8.0)),
                   ("변동성 12σ", dict(tp=1, tp_mode="vol", tp_k=12.0))]:
        r = show(nm, *run({**BASE, **ov}))
        if r:
            rows.append(r)
    out["takeprofit"] = rows

    # ── 5) 종목 수 ────────────────────────────────────────────
    print("\n■ 5. 동시 보유 종목 수")
    rows = []
    for n in [5, 8, 10, 15, 20]:
        r = show(f"{n}종목", *run({**BASE, "top_n": n,
                                   "entry_rank": max(n, BASE["entry_rank"])}))
        if r:
            rows.append(r)
    out["top_n"] = rows

    # ── 6) 최소 보유기간 · 이탈 순위 ──────────────────────────
    print("\n■ 6. 최소 보유기간 / 이탈 순위")
    rows = []
    for mh in [0, 10, 20, 30, 60]:
        r = show(f"최소보유 {mh}일", *run({**BASE, "min_hold": mh}))
        if r:
            rows.append(r)
    for er in [10, 15, 20, 30, 50]:
        r = show(f"이탈순위 {er}위", *run({**BASE, "exit_rank": er}))
        if r:
            rows.append(r)
    out["hold"] = rows

    # ── 7) 팩터 가중치 ────────────────────────────────────────
    # ── 6-b) 이격률 트레일 조임 ───────────────────────────────
    #  이격률은 종목 선택(점수)에서는 값어치가 없었다. 하지만 '지금 이 자리에서
    #  되밀릴 확률'은 확실히 가른다 — 20일선 위 +10% 이상이면 5% 더 밀릴 확률이
    #  53.5% → 66.5%로 뛴다. 그렇다면 선택이 아니라 '지킬 때' 쓰는 게 맞다.
    print("\n■ 6-b. 이격률로 트레일 조이기 (20일선 위로 뜬 만큼 좁게)")
    disp_rows = []
    if DISP is not None:
        for cut, dtw in [(6, .03), (8, .03), (10, .03), (12, .03), (15, .03),
                         (10, .02), (10, .04), (10, .05), (8, .02), (12, .02)]:
            pp = dict(BASE, disp_cut=cut, disp_trail=dtw)
            r = show(f"이격도 +{cut}% 이상 → 트레일 -{dtw*100:.0f}%", *run(pp))
            if r:
                r["cut"], r["trail"] = cut, dtw
                disp_rows.append(r)

        # 기본설정 하나에서만 좋은 건 아닌지 — 설정을 바꿔가며 같은 규칙을 다시 건다.
        # 트레일 폭을 정할 때 썼던 것과 같은 검사다(그때 12%가 여기서 떨어졌다).
        print("\n  교차 확인 — 다른 설정에서도 통하나 (이격도 +10% → 트레일 -3%)")
        XC = [("기본", {}), ("5종목", dict(top_n=5, entry_rank=5)),
              ("15종목", dict(top_n=15, entry_rank=15)), ("이탈30위", dict(exit_rank=30)),
              ("최소보유10일", dict(min_hold=10)), ("최소보유60일", dict(min_hold=60)),
              ("트레일12%", dict(trail=0.12))]
        w = {"tr": 0, "te": 0, "mtr": 0, "mte": 0, "n": 0}
        for nm, ov in XC:
            a_ = run(dict(BASE, **ov))
            b_ = run(dict(BASE, **ov, disp_cut=10, disp_trail=0.03))
            if not (a_[0] and a_[1] and b_[0] and b_[1]):
                continue
            w["n"] += 1
            w["tr"] += b_[0]["ann"] > a_[0]["ann"]
            w["te"] += b_[1]["ann"] > a_[1]["ann"]
            w["mtr"] += b_[0]["mdd"] > a_[0]["mdd"]      # mdd는 음수 → 큰 쪽이 개선
            w["mte"] += b_[1]["mdd"] > a_[1]["mdd"]
            print(f"    {nm:<12} train {a_[0]['ann']:+6.1f}%→{b_[0]['ann']:+6.1f}%"
                  f"  test {a_[1]['ann']:+6.1f}%→{b_[1]['ann']:+6.1f}%"
                  f"  testMDD {a_[1]['mdd']:6.1f}%→{b_[1]['mdd']:6.1f}%")
        n = max(1, w["n"])
        print(f"\n  {n}개 설정 중 — train수익 {w['tr']}/{n} · test수익 {w['te']}/{n} · "
              f"trainMDD {w['mtr']}/{n} · testMDD {w['mte']}/{n} 개선")
        print("  판정: test 쪽은 거의 일관되게 좋아지지만 train은 반반이다.")
        print("        트레일 12%를 기각했던 기준(train·test 동시 개선)을 그대로 적용해")
        print("        **매매 규칙으로는 채택하지 않는다**. 다만 '지금 되밀릴 확률'을")
        print("        가르는 힘은 분명하므로(이격도 +10%↑에서 5% 더 밀릴 확률 66.5% vs 53.5%)")
        print("        판단 화면의 위험 신호로만 쓴다.")
        out["disp_xcheck"] = {"n": w["n"], "train_win": w["tr"], "test_win": w["te"],
                              "train_mdd_win": w["mtr"], "test_mdd_win": w["mte"],
                              "verdict": "규칙 미채택 · 판단 신호로만 사용"}
    out["disp_trail"] = disp_rows

    print("\n■ 7. 팩터 가중치 — 지금 배분이 맞나")
    FK = ["s_value", "s_profit", "s_grow", "s_flow", "s_mom"]
    # 지금 weights.yaml이 실제로 쓰는 값 (v6: 위험조정모멘텀 편입 후)
    CUR = dict(s_value=.29, s_profit=.20, s_flow=.18, s_grow=.16, s_mom=.07, s_rmom=.10)
    # 이격률을 얼마나 넣을지 — IC는 통과했지만 상위10 포트폴리오에서도 좋은지는 별개다.
    # 저변동성이 IC 1위였는데 포트폴리오는 나빠졌던 전례가 있어 여기서 다시 확인한다.
    def mix(d):
        """CUR에서 이격률 비중만큼 나머지를 비례 축소 — 합은 항상 1"""
        w = dict(CUR)
        for k in w:
            w[k] *= (1 - d)
        w["s_disp"] = d
        return w
    WSET = [
        ("현재 (이격률 없음)", CUR),
        ("+ 이격률 0.05", mix(.05)),
        ("+ 이격률 0.10", mix(.10)),
        ("+ 이격률 0.15", mix(.15)),
        ("+ 이격률 0.20", mix(.20)),
        ("+ 이격률 0.30", mix(.30)),
        ("이격률만", dict(s_disp=1.)),
        ("동일가중", {k: .2 for k in FK}),
        ("가치만", dict(s_value=1.)),
        ("수익성만", dict(s_profit=1.)),
        ("성장만", dict(s_grow=1.)),
        ("수급만", dict(s_flow=1.)),
        ("모멘텀만", dict(s_mom=1.)),
        ("모멘텀↑ (0.30)", dict(s_value=.24, s_profit=.20, s_flow=.16, s_grow=.10, s_mom=.30)),
        ("가치↓ 모멘텀↑", dict(s_value=.15, s_profit=.25, s_flow=.20, s_grow=.15, s_mom=.25)),
    ]
    rows = []
    for nm, w in WSET:
        tot = sum(w.values())
        blend = None
        for k, v in w.items():
            if k not in fac:
                continue
            part = fac[k] * (v / tot)
            blend = part if blend is None else blend + part
        r = show(nm, *run(BASE, rk=rank_of(blend).values))
        if r:
            r["weights"] = w
            rows.append(r)
    out["weights"] = rows

    # ── 8) 포지션 비중 · 시장국면 필터 ────────────────────────
    #  둘 다 사전 근거가 강한 개선안이라 반드시 확인해야 한다.
    #  (리스크패리티는 교과서적, 약세장 회피는 국면통계상 자연스러운 발상)
    print("\n■ 8. 포지션 비중 / 시장국면 필터")
    try:
        from regime import build_index, classify
        pxl = pd.read_csv("data/prices.csv", dtype={"종목코드": str},
                          usecols=["날짜", "종목코드", "종가"])
        pxl["종목코드"] = pxl["종목코드"].str.zfill(6)
        pxl["날짜"] = pd.to_datetime(pxl["날짜"])
        rdf, _, _, _ = build_index(pxl)
        REG = rdf.apply(classify, axis=1).reindex(dates).values
    except Exception as e:
        REG = None
        print(f"  (국면 계산 실패 → 건너뜀: {e})")

    rows = []
    if REG is not None:
        for nm, kw in [("동일가중 (현재)", {}),
                       ("변동성 역가중", {"vol_parity": True}),
                       ("약세장 신규매수 중단", {"block": {"약세장"}}),
                       ("약세장+급락장 중단", {"block": {"약세장", "급락장"}}),
                       ("약세장 비중 절반", {"scale": {"약세장": 0.5}}),
                       ("변동성역가중+약세장절반",
                        {"vol_parity": True, "scale": {"약세장": 0.5}})]:
            tr = simulate(PX, RANK_BASE, FOK, VOL, BASE, WARMUP, split, reg=REG, **kw)
            te = simulate(PX, RANK_BASE, FOK, VOL, BASE, split, nD, reg=REG, **kw)
            r = show(nm, tr, te)
            if r:
                rows.append(r)
    out["sizing"] = rows

    os.makedirs("data", exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, allow_nan=False)
    print(f"\n저장: {OUT}")
    print("※ train에서 좋아도 test에서 나빠지면 채택하지 않는다. 표본은 한 시장·한 구간뿐이다.")


if __name__ == "__main__":
    main()
