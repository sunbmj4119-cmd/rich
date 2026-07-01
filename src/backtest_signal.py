"""
신호 기반 백테스트 엔진 (signal.py 로직 충실 재현)
목표: 수익 최대화 + 안정성(Sharpe/MDD) + 승률 동시 개선
4층 보완을 토글로 켜고 끄며 효과를 정량 측정한다.

  1층 weights      : 팩터 가중치 재조합 (s_flow 포함)
  2층 regime       : 시장 200일선 아래면 신규매수 중단
  3층 trail_stop   : 트레일링 스탑 (수익 길게)
  4층 flow_filter  : 외국인 순매도 종목 매수보류
"""
import pandas as pd
import numpy as np

TOP_N = 20
ENTRY_RANK = 10
EXIT_RANK = 20
MIN_HOLD = 30
STOP_LOSS = -0.10
COST = 0.0015  # 편도 0.15%

FACTOR_COL = {
    "momentum": "s_mom", "value": "s_value", "profit": "s_profit",
    "stability": "s_stab", "growth": "s_grow", "vtrend": "s_vtrend",
    "flow": "s_flow",
}


def load_data():
    s = pd.read_csv("data/scores.csv", dtype={"종목코드": str})
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])

    # s_flow 재생성 (외국인 20일 순매수 / 시가총액)  -- 횡단면 표준화
    fl = pd.read_csv("data/flows.csv", dtype={"종목코드": str})
    fl["종목코드"] = fl["종목코드"].str.zfill(6)
    fl["날짜"] = pd.to_datetime(fl["날짜"])
    fl = fl.sort_values(["종목코드", "날짜"])
    fl["외국인20"] = fl.groupby("종목코드")["외국인순매수"].transform(
        lambda x: x.rolling(20, min_periods=10).sum())
    mc = pd.read_csv("data/marketcap.csv", dtype={"종목코드": str})
    mc["종목코드"] = mc["종목코드"].str.zfill(6)
    mc["날짜"] = pd.to_datetime(mc["날짜"])
    fl = fl.merge(mc[["날짜", "종목코드", "시가총액"]], on=["날짜", "종목코드"], how="left")
    fl["강도"] = fl["외국인20"] / fl["시가총액"]
    s = s.merge(fl[["날짜", "종목코드", "강도"]], on=["날짜", "종목코드"], how="left")
    s = s.rename(columns={"강도": "flow_strength"})
    # s_flow = 일자별 횡단면 z-score (점수 스케일에 맞춤)
    g = s.groupby("날짜")["flow_strength"]
    s["s_flow"] = ((s["flow_strength"] - g.transform("mean")) /
                   g.transform("std").replace(0, np.nan))
    s["s_flow"] = s["s_flow"].fillna(0) * 10 + 50  # 다른 팩터와 유사 스케일

    return s


def compute_score(s, weights):
    tot = sum(weights.values())
    sc = np.zeros(len(s))
    for fac, w in weights.items():
        if w == 0:
            continue
        col = FACTOR_COL[fac]
        sc = sc + (w / tot) * s[col].fillna(s[col].median()).values
    out = s.copy()
    out["score"] = sc
    return out


def market_ma(s, win=200):
    """동일가중 시장지수와 이동평균(레짐 필터용)"""
    piv = s.pivot_table(index="날짜", columns="종목코드", values="종가")
    idx = piv.pct_change().mean(axis=1).add(1).cumprod()
    ma = idx.rolling(win, min_periods=win // 2).mean()
    regime_ok = (idx >= ma)  # True면 매수 허용
    return regime_ok


def backtest(s, weights, regime=False, trail_stop=None,
             flow_filter=True, verbose=False):
    """
    룩어헤드 제거: t일 종가로 점수/순위/필터 판단 -> t+1일 종가로 체결.
    포지션의 일간수익도 보유 중인 종목의 실제 가격변화로 계산.
    """
    s = compute_score(s, weights)
    dates = sorted(s["날짜"].unique())
    by_date = {d: g for d, g in s.groupby("날짜")}
    regime_ok = market_ma(s) if regime else None
    price = s.pivot_table(index="날짜", columns="종목코드", values="종가")

    positions = {}
    equity = 1.0
    nav = []
    trades = []
    n_trades = 0
    date_idx = {d: i for i, d in enumerate(dates)}
    pending_signal = None  # 전일 t에서 만든 (매수후보, 청산대상)

    for i, d in enumerate(dates):
        px = dict(zip(by_date[d]["종목코드"], by_date[d]["종가"]))

        # --- (A) 전일 신호를 오늘 종가로 체결 ---
        if pending_signal is not None:
            to_close, buy_list = pending_signal
            for code in to_close:
                if code in positions:
                    pos = positions.pop(code)
                    p = px.get(code, pos["last_px"])
                    ret = (p / pos["entry_px"] - 1) - 2 * COST
                    trades.append(ret)
            n_buyable = TOP_N - len(positions)
            for code in buy_list:
                if n_buyable <= 0:
                    break
                if code in positions:
                    continue
                p = px.get(code)
                if p is None:
                    continue
                positions[code] = {"entry_date": d, "entry_px": p,
                                   "peak_px": p, "last_px": p}
                n_buyable -= 1
                n_trades += 1

        # --- (B) 오늘 포지션 일간수익으로 NAV 갱신 ---
        if positions and i > 0:
            prev = dates[i - 1]
            rets = []
            for code, pos in positions.items():
                p_now = px.get(code)
                p_prev = price.loc[prev, code] if code in price.columns else None
                if p_now and p_prev and p_prev > 0:
                    rets.append(p_now / p_prev - 1)
                    pos["last_px"] = p_now
                    pos["peak_px"] = max(pos["peak_px"], p_now)
            day_ret = np.mean(rets) if rets else 0.0
        else:
            day_ret = 0.0
        equity *= (1 + day_ret)
        nav.append((d, equity))

        # --- (C) 오늘 종가로 내일 체결할 신호 생성 ---
        g = by_date[d].sort_values("score", ascending=False).reset_index(drop=True)
        g["순위"] = g.index + 1
        rank = dict(zip(g["종목코드"], g["순위"]))
        flow = dict(zip(g["종목코드"], g["flow_strength"]))

        to_close = []
        for code, pos in positions.items():
            p = px.get(code, pos["last_px"])
            ret = p / pos["entry_px"] - 1
            held = date_idx[d] - date_idx[pos["entry_date"]]
            r = rank.get(code, 999)
            if ret <= STOP_LOSS:
                to_close.append(code); continue
            if trail_stop is not None:
                if (p / pos["peak_px"] - 1) <= -trail_stop:
                    to_close.append(code); continue
            if r > EXIT_RANK and held >= MIN_HOLD:
                to_close.append(code)

        buy_allowed = True
        if regime is not False and regime_ok is not None:
            buy_allowed = bool(regime_ok.get(d, True))
        buy_list = []
        if buy_allowed:
            remain = set(positions.keys()) - set(to_close)
            cands = g[(g["순위"] <= ENTRY_RANK) & (~g["종목코드"].isin(remain))]
            for _, c in cands.iterrows():
                code = c["종목코드"]
                if flow_filter:
                    st = flow.get(code, 0)
                    if pd.notna(st) and st < 0:
                        continue
                buy_list.append(code)
        pending_signal = (to_close, buy_list)

    nav = pd.DataFrame(nav, columns=["날짜", "nav"]).set_index("날짜")
    return summarize(nav, trades, n_trades, s)


def summarize(nav, trades, n_trades, s):
    r = nav["nav"].pct_change().dropna()
    total = nav["nav"].iloc[-1] - 1
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    mdd = (nav["nav"] / nav["nav"].cummax() - 1).min()
    win = np.mean([t > 0 for t in trades]) if trades else 0

    # 구간별
    def seg(a, b):
        sub = nav[(nav.index >= a) & (nav.index <= b)]
        if len(sub) < 2:
            return None
        rr = sub["nav"].pct_change().dropna()
        tt = sub["nav"].iloc[-1] / sub["nav"].iloc[0] - 1
        sh = (rr.mean() / rr.std() * np.sqrt(252)) if rr.std() > 0 else 0
        return tt, sh

    segs = {
        "train(18-23)": seg("2018-01-01", "2023-12-31"),
        "valid(24-25)": seg("2024-01-01", "2025-12-31"),
        "live(26)": seg("2026-01-01", "2026-12-31"),
    }
    return {"total": total, "sharpe": sharpe, "mdd": mdd, "win": win,
            "n_trades": n_trades, "segs": segs, "nav": nav}


def market_bench(s):
    piv = s.pivot_table(index="날짜", columns="종목코드", values="종가")
    idx = piv.pct_change().mean(axis=1).add(1).cumprod()
    nav = idx.to_frame("nav")
    return summarize(nav, [], 0, s)


def fmt(name, res):
    segs = res["segs"]
    line = (f"{name:22s} 누적 {res['total']*100:+7.1f}%  Sharpe {res['sharpe']:5.2f}  "
            f"MDD {res['mdd']*100:6.1f}%  승률 {res['win']*100:4.1f}%  거래 {res['n_trades']:4d}")
    print(line)
    parts = []
    for k, v in segs.items():
        if v:
            parts.append(f"{k} {v[0]*100:+6.1f}%/Sh{v[1]:4.2f}")
    print("   " + "  ".join(parts))

