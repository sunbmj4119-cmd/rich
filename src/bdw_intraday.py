"""
불장단타왕 분봉 규칙 엔진 + 다종목 스캐너 — '자리가 오면 사고, 손실은 짧게·수익은 길게'.

  그의 반복 순위 상위 규칙을 그가 말한 숫자 그대로 옮긴다 (최적화 없음 — 바꾸면 시도로 센다).
  종목 고르기  돈이 몰리는 곳만(16위): 최근 하루 거래대금 상위 N · 상대강도(14위): 최근 하루 수익이 기준(BTC·지수)보다 높음
  할 만한 장   상위봉(1시간) 종가가 VWMA100 위(5위) + 50선 > 200선 정배열(9위)
  매수 자리    진입봉(5분)에서 최근 12봉 안에 VWMA100 을 거래량 2배로 돌파(12위) → 다시 선까지 밟고(저가 ≤ 선×1.002) 종가가 선 위(19·23위)
               + 선에서 1% 안 (추격 금지 23위) → 다음 봉 시가 매수
  손실 짧게    -1% 손절(짧은 손절 · 손익비) · 종가가 선 아래면 근거 이탈 → 다음 봉 시가(6위) · 6봉 지나도 손실이면 시간 손절(6위)
  수익 길게    +2% 에서 절반 익절 + 손절을 본전으로(2위 분할 익절·본절) · 나머지는 선 이탈 또는 +6% (손익비 1:3 이상)
  비중·한도    한 번에 계좌의 0.5% 만 잃게(비중 = 0.5% ÷ 손절 폭) · 동시에 2종목 · 하루 -2.5%·한 달 -7.5% 면 새 진입 멈춤(25위)
               같은 종목 하루 2번까지 (24위 '세 번째는 놓아준다')
  시간         주식: 장 시작 뒤 2시간 안에만 진입, 장 마감 전에 모두 정리(단타로 끝냄) · 코인: 24시간, 최대 보유 24시간

  비용(한쪽): venue 별 — upbit 0.05%+미끄러짐 0.02% · binance 0.10%+0.02% · us(토스 해외주식) 0.10%+0.02% · kr 0.015%+0.05% (+매도 거래세 0.20%)

  입력: {종목: DataFrame(ts, open, high, low, close, volume)} 같은 봉 단위(진입봉). 상위봉은 안에서 다시 묶는다.
"""
import math

import numpy as np
import pandas as pd

VENUES = {
    "upbit": {"fee": 0.0005, "slip": 0.0002, "tax": 0.0, "session": None},
    "binance": {"fee": 0.0010, "slip": 0.0002, "tax": 0.0, "session": None},
    "us": {"fee": 0.0010, "slip": 0.0002, "tax": 0.0, "session": ("09:30", "16:00")},
    "kr": {"fee": 0.00015, "slip": 0.0005, "tax": 0.0020, "session": ("09:00", "15:30")},
}
RULES = {"top_n": 20, "line": 100, "hi_rule": "1h", "brk_lookback": 12, "vol_mult": 2.0, "touch": 0.002, "chase": 0.01,
         "stop": 0.01, "tp1": 0.02, "tp1_frac": 0.5, "tp2": 0.06, "time_bars": 6, "risk": 0.005, "max_pos": 2,
         "day_limit": 0.025, "month_limit": 0.075, "per_day_symbol": 2, "entry_window_min": 120, "crypto_max_hold_h": 24}


def vwma(close, vol, n):
    cv = (close * vol).rolling(n, min_periods=int(n * 0.8)).sum()
    vv = vol.rolling(n, min_periods=int(n * 0.8)).sum()
    return cv / vv.replace(0, np.nan)


def higher(df, rule):
    """상위봉 (진입봉을 묶음) — 그 봉이 '끝난 뒤'에만 쓰도록 한 칸 늦춘다 (미래 정보 없음)."""
    g = df.set_index("ts").resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    g["vw"] = vwma(g["close"], g["volume"], RULES["line"])
    g["ma50"] = g["close"].rolling(50, min_periods=40).mean()
    g["ma200"] = g["close"].rolling(200, min_periods=160).mean()
    g["trend_ok"] = (g["close"] > g["vw"]) & (g["ma50"] > g["ma200"])
    return g[["trend_ok"]].shift(1)                                   # 막 끝난 상위봉까지만


def prepare(df, venue, rules=None):
    """종목 하나: 신호·선·상위봉 추세·세션 정보를 붙인다."""
    R = {**RULES, **(rules or {})}
    d = df.sort_values("ts").reset_index(drop=True).copy()
    d["vw"] = vwma(d["close"], d["volume"], R["line"])
    v20 = d["volume"].rolling(20, min_periods=15).mean().shift(1)
    cross = (d["close"] > d["vw"]) & (d["close"].shift(1) <= d["vw"].shift(1)) & (d["volume"] >= R["vol_mult"] * v20)
    recent = cross.rolling(R["brk_lookback"], min_periods=1).max().shift(1).fillna(0) > 0
    retest = (d["low"] <= d["vw"] * (1 + R["touch"])) & (d["close"] > d["vw"]) & (d["close"] <= d["vw"] * (1 + R["chase"]))
    hi = higher(d, R["hi_rule"])
    d = pd.merge_asof(d, hi.reset_index().rename(columns={"index": "ts"}), on="ts", direction="backward")
    d["setup"] = recent & retest & ~cross & d["trend_ok"].fillna(False).astype(bool)
    d["value"] = d["close"] * d["volume"]
    ses = VENUES[venue]["session"]
    if ses:
        t = d["ts"].dt.strftime("%H:%M")
        d["day"] = d["ts"].dt.date
        start = pd.to_datetime(d["ts"].dt.strftime("%Y-%m-%d ") + ses[0])
        d["mins_open"] = (d["ts"] - start).dt.total_seconds() / 60
        d["in_session"] = (t >= ses[0]) & (t < ses[1])
        d["last_bar"] = d.groupby("day")["ts"].transform("max") == d["ts"]
        d["setup"] &= d["in_session"] & (d["mins_open"] <= R["entry_window_min"])
    else:
        d["day"] = d["ts"].dt.date
        d["last_bar"] = False
    return d


def arrays(d):
    """한 종목의 배열 (거래마다 다시 만들지 않게)."""
    return {"o": d["open"].to_numpy(float), "h": d["high"].to_numpy(float), "lo": d["low"].to_numpy(float),
            "c": d["close"].to_numpy(float), "vw": d["vw"].to_numpy(float), "last": d["last_bar"].to_numpy(bool),
            "day": d["day"].to_numpy(), "ts": d["ts"].to_numpy()}


def simulate_trade(A, i, venue, R):
    """i 봉에서 신호 → i+1 봉 시가 진입. 반환: (청산 봉, 순수익률(들어간 돈 기준), 진입가, 이유).
    한 봉 안에서 손절과 익절이 둘 다 닿으면 손절이 먼저라고 본다 (보수적)."""
    V = VENUES[venue]
    o, h, lo, c, vw, last, day, ts = A["o"], A["h"], A["lo"], A["c"], A["vw"], A["last"], A["day"], A["ts"]
    n = len(o)
    e = i + 1
    if e >= n or not np.isfinite(o[e]) or o[e] <= 0:
        return None
    entry = o[e] * (1 + V["slip"])
    cost_in = entry * (1 + V["fee"])
    stop = entry * (1 - R["stop"])
    tp1, tp2 = entry * (1 + R["tp1"]), entry * (1 + R["tp2"])
    left, got, half = 1.0, 0.0, False
    sell = 1 - (V["fee"] + V["slip"] + V["tax"])
    max_k = n - 1
    if not V["session"]:
        t_end = ts[e] + np.timedelta64(int(R["crypto_max_hold_h"] * 60), "m")
        max_k = min(max(int(np.searchsorted(ts, t_end)) - 1, e), n - 1)
    for k in range(e, n):
        if V["session"] and day[k] != day[e]:
            got += left * o[k] * sell
            return k, got / cost_in - 1, entry, "다음 날 시가 (장 마감 정리 실패)"
        if lo[k] <= stop:
            px = min(stop, o[k]) if k > e else stop
            got += left * px * sell
            return k, got / cost_in - 1, entry, "본전 손절" if half else "손절"
        if not half and h[k] >= tp1:
            got += R["tp1_frac"] * tp1 * sell
            left -= R["tp1_frac"]
            half = True
            stop = max(stop, entry)                                    # 본절로 끌어올림
        if half and h[k] >= tp2:
            got += left * tp2 * sell
            return k, got / cost_in - 1, entry, "목표 +6%"
        reason = None
        if c[k] < vw[k]:
            reason = "선 이탈"
        elif k - e + 1 >= R["time_bars"] and c[k] < entry and not half:
            reason = "시간 손절"
        elif V["session"] and last[k]:
            got += left * c[k] * sell
            return k, got / cost_in - 1, entry, "장 마감 정리"
        elif k >= max_k:
            reason = "최대 보유"
        if reason and k + 1 < n:
            got += left * o[k + 1] * sell
            return k + 1, got / cost_in - 1, entry, reason
    got += left * c[n - 1] * sell
    return n - 1, got / cost_in - 1, entry, "자료 끝"


def candidates(data, venue, rules=None, bench=None):
    """종목별 신호 → 거래 후보 (진입 시각, 청산 시각, 순수익률, 이유, 종목, 상대강도·거래대금 순위용 값)."""
    R = {**RULES, **(rules or {})}
    rows = []
    preps = {}
    for sym, df in data.items():
        d = prepare(df, venue, R)
        preps[sym] = d
    # 하루 거래대금 (직전 하루) · 하루 수익 — 고르는 데 쓰는 값
    bench_ret = None
    if bench is not None and bench in preps:
        b = preps[bench].set_index("ts")["close"]
        bench_ret = b.pct_change(periods=_bars_per_day(preps[bench]))
    for sym, d in preps.items():
        if sym == bench:
            continue
        bpd = _bars_per_day(d)
        val_day = d["value"].rolling(bpd, min_periods=max(1, bpd // 2)).sum()
        ret_day = d["close"].pct_change(periods=bpd)
        A = arrays(d)
        for i in np.where(d["setup"].fillna(False).to_numpy(bool))[0]:
            if i + 1 >= len(d):
                continue
            rs_ok = True
            if bench_ret is not None:
                br = bench_ret.asof(d["ts"].iloc[i])
                rs_ok = bool(np.isfinite(ret_day.iloc[i]) and (not np.isfinite(br) or ret_day.iloc[i] > br))
            if not rs_ok:
                continue
            t = simulate_trade(A, i, venue, R)
            if t is None:
                continue
            k, net, entry, why = t
            rows.append({"sym": sym, "t_sig": d["ts"].iloc[i], "t_in": d["ts"].iloc[i + 1], "t_out": d["ts"].iloc[k],
                         "net": net, "why": why, "value_day": float(val_day.iloc[i]) if np.isfinite(val_day.iloc[i]) else 0.0,
                         "ret_day": float(ret_day.iloc[i]) if np.isfinite(ret_day.iloc[i]) else 0.0})
    return pd.DataFrame(rows).sort_values("t_in").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["sym", "t_sig", "t_in", "t_out", "net", "why", "value_day", "ret_day"])


def _bars_per_day(d):
    if len(d) < 3:
        return 1
    step = d["ts"].diff().dropna().median()
    if d["last_bar"].any():                                            # 세션 시장: 하루 봉 수
        return max(1, int(d.groupby("day").size().median()))
    return max(1, int(pd.Timedelta(days=1) / step))


def portfolio(cands, rules=None, equity0=1.0):
    """후보 거래를 시간 순으로: 거래대금 상위 N 안 · 동시 2종목 · 같은 종목 하루 2번 · 하루·한 달 손실 한도.
    비중 = 위험 0.5% ÷ 손절 폭 1% = 계좌의 50% (동시 2종목이면 최대 100%). 반환: 받아들인 거래·자산곡선(청산 시점)."""
    R = {**RULES, **(rules or {})}
    if cands.empty:
        return cands.assign(w=[]), pd.Series(dtype=float)
    w = min(1.0 / R["max_pos"], R["risk"] / R["stop"])
    eq = equity0
    open_ = []                                                         # (t_out, sym, w·eq, net)
    taken, curve = [], []
    day_pnl, month_pnl, day_cnt = {}, {}, {}
    # 같은 시각 후보가 여럿이면 하루 거래대금 큰 것부터 (돈이 몰리는 곳)
    cands = cands.assign(_d=cands["t_in"].dt.date).sort_values(["t_in", "value_day"], ascending=[True, False])
    rank_ok = _top_n_filter(cands, R["top_n"])
    for idx, r in cands[rank_ok].iterrows():
        # 끝난 거래 정산
        still = []
        for (to, sym, amt, net, d0) in open_:
            if to <= r["t_in"]:
                pnl = amt * net
                eq += pnl
                day_pnl[d0] = day_pnl.get(d0, 0) + pnl
                m0 = (d0.year, d0.month)
                month_pnl[m0] = month_pnl.get(m0, 0) + pnl
                curve.append((to, eq))
            else:
                still.append((to, sym, amt, net, d0))
        open_ = still
        d0 = r["t_in"].date()
        m0 = (d0.year, d0.month)
        if len(open_) >= R["max_pos"] or any(s == r["sym"] for _, s, _, _, _ in open_):
            continue
        if day_cnt.get((d0, r["sym"]), 0) >= R["per_day_symbol"]:
            continue
        if day_pnl.get(d0, 0) <= -R["day_limit"] * eq or month_pnl.get(m0, 0) <= -R["month_limit"] * eq:
            continue
        amt = w * eq
        open_.append((r["t_out"], r["sym"], amt, r["net"], d0))
        day_cnt[(d0, r["sym"])] = day_cnt.get((d0, r["sym"]), 0) + 1
        taken.append({**r.to_dict(), "w": w})
    for (to, sym, amt, net, d0) in sorted(open_):
        eq += amt * net
        curve.append((to, eq))
    tk = pd.DataFrame(taken)
    return tk, pd.Series(dict(curve)).sort_index() if curve else pd.Series(dtype=float)


def _top_n_filter(c, n):
    """신호 시각 기준 직전 하루 거래대금이 그 날 신호들 중 상위 n 안 (같은 날 비교)."""
    if c.empty:
        return pd.Series(dtype=bool)
    rk = c.groupby("_d")["value_day"].rank(ascending=False, method="first")
    return rk <= n


def summary(tk, curve, days=None):
    if tk is None or len(tk) == 0 or curve.empty:
        return {"trades": 0}
    daily = curve.resample("D").last().ffill()
    r = daily.pct_change().dropna()
    yrs = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    by_day = tk.assign(d=pd.to_datetime(tk["t_in"]).dt.date).groupby("d")["net"].mean()
    t = by_day.mean() / by_day.std() * math.sqrt(len(by_day)) if len(by_day) > 2 and by_day.std() > 0 else None
    wins, losses = tk["net"][tk["net"] > 0], tk["net"][tk["net"] <= 0]
    return {"trades": int(len(tk)), "per_month": round(len(tk) / max(yrs * 12, 1e-9), 1), "win": round(float((tk["net"] > 0).mean()), 3),
            "avg_win": round(float(wins.mean()), 4) if len(wins) else 0, "avg_loss": round(float(losses.mean()), 4) if len(losses) else 0,
            "payoff": round(float(wins.mean() / -losses.mean()), 2) if len(wins) and len(losses) and losses.mean() < 0 else None,
            "mean": round(float(tk["net"].mean()), 5), "t_day": round(float(t), 2) if t is not None else None,
            "total": round(float(curve.iloc[-1] - 1), 4), "cagr": round(float(curve.iloc[-1] ** (1 / yrs) - 1), 4) if curve.iloc[-1] > 0 else -1,
            "mdd": round(float((curve / curve.cummax() - 1).min()), 4),
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(365 if days is None else days)), 2) if len(r) > 5 and r.std() > 0 else None,
            "why": tk["why"].value_counts().to_dict()}
