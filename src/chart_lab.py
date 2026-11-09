"""
차트 자리 실험실 — '확실한 자리 + 정해 둔 손절선·익절선'이 승률과 비용을 따졌을 때 이득인지, 차트 분석에서 흔히 쓰는 자리 20개로 검증한다.
종목은 그때의 시총 상위 100 (상장폐지 포함 — 살아남은 종목만 보는 착시 없음). 일봉의 고가·저가로 손절·익절 중 먼저 닿은 쪽을 판정.

사전 등록 (2026-11-09 · 결과를 보기 전에 정함 — 자리·손절·익절·보유·합격선 고정, 칸 수는 시도 장부에 센다)
  자리 20개 (t일 장 마감 정보로 판단 → t+1일 시가에 산다)            · 손절 −3 · −5 · −7 · −10%
                                                                   · 익절 +3 · +5 · +10 · +15 · +20%
                                                                   · 최대 보유 5 · 10 · 20 거래일 (그때까지 안 닿으면 그날 종가)
  판정  날마다 저가가 손절선 이하면 손절, 고가가 익절선 이상이면 익절 · 같은 날 둘 다면 손절 (보수적) ·
        시가가 손절선 아래로 열리면(갭 하락) 그 시가에 손절 · 시가가 익절선 위로 열려도 익절선 값으로 (보수적)
        같은 종목은 보유 기간 동안 다시 사지 않음 · 기업 행위로 끊긴 구간은 뺀다
  비용  매수 0.0186% + 미끄러짐 0.05% · 매도 같고 + 거래세 0.20% (왕복 약 0.34%) — setup_lab 과 같음
  핵심 비교 '같은 날, 그날 상위 100 의 다른 종목을 아무거나 사서 같은 손절·익절' (무작위 10번) — 손절·익절 폭만으로도 승률은 얼마든지 바뀌므로
        자리의 값어치 = (자리 건당 순수익) − (같은 날 아무 종목 건당 순수익)
        (처음엔 '같은 종목 ±10거래일 아무 날'로 정했으나, 사건 전후가 섞여 반등형은 유리·돌파형은 불리하게 치우쳐 결과를 본 뒤 바꿨다 — 기록)
  합격 (A 2002~2018 · B 2019~ 두 시대 모두): 100건 이상 · 건당 순수익 > 0 이고 달별 t ≥ 2 · 아무 날보다 나음(달별 t ≥ 2) ·
        비용 2배에서도 건당 > 0 → 통과한 칸만 20만 원 '한 번에 한 종목'으로 돌려 본다

  python src/chart_lab.py → data/chart_lab.json
"""
import itertools
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_lab as SL  # noqa: E402

OUT = "data/chart_lab.json"
ERA = SL.ERA
COMM, SLIP, TAX = SL.COMM, SL.SLIP, SL.TAX
STOPS = (0.03, 0.05, 0.07, 0.10)
TARGETS = (0.03, 0.05, 0.10, 0.15, 0.20)
HOLDS = (5, 10, 20)
CRIT = {"min_n": 100, "t_net": 2.0, "t_edge": 2.0, "n_random": 10}
SETUPS = {
    "C01": "정배열(20>60>120) 20일선 눌림 — 저가가 20일선 아래로 닿고 종가는 20일선 위",
    "C02": "정배열(60>120) 60일선 눌림 — 저가가 60일선 닿고 종가는 위",
    "C03": "상승 추세(120일선 위) 이격도 과매도 — 종가가 20일선보다 5% 넘게 아래",
    "C04": "20일 고점 돌파 + 거래량 2배 (박스 상단 돌파)",
    "C05": "52주 신고가 첫 돌파",
    "C06": "볼린저 수축(밴드 폭 120일 하위 20%) 뒤 상단 돌파",
    "C07": "갭상승 2% 이상 + 거래량 2배 + 양봉 (갭 앤 고)",
    "C08": "장대양봉 (몸통 +5% 이상 · 윗꼬리 짧음 · 거래량 2배)",
    "C09": "RSI(14) 30 아래에서 30 위로 복귀",
    "C10": "볼린저 하단 이탈 뒤 다시 안으로 복귀",
    "C11": "스토캐스틱(14,3) 20 아래 골든크로스",
    "C12": "200일선 위에서 3일 연속 하락 뒤 양봉",
    "C13": "망치형 캔들 (아래꼬리 ≥ 몸통 2배 · 20일 저점)",
    "C14": "3일 하락 뒤 상승장악형 (오늘 양봉이 어제 음봉을 감쌈)",
    "C15": "5일선이 20일선 위로 골든크로스 (60일선 위)",
    "C16": "MACD 시그널 상향 돌파 (0선 위)",
    "C17": "MACD 0선 상향 돌파",
    "C18": "일목균형표 구름대 상향 돌파",
    "C19": "거래량 3배 + 양봉 + 20일선 위 (세력 매집 흔적)",
    "C20": "쌍바닥 근사 — 20~60일 전 저점 ±3% 안에서 다시 지지 + 그 사이 반등 8% 이상 + 양봉",
}


def _df(x):
    return pd.DataFrame(x)


def _ema(x, n):
    return _df(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def setup_signals(P):
    """t행 = t일 장 마감 뒤 아는 정보로 '자리'인가 (그때의 상위 100 안에서만)."""
    C = P["Cf"] if "Cf" in P else _df(P["C"]).ffill().to_numpy()
    O, H, Lo, V = P["O"], P["H"], P["L"], P["V"]
    sh = lambda x, k=1: _df(x).shift(k).to_numpy()                      # noqa: E731
    ma = {n: SL.roll(C, n) for n in (5, 20, 60, 120, 200)}
    sd20 = SL.roll(C, 20, "std")
    up, lo = ma[20] + 2 * sd20, ma[20] - 2 * sd20
    bw = 4 * sd20 / ma[20]
    bw_q = _df(bw).rolling(120, min_periods=100).quantile(0.2).to_numpy()
    v20 = _df(V).rolling(20, min_periods=15).mean().shift(1).to_numpy()
    hi20 = _df(H).rolling(20, min_periods=15).max().shift(1).to_numpy()
    hi252 = _df(H).rolling(252, min_periods=200).max().shift(1).to_numpy()
    lo20 = _df(Lo).rolling(20, min_periods=15).min().to_numpy()
    d = _df(C).diff().to_numpy()
    rsi = 100 * SL.roll(np.clip(np.nan_to_num(d), 0, None), 14) / (SL.roll(np.abs(np.nan_to_num(d)), 14) + 1e-12)
    ll14 = _df(Lo).rolling(14, min_periods=12).min().to_numpy()
    hh14 = _df(H).rolling(14, min_periods=12).max().to_numpy()
    k = 100 * (C - ll14) / (hh14 - ll14 + 1e-12)
    dk = SL.roll(k, 3, mp=3)
    macd = _ema(C, 12) - _ema(C, 26)
    sig = _df(macd).ewm(span=9, adjust=False, min_periods=9).mean().to_numpy()
    mid = lambda n: (_df(H).rolling(n, min_periods=n).max().to_numpy() + _df(Lo).rolling(n, min_periods=n).min().to_numpy()) / 2   # noqa: E731
    ten, kij, s52 = mid(9), mid(26), mid(52)
    spanA, spanB = sh((ten + kij) / 2, 26), sh(s52, 26)
    cloud = np.fmax(spanA, spanB)
    body = C - O
    rng = H - Lo
    lw = np.fmin(O, C) - Lo
    uw = H - np.fmax(O, C)
    dn = (C < sh(C)).astype(float)
    dn3 = (dn + sh(dn) + sh(dn, 2)) == 3
    # 쌍바닥: 20~60일 전 구간의 최저가와 오늘 저가가 ±3% · 그 사이 최고가가 그 저점보다 8% 이상
    low_old = _df(Lo).shift(20).rolling(41, min_periods=30).min().to_numpy()
    hi_mid = _df(H).shift(1).rolling(40, min_periods=30).max().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        S = {
            "C01": (ma[20] > ma[60]) & (ma[60] > ma[120]) & (Lo <= ma[20]) & (C >= ma[20]),
            "C02": (ma[60] > ma[120]) & (C > ma[120]) & (Lo <= ma[60]) & (C >= ma[60]),
            "C03": (C > ma[120]) & (C / ma[20] <= 0.95),
            "C04": (C > hi20) & (sh(C) <= sh(hi20)) & (V >= 2 * v20),
            "C05": (C > hi252) & (sh(C) <= sh(hi252)),
            "C06": (C > up) & (sh(C) <= sh(up)) & (sh(bw) <= sh(bw_q)),
            "C07": (O / sh(C) >= 1.02) & (V >= 2 * v20) & (C >= O),
            "C08": (body / O >= 0.05) & (uw <= 0.25 * np.fmax(body, 1e-12)) & (V >= 2 * v20),
            "C09": (rsi > 30) & (sh(rsi) <= 30),
            "C10": (C > lo) & (sh(C) < sh(lo)),
            "C11": (k > dk) & (sh(k) <= sh(dk)) & (sh(k) < 20),
            "C12": (C > ma[200]) & (sh(dn3.astype(float)) == 1) & (C > O),
            "C13": (lw >= 2 * np.abs(body)) & (C >= Lo + 2 / 3 * rng) & (Lo <= lo20) & (rng > 0),
            "C14": (sh(dn3.astype(float)) == 1) & (sh(C) < sh(O)) & (C > O) & (O <= sh(C)) & (C >= sh(O)),
            "C15": (ma[5] > ma[20]) & (sh(ma[5]) <= sh(ma[20])) & (C > ma[60]),
            "C16": (macd > sig) & (sh(macd) <= sh(sig)) & (macd > 0),
            "C17": (macd > 0) & (sh(macd) <= 0),
            "C18": (C > cloud) & (sh(C) <= sh(cloud)),
            "C19": (V >= 3 * v20) & (C > O) & (C > ma[20]),
            "C20": (np.abs(Lo / low_old - 1) <= 0.03) & (hi_mid / low_old >= 1.08) & (C > O) & (Lo > 0),
        }
    U = P["U"]
    ok = np.isfinite(P["O"]) & (P["O"] > 0)
    return {k: np.nan_to_num(v, nan=False).astype(bool) & U & ok for k, v in S.items()}


# ── 길 (진입 다음 날부터 20거래일, 진입 시가 대비) ─────────────────────────────

class Paths:
    def __init__(self, P, ts, js):
        H = max(HOLDS)
        O, Hh, Lo = P["O"], P["H"], P["L"]
        C = P["Cf"] if "Cf" in P else _df(P["C"]).ffill().to_numpy()
        T = O.shape[0]
        ts, js = np.asarray(ts, int), np.asarray(js, int)
        e = ts + 1
        d = e[:, None] + np.arange(H)[None, :]
        dc = np.minimum(d, T - 1)
        po = O[e, js][:, None]
        self.n = len(ts)
        self.o = O[dc, js[:, None]] / po
        self.h = np.where(np.isfinite(Hh[dc, js[:, None]]), Hh[dc, js[:, None]] / po, 0.0)
        self.l = np.where(np.isfinite(Lo[dc, js[:, None]]), Lo[dc, js[:, None]] / po, np.inf)
        self.c = _df(C[dc, js[:, None]] / po).ffill(axis=1).fillna(1.0).to_numpy()   # 빈 종가는 그 전 종가
        self.inside = d < T

    def exit(self, s, g, h, cost_mult=1.0, detail=False):
        """반환 순수익 (그리고 detail 이면 청산 종류 0 손절 · 1 익절 · 2 기한, 보유 일수)."""
        n = self.n
        r = np.arange(n)
        L, Hh = self.l[:, :h], self.h[:, :h]
        sm, gm = L <= 1 - s, Hh >= 1 + g
        sd = np.where(sm.any(1), sm.argmax(1), h)
        gd = np.where(gm.any(1), gm.argmax(1), h)
        stop = (sd <= gd) & (sd < h)
        tgt = ~stop & (gd < h)
        px = np.empty(n)
        o_at = self.o[r, np.minimum(sd, h - 1)]
        px[stop] = np.where(sd[stop] == 0, 1 - s, np.fmin(1 - s, np.nan_to_num(o_at[stop], nan=1 - s)))
        px[tgt] = 1 + g
        tim = ~stop & ~tgt
        px[tim] = self.c[tim, h - 1]
        cb = COMM + SLIP * cost_mult
        cs = COMM + SLIP * cost_mult + TAX
        net = px * (1 - cs) / (1 + cb) - 1
        if not detail:
            return net
        kind = np.where(stop, 0, np.where(tgt, 1, 2))
        days = np.where(stop, sd + 1, np.where(tgt, gd + 1, h))
        return net, kind, days


def entries(P, S, a, b, h):
    """신호 → (t, j) 목록 (같은 종목은 h 거래일 동안 다시 사지 않음 · 끊긴 구간 제외)."""
    T, N = S.shape
    dates = P["dates"]
    i0 = int(np.searchsorted(dates, pd.Timestamp(a)))
    i1 = T if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
    brk = P.get("breaks")
    cb = np.cumsum(brk, axis=0) if brk is not None else None
    busy = np.full(N, -1)
    ts, js = [], []
    O = P["O"]
    for t in range(i0, min(i1, T - h - 1)):
        for j in np.flatnonzero(S[t] & (busy < t)):
            if not (np.isfinite(O[t + 1, j]) and O[t + 1, j] > 0):         # 다음 날 시가가 없으면 (거래정지·상장폐지) 못 산다
                continue
            if cb is not None and cb[min(t + 1 + h, T - 1), j] - cb[t, j] > 0:
                continue
            ts.append(t)
            js.append(j)
            busy[j] = t + h
    return np.array(ts, int), np.array(js, int)


def random_entries(P, ts, js, k, seed):
    """같은 날 · 그날 상위 100 안의 다른 종목 (다음 날 시가가 있는) — k번씩. 같은 날 시장 전체의 움직임은 똑같이 받는다."""
    rng = np.random.default_rng(seed)
    T = P["O"].shape[0]
    nxt = np.zeros_like(P["U"])
    nxt[:-1] = np.isfinite(P["O"][1:]) & (P["O"][1:] > 0)
    ok = P["U"] & nxt
    rt, rj, src = [], [], []
    for q, (t, j) in enumerate(zip(ts, js)):
        cand = np.flatnonzero(ok[t])
        cand = cand[cand != j]
        if not len(cand) or t + max(HOLDS) + 1 >= T:
            continue
        rt.extend([t] * k)
        rj.extend(rng.choice(cand, size=k, replace=True))
        src.extend([q] * k)
    return np.array(rt, int), np.array(rj, int), np.array(src, int)


def mstat(x, dates):
    t, m = SL.tstat_month(x, dates)
    return t


def cell_stats(net, kind, days, dates, rnd_mean_by_trade):
    win = net > 0
    aw = float(net[win].mean()) if win.any() else 0.0
    al = float(net[~win].mean()) if (~win).any() else 0.0
    edge = net - rnd_mean_by_trade
    ok = np.isfinite(edge)
    return {"n": int(len(net)), "win": round(float(win.mean()), 3), "avg_win": round(aw, 4), "avg_loss": round(al, 4),
            "be_win": round(-al / (aw - al), 3) if aw > al and al < 0 else None, "mean": round(float(net.mean()), 5),
            "t": mstat(net, dates), "rand_mean": round(float(np.nanmean(rnd_mean_by_trade)), 5),
            "edge": round(float(np.nanmean(edge)), 5), "t_edge": mstat(edge[ok], dates[ok]) if ok.sum() > 10 else None,
            "stop_share": round(float((kind == 0).mean()), 3), "tgt_share": round(float((kind == 1).mean()), 3),
            "days": round(float(days.mean()), 1)}


def run_era(era, log=print):
    a, b = ERA[era]
    P, idx = SL.panel(era)
    dates = P["dates"]
    S = setup_signals(P)
    out = {}
    for name in SETUPS:
        for h in HOLDS:
            ts, js = entries(P, S[name], a, b, h)
            if len(ts) < 20:
                for s, g in itertools.product(STOPS, TARGETS):
                    out[(name, s, g, h)] = {"n": int(len(ts))}
                continue
            PA = Paths(P, ts, js)
            rt, rj, src = random_entries(P, ts, js, CRIT["n_random"], seed=zlib.crc32(f"{name}|{h}|{era}".encode()))
            PR = Paths(P, rt, rj)
            d_in = dates[ts + 1]
            for s, g in itertools.product(STOPS, TARGETS):
                net, kind, days = PA.exit(s, g, h, detail=True)
                rn = PR.exit(s, g, h)
                rm = np.full(len(ts), np.nan)
                if len(src):
                    sums = np.bincount(src, weights=rn, minlength=len(ts))
                    cnt = np.bincount(src, minlength=len(ts))
                    rm = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
                st = cell_stats(net, kind, days, d_in, rm)
                st["mean_x2"] = round(float(PA.exit(s, g, h, cost_mult=2.0).mean()), 5)
                out[(name, s, g, h)] = st
        log(f"  {era} {name} 신호 {int(S[name].sum()):,}개 · {SETUPS[name][:30]}")
    return out, P


def passed(a, b):
    def ok(x):
        return (x.get("n", 0) >= CRIT["min_n"] and x.get("mean", -1) > 0 and (x.get("t") or 0) >= CRIT["t_net"]
                and x.get("edge", -1) > 0 and (x.get("t_edge") or 0) >= CRIT["t_edge"] and x.get("mean_x2", -1) > 0)
    return ok(a) and ok(b)


def one_slot(P, S, s, g, h, a, b, cash0=200_000):
    """20만 원 '한 번에 한 종목': 비어 있으면 그날 신호 중 시총 큰 종목을 다음 날 시가에 전부 → 손절·익절·기한."""
    O, Hh, Lo = P["O"], P["H"], P["L"]
    C = P["Cf"] if "Cf" in P else _df(P["C"]).ffill().to_numpy()
    MC = np.nan_to_num(P["MC"])
    dates = P["dates"]
    T = len(dates)
    i0 = int(np.searchsorted(dates, pd.Timestamp(a)))
    i1 = T if b is None else int(np.searchsorted(dates, pd.Timestamp(b), side="right"))
    cb, cs = COMM + SLIP, COMM + SLIP + TAX
    eq = cash0
    E = []
    n = wins = 0
    t = i0
    while t < min(i1, T - h - 2):
        js = [k for k in np.flatnonzero(S[t]) if np.isfinite(O[t + 1, k]) and O[t + 1, k] > 0]
        if not js:
            E.append((dates[t], eq))
            t += 1
            continue
        j = max(js, key=lambda k: MC[t, k])
        po = O[t + 1, j]
        units = eq / (po * (1 + cb))
        px, x = None, t + h
        for dd in range(h):
            d = t + 1 + dd
            if dd > 0 and np.isfinite(O[d, j]) and O[d, j] <= po * (1 - s):
                px, x = O[d, j], d
                break
            if np.isfinite(Lo[d, j]) and Lo[d, j] <= po * (1 - s):
                px, x = po * (1 - s), d
                break
            if np.isfinite(Hh[d, j]) and Hh[d, j] >= po * (1 + g):
                px, x = po * (1 + g), d
                break
            E.append((dates[d], units * C[d, j]))
        if px is None:
            px = C[t + h, j]
        new = units * px * (1 - cs)
        wins += new > eq
        eq = new
        n += 1
        E.append((dates[x], eq))
        t = x + 1
    s_ = pd.Series(dict(E)).sort_index()
    r = s_.pct_change().dropna()
    if len(r) < 60:
        return None
    yrs = len(r) / 252
    return {"cagr": round(float((s_.iloc[-1] / cash0) ** (1 / yrs) - 1), 4), "mdd": round(float((s_ / s_.cummax() - 1).min()), 4),
            "trades": n, "win": round(wins / max(n, 1), 3), "final": round(float(s_.iloc[-1]))}


def key_str(k):
    name, s, g, h = k
    return f"{name}|손절{int(s * 100)}|익절{int(g * 100)}|{h}일"


def main():
    t0 = time.time()
    res = {}
    panels = {}
    for era in ("A", "B"):
        res[era], panels[era] = run_era(era)
    keys = list(res["A"])
    cells = []
    for k in keys:
        A, B = res["A"][k], res["B"][k]
        cells.append({"key": key_str(k), "setup": k[0], "stop": k[1], "target": k[2], "hold": k[3], "A": A, "B": B, "pass": passed(A, B)})
    good = [c for c in cells if c["pass"]]
    for c in good:
        out = {}
        for era in ("A", "B"):
            P = panels[era]
            S = setup_signals(P)[c["setup"]]
            out[era] = one_slot(P, S, c["stop"], c["target"], c["hold"], *ERA[era])
        c["one_slot"] = out
    # 요약: 자리별 가장 나은 칸 · 승률 높은 칸이 정말 이득인가 · A 와 B 의 일관성
    by_setup = {}
    for name in SETUPS:
        cs_ = [c for c in cells if c["setup"] == name and c["A"].get("n", 0) >= CRIT["min_n"] and c["B"].get("n", 0) >= CRIT["min_n"]]
        if not cs_:
            by_setup[name] = {"desc": SETUPS[name], "note": "거래 부족"}
            continue
        best = max(cs_, key=lambda c: min(c["A"]["edge"], c["B"]["edge"]))
        by_setup[name] = {"desc": SETUPS[name], "signals_A": int(best["A"]["n"]), "signals_B": int(best["B"]["n"]),
                          "best": {k: best[k] for k in ("key", "A", "B", "pass")},
                          "share_edge_pos_both": round(float(np.mean([(c["A"]["edge"] > 0) and (c["B"]["edge"] > 0) for c in cs_])), 3),
                          "share_mean_pos_both": round(float(np.mean([(c["A"]["mean"] > 0) and (c["B"]["mean"] > 0) for c in cs_])), 3)}
    valid = [c for c in cells if c["A"].get("n", 0) >= CRIT["min_n"] and c["B"].get("n", 0) >= CRIT["min_n"]]
    hiwin = sorted(valid, key=lambda c: -(c["A"]["win"] + c["B"]["win"]))[:10]
    ea = np.array([c["A"]["edge"] for c in valid])
    eb = np.array([c["B"]["edge"] for c in valid])
    summary = {"cells": len(cells), "cells_valid": len(valid), "passed": [c["key"] for c in good],
               "share_mean_pos_both": round(float(np.mean([(c["A"]["mean"] > 0) and (c["B"]["mean"] > 0) for c in valid])), 4),
               "share_edge_pos_both": round(float(np.mean((ea > 0) & (eb > 0))), 4),
               "edge_corr_A_B": round(float(pd.Series(ea).rank().corr(pd.Series(eb).rank())), 3) if len(valid) > 10 else None,
               "high_win_cells": [{k: c[k] for k in ("key", "A", "B")} for c in hiwin]}
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "era": ERA, "costs": {"comm": COMM, "slip": SLIP, "tax": TAX},
           "grid": {"stops": STOPS, "targets": TARGETS, "holds": HOLDS}, "setups": SETUPS, "criteria": CRIT,
           "n_trials": len(cells), "summary": summary, "by_setup": by_setup, "passed_cells": good,
           "cells": [{"key": c["key"], "A": {k: c["A"].get(k) for k in ("n", "win", "be_win", "mean", "t", "edge", "t_edge", "mean_x2")},
                      "B": {k: c["B"].get(k) for k in ("n", "win", "be_win", "mean", "t", "edge", "t_edge", "mean_x2")}, "pass": c["pass"]} for c in cells],
           "secs": round(time.time() - t0)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"■ {len(cells)}칸 (거래 100건 넘는 칸 {len(valid)}) · 통과 {len(good)} · 두 시대 모두 건당 플러스 {summary['share_mean_pos_both']:.1%} · "
          f"두 시대 모두 아무 날보다 나음 {summary['share_edge_pos_both']:.1%} · A·B 자리값 순위 상관 {summary['edge_corr_A_B']}")
    for name, v in by_setup.items():
        if "best" not in v:
            print(f"  {name} {v['desc'][:34]:34s} 거래 부족")
            continue
        bA, bB = v["best"]["A"], v["best"]["B"]
        print(f"  {name} {v['desc'][:34]:34s} 최선 {v['best']['key'].split('|', 1)[1]:18s} "
              f"A 승률 {bA['win']:.0%}(본전 {bA['be_win']}) 건당 {bA['mean']:+.2%} 자리값 {bA['edge']:+.2%}(t {bA['t_edge']}) | "
              f"B 승률 {bB['win']:.0%} 건당 {bB['mean']:+.2%} 자리값 {bB['edge']:+.2%}(t {bB['t_edge']}) {'✓' if v['best']['pass'] else '✗'}")
    print(f"끝 ({out['secs']}초) → {OUT}")


if __name__ == "__main__":
    main()
