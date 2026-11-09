"""
분봉 규칙 격자 탐색 — 손절선·익절선·나눠 파는 비율·매수 자리를 바꿔 가며 '수익이 가장 큰 곳'을 찾되, 운으로 뽑힌 1등은 걸러 낸다.
자료는 서버에만 있다 (토스 1분봉 → 정규장 5분봉, intraday_lab 과 같은 자료). 요약만 data/intraday_sweep.json 으로 올린다.

사전 등록 (2026-11-09 · 격자 결과를 보기 전에 정함)
  격자   진입 손잡이 ENTRY × 청산 손잡이 EXIT (아래) — 칸 수를 그대로 시도 장부(trials.py)에 센다
  엔진   bdw_intraday 와 같은 규칙: 다음 봉 시가 진입 · 한 봉 안에서 손절이 익절보다 먼저 · 1차 목표에서 일부 팔고 손절을 본전으로 ·
         장 마지막 봉에서 모두 정리 · 위험 0.5% 비중(= 0.5% ÷ 손절 폭, 한 종목 최대 50%) · 동시 2종목 · 같은 종목 하루 2번 ·
         하루 -2.5%·한 달 -7.5% 면 새 진입 멈춤 · 상대강도(하루 수익 > 기준) · 거래대금 상위 N 은 '그 시각까지의' 하루 거래대금 순위 (미래 정보 없음)
  고르기 한 시대(반년)에서 총수익 1등 칸이 아니라, 그 칸과 '손잡이 하나만 한 칸 옮긴' 이웃 칸들의 총수익 중앙값이 가장 큰 칸
         (뾰족한 봉우리보다 넓은 언덕 — 숫자를 조금 바꿔도 버티는 설정) · 그 시대 거래 100번 이상인 칸만
  확인   고른 칸을 다른 시대에 숫자 그대로 — A 로 골라 B 에서, B 로 골라 A 에서 (두 방향)
  합격   확인 시대에서: 거래 100번 이상 · 건당 순수익 > 0 이고 날짜별 t ≥ 2 · 건당 순수익이 '같은 청산·같은 걸러내기로 진입 시각만 무작위' 30번의 95% 위 ·
         비용 2배에서도 건당 > 0 · 최대낙폭이 기준 보유보다 작음 → 두 방향 모두 합격해야 '쓸 만한 설정'
  참고   손잡이별 중앙값(대체로 어느 손절·익절 폭이 나은가) · A 성적과 B 성적의 순위 상관(좋은 설정이 계속 좋은가) ·
         그냥 1등(봉우리)을 골랐다면 확인 시대에서 어땠나

  2차 (--round 2, 기본): 1차에서 덜 잃는 곳이 격자 끝이었으므로 그 바깥 — 넉넉한 매수 자리 · 손절 −2~−10% · 익절 +3~+20% ·
       최대 5거래일 보유(밤사이 갭 반영) · 1시간봉 추세 이탈 청산 · 손절 폭과 상관없이 같은 돈(종목당 20%, 최대 5종목). 고르기·확인·합격은 같음.

  python src/intraday_sweep.py --venue kr --dir ~/.rich/candles/1m/kr --bench 069500 [--round 1|2]
  python src/intraday_sweep.py --venue us --dir ~/.rich/candles/1m/us --bench SPY
"""
import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdw_intraday as BI  # noqa: E402
import intraday_lab as IL  # noqa: E402

OUT = "data/intraday_sweep.json"
ENTRY = {"touch": [0.001, 0.002, 0.005],        # 되밟기: 저가가 선 위 몇 % 안까지 내려왔나
         "chase": [0.005, 0.01, 0.02],          # 추격 금지: 종가가 선 위 몇 % 안일 때만 산다
         "vol_mult": [1.5, 2.0, 3.0],           # 돌파 봉 거래량이 평소(20봉 평균)의 몇 배
         "window": [60, 120, 300],              # 장 시작 뒤 몇 분 안에만 진입
         "top_n": [20, 0]}                      # 하루 거래대금 상위 몇 종목만 (0 = 전부)
EXIT = {"stop": [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03],   # 손절선
        "tp1": [0.01, 0.02, 0.03, 0.05],                   # 1차 익절선
        "frac": [0.33, 0.5, 1.0],                          # 1차에서 파는 비율 (1 = 전부 → 2차 없음)
        "tp2": [0.04, 0.06, 0.10],                         # 2차 익절선 (1차보다 커야 함)
        "time_bars": [6, 12, 0],                           # 몇 봉(5분) 지나도 매입가 아래면 정리 (0 = 안 함)
        "line_exit": [1, 0]}                               # 5분봉 종가가 선(VWMA100) 아래면 정리 (1 = 함)
DEFAULT = {"touch": 0.002, "chase": 0.01, "vol_mult": 2.0, "window": 120, "top_n": 20,
           "stop": 0.01, "tp1": 0.02, "frac": 0.5, "tp2": 0.06, "time_bars": 6, "line_exit": 1, "max_days": 0}

# 2차 (2026-11-09 사전 등록, 1차 결과를 본 뒤): 1차에서 덜 잃는 곳이 격자 끝(손절 −3%·익절 +5%)이었으므로 그 바깥으로 —
# 매수 자리 넉넉하게(선에서 더 멀리까지) · 손절·익절 넓게 · 며칠 들고 가기(스윙) · 큰 추세가 꺾이면 정리.
# 손절 폭과 상관없이 같은 돈을 넣는다(종목당 20%·최대 5종목): 1차는 손절이 넓을수록 비중이 작아져 '덜 잃은' 것처럼 보였다.
ENTRY2 = {"touch": [0.002, 0.005, 0.01],         # 되밟기 (저가가 선 위 몇 % 안)
          "chase": [0.01, 0.02, 0.04],           # 추격 금지 (종가가 선 위 몇 % 안)
          "vol_mult": [1.5, 2.0, 3.0],
          "window": [120, 390],                  # 390 = 장중 아무 때나
          "top_n": [20, 0]}
EXIT2 = {"stop": [0.02, 0.03, 0.05, 0.07, 0.10],
         "tp1": [0.03, 0.05, 0.08, 0.12],
         "frac": [0.33, 0.5, 1.0],
         "tp2": [0.08, 0.12, 0.20],
         "time_bars": [0],
         "line_exit": [0, 1, 2],                 # 0 없음 · 1 5분봉 종가 < VWMA100 · 2 1시간봉 추세 이탈(종가 < 1시간 VWMA100 또는 50선 < 200선)
         "max_days": [0, 1, 3, 5]}               # 최대 보유: 0 = 그날 장 마감 · n = n 거래일 뒤 장 마감
ROUNDS = {1: {"entry": ENTRY, "exit": {**EXIT, "max_days": [0]}, "sizing": "risk", "max_pos": 2},
          2: {"entry": ENTRY2, "exit": EXIT2, "sizing": "equal", "max_pos": 5}}
CRIT = {"min_trades": 100, "t_day": 2.0, "random_pct": 0.95, "n_random": 30}
CFG = {"sizing": "risk", "max_pos": 2}           # 비중: risk = 위험 0.5% ÷ 손절 폭(최대 1/max_pos) · equal = 1/max_pos
WHY = ["손절", "본전 손절", "1차 목표(전부)", "2차 목표", "선 이탈", "시간 손절", "장 마감 정리", "선 이탈(1차 뒤)",
       "추세 이탈", "추세 이탈(1차 뒤)", "보유 기한 정리"]
XKEYS = ("stop", "tp1", "frac", "tp2", "time_bars", "line_exit", "max_days")
G = {}                                          # 작업자들이 나눠 쓰는 큰 배열 (fork 로 물려받음)


def grid(entry=None, exit_=None):
    E, X = entry or ENTRY, dict(exit_ or EXIT)
    X.setdefault("time_bars", [0])
    X.setdefault("max_days", [0])
    X = {k: X[k] for k in XKEYS}                # 청산 손잡이 순서를 고정
    return E, X, list(itertools.product(*E.values())), list(itertools.product(*X.values()))


def exit_key(p):
    """같은 규칙이면 같은 열쇠 (1차에서 전부 팔면 2차는 뜻이 없음) · 2차 ≤ 1차면 None (없는 칸)."""
    s, a, f, b, tb, ln, md = p
    if f >= 1:
        return (s, a, f, tb, ln, md)
    return p if b > a else None


def n_unique(entry=None, exit_=None):
    """실제로 다른 규칙인 칸 수 — 시도 장부에 이 수를 센다."""
    E, X, el, xl = grid(entry, exit_)
    return len(el) * len({k for k in map(exit_key, xl) if k is not None})


# ── 자료 ─────────────────────────────────────────────────────────────────────

def load(venue, path, cache=None):
    """1분봉 → 정규장 5분봉 (intraday_lab.load_dir) · 한 번 묶은 것은 서버 ~/.rich/cache 에 두고 다시 씀."""
    files = sorted(glob.glob(os.path.join(os.path.expanduser(path), "*.csv*")))
    sig = [(os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f))) for f in files]
    cp = os.path.join(os.path.expanduser(cache), f"sweep5m_{venue}.pkl") if cache else None
    if cp and os.path.exists(cp):
        try:
            with open(cp, "rb") as fh:
                c = pickle.load(fh)
            if c.get("sig") == sig:
                return c["data"]
        except (OSError, ValueError, pickle.UnpicklingError, EOFError):
            pass
    data = IL.load_dir(path, venue)
    if cp and data:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp + ".tmp", "wb") as fh:
            pickle.dump({"sig": sig, "data": data}, fh)
        os.replace(cp + ".tmp", cp)
    return data


def build(data, venue, bench=None, entry=None):
    """모든 종목을 이어 붙인 배열 하나: 신호 재료(선·거래량·상위봉 추세·세션·상대강도·거래대금 순위)를 한 번만 계산."""
    E = entry or ENTRY
    session = bool(BI.VENUES[venue]["session"])
    br = None
    if bench in data:
        b = BI.prepare(data[bench], venue)
        br = pd.DataFrame({"ts": b["ts"].to_numpy(), "br": b["close"].pct_change(BI._bars_per_day(b)).to_numpy()})
    syms = [s for s in data if s != bench]
    parts = []
    for si, s in enumerate(syms):
        d = BI.prepare(data[s], venue)
        bpd = BI._bars_per_day(d)
        rd = d["close"].pct_change(bpd).to_numpy()
        if br is not None:
            bb = pd.merge_asof(d[["ts"]], br, on="ts", direction="backward")["br"].to_numpy()
            rs = np.isfinite(rd) & (~np.isfinite(bb) | (rd > bb))
        else:
            rs = np.ones(len(d), bool)
        nxt_ok = (d["day"].shift(-1) == d["day"]).to_numpy() if session else (np.arange(len(d)) < len(d) - 1)
        base = d["trend_ok"].fillna(False).astype(bool).to_numpy() & rs & nxt_ok
        if session:
            base &= d["in_session"].to_numpy(bool)
        parts.append(pd.DataFrame({
            "ts": d["ts"].to_numpy(), "o": d["open"].to_numpy(float), "h": d["high"].to_numpy(float), "l": d["low"].to_numpy(float),
            "c": d["close"].to_numpy(float), "vw": d["vw"].to_numpy(float), "v": d["volume"].to_numpy(float),
            "v20": d["volume"].rolling(20, min_periods=15).mean().shift(1).to_numpy(),
            "xup": ((d["close"] > d["vw"]) & (d["close"].shift(1) <= d["vw"].shift(1))).to_numpy(),
            "val_day": d["value"].rolling(bpd, min_periods=max(1, bpd // 2)).sum().to_numpy(),
            "base": base, "mins": d["mins_open"].to_numpy(float) if session else np.zeros(len(d)), "sym": si,
            "trend": d["trend_ok"].fillna(False).astype(bool).to_numpy()}))
    X = pd.concat(parts, ignore_index=True)
    piv = X.pivot_table(index="ts", columns="sym", values="val_day", aggfunc="last").sort_index().ffill()
    rk = piv.rank(axis=1, ascending=False, method="first").stack().rename("rank").reset_index()
    X = X.merge(rk, on=["ts", "sym"], how="left")
    ts = X["ts"].to_numpy("datetime64[ns]").astype(np.int64)
    sym = X["sym"].to_numpy(np.int32)
    day = (ts // 86_400_000_000_000).astype(np.int32)
    dt = pd.DatetimeIndex(X["ts"])
    g = {"venue": venue, "session": session, "syms": syms, "ts": ts, "sym": sym, "day": day,
         "mon": (dt.year * 12 + dt.month).to_numpy(np.int32),
         "rank": X["rank"].fillna(1e9).to_numpy(float), "base": X["base"].to_numpy(bool), "mins": X["mins"].to_numpy(float),
         "trend": X["trend"].to_numpy(bool)}
    newday = np.r_[True, (np.diff(day) != 0) | (np.diff(sym) != 0)]
    g["dayseq"] = (np.cumsum(newday) - 1).astype(np.int32)                       # 거래일 번호 (종목마다 이어서 — 주말·휴일 건너뜀)
    for k in ("o", "h", "l", "c", "vw", "val_day"):
        g[k] = X[k].to_numpy(float)
    if session:
        g["H"] = int(X.groupby(["sym", day]).size().max())
    else:
        step = max(1, int(np.median(np.diff(ts[sym == 0])) // 60_000_000_000)) if (sym == 0).sum() > 2 else 5
        g["H"] = int(BI.RULES["crypto_max_hold_h"] * 60 // step)
    v, v20, xup = X["v"].to_numpy(float), X["v20"].to_numpy(float), X["xup"].to_numpy(bool)
    start = np.r_[0, np.flatnonzero(np.diff(sym)) + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(sym)]))
    L = BI.RULES["brk_lookback"]
    g["cross"], g["recent"] = {}, {}
    for vm in E["vol_mult"]:
        cross = xup & (v >= vm * v20)
        cs = np.r_[0, np.cumsum(cross)]
        i = np.arange(len(sym))
        g["cross"][vm] = cross
        g["recent"][vm] = (cs[i] - cs[np.maximum(i - L, first)]) > 0          # 직전 L 봉 안에 돌파 (같은 종목 안에서만)
    G.clear()
    G.update(g)
    return g


def signals(touch, chase, vol_mult, window, top_n):
    """매수 자리 (신호 봉 번호) — 진입 시각(다음 봉) 순, 같은 시각이면 하루 거래대금 큰 것부터."""
    g = G
    vw = g["vw"]
    ok = g["recent"][vol_mult] & ~g["cross"][vol_mult] & g["base"] & (g["mins"] <= window)
    ok &= (g["l"] <= vw * (1 + touch)) & (g["c"] > vw) & (g["c"] <= vw * (1 + chase))
    if top_n:
        ok &= g["rank"] <= top_n
    idx = np.flatnonzero(ok)
    idx = idx[idx + 1 < len(vw)]
    return idx[np.lexsort((-g["val_day"][idx], g["ts"][idx + 1]))]


def eligible(window, top_n, t0, t1):
    """무작위 진입용 자리: 신호와 같은 걸러내기(상위봉 추세·세션 시간대·상대강도·거래대금 순위)만 — 진입 시각만 다르다."""
    g = G
    ok = g["base"] & (g["mins"] <= window)
    if top_n:
        ok &= g["rank"] <= top_n
    idx = np.flatnonzero(ok)
    idx = idx[idx + 1 < len(ok)]
    tin = g["ts"][idx + 1]
    return idx[(tin >= t0) & (tin < t1)]


# ── 청산 (모든 신호를 한꺼번에) ───────────────────────────────────────────────

def _first(mask, inf):
    i = mask.argmax(1)
    return np.where(mask[np.arange(len(mask)), i], i, inf)


def _next(mask, inf):
    """[행, j] = j 봉부터 처음 조건이 맞는 봉 (없으면 inf) · 맨 뒤에 inf 한 칸 더."""
    H = mask.shape[1]
    idx = np.where(mask, np.arange(H, dtype=np.int16)[None, :], np.int16(inf))
    r = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    return np.concatenate([r, np.full((len(mask), 1), inf, np.int16)], 1)


class Paths:
    """신호마다 진입 봉부터 (최대 보유 기한의) 마지막 봉까지 가격을 '매입가 대비 비율'로 — 청산 숫자만 바꿔 가며 빠르게 다시 계산.
    bdw_intraday.simulate_trade 와 같은 순서: 한 봉 안에서 손절 → 1차 익절(→ 손절을 본전으로) → 2차 익절 →
    종가 규칙(보유 기한의 마지막 봉 정리 · 선/추세 이탈 · 시간 손절 → 다음 봉 시가). 밤사이 갭: 시가가 손절선 아래면 시가에 팖."""

    def __init__(self, sig, cost_mult=1.0, exit_=None):
        g = G
        X = grid(None, exit_ or EXIT)[1]
        V = BI.VENUES[g["venue"]]
        self.fee, self.slip, self.tax = V["fee"] * cost_mult, V["slip"] * cost_mult, V["tax"]
        dmax = max(X["max_days"])
        H = g["H"] * (dmax + 1)
        self.H, self.inf = H, H + 2
        ft = np.float64 if H <= 120 else np.float32                                  # 여러 날 보유는 배열이 커서 4바이트로
        sig = np.asarray(sig, np.int64)
        self.n = len(sig)
        e = self.e = sig + 1
        J = np.arange(H)
        N = len(g["o"])
        raw = e[:, None] + J[None, :]
        pos = np.minimum(raw, N - 1)
        valid = (raw < N) & (g["sym"][pos] == g["sym"][e][:, None])
        del raw
        if g["session"]:
            off = (g["dayseq"][pos] - g["dayseq"][e][:, None]).astype(np.int16)    # 진입일로부터 몇 거래일째
            valid &= off <= dmax
        valid = np.logical_and.accumulate(valid, axis=1)
        if g["session"]:
            self.lastD = {md: (valid & (off <= md)).sum(1) - 1 for md in X["max_days"]}
            del off
        else:
            lv = valid.sum(1) - 1
            self.lastD = {md: np.minimum(lv, g["H"] * (md + 1) - 1) for md in X["max_days"]}
        E = (g["o"][e] * (1 + self.slip))[:, None]
        self.o = np.where(valid, g["o"][pos] / E, np.nan).astype(ft)
        self.c = np.where(valid, g["c"][pos] / E, np.nan).astype(ft)
        h = np.where(valid, g["h"][pos] / E, 0.0).astype(ft)
        lo = np.where(valid, g["l"][pos] / E, np.inf).astype(ft)
        line = valid & (self.c < (g["vw"][pos] / E))
        trendbrk = valid & ~g["trend"][pos]
        below = valid & (self.c < 1.0)
        del pos
        inf = self.inf
        self.lines = {1: line, 2: trendbrk}
        self.S1 = {s: _first(lo <= 1 - s, inf) for s in X["stop"]}
        self.T1 = {a: _first(h >= 1 + a, inf) for a in X["tp1"]}
        self.C1 = {}
        for tb in X["time_bars"]:
            tm = (below & (J[None, :] + 1 >= tb)) if tb else None
            for ln in X["line_exit"]:
                m = self.lines[ln] if ln else None
                m = tm if m is None else (m if tm is None else (m | tm))
                self.C1[(tb, ln)] = _first(m, inf) if m is not None else np.full(self.n, inf)
        self.NT2 = {b: _next(h >= 1 + b, inf) for b in X["tp2"]}
        self.NBE = _next(lo <= 1.0, inf)
        self.NL = {ln: _next(self.lines[ln], inf) for ln in X["line_exit"] if ln}
        del h, lo, below
        self.lines = {k: v for k, v in self.lines.items() if k in X["line_exit"]}

    def exit(self, s, a, f, b, tb, ln, md=0, why=False):
        """반환: 순수익률(들어간 돈 기준, 비용 뺀 뒤) · 청산 봉 번호(전체 배열) [· 이유 번호]."""
        n, H = self.n, self.H
        r = np.arange(n)
        LD = self.lastD[md]
        s1, t1 = self.S1[s], self.T1[a]
        c1 = np.minimum(self.C1[(tb, ln)], LD)                 # 종가 규칙 또는 보유 기한의 마지막 봉
        stopF = s1 <= np.minimum(t1, c1)                       # 같은 봉이면 손절이 먼저 (보수적)
        tpF = ~stopF & (t1 <= c1)
        clF = ~stopF & ~tpF
        px = np.empty(n)
        xb = np.empty(n, np.int64)
        wy = np.empty(n, np.int8) if why else None
        end = 6 if md == 0 else 10
        j = s1[stopF]
        px[stopF] = np.where(j == 0, 1 - s, np.minimum(1 - s, self.o[r[stopF], np.minimum(j, H - 1)]))
        xb[stopF] = j
        if why:
            wy[stopF] = 0
        j, rr, L = c1[clF], r[clF], LD[clF]
        at = j >= L
        px[clF] = np.where(at, self.c[rr, np.minimum(j, H - 1)], self.o[rr, np.minimum(j + 1, H - 1)])
        xb[clF] = np.where(at, j, j + 1)
        if why:
            lab = 4 if ln == 1 else 8
            hit = self.lines[ln][rr, np.minimum(j, H - 1)] if ln else np.zeros(len(j), bool)
            wy[clF] = np.where(at, end, np.where(hit, lab, 5))
        rr, bb = r[tpF], t1[tpF]
        if f >= 1:
            px[tpF] = 1 + a
            xb[tpF] = bb
            if why:
                wy[tpF] = 2
        else:
            L = LD[tpF]
            c2 = np.minimum(self.NL[ln][rr, bb], L) if ln else L
            s2, t2 = self.NBE[rr, bb + 1], self.NT2[b][rr, bb]
            w = np.stack([s2.astype(np.int64) * 3, t2.astype(np.int64) * 3 + 1, c2.astype(np.int64) * 3 + 2]).argmin(0)
            jj = np.choose(w, [s2.astype(np.int64), t2.astype(np.int64), c2.astype(np.int64)])
            jc = np.minimum(jj, H - 1)
            at = jj >= L
            p2 = np.where(w == 0, np.minimum(1.0, self.o[rr, jc]),
                          np.where(w == 1, 1 + b, np.where(at, self.c[rr, jc], self.o[rr, np.minimum(jc + 1, H - 1)])))
            px[tpF] = f * (1 + a) + (1 - f) * p2
            xb[tpF] = np.where((w == 2) & ~at, jj + 1, jj)
            if why:
                wy[tpF] = np.where(w == 0, 1, np.where(w == 1, 3, np.where(at, end, 7 if ln == 1 else 9)))
        net = px * (1 - (self.fee + self.slip + self.tax)) / (1 + self.fee) - 1
        out = (net, self.e + xb)
        return out + (wy,) if why else out


# ── 장부 (bdw_intraday.portfolio 와 같은 규칙의 빠른 판) ──────────────────────

def book(t_in, t_out, sym, day, mon, net, w, i0=0, i1=None, keep=False):
    """시간 순 후보 → 동시 max_pos · 같은 종목 동시 1번 · 같은 종목 하루 per_day_symbol 번 · 하루·한 달 손실 한도.
    반환: 거래 수 · 건당 평균 · 총수익 · 최대낙폭 · (keep) 받아들인 번호 · 자산곡선 [(청산 시각, 자산)]."""
    R = BI.RULES
    mx, per_day, dl, ml = CFG["max_pos"], R["per_day_symbol"], R["day_limit"], R["month_limit"]
    eq = peak = 1.0
    mdd = s = 0.0
    n = 0
    op, dp, mp_, cnt = [], {}, {}, {}
    taken, curve = [], []
    for k in range(i0, len(t_in) if i1 is None else i1):
        ti = t_in[k]
        if op:
            still = []
            for p in op:
                if p[0] <= ti:
                    pnl = p[2] * p[3]
                    eq += pnl
                    dp[p[4]] = dp.get(p[4], 0.0) + pnl
                    mp_[p[5]] = mp_.get(p[5], 0.0) + pnl
                    if eq > peak:
                        peak = eq
                    elif eq / peak - 1 < mdd:
                        mdd = eq / peak - 1
                    if keep:
                        curve.append((p[0], eq))
                else:
                    still.append(p)
            op = still
        if len(op) >= mx:
            continue
        sk = sym[k]
        if any(p[1] == sk for p in op):
            continue
        dk = day[k]
        key = dk * 100_000 + sk
        if cnt.get(key, 0) >= per_day or dp.get(dk, 0.0) <= -dl * eq or mp_.get(mon[k], 0.0) <= -ml * eq:
            continue
        op.append((t_out[k], sk, w * eq, net[k], dk, mon[k]))
        cnt[key] = cnt.get(key, 0) + 1
        n += 1
        s += net[k]
        if keep:
            taken.append(k)
    for p in sorted(op):
        eq += p[2] * p[3]
        if eq > peak:
            peak = eq
        elif eq / peak - 1 < mdd:
            mdd = eq / peak - 1
        if keep:
            curve.append((p[0], eq))
    return n, (s / n if n else 0.0), eq - 1, mdd, taken, curve


def weight(stop):
    if CFG["sizing"] == "equal":
        return 1.0 / CFG["max_pos"]
    return min(1.0 / CFG["max_pos"], BI.RULES["risk"] / stop)


def _lists(sig):
    g = G
    e = sig + 1
    return g["ts"][e], [g["ts"][e].tolist(), g["sym"][sig].tolist(), g["day"][e].tolist(), g["mon"][e].tolist()]


def eval_entry(ei):
    """진입 칸 하나 × 청산 칸 전부 → [청산 칸, 시대(A·B), (거래 수·건당·총수익·낙폭)]."""
    g = G
    ec = g["entry_list"][ei]
    sig = signals(*ec)
    out = np.full((len(g["exit_list"]), 2, 4), np.nan, np.float32)
    if len(sig) == 0:
        out[..., 0] = 0
        return ei, out, 0
    tin, (tinL, symL, dayL, monL) = _lists(sig)
    m = int(np.searchsorted(tin, g["mid"]))
    m2 = int(np.searchsorted(tin, g["hi"]))
    P = Paths(sig, exit_=g["exit_grid"])
    done = {}
    for xi, xc in enumerate(g["exit_list"]):
        key = exit_key(xc)
        if key is None:
            continue
        if key in done:
            out[xi] = out[done[key]]
            continue
        done[key] = xi
        s = xc[0]
        net, xo = P.exit(*xc)
        netL, toutL = net.tolist(), g["ts"][xo].tolist()
        w = weight(s)
        for era, (i0, i1) in enumerate(((0, m), (m, m2))):
            nn, mean, tot, mdd, _, _ = book(tinL, toutL, symL, dayL, monL, netL, w, i0, i1)
            out[xi, era] = (nn, mean, tot, mdd)
    return ei, out, len(sig)


# ── 고르기·확인 ─────────────────────────────────────────────────────────────

def neighbor_median(T):
    """칸마다: 자기 + 손잡이 하나만 한 칸 옮긴 이웃들의 중앙값 (격자 밖·없는 칸은 뺀다)."""
    st = [T]
    for ax in range(T.ndim):
        if T.shape[ax] < 2:
            continue
        for sh in (1, -1):
            x = np.roll(T, sh, axis=ax).copy()
            ix = [slice(None)] * T.ndim
            ix[ax] = 0 if sh == 1 else -1
            x[tuple(ix)] = np.nan
            st.append(x)
    return np.nanmedian(np.stack(st), axis=0)


def combo_of(ix, E, X):
    keys = list(E) + list(X)
    vals = list(E.values()) + list(X.values())
    return {k: v[i] for k, v, i in zip(keys, vals, ix)}


def detail(c, t0, t1, cost_mult=1.0, sig=None):
    """한 칸을 한 시대에 자세히: 받아들인 거래·자산곡선 → bdw_intraday.summary."""
    if sig is None:
        sig = signals(c["touch"], c["chase"], c["vol_mult"], c["window"], c["top_n"])
    if len(sig) == 0:
        return {"trades": 0}, sig
    tin, (tinL, symL, dayL, monL) = _lists(sig)
    i0, i1 = int(np.searchsorted(tin, t0)), int(np.searchsorted(tin, t1))
    P = Paths(sig, cost_mult, exit_=G.get("exit_grid"))
    net, xo, wy = P.exit(c["stop"], c["tp1"], c["frac"], c["tp2"], c.get("time_bars", 0), c["line_exit"], c.get("max_days", 0), why=True)
    n, mean, tot, mdd, taken, curve = book(tinL, G["ts"][xo].tolist(), symL, dayL, monL, net.tolist(), weight(c["stop"]), i0, i1, keep=True)
    if not taken:
        return {"trades": 0}, sig
    tk = pd.DataFrame({"t_in": pd.to_datetime(tin[taken]), "net": net[taken], "why": [WHY[k] for k in wy[taken]]})
    cv = pd.Series(dict((pd.Timestamp(t), e) for t, e in curve)).sort_index()
    sm = BI.summary(tk, cv)
    sm["total"], sm["mdd"] = round(float(tot), 4), round(float(mdd), 4)
    return sm, sig


def random_mean(c, sig, t0, t1, seed):
    """같은 칸·같은 시대: 종목마다 실제 신호 수만큼 같은 걸러내기 안에서 진입 시각만 무작위 → 건당 순수익.
    (총수익으로 비교하면 무작위 쪽은 신호가 덜 몰려 거래가 더 많이 받아들여지고, 손해 보는 규칙에선 그만큼 더 잃어 보인다 → 건당으로 비교)"""
    g = G
    tin = g["ts"][sig + 1]
    s_era = sig[(tin >= t0) & (tin < t1)]
    k_sym = np.bincount(g["sym"][s_era], minlength=len(g["syms"]))
    pool = eligible(c["window"], c["top_n"], t0, t1)
    ps = g["sym"][pool]
    rng = np.random.default_rng(seed)
    pick = []
    for si in np.flatnonzero(k_sym):
        cand = pool[ps == si]
        if len(cand):
            pick.append(rng.choice(cand, size=min(int(k_sym[si]), len(cand)), replace=False))
    if not pick:
        return 0.0
    rs = np.concatenate(pick)
    rs = rs[np.lexsort((-g["val_day"][rs], g["ts"][rs + 1]))]
    sm, _ = detail(c, t0, t1, sig=rs)
    return float(sm["mean"]) if sm.get("trades") else None


def confirm(c, t0, t1, hold_):
    """고른 칸을 '보지 않은' 시대에 숫자 그대로: 성적 · 무작위 진입 30번 · 비용 2배 · 기준 보유 → 합격 여부."""
    sm, sig = detail(c, t0, t1)
    rnd = [x for x in (random_mean(c, sig, t0, t1, sd) for sd in range(CRIT["n_random"])) if x is not None]
    s2, _ = detail(c, t0, t1, cost_mult=2.0, sig=sig)
    pct = float((np.array(rnd) < sm.get("mean", -9)).mean()) if rnd else None
    ok = (sm.get("trades", 0) >= CRIT["min_trades"] and sm.get("mean", -1) > 0 and (sm.get("t_day") or 0) >= CRIT["t_day"]
          and pct is not None and pct >= CRIT["random_pct"] and s2.get("mean", -1) > 0 and (hold_ is None or sm.get("mdd", -1) > hold_["mdd"]))
    return {"strategy": sm, "random_means": {"median": round(float(np.median(rnd)), 5), "p95": round(float(np.percentile(rnd, 95)), 5)} if rnd else None,
            "beats_random_pct": pct, "cost_x2": {k: s2.get(k) for k in ("trades", "mean", "total", "mdd")}, "hold": hold_, "pass": bool(ok)}


def _row(R, ix, E, X):
    c = combo_of(ix, E, X)
    a, b = R[ix]
    f = lambda v: None if not np.isfinite(v) else round(float(v), 5)       # noqa: E731
    return {"rule": c, "A": {"trades": int(a[0]), "mean": f(a[1]), "total": f(a[2]), "mdd": f(a[3])},
            "B": {"trades": int(b[0]), "mean": f(b[1]), "total": f(b[2]), "mdd": f(b[3])}}


def analyze(R, E, X, eras_ns, hold):
    """R[진입…, 청산…, 시대, 통계] → 고르기(이웃 중앙값) · 반대 시대 확인 · 손잡이별 중앙값 · 순위 상관 · 1등 목록."""
    keys = list(E) + list(X)
    out = {"picks": {}, "peaks": {}, "top": {}, "marginals": {}}
    few = min(30, CRIT["min_trades"])                          # 손잡이별·상관 계산에 넣는 최소 거래 수
    (a0, a1), (b0, b1) = eras_ns["A"], eras_ns["B"]
    for sel, oth, (t0, t1), name in ((0, 1, (b0, b1), "A→B"), (1, 0, (a0, a1), "B→A")):
        T, N = R[..., sel, 2].astype(float), R[..., sel, 0].astype(float)
        elig = (N >= CRIT["min_trades"]) & np.isfinite(T)
        if not elig.any():
            out["picks"][name] = {"note": f"고르는 시대에 거래 {CRIT['min_trades']}번 넘는 칸이 없음"}
            continue
        sm = neighbor_median(T)
        sc = np.where(elig, sm, -np.inf)
        best = np.flatnonzero(sc.ravel() == sc.max())
        pick = np.unravel_index(best[np.argmax(T.ravel()[best])], T.shape)
        c = combo_of(pick, E, X)
        hold_ = hold["B" if name == "A→B" else "A"]
        out["picks"][name] = {"rule": c, "select_era": {"total": round(float(T[pick]), 4), "neighbor_median": round(float(sm[pick]), 4),
                                                         "trades": int(N[pick])}, "confirm": confirm(c, t0, t1, hold_)}
        peak = np.unravel_index(np.argmax(np.where(elig, T, -np.inf)), T.shape)
        out["peaks"][name] = {"rule": combo_of(peak, E, X), "select_total": round(float(T[peak]), 4),
                              "confirm_total": None if not np.isfinite(R[peak][oth, 2]) else round(float(R[peak][oth, 2]), 4),
                              "confirm_trades": int(R[peak][oth, 0])}
        order = np.argsort(-np.where(elig, T, -np.inf).ravel())[:10]
        out["top"][["A", "B"][sel]] = [_row(R, np.unravel_index(i, T.shape), E, X) for i in order if np.isfinite(T.ravel()[i]) and elig.ravel()[i]]
    for ax, k in enumerate(keys):
        vals = (list(E.values()) + list(X.values()))[ax]
        out["marginals"][k] = {}
        for i, v in enumerate(vals):
            sl = np.take(R, i, axis=ax)
            row = {}
            for era, nm in ((0, "A"), (1, "B")):
                t, n = sl[..., era, 2].ravel(), sl[..., era, 0].ravel()
                t = t[np.isfinite(t) & (n >= few)]
                row[nm] = {"median_total": round(float(np.median(t)), 4) if len(t) else None,
                           "share_positive": round(float((t > 0).mean()), 3) if len(t) else None}
            out["marginals"][k][str(v)] = row
    TA, TB = R[..., 0, 2].ravel(), R[..., 1, 2].ravel()
    ok = np.isfinite(TA) & np.isfinite(TB) & (R[..., 0, 0].ravel() >= few) & (R[..., 1, 0].ravel() >= few)
    if ok.sum() > 10:
        ra, rb = pd.Series(TA[ok]).rank(), pd.Series(TB[ok]).rank()
        topA = TA[ok] >= np.quantile(TA[ok], 0.99)
        out["persistence"] = {"cells": int(ok.sum()), "rank_corr_A_B": round(float(np.corrcoef(ra, rb)[0, 1]), 3),
                              "B_median_all": round(float(np.median(TB[ok])), 4), "B_median_of_A_top1pct": round(float(np.median(TB[ok][topA])), 4),
                              "share_positive_both": round(float(((TA[ok] > 0) & (TB[ok] > 0)).mean()), 4),
                              "share_positive_A": round(float((TA[ok] > 0).mean()), 4), "share_positive_B": round(float((TB[ok] > 0).mean()), 4)}
    try:
        dix = tuple((list(E.values()) + list(X.values()))[i].index(DEFAULT[k]) for i, k in enumerate(keys))
        out["default"] = _row(R, dix, E, X)
    except (ValueError, KeyError):
        out["default"] = None
    return out


def auto_workers(mb=700):
    cpu = os.cpu_count() or 1
    try:
        avail = int([x for x in open("/proc/meminfo") if x.startswith("MemAvailable")][0].split()[1]) / 1024
    except (OSError, IndexError, ValueError):
        avail = 2048
    return max(1, min(cpu, 4, int(avail // mb)))


def run(venue, path, bench=None, workers=0, cache="~/.rich/cache", entry=None, exit_=None, data=None, log=print, round_=1):
    t_start = time.time()
    rd = ROUNDS[round_]
    CFG.update({"sizing": rd["sizing"], "max_pos": rd["max_pos"]})
    E, X, el, xl = grid(entry or rd["entry"], exit_ or rd["exit"])
    data = data if data is not None else load(venue, path, cache)
    if not data:
        raise SystemExit(f"분봉 자료가 없습니다: {path}")
    eras = IL.era_bounds(data)
    hold = {k: IL.hold(data, bench, a, b) for k, (a, b) in eras.items()}
    log(f"  자료 {len(data)}종목 · 시대 A {eras['A'][0]} ~ {eras['A'][1]} · B ~ {eras['B'][1]} ({time.time() - t_start:.0f}초)")
    build(data, venue, bench, E)
    del data
    lo, mid, hi = (pd.Timestamp(x).value for x in (eras["A"][0], eras["A"][1], eras["B"][1]))
    G.update({"entry_list": el, "exit_list": xl, "exit_grid": X, "mid": mid, "hi": hi})
    eras_ns = {"A": (lo, mid), "B": (mid, hi)}
    nw = workers or auto_workers(700 if max(X["max_days"]) == 0 else 1500)
    log(f"  배열 {len(G['o']):,}봉 · 진입 {len(el)}칸 × 청산 {len(xl)}칸 (서로 다른 규칙 {n_unique(E, X):,}칸) · 작업자 {nw}")
    res = np.full((len(el), len(xl), 2, 4), np.nan, np.float32)
    nsig = {}
    t1 = time.time()
    if nw > 1:
        with mp.get_context("fork").Pool(nw) as pool:
            it = pool.imap_unordered(eval_entry, range(len(el)))
            for k, (ei, out, ns) in enumerate(it, 1):
                res[ei], nsig[ei] = out, ns
                if k % max(1, len(el) // 20) == 0 or k == len(el):
                    log(f"  {k}/{len(el)} · {(time.time() - t1) / 60:.1f}분 (남은 예상 {(time.time() - t1) / k * (len(el) - k) / 60:.0f}분)")
    else:
        for k, ei in enumerate(range(len(el)), 1):
            _, res[ei], nsig[ei] = eval_entry(ei)
            if k % max(1, len(el) // 20) == 0 or k == len(el):
                log(f"  {k}/{len(el)} · {(time.time() - t1) / 60:.1f}분 (남은 예상 {(time.time() - t1) / k * (len(el) - k) / 60:.0f}분)")
    shape = tuple(len(v) for v in E.values()) + tuple(len(v) for v in X.values())
    R = res.reshape(shape + (2, 4))
    an = analyze(R, E, X, eras_ns, hold)
    return {"generated": time.strftime("%Y-%m-%d %H:%M"), "venue": venue, "bench": bench, "symbols": len(G["syms"]),
            "eras": {k: [str(a), str(b)] for k, (a, b) in eras.items()}, "grid": {"entry": E, "exit": X},
            "cells": len(el) * len(xl), "cells_unique": n_unique(E, X), "criteria": CRIT, "costs": BI.VENUES[venue],
            "round": round_, "sizing": dict(CFG),
            "fixed": {k: BI.RULES[k] for k in ("line", "hi_rule", "brk_lookback", "risk", "per_day_symbol", "day_limit", "month_limit")},
            "signals_median": int(np.median(list(nsig.values()))) if nsig else 0, "hold": hold, **an,
            "pass": all((p.get("confirm") or {}).get("pass", False) for p in an["picks"].values()) and len(an["picks"]) == 2,
            "secs": round(time.time() - t_start)}


def n_trials(allres):
    """1차(맨 위 시장별) + 2차 이후(round2·round3 … 아래 시장별) 서로 다른 칸 수 합."""
    n = sum(v.get("cells_unique", 0) for v in allres.values() if isinstance(v, dict) and v.get("venue"))
    for k, v in allres.items():
        if k.startswith("round") and isinstance(v, dict):
            n += sum(x.get("cells_unique", 0) for x in v.values() if isinstance(x, dict))
    return int(n)


def _pct(x):
    return "—" if x is None else f"{x * 100:+.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=list(BI.VENUES))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cache", default="~/.rich/cache")
    ap.add_argument("--round", type=int, default=2, choices=list(ROUNDS))
    a = ap.parse_args()
    print(f"■ {a.venue} 격자 탐색 {a.round}차 시작 {time.strftime('%H:%M')}", flush=True)
    res = run(a.venue, a.dir, a.bench, a.workers, a.cache, log=lambda m: print(m, flush=True), round_=a.round)
    allres = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    if a.round == 1:
        allres[a.venue] = res                                    # 1차는 맨 위 (예전 형식 그대로)
    else:
        allres.setdefault(f"round{a.round}", {})[a.venue] = res
    allres["n_trials"] = n_trials(allres)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(allres, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"■ {a.venue} · {res['cells_unique']:,}칸 · {res['secs'] // 60}분")
    if res.get("default"):
        d = res["default"]
        print(f"  기준 칸(그의 숫자): A 총 {_pct(d['A']['total'])} ({d['A']['trades']}번) · B 총 {_pct(d['B']['total'])} ({d['B']['trades']}번)")
    ps = res.get("persistence") or {}
    if ps:
        print(f"  A·B 순위 상관 {ps['rank_corr_A_B']} · 두 시대 모두 이익인 칸 {ps['share_positive_both'] * 100:.1f}% · "
              f"A 상위 1% 칸들의 B 중앙값 {_pct(ps['B_median_of_A_top1pct'])} (전체 {_pct(ps['B_median_all'])})")
    for name, p in res["picks"].items():
        if "confirm" not in p:
            print(f"  {name}: {p.get('note')}")
            continue
        s, cf = p["confirm"]["strategy"], p["confirm"]
        print(f"  {name}: {p['rule']}\n     고른 시대 총 {_pct(p['select_era']['total'])} (이웃 중앙값 {_pct(p['select_era']['neighbor_median'])}) → "
              f"확인 시대 거래 {s.get('trades')} · 승률 {s.get('win')} · 건당 {s.get('mean')} (t {s.get('t_day')}) · 총 {_pct(s.get('total'))} · "
              f"낙폭 {_pct(s.get('mdd'))} | 무작위 위 {cf['beats_random_pct']} · 비용2배 건당 {cf['cost_x2'].get('mean')} {'✓' if cf['pass'] else '✗'}")
    for name, p in res["peaks"].items():
        print(f"  그냥 1등 {name}: 고른 시대 {_pct(p['select_total'])} → 확인 시대 {_pct(p['confirm_total'])}")
    print(f"판정: {'통과' if res['pass'] else '탈락'} → {OUT}")


if __name__ == "__main__":
    main()
