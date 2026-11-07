"""
분봉 실험실 — bdw_intraday(불장단타왕 규칙 엔진)를 실제 분봉으로 검증한다. 자료가 있는 곳에서 돈다
(토스 분봉 = 서버 ~/.rich/candles/… · 코인·미국 = 이 저장소 data/intraday/raw/… — 둘 다 원본은 저장소에 올리지 않음).

사전 등록 (자료를 보기 전에 정함)
  두 시대 = 받은 기간을 시간 순으로 반 나눈 앞(A)·뒤(B)
  기준 1 무작위 진입: 같은 종목·같은 '할 만한 장'(상위봉 추세)·같은 진입 시간대 안에서 진입 시각만 무작위, 청산 규칙은 똑같이 → 30번
         → 그의 '매수 자리'가 우연보다 나은가
  기준 2 보유: 기준 종목(지수 ETF·BTC)을 그 기간 그대로 들고 있기 → 손실을 줄였나(최대낙폭)
  합격 (두 시대 모두): 거래 100번 이상 · 건당 순수익 > 0 이고 날짜별로 묶은 t ≥ 3 · 총수익이 무작위 30번의 95% 위 ·
        최대낙폭이 기준 보유보다 작음 · 비용 2배에서도 건당 순수익 > 0

  python src/intraday_lab.py --venue kr --dir ~/.rich/candles/5m/kr --bench 069500
  python src/intraday_lab.py --venue upbit --dir data/intraday/raw/upbit/5m --bench KRW-BTC
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bdw_intraday as BI  # noqa: E402

OUT = "data/intraday_lab.json"
TZ = {"kr": "Asia/Seoul", "us": "America/New_York", "upbit": "Asia/Seoul", "binance": "UTC"}
CRIT = {"min_trades": 100, "t_day": 3.0, "random_pct": 0.95, "n_random": 30}


def load_dir(path, venue):
    data = {}
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(path), "*.csv*"))):
        sym = os.path.basename(f).split(".")[0]
        d = pd.read_csv(f)
        if d.empty:
            continue
        ts = pd.to_datetime(d["ts"], utc=True, errors="coerce")
        d["ts"] = ts.dt.tz_convert(TZ[venue]).dt.tz_localize(None)
        d = d.dropna(subset=["ts", "close"]).drop_duplicates("ts").sort_values("ts")
        for c in ("open", "high", "low", "close", "volume"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["open"] = d["open"].fillna(d["close"])
        d["high"] = d[["high", "open", "close"]].max(axis=1)
        d["low"] = d[["low", "open", "close"]].min(axis=1)
        d["volume"] = d["volume"].fillna(0)
        if len(d) > 500:
            data[sym] = d.reset_index(drop=True)
    return data


def random_candidates(data, venue, n_like, seed, rules=None):
    """같은 '할 만한 장'·시간대 안에서 진입 시각만 무작위 (종목별 실제 신호 수만큼)."""
    R = {**BI.RULES, **(rules or {})}
    rng = np.random.default_rng(seed)
    rows = []
    for sym, df in data.items():
        k = n_like.get(sym, 0)
        if not k:
            continue
        d = BI.prepare(df, venue, R)
        ok = d["trend_ok"].fillna(False).astype(bool)
        if "in_session" in d:
            ok &= d["in_session"] & (d["mins_open"] <= R["entry_window_min"])
        idx = np.where(ok.to_numpy())[0]
        idx = idx[idx < len(d) - 2]
        if not len(idx):
            continue
        A = BI.arrays(d)
        bpd = BI._bars_per_day(d)
        val_day = d["value"].rolling(bpd, min_periods=max(1, bpd // 2)).sum()
        for i in rng.choice(idx, size=min(k, len(idx)), replace=False):
            t = BI.simulate_trade(A, int(i), venue, R)
            if t is None:
                continue
            kk, net, entry, why = t
            rows.append({"sym": sym, "t_sig": d["ts"].iloc[i], "t_in": d["ts"].iloc[i + 1], "t_out": d["ts"].iloc[kk], "net": net, "why": why,
                         "value_day": float(val_day.iloc[i]) if np.isfinite(val_day.iloc[i]) else 0.0, "ret_day": 0.0})
    return pd.DataFrame(rows).sort_values("t_in").reset_index(drop=True) if rows else pd.DataFrame(columns=["sym", "t_sig", "t_in", "t_out", "net", "why", "value_day", "ret_day"])


def era_bounds(data):
    lo = min(d["ts"].iloc[0] for d in data.values())
    hi = max(d["ts"].iloc[-1] for d in data.values())
    mid = lo + (hi - lo) / 2
    return {"A": (lo, mid), "B": (mid, hi)}


def cut(c, a, b):
    return c[(c["t_in"] >= a) & (c["t_in"] < b)] if len(c) else c


def hold(data, bench, a, b):
    if bench not in data:
        return None
    s = data[bench].set_index("ts")["close"]
    s = s[(s.index >= a) & (s.index < b)]
    if len(s) < 10:
        return None
    e = s / s.iloc[0]
    return {"total": round(float(e.iloc[-1] - 1), 4), "mdd": round(float((e / e.cummax() - 1).min()), 4)}


def run(venue, path, bench, rules=None):
    t0 = time.time()
    data = load_dir(path, venue)
    if not data:
        raise SystemExit(f"분봉 자료가 없습니다: {path}")
    eras = era_bounds(data)
    cand = BI.candidates(data, venue, rules, bench=bench if bench in data else None)
    old = {k: dict(v) for k, v in BI.VENUES.items()}
    BI.VENUES[venue]["fee"] *= 2
    BI.VENUES[venue]["slip"] *= 2
    cand2 = BI.candidates(data, venue, rules, bench=bench if bench in data else None)
    BI.VENUES.update(old)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "venue": venue, "rules": {**BI.RULES, **(rules or {})}, "costs": BI.VENUES[venue],
           "symbols": len(data), "bench": bench, "criteria": CRIT, "eras": {k: [str(a), str(b)] for k, (a, b) in eras.items()}, "rows": {}}
    for era, (a, b) in eras.items():
        c = cut(cand, a, b)
        tk, curve = BI.portfolio(c, rules)
        sm = BI.summary(tk, curve)
        n_like = c["sym"].value_counts().to_dict() if len(c) else {}
        rnd = []
        for sd in range(CRIT["n_random"]):
            rc = cut(random_candidates(data, venue, n_like, sd, rules), a, b)
            rtk, rcurve = BI.portfolio(rc, rules)
            rs = BI.summary(rtk, rcurve)
            rnd.append(rs.get("total", 0.0) if rs.get("trades") else 0.0)
        tk2, curve2 = BI.portfolio(cut(cand2, a, b), rules)
        s2 = BI.summary(tk2, curve2)
        hb = hold(data, bench, a, b)
        pct = float((np.array(rnd) < sm.get("total", -9)).mean()) if rnd else None
        ok = (sm.get("trades", 0) >= CRIT["min_trades"] and sm.get("mean", -1) > 0 and (sm.get("t_day") or 0) >= CRIT["t_day"]
              and pct is not None and pct >= CRIT["random_pct"] and (hb is None or sm.get("mdd", -1) > hb["mdd"]) and s2.get("mean", -1) > 0)
        res["rows"][era] = {"strategy": sm, "random_totals": {"median": round(float(np.median(rnd)), 4), "p95": round(float(np.percentile(rnd, 95)), 4)} if rnd else None,
                            "beats_random_pct": pct, "hold": hb, "cost_x2": {k: s2.get(k) for k in ("trades", "mean", "total", "mdd")}, "pass": bool(ok)}
    res["pass"] = all(v["pass"] for v in res["rows"].values())
    res["secs"] = round(time.time() - t0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=list(BI.VENUES))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--bench", default=None)
    a = ap.parse_args()
    res = run(a.venue, a.dir, a.bench)
    allres = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    allres[a.venue] = res
    allres["n_trials"] = sum(1 for k, v in allres.items() if isinstance(v, dict) and v.get("venue"))
    json.dump(allres, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"■ {a.venue} · {res['symbols']}종목 · 시대 A {res['eras']['A'][0][:10]}~{res['eras']['A'][1][:10]} · B ~{res['eras']['B'][1][:10]}")
    for era, r in res["rows"].items():
        s = r["strategy"]
        print(f"  {era}: 거래 {s.get('trades')} (월 {s.get('per_month')}) · 승률 {s.get('win')} · 이길 때 {s.get('avg_win')} / 질 때 {s.get('avg_loss')} (손익비 {s.get('payoff')}) · "
              f"건당 {s.get('mean')} (t {s.get('t_day')}) · 총 {s.get('total')} · 낙폭 {s.get('mdd')} | 무작위 위 {r['beats_random_pct']} · 보유 {r['hold']} · 비용2배 {r['cost_x2']} {'✓' if r['pass'] else '✗'}")
    print(f"판정: {'통과' if res['pass'] else '탈락'} ({res['secs']}초) → {OUT}")


if __name__ == "__main__":
    main()
