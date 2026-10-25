"""
성과 보고서 + 전략 설정 시뮬레이터용 격자 -> docs/backtest.json

시중 도구에서 가져온 구성
  · TradingView Strategy Tester : 요약 지표 · 자본곡선 · 낙폭 · 거래 목록
  · QuantStats / pyfolio        : 월별 수익 히트맵 · 수중(underwater) 차트 · 롤링 수익 · Sortino/Calmar
  · Trade 분석 (MAE/MFE)        : 거래 중 최대 역행·최대 순행 → 손절·익절 폭이 적절한지
  · 젠포트·인텔리퀀트·Composer  : 설정을 바꾸면 결과가 어떻게 달라지나 → 36개 조합을 미리 계산

모든 수치는 decision_engine.decide() 를 그대로 돌린 결과 (라이브와 같은 코드),
t일 판단 → t+1일 종가 체결, 편도 비용 0.15%. **가상(hypothetical) 성과이며 실제 계좌 결과가 아니다.**

사용: python src/tearsheet.py            # 보고서 + 격자 (4코어 약 3~4분)
      python src/tearsheet.py --no-grid  # 보고서만 (약 15초)
"""
import os
import sys

# src/signal.py 가 표준 모듈 signal 과 이름이 같아, 병렬 처리(multiprocessing)가 엉뚱한 파일을 불러온다.
# 스크립트 폴더를 경로에서 잠시 빼고 표준 signal 을 먼저 올려 둔다.
_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _SRC]
import signal  # noqa: E402,F401
sys.path.append(_SRC)

import argparse                                  # noqa: E402
import math                                      # noqa: E402
from statistics import NormalDist               # noqa: E402
import copy                                      # noqa: E402
import itertools                                 # noqa: E402
import json                                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor   # noqa: E402

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import features as F                         # noqa: E402
import settings                                # noqa: E402
from engine_backtest import run, WARMUP, SPLIT, CFG   # noqa: E402

OUT = "docs/backtest.json"
# 손절선은 -10%·-15% 결과가 같았다 (고점 -8% 트레일이 거의 항상 먼저 걸린다) → 트레일 폭을 축으로 쓴다
GRID = {"max_positions": [5, 10, 15], "trail_base": [0.06, 0.08, 0.12], "stop_loss": [0.07, 0.10],
        "crash_gate": [True, False], "take_profit": [True, False]}


def r(x, k=2):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def metrics(curve, bench=None):
    c = np.asarray(curve, float)
    dr = np.diff(c) / c[:-1]
    yrs = len(c) / 252
    dd = c / np.maximum.accumulate(c) - 1
    # 최장 수중 기간 (고점 회복까지 걸린 거래일)
    longest, cur = 0, 0
    for x in dd:
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    down = dr[dr < 0]
    cagr = c[-1] ** (1 / yrs) - 1
    out = {"cagr": r(cagr * 100, 1), "total": r((c[-1] - 1) * 100, 1),
           "vol": r(dr.std() * np.sqrt(252) * 100, 1),
           "sharpe": r(dr.mean() / (dr.std() + 1e-12) * np.sqrt(252)),
           "sortino": r(dr.mean() / (down.std() + 1e-12) * np.sqrt(252)),
           "mdd": r(dd.min() * 100, 1), "calmar": r(cagr / abs(dd.min()) if dd.min() < 0 else None),
           "longest_dd_days": int(longest)}
    if bench is not None:
        b = np.asarray(bench, float)
        br = np.diff(b) / b[:-1]
        cov = np.cov(dr, br)
        beta = cov[0, 1] / cov[1, 1]
        out["beta"] = r(beta)
        out["alpha"] = r(((1 + dr.mean() - beta * br.mean()) ** 252 - 1) * 100, 1)
        out["corr"] = r(np.corrcoef(dr, br)[0, 1])
    return out


def psr(dr, sr0=0.0):
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012) — 진짜 Sharpe가 sr0(일간)보다 클 확률.
    표본 길이·왜도·첨도를 반영해 '우연히 좋아 보이는' 정도를 깎는다."""
    dr = np.asarray(dr, float)
    n = len(dr)
    sr = dr.mean() / (dr.std() + 1e-12)
    sk = float(((dr - dr.mean()) ** 3).mean() / (dr.std() ** 3 + 1e-12))
    ku = float(((dr - dr.mean()) ** 4).mean() / (dr.std() ** 4 + 1e-12))
    den = math.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr * sr, 1e-12))
    return NormalDist().cdf((sr - sr0) * math.sqrt(n - 1) / den)


def deflated_sr0(srs):
    """여러 설정(N개)을 시험했을 때 '최고값'이 우연히 나올 기대 Sharpe (Deflated Sharpe의 기준선)."""
    srs = np.asarray(srs, float)
    n = len(srs)
    g = 0.5772156649
    z = NormalDist().inv_cdf
    return float(np.std(srs) * ((1 - g) * z(1 - 1 / n) + g * z(1 - 1 / (n * math.e))))


def worst_drawdowns(curve, dates, k=5):
    """낙폭 구간: 고점일 → 바닥일 → 회복일 (QuantStats 'Worst Drawdowns')."""
    c = np.asarray(curve, float)
    peak_i, out, i = 0, [], 1
    while i < len(c):
        if c[i] >= c[peak_i]:
            peak_i = i
            i += 1
            continue
        j = i
        while j < len(c) and c[j] < c[peak_i]:
            j += 1
        seg = c[peak_i:j]
        tr = peak_i + int(np.argmin(seg))
        out.append({"start": str(dates[peak_i].date()), "trough": str(dates[tr].date()),
                    "end": str(dates[j].date()) if j < len(c) else None,
                    "depth": r((c[tr] / c[peak_i] - 1) * 100, 1),
                    "days": int((j if j < len(c) else len(c) - 1) - peak_i)})
        peak_i, i = (j, j + 1) if j < len(c) else (peak_i, len(c))
    return sorted(out, key=lambda x: x["depth"])[:k]


def period_returns(curve, bench):
    """기간별 수익률 표 (키우GO·RATB 화면 형식)."""
    rows = []
    for lab, n in [("1주", 5), ("1개월", 21), ("3개월", 63), ("6개월", 126), ("1년", 252), ("2년", 504)]:
        if len(curve) > n:
            rows.append({"label": lab, "strat": r((curve[-1] / curve[-1 - n] - 1) * 100, 1),
                         "bench": r((bench[-1] / bench[-1 - n] - 1) * 100, 1)})
    return rows


def cone(dr, horizon=250):
    """실전 추적용 기대 범위 (pyfolio cone) — 백테스트 일간 수익의 평균·표준편차로 n일 뒤 ±1σ·±2σ."""
    mu, sd = float(np.mean(dr)), float(np.std(dr))
    out = []
    for h in range(0, horizon + 1, 5):
        m = (1 + mu) ** h
        w = sd * math.sqrt(h)
        out.append([h, r(m, 4), r(m * (1 - w), 4), r(m * (1 + w), 4), r(m * (1 - 2 * w), 4), r(m * (1 + 2 * w), 4)])
    return out


def by_stock(trades, names, dates):
    """종목별 엔진 매매 성적 (라씨 매매비서의 종목 신호 성과 카드)."""
    g = {}
    for t in trades:
        if t["frac"] < 1 and t["why"] == "부분익절":
            continue                                   # 부분익절은 같은 거래의 일부 — 횟수에서 뺀다
        g.setdefault(t["code"], []).append(t)
    out = {}
    for c, ts in g.items():
        ret = np.array([x["ret"] for x in ts])
        out[c] = {"name": names.get(c, c), "n": len(ts), "win": r((ret > 0).mean() * 100, 0),
                  "avg": r(ret.mean() * 100, 1), "med": r(float(np.median(ret)) * 100, 1),
                  "hold": r(np.mean([x["held"] for x in ts]), 0),
                  "last": str(dates[ts[-1]["t1"]].date())}
    return out


def trade_stats(trades):
    if not trades:
        return {}
    ret = np.array([t["ret"] for t in trades])
    w = np.array([t["w"] * t["frac"] for t in trades])        # 계좌 대비 크기로 가중
    gains, losses = (ret[ret > 0] * w[ret > 0]).sum(), -(ret[ret <= 0] * w[ret <= 0]).sum()
    wins, loss = ret[ret > 0], ret[ret <= 0]
    by = {}
    for t in trades:
        b = by.setdefault(t["why"], [])
        b.append(t["ret"])
    return {"n": len(trades), "win": r((ret > 0).mean() * 100, 1),
            "profit_factor": r(gains / losses if losses > 0 else None),
            "payoff": r(wins.mean() / -loss.mean() if len(wins) and len(loss) else None),
            "avg": r(ret.mean() * 100), "avg_win": r(wins.mean() * 100 if len(wins) else None),
            "avg_loss": r(loss.mean() * 100 if len(loss) else None),
            "best": r(ret.max() * 100, 1), "worst": r(ret.min() * 100, 1),
            "avg_hold": r(np.mean([t["held"] for t in trades]), 0),
            "expectancy": r(ret.mean() * 100),
            "exits": [{"why": k, "n": len(v), "avg": r(np.mean(v) * 100), "win": r(np.mean(np.array(v) > 0) * 100, 1)}
                      for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))]}


def monthly(curve, dates):
    s = pd.Series(curve, index=dates)
    m = s.resample("ME").last()
    ret = m.pct_change()
    ret.iloc[0] = m.iloc[0] / s.iloc[0] - 1
    rows = {}
    for d, v in ret.items():
        rows.setdefault(d.year, [None] * 12)[d.month - 1] = r(v * 100, 1)
    yearly = s.resample("YE").last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / s.iloc[0] - 1
    return [{"year": y, "months": v, "year_ret": r(yr[yr.index.year == y].iloc[0] * 100, 1)}
            for y, v in sorted(rows.items())]


def _grid_one(args):
    combo, cfg = args
    panel, feats = _CACHE["pf"]
    c = copy.deepcopy(cfg)
    st = c["strategy"]
    st["max_positions"] = combo["max_positions"]
    st["entry_rank"] = combo["max_positions"]            # 진입 순위 = 보유 종목 수 (기존 규칙과 같은 관계)
    st["stop_loss"] = combo["stop_loss"]
    st["trail_base"] = combo["trail_base"]
    st["crash_gate"] = combo["crash_gate"]
    if not combo["take_profit"]:
        st["take_profit_gain"] = 9.99                     # 사실상 끔
    res = run(panel, feats, c, WARMUP, len(panel["dates"]), detail=True)
    raw = res["raw"]
    cv = raw["curve"]
    dates = panel["dates"][WARMUP:WARMUP + len(cv)]
    k = int(np.searchsorted(dates.values, np.datetime64(SPLIT)))
    tr_, te_ = cv[:k], cv[k - 1:] / cv[k - 1]
    ts = trade_stats(raw["trades"])
    dr = np.diff(cv) / cv[:-1]
    return {**combo, "sr_daily": float(dr.mean() / (dr.std() + 1e-12)),
            "train": metrics(tr_), "test": metrics(te_),
            "full": metrics(cv), "win": ts.get("win"), "n_trades": ts.get("n"),
            "profit_factor": ts.get("profit_factor"),
            "curve": [r(x, 3) for x in cv[::10]]}


_CACHE = {}


def _init():
    p, f = F.build_all()
    _CACHE["pf"] = (p, F.apply_strategy(p, f, settings.load(CFG)["strategy"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-grid", action="store_true")
    a = ap.parse_args()
    cfg = settings.load(CFG)
    panel, feats = F.build_all()
    feats = F.apply_strategy(panel, feats, cfg["strategy"])
    dates = panel["dates"]
    nD = len(dates)
    print("본 보고서 계산…")
    res = run(panel, feats, cfg, WARMUP, nD, detail=True)
    raw = res["raw"]
    cv = raw["curve"]
    dd_ = dates[WARMUP:WARMUP + len(cv)]
    idx = panel["market"]["idx"].values[WARMUP:WARMUP + len(cv)]
    bench = idx / idx[0]
    k = int(np.searchsorted(dd_.values, np.datetime64(SPLIT)))
    dd = cv / np.maximum.accumulate(cv) - 1
    bdd = bench / np.maximum.accumulate(bench) - 1
    roll = pd.Series(cv).pct_change(126).values
    broll = pd.Series(bench).pct_change(126).values
    names = panel["names"]
    trades = raw["trades"]
    tl = [{"name": names.get(t["code"], t["code"]), "code": t["code"],
           "in": str(dates[t["t0"]].date()), "out": str(dates[t["t1"]].date()),
           "ret": r(t["ret"] * 100, 1), "held": t["held"], "why": t["why"],
           "frac": r(t["frac"], 2), "mae": r(t["mae"] * 100, 1), "mfe": r(t["mfe"] * 100, 1)}
          for t in trades]
    step = 5
    out = {
        "generated": str(pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")),
        "period": [str(dd_[0].date()), str(dd_[-1].date())], "split": SPLIT,
        "assumptions": "판단 엔진(decide) 그대로 · t일 판단→t+1일 종가 체결 · 편도 0.15% · 소수점 주식 · "
                       "생존 100종목(상장폐지 미포함) · 가상 성과",
        "settings": {k_: cfg["strategy"][k_] for k_ in ("max_positions", "entry_rank", "exit_rank",
                                                         "min_hold_days", "stop_loss", "trail_base",
                                                         "take_profit_gain", "crash_gate")},
        "summary": {"full": metrics(cv, bench), "train": metrics(cv[:k], bench[:k]),
                    "test": metrics(cv[k - 1:] / cv[k - 1], bench[k - 1:] / bench[k - 1]),
                    "bench_full": metrics(bench), "bench_train": metrics(bench[:k]),
                    "bench_test": metrics(bench[k - 1:] / bench[k - 1])},
        "trades": trade_stats(trades),
        "exposure_avg": r(np.mean(raw["expo"]) * 100, 1),
        "series": {"d": [str(x.date()) for x in dd_[::step]], "eq": [r(x, 3) for x in cv[::step]],
                   "bench": [r(x, 3) for x in bench[::step]], "dd": [r(x * 100, 1) for x in dd[::step]],
                   "bdd": [r(x * 100, 1) for x in bdd[::step]],
                   "roll": [r(x * 100, 1) for x in roll[::step]], "broll": [r(x * 100, 1) for x in broll[::step]],
                   "expo": [r(x * 100, 0) for x in raw["expo"][::step]], "split_i": k // step},
        "monthly": monthly(cv, dd_),
        "bench_monthly": monthly(bench, dd_),
        "trade_list": tl[-80:][::-1],
        "worst_dd": worst_drawdowns(cv, dd_),
        "periods": period_returns(cv, bench),
        "psr": r(psr(np.diff(cv) / cv[:-1]) * 100, 1),
        "psr_test": r(psr(np.diff(cv[k - 1:]) / cv[k - 1:-1]) * 100, 1),
        "cone": cone(np.diff(cv) / cv[:-1]),
        "by_stock": by_stock(trades, names, dates),
        "label": "BACKTEST (가설 성과)",
        "disclaimer": ("가설(시뮬레이션) 성과입니다. 실제 매매 기록이 아니며, 사후에 알고 있는 정보로 규칙을 "
                       "정했다는 한계가 있습니다. 유동성·체결 실패·세금 변화를 완전히 반영하지 못합니다. "
                       "어떤 계좌도 이와 비슷한 수익이나 손실을 낼 것이라는 보장은 없습니다."),
        "mae_mfe": [[t["mae"], t["mfe"], t["ret"]] for t in tl if t["frac"] == 1],
    }
    s = out["summary"]
    print(f"  전체 CAGR {s['full']['cagr']}% Sharpe {s['full']['sharpe']} MDD {s['full']['mdd']}% "
          f"· 시장 {s['bench_full']['cagr']}% / {s['bench_full']['mdd']}%")
    print(f"  거래 {out['trades']['n']} · 승률 {out['trades']['win']}% · PF {out['trades']['profit_factor']}"
          f" · 손익비 {out['trades']['payoff']} · 평균노출 {out['exposure_avg']}%")

    if not a.no_grid:
        combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
        print(f"설정 격자 {len(combos)}개 계산…")
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 2, initializer=_init) as ex:
            grid = list(ex.map(_grid_one, [(c, cfg) for c in combos]))
        out["grid"] = {"axes": GRID, "rows": grid, "curve_step": 10,
                       "curve_d": [str(x.date()) for x in dd_[::10]]}
        cur = cfg["strategy"]
        base = next(g for g in grid if g["max_positions"] == cur["max_positions"]
                    and abs(g["stop_loss"] - cur["stop_loss"]) < 1e-9 and g["crash_gate"] == cur["crash_gate"]
                    and abs(g["trail_base"] - cur["trail_base"]) < 1e-9
                    and g["take_profit"])
        better_both = sum(1 for g in grid if g["train"]["sharpe"] > base["train"]["sharpe"]
                          and g["test"]["sharpe"] > base["test"]["sharpe"])
        sr0 = deflated_sr0([g["sr_daily"] for g in grid])
        best = max(grid, key=lambda g: g["sr_daily"])
        nfull = len(cv) - 1
        for g in grid:                                 # 각 조합의 '72개 중 고른 값' 보정 확률
            g["dsr"] = r(NormalDist().cdf((g["sr_daily"] - sr0) * math.sqrt(nfull)) * 100, 0)   # 왜도·첨도 보정은 생략한 근사
        out["grid"]["deflated"] = {"n_trials": len(grid), "sr0_annual": r(sr0 * math.sqrt(252)),
                                   "best": {k_: best[k_] for k_ in GRID},
                                   "best_sharpe": best["full"]["sharpe"], "best_dsr": best["dsr"],
                                   "current_dsr": base["dsr"]}
        out["grid"]["note"] = (f"현재 설정보다 2019~2023·2024~ 두 구간 Sharpe가 모두 높은 조합: {better_both}/{len(grid)}개. "
                               f"{len(grid)}개를 시험했으므로 운만으로도 최고 Sharpe가 약 {r(sr0*math.sqrt(252))} "
                               "높아 보일 수 있습니다. 최근 구간만 보고 고르면 과적합입니다.")
        print("  " + out["grid"]["note"])
    elif os.path.exists(OUT):
        try:
            old = json.load(open(OUT, encoding="utf-8"))
            if "grid" in old:
                out["grid"] = old["grid"]               # 격자는 주 1회만 다시 계산 — 지난 값을 유지
        except Exception:
            pass
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"),
              default=lambda o: o.item() if hasattr(o, "item") else None)
    print(f"저장: {OUT} ({os.path.getsize(OUT)//1024}KB)")


if __name__ == "__main__":
    main()
