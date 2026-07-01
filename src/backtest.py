"""
백테스트 - 점수와 미래수익률의 관계 검증
- 입력: data/scores.csv (종합점수), data/prices.csv (수익률 계산용)
- 기간 분리: train 2018-2023 / validation 2024-2025 / live 2026+
- 지표:
  1) 지연 상관(lag N=1,5,10,20): 오늘 점수 vs N일 후 수익률 (Spearman)
  2) IC / IR : 일별 단면 Spearman 상관의 평균(IC)과 평균/표준편차(IR)
  3) 5분위 스프레드: 점수 상위20% vs 하위20% 평균 미래수익 (단조성 체크)
  4) 롱숏(상위20% 매수 - 하위20% 공매도) vs 동일가중 시장 누적수익
     · 보유기간 --hold 일 마다 리밸런싱
  5) 연율 Sharpe, MDD
- 사용:
  python src/backtest.py --score 종합점수 --hold 20
"""
import argparse
import numpy as np
import pandas as pd

PRICES = "data/prices.csv"
SCORES = "data/scores.csv"

PERIODS = {
    "train": ("2018-01-01", "2023-12-31"),
    "validation": ("2024-01-01", "2025-12-31"),
    "live": ("2026-01-01", "2100-01-01"),
}


def load(score_col):
    px = pd.read_csv(PRICES, dtype={"종목코드": str})
    px["종목코드"] = px["종목코드"].str.zfill(6)
    px["날짜"] = pd.to_datetime(px["날짜"])
    px = px[["날짜", "종목코드", "종가"]].sort_values(["종목코드", "날짜"])

    sc = pd.read_csv(SCORES, dtype={"종목코드": str})
    sc["종목코드"] = sc["종목코드"].str.zfill(6)
    sc["날짜"] = pd.to_datetime(sc["날짜"])
    if score_col not in sc.columns:
        raise SystemExit(f"점수 컬럼 '{score_col}' 없음. 사용 가능: {list(sc.columns)}")
    sc = sc[["날짜", "종목코드", score_col]].rename(columns={score_col: "score"})

    df = pd.merge(px, sc, on=["날짜", "종목코드"], how="inner")
    df = df.sort_values(["종목코드", "날짜"]).reset_index(drop=True)
    return df


def add_fwd_returns(df, horizons):
    parts = []
    for code, g in df.groupby("종목코드"):
        g = g.sort_values("날짜").copy()
        for n in horizons:
            g[f"fwd{n}"] = g["종가"].shift(-n) / g["종가"] - 1
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)


def slice_period(df, period):
    a, b = PERIODS[period]
    m = (df["날짜"] >= a) & (df["날짜"] <= b)
    return df[m].copy()


def lag_corr(df, horizons):
    out = {}
    for n in horizons:
        sub = df.dropna(subset=["score", f"fwd{n}"])
        if len(sub) < 50:
            out[n] = np.nan
            continue
        out[n] = sub["score"].corr(sub[f"fwd{n}"], method="spearman")
    return out


def daily_ic(df, n):
    """일별 단면 Spearman IC 시계열 → IC평균, IR"""
    ics = []
    for d, g in df.groupby("날짜"):
        g = g.dropna(subset=["score", f"fwd{n}"])
        if g["종목코드"].nunique() < 5:
            continue
        ics.append(g["score"].corr(g[f"fwd{n}"], method="spearman"))
    ics = pd.Series(ics, dtype=float).dropna()
    if len(ics) < 5:
        return np.nan, np.nan
    ic = ics.mean()
    ir = ic / ics.std() if ics.std() > 0 else np.nan
    return ic, ir


def quantile_spread(df, n, q=5):
    """일별 점수 5분위, 분위별 평균 미래수익. 상위-하위 스프레드 + 단조성"""
    rows = []
    for d, g in df.groupby("날짜"):
        g = g.dropna(subset=["score", f"fwd{n}"])
        if g["종목코드"].nunique() < q:
            continue
        try:
            g["bin"] = pd.qcut(g["score"], q, labels=False, duplicates="drop")
        except ValueError:
            continue
        rows.append(g.groupby("bin")[f"fwd{n}"].mean())
    if not rows:
        return None
    qmean = pd.concat(rows, axis=1).mean(axis=1)
    qmean.index = [f"Q{int(i)+1}" for i in qmean.index]
    spread = qmean.iloc[-1] - qmean.iloc[0]
    mono = qmean.is_monotonic_increasing
    return qmean, spread, mono


def long_short_curve(df, hold, q=5):
    """hold일 리밸런싱. 상위20% 롱 - 하위20% 숏 (동일가중) vs 시장 동일가중.
    리밸런싱일에 보유종목 확정 → 다음 리밸런싱까지 fwd{hold} 수익 1회 적용(비중첩)."""
    dates = np.sort(df["날짜"].unique())
    rebal = dates[::hold]
    ls_rets, mkt_rets, used = [], [], []
    for d in rebal:
        g = df[df["날짜"] == d].dropna(subset=["score", f"fwd{hold}"])
        if g["종목코드"].nunique() < q:
            continue
        try:
            g = g.copy()
            g["bin"] = pd.qcut(g["score"], q, labels=False, duplicates="drop")
        except ValueError:
            continue
        top = g[g["bin"] == g["bin"].max()][f"fwd{hold}"].mean()
        bot = g[g["bin"] == g["bin"].min()][f"fwd{hold}"].mean()
        mkt = g[f"fwd{hold}"].mean()
        ls_rets.append(top - bot)
        mkt_rets.append(mkt)
        used.append(d)
    if not ls_rets:
        return None
    ls = pd.Series(ls_rets, index=pd.to_datetime(used))
    mkt = pd.Series(mkt_rets, index=pd.to_datetime(used))
    return ls, mkt


def perf_stats(period_rets, hold):
    """비중첩 보유수익 시계열 → 연율 수익/Sharpe/MDD"""
    if period_rets is None or len(period_rets) < 2:
        return {}
    cum = (1 + period_rets).cumprod()
    n_per_year = 252 / hold
    total = cum.iloc[-1] - 1
    years = len(period_rets) / n_per_year
    cagr = (cum.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = (period_rets.mean() / period_rets.std() * np.sqrt(n_per_year)
              if period_rets.std() > 0 else np.nan)
    peak = cum.cummax()
    mdd = ((cum - peak) / peak).min()
    return {"누적": total, "CAGR": cagr, "Sharpe": sharpe, "MDD": mdd}


def run_period(df, period, horizons, hold):
    sub = slice_period(df, period)
    if sub.empty:
        print(f"\n===== [{period}] 데이터 없음 =====")
        return
    print(f"\n===== [{period}]  {PERIODS[period][0]} ~ {PERIODS[period][1]} "
          f"| {sub['날짜'].nunique()}일 {sub['종목코드'].nunique()}종목 =====")

    lc = lag_corr(sub, horizons)
    print("· 지연 상관(Spearman):",
          "  ".join(f"N{n}={lc[n]:.3f}" if pd.notna(lc[n]) else f"N{n}=NA"
                    for n in horizons))

    for n in horizons:
        ic, ir = daily_ic(sub, n)
        ic_s = f"{ic:.3f}" if pd.notna(ic) else "NA"
        ir_s = f"{ir:.3f}" if pd.notna(ir) else "NA"
        print(f"· IC(N{n})={ic_s}  IR(N{n})={ir_s}")

    qs = quantile_spread(sub, hold)
    if qs:
        qmean, spread, mono = qs
        print(f"· 5분위 평균 미래{hold}일수익:",
              "  ".join(f"{k}={v*100:.2f}%" for k, v in qmean.items()))
        print(f"  상위-하위 스프레드={spread*100:.2f}%  단조증가={mono}")

    lsmkt = long_short_curve(sub, hold)
    if lsmkt:
        ls, mkt = lsmkt
        ls_p = perf_stats(ls, hold)
        mkt_p = perf_stats(mkt, hold)
        if ls_p:
            print(f"· 롱숏(hold={hold}): 누적={ls_p['누적']*100:.1f}%  "
                  f"CAGR={ls_p['CAGR']*100:.1f}%  Sharpe={ls_p['Sharpe']:.2f}  "
                  f"MDD={ls_p['MDD']*100:.1f}%")
        if mkt_p:
            print(f"· 시장(동일가중):    누적={mkt_p['누적']*100:.1f}%  "
                  f"CAGR={mkt_p['CAGR']*100:.1f}%  Sharpe={mkt_p['Sharpe']:.2f}  "
                  f"MDD={mkt_p['MDD']*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default="종합점수")
    ap.add_argument("--hold", type=int, default=20)
    args = ap.parse_args()

    horizons = sorted(set([1, 5, 10, 20, args.hold]))
    df = load(args.score)
    df = add_fwd_returns(df, horizons)
    print(f"백테스트 점수컬럼='{args.score}'  보유={args.hold}일  "
          f"전체 {df['날짜'].nunique()}일 {df['종목코드'].nunique()}종목")

    for period in ["train", "validation", "live"]:
        run_period(df, period, [1, 5, 10, 20], args.hold)


if __name__ == "__main__":
    main()

