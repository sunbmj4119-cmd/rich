"""
시장국면(regime) 엔진 -> data/regime.json

왜 필요한가
  종목 점수는 '같은 시장 안에서 누가 나은가'(횡단면)만 말한다. 시장 자체가 무너지면
  1위 종목도 같이 빠진다. 그래서 '지금이 어떤 시장인가'와 '그 시장에서 이 전략이
  과거 어땠나'를 따로 계산해 판단에 얹는다.

계산
  1) KOSPI100 등가중 지수 — prices.csv 100종목 일간수익 평균의 누적곱.
  2) 국면 4분류 (그날 알 수 있는 정보만 사용, 미래참조 없음)
       급락장 : 지수 고점대비 낙폭 <= -15%
       약세장 : 200일 이동평균 아래
       강세장 : 200일선 위 + 60일선 위 종목비율(breadth) >= 55%
       횡보장 : 나머지
  3) 국면별 조건부 성과 — 각 거래일 종합점수 상위 10종목 동일가중의 30거래일 수익을
       그날의 국면으로 묶어 승률·중앙값·사분위 집계. 시장 자체 수익과 비교해 '초과분'도.
  4) 국면 전이확률 — t일 국면 → t+30거래일 국면의 실측 빈도.
  5) 종목별 베타 / 하락베타 — 최근 250거래일. 하락베타는 시장이 내린 날만 사용.
       (스트레스 테스트: 시장 -10%면 이 종목은 대략 -하락베타×10%)

한계 (정직 고지)
  - 표본은 생존 100종목 2018~2026. 급락장 표본은 적어 그 칸의 통계는 불확실하다.
  - 30일 수익을 매일 겹쳐 표집하므로 표시 N보다 독립 관측치는 훨씬 적다(≈N/30).
"""
import os
import json

import numpy as np
import pandas as pd

PRICES = "data/prices.csv"
SCORES = "data/scores.csv"
OUT = "data/regime.json"

FWD = 30          # 예측 지평(거래일)
TOP_N = 10        # 전략 = 종합점수 상위 10
BETA_WIN = 250    # 베타 추정 창
CRASH_DD = -0.15  # 급락장 기준(고점대비)
BULL_BREADTH = 0.55


def _q(x, p):
    return float(np.nanpercentile(x, p)) if len(x) else float("nan")


def build_index(px: pd.DataFrame):
    """100종목 등가중 지수 + 국면 피처. 반환: DataFrame(index=날짜)"""
    piv = px.pivot_table(index="날짜", columns="종목코드", values="종가").sort_index()
    ret = piv.pct_change()
    # 등가중 일간수익 (그날 데이터 있는 종목만 평균)
    mret = ret.mean(axis=1, skipna=True)
    idx = (1 + mret.fillna(0)).cumprod()

    ma200 = idx.rolling(200, min_periods=100).mean()
    ma60_stock = piv.rolling(60, min_periods=30).mean()
    breadth = (piv > ma60_stock).sum(axis=1) / piv.notna().sum(axis=1).replace(0, np.nan)
    vol = mret.rolling(20, min_periods=10).std() * np.sqrt(252)
    dd = idx / idx.cummax() - 1

    df = pd.DataFrame({"idx": idx, "mret": mret, "ma200": ma200,
                       "breadth": breadth, "vol": vol, "dd": dd})
    df["trend"] = df["idx"] / df["ma200"] - 1
    return df, piv, ret, mret


def classify(row):
    if pd.isna(row["ma200"]) or pd.isna(row["breadth"]):
        return None
    if row["dd"] <= CRASH_DD:
        return "급락장"
    if row["trend"] < 0:
        return "약세장"
    if row["breadth"] >= BULL_BREADTH:
        return "강세장"
    return "횡보장"


REGIME_META = {
    "강세장": {"emoji": "🔥", "color": "#34c759",
               "desc": "지수가 200일선 위 + 절반 이상 종목이 60일선 위. 추세추종이 통하는 구간."},
    "횡보장": {"emoji": "😐", "color": "#ff9500",
               "desc": "200일선 위지만 상승 종목 폭이 좁다. 종목별 편차가 커지는 구간."},
    "약세장": {"emoji": "🌧", "color": "#ff3b30",
               "desc": "지수가 200일선 아래. 좋은 점수도 시장에 끌려 내려가기 쉬운 구간."},
    "급락장": {"emoji": "⛈", "color": "#8e0000",
               "desc": "고점대비 -15% 이상. 손절이 연쇄 발동하기 쉬우니 신규진입은 소액으로."},
}
ORDER = ["강세장", "횡보장", "약세장", "급락장"]


def main():
    px = pd.read_csv(PRICES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종가"])
    px["종목코드"] = px["종목코드"].str.zfill(6)
    px["날짜"] = pd.to_datetime(px["날짜"])
    df, piv, ret, mret = build_index(px)
    df["regime"] = df.apply(classify, axis=1)

    # ── 전략(상위10) 30일 선도수익 ─────────────────────────────
    sc = pd.read_csv(SCORES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종가", "종합점수"])
    sc["종목코드"] = sc["종목코드"].str.zfill(6)
    sc["날짜"] = pd.to_datetime(sc["날짜"])
    sc = sc.sort_values(["종목코드", "날짜"])
    sc["fwd"] = sc.groupby("종목코드")["종가"].shift(-FWD) / sc["종가"] - 1
    # 각 거래일 상위 TOP_N 종목의 평균 선도수익 = 그날 진입했을 때의 전략 수익
    sc["rk"] = sc.groupby("날짜")["종합점수"].rank(ascending=False, method="first")
    strat = sc[sc["rk"] <= TOP_N].groupby("날짜")["fwd"].mean().rename("strat")

    # 시장 자체 30일 선도수익
    mfwd = (df["idx"].shift(-FWD) / df["idx"] - 1).rename("mkt")

    j = pd.concat([df["regime"], strat, mfwd], axis=1).dropna(subset=["regime"])

    # 시장 30일 수익 구간별 확률 (스트레스 시나리오에 '얼마나 흔한 일인가'를 붙이기 위함)
    BUCKETS = [(-99, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 99)]

    def mkt_probs(series):
        v = series.dropna() * 100
        if len(v) < FWD:
            return None
        return [round(float(((v > lo) & (v <= hi)).mean() * 100), 1) for lo, hi in BUCKETS]

    hist = []
    for name in ORDER:
        g = j[j["regime"] == name]
        gv = g.dropna(subset=["strat"])
        n_days = int(len(g))
        if n_days == 0:
            continue
        row = {"name": name, "emoji": REGIME_META[name]["emoji"],
               "color": REGIME_META[name]["color"], "desc": REGIME_META[name]["desc"],
               "days": n_days, "share": round(n_days / len(j) * 100, 1),
               "n": int(len(gv)), "eff_n": round(len(gv) / FWD, 1)}
        if len(gv) >= FWD:      # 독립 관측치 1개 미만이면 통계를 내지 않는다
            r = gv["strat"]
            m = gv["mkt"].dropna()
            row.update(win=round(float((r > 0).mean() * 100), 1),
                       med=round(float(r.median() * 100), 1),
                       avg=round(float(r.mean() * 100), 1),
                       p25=round(_q(r, 25) * 100, 1), p75=round(_q(r, 75) * 100, 1),
                       worst=round(float(r.min() * 100), 1),
                       mkt_med=round(float(m.median() * 100), 1) if len(m) else None,
                       edge=round(float((r.mean() - gv["mkt"].mean()) * 100), 1)
                       if gv["mkt"].notna().any() else None,
                       mkt_probs=mkt_probs(g["mkt"]))
        hist.append(row)

    # ── 국면 전이확률 (t → t+FWD) ─────────────────────────────
    reg = df["regime"].dropna()
    nxt = reg.shift(-FWD)
    trans = {}
    for name in ORDER:
        sub = nxt[(reg == name)].dropna()
        if len(sub) < FWD:
            continue
        vc = sub.value_counts(normalize=True) * 100
        trans[name] = {k: round(float(v), 1) for k, v in vc.items() if v >= 0.05}

    # ── 종목별 베타 / 하락베타 ────────────────────────────────
    tail_ret = ret.tail(BETA_WIN)
    tail_m = mret.tail(BETA_WIN)
    down = tail_m < 0
    betas = {}
    mv = float(tail_m.var())
    mv_d = float(tail_m[down].var())
    for code in tail_ret.columns:
        r = tail_ret[code]
        ok = r.notna() & tail_m.notna()
        if ok.sum() < 60 or not np.isfinite(mv) or mv <= 0:
            continue
        b = float(np.cov(r[ok], tail_m[ok])[0, 1] / mv)
        d = down & ok
        bd = (float(np.cov(r[d], tail_m[d])[0, 1] / mv_d)
              if d.sum() >= 30 and np.isfinite(mv_d) and mv_d > 0 else b)
        vol_ann = float(r[ok].std() * np.sqrt(252) * 100)
        betas[code] = {"beta": round(b, 2), "down": round(bd, 2), "vol": round(vol_ann, 1)}
    # 변동성 백분위 (100종목 중 몇 번째로 출렁이나)
    if betas:
        vs = pd.Series({c: v["vol"] for c, v in betas.items()})
        vr = vs.rank(pct=True) * 100
        for c in betas:
            betas[c]["vol_rank"] = int(round(float(vr[c])))

    # ── 현재 국면 ────────────────────────────────────────────
    last = df.dropna(subset=["regime"]).iloc[-1]
    cur_name = last["regime"]
    # 현재 국면이 며칠째 이어지는가
    streak = 0
    for v in reg.iloc[::-1]:
        if v == cur_name:
            streak += 1
        else:
            break
    since = reg.index[-streak].strftime("%Y-%m-%d") if streak <= len(reg) else ""

    # 지수 최근 120거래일 (차트용)
    tail_idx = df["idx"].tail(120)
    base = float(tail_idx.iloc[0])

    out = {
        "asof": df.index[-1].strftime("%Y-%m-%d"),
        "fwd": FWD, "top_n": TOP_N,
        "current": {
            "name": cur_name,
            "emoji": REGIME_META[cur_name]["emoji"],
            "color": REGIME_META[cur_name]["color"],
            "desc": REGIME_META[cur_name]["desc"],
            "trend": round(float(last["trend"]) * 100, 1),
            "breadth": round(float(last["breadth"]) * 100, 1),
            "vol": round(float(last["vol"]) * 100, 1),
            "dd": round(float(last["dd"]) * 100, 1),
            "streak": int(streak), "since": since,
        },
        "history": hist,
        "transition": trans,
        "betas": betas,
        "index": [{"date": d.strftime("%m/%d"), "v": round(float(v) / base * 100, 1)}
                  for d, v in tail_idx.items()],
        "mkt_med_all": round(float(mfwd.dropna().median() * 100), 1),
        "mkt_buckets": [f"{lo}~{hi}%" for lo, hi in BUCKETS],
        "mkt_probs_all": mkt_probs(mfwd),
        "mkt_probs_now": next((h.get("mkt_probs") for h in hist if h["name"] == cur_name), None),
    }
    os.makedirs("data", exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, allow_nan=False)

    c = out["current"]
    print(f"국면: {c['emoji']} {c['name']} ({c['streak']}거래일째, {c['since']}~) · "
          f"200일선대비 {c['trend']:+.1f}% · breadth {c['breadth']:.0f}% · 변동성 {c['vol']:.0f}%")
    for h in hist:
        if "win" in h:
            print(f"  {h['name']}: 승률 {h['win']}% 중앙 {h['med']:+}% "
                  f"(독립표본 ≈{h['eff_n']}, 시장대비 {h['edge']:+}%p)")
        else:
            print(f"  {h['name']}: 표본부족({h['n']}일)")


if __name__ == "__main__":
    main()
