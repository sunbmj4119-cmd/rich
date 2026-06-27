"""
점수 엔진 v2 - 논리 팩터 강화 버전
- 정규화: 각 거래일에 종목 간 백분위(0~100) = cross-sectional rank
- 미래참조 차단: 재무는 '사용가능일' 기준 as-of join (backward)
- 가치 팩터: PER = 종가/EPS, PBR = 종가/BPS (낮을수록 고득점). 결측은 중립 50.

[v2 변경점]
- 모멘텀: 기존 20/60일 + 12-1개월 모멘텀(252~21일, 단기반전 제거) 추가
- 성장: 매출성장 + 이익성장(영업이익·순이익 전년동기) 통합
- 수익성: ROE + 영업이익률 + ROE개선추세(직전분기 대비)
- 거래량추세: '당일급증'(노이즈) 대신 '장기 거래대금 추세'를 논리로 사용
- 감성점수: 노이즈로 확인되어 비중 0 기본. weights에서 조정 가능.

- 결과: data/scores.csv
- 가중치: config/weights.yaml
"""
import os
import pandas as pd
import numpy as np

PRICES = "data/prices.csv"
FIN = "data/financials.csv"
OUT = "data/scores.csv"
WEIGHTS = "config/weights.yaml"

# 기본 가중치 (v2: 논리 6팩터, 감성 비중 낮음)
DEFAULT_W = {
    "logic_weight": 0.9, "emotion_weight": 0.1,
    "logic": {"momentum": 0.20, "value": 0.20, "profit": 0.20,
              "stability": 0.10, "growth": 0.15, "vtrend": 0.15},
    "emotion": {"volume": 0.5, "volatility": 0.5},
}


def load_weights():
    if os.path.exists(WEIGHTS):
        try:
            import yaml
            with open(WEIGHTS, encoding="utf-8") as f:
                w = yaml.safe_load(f)
            w.setdefault("logic", {})
            for k, v in DEFAULT_W["logic"].items():
                w["logic"].setdefault(k, v)
            w.setdefault("emotion", DEFAULT_W["emotion"])
            return w
        except Exception as e:
            print(f"가중치 로드 실패, 기본값 사용: {e}")
    return DEFAULT_W


def xs_rank(s):
    """그날 종목간 백분위 0~100"""
    return s.rank(pct=True) * 100


def main():
    w = load_weights()
    lw, ew = w["logic_weight"], w["emotion_weight"]

    # 1) 가격 데이터 + 가격기반 지표
    df = pd.read_csv(PRICES, dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].str.zfill(6)
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values(["종목코드", "날짜"]).reset_index(drop=True)

    parts = []
    for code, g in df.groupby("종목코드"):
        g = g.sort_values("날짜").copy()
        g["mom20"] = g["종가"].pct_change(20)
        g["mom60"] = g["종가"].pct_change(60)
        # 12-1개월 모멘텀: 21일 전 종가 대비 252일 전 종가 (최근 1개월 제외)
        g["mom12_1"] = g["종가"].shift(21) / g["종가"].shift(252) - 1
        g["volatility"] = g["종가"].pct_change().rolling(20).std()
        # 거래대금 추세: 60일 평균 거래대금 / 120일 평균 거래대금 (장기 관심 증가)
        g["거래대금"] = g["종가"] * g["거래량"]
        g["vtrend"] = (g["거래대금"].rolling(60).mean()
                       / g["거래대금"].rolling(120).mean())
        # 당일 거래량 급증(기존 감성용, 호환 위해 유지)
        g["vol_ratio"] = g["거래량"] / g["거래량"].rolling(20).mean()
        parts.append(g)
    df = pd.concat(parts)

    # 2) 재무 as-of join
    has_fin = os.path.exists(FIN)
    fin_cols = ["ROE", "영업이익률", "부채비율", "매출성장률",
                "영업이익성장률", "순이익성장률", "ROE개선", "EPS", "BPS"]
    if has_fin:
        fin = pd.read_csv(FIN, dtype={"종목코드": str})
        fin["종목코드"] = fin["종목코드"].str.zfill(6)
        fin["사용가능일"] = pd.to_datetime(fin["사용가능일"], errors="coerce")
        fin = fin.dropna(subset=["사용가능일"])
        for c in ["EPS", "BPS"]:
            if c not in fin.columns:
                fin[c] = np.nan

        # 전년동기 대비 성장률 (같은 분기끼리)
        fin = fin.sort_values(["종목코드", "연도", "분기"])
        fin["매출성장률"] = fin.groupby(["종목코드", "분기"])["매출액"].pct_change() * 100
        fin["영업이익성장률"] = fin.groupby(["종목코드", "분기"])["영업이익"].pct_change() * 100
        fin["순이익성장률"] = fin.groupby(["종목코드", "분기"])["당기순이익"].pct_change() * 100
        # ROE 개선: 직전 분기 대비 ROE 변화 (추세)
        fin["ROE개선"] = fin.groupby("종목코드")["ROE"].diff()
        fin = fin.sort_values("사용가능일")

        merged = []
        for code, g in df.groupby("종목코드"):
            g = g.sort_values("날짜")
            f = fin[fin["종목코드"] == code].sort_values("사용가능일")
            if f.empty:
                for c in fin_cols:
                    g[c] = np.nan
            else:
                g = pd.merge_asof(
                    g, f[["사용가능일"] + fin_cols],
                    left_on="날짜", right_on="사용가능일", direction="backward",
                )
            merged.append(g)
        df = pd.concat(merged)
    else:
        for c in fin_cols:
            df[c] = np.nan
        print("재무 데이터 없음 - 가격 기반 점수만 계산")

    # 2-b) PER / PBR
    df["PER"] = np.where(df["EPS"] > 0, df["종가"] / df["EPS"], np.nan)
    df["PBR"] = np.where(df["BPS"] > 0, df["종가"] / df["BPS"], np.nan)

    # 2-c) 외국인 수급 (검증된 신규 팩터: valid IC +0.055, 모멘텀과 독립)
    #     외국인이 최근 20일 순매수한 종목 = 향후 강세. 시총 대비 정규화.
    FLOWS = "data/flows.csv"
    MCAP = "data/marketcap.csv"
    if os.path.exists(FLOWS):
        fldf = pd.read_csv(FLOWS, dtype={"종목코드": str})
        fldf["종목코드"] = fldf["종목코드"].str.zfill(6)
        fldf["날짜"] = pd.to_datetime(fldf["날짜"])
        fldf = fldf.sort_values(["종목코드", "날짜"])
        # 20일 누적 외국인 순매수
        fldf["외국인20"] = fldf.groupby("종목코드")["외국인순매수"].transform(
            lambda x: x.rolling(20, min_periods=10).sum())
        # 시총 정규화 (있으면)
        if os.path.exists(MCAP):
            mcdf = pd.read_csv(MCAP, dtype={"종목코드": str})
            mcdf["종목코드"] = mcdf["종목코드"].str.zfill(6)
            mcdf["날짜"] = pd.to_datetime(mcdf["날짜"])
            fldf = fldf.merge(mcdf[["날짜", "종목코드", "시가총액"]],
                              on=["날짜", "종목코드"], how="left")
            fldf["외국인강도"] = fldf["외국인20"] / fldf["시가총액"]
        else:
            fldf["외국인강도"] = fldf["외국인20"]
        df = df.merge(fldf[["날짜", "종목코드", "외국인강도"]],
                      on=["날짜", "종목코드"], how="left")
    else:
        df["외국인강도"] = np.nan
        print("수급 데이터(flows.csv) 없음 - 외국인 팩터 건너뜀")

    # 3) 백분위 정규화
    g = df.groupby("날짜")

    # 모멘텀: 20일·60일·12-1개월 평균
    df["s_mom"] = (g["mom20"].transform(xs_rank).fillna(50) * 0.3
                   + g["mom60"].transform(xs_rank).fillna(50) * 0.3
                   + g["mom12_1"].transform(xs_rank).fillna(50) * 0.4)

    # 수익성: ROE + 영업이익률 + ROE개선추세
    df["s_profit"] = (g["ROE"].transform(xs_rank).fillna(50) * 0.4
                      + g["영업이익률"].transform(xs_rank).fillna(50) * 0.3
                      + g["ROE개선"].transform(xs_rank).fillna(50) * 0.3)

    df["s_stab"] = (100 - g["부채비율"].transform(xs_rank)).fillna(50)

    # 성장: 매출 + 영업이익 + 순이익 성장률
    df["s_grow"] = (g["매출성장률"].transform(xs_rank).fillna(50) * 0.34
                    + g["영업이익성장률"].transform(xs_rank).fillna(50) * 0.33
                    + g["순이익성장률"].transform(xs_rank).fillna(50) * 0.33)

    # 가치
    s_per = (100 - g["PER"].transform(xs_rank)).fillna(50)
    s_pbr = (100 - g["PBR"].transform(xs_rank)).fillna(50)
    df["s_value"] = s_per * 0.5 + s_pbr * 0.5

    # 거래대금 추세 (신규 논리 팩터)
    df["s_vtrend"] = g["vtrend"].transform(xs_rank).fillna(50)

    # 외국인 수급 (검증된 신규 팩터: 외국인 매수강도 높을수록 高점수)
    df["s_flow"] = g["외국인강도"].transform(xs_rank).fillna(50)

    # 감성(호환용): 당일거래량급증 + 저변동성
    df["s_vol"] = g["vol_ratio"].transform(xs_rank).fillna(50)
    df["s_volat"] = (100 - g["volatility"].transform(xs_rank)).fillna(50)

    # 4) 합성
    L = w["logic"]
    E = w["emotion"]
    df["논리점수"] = (df["s_mom"] * L["momentum"]
                  + df["s_value"] * L["value"]
                  + df["s_profit"] * L["profit"]
                  + df["s_stab"] * L["stability"]
                  + df["s_grow"] * L["growth"]
                  + df["s_vtrend"] * L.get("vtrend", 0.0)
                  + df["s_flow"] * L.get("flow", 0.0))
    df["감성점수"] = df["s_vol"] * E["volume"] + df["s_volat"] * E["volatility"]
    df["종합점수"] = df["논리점수"] * lw + df["감성점수"] * ew

    # 5) 저장
    out = df.dropna(subset=["s_mom"]).copy()
    cols = ["날짜", "종목코드", "종목명", "종가",
            "논리점수", "감성점수", "종합점수",
            "s_mom", "s_value", "s_profit", "s_stab", "s_grow", "s_vtrend",
            "s_flow", "s_vol", "s_volat"]
    out = out[cols]
    out["날짜"] = out["날짜"].dt.strftime("%Y-%m-%d")
    for c in cols[4:]:
        out[c] = out[c].round(1)

    os.makedirs("data", exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"점수 저장: {len(out)}행, {out['종목코드'].nunique()}종목, "
          f"{out['날짜'].nunique()}일")


if __name__ == "__main__":
    main()
