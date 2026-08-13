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
MCAP = "data/marketcap.csv"   # 이 시스템이 실제로 관리 중인 유니버스
OUT = "data/scores.csv"
WEIGHTS = "config/weights.yaml"

# 기본 가중치 (v2: 논리 6팩터, 감성 비중 낮음)
DEFAULT_W = {
    "logic_weight": 0.9, "emotion_weight": 0.1,
    "logic": {"momentum": 0.20, "rmom": 0.0, "value": 0.20, "profit": 0.20,
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


def limit_universe(df):
    """점수를 매길 종목을 '데이터가 갖춰진 것'으로 제한한다.

    왜 필요한가
      점수는 그날 종목 간 백분위다. 그래서 **표본에 누가 들어있느냐가 모든 종목의
      점수를 바꾼다**. 재무가 없는 종목을 끼워넣으면 그 종목은 가치·수익성·성장에서
      일괄 중립 50점을 받고(가중 합계 65%), 동시에 나머지 종목의 백분위를 희석시킨다.
      모멘텀 하나만으로 상위 10위에 올라 매수 신호까지 날 수 있다.

      prices.csv는 position_lab 같은 '가격 경로 통계'용으로 종목을 넓혀 받는다.
      그 확장이 매매 신호를 조용히 바꾸지 않도록 여기서 문을 따로 잠근다.

    기준
      시총 데이터가 있는 종목만. 시총은 collect_flows가 매일 유지하는 파일이라
      '이 시스템이 실제로 관리 중인 유니버스'와 같은 뜻이다.
      유니버스를 넓히려면 그 종목들의 재무·시총부터 모은 뒤 이 필터가 자동으로 열린다.
    """
    if not os.path.exists(MCAP):
        return df
    try:
        mc = pd.read_csv(MCAP, dtype={"종목코드": str}, usecols=["종목코드"])
    except Exception:
        return df
    ok = set(mc["종목코드"].str.zfill(6))
    if not ok:
        return df
    have = set(df["종목코드"].unique())
    drop = have - ok
    if drop:
        print(f"유니버스 제한: {len(have)}종목 중 시총·재무가 갖춰진 {len(have & ok)}종목만 채점 "
              f"({len(drop)}종목 제외 — 가격만 있어 백분위를 왜곡시킴)")
    return df[df["종목코드"].isin(ok)].reset_index(drop=True)


def main():
    w = load_weights()
    lw, ew = w["logic_weight"], w["emotion_weight"]

    # 1) 가격 데이터 + 가격기반 지표
    df = pd.read_csv(PRICES, dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].str.zfill(6)
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values(["종목코드", "날짜"]).reset_index(drop=True)
    df = limit_universe(df)

    parts = []
    for code, g in df.groupby("종목코드"):
        g = g.sort_values("날짜").copy()
        g["mom20"] = g["종가"].pct_change(20)
        g["mom60"] = g["종가"].pct_change(60)
        # 12-1개월 모멘텀: 21일 전 종가 대비 252일 전 종가 (최근 1개월 제외)
        g["mom12_1"] = g["종가"].shift(21) / g["종가"].shift(252) - 1
        # 위험조정 모멘텀 — 같은 상승이라도 '덜 출렁이며' 오른 종목을 높게 본다.
        # factor_lab 검증: 원본 모멘텀 IC는 -0.010(음수)인데 변동성으로 나누면 +0.041,
        # 9년 중 7년 양수. 학술적으로도 residual momentum이 raw보다 낫다고 보고됨
        # (Blitz·Huij·Martens 2011). 종합점수와의 상관 +0.23으로 새 정보를 담는다.
        g["_vol60d"] = g["종가"].pct_change().rolling(60, min_periods=30).std()
        g["rmom"] = g["mom12_1"] / (g["_vol60d"] * np.sqrt(252) + 1e-9)
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

    # 2-d) 공매도 잔고비중 (약세 신호: 많이 공매도된 종목은 향후 약세 — Boehmer·Jones·Zhang 2008)
    #      KRX 잔고는 T+2 지연 공시 → 재무처럼 as-of(backward) 조인으로 '최신 가용값' 사용.
    #      데이터는 최근 수집분만 존재 → 과거는 결측(중립 50). weights의 short=0이면 점수 무영향.
    #      백필(Actions) + verify_factor 검증 후 weights에서 short 가중 부여.
    SHORTS = "data/shorts.csv"
    df["공매도잔고비중"] = np.nan
    if os.path.exists(SHORTS):
        shdf = pd.read_csv(SHORTS, dtype={"종목코드": str})
        shdf["종목코드"] = shdf["종목코드"].str.zfill(6)
        shdf["날짜"] = pd.to_datetime(shdf["날짜"], errors="coerce")
        if "공매도잔고비중" in shdf.columns:
            shdf = shdf.dropna(subset=["날짜", "공매도잔고비중"]).sort_values("날짜")
            merged_s = []
            for code, gg in df.groupby("종목코드"):
                gg = gg.sort_values("날짜")
                f = shdf[shdf["종목코드"] == code][["날짜", "공매도잔고비중"]].sort_values("날짜")
                if not f.empty:
                    gg = gg.drop(columns=["공매도잔고비중"])
                    gg = pd.merge_asof(gg, f, on="날짜", direction="backward")
                merged_s.append(gg)
            df = pd.concat(merged_s)

    # 3) 백분위 정규화
    g = df.groupby("날짜")

    # 모멘텀: 20일·60일·12-1개월 평균
    df["s_rmom"] = g["rmom"].transform(xs_rank).fillna(50)
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

    # 공매도 부담 (잔고비중 높을수록 약세 → 낮은 점수). 결측=중립 50.
    df["s_short"] = (100 - g["공매도잔고비중"].transform(xs_rank)).fillna(50)

    # 감성(호환용): 당일거래량급증 + 저변동성
    df["s_vol"] = g["vol_ratio"].transform(xs_rank).fillna(50)
    df["s_volat"] = (100 - g["volatility"].transform(xs_rank)).fillna(50)

    # 4) 합성
    L = w["logic"]
    E = w["emotion"]
    df["논리점수"] = (df["s_rmom"] * L.get("rmom", 0)
                     + df["s_mom"] * L["momentum"]
                  + df["s_value"] * L["value"]
                  + df["s_profit"] * L["profit"]
                  + df["s_stab"] * L["stability"]
                  + df["s_grow"] * L["growth"]
                  + df["s_vtrend"] * L.get("vtrend", 0.0)
                  + df["s_flow"] * L.get("flow", 0.0)
                  + df["s_short"] * L.get("short", 0.0))
    df["감성점수"] = df["s_vol"] * E["volume"] + df["s_volat"] * E["volatility"]
    df["종합점수"] = df["논리점수"] * lw + df["감성점수"] * ew

    # 5) 저장
    out = df.dropna(subset=["s_mom"]).copy()
    cols = ["날짜", "종목코드", "종목명", "종가",
            "논리점수", "감성점수", "종합점수",
            "s_mom", "s_rmom", "s_value", "s_profit", "s_stab", "s_grow", "s_vtrend",
            "s_flow", "s_short", "s_vol", "s_volat"]
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
