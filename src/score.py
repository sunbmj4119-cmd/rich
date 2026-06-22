"""
점수 엔진 - 가격(필수) + 재무(있으면) 합쳐 종합점수 산출
- 정규화: 각 거래일에 종목 간 백분위(0~100) = cross-sectional rank
- 미래참조 차단: 재무는 '사용가능일' 기준 as-of join (backward)
- 가치 팩터: PER = 종가/EPS, PBR = 종가/BPS (낮을수록 고득점). 결측은 중립 50.
- 결과: data/scores.csv (전 기간 전 종목 점수)
- 가중치: config/weights.yaml (없으면 기본값)
"""
import os
import pandas as pd
import numpy as np

PRICES = "data/prices.csv"
FIN = "data/financials.csv"
OUT = "data/scores.csv"
WEIGHTS = "config/weights.yaml"

# 기본 가중치 (value 포함)
DEFAULT_W = {
    "logic_weight": 0.6, "emotion_weight": 0.4,
    "logic": {"momentum": 0.25, "value": 0.20, "profit": 0.25,
              "stability": 0.15, "growth": 0.15},
    "emotion": {"volume": 0.5, "volatility": 0.5},
}


def load_weights():
    if os.path.exists(WEIGHTS):
        try:
            import yaml
            with open(WEIGHTS, encoding="utf-8") as f:
                w = yaml.safe_load(f)
            # value 키 누락 방어
            w.setdefault("logic", {})
            for k, v in DEFAULT_W["logic"].items():
                w["logic"].setdefault(k, v)
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
        g["vol_ratio"] = g["거래량"] / g["거래량"].rolling(20).mean()
        g["volatility"] = g["종가"].pct_change().rolling(20).std()
        parts.append(g)
    df = pd.concat(parts)

    # 2) 재무 as-of join (있으면)
    has_fin = os.path.exists(FIN)
    fin_cols = ["ROE", "영업이익률", "부채비율", "매출성장률", "EPS", "BPS"]
    if has_fin:
        fin = pd.read_csv(FIN, dtype={"종목코드": str})
        fin["종목코드"] = fin["종목코드"].str.zfill(6)
        fin["사용가능일"] = pd.to_datetime(fin["사용가능일"], errors="coerce")
        fin = fin.dropna(subset=["사용가능일"])

        # EPS/BPS 없는 옛 파일 방어
        for c in ["EPS", "BPS"]:
            if c not in fin.columns:
                fin[c] = np.nan

        # 매출성장률(전년동기)
        fin = fin.sort_values(["종목코드", "연도", "분기"])
        fin["매출성장률"] = fin.groupby(["종목코드", "분기"])["매출액"].pct_change() * 100
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

    # 2-b) PER / PBR 계산 (낮을수록 저평가 = 고득점)
    #  EPS/BPS 양수일 때만 유효. 적자/결손(EPS<=0)·결측은 NaN → 중립 50.
    df["PER"] = np.where(df["EPS"] > 0, df["종가"] / df["EPS"], np.nan)
    df["PBR"] = np.where(df["BPS"] > 0, df["종가"] / df["BPS"], np.nan)

    # 3) 그날 종목간 백분위 정규화
    g = df.groupby("날짜")
    df["s_mom"] = g["mom20"].transform(xs_rank) * 0.5 + g["mom60"].transform(xs_rank) * 0.5
    df["s_vol"] = g["vol_ratio"].transform(xs_rank)
    df["s_volat"] = 100 - g["volatility"].transform(xs_rank)  # 낮을수록 高
    df["s_profit"] = (g["ROE"].transform(xs_rank).fillna(50) * 0.5
                      + g["영업이익률"].transform(xs_rank).fillna(50) * 0.5)
    df["s_stab"] = (100 - g["부채비율"].transform(xs_rank)).fillna(50)  # 낮을수록 高
    df["s_grow"] = g["매출성장률"].transform(xs_rank).fillna(50)
    # 가치: PER·PBR 낮을수록 高. 각 (100 - rank), 둘 평균. 결측 중립 50.
    s_per = (100 - g["PER"].transform(xs_rank)).fillna(50)
    s_pbr = (100 - g["PBR"].transform(xs_rank)).fillna(50)
    df["s_value"] = s_per * 0.5 + s_pbr * 0.5

    # 4) 논리/감성 합성
    L = w["logic"]
    E = w["emotion"]
    df["논리점수"] = (df["s_mom"] * L["momentum"]
                  + df["s_value"] * L["value"]
                  + df["s_profit"] * L["profit"]
                  + df["s_stab"] * L["stability"]
                  + df["s_grow"] * L["growth"])
    df["감성점수"] = df["s_vol"] * E["volume"] + df["s_volat"] * E["volatility"]
    df["종합점수"] = df["논리점수"] * lw + df["감성점수"] * ew

    # 5) 저장 (점수 계산 가능한 행만)
    out = df.dropna(subset=["s_mom", "s_vol"]).copy()
    cols = ["날짜", "종목코드", "종목명", "종가",
            "논리점수", "감성점수", "종합점수",
            "s_mom", "s_value", "s_profit", "s_stab", "s_grow", "s_vol", "s_volat"]
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

