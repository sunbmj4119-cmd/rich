"""
신규 팩터 검증기 - 수집한 수급/시세 데이터가 노이즈인지 신호인지 판정
== 반드시 score.py에 추가하기 전에 실행 ==

검증 항목:
  1) IC (정보계수): 팩터와 미래수익의 일별 상관 평균 (train/valid 구분)
  2) 기존 팩터와의 상관: 모멘텀 등과 겹치면 추가가치 없음
  3) 5분위 단조성: 팩터 상위그룹이 정말 더 오르나
  4) 부호 일관성: train과 valid에서 같은 방향인가

판정:
  ✅ 채택: valid IC 절댓값 > 0.02, 기존팩터 상관 < 0.5, 부호 일관
  ❌ 노이즈: valid IC 절댓값 < 0.02
  ⚠️ 중복: 기존팩터 상관 > 0.5 (추가가치 낮음)

사용: python src/verify_factor.py
"""
import os
import pandas as pd
import numpy as np

PRICES = "data/prices.csv"
SCORES = "data/scores.csv"
FLOWS = "data/flows.csv"
OHLCV = "data/ohlcv.csv"
MCAP = "data/marketcap.csv"

FWD = 40  # 미래수익 평가 기간


def daily_ic(df, fcol, ycol="fwd"):
    """일별 IC의 평균"""
    def _ic(x):
        if len(x) < 10:
            return np.nan
        return x[fcol].corr(x[ycol], method="spearman")
    return df.groupby("날짜")[[fcol, ycol]].apply(_ic).mean()


def evaluate_factor(merged, fcol, name, existing_cols):
    """한 팩터의 IC·상관·단조성 평가"""
    print(f"\n{'='*60}")
    print(f"  팩터 검증: {name}")
    print(f"{'='*60}")

    res = {}
    for label, (a, b) in {"train(18-23)": ("2018-01-01", "2023-12-31"),
                          "valid(24-25)": ("2024-01-01", "2025-12-31")}.items():
        sub = merged[(merged["날짜"] >= a) & (merged["날짜"] <= b)].dropna(subset=[fcol, "fwd"])
        if len(sub) < 100:
            print(f"  {label}: 데이터 부족 ({len(sub)}행)")
            res[label] = np.nan
            continue
        ic = daily_ic(sub, fcol)
        res[label] = ic
        # 5분위 단조성
        try:
            sub = sub.copy()
            sub["q"] = sub.groupby("날짜")[fcol].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
            q_ret = sub.groupby("q")["fwd"].mean()
            spread = q_ret.iloc[-1] - q_ret.iloc[0]
            mono = q_ret.is_monotonic_increasing or q_ret.is_monotonic_decreasing
            print(f"  {label}: IC={ic:+.3f} | 5분위스프레드={spread*100:+.1f}% | "
                  f"단조={'✅' if mono else '✗'}")
        except Exception:
            print(f"  {label}: IC={ic:+.3f}")

    # 기존 팩터와 상관
    print(f"\n  [기존 팩터와 상관관계]")
    v = merged[merged["날짜"] >= "2024-01-01"].dropna(subset=[fcol])
    max_corr = 0
    for ec in existing_cols:
        if ec in v.columns:
            corr = v[fcol].corr(v[ec])
            if abs(corr) > abs(max_corr):
                max_corr = corr
            flag = "⚠️중복" if abs(corr) > 0.5 else ""
            print(f"    vs {ec}: {corr:+.2f} {flag}")

    # 최종 판정
    vic = res.get("valid(24-25)", np.nan)
    tic = res.get("train(18-23)", np.nan)
    print(f"\n  ▶ 판정: ", end="")
    if pd.isna(vic) or abs(vic) < 0.02:
        print("❌ 노이즈 (valid IC 너무 약함) — 추가 비권장")
    elif abs(max_corr) > 0.6:
        print(f"⚠️ 중복 (기존팩터와 상관 {max_corr:+.2f}) — 추가가치 낮음")
    elif not pd.isna(tic) and np.sign(tic) != np.sign(vic) and abs(tic) > 0.02:
        print(f"⚠️ 부호 불일치 (train {tic:+.3f} vs valid {vic:+.3f}) — 불안정")
    else:
        print(f"✅ 채택 권장 (valid IC {vic:+.3f}, 기존과 독립적)")


def main():
    px = pd.read_csv(PRICES, dtype={"종목코드": str})
    px["종목코드"] = px["종목코드"].str.zfill(6)
    px["날짜"] = pd.to_datetime(px["날짜"])
    px = px.sort_values(["종목코드", "날짜"])
    # 미래수익
    px["fwd"] = px.groupby("종목코드")["종가"].shift(-FWD) / px["종가"] - 1

    # 기존 점수 팩터 (상관 비교용)
    sc = pd.read_csv(SCORES, dtype={"종목코드": str})
    sc["종목코드"] = sc["종목코드"].str.zfill(6)
    sc["날짜"] = pd.to_datetime(sc["날짜"])
    existing = ["s_mom", "s_value", "s_profit", "s_grow"]
    base = px[["날짜", "종목코드", "종가", "fwd"]].merge(
        sc[["날짜", "종목코드"] + existing], on=["날짜", "종목코드"], how="left")

    has_any = False

    # ── 수급 팩터 검증 ──
    if os.path.exists(FLOWS):
        has_any = True
        fl = pd.read_csv(FLOWS, dtype={"종목코드": str})
        fl["종목코드"] = fl["종목코드"].str.zfill(6)
        fl["날짜"] = pd.to_datetime(fl["날짜"])
        m = base.merge(fl, on=["날짜", "종목코드"], how="inner")
        m = m.sort_values(["종목코드", "날짜"])

        # 팩터1: 외국인 20일 누적 순매수 (시총 정규화 전 단순버전)
        m["외국인20"] = m.groupby("종목코드")["외국인순매수"].transform(
            lambda x: x.rolling(20, min_periods=10).sum())
        evaluate_factor(m, "외국인20", "외국인 20일 누적순매수", existing)

        # 팩터2: 기관 20일 누적 (비교용 - 학술상 약할 것)
        m["기관20"] = m.groupby("종목코드")["기관순매수"].transform(
            lambda x: x.rolling(20, min_periods=10).sum())
        evaluate_factor(m, "기관20", "기관 20일 누적순매수", existing)

        # 팩터3: 외국인+기관 동반매수 (스마트머니)
        m["스마트20"] = m["외국인20"] + m["기관20"]
        evaluate_factor(m, "스마트20", "외국인+기관 동반순매수", existing)
    else:
        print("\n⚠️ data/flows.csv 없음 — 먼저 collect_flows.py 실행 필요")

    # ── 시세 보강 팩터 검증 ──
    if os.path.exists(OHLCV):
        has_any = True
        oh = pd.read_csv(OHLCV, dtype={"종목코드": str})
        oh["종목코드"] = oh["종목코드"].str.zfill(6)
        oh["날짜"] = pd.to_datetime(oh["날짜"])
        m2 = base.merge(oh, on=["날짜", "종목코드"], how="inner", suffixes=("", "_o"))
        m2 = m2.sort_values(["종목코드", "날짜"])
        # 진짜 변동성: 고저폭 20일 평균
        m2["고저폭"] = (m2["고가"] - m2["저가"]) / m2["종가"]
        m2["변동성20"] = m2.groupby("종목코드")["고저폭"].transform(
            lambda x: x.rolling(20, min_periods=10).mean())
        evaluate_factor(m2, "변동성20", "고저 변동성(저변동 효과)", existing)

    # ── 공매도 팩터 검증 (백필 후 유효; 표본 부족 시 자동 스킵) ──
    SHORTS = "data/shorts.csv"
    if os.path.exists(SHORTS):
        sh = pd.read_csv(SHORTS, dtype={"종목코드": str})
        sh["종목코드"] = sh["종목코드"].str.zfill(6)
        sh["날짜"] = pd.to_datetime(sh["날짜"], errors="coerce")
        if "공매도잔고비중" in sh.columns:
            has_any = True
            m3 = base.merge(sh[["날짜", "종목코드", "공매도잔고비중"]],
                            on=["날짜", "종목코드"], how="inner").dropna(subset=["공매도잔고비중"])
            # 약세신호: 잔고비중↑ → 향후수익↓. 부호 반전해 '높을수록 好' 팩터로 평가.
            m3["공매도역"] = -m3["공매도잔고비중"]
            evaluate_factor(m3, "공매도역", "공매도잔고비중 반전(낮을수록 好)", existing)

    if not has_any:
        print("\n수집된 데이터가 없습니다. 먼저 collect_flows.py를 실행하세요.")


if __name__ == "__main__":
    main()

