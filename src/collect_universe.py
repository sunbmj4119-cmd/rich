"""
생존편향 데이터 수집 (무료) — 현실적 버전
  과거 시점 구성종목은 FDR/pykrx 둘 다 막힘(KRX 인증·날짜제한).
  대안:
   1) 현재 KOSPI200 구성종목만 FDR로 수집 → 매년 누적하면 진짜 이력 생성
   2) 상장폐지 종목 전체(KRX-DELISTING) → 생존편향 정량화의 핵심 (이건 잘 됨)
   3) 과거 구성종목이 꼭 필요하면 수동 Excel(아래 안내) → data/universe_manual.csv 로 올리면 자동 병합
"""
import os, time
import pandas as pd

UNIV = "data/universe.csv"
MANUAL = "data/universe_manual.csv"
DELIST = "data/delisted.csv"


def today_kospi200():
    """오늘 현재 KOSPI200 구성종목 (FDR StockListing 사용 — 인증불필요)"""
    import FinanceDataReader as fdr
    # KRX 전체에서 KOSPI200 편입 여부 컬럼 활용 시도
    for src in ["KOSPI200", "KRX/INDEX/STOCK/1028"]:
        try:
            if src == "KOSPI200":
                df = fdr.StockListing("KOSPI")  # 전체 코스피
                # 시총 상위 200 근사 (정확한 편입표 없을 때 폴백)
                if "Marcap" in df.columns:
                    df = df.sort_values("Marcap", ascending=False).head(200)
                codes = df["Code"].astype(str).str.zfill(6).tolist() if "Code" in df.columns else \
                        df["Symbol"].astype(str).str.zfill(6).tolist()
            else:
                df = fdr.SnapDataReader(src)
                col = "Code" if "Code" in df.columns else df.columns[0]
                codes = df[col].astype(str).str.zfill(6).tolist()
            if codes:
                return codes
        except Exception as e:
            print(f"  {src} 실패: {str(e)[:60]}")
    return []


def main():
    os.makedirs("data", exist_ok=True)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    # 1) 현재 구성종목 스냅샷 (누적)
    print("■ 현재 KOSPI200 구성종목 수집")
    codes = []
    try:
        codes = today_kospi200()
    except ImportError:
        print("  FinanceDataReader 미설치")
    rows = [{"날짜": today, "종목코드": c} for c in codes]
    print(f"  {today}: {len(codes)}종목")

    # 기존 universe + 수동 병합
    frames = []
    if os.path.exists(UNIV):
        frames.append(pd.read_csv(UNIV, dtype={"종목코드": str}))
    if rows:
        frames.append(pd.DataFrame(rows))
    if os.path.exists(MANUAL):
        m = pd.read_csv(MANUAL, dtype={"종목코드": str})
        frames.append(m)
        print(f"  수동 파일 병합: {MANUAL} ({m['날짜'].nunique()}스냅샷)")
    if frames:
        df = pd.concat(frames).drop_duplicates(["날짜", "종목코드"])
        df["종목코드"] = df["종목코드"].str.zfill(6)
        df.to_csv(UNIV, index=False)
        print(f"  유니버스 저장: {UNIV} ({df['날짜'].nunique()}스냅샷, 고유 {df['종목코드'].nunique()}종목)")
    else:
        print("  ⚠️ 구성종목 수집 실패 — 수동 Excel 안내 참고")

    # 2) 상장폐지 종목 (생존편향 실체)
    print("\n■ 상장폐지 종목 수집")
    try:
        import FinanceDataReader as fdr
        d = fdr.StockListing("KRX-DELISTING")
        d.to_csv(DELIST, index=False)
        print(f"  저장: {DELIST} ({len(d)}건)")
    except Exception as e:
        print(f"  실패: {str(e)[:60]}")


if __name__ == "__main__":
    main()
