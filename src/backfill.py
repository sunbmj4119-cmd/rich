"""
가격 백필 (1회용) - 코스피 주요종목 8년치 종가/거래량/시총/PER/PBR
- 종목 리스트: FDR(StockListing) 우선 → 실패시 하드코딩 폴백
- 개별종목 시계열(get_market_ohlcv 등)은 로그인 없이 동작 확인됨
- 이어받기: 이미 받은 종목 건너뜀
"""
import os
import time
import pandas as pd
from datetime import datetime
from pykrx import stock

os.makedirs("data", exist_ok=True)
START = "20180101"
END = datetime.now().strftime("%Y%m%d")
OUT = "data/prices.csv"
COLS = ["날짜", "종목코드", "종목명", "종가", "거래량", "시가총액", "PER", "PBR"]
TARGET_N = 200  # 시총 상위 N

# 폴백용 코스피 대형주 코드 (KRX/FDR 모두 막힐 때 최소 동작 보장)
FALLBACK = [
    "005930","000660","373220","207940","005380","000270","005490","068270",
    "035420","051910","006400","028260","105560","055550","012330","032830",
    "066570","003670","096770","015760","034730","017670","086790","033780",
    "009150","011200","010130","024110","316140","138040","259960","267260",
    "010950","018260","047810","051900","090430","000810","021240","011170",
]


def get_universe():
    """코스피 시총 상위 N 종목코드 - FDR 우선, 실패시 폴백"""
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KOSPI")
        # 시가총액 컬럼명이 'Marcap' 또는 'MarketCap' 등으로 올 수 있음
        capcol = None
        for c in ["Marcap", "MarketCap", "Market Cap", "시가총액"]:
            if c in df.columns:
                capcol = c
                break
        codecol = "Code" if "Code" in df.columns else "Symbol"
        if capcol:
            df = df.dropna(subset=[capcol]).sort_values(capcol, ascending=False)
            codes = df[codecol].astype(str).str.zfill(6).head(TARGET_N).tolist()
            if codes:
                print(f"FDR 종목 수신: {len(codes)}개")
                return codes
        # 시총 컬럼 없으면 코드만이라도
        codes = df[codecol].astype(str).str.zfill(6).head(TARGET_N).tolist()
        if codes:
            print(f"FDR 종목(시총없음): {len(codes)}개")
            return codes
    except Exception as e:
        print(f"FDR 실패: {e}")
    print(f"폴백 리스트 사용: {len(FALLBACK)}개")
    return FALLBACK


def flush(buffer):
    if not buffer:
        return
    df = pd.concat(buffer)[COLS]
    header = not os.path.exists(OUT)
    df.to_csv(OUT, mode="a", header=header, index=False)


def main():
    tickers = get_universe()

    # 종목명 (개별 조회는 동작함)
    names = {}
    for t in tickers:
        try:
            names[t] = stock.get_market_ticker_name(t)
        except Exception:
            names[t] = t

    done = set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"종목코드": str})
        done = set(old["종목코드"].astype(str).str.zfill(6).unique())
        print(f"이미 받은 종목: {len(done)}개 (이어받기)")

    buffer = []
    for i, code in enumerate(tickers):
        code = str(code).zfill(6)
        if code in done:
            continue
        try:
            ohlcv = stock.get_market_ohlcv(START, END, code)
            if ohlcv.empty:
                print(f"  {names.get(code,code)}: 데이터 없음")
                continue

            df = ohlcv[["종가", "거래량"]].copy()
            # 재무/시총은 개별 시계열로 받기 (전종목 스냅샷 회피)
            try:
                fund = stock.get_market_fundamental(START, END, code)
                df["PER"] = fund["PER"]
                df["PBR"] = fund["PBR"]
            except Exception:
                df["PER"] = None
                df["PBR"] = None
            try:
                cap = stock.get_market_cap(START, END, code)
                df["시가총액"] = cap["시가총액"]
            except Exception:
                df["시가총액"] = None

            df["종목코드"] = code
            df["종목명"] = names.get(code, code)
            df = df.reset_index()
            df["날짜"] = df["날짜"].dt.strftime("%Y-%m-%d")
            df = df[COLS]
            buffer.append(df)

            print(f"[{i+1}/{len(tickers)}] {names.get(code,code)} {len(df)}줄")
            time.sleep(0.5)

            if len(buffer) >= 20:
                flush(buffer)
                buffer = []
        except Exception as e:
            print(f"  {code} 실패: {e}")
            time.sleep(1)

    if buffer:
        flush(buffer)
    print("가격 백필 완료")


if __name__ == "__main__":
    main()

