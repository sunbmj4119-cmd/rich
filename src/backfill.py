"""
가격 백필 (1회용) - 코스피200 8년치 종가/거래량/시총/PER/PBR
- 종목 리스트 받기를 날짜 지정 + 재시도로 견고하게 처리
- 이어받기: 이미 받은 종목은 건너뜀 (끊겨도 다시 돌리면 이어서)
"""
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pykrx import stock

os.makedirs("data", exist_ok=True)
START = "20180101"
END = datetime.now().strftime("%Y%m%d")
OUT = "data/prices.csv"
COLS = ["날짜", "종목코드", "종목명", "종가", "거래량", "시가총액", "PER", "PBR"]


def recent_bday():
    """오늘 기준 최근 영업일 yyyymmdd (주말/공휴일 보정)"""
    d = datetime.now()
    for _ in range(10):
        ymd = d.strftime("%Y%m%d")
        try:
            df = stock.get_index_ohlcv(ymd, ymd, "1028")  # 코스피200 지수
            if not df.empty:
                return ymd
        except Exception:
            pass
        d -= timedelta(days=1)
    return END


def get_kospi200(base_date):
    """코스피200 구성종목 - 날짜 지정 + 재시도"""
    for attempt in range(5):
        try:
            tickers = stock.get_index_portfolio_deposit_file("1028", base_date)
            if tickers:
                return tickers
        except Exception as e:
            print(f"  종목리스트 재시도 {attempt+1}: {e}")
        time.sleep(2)
    print("  코스피200 직접 수신 실패 → 시총 상위 200 대체")
    cap = stock.get_market_cap(base_date, market="KOSPI")
    return cap.sort_values("시가총액", ascending=False).head(200).index.tolist()


def flush(buffer):
    if not buffer:
        return
    df = pd.concat(buffer)[COLS]
    header = not os.path.exists(OUT)
    df.to_csv(OUT, mode="a", header=header, index=False)


def main():
    base = recent_bday()
    print(f"기준 영업일: {base}")

    tickers = get_kospi200(base)
    names = {t: stock.get_market_ticker_name(t) for t in tickers}
    print(f"코스피200 종목 수: {len(tickers)}")

    done = set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"종목코드": str})
        done = set(old["종목코드"].unique())
        print(f"이미 받은 종목: {len(done)}개 (이어받기)")

    buffer = []
    for i, code in enumerate(tickers):
        if code in done:
            continue
        try:
            ohlcv = stock.get_market_ohlcv(START, END, code)
            if ohlcv.empty:
                print(f"  {names[code]}: 데이터 없음")
                continue
            fund = stock.get_market_fundamental(START, END, code)
            cap = stock.get_market_cap(START, END, code)

            df = ohlcv[["종가", "거래량"]].copy()
            df["시가총액"] = cap["시가총액"]
            df["PER"] = fund["PER"]
            df["PBR"] = fund["PBR"]
            df["종목코드"] = code
            df["종목명"] = names[code]
            df = df.reset_index()
            df["날짜"] = df["날짜"].dt.strftime("%Y-%m-%d")
            buffer.append(df)

            print(f"[{i+1}/{len(tickers)}] {names[code]} {len(df)}줄")
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
