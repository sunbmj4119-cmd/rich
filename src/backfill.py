import os
import time
import pandas as pd
from datetime import datetime
from pykrx import stock

os.makedirs("data", exist_ok=True)
START = "20180101"
END = datetime.now().strftime("%Y%m%d")
OUT = "data/prices.csv"

# 1) 코스피200 구성종목
today = stock.get_nearest_business_day_in_a_week()
tickers = stock.get_index_portfolio_deposit_file("1028")  # 1028 = 코스피200
names = {t: stock.get_market_ticker_name(t) for t in tickers}
print(f"코스피200 종목 수: {len(tickers)}")

# 2) 이미 받은 거 이어받기
done_codes = set()
if os.path.exists(OUT):
    old = pd.read_csv(OUT, dtype={"종목코드": str})
    done_codes = set(old["종목코드"].unique())
    print(f"이미 받은 종목: {len(done_codes)}개 (이어받기)")

# 3) 종목별 8년치 수집
buffer = []
for i, code in enumerate(tickers):
    if code in done_codes:
        continue
    try:
        ohlcv = stock.get_market_ohlcv(START, END, code)
        fund = stock.get_market_fundamental(START, END, code)
        cap = stock.get_market_cap(START, END, code)

        df = ohlcv[["종가", "거래량"]].copy()
        df["시가총액"] = cap["시가총액"]
        df["PER"] = fund["PER"]
        df["PBR"] = fund["PBR"]
        df["종목코드"] = code
        df["종목명"] = names[code]
        df = df.reset_index().rename(columns={"날짜": "날짜"})
        df["날짜"] = df["날짜"].dt.strftime("%Y-%m-%d")
        buffer.append(df)

        print(f"[{i+1}/{len(tickers)}] {names[code]} {len(df)}줄")
        time.sleep(0.5)

        # 20종목마다 중간 저장 (끊겨도 보존)
        if len(buffer) >= 20:
            _flush(buffer, OUT)
            buffer = []

    except Exception as e:
        print(f"{code} 실패: {e}")
        time.sleep(1)

# 남은 거 저장
if buffer:
    _flush(buffer, OUT)

print("백필 완료")
