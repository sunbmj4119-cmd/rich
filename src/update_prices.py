"""
매일 주가 갱신 - prices.csv 에 최신 종가 추가 (100종목)
- 기존 prices.csv 종목 리스트 유지
- 마지막 날짜 이후 ~ 오늘까지의 일봉만 받아서 append (중복 방지)
- 장 안 열린 날/데이터 없으면 그냥 넘어감
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from pykrx import stock

PRICES = "data/prices.csv"


def main():
    df = pd.read_csv(PRICES, dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].str.zfill(6)
    uni = df[["종목코드", "종목명"]].drop_duplicates()
    last_date = pd.to_datetime(df["날짜"]).max()
    start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
    today = datetime.now().strftime("%Y%m%d")

    if start > today:
        print("이미 최신 — 추가 없음")
        return

    existing = set(zip(df["날짜"].astype(str), df["종목코드"]))
    rows = []
    for code, name in zip(uni["종목코드"], uni["종목명"]):
        try:
            o = stock.get_market_ohlcv(start, today, code)
            if o.empty:
                continue
            for idx, r in o.iterrows():
                d = idx.strftime("%Y-%m-%d")
                if (d, code) in existing:
                    continue
                rows.append({
                    "날짜": d, "종목코드": code, "종목명": name,
                    "종가": int(r["종가"]), "거래량": int(r["거래량"]),
                    "시가총액": "", "PER": "", "PBR": "",
                })
        except Exception as e:
            print(f"  {name} 실패: {e}")

    if not rows:
        print("추가할 신규 데이터 없음 (휴장일 등)")
        return

    add = pd.DataFrame(rows)
    add.to_csv(PRICES, mode="a", header=False, index=False)
    print(f"주가 갱신: {len(rows)}행 추가 ({add['날짜'].min()}~{add['날짜'].max()})")


if __name__ == "__main__":
    main()

