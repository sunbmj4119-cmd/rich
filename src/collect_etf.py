"""
지수 ETF 일봉 수집 (pykrx, KRX 로그인 — KRX_ID/KRX_PW 이미 연동됨) → data/etf_ohlcv.csv

  069500 KODEX 200 · 114800 KODEX 인버스
  국내 주식형 ETF 는 매도 때 증권거래세가 없어 단기매매 비용이 주식의 1/5 수준 → 단기 연구·모의투자에 쓴다.

  python src/collect_etf.py --backfill 2018-01-01   # 처음 한 번
  python src/collect_etf.py                          # 매일 (최근 2주 다시 받아 덮어씀 · 파일이 없으면 저절로 2018년부터)
"""
import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

OUT = "data/etf_ohlcv.csv"
TICKERS = {"069500": "KODEX 200", "114800": "KODEX 인버스"}
COLS = ["날짜", "종목코드", "종목명", "시가", "고가", "저가", "종가", "거래량", "거래대금", "NAV"]


def fetch(code, start, end):
    from pykrx import stock
    df = None
    try:
        df = stock.get_etf_ohlcv_by_date(start, end, code)
    except Exception as e:
        print(f"  ETF 전용 조회 실패({code}): {type(e).__name__} {str(e)[:100]} → 일반 시세로")
    if df is None or df.empty:
        df = stock.get_market_ohlcv(start, end, code)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLS)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "날짜"})
    df["날짜"] = pd.to_datetime(df["날짜"]).dt.strftime("%Y-%m-%d")
    df["종목코드"] = code
    df["종목명"] = TICKERS[code]
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df[COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill")
    a = ap.parse_args()
    end = datetime.now().strftime("%Y%m%d")
    old = pd.read_csv(OUT, dtype={"종목코드": str}) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    if not a.backfill and (old.empty or str(old["날짜"].min()) > "2018-01-05"):
        a.backfill = "2018-01-01"                       # 처음이면 저절로 되채우기 (요청 2번이면 끝)
        print("  ETF 일봉이 없어 2018년부터 받습니다")
    start = (a.backfill.replace("-", "") if a.backfill
             else (datetime.now() - timedelta(days=14)).strftime("%Y%m%d"))
    parts = [old]
    for code, name in TICKERS.items():
        df = fetch(code, start, end)
        print(f"  {name}: {len(df)}행 ({df['날짜'].min() if len(df) else '-'} ~ {df['날짜'].max() if len(df) else '-'})")
        if (df["시가"].fillna(0) <= 0).mean() > 0.5 if len(df) else False:
            print(f"  ⚠ {name} 시가가 비어 있음 — 단기 연구에서 근사값을 씁니다")
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["종목코드"] = out["종목코드"].astype(str).str.zfill(6)
    out = out.drop_duplicates(["날짜", "종목코드"], keep="last").sort_values(["종목코드", "날짜"])
    out.to_csv(OUT, index=False)
    print(f"ETF 일봉: {len(out)}행 → {OUT}")


if __name__ == "__main__":
    main()
