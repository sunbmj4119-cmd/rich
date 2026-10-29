"""
미국 주식 일봉 수집 → data/us_ohlcv.csv · 원/달러 환율 → data/fx_usdkrw.csv

  유니버스: S&P 100 (2026년 기준 구성 — 지금 남아 있는 대형주라 생존편향이 있다) + 비교용 SPY·QQQ
  자료원: Yahoo 차트(키 없음, 무료) — 실패하면 Stooq CSV.
          가격은 분할·배당을 반영한 수정가(Yahoo adjclose 비율을 시가·고가·저가에도 곱함) → 수익률에 배당 포함.
  GitHub Actions 에서 돈다 (이 저장소의 연구 환경에서는 바깥 사이트 접속이 막혀 있음).

  python src/collect_us.py              # 매일: 최근 30일 다시 받아 덮어씀, 처음 보는 종목은 2014년부터
  python src/collect_us.py --backfill   # 전부 다시
"""
import argparse
import io
import os
import sys
import time

import pandas as pd
import requests

OUT = "data/us_ohlcv.csv"
FX = "data/fx_usdkrw.csv"
SP100 = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN", "AVGO", "AXP", "BA", "BAC", "BK",
         "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
         "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOGL", "GS", "HD", "HON",
         "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ",
         "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP",
         "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
         "TSLA", "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM"]
BENCH = ["SPY", "QQQ"]
COLS = ["날짜", "종목코드", "시가", "고가", "저가", "종가", "거래량"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def yahoo(sym, start="2014-01-01"):
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp.today().timestamp()) + 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    r = requests.get(url, params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"}, headers=UA, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    df = pd.DataFrame({"날짜": pd.to_datetime(res["timestamp"], unit="s").tz_localize("UTC")
                       .tz_convert(res.get("meta", {}).get("exchangeTimezoneName", "America/New_York")).date,
                       "시가": q["open"], "고가": q["high"], "저가": q["low"], "종가": q["close"], "거래량": q["volume"]})
    if adj:
        f = pd.Series(adj) / df["종가"]
        for c in ("시가", "고가", "저가", "종가"):
            df[c] = df[c] * f.values
    df["날짜"] = pd.to_datetime(df["날짜"]).dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["종가"])


def stooq(sym, start="2014-01-01"):
    s = sym.lower().replace("-", ".") + (".us" if not sym.endswith("=X") else "")
    if sym == "KRW=X":
        s = "usdkrw"
    r = requests.get("https://stooq.com/q/d/l/", params={"s": s, "i": "d", "d1": start.replace("-", "")}, headers=UA, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={"Date": "날짜", "Open": "시가", "High": "고가", "Low": "저가", "Close": "종가", "Volume": "거래량"})
    if "거래량" not in df:
        df["거래량"] = 0
    return df[["날짜", "시가", "고가", "저가", "종가", "거래량"]].dropna(subset=["종가"])


def fetch(sym, start):
    last = ""
    for f in (yahoo, stooq):
        try:
            d = f(sym, start)
            if len(d):
                return d, f.__name__
        except Exception as e:
            last = f"{f.__name__}: {type(e).__name__} {str(e)[:80]}"
        time.sleep(0.3)
    print(f"  {sym} 실패 ({last})")
    return pd.DataFrame(columns=COLS[:1] + COLS[2:]), "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    have = old.groupby("종목코드")["날짜"].count().to_dict() if len(old) else {}
    recent = (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    parts, src_n, bad = [old], {}, []
    for sym in SP100 + BENCH:
        full = a.backfill or have.get(sym, 0) < 500
        d, src = fetch(sym, "2014-01-01" if full else recent)
        if d.empty:
            bad.append(sym)
            continue
        d["종목코드"] = sym
        parts.append(d[COLS])
        src_n[src] = src_n.get(src, 0) + 1
        time.sleep(0.2)
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out = out.drop_duplicates(["날짜", "종목코드"], keep="last").sort_values(["종목코드", "날짜"])
    out.to_csv(OUT, index=False)
    print(f"미국 일봉: {len(out):,}행 · {out['종목코드'].nunique()}종목 · {out['날짜'].min()} ~ {out['날짜'].max()} · 자료원 {src_n}"
          + (f" · 실패 {bad}" if bad else ""))
    fx_old = pd.read_csv(FX) if os.path.exists(FX) else pd.DataFrame(columns=["날짜", "환율"])
    d, src = fetch("KRW=X", "2014-01-01" if a.backfill or len(fx_old) < 500 else recent)
    if len(d):
        fx = pd.concat([fx_old, d.rename(columns={"종가": "환율"})[["날짜", "환율"]]], ignore_index=True)
        fx = fx.drop_duplicates("날짜", keep="last").sort_values("날짜")
        fx.to_csv(FX, index=False)
        print(f"원/달러: {len(fx)}행 · {fx['날짜'].min()} ~ {fx['날짜'].max()} · 마지막 {fx['환율'].iloc[-1]:.1f} · {src}")
    if len(bad) > len(SP100) // 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
