"""
분봉 자료 받기 (무료) — 원본은 data/intraday/raw/ 에만 (저장소에 올리지 않음: .gitignore).

  binance  data.binance.vision 의 월별 묶음(1분·5분봉, 2017~) — 그의 주 무대(코인)와 같은 시장
  upbit    api.upbit.com 분봉 (원화 마켓, 200봉씩 거슬러) — 그가 실제로 매매한 거래소
  yahoo    query1.finance.yahoo.com 1시간봉(최근 2년)·5분봉(최근 60일) — 미국 주식

  python src/intraday_data.py binance --interval 5m --start 2021-01 --end 2025-10
  python src/intraday_data.py upbit --unit 5 --days 900
  python src/intraday_data.py yahoo --interval 60m
"""
import argparse
import io
import os
import sys
import time
import zipfile

import pandas as pd

RAW = "data/intraday/raw"
BINANCE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT",
           "BNBUSDT", "TRXUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "BCHUSDT", "ETCUSDT"]
UPBIT = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE", "KRW-ADA", "KRW-AVAX", "KRW-LINK", "KRW-DOT", "KRW-TRX",
         "KRW-ETC", "KRW-BCH", "KRW-ATOM", "KRW-NEAR", "KRW-SUI", "KRW-SAND", "KRW-HBAR", "KRW-STX", "KRW-SEI", "KRW-ARB"]
YAHOO = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "MU", "NFLX", "PLTR", "COIN",
         "SMCI", "ARM", "INTC", "ORCL", "CRM", "UBER", "SHOP", "MSTR", "TSM", "ASML", "QCOM", "ADBE", "JPM", "XOM", "LLY"]


def _get(url, params=None, tries=5):
    import requests
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "rich-research"})
        except requests.RequestException:
            time.sleep(2 ** i)
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (418, 429, 500, 502, 503, 504):
            time.sleep(2 ** i + 1)
            continue
        return r
    return None


def parse_binance_csv(raw_bytes):
    """바이낸스 묶음 CSV (머리글 없음): open_time, o, h, l, c, v, close_time, … — 2025 부터 마이크로초."""
    d = pd.read_csv(io.BytesIO(raw_bytes), header=None)
    if not str(d.iloc[0, 0]).isdigit():                                   # 머리글 있는 판
        d = d.iloc[1:]
    t = pd.to_numeric(d[0])
    unit = "us" if t.max() > 1e14 else "ms"
    return pd.DataFrame({"ts": pd.to_datetime(t, unit=unit, utc=True), "open": pd.to_numeric(d[1]), "high": pd.to_numeric(d[2]),
                         "low": pd.to_numeric(d[3]), "close": pd.to_numeric(d[4]), "volume": pd.to_numeric(d[5])})


def binance(interval, start, end, symbols=None):
    out = os.path.join(RAW, "binance", interval)
    os.makedirs(out, exist_ok=True)
    months = pd.period_range(start, end, freq="M")
    for sym in symbols or BINANCE:
        path = os.path.join(out, f"{sym}.csv.gz")
        have = pd.read_csv(path, parse_dates=["ts"]) if os.path.exists(path) else None
        parts = [have] if have is not None else []
        done = set(have["ts"].dt.strftime("%Y-%m")) if have is not None else set()
        for m in months:
            if str(m) in done:
                continue
            url = f"https://data.binance.vision/data/spot/monthly/klines/{sym}/{interval}/{sym}-{interval}-{m}.zip"
            r = _get(url)
            if r is None or r.status_code != 200:
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            parts.append(parse_binance_csv(z.read(z.namelist()[0])))
        if parts:
            d = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
            d.to_csv(path, index=False)
            print(f"  binance {sym}: {len(d):,}봉 {d['ts'].iloc[0]} ~ {d['ts'].iloc[-1]}", flush=True)


def parse_upbit(js):
    return pd.DataFrame({"ts": pd.to_datetime([x["candle_date_time_utc"] for x in js], utc=True),
                         "open": [x["opening_price"] for x in js], "high": [x["high_price"] for x in js], "low": [x["low_price"] for x in js],
                         "close": [x["trade_price"] for x in js], "volume": [x["candle_acc_trade_volume"] for x in js]})


def upbit(unit, days, markets=None):
    out = os.path.join(RAW, "upbit", f"{unit}m")
    os.makedirs(out, exist_ok=True)
    stop_at = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    for mk in markets or UPBIT:
        path = os.path.join(out, f"{mk}.csv.gz")
        parts, to = [], None
        while True:
            prm = {"market": mk, "count": 200, **({"to": to} if to else {})}
            r = _get(f"https://api.upbit.com/v1/candles/minutes/{unit}", prm)
            if r is None or r.status_code != 200 or not r.json():
                break
            d = parse_upbit(r.json())
            parts.append(d)
            oldest = d["ts"].min()
            if oldest <= stop_at:
                break
            to = oldest.strftime("%Y-%m-%dT%H:%M:%SZ")
            time.sleep(0.12)                                                  # 초당 10번 아래
        if parts:
            d = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
            d.to_csv(path, index=False)
            print(f"  upbit {mk}: {len(d):,}봉 {d['ts'].iloc[0]} ~ {d['ts'].iloc[-1]}", flush=True)


def parse_yahoo(js):
    res = js["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    return pd.DataFrame({"ts": pd.to_datetime(res["timestamp"], unit="s", utc=True), "open": q["open"], "high": q["high"],
                         "low": q["low"], "close": q["close"], "volume": q["volume"]}).dropna(subset=["close"])


def yahoo(interval, symbols=None):
    rng = {"60m": "730d", "1h": "730d", "15m": "60d", "5m": "60d", "1m": "7d"}[interval]
    out = os.path.join(RAW, "yahoo", interval)
    os.makedirs(out, exist_ok=True)
    for sym in symbols or YAHOO:
        r = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", {"interval": interval, "range": rng, "includePrePost": "false"})
        if r is None or r.status_code != 200:
            print(f"  yahoo {sym}: 실패 {getattr(r, 'status_code', None)}")
            continue
        d = parse_yahoo(r.json())
        path = os.path.join(out, f"{sym}.csv.gz")
        if os.path.exists(path):
            d = pd.concat([pd.read_csv(path, parse_dates=["ts"]), d]).drop_duplicates("ts").sort_values("ts")
        d.to_csv(path, index=False)
        print(f"  yahoo {sym}: {len(d):,}봉 {d['ts'].iloc[0]} ~ {d['ts'].iloc[-1]}", flush=True)
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["binance", "upbit", "yahoo"])
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m"))
    ap.add_argument("--unit", type=int, default=5)
    ap.add_argument("--days", type=int, default=900)
    a = ap.parse_args()
    if a.source == "binance":
        binance(a.interval, a.start, a.end)
    elif a.source == "upbit":
        upbit(a.unit, a.days)
    else:
        yahoo(a.interval if a.interval != "5m" else "60m")


if __name__ == "__main__":
    main()
