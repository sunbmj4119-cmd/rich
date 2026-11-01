"""
KRX 긴 역사 수집기 — Oracle 서버(또는 집 PC)에서 '한 번' 돌린다. GitHub 서버는 KRX 가 IP 를 막아 못 받는다.

  왜: 지금 단기 몫에서 가장 많이 뽑히는 신호가 외국인·기관 수급인데 2018년 전 자료가 없어 긴 역사 검증을 못 했다(가장 큰 빈칸).
      또 긴 역사 검증의 종목이 '지금 살아남은 대형주'뿐이라 생존편향이 있었다.
  받는 것 (pykrx, KRX 로그인 KRX_ID/KRX_PW — ~/.rich/secrets.env)
    krx_universe.csv.gz : 해마다 마지막 거래일의 코스피 시가총액 상위 150 (상장폐지된 회사 포함 = 그때 실제 대형주)
    krx_ohlcv.csv.gz    : 그 종목들의 수정 일봉 2000~2018
    krx_flows.csv.gz    : 같은 종목들의 외국인·기관·개인 순매수 금액 2000~2018
    krx_etf.csv.gz      : 국내 ETF 12종 상장일부터 전 기간
  이어 받기: 종목·연도마다 data/history/krx_cache/ 에 저장 → 끊겨도 다시 돌리면 이어서 (KRX 차단을 피하려 호출마다 쉼)
  시간: 대략 1~2시간

  서버: bash ~/rich/deploy/oracle/collect_history.sh   (받고 → GitHub 에 올림)
  직접: python src/collect_krx_history.py [--years 2000 2018] [--top 150]
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DIR = "data/history"
CACHE = os.path.join(DIR, "krx_cache")
ETF_CODES = ["069500", "114800", "122630", "252670", "229200", "153130", "114260", "148070", "261240", "132030", "143850", "133690"]


def stock_api():
    try:
        from pykrx import stock          # KRX_ID 가 있으면 가져오는 순간 로그인
        return stock
    except Exception as ex:
        sys.exit(f"pykrx 를 쓸 수 없습니다 ({type(ex).__name__}: {ex}) — pip install pykrx, KRX_ID/KRX_PW 확인")


def cached(name, fn):
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p):
        return pd.read_pickle(p)
    df = fn()
    if df is not None:
        os.makedirs(CACHE, exist_ok=True)
        df.to_pickle(p)
    return df


def year_end(stock, y):
    """그해 마지막 거래일 (YYYYMMDD)."""
    for d in pd.date_range(f"{y}-12-31", periods=15, freq="-1D"):
        s = d.strftime("%Y%m%d")
        try:
            df = stock.get_market_cap(s, market="KOSPI")
            if df is not None and len(df) > 100 and df["시가총액"].sum() > 0:
                return s, df
        except Exception:
            pass
        time.sleep(0.3)
    return None, None


def universe(stock, years, top):
    rows = []
    for y in years:
        def get():
            s, df = year_end(stock, y)
            if df is None:
                return None
            df = df.sort_values("시가총액", ascending=False).head(top).reset_index()
            df = df.rename(columns={df.columns[0]: "종목코드"})
            df["기준일"] = pd.Timestamp(s).strftime("%Y-%m-%d")
            df["순위"] = range(1, len(df) + 1)
            return df[["기준일", "종목코드", "시가총액", "순위"]]
        d = cached(f"univ_{y}_{top}", get)
        if d is not None:
            rows.append(d)
            print(f"  {y} 기준일 {d['기준일'].iloc[0]} 상위 {len(d)}")
    u = pd.concat(rows, ignore_index=True)
    try:
        names = {}
        for c in u["종목코드"].unique():
            try:
                names[c] = stock.get_market_ticker_name(c)
            except Exception:
                names[c] = ""
        u["종목명"] = u["종목코드"].map(names)
    except Exception:
        u["종목명"] = ""
    return u


def per_code(stock, codes, years, kind):
    out = []
    for i, c in enumerate(codes):
        for y in years:
            s, e = f"{y}0101", f"{y}1231"

            def get():
                try:
                    if kind == "ohlcv":
                        df = stock.get_market_ohlcv_by_date(s, e, c, adjusted=True)
                    elif kind == "etf":
                        df = stock.get_etf_ohlcv_by_date(s, e, c)
                    else:
                        df = stock.get_market_trading_value_by_date(s, e, c, detail=False)
                except Exception:
                    return pd.DataFrame()
                finally:
                    time.sleep(0.35)
                return df if df is not None else pd.DataFrame()
            df = cached(f"{kind}_{c}_{y}", get)
            if df is None or df.empty:
                continue
            df = df.reset_index().rename(columns={df.reset_index().columns[0]: "날짜"})
            df["종목코드"] = c
            out.append(df)
        if (i + 1) % 20 == 0:
            print(f"    {kind} {i + 1}/{len(codes)}")
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d["날짜"] = pd.to_datetime(d["날짜"]).dt.strftime("%Y-%m-%d")
    return d


def tidy_flows(f):
    if f.empty:
        return f
    pick = {"외국인순매수": ["외국인", "외국인합계"], "기관순매수": ["기관합계", "기관"], "개인순매수": ["개인"]}
    out = f[["날짜", "종목코드"]].copy()
    for k, cands in pick.items():
        col = next((c for c in cands if c in f.columns), None)
        out[k] = f[col] if col else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2000, 2018])
    ap.add_argument("--top", type=int, default=150)
    a = ap.parse_args()
    stock = stock_api()
    years = list(range(a.years[0], a.years[1] + 1))
    os.makedirs(DIR, exist_ok=True)
    t0 = time.time()
    print("1) 해마다 시가총액 상위 (상장폐지 포함)")
    u = universe(stock, years, a.top)
    u.to_csv(os.path.join(DIR, "krx_universe.csv.gz"), index=False)
    codes = sorted(u["종목코드"].unique())
    print(f"   종목 {len(codes)}개 (해마다 상위 {a.top}의 합집합)")
    print("2) 수정 일봉")
    o = per_code(stock, codes, years, "ohlcv")
    if not o.empty:
        keep = [c for c in ("날짜", "종목코드", "시가", "고가", "저가", "종가", "거래량") if c in o.columns]
        o[keep].to_csv(os.path.join(DIR, "krx_ohlcv.csv.gz"), index=False)
    print(f"   {len(o):,}행")
    print("3) 투자자별 순매수 (외국인·기관·개인)")
    f = tidy_flows(per_code(stock, codes, years, "flows"))
    if not f.empty:
        f.to_csv(os.path.join(DIR, "krx_flows.csv.gz"), index=False)
    print(f"   {len(f):,}행")
    print("4) 국내 ETF 12종 전 기간")
    e = per_code(stock, ETF_CODES, list(range(2002, pd.Timestamp.now().year + 1)), "etf")
    if not e.empty:
        keep = [c for c in ("날짜", "종목코드", "시가", "고가", "저가", "종가", "거래량") if c in e.columns]
        e[keep].to_csv(os.path.join(DIR, "krx_etf.csv.gz"), index=False)
    print(f"   {len(e):,}행")
    print(f"끝 ({(time.time() - t0) / 60:.0f}분) → {DIR}/krx_*.csv.gz")


if __name__ == "__main__":
    main()
