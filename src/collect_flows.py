"""
수급·시세 보강 수집기 (pykrx 무료, KRX 공개데이터, 키 불필요)
== 수집 항목 ==
  1) 투자자별 순매수 (외국인/기관/개인) - 종목별 일별
  2) OHLCV (시가/고가/저가/종가/거래량/거래대금) - 날짜별 전종목
  3) 시가총액 / 상장주식수 - 날짜별 전종목

== 출력 ==
  data/flows.csv     : 날짜,종목코드,외국인순매수,기관순매수,개인순매수
  data/ohlcv.csv     : 날짜,종목코드,시가,고가,저가,종가,거래량,거래대금
  data/marketcap.csv : 날짜,종목코드,시가총액,상장주식수

== 사용 ==
  python src/collect_flows.py --backfill 2018-01-01   # 과거 전체 1회 수집
  python src/collect_flows.py                          # 최근일만 추가(매일)
  python src/collect_flows.py --check                  # 수집 데이터 검증만

[검증 철학]
  - 빈 DataFrame, 전부 0, 결측 과다, 종목수 부족을 모두 체크하고 로그 출력
  - 실패 종목은 건너뛰되 기록 (조용한 실패 방지)
"""
import argparse
import os
import time
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PRICES = "data/prices.csv"
FLOWS = "data/flows.csv"
OHLCV = "data/ohlcv.csv"
MCAP = "data/marketcap.csv"

try:
    from pykrx import stock
except ImportError:
    print("pykrx 필요: pip install pykrx")
    sys.exit(1)


def get_universe():
    """우리 100종목 코드 목록 (prices.csv 기준)"""
    px = pd.read_csv(PRICES, dtype={"종목코드": str})
    px["종목코드"] = px["종목코드"].str.zfill(6)
    return sorted(px["종목코드"].unique())


def trading_dates(start, end):
    """KRX 영업일 목록"""
    return stock.get_previous_business_days(fromdate=start.replace("-", ""),
                                            todate=end.replace("-", ""))


# ─────────────────────────────────────────────────────────
# 1) 투자자별 순매수 (종목별 시계열)
# ─────────────────────────────────────────────────────────
def collect_flows(codes, start, end, existing=None):
    """각 종목의 외국인/기관/개인 순매수 금액 일별 수집"""
    rows = []
    fails = []
    s = start.replace("-", "")
    e = end.replace("-", "")
    for i, code in enumerate(codes):
        try:
            df = stock.get_market_trading_value_by_date(s, e, code, detail=True)
            if df is None or df.empty:
                fails.append(code)
                continue
            df = df.reset_index()
            # 컬럼: 날짜, 기관합계, ..., 외국인, 개인, 전체 등 (detail에 따라)
            datecol = df.columns[0]
            df = df.rename(columns={datecol: "날짜"})
            # 외국인/기관/개인 컬럼 탐색 (버전마다 이름 약간 다름)
            def pick(cands):
                for c in cands:
                    if c in df.columns:
                        return c
                return None
            f_col = pick(["외국인", "외국인합계"])
            i_col = pick(["기관합계", "기관"])
            p_col = pick(["개인"])
            for _, r in df.iterrows():
                rows.append({
                    "날짜": pd.to_datetime(r["날짜"]).strftime("%Y-%m-%d"),
                    "종목코드": code,
                    "외국인순매수": r[f_col] if f_col else np.nan,
                    "기관순매수": r[i_col] if i_col else np.nan,
                    "개인순매수": r[p_col] if p_col else np.nan,
                })
        except Exception as ex:
            fails.append(code)
        if (i + 1) % 20 == 0:
            print(f"    수급 {i+1}/{len(codes)} 종목...")
        time.sleep(0.3)  # KRX 차단 방지
    return pd.DataFrame(rows), fails


# ─────────────────────────────────────────────────────────
# 2) OHLCV + 3) 시가총액 (날짜별 전종목)
# ─────────────────────────────────────────────────────────
def collect_ohlcv_cap(codes, dates):
    """날짜별로 전종목 OHLCV·시총 수집 (날짜 단위가 호출수 적음)"""
    code_set = set(codes)
    ohlcv_rows = []
    cap_rows = []
    fails = []
    for i, d in enumerate(dates):
        ds = d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d).replace("-", "")
        try:
            o = stock.get_market_ohlcv_by_ticker(ds, market="KOSPI")
            c = stock.get_market_cap_by_ticker(ds, market="KOSPI")
            if o is None or o.empty:
                fails.append(ds)
                continue
            o = o.reset_index()
            tk = o.columns[0]
            for _, r in o.iterrows():
                code = str(r[tk]).zfill(6)
                if code not in code_set:
                    continue
                ohlcv_rows.append({
                    "날짜": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                    "종목코드": code,
                    "시가": r.get("시가"), "고가": r.get("고가"),
                    "저가": r.get("저가"), "종가": r.get("종가"),
                    "거래량": r.get("거래량"), "거래대금": r.get("거래대금"),
                })
            if c is not None and not c.empty:
                c = c.reset_index()
                tkc = c.columns[0]
                for _, r in c.iterrows():
                    code = str(r[tkc]).zfill(6)
                    if code not in code_set:
                        continue
                    cap_rows.append({
                        "날짜": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                        "종목코드": code,
                        "시가총액": r.get("시가총액"),
                        "상장주식수": r.get("상장주식수"),
                    })
        except Exception as ex:
            fails.append(ds)
        if (i + 1) % 20 == 0:
            print(f"    OHLCV {i+1}/{len(dates)} 일...")
        time.sleep(0.3)
    return pd.DataFrame(ohlcv_rows), pd.DataFrame(cap_rows), fails


# ─────────────────────────────────────────────────────────
# 저장 (중복 제거 append)
# ─────────────────────────────────────────────────────────
def save_merge(df, path, keys=("날짜", "종목코드")):
    if df.empty:
        print(f"    ⚠️ {path}: 수집된 행 없음 — 저장 건너뜀")
        return
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"종목코드": str})
        old["종목코드"] = old["종목코드"].str.zfill(6)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=list(keys), keep="last")
    df = df.sort_values(list(keys))
    df.to_csv(path, index=False)
    print(f"    ✅ {path}: 총 {len(df):,}행 저장")


# ─────────────────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────────────────
def check():
    print("\n=== 수집 데이터 검증 ===")
    for path, cols in [(FLOWS, ["외국인순매수", "기관순매수", "개인순매수"]),
                       (OHLCV, ["시가", "고가", "저가", "종가", "거래대금"]),
                       (MCAP, ["시가총액", "상장주식수"])]:
        if not os.path.exists(path):
            print(f"\n❌ {path}: 파일 없음 (아직 수집 안 됨)")
            continue
        df = pd.read_csv(path, dtype={"종목코드": str})
        print(f"\n■ {path}")
        print(f"   행수: {len(df):,} | 종목수: {df['종목코드'].nunique()} | "
              f"기간: {df['날짜'].min()} ~ {df['날짜'].max()}")
        for c in cols:
            if c not in df.columns:
                print(f"   ⚠️ '{c}' 컬럼 없음")
                continue
            nn = df[c].notna().sum()
            zero = (df[c] == 0).sum()
            print(f"   {c}: 값있음 {nn/len(df)*100:.0f}% | "
                  f"0값 {zero/len(df)*100:.0f}% | "
                  f"예시 {df[c].dropna().iloc[0] if nn else 'N/A'}")


# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", metavar="YYYY-MM-DD",
                    help="이 날짜부터 전체 수집")
    ap.add_argument("--check", action="store_true", help="검증만")
    args = ap.parse_args()

    if args.check:
        check()
        return

    codes = get_universe()
    print(f"유니버스: {len(codes)}종목")

    if args.backfill:
        start = args.backfill
        end = datetime.today().strftime("%Y-%m-%d")
        print(f"백필 모드: {start} ~ {end}")
    else:
        # 데일리: 최근 7일만 (놓친 날 보충)
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        print(f"데일리 모드: {start} ~ {end}")

    # 1) 수급
    print("\n[1/2] 투자자별 순매수 수집...")
    flows, ff = collect_flows(codes, start, end)
    if ff:
        print(f"    수급 실패 {len(ff)}종목: {ff[:5]}{'...' if len(ff)>5 else ''}")
    if flows.empty:
        print("    🔴 수급 0건! 원인 추정:")
        print(f"       - 기간({start}~{end})이 너무 짧거나 영업일 없음")
        print(f"       - 백필 안 했으면 --backfill 2018-01-01 로 재실행 권장")
        print(f"       - 전종목 실패면 KRX 일시 접속불가 (재시도)")
    save_merge(flows, FLOWS)

    # 2) OHLCV + 시총
    print("\n[2/2] OHLCV·시총 수집...")
    dates = trading_dates(start, end)
    print(f"    대상 영업일: {len(dates)}일")
    ohlcv, cap, of = collect_ohlcv_cap(codes, dates)
    if of:
        print(f"    OHLCV 실패 {len(of)}일")
    save_merge(ohlcv, OHLCV)
    save_merge(cap, MCAP)

    check()


if __name__ == "__main__":
    main()


