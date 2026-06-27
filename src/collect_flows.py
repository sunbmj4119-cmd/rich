"""
수급·시세·공매도 보강 수집기 (pykrx, KRX 로그인 필요 — KRX_ID/KRX_PW 이미 연동됨)

== 수집 항목 ==
  1) 투자자별 순매수 (외국인/기관/개인)        - 종목별 일별  → data/flows.csv
  2) 공매도 (거래량비중 + 잔고비중)             - 종목별 일별  → data/shorts.csv
  3) OHLCV (시/고/저/종/거래량/거래대금)        - 날짜별 전종목 → data/ohlcv.csv
  4) 시가총액 / 상장주식수                      - 날짜별 전종목 → data/marketcap.csv

[설계 의도]
  - 백필을 단 1회만 돌리도록, 종목 단위로 도는 수급+공매도를 같은 루프에 묶음
    (KRX 호출을 한 번에 처리 → 차단 위험·소요시간 최소화)
  - 공매도는 일단 '수집·검증'까지만. score.py 반영은 verify_factor.py 검증 후 결정.

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
SHORTS = "data/shorts.csv"
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


def _pick(df, cands):
    """버전마다 컬럼명이 달라 후보 중 존재하는 첫 컬럼 반환"""
    for c in cands:
        if c in df.columns:
            return c
    return None


# ─────────────────────────────────────────────────────────
# 1+2) 종목별 시계열: 투자자 순매수 + 공매도 (같은 루프)
# ─────────────────────────────────────────────────────────
def collect_per_ticker(codes, start, end):
    """각 종목의 외국인/기관/개인 순매수 + 공매도(거래량·잔고)를 한 번에 수집.

    수급과 공매도 모두 (fromdate, todate, ticker) 시그니처라 종목당 호출을 묶어
    백필 시 KRX 접속 횟수를 절반으로 줄인다.
    """
    flow_rows, short_rows = [], []
    flow_fails, short_fails = [], []
    s = start.replace("-", "")
    e = end.replace("-", "")

    for i, code in enumerate(codes):
        # --- (1) 투자자별 순매수 ---
        try:
            df = stock.get_market_trading_value_by_date(s, e, code, detail=False)
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={df.reset_index().columns[0]: "날짜"})
                f_col = _pick(df, ["외국인", "외국인합계"])
                i_col = _pick(df, ["기관합계", "기관"])
                p_col = _pick(df, ["개인"])
                for _, r in df.iterrows():
                    flow_rows.append({
                        "날짜": pd.to_datetime(r["날짜"]).strftime("%Y-%m-%d"),
                        "종목코드": code,
                        "외국인순매수": r[f_col] if f_col else np.nan,
                        "기관순매수": r[i_col] if i_col else np.nan,
                        "개인순매수": r[p_col] if p_col else np.nan,
                    })
            else:
                flow_fails.append(code)
        except Exception:
            flow_fails.append(code)

        time.sleep(0.3)  # KRX 차단 방지

        # --- (2) 공매도: 거래량비중 + 잔고비중 ---
        try:
            vol = stock.get_shorting_volume_by_date(s, e, code)   # 공매도/거래/비중
            bal = stock.get_shorting_balance_by_date(s, e, code)  # 잔고/금액/비중
            sv = {}  # 날짜 -> dict
            if vol is not None and not vol.empty:
                vol = vol.reset_index().rename(columns={vol.reset_index().columns[0]: "날짜"})
                # 컬럼 예: '공매도','매수','비중'  (비중 = 공매도/거래량, 단위 %)
                vw_col = _pick(vol, ["비중", "공매도비중"])
                for _, r in vol.iterrows():
                    d = pd.to_datetime(r["날짜"]).strftime("%Y-%m-%d")
                    sv.setdefault(d, {})["공매도거래비중"] = r[vw_col] if vw_col else np.nan
            if bal is not None and not bal.empty:
                bal = bal.reset_index().rename(columns={bal.reset_index().columns[0]: "날짜"})
                # 컬럼 예: '잔고수량','잔고금액','비중'  (비중 = 잔고/상장주식수, 단위 %)
                bw_col = _pick(bal, ["비중", "잔고비중"])
                bq_col = _pick(bal, ["잔고수량", "잔고"])
                for _, r in bal.iterrows():
                    d = pd.to_datetime(r["날짜"]).strftime("%Y-%m-%d")
                    sv.setdefault(d, {})["공매도잔고비중"] = r[bw_col] if bw_col else np.nan
                    sv[d]["공매도잔고수량"] = r[bq_col] if bq_col else np.nan
            if sv:
                for d, vals in sv.items():
                    short_rows.append({
                        "날짜": d, "종목코드": code,
                        "공매도거래비중": vals.get("공매도거래비중", np.nan),
                        "공매도잔고비중": vals.get("공매도잔고비중", np.nan),
                        "공매도잔고수량": vals.get("공매도잔고수량", np.nan),
                    })
            else:
                short_fails.append(code)
        except Exception:
            short_fails.append(code)

        time.sleep(0.3)

        if (i + 1) % 20 == 0:
            print(f"    수급+공매도 {i+1}/{len(codes)} 종목...")

    return (pd.DataFrame(flow_rows), pd.DataFrame(short_rows),
            flow_fails, short_fails)


# ─────────────────────────────────────────────────────────
# 3+4) OHLCV + 시가총액 (날짜별 전종목 = 호출수 적음)
# ─────────────────────────────────────────────────────────
def collect_ohlcv_cap(codes, dates):
    code_set = set(codes)
    ohlcv_rows, cap_rows, fails = [], [], []
    for i, d in enumerate(dates):
        ds = d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d).replace("-", "")
        dstr = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        try:
            o = stock.get_market_ohlcv_by_ticker(ds, market="ALL")
            c = stock.get_market_cap_by_ticker(ds, market="ALL")
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
                    "날짜": dstr, "종목코드": code,
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
                        "날짜": dstr, "종목코드": code,
                        "시가총액": r.get("시가총액"),
                        "상장주식수": r.get("상장주식수"),
                    })
        except Exception:
            fails.append(ds)
        if (i + 1) % 20 == 0:
            print(f"    OHLCV {i+1}/{len(dates)} 일...")
        time.sleep(0.3)
    return pd.DataFrame(ohlcv_rows), pd.DataFrame(cap_rows), fails


# ─────────────────────────────────────────────────────────
# 저장 (중복 제거 append)
# ─────────────────────────────────────────────────────────
def save_merge(df, path, keys=("날짜", "종목코드")):
    if df is None or df.empty:
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
    specs = [
        (FLOWS, ["외국인순매수", "기관순매수", "개인순매수"]),
        (SHORTS, ["공매도거래비중", "공매도잔고비중", "공매도잔고수량"]),
        (OHLCV, ["시가", "고가", "저가", "종가", "거래대금"]),
        (MCAP, ["시가총액", "상장주식수"]),
    ]
    for path, cols in specs:
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
            ex = df[c].dropna().iloc[0] if nn else "N/A"
            print(f"   {c}: 값있음 {nn/len(df)*100:.0f}% | "
                  f"0값 {zero/len(df)*100:.0f}% | 예시 {ex}")


# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", metavar="YYYY-MM-DD", help="이 날짜부터 전체 수집")
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
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        print(f"데일리 모드: {start} ~ {end}")

    # 1+2) 수급 + 공매도 (종목 루프 1회)
    print("\n[1/2] 투자자 순매수 + 공매도 수집...")
    flows, shorts, ff, sf = collect_per_ticker(codes, start, end)
    if ff:
        print(f"    수급 실패 {len(ff)}종목: {ff[:5]}{'...' if len(ff)>5 else ''}")
    if sf:
        print(f"    공매도 실패 {len(sf)}종목: {sf[:5]}{'...' if len(sf)>5 else ''}")
    if flows.empty:
        print("    🔴 수급 0건! 기간이 너무 짧거나 KRX 접속불가 — 재시도 권장")
    if shorts.empty:
        print("    🔴 공매도 0건! KRX 로그인(KRX_ID/KRX_PW) 확인 필요")
    save_merge(flows, FLOWS)
    save_merge(shorts, SHORTS)

    # 3+4) OHLCV + 시총
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

