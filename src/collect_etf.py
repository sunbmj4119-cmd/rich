"""
ETF 일봉 수집 → data/etf_ohlcv.csv  (전천후 모델이 국면마다 갈아탈 자산들)

  자료원: 네이버 차트(로그인 없음, 한 번에 수천 일) — 실패하면 pykrx(KRX) 로 한 번 더.
          KRX 는 2026-10-27 GitHub 서버 IP 를 막은 적이 있어 첫 순서에서 뺐다.

  자산 (코드 · 이름 · 세금 구분)
    국내 주식      069500 KODEX 200 · 229200 KODEX 코스닥150                    매매차익 비과세
    국내 파생      122630 KODEX 레버리지 · 114800 KODEX 인버스 · 252670 KODEX 200선물인버스2X
                   (매매차익과 과표기준가 증가분 중 작은 쪽에 15.4% — 주가지수 선물 이익은 과표에 거의 안 잡혀 사실상 0에 가깝다)
    채권·현금      153130 KODEX 단기채권 · 114260 KODEX 국고채3년 · 148070 KOSEF 국고채10년
    달러·금·미국   261240 KODEX 미국달러선물 · 132030 KODEX 골드선물(H)
                   143850 TIGER 미국S&P500선물(H) · 133690 TIGER 미국나스닥100
                   (기타 ETF — 매매차익에 15.4% 배당소득세. 연구·모의투자는 이익 난 매도마다 15.4%를 뗀다 = 보수적)
  ETF 는 모두 매도 때 증권거래세가 없다. 분배금은 가격에 안 들어가 있어 수익이 조금 낮게 잡힌다(보수적).

  python src/collect_etf.py                 # 매일: 최근 30거래일 다시 받아 덮어씀, 처음 보는 자산은 저절로 전체 기간
  python src/collect_etf.py --backfill      # 모든 자산 전체 기간 다시
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/etf_ohlcv.csv"
COLS = ["날짜", "종목코드", "종목명", "시가", "고가", "저가", "종가", "거래량", "거래대금", "NAV"]
# code: (이름, 세금 구분, 자산군)
ASSETS = {
    "069500": ("KODEX 200", "equity", "국내주식"),
    "114800": ("KODEX 인버스", "kderiv", "국내주식 하락"),
    "122630": ("KODEX 레버리지", "kderiv", "국내주식 2배"),
    "252670": ("KODEX 200선물인버스2X", "kderiv", "국내주식 하락 2배"),
    "229200": ("KODEX 코스닥150", "equity", "코스닥"),
    "153130": ("KODEX 단기채권", "other", "현금성"),
    "114260": ("KODEX 국고채3년", "other", "국내채권"),
    "148070": ("KOSEF 국고채10년", "other", "국내 장기채"),
    "261240": ("KODEX 미국달러선물", "other", "달러"),
    "132030": ("KODEX 골드선물(H)", "other", "금"),
    "143850": ("TIGER 미국S&P500선물(H)", "other", "미국주식"),
    "133690": ("TIGER 미국나스닥100", "other", "미국 기술주"),
}
TICKERS = {k: v[0] for k, v in ASSETS.items()}
FULL = 5000          # 전체 기간 요청 일수 (약 20년)


def fetch_naver(code, count):
    import naver
    d = naver.daily(code, count)
    if d.empty:
        return pd.DataFrame(columns=COLS)
    d["종목코드"] = code
    d["종목명"] = TICKERS[code]
    d["거래대금"] = (d["종가"] * d["거래량"]).round()
    d["NAV"] = None
    return d[COLS]


def fetch_krx(code, start, end):
    from pykrx import stock          # KRX_ID 가 있으면 가져오는 순간 로그인 — 막혀 있으면 여기서 예외
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
    ap.add_argument("--backfill", nargs="?", const="all")
    a = ap.parse_args()
    old = pd.read_csv(OUT, dtype={"종목코드": str}) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    if len(old):
        old["종목코드"] = old["종목코드"].astype(str).str.zfill(6)
    parts, bad = [old], []
    for code, (name, _, _) in ASSETS.items():
        have = old[old["종목코드"] == code]
        full = bool(a.backfill) or len(have) < 500
        n = FULL if full else 30
        df, src = pd.DataFrame(columns=COLS), "-"
        try:
            df, src = fetch_naver(code, n), "네이버"
        except Exception as e:
            print(f"  {name}: 네이버 실패 ({str(e)[:80]}) → KRX 로 다시")
        if df.empty:
            try:
                start = "20100101" if full else (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y%m%d")
                df, src = fetch_krx(code, start, pd.Timestamp.today().strftime("%Y%m%d")), "KRX"
            except Exception as e:
                print(f"  {name}: KRX 도 실패 ({type(e).__name__} {str(e)[:60]})")
        if df.empty:
            bad.append(name)
            continue
        print(f"  {name}: {len(df)}행 ({df['날짜'].min()} ~ {df['날짜'].max()}) · {src}{' · 전체 기간' if full else ''}")
        parts.append(df)
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out["종목코드"] = out["종목코드"].astype(str).str.zfill(6)
    out = out.drop_duplicates(["날짜", "종목코드"], keep="last").sort_values(["종목코드", "날짜"])
    out.to_csv(OUT, index=False)
    print(f"ETF 일봉: {len(out)}행 · {out['종목코드'].nunique()}개 자산 → {OUT}" + (f" · 실패 {bad}" if bad else ""))
    if len(bad) == len(ASSETS):
        sys.exit(1)


if __name__ == "__main__":
    main()
