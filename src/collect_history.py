"""
긴 역사 자료 (연구 전용) → data/history/hist.csv · data/history/us_stocks.csv

  왜: 2014~2026 자료는 규칙을 만드는 데 이미 썼다. 그 전(1997~2013) 자료는 한 번도 안 봤으므로,
      지금 규칙을 고정한 채 거기에 돌리면 '진짜 처음 보는 시험'(표본 외 검증)이 된다 — IMF·닷컴·2008 금융위기 포함.
  자료원: Yahoo 차트 (키 없음). 수정가(분할·배당 반영). GitHub Actions 에서 돈다.

  지수·ETF (hist.csv, 종목코드 = Yahoo 기호)
    한국: ^KS11 코스피 · ^KQ11 코스닥 · 069500.KS KODEX 200 · 업종 ETF(반도체·은행·자동차·증권·에너지화학·건설·철강 …)
    미국: SPY · QQQ · IWM · 업종 SPDR 9종 · TLT · IEF · SHY · GLD · UUP · EFA · EEM · DBC · VNQ
    환율: KRW=X
  미국 대형주 (us_stocks.csv): S&P 100 (지금 구성, collect_us.SP100) 2000년부터 — 생존편향 주의
  국내 대형주 (kr_stocks.csv): 최근 코스피 100 에 든 적 있는 종목 전부(data/universe.csv) 2000년부터 — 역시 생존편향

  python src/collect_history.py        → 전부
  python src/collect_history.py kr     → 국내 대형주만
  python src/collect_history.py assets → 지수·ETF·환율(hist.csv)만
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_us as CU  # noqa: E402

DIR = "data/history"
HIST = os.path.join(DIR, "hist.csv")
STOCKS = os.path.join(DIR, "us_stocks.csv")
KR_STOCKS = os.path.join(DIR, "kr_stocks.csv")
KR = {"^KS11": "코스피 지수", "^KQ11": "코스닥 지수", "069500.KS": "KODEX 200", "091160.KS": "KODEX 반도체", "091170.KS": "KODEX 은행",
      "091180.KS": "KODEX 자동차", "102970.KS": "KODEX 증권", "117460.KS": "KODEX 에너지화학", "117700.KS": "KODEX 건설",
      "117680.KS": "KODEX 철강", "140710.KS": "KODEX 운송", "266420.KS": "KODEX 헬스케어", "114800.KS": "KODEX 인버스"}
US = {"SPY": "S&P 500", "QQQ": "나스닥 100", "IWM": "러셀 2000", "XLK": "기술", "XLF": "금융", "XLE": "에너지", "XLV": "헬스케어",
      "XLI": "산업재", "XLP": "필수소비재", "XLY": "임의소비재", "XLU": "유틸리티", "XLB": "소재", "TLT": "미국 장기채", "IEF": "미국 중기채",
      "SHY": "미국 단기채", "GLD": "금", "UUP": "달러 지수", "EFA": "선진국 주식", "EEM": "신흥국 주식", "DBC": "원자재", "VNQ": "리츠",
      # 자산배분 전략(VAA·DAA·HAA·듀얼 모멘텀·영구·올웨더)이 쓰는 것
      "AGG": "미국 종합채권", "BND": "미국 종합채권(뱅가드)", "LQD": "미국 투자등급 회사채", "HYG": "미국 하이일드", "TIP": "미국 물가연동채",
      "BIL": "미국 초단기 국채", "VEA": "선진국 주식(뱅가드)", "VWO": "신흥국 주식(뱅가드)", "VGK": "유럽 주식", "EWJ": "일본 주식",
      "GSG": "원자재(GSCI)", "VTI": "미국 전체 주식"}
FX = {"KRW=X": "원/달러"}


def kr_stocks():
    """최근 코스피 100 에 든 적 있는 종목 (universe.csv 전체 기간) — Yahoo .KS (안 되면 .KQ), 2000년부터 수정가."""
    codes = sorted(pd.read_csv("data/universe.csv", dtype=str)["종목코드"].str.zfill(6).unique())
    names = {}
    if os.path.exists("data/sectors.csv"):
        names = pd.read_csv("data/sectors.csv", dtype={"종목코드": str}).set_index("종목코드")["종목명"].to_dict()
    rows, bad = [], []
    for c in codes:
        d = None
        for suf in (".KS", ".KQ"):
            try:
                d = CU.yahoo(c + suf, "2000-01-01")
            except Exception:
                d = None
            if d is not None and not d.empty:
                break
        if d is None or d.empty:
            bad.append(c)
            continue
        d["종목코드"], d["종목명"] = c, names.get(c, "")
        rows.append(d)
        time.sleep(0.2)
    s = pd.concat(rows, ignore_index=True)
    for col in ("시가", "고가", "저가", "종가"):
        s[col] = s[col].round(2)
    s.to_csv(KR_STOCKS, index=False)
    first = s.groupby("종목코드")["날짜"].min()
    print(f"국내 대형주: {s['종목코드'].nunique()}종목 · {len(s):,}행 · 2005년 전부터 있는 종목 {(first < '2005-01-01').sum()}개 → {KR_STOCKS}"
          + (f" · 실패 {bad}" if bad else ""))


def main():
    os.makedirs(DIR, exist_ok=True)
    if sys.argv[1:] == ["kr"]:
        return kr_stocks()
    assets_only = sys.argv[1:] == ["assets"]
    rows, bad = [], []
    for sym, name in {**KR, **US, **FX}.items():
        try:
            d = CU.yahoo(sym, "1990-01-01")
        except Exception as e:
            bad.append(f"{sym}({type(e).__name__})")
            continue
        if d.empty:
            bad.append(sym)
            continue
        d["종목코드"], d["종목명"] = sym, name
        rows.append(d)
        print(f"  {sym:10s} {name:10s} {len(d):5d}일 {d['날짜'].min()} ~ {d['날짜'].max()}")
        time.sleep(0.3)
    h = pd.concat(rows, ignore_index=True)
    for c in ("시가", "고가", "저가", "종가"):
        h[c] = h[c].round(4)
    h.to_csv(HIST, index=False)
    print(f"지수·ETF: {h['종목코드'].nunique()}개 · {len(h):,}행 → {HIST}" + (f" · 실패 {bad}" if bad else ""))
    if assets_only:
        return
    srows, sbad = [], []
    for sym in CU.SP100:
        try:
            d = CU.yahoo(sym, "2000-01-01")
            d["종목코드"] = sym
            srows.append(d)
        except Exception:
            sbad.append(sym)
        time.sleep(0.2)
    s = pd.concat(srows, ignore_index=True)
    for c in ("시가", "고가", "저가", "종가"):
        s[c] = s[c].round(4)
    s.to_csv(STOCKS, index=False)
    print(f"미국 대형주: {s['종목코드'].nunique()}종목 · {len(s):,}행 · {s['날짜'].min()} ~ → {STOCKS}" + (f" · 실패 {sbad}" if sbad else ""))
    kr_stocks()


if __name__ == "__main__":
    main()
