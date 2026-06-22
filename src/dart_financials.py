"""
DART 분기 재무 수집 (1회 백필) — EPS/BPS/주식수 포함 버전
- 종목 리스트는 data/prices.csv 에서 읽음 (KRX 지수 API 회피)
- 손익/재무상태 항목 + 파생지표(부채비율/영업이익률/순이익률/ROE)
- 추가: 주식총수(주식의 총수 보고서) → EPS = 순이익/주식수, BPS = 자본총계/주식수
  · score.py 에서 PER = 종가/EPS, PBR = 종가/BPS 로 가치 팩터 계산
- data/financials.csv 로 분리 저장. 점수 계산 시 as-of join.
- 이어받기: 이미 받은 종목 건너뜀
- 고정 컬럼 스키마(COLS)로 저장 → resume 시 컬럼 어긋남 방지
"""
import os
import time
import pandas as pd
from datetime import datetime
import OpenDartReader

API_KEY = os.environ["DART_API_KEY"]
dart = OpenDartReader(API_KEY)

os.makedirs("data", exist_ok=True)
OUT = "data/financials.csv"
PRICES = "data/prices.csv"

REPRTS = {"11013": "1Q", "11012": "2Q", "11014": "3Q", "11011": "4Q"}
AVAIL = {"1Q": "-05-15", "2Q": "-08-15", "3Q": "-11-15", "4Q": "-04-01"}
START_YEAR = 2016

# 고정 컬럼 스키마 (resume 안전)
COLS = [
    "종목코드", "종목명", "연도", "분기", "사용가능일",
    "매출액", "영업이익", "당기순이익",
    "자산총계", "부채총계", "자본총계",
    "부채비율", "영업이익률", "순이익률", "ROE",
    "주식수", "EPS", "BPS",
]


def to_num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None


def pick(df, account_names):
    for nm in account_names:
        row = df[df["account_nm"].str.contains(nm, na=False)]
        if not row.empty:
            return to_num(row.iloc[0].get("thstrm_amount"))
    return None


def get_universe():
    """prices.csv 에서 종목코드/이름 추출"""
    df = pd.read_csv(PRICES, dtype={"종목코드": str})
    df["종목코드"] = df["종목코드"].str.zfill(6)
    uni = df[["종목코드", "종목명"]].drop_duplicates()
    return list(zip(uni["종목코드"], uni["종목명"]))


def get_shares(code, year, rcode):
    """주식의 총수 현황 보고서에서 보통주 발행주식총수 추출.
    실패하면 None (그 분기 EPS/BPS는 비움 → score 에서 중립 50 처리)."""
    try:
        rpt = dart.report(code, "주식총수", year, reprt_code=rcode)
        if rpt is None or len(rpt) == 0:
            return None
        df = rpt.copy()
        df.columns = [c.strip() for c in df.columns]

        cand_col = None
        for c in ["istc_totqy", "distb_stock_co", "now_to_isu_stock_totqy"]:
            if c in df.columns:
                cand_col = c
                break
        if cand_col is None:
            return None

        se_col = "se" if "se" in df.columns else None
        val = None
        if se_col:
            for kw in ["보통주", "합계"]:
                row = df[df[se_col].astype(str).str.contains(kw, na=False)]
                if not row.empty:
                    v = to_num(row.iloc[0].get(cand_col))
                    if v and v > 0:
                        val = v
                        break
        if val is None:
            for _, r in df.iterrows():
                v = to_num(r.get(cand_col))
                if v and v > 0:
                    val = v
                    break
        return val
    except Exception:
        return None


def collect_one(code, name):
    rows = []
    this_year = datetime.now().year
    for year in range(START_YEAR, this_year + 1):
        for rcode, qlabel in REPRTS.items():
            try:
                fs = dart.finstate(code, year, reprt_code=rcode)
                if fs is None or len(fs) == 0:
                    continue
                if "fs_div" in fs.columns:
                    cfs = fs[fs["fs_div"] == "CFS"]
                    fs = cfs if not cfs.empty else fs

                revenue = pick(fs, ["매출액", "수익(매출액)", "영업수익"])
                op = pick(fs, ["영업이익"])
                net = pick(fs, ["당기순이익"])
                assets = pick(fs, ["자산총계"])
                liab = pick(fs, ["부채총계"])
                equity = pick(fs, ["자본총계"])

                debt_ratio = (liab / equity * 100) if (liab and equity) else None
                op_margin = (op / revenue * 100) if (op and revenue) else None
                net_margin = (net / revenue * 100) if (net and revenue) else None
                roe = (net / equity * 100) if (net and equity) else None

                shares = get_shares(code, year, rcode)
                eps = (net / shares) if (net and shares) else None
                bps = (equity / shares) if (equity and shares) else None

                avail = f"{year+1}{AVAIL[qlabel]}" if qlabel == "4Q" else f"{year}{AVAIL[qlabel]}"

                rows.append({
                    "종목코드": code, "종목명": name,
                    "연도": year, "분기": qlabel, "사용가능일": avail,
                    "매출액": revenue, "영업이익": op, "당기순이익": net,
                    "자산총계": assets, "부채총계": liab, "자본총계": equity,
                    "부채비율": debt_ratio, "영업이익률": op_margin,
                    "순이익률": net_margin, "ROE": roe,
                    "주식수": shares, "EPS": eps, "BPS": bps,
                })
                time.sleep(0.3)
            except Exception as e:
                print(f"  {name} {year}{qlabel} 실패: {e}")
                time.sleep(0.3)
    return rows


def flush(buffer):
    if not buffer:
        return
    df = pd.DataFrame(buffer, columns=COLS)
    header = not os.path.exists(OUT)
    df.to_csv(OUT, mode="a", header=header, index=False)


def main():
    universe = get_universe()
    print(f"대상 종목: {len(universe)}개")

    done = set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"종목코드": str})
        done = set(old["종목코드"].astype(str).str.zfill(6).unique())
        print(f"이미 받은 종목: {len(done)}개")

    buffer = []
    for i, (code, name) in enumerate(universe):
        if code in done:
            continue
        rows = collect_one(code, name)
        buffer.extend(rows)
        print(f"[{i+1}/{len(universe)}] {name} {len(rows)}분기")
        if len(buffer) >= 200:
            flush(buffer)
            buffer = []

    if buffer:
        flush(buffer)
    print("DART 재무 수집 완료 (EPS/BPS 포함)")


if __name__ == "__main__":
    main()

