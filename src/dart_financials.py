"""
DART 분기 재무 수집 (1회 백필 + 매일 최신분기 갱신 겸용)
- 종목별로 매출/영업이익/순이익/자산/부채/자본을 분기 단위로 받아
  파생지표(부채비율, 영업이익률, 순이익률)를 계산해 CSV로 저장한다.
- 가격 데이터와 갱신 주기가 다르므로 data/financials.csv 로 분리 저장.
- 점수 계산 시 거래일별로 '그 시점 이전 최신 분기'를 as-of join 한다(미래참조 차단).
"""
import os
import time
import pandas as pd
from datetime import datetime
import OpenDartReader
from pykrx import stock

API_KEY = os.environ["DART_API_KEY"]
dart = OpenDartReader(API_KEY)

os.makedirs("data", exist_ok=True)
OUT = "data/financials.csv"

# 분기 보고서 코드
REPRTS = {
    "11013": "1Q",   # 1분기보고서
    "11012": "2Q",   # 반기보고서(누적 2Q)
    "11014": "3Q",   # 3분기보고서
    "11011": "4Q",   # 사업보고서(연간)
}

# 발표 시점 근사 (보고서 마감일 기준, 미래참조 방지용 '사용가능일')
# 실제 공시일은 종목마다 다르나, 보수적으로 분기말+약 3개월 후로 잡는다.
AVAIL = {
    "1Q": "-05-15",
    "2Q": "-08-15",
    "3Q": "-11-15",
    "4Q": "-04-01",  # 다음해 4월 (사업보고서)
}

START_YEAR = 2016  # DART 재무 안정 제공 시작


def to_num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None


def pick(df, account_names):
    """주요계정 df에서 계정명으로 당기금액 추출"""
    for nm in account_names:
        row = df[df["account_nm"].str.contains(nm, na=False)]
        if not row.empty:
            return to_num(row.iloc[0].get("thstrm_amount"))
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
                # 연결재무제표(CFS) 우선, 없으면 개별
                if "fs_div" in fs.columns:
                    cfs = fs[fs["fs_div"] == "CFS"]
                    fs = cfs if not cfs.empty else fs

                revenue = pick(fs, ["매출액", "수익(매출액)", "영업수익"])
                op = pick(fs, ["영업이익"])
                net = pick(fs, ["당기순이익"])
                assets = pick(fs, ["자산총계"])
                liab = pick(fs, ["부채총계"])
                equity = pick(fs, ["자본총계"])

                # 파생지표
                debt_ratio = (liab / equity * 100) if (liab and equity) else None
                op_margin = (op / revenue * 100) if (op and revenue) else None
                net_margin = (net / revenue * 100) if (net and revenue) else None
                roe = (net / equity * 100) if (net and equity) else None

                avail = f"{year}{AVAIL[qlabel]}" if qlabel != "4Q" else f"{year+1}{AVAIL[qlabel]}"

                rows.append({
                    "종목코드": code, "종목명": name,
                    "연도": year, "분기": qlabel, "사용가능일": avail,
                    "매출액": revenue, "영업이익": op, "당기순이익": net,
                    "자산총계": assets, "부채총계": liab, "자본총계": equity,
                    "부채비율": debt_ratio, "영업이익률": op_margin,
                    "순이익률": net_margin, "ROE": roe,
                })
                time.sleep(0.3)  # DART rate limit 배려
            except Exception as e:
                print(f"  {name} {year}{qlabel} 실패: {e}")
                time.sleep(0.3)
    return rows


def main():
    tickers = stock.get_index_portfolio_deposit_file("1028")  # 코스피200
    names = {t: stock.get_market_ticker_name(t) for t in tickers}
    print(f"대상 종목: {len(tickers)}개")

    # 이어받기
    done = set()
    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype={"종목코드": str})
        done = set(old["종목코드"].unique())
        print(f"이미 받은 종목: {len(done)}개")

    buffer = []
    for i, code in enumerate(tickers):
        if code in done:
            continue
        rows = collect_one(code, names[code])
        buffer.extend(rows)
        print(f"[{i+1}/{len(tickers)}] {names[code]} {len(rows)}분기")

        if len(buffer) >= 200:
            _flush(buffer)
            buffer = []

    if buffer:
        _flush(buffer)
    print("DART 재무 수집 완료")


def _flush(buffer):
    df = pd.DataFrame(buffer)
    header = not os.path.exists(OUT)
    df.to_csv(OUT, mode="a", header=header, index=False)


if __name__ == "__main__":
    main()

