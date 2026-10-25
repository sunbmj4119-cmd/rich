"""
DART 분기 재무 수집 (1회 백필) — EPS/BPS/주식수 포함 버전
- 종목 리스트는 data/prices.csv 에서 읽음 (KRX 지수 API 회피)
- 손익/재무상태 항목 + 파생지표(부채비율/영업이익률/순이익률/ROE)
- 추가: 주식총수(주식의 총수 보고서) → EPS = 순이익/주식수, BPS = 자본총계/주식수
  · 1Q·3Q 등에서 주식총수 보고서가 비면, 같은 종목의 직전 주식수를 carry-forward
- data/financials.csv 로 분리 저장. 점수 계산 시 as-of join.
- 이어받기: 이미 받은 종목 건너뜀
- 고정 컬럼 스키마(COLS)로 저장 → resume 시 컬럼 어긋남 방지

[갱신 모드] python src/dart_financials.py --update
  점수 대상 종목(우선주 제외)은 최근 2개년 중 '공시 마감이 지났는데 아직 없는 분기'만 다시 받는다 (financials.yml 이 주기 실행).
  같은 (종목, 연도, 분기) 행은 새로 받은 값으로 교체.

[테스트 모드] 환경변수로 범위 축소 (없으면 전체 수집):
  TEST_LIMIT=3        → 상위 3종목만
  TEST_START_YEAR=2024 → 2024년부터만
"""
import os
import sys
import time
import pandas as pd
from datetime import datetime
import OpenDartReader

API_KEY = os.environ["DART_API_KEY"]
dart = OpenDartReader(API_KEY)

os.makedirs("data", exist_ok=True)
OUT = "data/financials.csv"
PRICES = "data/prices.csv"
SCORES = "data/scores.csv"

REPRTS = {"11013": "1Q", "11012": "2Q", "11014": "3Q", "11011": "4Q"}
AVAIL = {"1Q": "-05-15", "2Q": "-08-15", "3Q": "-11-15", "4Q": "-04-01"}
# 분기 순서 (carry-forward 시 직전 주식수 사용)
QORDER = ["1Q", "2Q", "3Q", "4Q"]
START_YEAR = 2016

# 테스트 모드 스위치 (환경변수 없으면 전체)
TEST_LIMIT = int(os.environ.get("TEST_LIMIT", "0"))          # 종목 수 (0=전체)
TEST_START_YEAR = int(os.environ.get("TEST_START_YEAR", str(START_YEAR)))

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
    실패하면 None → 호출부에서 직전 주식수를 carry-forward."""
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


def due_quarters(start_year, today=None):
    """공시 마감(사용가능일)이 지난 (연도, 분기) 목록 — 분기 순서대로."""
    today = today or datetime.now()
    out = []
    for year in range(start_year, today.year + 1):
        for q in QORDER:
            avail = f"{year+1}{AVAIL[q]}" if q == "4Q" else f"{year}{AVAIL[q]}"
            if datetime.strptime(avail, "%Y-%m-%d") <= today:
                out.append((year, q))
    return out


def collect_one(code, name, targets=None, last_shares=None):
    """targets: [(연도, 분기)] — 없으면 TEST_START_YEAR 부터 전부.
    last_shares: 갱신 모드에서 기존 데이터의 마지막 주식수 (carry-forward 시작값)."""
    rows = []
    this_year = datetime.now().year
    if targets is None:
        targets = [(y, q) for y in range(TEST_START_YEAR, this_year + 1) for q in QORDER]
    for year, qlabel in targets:                    # 분기 순서대로 (carry-forward 정확)
        rcode = [k for k, v in REPRTS.items() if v == qlabel][0]
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

            # 주식수: 못 구하면 직전 주식수 사용 (1Q·3Q 빈 분기 보정)
            shares = get_shares(code, year, rcode)
            if shares and shares > 0:
                last_shares = shares
            else:
                shares = last_shares

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


def scored_universe():
    """갱신 대상 = 점수를 매기는 종목(scores.csv 최근일). 우선주(코드 끝자리가 0이 아님)는 DART 재무가 없어 뺀다."""
    if not os.path.exists(SCORES):
        return get_universe()
    s = pd.read_csv(SCORES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종목명"])
    s["종목코드"] = s["종목코드"].str.zfill(6)
    last = s[s["날짜"] == s["날짜"].max()].drop_duplicates("종목코드")
    return [(c, n) for c, n in zip(last["종목코드"], last["종목명"]) if c.endswith("0")]


def save_merged(old, new):
    df = pd.concat([old, pd.DataFrame(new, columns=COLS)], ignore_index=True)
    df = df.drop_duplicates(["종목코드", "연도", "분기"], keep="last").sort_values(["종목코드", "연도", "분기"])
    df.to_csv(OUT, index=False)
    return df


def update():
    """점수 대상 종목의 빠진 최근 분기만 채운다. 새로 편입된 종목은 전체 이력을 받는다.
    10종목마다 저장하므로 시간 초과로 끊겨도 받은 만큼은 남는다."""
    universe = scored_universe()
    if TEST_LIMIT > 0:
        universe = universe[:TEST_LIMIT]
    old = pd.read_csv(OUT, dtype={"종목코드": str}) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    old["종목코드"] = old["종목코드"].astype(str).str.zfill(6)
    have = set(zip(old["종목코드"], old["연도"].astype(int), old["분기"]))
    due = due_quarters(datetime.now().year - 1)
    print(f"대상 {len(universe)}종목 (점수 대상 · 우선주 제외) · 확인 분기 {due[0]}~{due[-1]}")
    new, done = [], 0
    for i, (code, name) in enumerate(universe):
        mine = old[old["종목코드"] == code]
        if mine.empty:
            rows = collect_one(code, name)
        else:
            want = [(y, q) for y, q in due if (code, y, q) not in have]
            if not want:
                continue
            sh = pd.to_numeric(mine.sort_values(["연도", "분기"])["주식수"], errors="coerce").dropna()
            rows = collect_one(code, name, want, float(sh.iloc[-1]) if len(sh) else None)
        new.extend(rows)
        done += 1
        print(f"[{i+1}/{len(universe)}] {name} +{len(rows)}분기")
        if new and done % 10 == 0:
            save_merged(old, new)
    if not new:
        print("새로 받을 분기 없음 (아직 공시 전이거나 이미 최신)")
        return
    df = save_merged(old, new)
    print(f"DART 재무 갱신 완료: +{len(new)}분기 · 최신 사용가능일 {df['사용가능일'].max()}")


def main():
    if "--update" in sys.argv:
        return update()
    universe = get_universe()

    if TEST_LIMIT > 0:
        universe = universe[:TEST_LIMIT]
        print(f"⚡ 테스트 모드: 상위 {TEST_LIMIT}종목, {TEST_START_YEAR}년부터")

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

