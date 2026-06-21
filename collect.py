import os
import json
from datetime import datetime
import gspread
from pykrx import stock

# 1) 시트 연결
creds = json.loads(os.environ["GOOGLE_CREDENTIALS"])
sheet_id = os.environ["SHEET_ID"]
gc = gspread.service_account_from_dict(creds)
sh = gc.open_by_key(sheet_id)
ws = sh.sheet1

# 2) 테스트 종목 5개
stocks = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "005380": "현대차",
}

HEADER = ["날짜", "종목코드", "종목명", "종가", "거래량",
          "외국인순매수", "기관순매수", "개인순매수"]

# 3) 기존 데이터 + 헤더 점검
existing = ws.get_all_values()
if not existing or existing[0] != HEADER:
    ws.clear()
    ws.append_row(HEADER)
    existing = ws.get_all_values()

already = set()
for r in existing[1:]:
    if len(r) >= 2:
        already.add((r[0], r[1]))

# 4) 데이터 수집
today = datetime.now().strftime("%Y%m%d")
new_rows = []

for code, name in stocks.items():
    try:
        # 종가 + 거래량
        df = stock.get_market_ohlcv(today, today, code)
        if df.empty:
            df = stock.get_market_ohlcv("20240101", today, code).tail(1)
        close = int(df["종가"].iloc[-1])
        volume = int(df["거래량"].iloc[-1])
        date = df.index[-1].strftime("%Y-%m-%d")
        ymd = df.index[-1].strftime("%Y%m%d")

        if (date, code) in already:
            print(f"건너뜀(중복): {name} {date}")
            continue

        # 투자자별 순매수 (금액)
        foreign = inst = indiv = ""
        try:
            tv = stock.get_market_trading_value_by_date(ymd, ymd, code)
            if not tv.empty:
                row0 = tv.iloc[-1]
                # 컬럼명이 버전 따라 다를 수 있어 안전하게 추출
                def pick(*names):
                    for n in names:
                        if n in tv.columns:
                            return int(row0[n])
                    return ""
                foreign = pick("외국인합계", "외국인")
                inst = pick("기관합계", "기관")
                indiv = pick("개인")
        except Exception as e:
            print(f"  순매수 실패({name}): {e}")

        new_rows.append([date, code, name, close, volume, foreign, inst, indiv])
        print(f"{name}: 종가{close} 거래량{volume} 외인{foreign} 기관{inst}")

    except Exception as e:
        print(f"{name} 실패: {e}")

# 5) 저장
if new_rows:
    ws.append_rows(new_rows)
    print(f"저장 완료: {len(new_rows)}개")
else:
    print("새로 추가할 데이터 없음")
