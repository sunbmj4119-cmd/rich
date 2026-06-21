import os
import json
from datetime import datetime
import gspread
from pykrx import stock

# 1) 키 꺼내서 시트 연결
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

# 3) 기존 데이터 읽어서 (날짜,코드) 조합 모으기 = 중복 체크용
existing = ws.get_all_values()

# 헤더가 없으면 새로 깔기
if not existing or existing[0] != ["날짜", "종목코드", "종목명", "종가"]:
    ws.clear()
    ws.append_row(["날짜", "종목코드", "종목명", "종가"])
    existing = ws.get_all_values()

# 이미 들어있는 (날짜, 코드) 집합
already = set()
for r in existing[1:]:            # 헤더 빼고
    if len(r) >= 2:
        already.add((r[0], r[1]))

# 4) 종가 가져와서, 중복 아닌 것만 모으기
today = datetime.now().strftime("%Y%m%d")
new_rows = []
for code, name in stocks.items():
    try:
        df = stock.get_market_ohlcv(today, today, code)
        if df.empty:
            df = stock.get_market_ohlcv("20240101", today, code).tail(1)
        close = int(df["종가"].iloc[-1])
        date = df.index[-1].strftime("%Y-%m-%d")

        if (date, code) in already:
            print(f"건너뜀(중복): {name} {date}")
            continue

        new_rows.append([date, code, name, close])
        print(f"{name}: {close}")
    except Exception as e:
        print(f"{name} 실패: {e}")

# 5) 새 데이터만 한 번에 추가
if new_rows:
    ws.append_rows(new_rows)
    print(f"저장 완료: {len(new_rows)}개")
else:
    print("새로 추가할 데이터 없음 (전부 중복)")
