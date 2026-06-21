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

# 2) 테스트 종목 5개 (코드: 이름)
stocks = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "005380": "현대차",
}

# 3) 최근 거래일 종가 가져오기
today = datetime.now().strftime("%Y%m%d")
rows = []
for code, name in stocks.items():
    try:
        df = stock.get_market_ohlcv(today, today, code)
        if df.empty:
            # 오늘 데이터 없으면(휴장 등) 최근 5일 중 마지막
            df = stock.get_market_ohlcv("20240101", today, code).tail(1)
        close = int(df["종가"].iloc[-1])
        date = df.index[-1].strftime("%Y-%m-%d")
        rows.append([date, code, name, close])
        print(f"{name}: {close}")
    except Exception as e:
        print(f"{name} 실패: {e}")

# 4) 헤더가 없으면 먼저 넣기
existing = ws.get_all_values()
if not existing or existing[0] != ["날짜", "종목코드", "종목명", "종가"]:
    if not existing:
        ws.append_row(["날짜", "종목코드", "종목명", "종가"])

# 5) 시트 맨 아래에 누적
for row in rows:
    ws.append_row(row)

print(f"저장 완료: {len(rows)}개")
