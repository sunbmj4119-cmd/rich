import os
import json
import time
from datetime import datetime
import gspread
from pykrx import stock

# 시트 연결
creds = json.loads(os.environ["GOOGLE_CREDENTIALS"])
sheet_id = os.environ["SHEET_ID"]
gc = gspread.service_account_from_dict(creds)
sh = gc.open_by_key(sheet_id)
ws = sh.sheet1

stocks = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "005380": "현대차",
}

HEADER = ["날짜", "종목코드", "종목명", "종가", "거래량"]

# 헤더 점검 (기존 데이터 보존 - clear 안 함)
existing = ws.get_all_values()
if not existing or existing[0] != HEADER:
    ws.clear()
    ws.append_row(HEADER)
    existing = ws.get_all_values()

# 이미 들어있는 (날짜,코드) = 이어받기/중복방지
already = set()
for r in existing[1:]:
    if len(r) >= 2:
        already.add((r[0], r[1]))

START = "20180101"
END = datetime.now().strftime("%Y%m%d")

for code, name in stocks.items():
    try:
        df = stock.get_market_ohlcv(START, END, code)
        if df.empty:
            print(f"{name}: 데이터 없음")
            continue

        rows = []
        for idx, row in df.iterrows():
            date = idx.strftime("%Y-%m-%d")
            if (date, code) in already:
                continue
            rows.append([date, code, name, int(row["종가"]), int(row["거래량"])])

        # 1000줄씩 끊어서 저장 (안정성)
        for i in range(0, len(rows), 1000):
            ws.append_rows(rows[i:i+1000])
            time.sleep(1)

        print(f"{name}: {len(rows)}줄 저장")
        time.sleep(2)  # KRX 서버 배려

    except Exception as e:
        print(f"{name} 실패: {e}")

print("백필 완료")
