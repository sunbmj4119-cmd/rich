import os
import json
import gspread

# GitHub Secrets에서 키 꺼내기
creds = json.loads(os.environ["GOOGLE_CREDENTIALS"])
sheet_id = os.environ["SHEET_ID"]

# 구글 시트 연결
gc = gspread.service_account_from_dict(creds)
sh = gc.open_by_key(sheet_id)
ws = sh.sheet1

# 테스트로 한 줄 쓰기
ws.update("A1", [["연결 성공!"]])
print("시트에 쓰기 완료")
