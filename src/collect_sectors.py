"""
업종/섹터 수집기 → data/sectors.csv
- FinanceDataReader의 KRX 상장기업 상세(StockListing 'KRX-DESC')에서 업종(Industry) 취득
- 우리 유니버스(prices.csv 종목)만 필터
- 세부 업종 문자열을 폭넓은 '섹터' 버킷으로 매핑(포트 편중 분석용)
- 검증: 매칭률·섹터 분포 출력

사용: python src/collect_sectors.py     (수동/주기적으로 1회씩. 업종은 자주 안 바뀜)
"""
import os
import pandas as pd

PRICES = "data/prices.csv"
OUT = "data/sectors.csv"

# 세부 업종(Industry) 키워드 → 폭넓은 섹터. 위에서부터 먼저 매칭.
RULES = [
    ("배터리·2차전지", ["이차전지", "일차전지", "전지", "배터리"]),
    ("지주", ["지주", "투자회사", "기타 금융업"]),  # 금융보다 먼저: 지주사는 '기타 금융업'으로 분류됨
    ("금융", ["은행", "보험", "증권", "카드", "금융", "신용", "캐피탈", "저축"]),
    ("반도체", ["반도체", "전자부품", "웨이퍼"]),
    ("IT·전자", ["전자", "전기", "디스플레이", "컴퓨터", "통신장비", "정밀기기", "광학"]),
    ("통신·인터넷", ["통신", "방송", "인터넷", "소프트웨어", "게임", "포털", "정보서비스", "출판"]),
    ("자동차", ["자동차", "타이어", "차체", "운송장비 부품"]),
    ("조선·기계·방산", ["조선", "선박", "중공업", "기계", "항공", "우주", "방위", "무기"]),
    ("화학·에너지", ["화학", "석유", "정유", "에너지", "가스", "전력", "고무", "플라스틱", "비료"]),
    ("바이오·제약", ["제약", "의약", "바이오", "의료", "생명", "연구개발"]),
    ("철강·금속", ["철강", "금속", "비철", "제철", "주조", "합금"]),
    ("건설·건자재", ["건설", "토목", "건축", "시멘트", "요업", "유리"]),
    ("소비·유통", ["식품", "음료", "유통", "소매", "도매", "백화점", "화장품", "생활", "섬유",
                  "의복", "담배", "호텔", "여행", "교육", "가구", "서비스", "중개", "무역", "상사"]),
    ("운송·물류", ["운송", "해운", "물류", "항만", "창고", "택배"]),
]


def to_sector(industry: str) -> str:
    s = str(industry or "")
    for sector, kws in RULES:
        if any(k in s for k in kws):
            return sector
    return "기타"


def main():
    import re
    import FinanceDataReader as fdr
    px = pd.read_csv(PRICES, dtype={"종목코드": str})
    px["종목코드"] = px["종목코드"].str.zfill(6)
    uni = px[["종목코드", "종목명"]].drop_duplicates()
    codes = set(uni["종목코드"])

    full = fdr.StockListing("KRX-DESC")
    full["Code"] = full["Code"].astype(str).str.zfill(6)
    lst = full[full["Code"].isin(codes)]

    def parent_industry(name):
        """우선주는 모주(보통주)의 업종을 상속"""
        base = re.sub(r"(\d?우[BC]?|우선주)$", "", name).strip()
        m = full[full["Name"] == base]
        if len(m) and pd.notna(m["Industry"].iloc[0]):
            return str(m["Industry"].iloc[0])
        return ""

    rows = []
    for _, u in uni.iterrows():
        code = u["종목코드"]; nm = u["종목명"]
        m = lst[lst["Code"] == code]
        industry = str(m["Industry"].iloc[0]) if len(m) and pd.notna(m["Industry"].iloc[0]) else ""
        if not industry:
            industry = parent_industry(nm)
        sector = to_sector(industry)
        rows.append({"종목코드": code, "종목명": nm,
                     "업종": industry, "섹터": sector})
    out = pd.DataFrame(rows)

    # 검증
    matched = (out["업종"].astype(str).str.len() > 0).sum()
    print(f"업종 매칭: {matched}/{len(out)}종목")
    print("섹터 분포:")
    for sec, n in out["섹터"].value_counts().items():
        print(f"   {sec}: {n}")
    unknown = out[out["섹터"] == "기타"]
    if len(unknown):
        print(f"⚠️ '기타'로 분류된 {len(unknown)}종목 (업종 확인 필요):")
        for _, r in unknown.iterrows():
            print(f"   {r['종목명']}: '{r['업종']}'")

    os.makedirs("data", exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"저장: {OUT} ({len(out)}행)")


if __name__ == "__main__":
    main()
