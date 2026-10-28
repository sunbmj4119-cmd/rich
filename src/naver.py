"""
네이버 금융 — KRX(data.krx.co.kr)가 막혔을 때 쓰는 대체 자료원 (로그인·키 없음, 무료)

왜 필요한가
  2026-10-27 daily 실행 중 KRX 가 GitHub 서버 IP 를 차단했다 (응답이 '에러페이지 - 한국거래소' HTML).
  그날 OHLCV·시총·ETF 일봉이 0건이 되어 단기 모의투자가 10-23 에 멈췄다.
  네이버 차트 자료는 종목·ETF 모두 일봉(시가·고가·저가·종가·거래량)을 한 번 요청으로 수천 일까지 준다.

  daily(code, count)      일봉 (수정주가 기준 — 최근 며칠을 덧붙이는 용도로는 원본과 같다)
  investor(code, pages)   외국인·기관 순매매 '수량' (종목 화면의 '외국인·기관' 표) — 금액은 수량 × 종가로 근사

  python src/naver.py --selftest     # 접속·형식 점검 (GitHub Actions 에서 실행)
"""
import re
import sys
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Referer": "https://finance.naver.com/"}
CHART = "https://fchart.stock.naver.com/sise.nhn"
FRGN = "https://finance.naver.com/item/frgn.naver"


def _get(url, params, tries=3):
    err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=15)
            if r.status_code == 200 and r.content:
                return r
            err = f"HTTP {r.status_code}"
        except Exception as e:                       # 네트워크 오류는 잠깐 쉬고 다시
            err = f"{type(e).__name__}: {str(e)[:80]}"
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"네이버 요청 실패 {url} {params}: {err}")


def parse_chart(text):
    """<item data="20261027|시가|고가|저가|종가|거래량" /> 목록 → DataFrame"""
    rows = []
    for d in re.findall(r'data="([^"]+)"', text):
        p = d.split("|")
        if len(p) < 6 or not re.fullmatch(r"\d{8}", p[0]):
            continue
        try:
            o, h, l, c, v = (float(x) if x not in ("", "null") else float("nan") for x in p[1:6])
        except ValueError:
            continue
        rows.append({"날짜": f"{p[0][:4]}-{p[0][4:6]}-{p[0][6:]}", "시가": o, "고가": h, "저가": l, "종가": c, "거래량": v})
    return pd.DataFrame(rows, columns=["날짜", "시가", "고가", "저가", "종가", "거래량"])


def daily(code, count=30):
    r = _get(CHART, {"symbol": code, "timeframe": "day", "count": int(count), "requestType": 0})
    df = parse_chart(r.content.decode("euc-kr", errors="ignore"))
    m = re.search(r'name="([^"]*)"', r.content.decode("euc-kr", errors="ignore"))
    df.attrs["name"] = m.group(1) if m else ""
    return df


def _num(s):
    s = s.replace(",", "").replace("%", "").replace("+", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_investor(html):
    """종목 '외국인·기관' 표 한 쪽 → [날짜, 종가, 기관, 외국인] (수량). 열은 끝에서부터 센다:
       … 거래량 | 기관 순매매량 | 외국인 순매매량 | 외국인 보유주수 | 보유율"""
    rows = []
    for chunk in html.split("<tr")[1:]:
        m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", chunk)
        if not m:
            continue
        vals = [v.strip() for v in re.findall(r"<span[^>]*>\s*([^<]*?)\s*</span>", chunk)]
        vals = [v for v in vals if v != ""]
        if len(vals) < 7 or not re.match(r"\d{4}\.\d{2}\.\d{2}", vals[0]):
            continue
        rows.append({"날짜": f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "종가": _num(vals[1]),
                     "기관": _num(vals[-4]), "외국인": _num(vals[-3])})
    return pd.DataFrame(rows, columns=["날짜", "종가", "기관", "외국인"])


def investor(code, pages=1):
    out = []
    for p in range(1, pages + 1):
        r = _get(FRGN, {"code": code, "page": p})
        out.append(parse_investor(r.content.decode("euc-kr", errors="ignore")))
        time.sleep(0.2)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["날짜", "종가", "기관", "외국인"])
    return df.drop_duplicates("날짜")


def selftest():
    ok = True
    for code in ("069500", "005930", "114800"):
        try:
            d = daily(code, 5)
            print(f"  일봉 {code} {d.attrs.get('name', '')}: {len(d)}행 · 마지막 {d.iloc[-1].to_dict() if len(d) else '-'}")
            ok &= len(d) > 0
        except Exception as e:
            print(f"  일봉 {code} 실패: {e}")
            ok = False
    try:
        f = investor("005930")
        print(f"  외국인·기관 005930: {len(f)}행 · 첫 줄 {f.iloc[0].to_dict() if len(f) else '-'}")
        ok &= len(f) > 0
    except Exception as e:
        print(f"  외국인·기관 실패: {e}")
        ok = False
    print("네이버 점검:", "정상" if ok else "문제 있음")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
