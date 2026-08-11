"""
종목별 최신 뉴스 수집 -> data/news.csv

목적: 숫자(팩터)만으로는 안 보이는 '사건'을 투자 논리 점검에 넣는다.
  - 점수엔 반영하지 않는다(감성 팩터는 검증 전). 오직 '내 논리를 반박할 뉴스가 있나?' 체크용.
  - 기사 제목만 키워드 사전으로 채점 → 강한 부정어가 있으면 대시보드에 반대논리로 승격.

설계 원칙
  - 무료·무인증(Google News RSS)만 사용. 키 없이 Actions에서 동작.
  - 네트워크 실패는 절대 파이프라인을 죽이지 않는다(빈 파일이라도 남기고 종료 0).
  - 전 종목이 아니라 '오늘 판단이 필요한 종목'만 조회(보유·신호·상위권) → 요청 수 절감.
  - 누적 저장하되 30일 지난 기사는 버려 파일 크기를 묶어둔다.
"""
import os
import re
import sys
import csv
import time
import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pandas as pd

SCORES = "data/scores.csv"
SIGNALS = "data/signals_today.csv"
OUT = "data/news.csv"

KST = timezone(timedelta(hours=9))
KEEP_DAYS = 30        # news.csv 보관 기간
FRESH_DAYS = 14       # 대시보드에 '최신'으로 취급할 기간
PER_STOCK = 6         # 종목당 저장 기사 수
TOP_SCORE_N = 30      # 점수 상위 몇 종목까지 조회할지
TIMEOUT = 12
SLEEP = 0.35          # 요청 간 간격(예의 + 차단 회피)
UA = "Mozilla/5.0 (compatible; rich-dashboard/1.0; +https://github.com/)"

# ── 감성 키워드 사전 ──────────────────────────────────────────
# 가중치: 2 = 주가에 직접적인 강한 사건, 1 = 방향성 있는 일반 뉴스
POS = {
    2: ["신고가", "흑자전환", "어닝서프라이즈", "실적 서프라이즈", "대규모 수주", "수주 확보",
        "자사주 매입", "자사주 소각", "배당 확대", "배당 인상", "목표주가 상향", "목표가 상향",
        "투자의견 상향", "상한가", "급등", "최대 실적", "사상 최대", "인수 성공", "수출 확대"],
    1: ["수주", "계약 체결", "상승", "강세", "호실적", "실적 개선", "이익 증가", "매출 증가",
        "성장", "신제품", "증설", "공급 계약", "인증 획득", "특허", "점유율 확대",
        "외국인 순매수", "기관 순매수", "저평가", "수혜", "반등", "회복", "낙관"],
}
NEG = {
    2: ["유상증자", "횡령", "배임", "분식회계", "상장폐지", "거래정지", "감사의견 거절",
        "적자전환", "어닝쇼크", "실적 쇼크", "리콜", "영업정지", "블록딜", "대량 매도",
        "목표주가 하향", "목표가 하향", "투자의견 하향", "하한가", "급락", "폭락",
        "화재", "사고", "파업", "소송 패소", "과징금", "압수수색", "구속"],
    1: ["하락", "약세", "부진", "실적 악화", "이익 감소", "매출 감소", "적자", "손실 확대",
        "감산", "수주 취소", "계약 해지", "소송", "제재", "조사 착수", "우려", "경고",
        "외국인 순매도", "기관 순매도", "고평가", "차익실현", "규제", "불확실"],
}
# 강한 부정어(가중치 2)는 대시보드에서 '반대논리'로 승격 → 그 목록
STRONG_NEG = set(NEG[2])


def _score_title(title: str):
    """제목 → (감성점수, 매칭 키워드 리스트). 점수는 -5~+5로 클립."""
    t = re.sub(r"\s+", " ", title)
    sc = 0
    hits = []
    for w, words in POS.items():
        for kw in words:
            if kw in t:
                sc += w
                hits.append("+" + kw)
    for w, words in NEG.items():
        for kw in words:
            if kw in t:
                sc -= w
                hits.append("-" + kw)
    return max(-5, min(5, sc)), hits


def _parse_rss(xml_text: str):
    """Google News RSS → [{title, link, pub, source}]"""
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for it in root.iter("item"):
        def g(tag):
            e = it.find(tag)
            return (e.text or "").strip() if e is not None and e.text else ""
        title = html.unescape(g("title"))
        if not title:
            continue
        # Google News 제목은 "제목 - 언론사" 형식 → 언론사 꼬리표를 제목에서 떼어낸다
        source = ""
        se = it.find("source")
        if se is not None and se.text:
            source = se.text.strip()
        if " - " in title:
            head, tail = title.rsplit(" - ", 1)
            if not source:
                title, source = head, tail
            elif tail == source:
                title = head
        pub = ""
        raw = g("pubDate")
        if raw:
            for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    pub = dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")
                    break
                except ValueError:
                    continue
        out.append({"title": title.strip(), "link": g("link"), "pub": pub, "source": source})
    return out


def _fetch(query: str):
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def _targets():
    """오늘 판단이 필요한 종목만 고른다: 신호 있는 종목 + 점수 상위 N."""
    s = pd.read_csv(SCORES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종목명", "종합점수"])
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])
    cur = s[s["날짜"] == s["날짜"].max()].sort_values("종합점수", ascending=False)
    picked = {}
    if os.path.exists(SIGNALS):
        try:
            sg = pd.read_csv(SIGNALS, dtype={"종목코드": str})
            for _, r in sg.iterrows():
                picked[str(r["종목코드"]).zfill(6)] = str(r["종목명"])
        except Exception:
            pass
    for _, r in cur.head(TOP_SCORE_N).iterrows():
        picked.setdefault(r["종목코드"], r["종목명"])
    return picked


def _load_existing():
    if not os.path.exists(OUT):
        return pd.DataFrame()
    try:
        return pd.read_csv(OUT, dtype={"종목코드": str})
    except Exception:
        return pd.DataFrame()


def main():
    os.makedirs("data", exist_ok=True)
    try:
        targets = _targets()
    except Exception as e:
        print(f"뉴스: 대상 선정 실패 → 건너뜀 ({e})")
        return

    rows = []
    ok = fail = 0
    for code, name in targets.items():
        try:
            arts = _parse_rss(_fetch(f"{name} 주가"))
            ok += 1
        except Exception:
            fail += 1
            arts = []
        for a in arts[:PER_STOCK]:
            sc, hits = _score_title(a["title"])
            rows.append({"종목코드": code, "종목명": name, "제목": a["title"],
                         "링크": a["link"], "발행일": a["pub"], "출처": a["source"],
                         "감성": sc, "키워드": " ".join(hits[:6])})
        time.sleep(SLEEP)

    # 시장 전체 뉴스 (종목코드 MARKET)
    try:
        for a in _parse_rss(_fetch("코스피 증시 전망"))[:8]:
            sc, hits = _score_title(a["title"])
            rows.append({"종목코드": "MARKET", "종목명": "시장", "제목": a["title"],
                         "링크": a["link"], "발행일": a["pub"], "출처": a["source"],
                         "감성": sc, "키워드": " ".join(hits[:6])})
    except Exception:
        fail += 1

    new = pd.DataFrame(rows)
    old = _load_existing()
    if len(new) == 0 and len(old) == 0:
        # 헤더만이라도 남겨 build_data가 안전하게 읽도록
        pd.DataFrame(columns=["종목코드", "종목명", "제목", "링크", "발행일", "출처", "감성", "키워드"]) \
          .to_csv(OUT, index=False)
        print(f"뉴스: 0건 (성공 {ok} / 실패 {fail}) — 네트워크 차단 환경일 수 있음")
        return

    df = pd.concat([old, new], ignore_index=True) if len(old) else new
    df["종목코드"] = df["종목코드"].astype(str)
    df = df.drop_duplicates(subset=["종목코드", "제목"], keep="last")
    # 보관기간 정리 (발행일 파싱 실패 건은 남긴다)
    cut = (datetime.now(KST) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    keep = df["발행일"].astype(str).str[:10]
    df = df[(keep >= cut) | (keep.str.len() < 10)]
    df = df.sort_values(["종목코드", "발행일"], ascending=[True, False])
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)

    strong = sum(1 for _, r in new.iterrows()
                 if any(k in str(r["제목"]) for k in STRONG_NEG)) if len(new) else 0
    print(f"뉴스: 신규 {len(new)}건 · 누적 {len(df)}건 · 조회성공 {ok}/실패 {fail} · 강한 악재 제목 {strong}건")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                       # 파이프라인은 절대 죽이지 않는다
        print(f"뉴스 수집 실패(무시): {type(e).__name__} {e}", file=sys.stderr)
