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

# ── 감성 사전 (정규식) ────────────────────────────────────────
# 첫 CI 수집 155건을 사람이 읽고 고친 판이다. 단순 부분일치는 실패가 많았다:
#   "외국인 5일 연속 순매수행진" ← '외국인 순매수'로는 안 잡힌다(사이에 말이 낀다)
#   "목표주가 2배 넘게 벌어진"   ← '목표주가 상향'이 아닌데 상향으로 오인될 수 있다
# 그래서 사이에 낄 수 있는 말을 허용하는 정규식으로 바꿨다.
# 가중치: 2 = 주가에 직접 꽂히는 사건, 1 = 방향성 있는 일반 기사
POS = {
    2: [r"신고가", r"흑자전환", r"어닝\s?서프라이즈", r"실적\s?서프라이즈",
        r"대규모\s?수주", r"수주\s?(확보|잭팟)", r"자사주\s?(매입|취득|소각)",
        r"배당\s?(확대|인상|증액)", r"(목표주?가|목표가)[^가-힣]{0,8}상향",
        r"투자의견[^가-힣]{0,8}(상향|매수로)", r"상한가", r"급등", r"최대\s?실적",
        r"사상\s?최대", r"수출\s?확대", r"무상증자"],
    1: [r"수주", r"공급\s?계약", r"계약\s?(체결|수주)", r"상승", r"강세", r"호실적",
        r"실적\s?개선", r"이익\s?(증가|급증)", r"매출\s?(증가|급증)", r"신제품", r"증설",
        r"인증\s?획득", r"특허", r"점유율\s?확대", r"배당\s?결정",
        r"외국인[^가-힣]{0,10}순매수", r"기관[^가-힣]{0,10}순매수", r"순매수\s?(행진|세)",
        r"매수세\s?유입", r"저평가", r"수혜", r"반등", r"회복", r"호평", r"극찬",
        r"목표주?가\s?(상향|올려)"],
}
NEG = {
    2: [r"유상증자", r"횡령", r"배임", r"분식회계", r"상장폐지", r"거래정지",
        r"감사의견\s?거절", r"적자전환", r"어닝\s?쇼크", r"실적\s?쇼크", r"리콜",
        r"영업정지", r"블록딜", r"(목표주?가|목표가)[^가-힣]{0,8}하향",
        r"투자의견[^가-힣]{0,8}하향", r"하한가", r"급락", r"폭락", r"화재", r"파업",
        r"소송\s?패소", r"과징금", r"압수수색", r"구속", r"전환사채", r"감자\s?결정"],
    1: [r"하락", r"약세", r"부진", r"실적\s?악화", r"이익\s?감소", r"매출\s?감소",
        r"적자", r"손실\s?확대", r"감산", r"수주\s?취소", r"계약\s?해지", r"소송",
        r"제재", r"조사\s?착수", r"우려", r"경고", r"외국인[^가-힣]{0,10}순매도",
        r"기관[^가-힣]{0,10}순매도", r"순매도\s?(행진|세)", r"고평가", r"차익실현",
        r"규제", r"불확실", r"장내\s?매도", r"지분[^가-힣]{0,6}(감소|축소)", r"반토막"],
}
# 강한 악재(가중치 2)는 대시보드에서 '반대논리'로 승격된다
STRONG_NEG_RE = re.compile("|".join(NEG[2]))

# 정보가 거의 없는 자동생성·콘텐츠팜 기사 (CI 1회 수집에서 확인된 패턴)
JUNK = re.compile(
    r"투자분석\s*\d{4}|주가\s?분석\s*\d{4}|살펴볼\s?적기|"
    r"기업주식정보|오늘의?\s?급등주|추천주\s?TOP|무료\s?추천|"
    r"운세|로또|스타뉴스")

_POS_RE = {w: re.compile("|".join(p)) for w, p in POS.items()}
_NEG_RE = {w: re.compile("|".join(p)) for w, p in NEG.items()}

# 종목명 뒤에 붙으면 '다른 회사'가 되는 꼬리표
#  예) "KCC" 검색에 "KCC건설" 기사가, "아모레퍼시픽홀딩스"에 "…3우C"가 섞여 들어온다
SUFFIX_RE = (r"(?:건설|홀딩스|지주|생명|증권|화재|해상|케미칼|에너지솔루션|"
             r"이노베이션|바이오|퓨처엠|인터내셔널|\d?우[A-C]?)")


def _score_title(title: str):
    """제목 → (감성점수, 매칭 근거). 점수는 -5~+5."""
    t = re.sub(r"\s+", " ", title)
    sc, hits = 0, []
    for w, rx in _POS_RE.items():
        found = {m.group(0) for m in rx.finditer(t)}
        for f in found:
            sc += w
            hits.append("+" + f)
    for w, rx in _NEG_RE.items():
        found = {m.group(0) for m in rx.finditer(t)}
        for f in found:
            sc -= w
            hits.append("-" + f)
    return max(-5, min(5, sc)), hits


def _relevant(title: str, name: str, all_names=None) -> bool:
    """
    이 기사가 정말 이 종목 기사인가.

    종목명이 제목에 없다고 버리지는 않는다 — 기사는 그 종목을 검색해 받아온 것이고
    제목에서 이름을 줄여 쓰는 경우가 많다("SK하이닉스" → "하이닉스", "삼전닉스").
    버리는 건 두 경우뿐이다:
      · 정보가 없는 자동생성·광고성 기사
      · 이름이 **오직 다른 법인의 일부로만** 등장하는 기사 (KCC 검색 → KCC건설 기사)
    """
    if JUNK.search(title):
        return False
    if name in title:
        # 이름 뒤에 꼬리표가 안 붙은 '순수' 등장이 하나라도 있으면 이 종목 기사다
        plain = re.search(re.escape(name) + r"(?!" + SUFFIX_RE + r")", title)
        if not plain:
            return False
    return True


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


# ── DART 공시 분류 ────────────────────────────────────────────
# 공시 제목은 법정 서식이라 표기가 일정하다 → 키워드 매칭 정확도가 기사보다 훨씬 높다.
DART_NEG = {
    r"유상증자결정": (-2, "유상증자 — 주식 수가 늘어 기존 주주 지분이 희석된다"),
    r"전환사채권?발행결정": (-2, "전환사채 — 나중에 주식으로 바뀌어 희석 요인"),
    r"신주인수권부사채권?발행결정": (-2, "신주인수권부사채 — 희석 요인"),
    r"감자결정": (-2, "감자 — 자본 구조 악화 신호일 수 있다"),
    r"횡령|배임": (-2, "횡령·배임 — 지배구조 리스크"),
    r"소송[등]?의?\s?제기": (-1, "소송 제기"),
    r"불성실공시법인지정": (-2, "불성실공시 — 공시 신뢰도 문제"),
    r"관리종목지정|상장폐지": (-2, "관리종목·상장폐지 관련"),
    r"영업정지|영업양도": (-1, "영업 관련 중대 변경"),
}
DART_POS = {
    r"자기주식취득": (2, "자사주 취득 — 유통 물량이 줄어든다"),
    r"자기주식소각": (2, "자사주 소각 — 주식 수가 영구히 줄어든다"),
    r"현금ㆍ?현물배당결정|현금배당": (1, "배당 결정"),
    r"단일판매ㆍ?공급계약체결": (2, "대규모 공급계약 체결"),
    r"무상증자결정": (1, "무상증자"),
    r"주식분할": (1, "액면분할"),
}
_DN = [(re.compile(k), v) for k, v in DART_NEG.items()]
_DP = [(re.compile(k), v) for k, v in DART_POS.items()]


def _collect_dart(targets, rows):
    """
    최근 공시를 rows에 추가. 출처는 'DART'로 표시해 대시보드가 구분한다.

    주의: kind를 지정하지 말 것.
      kind="A"는 **정기공시(사업·분기보고서)** 만 준다. 우리가 보고 싶은
      유상증자·자사주취득·공급계약은 주요사항보고(B)·거래소공시(I)라
      A로 조회하면 30종목 전부 "013 조회된 데이타가 없습니다"가 돌아온다.
      (첫 CI 실행에서 실제로 그렇게 나왔다.)
      전체를 받아 제목 키워드로 거르는 편이 단순하고 빠뜨림도 없다.
    """
    import io
    import contextlib
    import OpenDartReader
    with contextlib.redirect_stdout(io.StringIO()):     # 라이브러리가 찍는 오류 JSON 억제
        dart = OpenDartReader(os.environ["DART_API_KEY"])
    since = (datetime.now(KST) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    today = datetime.now(KST).strftime("%Y-%m-%d")
    added = hit = 0
    for code, name in targets.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                lst = dart.list(code, start=since, end=today)   # kind 미지정 = 전체
        except Exception:
            continue
        if lst is None or len(lst) == 0:
            continue
        hit += 1
        picked = 0
        for _, r in lst.iterrows():
            if picked >= 5:
                break
            title = str(r.get("report_nm", "")).strip()
            if not title:
                continue
            sc, why = 0, ""
            for rx, (v, txt) in _DN + _DP:
                if rx.search(title):
                    sc, why = v, txt
                    break
            if sc == 0:
                continue           # 정기보고서 등 방향성 없는 공시는 생략
            rcp = str(r.get("rcept_no", ""))
            rows.append({"종목코드": code, "종목명": name, "제목": f"[공시] {title}",
                         "링크": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp}",
                         "발행일": str(r.get("rcept_dt", ""))[:10], "출처": "DART",
                         "감성": sc, "키워드": why})
            added += 1
            picked += 1
        time.sleep(0.12)
    print(f"  DART: {hit}/{len(targets)}종목 공시 조회 성공 · 방향성 있는 공시 {added}건")
    return added


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

    all_names = set(targets.values())
    rows = []
    ok = fail = dropped = 0
    for code, name in targets.items():
        try:
            arts = _parse_rss(_fetch(f"{name} 주가"))
            ok += 1
        except Exception:
            fail += 1
            arts = []
        kept = 0
        for a in arts:
            if kept >= PER_STOCK:
                break
            if not _relevant(a["title"], name, all_names):
                dropped += 1
                continue
            sc, hits = _score_title(a["title"])
            rows.append({"종목코드": code, "종목명": name, "제목": a["title"],
                         "링크": a["link"], "발행일": a["pub"], "출처": a["source"],
                         "감성": sc, "키워드": " ".join(hits[:6])})
            kept += 1
        time.sleep(SLEEP)

    # ── DART 공시 (있으면 최우선 신호) ─────────────────────────
    # 기사 제목은 기자가 쓴 해석이지만, 공시는 회사가 낸 사실이다.
    # 유상증자·자사주소각 같은 건 '뉴스'가 아니라 확정된 사건이라 훨씬 신뢰도가 높다.
    dart_n = 0
    if os.environ.get("DART_API_KEY"):
        try:
            dart_n = _collect_dart(targets, rows)
        except Exception as e:
            print(f"  DART 공시 수집 실패(무시): {type(e).__name__} {e}")

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
    # 예전 판으로 수집해둔 자동생성·타사 기사도 이 기회에 걸러낸다(사전이 개선되면 소급 적용)
    before = len(df)
    keep = df.apply(lambda r: (str(r["출처"]) == "DART" or str(r["종목코드"]) == "MARKET"
                               or _relevant(str(r["제목"]), str(r["종목명"]))), axis=1)
    df = df[keep]
    cleaned = before - len(df)
    # 보관기간 정리 (발행일 파싱 실패 건은 남긴다)
    cut = (datetime.now(KST) - timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%d")
    keep = df["발행일"].astype(str).str[:10]
    df = df[(keep >= cut) | (keep.str.len() < 10)]
    df = df.sort_values(["종목코드", "발행일"], ascending=[True, False])
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)

    strong = sum(1 for _, r in new.iterrows()
                 if STRONG_NEG_RE.search(str(r["제목"]))) if len(new) else 0
    print(f"뉴스: 신규 {len(new)}건(공시 {dart_n}) · 누적 {len(df)}건 · "
          f"조회성공 {ok}/실패 {fail} · 무관·광고 제외 {dropped}건"
          + (f" · 과거분 정리 {cleaned}건" if cleaned else "")
          + f" · 강한 악재 {strong}건")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                       # 파이프라인은 절대 죽이지 않는다
        print(f"뉴스 수집 실패(무시): {type(e).__name__} {e}", file=sys.stderr)
