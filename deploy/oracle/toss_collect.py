"""
토스증권 오픈API 시세·수급 이력 수집 (서버에서 실행) — 조회(GET)만 하고 주문·잔고 API는 쓰지 않는다.

모으는 것 (종목별 일별, until 로 과거까지 넘겨 가며):
  investor-trading(투자자별) · program-trades(프로그램매매) · short-selling(공매도)
  credit-trades(신용) · securities-lending(대차)
  + 시장 전체 투자자별 매매대금(KOSPI, 1d) + 지수 ETF 일봉(KODEX 200·인버스)

저장 위치: 원본은 서버의 ~/.rich/toss/ 에만 둔다 (저장소가 공개라 토스 원자료를 그대로 올리지 않는다).
GitHub 에는 data/toss/collect_status.json (종목 수·기간·필드 이름·요청 수만, 값 없음)만 올린다.

  python toss_collect.py --backfill     # 처음 한 번: 2018년까지 거슬러 (중간에 끊겨도 다시 실행하면 이어서)
  python toss_collect.py                # 매일: 최근 것만 덧붙임
  python toss_collect.py --limit 3      # 시험: 3종목만

응답 필드 이름을 미리 알 수 없어(명세의 records 항목이 비어 있음) 받은 그대로 저장하고,
날짜 필드와 숫자 필드는 자동으로 찾는다 (toss_lab.py 가 같은 규칙으로 읽는다).
"""
import argparse
import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toss_probe as TP  # noqa: E402  — 키 파일 읽기 · 토큰 재사용(클라이언트당 토큰 1개만 유효)

BASE = TP.BASE
RAW = os.path.join(TP.STATE, "toss")
REPO = TP.REPO
STATUS = os.path.join(REPO, "data", "toss", "collect_status.json")
ENDPOINTS = ["investor-trading", "program-trades", "short-selling", "credit-trades", "securities-lending"]
ETFS = {"069500": "KODEX 200", "114800": "KODEX 인버스"}
START = "2018-01-01"
PAUSE = 0.25                     # 초당 4건 이하
DATE_KEYS = re.compile(r"(date|day|baseDt|tradeDt|tradingDate|timestamp)$", re.I)


class Api:
    def __init__(self):
        self.env = TP.load_env()
        self.tok, info = TP.get_token(self.env)
        if not self.tok:
            raise SystemExit(f"토큰 발급 실패: {info}")
        self.n = 0
        self.err = {}

    def get(self, path, params=None):
        for i in range(5):
            time.sleep(PAUSE)
            try:
                r = requests.get(BASE + path, params=params, timeout=20,
                                 headers={"Authorization": f"Bearer {self.tok}", "Accept": "application/json"})
            except requests.RequestException as e:
                self.err[type(e).__name__] = self.err.get(type(e).__name__, 0) + 1
                time.sleep(2 ** i)
                continue
            self.n += 1
            if r.status_code == 401 and i == 0:            # 토큰 만료 → 새로
                try:
                    os.remove(TP.TOKEN)
                except OSError:
                    pass
                self.tok, _ = TP.get_token(self.env)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                self.err[str(r.status_code)] = self.err.get(str(r.status_code), 0) + 1
                time.sleep(2 ** (i + 1))
                continue
            if r.status_code != 200:
                self.err[str(r.status_code)] = self.err.get(str(r.status_code), 0) + 1
                return None
            try:
                return r.json()
            except ValueError:
                return None
        return None


def find_date(rec):
    for k, v in rec.items():
        if isinstance(v, str) and DATE_KEYS.search(k) and re.match(r"\d{4}-\d{2}-\d{2}", v):
            return v[:10]
    for v in rec.values():
        if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v):
            return v[:10]
    return None


def flat(rec, pre=""):
    out = {}
    for k, v in rec.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def load_raw(path):
    rows = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                rows[r["_date"]] = r
            except (ValueError, KeyError):
                continue
    return rows


def save_raw(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for d in sorted(rows):
            fh.write(json.dumps(rows[d], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def collect_records(api, path, params, file, backfill, key="records", cursor="until", next_key="nextUntil"):
    """최신 → 과거로 넘기며 모은다.
    매일 모드: 이미 가진 날짜를 만나면 멈춘다 (보통 1쪽).
    되채우기: ① 최신부터 이미 가진 날짜까지 ② 가진 것 중 가장 오래된 날부터 2018년까지 (끊겨도 다시 실행하면 이어서)."""
    rows = load_raw(file)
    have = set(rows)
    added = pages = 0
    starts = [None]
    if backfill and have and min(have) > START:
        old = min(have)
        starts.append(old if cursor == "until" else f"{old}T00:00:00+09:00")
    for i, cur in enumerate(starts):
        while pages < 400:
            p = dict(params)
            if cur:
                p[cursor] = cur
            j = api.get(path, p)
            pages += 1
            res = j.get("result") if isinstance(j, dict) else None
            recs = res.get(key) if isinstance(res, dict) else None
            if not recs:
                break
            hit_old = too_old = False
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                d = find_date(rec)
                if not d:
                    continue
                if d < START:
                    too_old = True
                    continue
                if d in have:
                    hit_old = True
                if d not in rows:
                    added += 1
                r = flat(rec)
                r["_date"] = d
                rows[d] = r
            nxt = res.get(next_key)
            if too_old or not nxt or nxt == cur:
                break
            if i == 0 and hit_old:                      # 가진 자료와 이어짐
                break
            cur = nxt
    if added:
        save_raw(file, rows)
    return {"rows": len(rows), "added": added, "pages": pages,
            "first": min(rows) if rows else None, "last": max(rows) if rows else None,
            "fields": sorted({k for r in list(rows.values())[-50:] for k in r if k != "_date"})}


def universe(limit=None):
    """저장소 data/marketcap.csv 의 최신일 시총 상위 100 (+ 지수 ETF 는 따로)."""
    path = os.path.join(REPO, "data", "marketcap.csv")
    best = {}
    last = None
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            d, code, mc = line.strip().split(",")[:3]
            if last is None or d > last:
                last, best = d, {}
            if d == last and mc:
                best[code.zfill(6)] = float(mc)
    codes = [c for c, _ in sorted(best.items(), key=lambda x: -x[1])][:100]
    return codes[:limit] if limit else codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    api = Api()
    codes = universe(a.limit)
    t0 = time.time()
    st = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "backfill" if a.backfill else "daily",
          "symbols": len(codes), "endpoints": {}}
    for ep in ENDPOINTS:
        agg = {"symbols_ok": 0, "rows": 0, "added": 0, "first": None, "last": None, "fields": set()}
        for i, code in enumerate(codes):
            r = collect_records(api, f"/api/v1/stocks/{code}/{ep}", {"count": 100},
                                os.path.join(RAW, ep, f"{code}.jsonl"), a.backfill)
            if r["rows"]:
                agg["symbols_ok"] += 1
                agg["rows"] += r["rows"]
                agg["added"] += r["added"]
                agg["first"] = min(filter(None, [agg["first"], r["first"]]))
                agg["last"] = max(filter(None, [agg["last"], r["last"]]))
                agg["fields"].update(r["fields"])
            if (i + 1) % 10 == 0:
                print(f"  {ep}: {i + 1}/{len(codes)} · 요청 {api.n} · {time.time() - t0:.0f}s", flush=True)
        agg["fields"] = sorted(agg["fields"])
        st["endpoints"][ep] = agg
        print(f"✓ {ep}: {agg['symbols_ok']}종목 · {agg['rows']}행 (+{agg['added']}) · {agg['first']}~{agg['last']}", flush=True)
    # 시장 전체 투자자별 매매대금
    r = collect_records(api, "/api/v1/market-indicators/KOSPI/investor-trading", {"interval": "1d", "count": 100},
                        os.path.join(RAW, "market", "KOSPI_investor.jsonl"), a.backfill)
    st["market_investor"] = r
    print(f"✓ KOSPI 투자자별: {r['rows']}행 · {r['first']}~{r['last']}")
    # 지수 ETF 일봉 (interval 값은 명세에 예시가 없어 차례로 시도)
    st["etf"] = {}
    for code, name in ETFS.items():
        got = None
        for iv in ("1d", "day", "D"):
            r = collect_records(api, "/api/v1/candles", {"symbol": code, "interval": iv, "count": 200, "adjusted": "true"},
                                os.path.join(RAW, "candles", f"{code}.jsonl"), a.backfill,
                                key="candles", cursor="before", next_key="nextBefore")
            if r["rows"]:
                got = dict(r, interval=iv)
                break
        st["etf"][code] = got or {"rows": 0}
        print(f"{'✓' if got else '✗'} {name} 일봉: {(got or {}).get('rows', 0)}행")
    st["requests"] = api.n
    st["errors"] = api.err
    st["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    json.dump(st, open(STATUS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"끝 · 요청 {api.n}건 · {st['seconds']}초 · 오류 {api.err or '없음'} → data/toss/collect_status.json")


if __name__ == "__main__":
    main()
