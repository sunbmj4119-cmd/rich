"""
토스증권 캔들(분봉) 수집 — 서버에서만 (토스는 등록 IP 에서만 받는다). 조회(GET)만.

  종목: 국내 = 최근 시총 상위 100 (data/universe.csv) · 미국 = data/us_ohlcv.csv 의 종목 (레버리지·인버스 제외)
  봉 단위: data/toss/probe.json 의 캔들 점검(봉 단위 목록)에서 5분봉을 고른다 (--interval 로 바꿈)
  저장: 원본은 서버 ~/.rich/candles/<봉>/<종목>.csv.gz 에만 (공개 저장소에 시세 원본을 올리지 않음, 크기도 큼)
        GitHub 에는 data/toss/candles_status.json (종목 수·기간·봉 수만)
  이어 받기: 이미 받은 가장 최근 봉까지만 새로 받는다. --backfill 이면 받을 수 있는 가장 옛날까지 (끊겨도 다시 실행하면 이어서).

  python toss_candles.py --backfill          # 처음: 옛날까지 (2~3시간, 뒤에서)
  python toss_candles.py                     # 매일: 새 봉만
  python toss_candles.py --market kr --limit 3
"""
import argparse
import csv
import gzip
import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toss_probe as TP  # noqa: E402

BASE = TP.BASE
RAW = os.path.join(TP.STATE, "candles")
REPO = TP.REPO if os.path.isdir(os.path.join(TP.REPO, "data")) else os.path.abspath(os.path.join(HERE, "..", ".."))
STATUS = os.path.join(REPO, "data", "toss", "candles_status.json")
PAUSE = 0.25
GEARED_US = {"SOXL", "SOXS", "TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA", "LABU", "LABD", "FAS", "FAZ", "KORU", "TECL", "TECS"}
KEYS = {"ts": re.compile(r"(timestamp|time|datetime|date|dt)$", re.I), "open": re.compile(r"open", re.I), "high": re.compile(r"high", re.I),
        "low": re.compile(r"low", re.I), "close": re.compile(r"(close|last|trade)price$|^close$", re.I), "volume": re.compile(r"(volume|vol|qty|quantity)$", re.I)}


def pick_interval(want="5"):
    """명세의 봉 단위 목록에서 want 분 봉을 고른다 (예: '5m', 'MINUTE_5', '5')."""
    try:
        cp = json.load(open(os.path.join(REPO, "data", "toss", "probe.json"))).get("candles") or {}
    except (OSError, ValueError):
        cp = {}
    ivs = list(cp.get("intervals") or [])
    ivs += [k.split(":", 1)[1] for k, v in (cp.get("tests") or {}).items() if (v or {}).get("status") == 200 and k.split(":", 1)[1] not in ivs]
    for iv in ivs:
        s = str(iv)
        if re.fullmatch(rf"(?i)(m(in(ute)?)?_?{want}|{want}_?m(in(ute)?)?|{want})", s):
            return s
    return None


def symbols(market, limit=None):
    out = []
    if market == "kr":
        rows = list(csv.DictReader(open(os.path.join(REPO, "data", "universe.csv"), encoding="utf-8")))
        last = max(r["날짜"] for r in rows)
        out = ["069500"] + [r["종목코드"] for r in rows if r["날짜"] == last][:100]      # 069500 = 기준(KODEX 200)
    else:
        seen = []
        with open(os.path.join(REPO, "data", "us_ohlcv.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                c = r["종목코드"]
                if c not in seen and c not in GEARED_US:
                    seen.append(c)
        out = ["SPY", "QQQ"] + [c for c in seen if c not in ("SPY", "QQQ")]          # 기준(지수 ETF) 먼저
    return out[:limit] if limit else out


def norm(c):
    """캔들 하나 → (ts, o, h, l, c, v). 항목 이름을 모르니 이름으로 찾는다."""
    if not isinstance(c, dict):
        return None
    got = {}
    for k, v in c.items():
        for name, rx in KEYS.items():
            if name not in got and rx.search(k):
                got[name] = v
    if "ts" not in got or "close" not in got:
        return None

    def num(x):
        if isinstance(x, dict):
            x = x.get("amount", x.get("value"))
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    return (str(got["ts"]), num(got.get("open")), num(got.get("high")), num(got.get("low")), num(got.get("close")), num(got.get("volume")))


def read_have(path):
    if not os.path.exists(path):
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r[0]: r for r in csv.reader(f) if r and r[0] != "ts"}


def write_all(path, rows):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for k in sorted(rows):
            w.writerow(rows[k])
    os.replace(tmp, path)


def fetch(api, sym, iv, have, backfill, max_pages=2000):
    """① 가장 최근 → 이미 받은 가장 최근 봉까지 ② backfill 이면 이미 받은 가장 옛날 봉부터 더 옛날로 (끝날 때까지)."""
    new = pages = 0
    newest_have = max(have) if have else None

    def page(before):
        nonlocal new, pages
        j = api.get("/api/v1/candles", {"symbol": sym, "interval": iv, "count": 200, **({"before": before} if before else {})})
        pages += 1
        if j is None:
            return [], None
        res = j.get("result") or {}
        cs = [x for x in (norm(c) for c in res.get("candles") or []) if x]
        for row in cs:
            if row[0] not in have:
                have[row[0]] = list(row)
                new += 1
        return cs, res.get("nextBefore")
    before = None
    while pages < max_pages:
        cs, nxt = page(before)
        if not cs or not nxt:
            return new, pages                              # 끝까지 받음
        if newest_have and min(r[0] for r in cs) <= newest_have:
            break
        before = nxt
    if backfill and have:
        before = min(have)
        while pages < max_pages:
            cs, nxt = page(before)
            if not cs or not nxt:
                break
            before = nxt
    return new, pages


class Api:
    def __init__(self):
        self.env = TP.load_env()
        self.tok, info = TP.get_token(self.env)
        if not self.tok:
            raise SystemExit(f"토큰 발급 실패: {info}")
        seq = self.env.get("TOSS_ACCOUNT_SEQ")
        self.hdr = {"Authorization": f"Bearer {self.tok}", "Accept": "application/json",
                    **({"X-Tossinvest-Account": str(seq)} if seq not in (None, "") else {})}
        self.n, self.err = 0, {}

    def get(self, path, params):
        for i in range(5):
            time.sleep(PAUSE)
            try:
                r = requests.get(BASE + path, params=params, headers=self.hdr, timeout=20)
            except requests.RequestException as e:
                self.err[type(e).__name__] = self.err.get(type(e).__name__, 0) + 1
                time.sleep(2 ** i)
                continue
            self.n += 1
            if r.status_code == 200:
                return r.json()
            self.err[str(r.status_code)] = self.err.get(str(r.status_code), 0) + 1
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** i)
                continue
            return None
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--market", choices=["kr", "us", "all"], default="all")
    ap.add_argument("--interval", default=None, help="봉 단위 (기본: 명세에서 5분봉)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    iv = a.interval or pick_interval("5")
    if not iv:
        raise SystemExit("봉 단위를 모릅니다 — 먼저 setup.sh --probe (캔들 점검) 를 하세요")
    api = Api()
    status = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    status.setdefault("interval", iv)
    t0 = time.time()
    for market in (["kr", "us"] if a.market == "all" else [a.market]):
        d = os.path.join(RAW, iv, market)
        os.makedirs(d, exist_ok=True)
        syms = symbols(market, a.limit)
        tot_new, info = 0, {}
        for i, sym in enumerate(syms):
            path = os.path.join(d, f"{sym}.csv.gz")
            have = read_have(path)
            new, pages = fetch(api, sym, iv, have, a.backfill)
            if new:
                write_all(path, have)
            tot_new += new
            if have:
                info[sym] = {"rows": len(have), "first": min(have), "last": max(have)}
            if (i + 1) % 10 == 0:
                print(f"  {market} {i + 1}/{len(syms)} · 새 봉 {tot_new:,} · 요청 {api.n:,} · {(time.time() - t0) / 60:.0f}분", flush=True)
        rows = [v["rows"] for v in info.values()]
        status[market] = {"symbols": len(info), "rows": int(sum(rows)), "first": min((v["first"] for v in info.values()), default=None),
                          "last": max((v["last"] for v in info.values()), default=None),
                          "median_rows": int(sorted(rows)[len(rows) // 2]) if rows else 0}
        print(f"■ {market}: {len(info)}종목 · {sum(rows):,}봉 · {status[market]['first']} ~ {status[market]['last']}")
    status.update({"updated": time.strftime("%Y-%m-%d %H:%M"), "requests": api.n, "errors": api.err})
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    json.dump(status, open(STATUS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"끝 ({(time.time() - t0) / 60:.0f}분) → 원본 {RAW}/{iv}/ · 현황 data/toss/candles_status.json")


if __name__ == "__main__":
    main()
