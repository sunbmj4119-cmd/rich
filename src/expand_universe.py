"""
표본 늘리기 -> data/prices.csv 확장 + data/prices_delisted.csv (신규)

왜 필요한가
  지금 표본은 '2026년 현재까지 살아남은 코스피 100종목'의 2018년 이후 가격이다.
  이 표본에는 두 가지 구멍이 있다.

    1) 좁다. 100종목뿐이라 순위밴드를 나누면 칸이 금방 얇아진다.
    2) 살아남은 것만 있다. 그사이 상장폐지된 종목이 통째로 빠져 있어
       **하락 꼬리가 실제보다 얇다**. "많이 빠진 뒤 반등확률이 높다"는 통계가
       특히 이 편향을 크게 받는다 — 끝내 회복 못 한 종목은 표본에 없기 때문.

  그래서 두 가지를 받는다.
    A. universe.csv에 있는데 prices.csv에 없는 종목 (지금 살아있는 종목 확장)
    B. 2018년 이후 상장폐지된 코스피 주권 (사라진 종목 = 편향의 실체)

받은 뒤에 무엇이 달라지나
  A는 prices.csv에 그대로 합쳐져 score/signal 파이프라인이 바로 쓴다.
  B는 **따로** prices_delisted.csv에 둔다. 오늘 매매할 종목이 아니라
  통계용 표본이기 때문이다. position_lab / refclass가 선택적으로 읽는다.

주의
  · 네트워크가 필요하다. 로컬 샌드박스에서는 막히므로 GitHub Actions로 돌린다.
  · 이어받기를 지원한다. 중간에 끊겨도 다시 돌리면 못 받은 것부터 간다.
  · 한 번 '데이터 없음'으로 확인된 종목은 skip.csv에 적어 매번 재시도하지 않는다.
"""
import os
import sys
import time

import pandas as pd

PRICES = "data/prices.csv"
DPRICES = "data/prices_delisted.csv"
UNIV = "data/universe.csv"
DELIST = "data/delisted.csv"
SKIP = "data/expand_skip.csv"

START = "20180101"
COLS = ["날짜", "종목코드", "종목명", "종가", "거래량", "시가총액", "PER", "PBR"]
FLUSH_EVERY = 15
PAUSE = 0.4

# 상장폐지 사유 중 '기업이 망해서 사라진 것'과 무관한 것들.
# 수익증권 만기·신주인수권 행사만료·스팩 합병은 주가 하락과 관계없이 사라지므로
# 하락 꼬리를 채우는 목적에는 오히려 잡음이다. 다만 지우지 않고 사유를 남겨
# 나중에 필요하면 골라 쓸 수 있게 한다.
NOISE_REASON = ("수익증권", "신주인수권", "스팩", "상장지수", "파생결합")


def log(*a):
    print(*a, flush=True)


def load_done(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, dtype={"종목코드": str}, usecols=["종목코드"])
    return set(d["종목코드"].astype(str).str.zfill(6).unique())


def load_skip():
    if not os.path.exists(SKIP):
        return set()
    return set(pd.read_csv(SKIP, dtype=str)["종목코드"].str.zfill(6))


def add_skip(code, why):
    head = not os.path.exists(SKIP)
    pd.DataFrame([{"종목코드": code, "사유": why}]).to_csv(
        SKIP, mode="a", header=head, index=False)


def targets_alive():
    """universe.csv에 있는데 prices.csv에 없는 종목"""
    if not os.path.exists(UNIV):
        return []
    u = pd.read_csv(UNIV, dtype={"종목코드": str})
    have = load_done(PRICES)
    codes = sorted(set(u["종목코드"].str.zfill(6)) - have)
    return [(c, None, None) for c in codes]


def targets_delisted():
    """2018년 이후 상장폐지된 코스피 주권 — (코드, 상장폐지일, 사유)"""
    if not os.path.exists(DELIST):
        return []
    d = pd.read_csv(DELIST, dtype={"Symbol": str})
    d["DelistingDate"] = pd.to_datetime(d["DelistingDate"], errors="coerce")
    d = d[(d["DelistingDate"] >= "2018-01-01")
          & (d["Market"] == "KOSPI")
          & (d["SecuGroup"] == "주권")]
    d = d.dropna(subset=["Symbol"])
    have = load_done(DPRICES)
    out = []
    for _, r in d.iterrows():
        c = str(r["Symbol"]).zfill(6)
        if c in have:
            continue
        why = str(r.get("Reason") or "")
        if any(k in why for k in NOISE_REASON):
            continue
        out.append((c, r["DelistingDate"].strftime("%Y%m%d"), why))
    return sorted(out)


def fetch(code, end):
    """한 종목의 일별 종가·거래량·시총·PER·PBR. 없으면 None."""
    from pykrx import stock
    ohlcv = stock.get_market_ohlcv(START, end, code)
    if ohlcv is None or ohlcv.empty:
        return None
    df = ohlcv[["종가", "거래량"]].copy()
    # 시총·재무는 실패해도 가격은 살린다 — 가격만 있어도 경로 통계는 만들 수 있다
    for col, fn, src in (("시가총액", stock.get_market_cap, "시가총액"),):
        try:
            df[col] = fn(START, end, code)[src]
        except Exception:
            df[col] = None
    try:
        f = stock.get_market_fundamental(START, end, code)
        df["PER"], df["PBR"] = f["PER"], f["PBR"]
    except Exception:
        df["PER"] = df["PBR"] = None
    try:
        name = stock.get_market_ticker_name(code)
    except Exception:
        name = code
    df["종목코드"], df["종목명"] = code, name
    df = df.reset_index()
    df["날짜"] = pd.to_datetime(df["날짜"]).dt.strftime("%Y-%m-%d")
    return df[COLS]


def run(kind, tgts, out):
    if not tgts:
        log(f"■ {kind}: 받을 종목 없음")
        return 0
    log(f"■ {kind}: {len(tgts)}종목")
    skip = load_skip()
    today = pd.Timestamp.now().strftime("%Y%m%d")
    buf, got, extra = [], 0, {}
    for n, (code, dend, why) in enumerate(tgts, 1):
        if code in skip:
            continue
        try:
            df = fetch(code, dend or today)
        except Exception as e:
            log(f"  [{n}/{len(tgts)}] {code} 실패: {str(e)[:70]}")
            time.sleep(1.0)
            continue
        if df is None or df.empty:
            log(f"  [{n}/{len(tgts)}] {code} 데이터 없음")
            add_skip(code, "데이터 없음")
            continue
        if dend:                       # 상장폐지 종목은 언제·왜 사라졌는지를 같이 남긴다
            df["상장폐지일"] = pd.to_datetime(dend).strftime("%Y-%m-%d")
            df["사유"] = why
        buf.append(df)
        got += 1
        log(f"  [{n}/{len(tgts)}] {df['종목명'].iloc[0]}({code}) {len(df)}줄"
            + (f" · 폐지 {df['상장폐지일'].iloc[0]}" if dend else ""))
        time.sleep(PAUSE)
        if len(buf) >= FLUSH_EVERY:
            flush(buf, out)
            buf = []
    if buf:
        flush(buf, out)
    log(f"  → {kind} {got}종목 추가")
    return got


def flush(buf, out):
    df = pd.concat(buf, ignore_index=True)
    head = not os.path.exists(out)
    df.to_csv(out, mode="a", header=head, index=False)


def summary():
    for p in (PRICES, DPRICES):
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, dtype={"종목코드": str}, usecols=["날짜", "종목코드"])
        log(f"  {p}: {d['종목코드'].nunique()}종목 · {len(d):,}행 · "
            f"{d['날짜'].min()}~{d['날짜'].max()}")


def main():
    os.makedirs("data", exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"표본 확장 시작 (대상: {which})")
    if which in ("all", "alive"):
        run("살아있는 종목 확장", targets_alive(), PRICES)
    if which in ("all", "delisted"):
        run("상장폐지 종목", targets_delisted(), DPRICES)
    log("\n■ 결과")
    summary()


if __name__ == "__main__":
    main()
