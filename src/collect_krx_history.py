"""
KRX 긴 역사 수집기 — Oracle 서버(또는 집 PC)에서 '한 번' 돌린다. GitHub 서버는 KRX 가 IP 를 막아 못 받는다.

  왜: 지금 단기 몫에서 가장 많이 뽑히는 신호가 외국인·기관 수급인데 2018년 전 자료가 없어 긴 역사 검증을 못 했다(가장 큰 빈칸).
      또 긴 역사 검증의 종목이 '지금 살아남은 대형주'뿐이라 생존편향이 있었다.
  받는 것 (pykrx, KRX 로그인 KRX_ID/KRX_PW — ~/.rich/secrets.env)
    krx_universe.csv.gz : 해마다 마지막 거래일의 코스피 시가총액 상위 150 (상장폐지된 회사 포함 = 그때 실제 대형주)
    krx_ohlcv.csv.gz    : 그 종목들의 수정 일봉 2000~2018
    krx_flows.csv.gz    : 같은 종목들의 외국인·기관·개인 순매수 금액 2000~2018
    krx_etf.csv.gz      : 국내 ETF 12종 상장일부터 전 기간
  이어 받기: 종목·연도마다 data/history/krx_cache/ 에 저장 → 끊겨도 다시 돌리면 이어서 (KRX 차단을 피하려 호출마다 쉼)
  시간: 대략 2~3시간 (종목마다 그 종목이 상위 150 이던 해 ± 1년만 받는다)
  일봉은 KRX 에서 직접 — pykrx 기본(수정주가=네이버)은 상장폐지 종목이 비어 생존편향이 그대로 남는다.
  KRX 의 '수정주가' 요청은 실제로는 원본가로 온다 → 등락률(기준가 대비)도 남겨 history_lab.krx_adjust 가 분할·증자를 보정.
  진행 보기: tail -n 5 ~/.rich/logs/krx_history.log  (경고는 숨기고 진행 줄만 남김)

  서버: bash ~/rich/deploy/oracle/collect_history.sh   (받고 → GitHub 에 올림)
  직접: python src/collect_krx_history.py [--years 2000 2018] [--top 150]
"""
import argparse
import glob
import os
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")                     # pykrx 안의 pandas 경고가 로그를 덮어 진행이 안 보였다
try:
    sys.stdout.reconfigure(line_buffering=True)       # 파이프(tee)로 나가도 진행 줄이 바로 보이게
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DIR = "data/history"
CACHE = os.path.join(DIR, "krx_cache")
ETF_CODES = ["069500", "114800", "122630", "252670", "229200", "153130", "114260", "148070", "261240", "132030", "143850", "133690"]


def stock_api():
    try:
        from pykrx import stock          # KRX_ID 가 있으면 가져오는 순간 로그인
        return stock
    except Exception as ex:
        sys.exit(f"pykrx 를 쓸 수 없습니다 ({type(ex).__name__}: {ex}) — pip install pykrx, KRX_ID/KRX_PW 확인")


def cached(name, fn, keep_empty=True):
    """한 번 받은 것은 저장해 두고 다시 쓴다. 빈 결과는 (차단·일시 오류일 수 있어) keep_empty=False 면 저장하지 않는다."""
    p = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(p):
        try:
            return pd.read_pickle(p)
        except Exception:                             # 받는 중에 끊겨 반쯤 쓴 파일 → 다시 받는다
            os.remove(p)
    df = fn()
    if df is not None and (keep_empty or not df.empty):
        os.makedirs(CACHE, exist_ok=True)
        df.to_pickle(p + ".tmp")
        os.replace(p + ".tmp", p)
    return df


def year_end(stock, y):
    """그해 마지막 거래일 (YYYYMMDD)."""
    for d in pd.date_range(f"{y}-12-31", periods=15, freq="-1D"):
        s = d.strftime("%Y%m%d")
        try:
            df = stock.get_market_cap(s, market="KOSPI")
            if df is not None and len(df) > 100 and df["시가총액"].sum() > 0:
                return s, df
        except Exception:
            pass
        time.sleep(0.3)
    return None, None


def universe(stock, years, top):
    rows = []
    for y in years:
        def get():
            s, df = year_end(stock, y)
            if df is None:
                return None
            df = df.sort_values("시가총액", ascending=False).head(top).reset_index()
            df = df.rename(columns={df.columns[0]: "종목코드"})
            df["기준일"] = pd.Timestamp(s).strftime("%Y-%m-%d")
            df["순위"] = range(1, len(df) + 1)
            return df[["기준일", "종목코드", "시가총액", "순위"]]
        d = cached(f"univ_{y}_{top}", get)
        if d is not None:
            rows.append(d)
            print(f"  {y} 기준일 {d['기준일'].iloc[0]} 상위 {len(d)}")
    u = pd.concat(rows, ignore_index=True)
    try:
        names = {}
        for c in u["종목코드"].unique():
            try:
                names[c] = stock.get_market_ticker_name(c)
            except Exception:
                names[c] = ""
        u["종목명"] = u["종목코드"].map(names)
    except Exception:
        u["종목명"] = ""
    return u


class Blocked(Exception):
    pass


def per_code(stock, need, kind, t0=None):
    """need: {종목코드: [연도…]}. 연도마다 받아 저장(이어 받기). 빈 응답이 계속되면 차단·로그인 문제로 보고 멈춘다."""
    from pykrx.website import krx as KW                # KRX 직접 (수정주가, 상장폐지 종목 포함)
    out, empty_run, calls, t0 = [], 0, 0, t0 or time.time()
    codes = sorted(need)
    total = sum(len(v) for v in need.values())
    done = 0
    for i, c in enumerate(codes):
        for y in need[c]:
            s, e = f"{y}0101", f"{y}1231"
            hit = os.path.exists(os.path.join(CACHE, f"{kind}k_{c}_{y}.pkl"))

            def get():
                try:
                    if kind == "ohlcv":
                        df = KW.get_market_ohlcv_by_date(s, e, c, True)
                    elif kind == "etf":
                        df = stock.get_etf_ohlcv_by_date(s, e, c)
                    else:
                        df = stock.get_market_trading_value_by_date(s, e, c, detail=False)
                except Exception as ex:
                    print(f"    ! {kind} {c} {y}: {type(ex).__name__} {str(ex)[:80]}")
                    return pd.DataFrame()
                finally:
                    time.sleep(0.35)
                return df if df is not None else pd.DataFrame()
            df = cached(f"{kind}k_{c}_{y}", get, keep_empty=False)
            done += 1
            calls += 0 if hit else 1
            if df is not None and not df.empty:
                empty_run = 0                              # 받아 둔 것이든 새로 받은 것이든 자료가 있으면 정상
            elif not hit:
                empty_run += 1
                if empty_run >= 60:
                    raise Blocked(f"{kind}: 빈 응답이 {empty_run}번 이어짐 (마지막 {c} {y}) — KRX 차단이나 로그인 만료로 보입니다. "
                                  "잠시 뒤 다시 실행하면 받은 데까지 이어 받습니다")
            if df is None or df.empty:
                continue
            df = df.reset_index().rename(columns={df.reset_index().columns[0]: "날짜"})
            df["종목코드"] = c
            out.append(df)
        if (i + 1) % 10 == 0 or i + 1 == len(codes):
            el = time.time() - t0
            left = (total - done) * (el / max(calls, 1)) if calls else 0
            print(f"    {kind} 종목 {i + 1}/{len(codes)} · 연도 {done}/{total} · 새로 받음 {calls} · "
                  f"{el / 60:.0f}분 지남" + (f" · 남은 시간 약 {left / 60:.0f}분" if calls >= 20 else ""))
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    d["날짜"] = pd.to_datetime(d["날짜"]).dt.strftime("%Y-%m-%d")
    return d


def needed_years(u, y0, y1):
    """종목마다 받을 해: 상위 150 이던 해(연말 기준) 의 다음 해가 실제로 담는 해, 그 앞 1년은 신호 계산용."""
    yrs = u.assign(y=pd.to_datetime(u["기준일"]).dt.year).groupby("종목코드")["y"].agg(["min", "max"])
    return {c: list(range(max(y0, int(r["min"])), min(y1, int(r["max"]) + 1) + 1)) for c, r in yrs.iterrows()}


def tidy_flows(f):
    if f.empty:
        return f
    pick = {"외국인순매수": ["외국인", "외국인합계"], "기관순매수": ["기관합계", "기관"], "개인순매수": ["개인"]}
    out = f[["날짜", "종목코드"]].copy()
    for k, cands in pick.items():
        col = next((c for c in cands if c in f.columns), None)
        out[k] = f[col] if col else float("nan")
    return out


def check_login(stock):
    """시작하자마자 한 번 받아 본다 — 안 되면 몇 시간 돌다 빈 파일을 올리는 대신 바로 이유를 알린다."""
    for d in ("20181228", "20181227"):
        try:
            df = stock.get_market_cap(d, market="KOSPI")
            if df is not None and len(df) > 100:
                print(f"0) KRX 접속 확인 ✓ ({d} 코스피 {len(df)}종목)")
                return True
        except Exception as ex:
            print(f"   KRX 접속 오류: {type(ex).__name__} {str(ex)[:120]}")
    print("0) KRX 접속 확인 ✗ — 빈 응답. KRX_ID/KRX_PW(setup.sh --keys) 와 KRX 정보데이터시스템 로그인이 되는지 확인하세요")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2000, 2018])
    ap.add_argument("--top", type=int, default=150)
    a = ap.parse_args()
    stock = stock_api()
    if not check_login(stock):
        sys.exit(2)
    for f in glob.glob(os.path.join(CACHE, "ohlcv_*.pkl")) + glob.glob(os.path.join(CACHE, "flows_*.pkl")) + \
            glob.glob(os.path.join(CACHE, "etf_*.pkl")):
        os.remove(f)                                    # 예전 판(네이버 일봉, 빈 응답까지 저장) 캐시는 버린다
    years = list(range(a.years[0], a.years[1] + 1))
    os.makedirs(DIR, exist_ok=True)
    t0 = time.time()
    print("1) 해마다 시가총액 상위 (상장폐지 포함)")
    u = universe(stock, years, a.top)
    u.to_csv(os.path.join(DIR, "krx_universe.csv.gz"), index=False)
    need = needed_years(u, a.years[0], a.years[1])
    print(f"   종목 {len(need)}개 (해마다 상위 {a.top}의 합집합) · 받을 종목·연도 {sum(map(len, need.values()))}개")
    stopped = []

    def step(kind, need_k, out, keep=None, tidy=None):
        try:
            d = per_code(stock, need_k, kind)
        except Blocked as ex:
            print(f"   멈춤: {ex}")
            stopped.append(kind)
            return
        d = tidy(d) if tidy else d
        if not d.empty:
            cols = [c for c in keep if c in d.columns] if keep else list(d.columns)
            d[cols].to_csv(os.path.join(DIR, out), index=False)
        print(f"   {len(d):,}행 · 종목 {d['종목코드'].nunique() if not d.empty else 0}개 → {out}")

    print("2) 수정 일봉 (KRX 직접 — 상장폐지 종목 포함)")
    step("ohlcv", need, "krx_ohlcv.csv.gz", keep=("날짜", "종목코드", "시가", "고가", "저가", "종가", "거래량", "거래대금", "등락률"))
    print("3) 투자자별 순매수 (외국인·기관·개인)")
    step("flows", need, "krx_flows.csv.gz", tidy=tidy_flows)
    print("4) 국내 ETF 12종 전 기간")
    step("etf", {c: list(range(2002, pd.Timestamp.now().year + 1)) for c in ETF_CODES}, "krx_etf.csv.gz",
         keep=("날짜", "종목코드", "시가", "고가", "저가", "종가", "거래량"))
    if stopped:
        print(f"일부 멈춤 ({', '.join(stopped)}) — 다시 실행하면 받은 데까지 이어 받습니다")
    print(f"끝 ({(time.time() - t0) / 60:.0f}분) → {DIR}/krx_*.csv.gz")


if __name__ == "__main__":
    main()
