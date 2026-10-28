"""
수급·시세·공매도 보강 수집기 (pykrx, KRX 로그인 필요 — KRX_ID/KRX_PW 이미 연동됨)

== 수집 항목 ==
  1) 투자자별 순매수 (외국인/기관/개인)        - 종목별 일별  → data/flows.csv
  2) 공매도 (거래량비중 + 잔고비중)             - 종목별 일별  → data/shorts.csv
  3) OHLCV (시/고/저/종/거래량/거래대금)        - 날짜별 전종목 → data/ohlcv.csv
  4) 시가총액 / 상장주식수                      - 날짜별 전종목 → data/marketcap.csv

[설계 의도]
  - 백필을 단 1회만 돌리도록, 종목 단위로 도는 수급+공매도를 같은 루프에 묶음
    (KRX 호출을 한 번에 처리 → 차단 위험·소요시간 최소화)
  - 공매도는 일단 '수집·검증'까지만. score.py 반영은 verify_factor.py 검증 후 결정.

== 사용 ==
  python src/collect_flows.py --backfill 2018-01-01   # 과거 전체 1회 수집
  python src/collect_flows.py                          # 최근일만 추가(매일)
  python src/collect_flows.py --check                  # 수집 데이터 검증만

[검증 철학]
  - 빈 DataFrame, 전부 0, 결측 과다, 종목수 부족을 모두 체크하고 로그 출력
  - 실패 종목은 건너뛰되 기록 (조용한 실패 방지)
"""
import argparse
import os
import time
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PRICES = "data/prices.csv"
FLOWS = "data/flows.csv"
SHORTS = "data/shorts.csv"
OHLCV = "data/ohlcv.csv"
MCAP = "data/marketcap.csv"
STATUS = "data/collect_status.json"     # 마지막 수집 결과 — 자동매매·홈·해설이 '왜 자료가 낡았나'를 보여 줄 때 읽는다
GAP_MAX_DAYS = 120                      # 매일 모드: 빠진 기간이 있으면 이만큼까지 거슬러 메운다

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KRX_ERR = None
try:
    from pykrx import stock      # KRX_ID 가 있으면 가져오는 순간 로그인 — KRX 가 IP 를 막으면 여기서 예외가 난다
except ImportError:
    print("pykrx 필요: pip install pykrx")
    sys.exit(1)
except Exception as _e:          # 로그인 응답이 JSON 이 아님(차단 페이지) 등 → 네이버 대체 경로만으로 진행
    stock = None
    KRX_ERR = f"{type(_e).__name__}: {str(_e)[:120]}"
    print(f"⚠ KRX 접속 실패 ({KRX_ERR}) — 네이버 대체 경로로 받습니다")


def get_universe():
    """우리 100종목 코드 목록 (prices.csv 기준)"""
    px = pd.read_csv(PRICES, dtype={"종목코드": str})
    px["종목코드"] = px["종목코드"].str.zfill(6)
    return sorted(px["종목코드"].unique())


def trading_dates(start, end):
    """영업일 목록 — KRX 가 안 되면 prices.csv 에 있는 날짜(네이버로 갱신됨), 그것도 없으면 평일."""
    out = []
    if stock is not None:
        try:
            out = list(stock.get_previous_business_days(fromdate=start.replace("-", ""), todate=end.replace("-", "")))
        except Exception as ex:
            print(f"    ⚠ KRX 영업일 조회 실패: {type(ex).__name__} {str(ex)[:80]}")
    if not out:
        d = pd.to_datetime(pd.read_csv(PRICES, usecols=["날짜"])["날짜"].unique())
        out = sorted(x for x in d if pd.Timestamp(start) <= x <= pd.Timestamp(end))
        if not out:
            out = list(pd.bdate_range(start, end))
        print(f"    영업일은 prices.csv/평일 기준으로 셈 ({len(out)}일)")
    return out


# ─────────────────────────────────────────────────────────
# 네이버 대체 경로 — KRX 가 막혀 빠진 (날짜, 종목)만 채운다
# ─────────────────────────────────────────────────────────
def naver_fill_ohlcv(codes, dates, have):
    """OHLCV·시총 빈칸 채우기. 거래대금 = 종가×거래량(근사), 시총 = 종가 × 마지막으로 알려진 상장주식수."""
    import naver
    want = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates}
    shares = {}
    if os.path.exists(MCAP):
        m = pd.read_csv(MCAP, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "상장주식수"])
        m["종목코드"] = m["종목코드"].str.zfill(6)
        shares = m.sort_values("날짜").groupby("종목코드")["상장주식수"].last().to_dict()
    n = max(10, len(want) + 10)
    o_rows, c_rows, fails = [], [], []
    for code in codes:
        miss = {d for d in want if (d, code) not in have}
        if not miss:
            continue
        try:
            df = naver.daily(code, n)
        except Exception:
            fails.append(code)
            continue
        for _, r in df[df["날짜"].isin(miss)].iterrows():
            if not r["종가"] or r["종가"] <= 0:
                continue
            o_rows.append({"날짜": r["날짜"], "종목코드": code, "시가": r["시가"], "고가": r["고가"], "저가": r["저가"],
                           "종가": r["종가"], "거래량": r["거래량"], "거래대금": round(r["종가"] * r["거래량"])})
            sh = shares.get(code)
            if sh and sh > 0:
                c_rows.append({"날짜": r["날짜"], "종목코드": code, "시가총액": round(r["종가"] * sh), "상장주식수": sh})
        time.sleep(0.05)
    return pd.DataFrame(o_rows), pd.DataFrame(c_rows), fails


def naver_fill_flows(codes, start, end):
    """외국인·기관 순매수 금액 근사 = 순매매 수량 × 종가. 개인 = -(외국인+기관) (기타법인 무시 — 근사)."""
    import naver
    pages = max(1, int(np.ceil(len(pd.bdate_range(start, end)) / 20)))
    rows, fails = [], []
    for code in codes:
        try:
            df = naver.investor(code, pages)
        except Exception:
            fails.append(code)
            continue
        df = df[(df["날짜"] >= start) & (df["날짜"] <= end)]
        if df.empty:
            fails.append(code)
            continue
        for _, r in df.iterrows():
            if not np.isfinite(r["외국인"]) or not np.isfinite(r["종가"]):
                continue
            f, i = r["외국인"] * r["종가"], np.nan_to_num(r["기관"]) * r["종가"]
            p_ = r.get("개인", np.nan)
            p_ = p_ * r["종가"] if np.isfinite(p_) else -(f + i)
            rows.append({"날짜": r["날짜"], "종목코드": code, "외국인순매수": round(f), "기관순매수": round(i),
                         "개인순매수": round(p_)})
    return pd.DataFrame(rows), fails


def _pick(df, cands):
    """버전마다 컬럼명이 달라 후보 중 존재하는 첫 컬럼 반환"""
    for c in cands:
        if c in df.columns:
            return c
    return None


def _fetch_short(s, e, code):
    """공매도 잔고(핵심) + 거래량(보조)을 각각 독립 try로 수집.
    반환: (sv dict {날짜: {컬럼:값}}, 첫 에러문자열 or None)
    잔고 하나만 성공해도 sv는 채워진다 → 한쪽 버그에 강건."""
    sv = {}
    err = None
    # 잔고 (우선 — 약세 신호의 핵심)
    try:
        bal = stock.get_shorting_balance_by_date(s, e, code)
        if bal is not None and not bal.empty:
            bal = bal.reset_index().rename(columns={bal.reset_index().columns[0]: "날짜"})
            bw_col = _pick(bal, ["비중", "잔고비중"])
            bq_col = _pick(bal, ["잔고수량", "잔고"])
            for _, r in bal.iterrows():
                d = pd.to_datetime(r["날짜"]).strftime("%Y-%m-%d")
                sv.setdefault(d, {})["공매도잔고비중"] = r[bw_col] if bw_col else np.nan
                sv[d]["공매도잔고수량"] = r[bq_col] if bq_col else np.nan
    except Exception as ex:
        err = f"잔고 {type(ex).__name__}: {str(ex)[:80]}"
    # 거래량 (보조 — pykrx 버전 따라 KeyError 날 수 있음, 실패해도 잔고는 유지)
    try:
        vol = stock.get_shorting_volume_by_date(s, e, code)
        if vol is not None and not vol.empty:
            vol = vol.reset_index().rename(columns={vol.reset_index().columns[0]: "날짜"})
            vw_col = _pick(vol, ["비중", "공매도비중"])
            for _, r in vol.iterrows():
                d = pd.to_datetime(r["날짜"]).strftime("%Y-%m-%d")
                sv.setdefault(d, {})["공매도거래비중"] = r[vw_col] if vw_col else np.nan
    except Exception as ex:
        if err is None:
            err = f"거래량 {type(ex).__name__}: {str(ex)[:80]}"
    return sv, err


# ─────────────────────────────────────────────────────────
# 1+2) 종목별 시계열: 투자자 순매수 + 공매도 (같은 루프)
# ─────────────────────────────────────────────────────────
def collect_per_ticker(codes, start, end):
    """각 종목의 외국인/기관/개인 순매수 + 공매도(거래량·잔고)를 한 번에 수집.

    수급과 공매도 모두 (fromdate, todate, ticker) 시그니처라 종목당 호출을 묶어
    백필 시 KRX 접속 횟수를 절반으로 줄인다.
    """
    flow_rows, short_rows = [], []
    flow_fails, short_fails = [], []
    s = start.replace("-", "")
    e = end.replace("-", "")
    if stock is None:
        return pd.DataFrame(), pd.DataFrame(), list(codes), list(codes)
    blocked = 0

    for i, code in enumerate(codes):
        if blocked >= 15:                     # 연속 15종목 실패 = KRX 차단 → 나머지는 네이버로
            flow_fails.append(code)
            short_fails.append(code)
            continue
        # --- (1) 투자자별 순매수 ---
        try:
            df = stock.get_market_trading_value_by_date(s, e, code, detail=False)
            if df is not None and not df.empty:
                df = df.reset_index().rename(columns={df.reset_index().columns[0]: "날짜"})
                f_col = _pick(df, ["외국인", "외국인합계"])
                i_col = _pick(df, ["기관합계", "기관"])
                p_col = _pick(df, ["개인"])
                for _, r in df.iterrows():
                    flow_rows.append({
                        "날짜": pd.to_datetime(r["날짜"]).strftime("%Y-%m-%d"),
                        "종목코드": code,
                        "외국인순매수": r[f_col] if f_col else np.nan,
                        "기관순매수": r[i_col] if i_col else np.nan,
                        "개인순매수": r[p_col] if p_col else np.nan,
                    })
                blocked = 0
            else:
                flow_fails.append(code)
                blocked += 1
        except Exception:
            flow_fails.append(code)
            blocked += 1

        time.sleep(0.3)  # KRX 차단 방지

        # --- (2) 공매도: 잔고(핵심) + 거래량(보조) 별도 수집 ---
        # vol/bal을 분리: 한쪽이 pykrx 버전 버그(KeyError '거래량' 등)로 터져도
        # 다른 쪽(특히 잔고비중)은 살린다. 잔고비중이 진짜 약세 신호.
        sv, sv_err = _fetch_short(s, e, code)
        if sv:
            for d, vals in sv.items():
                short_rows.append({
                    "날짜": d, "종목코드": code,
                    "공매도거래비중": vals.get("공매도거래비중", np.nan),
                    "공매도잔고비중": vals.get("공매도잔고비중", np.nan),
                    "공매도잔고수량": vals.get("공매도잔고수량", np.nan),
                })
        else:
            short_fails.append(code)
            if sv_err and len(short_fails) <= 3:  # 첫 몇 개만 원인 출력
                print(f"    [공매도 실패 예시] {code}: {sv_err}")

        time.sleep(0.3)

        if (i + 1) % 20 == 0:
            print(f"    수급+공매도 {i+1}/{len(codes)} 종목...")

    return (pd.DataFrame(flow_rows), pd.DataFrame(short_rows),
            flow_fails, short_fails)


# ─────────────────────────────────────────────────────────
# 3+4) OHLCV + 시가총액 (날짜별 전종목 = 호출수 적음)
# ─────────────────────────────────────────────────────────
def collect_ohlcv_cap(codes, dates):
    code_set = set(codes)
    ohlcv_rows, cap_rows, fails = [], [], []
    if stock is None:
        return pd.DataFrame(), pd.DataFrame(), [str(d) for d in dates]
    for i, d in enumerate(dates):
        ds = d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d).replace("-", "")
        dstr = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        # OHLCV와 시가총액을 따로 받는다.
        #  전에는 한 try 안에 묶여 있어서 시가총액 한 번 실패하면 그날 OHLCV까지
        #  통째로 버려졌다. 게다가 except가 원인을 삼켜 왜 멈췄는지 알 수 없었다
        #  (실제로 ohlcv.csv가 8/07에 멈춘 걸 며칠 뒤에야 발견).
        o = c = None
        try:
            o = stock.get_market_ohlcv_by_ticker(ds, market="ALL")
        except Exception as ex:
            print(f"    ⚠ OHLCV {ds} 실패: {type(ex).__name__} {str(ex)[:120]}")
        try:
            c = stock.get_market_cap_by_ticker(ds, market="ALL")
        except Exception as ex:
            print(f"    ⚠ 시가총액 {ds} 실패: {type(ex).__name__} {str(ex)[:120]}")
        try:
            if o is None or o.empty:
                fails.append(ds)
                continue
            o = o.reset_index()
            tk = o.columns[0]
            for _, r in o.iterrows():
                code = str(r[tk]).zfill(6)
                if code not in code_set:
                    continue
                ohlcv_rows.append({
                    "날짜": dstr, "종목코드": code,
                    "시가": r.get("시가"), "고가": r.get("고가"),
                    "저가": r.get("저가"), "종가": r.get("종가"),
                    "거래량": r.get("거래량"), "거래대금": r.get("거래대금"),
                })
            if c is not None and not c.empty:
                c = c.reset_index()
                tkc = c.columns[0]
                for _, r in c.iterrows():
                    code = str(r[tkc]).zfill(6)
                    if code not in code_set:
                        continue
                    cap_rows.append({
                        "날짜": dstr, "종목코드": code,
                        "시가총액": r.get("시가총액"),
                        "상장주식수": r.get("상장주식수"),
                    })
        except Exception as ex:
            print(f"    ⚠ OHLCV {ds} 파싱 실패: {type(ex).__name__} {str(ex)[:120]}")
            fails.append(ds)
        if (i + 1) % 20 == 0:
            print(f"    OHLCV {i+1}/{len(dates)} 일...")
        time.sleep(0.3)
    return pd.DataFrame(ohlcv_rows), pd.DataFrame(cap_rows), fails


# ─────────────────────────────────────────────────────────
# 저장 (중복 제거 append)
# ─────────────────────────────────────────────────────────
def save_merge(df, path, keys=("날짜", "종목코드")):
    if df is None or df.empty:
        print(f"    ⚠️ {path}: 수집된 행 없음 — 저장 건너뜀")
        return
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"종목코드": str})
        old["종목코드"] = old["종목코드"].str.zfill(6)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=list(keys), keep="last")
    df = df.sort_values(list(keys))
    df.to_csv(path, index=False)
    print(f"    ✅ {path}: 총 {len(df):,}행 저장")


# ─────────────────────────────────────────────────────────
# 검증
# ─────────────────────────────────────────────────────────
def check():
    print("\n=== 수집 데이터 검증 ===")
    specs = [
        (FLOWS, ["외국인순매수", "기관순매수", "개인순매수"]),
        (SHORTS, ["공매도거래비중", "공매도잔고비중", "공매도잔고수량"]),
        (OHLCV, ["시가", "고가", "저가", "종가", "거래대금"]),
        (MCAP, ["시가총액", "상장주식수"]),
    ]
    for path, cols in specs:
        if not os.path.exists(path):
            print(f"\n❌ {path}: 파일 없음 (아직 수집 안 됨)")
            continue
        df = pd.read_csv(path, dtype={"종목코드": str})
        print(f"\n■ {path}")
        print(f"   행수: {len(df):,} | 종목수: {df['종목코드'].nunique()} | "
              f"기간: {df['날짜'].min()} ~ {df['날짜'].max()}")
        for c in cols:
            if c not in df.columns:
                print(f"   ⚠️ '{c}' 컬럼 없음")
                continue
            nn = df[c].notna().sum()
            zero = (df[c] == 0).sum()
            ex = df[c].dropna().iloc[0] if nn else "N/A"
            print(f"   {c}: 값있음 {nn/len(df)*100:.0f}% | "
                  f"0값 {zero/len(df)*100:.0f}% | 예시 {ex}")


# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# 공매도만 수집 (수급/OHLCV는 이미 받았을 때 — 시간 절약)
# ─────────────────────────────────────────────────────────
def collect_shorts_only(codes, start, end):
    """공매도(잔고비중+거래량비중)만 종목별로 수집. flows/ohlcv는 건드리지 않음."""
    short_rows, short_fails = [], []
    s = start.replace("-", "")
    e = end.replace("-", "")
    for i, code in enumerate(codes):
        sv, err = _fetch_short(s, e, code)
        if sv:
            for d, vals in sv.items():
                short_rows.append({
                    "날짜": d, "종목코드": code,
                    "공매도거래비중": vals.get("공매도거래비중", np.nan),
                    "공매도잔고비중": vals.get("공매도잔고비중", np.nan),
                    "공매도잔고수량": vals.get("공매도잔고수량", np.nan),
                })
        else:
            short_fails.append(code)
            if err and len(short_fails) <= 3:
                print(f"    [공매도 실패 예시] {code}: {err}")
        time.sleep(0.4)
        if (i + 1) % 20 == 0:
            print(f"    공매도 {i+1}/{len(codes)} 종목...")
    return pd.DataFrame(short_rows), short_fails


class _Tee:
    """화면 출력은 그대로 두고 내용을 모아 둔다 (pykrx 의 로그인 실패 문구를 잡기 위해)."""
    def __init__(self, out):
        self.out, self.buf = out, []

    def write(self, s):
        self.buf.append(s)
        return self.out.write(s)

    def flush(self):
        return self.out.flush()


def last_dates():
    out = {}
    for p in (FLOWS, OHLCV, MCAP):
        if os.path.exists(p):
            try:
                out[p] = str(pd.read_csv(p, usecols=["날짜"])["날짜"].max())
            except Exception:
                pass
    return out


def write_status(start, end, flows, ohlcv, log, filled=None):
    """수집 결과 기록. 네이버로 빈칸을 다 메웠으면 정상(ok)이고, 무엇을 대신 받았는지 note 에 남긴다."""
    import json
    filled = filled or {}
    text = "".join(log)
    lasts = last_dates()
    plast = None
    try:
        plast = str(pd.read_csv(PRICES, usecols=["날짜"])["날짜"].max())
    except Exception:
        pass
    krx = None
    if "패스워드 변경" in text or "비밀번호 변경" in text:
        krx = "KRX 로그인 실패 — KRX 계정이 '비밀번호 변경 필요' 상태입니다"
    elif KRX_ERR or "에러페이지 - 한국거래소" in text or "ip-block" in text:
        krx = "KRX 가 GitHub 서버 접속을 막았습니다 (IP 차단 페이지)"
    elif "로그인 실패" in text and "환경 변수" not in text:
        krx = "KRX 로그인 실패 — 아이디·비밀번호(GitHub Secrets KRX_ID/KRX_PW)를 확인하세요"
    problem = None
    behind = [k for k, p in (("시세(OHLCV)", OHLCV), ("시가총액", MCAP), ("수급", FLOWS))
              if plast and lasts.get(p) and lasts[p] < plast]
    if behind:
        problem = f"{'·'.join(behind)} 자료가 {min(lasts[p] for p in (OHLCV, MCAP, FLOWS) if lasts.get(p))}에서 멈춤" + (f" — {krx}" if krx else "")
    elif flows.empty and ohlcv.empty and len(pd.bdate_range(start, end)) > 1 and not filled:
        problem = krx or "KRX에서 자료를 하나도 받지 못했습니다 (접속 장애 또는 로그인 문제)"
    note = None
    if filled:
        note = (f"{krx or 'KRX 일부 실패'} → 네이버로 대신 받음: " +
                ", ".join(f"{k} {v}건" for k, v in filled.items() if v) +
                " (거래대금·시총·수급 금액은 종가 기준 근사)")
    fix = None
    if problem:
        fix = ("KRX 정보데이터시스템(data.krx.co.kr)에 로그인해 비밀번호를 바꾸거나 '다음에 변경'을 누른 뒤, "
               "비밀번호를 바꿨다면 GitHub → Settings → Secrets → KRX_PW 도 새 값으로 바꿔 주세요. "
               "다음 실행 때 빠진 날짜는 저절로 메웁니다.") if krx and "비밀번호" in krx else \
              "다음 실행 때 빠진 날짜를 저절로 다시 받습니다. 이틀 넘게 계속되면 확인이 필요합니다."
    st = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "start": start, "end": end,
          "flows_rows": int(len(flows)), "ohlcv_rows": int(len(ohlcv)), "prices_last": plast,
          "flows_last": lasts.get(FLOWS), "ohlcv_last": lasts.get(OHLCV), "marketcap_last": lasts.get(MCAP),
          "krx": krx, "filled": filled, "note": note,
          "ok": problem is None, "problem": problem, "fix": fix}
    json.dump(st, open(STATUS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(("✓ 수집 상태 정상" if st["ok"] else f"🔴 {problem}") + (f" · {note}" if note else "") + f" → {STATUS}")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", metavar="YYYY-MM-DD", help="이 날짜부터 전체 수집")
    ap.add_argument("--shorts-only", metavar="YYYY-MM-DD",
                    help="공매도만 이 날짜부터 수집 (수급/OHLCV 건너뜀)")
    ap.add_argument("--check", action="store_true", help="검증만")
    args = ap.parse_args()

    if args.check:
        check()
        return

    # 공매도만 수집 모드 (이미 받은 수급/OHLCV 보존, 시간 절약)
    if args.shorts_only:
        codes = get_universe()
        end = datetime.today().strftime("%Y-%m-%d")
        print(f"공매도 전용 수집: {args.shorts_only} ~ {end} ({len(codes)}종목)")
        shorts, sf = collect_shorts_only(codes, args.shorts_only, end)
        if sf:
            print(f"    공매도 실패 {len(sf)}종목: {sf[:5]}{'...' if len(sf)>5 else ''}")
        if shorts.empty:
            print("    🔴 공매도 0건! KRX 로그인(KRX_ID/KRX_PW) 확인 필요")
        save_merge(shorts, SHORTS)
        check()
        return

    codes = get_universe()
    print(f"유니버스: {len(codes)}종목")

    if args.backfill:
        start = args.backfill
        end = datetime.today().strftime("%Y-%m-%d")
        print(f"백필 모드: {start} ~ {end}")
    else:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        # 빠진 기간이 있으면(로그인 실패 등으로 며칠~몇 주 못 받았으면) 그 다음 날부터 다시 받는다
        lasts = last_dates()
        oldest = min(lasts.values()) if lasts else None
        floor = (datetime.today() - timedelta(days=GAP_MAX_DAYS)).strftime("%Y-%m-%d")
        if oldest and oldest < start:
            start = max(oldest, floor)
            print(f"빈 기간 메우기: 가장 늦은 파일의 마지막 날 {oldest} → {start}부터 다시 받음")
        print(f"데일리 모드: {start} ~ {end}")
    tee = _Tee(sys.stdout)
    sys.stdout = tee

    # 1+2) 수급 + 공매도 (종목 루프 1회)
    print("\n[1/2] 투자자 순매수 + 공매도 수집...")
    flows, shorts, ff, sf = collect_per_ticker(codes, start, end)
    if ff:
        print(f"    수급 실패 {len(ff)}종목: {ff[:5]}{'...' if len(ff)>5 else ''}")
    if sf:
        print(f"    공매도 실패 {len(sf)}종목: {sf[:5]}{'...' if len(sf)>5 else ''}")
    if flows.empty:
        print("    🔴 수급 0건! 기간이 너무 짧거나 KRX 접속불가 — 재시도 권장")
    if shorts.empty:
        print("    🔴 공매도 0건! KRX 로그인(KRX_ID/KRX_PW) 확인 필요")
    filled = {}
    if ff:
        nfl, nff = naver_fill_flows(ff, start, end)
        print(f"    네이버로 수급 대신 받음: {nfl['종목코드'].nunique() if len(nfl) else 0}/{len(ff)}종목" + (f" (실패 {len(nff)})" if nff else ""))
        filled["수급"] = int(len(nfl))
        flows = pd.concat([flows, nfl], ignore_index=True) if len(nfl) else flows
    save_merge(flows, FLOWS)
    save_merge(shorts, SHORTS)

    # 3+4) OHLCV + 시총
    print("\n[2/2] OHLCV·시총 수집...")
    dates = trading_dates(start, end)
    print(f"    대상 영업일: {len(dates)}일")
    ohlcv, cap, of = collect_ohlcv_cap(codes, dates)
    if of:
        print(f"    OHLCV 실패 {len(of)}일")
    want = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates}
    have = set(zip(ohlcv["날짜"], ohlcv["종목코드"])) if len(ohlcv) else set()
    if os.path.exists(OHLCV):
        o0 = pd.read_csv(OHLCV, dtype={"종목코드": str}, usecols=["날짜", "종목코드"])
        o0 = o0[o0["날짜"].isin(want)]
        have |= set(zip(o0["날짜"], o0["종목코드"].str.zfill(6)))
    no, nc, nof = naver_fill_ohlcv(codes, dates, have)
    if len(no):
        print(f"    네이버로 OHLCV 대신 받음: {len(no)}행 ({no['날짜'].min()} ~ {no['날짜'].max()})" + (f" · 실패 {len(nof)}종목" if nof else ""))
        filled["시세"] = int(len(no))
        ohlcv = pd.concat([ohlcv, no], ignore_index=True)
        cap = pd.concat([cap, nc], ignore_index=True) if len(nc) else cap
    save_merge(ohlcv, OHLCV)
    save_merge(cap, MCAP)
    sys.stdout = tee.out
    write_status(start, end, flows, ohlcv, tee.buf, filled)

    check()


if __name__ == "__main__":
    main()



