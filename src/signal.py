"""
매매 신호 생성 - 매일 장마감 후 실행
백테스트로 검증된 규칙을 실전 신호로 변환:
  · 상위 N종목(기본 20)을 목표 보유 종목으로
  · 보유기간 HOLD일(기본 40) 경과 후 재평가
  · 매수신호: 목표에 새로 진입한 종목 (지금 안 갖고 있는데 상위권 진입)
  · 매도신호: 보유 중인데 상위권에서 이탈 + 보유 HOLD일 경과
  · 유지: 보유 중이고 아직 상위권 or 보유기간 미달

상태파일 data/portfolio.json 으로 보유현황을 다음날까지 기억.
출력:
  · data/signals_today.csv (오늘 신호 표)
  · data/portfolio.json (갱신된 보유현황)
  · 콘솔/Summary 출력
환경변수:
  TOP_N (기본 20), HOLD_DAYS (기본 40)
"""
import os
import json
from datetime import datetime
import pandas as pd

SCORES = "data/scores.csv"
PORT = "data/portfolio.json"
OUT = "data/signals_today.csv"

TOP_N = int(os.environ.get("TOP_N", "20"))
HOLD_DAYS = int(os.environ.get("HOLD_DAYS", "40"))


def load_portfolio():
    if os.path.exists(PORT):
        with open(PORT, encoding="utf-8") as f:
            return json.load(f)
    return {}  # {종목코드: {"종목명":.., "진입일":.., "진입점수":.., "진입가":..}}


def save_portfolio(p):
    with open(PORT, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


def biz_days_between(d1, d2, trading_dates):
    """두 날짜 사이 거래일 수 (보유기간 계산)"""
    td = [d for d in trading_dates if d1 <= d <= d2]
    return max(len(td) - 1, 0)


def main():
    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    s["종목코드"] = s["종목코드"].str.zfill(6)
    last = s["날짜"].max()
    trading_dates = sorted(s["날짜"].unique())

    today = s[s["날짜"] == last].copy().sort_values("종합점수", ascending=False)
    today = today.reset_index(drop=True)
    today["순위"] = today.index + 1
    top_codes = set(today.head(TOP_N)["종목코드"])

    port = load_portfolio()
    score_map = dict(zip(today["종목코드"], today["종합점수"]))
    name_map = dict(zip(today["종목코드"], today["종목명"]))
    price_map = dict(zip(today["종목코드"], today["종가"]))
    rank_map = dict(zip(today["종목코드"], today["순위"]))

    signals = []  # 출력용

    # 1) 매도/유지 판정 (현재 보유종목 순회)
    still_hold = {}
    for code, info in port.items():
        cur_score = score_map.get(code)
        cur_rank = rank_map.get(code)
        cur_price = price_map.get(code)
        held = biz_days_between(info["진입일"], last, trading_dates)
        in_top = code in top_codes

        pnl = None
        if cur_price and info.get("진입가"):
            pnl = (cur_price / info["진입가"] - 1) * 100

        if not in_top and held >= HOLD_DAYS:
            action = "매도"
            reason = f"상위{TOP_N}위 이탈 + {held}일 보유(≥{HOLD_DAYS})"
        elif not in_top and held < HOLD_DAYS:
            action = "유지(관찰)"
            reason = f"상위 이탈했으나 보유 {held}일(<{HOLD_DAYS}), 기간 미달"
            still_hold[code] = info
        else:
            action = "유지"
            reason = f"상위{TOP_N}위 유지(현 {cur_rank}위), 보유 {held}일"
            still_hold[code] = info

        signals.append({
            "구분": action, "종목코드": code,
            "종목명": name_map.get(code, info.get("종목명", "")),
            "현재순위": cur_rank, "현재점수": cur_score,
            "보유일": held, "수익률%": round(pnl, 1) if pnl is not None else "",
            "사유": reason,
        })

    # 2) 매수 판정 (상위 N 중 아직 미보유)
    for _, r in today.head(TOP_N).iterrows():
        code = r["종목코드"]
        if code in port:
            continue  # 이미 보유 → 위에서 처리됨
        signals.append({
            "구분": "매수", "종목코드": code, "종목명": r["종목명"],
            "현재순위": int(r["순위"]), "현재점수": r["종합점수"],
            "보유일": 0, "수익률%": "",
            "사유": f"신규 상위{TOP_N}위 진입({int(r['순위'])}위)",
        })
        still_hold[code] = {
            "종목명": r["종목명"], "진입일": last,
            "진입점수": float(r["종합점수"]), "진입가": int(r["종가"]),
        }

    save_portfolio(still_hold)

    # 3) 출력
    sig = pd.DataFrame(signals)
    order = {"매수": 0, "매도": 1, "유지": 2, "유지(관찰)": 3}
    sig["_o"] = sig["구분"].map(order).fillna(9)
    sig = sig.sort_values(["_o", "현재순위"]).drop(columns="_o")
    sig.to_csv(OUT, index=False)

    buys = sig[sig["구분"] == "매수"]
    sells = sig[sig["구분"] == "매도"]

    print(f"\n{'='*50}")
    print(f"  {last} 매매 신호  (상위{TOP_N} / {HOLD_DAYS}일 보유)")
    print(f"{'='*50}")
    print(f"\n■ 내일 매수 ({len(buys)}종목)")
    if len(buys):
        for _, r in buys.iterrows():
            print(f"   [{r['현재순위']:>2}위] {r['종목명']}  점수 {r['현재점수']}")
    else:
        print("   없음")

    print(f"\n■ 내일 매도 ({len(sells)}종목)")
    if len(sells):
        for _, r in sells.iterrows():
            pnl = f"{r['수익률%']:+}%" if r['수익률%'] != "" else ""
            print(f"   {r['종목명']}  ({r['사유']}) {pnl}")
    else:
        print("   없음")

    holds = sig[sig["구분"].str.startswith("유지")]
    print(f"\n■ 보유 유지 ({len(holds)}종목)")
    for _, r in holds.iterrows():
        pnl = f"{r['수익률%']:+}%" if r['수익률%'] != "" else ""
        print(f"   {r['종목명']}  {r['현재순위']}위  보유{r['보유일']}일  {pnl}")

    print(f"\n총 보유 예정: {len(still_hold)}종목\n")


if __name__ == "__main__":
    main()

