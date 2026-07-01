"""
매매 신호 생성 v3 - 실제 거래내역(my_trades.csv) 기반 · 주식수 회계
확정 전략: 상위20종목/진입10위/이탈20위/최소30일/손절-10% (워크포워드 검증)
입력: data/my_trades.csv  (날짜,종목명,구분,금액[,수량]) — 종목명만 적으면 코드/가격 자동
  · 매수: 금액 또는 수량 중 하나 (둘 다 있으면 수량=주식수, 금액=실제원가)
  · 매도: 수량(부분매도) 또는 금액 또는 공란(전량). 실현손익 자동 집계.
  · 주식수 기반이라 물타기 평단·부분매도가 정확히 계산됨.
"""
import os
import json
import pandas as pd
import numpy as np

SCORES = "data/scores.csv"
PRICES = "data/prices.csv"
TRADES = "data/my_trades.csv"
OUT = "data/signals_today.csv"

TOP_N = 20
ENTRY_RANK = 10
EXIT_RANK = 20
MIN_HOLD = 30
STOP_LOSS = -0.10
TRAIL_STOP = 0.08   # 고점 대비 -8% 하락 시 청산 (백테스트: MDD -42%→-39%, 누적 +368%→+442%)


def _pnum(v):
    """빈칸/NaN 안전 float (없으면 None)"""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in ("", "nan", "None"):
        return None
    try:
        f = float(s)
        return f if f > 0 else None
    except ValueError:
        return None


def load_trades(name2code, price_lookup):
    """주식수 기반 회계. 반환: (열린포지션, 경고, 실현손익합계원)"""
    if not os.path.exists(TRADES):
        return {}, [], 0.0
    t = pd.read_csv(TRADES, dtype=str)
    t.columns = [c.strip() for c in t.columns]
    t["날짜"] = pd.to_datetime(t["날짜"], errors="coerce")
    t = t.dropna(subset=["날짜"]).sort_values("날짜")
    has_qty = "수량" in t.columns
    positions = {}          # code -> {종목명,진입일,shares,평단가,투자금액,realized}
    realized_total = 0.0
    warnings = []
    for _, r in t.iterrows():
        name = str(r["종목명"]).strip()
        action = str(r["구분"]).strip()
        d = r["날짜"]
        code = name2code.get(name)
        if code is None:
            warnings.append(f"종목명 '{name}' 매칭 실패 (오타 확인)")
            continue
        price = price_lookup(code, d)
        if price is None:
            warnings.append(f"{name} {d.date()} 가격 없음")
            continue
        qty = _pnum(r.get("수량")) if has_qty else None
        amt = _pnum(r.get("금액"))

        if action == "매수":
            # 수량 우선(주식수), 없으면 금액/가격. 원가는 금액 있으면 금액, 없으면 수량*가격
            if qty:
                sh = qty
                cost = amt if amt else qty * price
            elif amt:
                sh = amt / price
                cost = amt
            else:
                warnings.append(f"{name} {d.date()} 매수 금액/수량 둘 다 비어있음 → 건너뜀")
                continue
            if code in positions:
                p = positions[code]
                p["shares"] += sh
                p["투자금액"] += cost
                p["평단가"] = p["투자금액"] / p["shares"] if p["shares"] else price
            else:
                positions[code] = {"종목명": name, "진입일": d, "shares": sh,
                                   "평단가": cost / sh if sh else price,
                                   "투자금액": cost, "realized": 0.0}

        elif action == "매도":
            if code not in positions:
                warnings.append(f"{name} {d.date()} 보유 없음인데 매도 기록 → 무시")
                continue
            p = positions[code]
            # 매도 수량: 수량 지정 > 금액/가격 > 전량
            if qty:
                sell_sh = min(qty, p["shares"])
            elif amt:
                sell_sh = min(amt / price, p["shares"])
            else:
                sell_sh = p["shares"]
            avg = p["평단가"]
            pnl = sell_sh * (price - avg)      # 실현손익(원)
            realized_total += pnl
            p["realized"] += pnl
            p["shares"] -= sell_sh
            p["투자금액"] = p["shares"] * avg   # 남은 원가
            if p["shares"] <= 1e-6:            # 전량 청산 → 포지션 종료
                positions.pop(code, None)
    return positions, warnings, realized_total


def main():
    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])
    last = s["날짜"].max()
    trading_dates = sorted(s["날짜"].unique())

    px = pd.read_csv(PRICES, dtype={"종목코드": str})
    px["종목코드"] = px["종목코드"].str.zfill(6)
    px["날짜"] = pd.to_datetime(px["날짜"])
    name2code = dict(zip(px["종목명"], px["종목코드"]))
    px_sorted = px.sort_values("날짜")

    def price_lookup(code, d):
        sub = px_sorted[(px_sorted["종목코드"] == code) & (px_sorted["날짜"] <= d)]
        return int(sub.iloc[-1]["종가"]) if len(sub) else None

    today = s[s["날짜"] == last].sort_values("종합점수", ascending=False).reset_index(drop=True)
    today["순위"] = today.index + 1
    rank_map = dict(zip(today["종목코드"], today["순위"]))
    score_map = dict(zip(today["종목코드"], today["종합점수"]))
    price_now = dict(zip(today["종목코드"], today["종가"]))

    # 외국인 수급 필터 (검증: 순매도 종목 제외 시 Sharpe 1.01→1.31, 최악폴드 -0.98→+0.10)
    # 학술 근거: 외국인은 한국 대형주 방향 예측에 우위 (Kim et al. 2014)
    flow_strength = {}  # 종목코드 -> 최근 20일 외국인 매수강도(시총대비)
    FLOWS = "data/flows.csv"
    MCAP = "data/marketcap.csv"
    if os.path.exists(FLOWS):
        fl = pd.read_csv(FLOWS, dtype={"종목코드": str})
        fl["종목코드"] = fl["종목코드"].str.zfill(6)
        fl["날짜"] = pd.to_datetime(fl["날짜"])
        fl = fl.sort_values(["종목코드", "날짜"])
        fl["외국인20"] = fl.groupby("종목코드")["외국인순매수"].transform(
            lambda x: x.rolling(20, min_periods=10).sum())
        if os.path.exists(MCAP):
            mc = pd.read_csv(MCAP, dtype={"종목코드": str})
            mc["종목코드"] = mc["종목코드"].str.zfill(6)
            mc["날짜"] = pd.to_datetime(mc["날짜"])
            fl = fl.merge(mc[["날짜", "종목코드", "시가총액"]], on=["날짜", "종목코드"], how="left")
            fl["강도"] = fl["외국인20"] / fl["시가총액"]
        else:
            fl["강도"] = fl["외국인20"]
        latest_fl = fl[fl["날짜"] == fl["날짜"].max()]
        flow_strength = dict(zip(latest_fl["종목코드"], latest_fl["강도"]))

    positions, warns, realized_total = load_trades(name2code, price_lookup)

    def held_days(entry):
        return sum(1 for d in trading_dates if entry <= d <= last) - 1

    rows = []
    for code, pos in positions.items():
        r = rank_map.get(code)
        p_now = price_now.get(code) or price_lookup(code, last)
        ret = (p_now / pos["평단가"] - 1) if pos["평단가"] else 0
        days = held_days(pos["진입일"])
        # 진입 후 고점 추적 (트레일링 스탑용)
        peak = px_sorted[(px_sorted["종목코드"] == code) &
                         (px_sorted["날짜"] >= pos["진입일"]) &
                         (px_sorted["날짜"] <= last)]["종가"].max()
        peak = max(peak, p_now) if pd.notna(peak) else p_now
        trail_dd = (p_now / peak - 1) if peak else 0
        if ret <= STOP_LOSS:
            action, reason = "🔴손절", f"{ret*100:.1f}% (손절선 {STOP_LOSS*100:.0f}%)"
        elif trail_dd <= -TRAIL_STOP:
            action, reason = "🔴손절", f"고점대비 {trail_dd*100:.1f}% (트레일 -{TRAIL_STOP*100:.0f}%)"
        elif (r is None or r > EXIT_RANK) and days >= MIN_HOLD:
            action, reason = "🔵매도", f"{r if r else '권외'}위 이탈+{days}일"
        elif (r is None or r > EXIT_RANK):
            action, reason = "⏳보유", f"{r if r else '권외'}위지만 {days}일(<{MIN_HOLD})"
        else:
            action, reason = "🟢유지", f"{r}위 유지, {days}일"
        stop_price = int(round(pos["평단가"] * (1 + STOP_LOSS)))      # 평단 -10%
        trail_price = int(round(peak * (1 - TRAIL_STOP))) if peak else stop_price  # 고점 -8%
        # 실제 손절 발동가 = 둘 중 높은 쪽(먼저 닿는 쪽)
        guard_price = max(stop_price, trail_price)
        rows.append({"구분": action, "종목코드": code, "종목명": pos["종목명"],
                     "순위": r, "점수": score_map.get(code), "보유일": days,
                     "평단가": int(pos["평단가"]), "현재가": int(p_now),
                     "수익률%": round(ret*100, 1), "사유": reason,
                     "손절가": stop_price, "트레일가": trail_price, "감시가": guard_price,
                     "투자금액": int(pos.get("투자금액", 0) or 0),
                     "수량": round(pos.get("shares", 0), 4),
                     "실현손익": int(round(pos.get("realized", 0) or 0))})

    held_codes = set(positions.keys())
    n_buyable = TOP_N - len(held_codes)
    if n_buyable > 0:
        cands = today[(today["순위"] <= ENTRY_RANK) & (~today["종목코드"].isin(held_codes))]
        bought = 0
        for _, c in cands.iterrows():
            if bought >= n_buyable:
                break
            code = c["종목코드"]
            # 외국인 필터: 순매도 중이면 매수 보류
            strength = flow_strength.get(code, 0)
            if pd.notna(strength) and strength < 0:
                rows.append({"구분": "⚪보류", "종목코드": code, "종목명": c["종목명"],
                             "순위": int(c["순위"]), "점수": c["종합점수"], "보유일": 0,
                             "평단가": "", "현재가": int(c["종가"]), "수익률%": "",
                             "사유": f"{int(c['순위'])}위지만 외국인 순매도중 (매수보류)",
                             "손절가": "", "트레일가": "", "감시가": "", "투자금액": "",
                             "수량": "", "실현손익": ""})
                continue
            buy_px = int(c["종가"])
            rows.append({"구분": "🟡매수", "종목코드": code, "종목명": c["종목명"],
                         "순위": int(c["순위"]), "점수": c["종합점수"], "보유일": 0,
                         "평단가": "", "현재가": buy_px, "수익률%": "",
                         "사유": f"상위{ENTRY_RANK}위+외국인매수({int(c['순위'])}위)",
                         "손절가": int(round(buy_px*(1+STOP_LOSS))), "트레일가": "", "감시가": "",
                         "투자금액": "", "수량": "", "실현손익": ""})
            bought += 1

    sig = pd.DataFrame(rows)
    if len(sig):
        order = {"🔴손절": 0, "🔵매도": 1, "🟡매수": 2, "⚪보류": 3, "🟢유지": 4, "⏳보유": 5}
        sig["_o"] = sig["구분"].map(order).fillna(9)
        sig = sig.sort_values(["_o", "순위"]).drop(columns="_o")
        sig.to_csv(OUT, index=False)

    # 계좌 요약(실현손익 포함 — 청산된 종목까지) → build_data가 읽어 대시보드에 표시
    acct = {"realized_total": int(round(realized_total)),
            "invested": int(round(sum(p["투자금액"] for p in positions.values()))),
            "n_positions": len(positions)}
    json.dump(acct, open("data/account.json", "w", encoding="utf-8"), ensure_ascii=False)

    print(f"\n{'='*54}")
    print(f"  {last.date()} 매매 신호  (상위20/진입10/30일/손절-10%)")
    print(f"{'='*54}")
    if warns:
        print("\n⚠️ 입력 확인 필요:")
        for w in warns:
            print(f"   - {w}")

    def show(tag, label):
        sub = sig[sig["구분"] == tag] if len(sig) else pd.DataFrame()
        print(f"\n■ {label} ({len(sub)})")
        if not len(sub):
            print("   없음"); return
        for _, r in sub.iterrows():
            pnl = f" ({r['수익률%']:+}%)" if r["수익률%"] != "" else ""
            gp = r.get("감시가", "")
            sp = r.get("손절가", "")
            extra = ""
            if tag in ("🔴손절", "🔵매도") and gp != "":
                extra = f"  → 지정가매도 {int(r['현재가']):,}원 / 손절감시가 {int(gp):,}원"
            elif tag == "🟡매수" and sp != "":
                extra = f"  → 매수 {int(r['현재가']):,}원 / 손절가 {int(sp):,}원 설정"
            print(f"   {r['종목명']}  {r['순위']}위{pnl}  {r['사유']}{extra}")

    show("🔴손절", "손절 (즉시)")
    show("🔵매도", "매도")
    show("🟡매수", "매수 추천")
    show("⚪보류", "매수 보류 (외국인 순매도)")
    show("🟢유지", "보유 유지")
    show("⏳보유", "보유(기간 미달)")
    inv = sum(p["투자금액"] for p in positions.values())
    print(f"\n현재 보유: {len(positions)}종목" + (f", 투자원금 {inv:,.0f}원" if inv else ""))
    if abs(realized_total) >= 1:
        print(f"누적 실현손익: {realized_total:+,.0f}원")
    print()


if __name__ == "__main__":
    main()



