"""
4상태 판단 엔진 — 모든 종목을 매일 **매수 / 유지 / 매도 / 관망** 중 하나로 확정한다.

이 파일은 라이브(autotrader.py)와 검증(engine_backtest.py)이 **같이** import 한다.
백테스트에서 본 성과를 낸 코드와 실제 주문을 내는 코드가 같아야 검증이 의미를 갖는다.

규칙은 세 층으로 나뉜다 (README의 신뢰도 배지와 같은 체계)
  🟩 검증됨 — 이 저장소 데이터로 train/test·교차확인을 통과한 규칙 → 매매를 결정한다
      종합순위 · 외국인 수급 필터 · 손절/트레일/익절 사다리 · 순위이탈 · 급락장 바닥확인 게이트
  🟧 안전장치 — 검증할 수 없거나(뉴스 이력 1개월) 수익이 아니라 사고를 막는 규칙
      데이터 신선도 · 중대 공시 거부권 · 계좌 서킷브레이커 · 주문 한도
  ⬜ 참고 — 논문·서적에서 가져왔으나 검증에서 탈락한 지표 → 화면에만 보여주고 결정엔 쓰지 않는다
      추세정렬·52주고가·RSI·F-score·SUE·섹터 상대가치·섹터 모멘텀 (engine_lab.py 결과)

상태 정의
  매수 — 보유 X · 순위 ≤ entry_rank · 빈 자리 있음 · 모든 게이트 통과
  유지 — 보유 O · 어떤 매도 조건에도 걸리지 않음
  매도 — 보유 O · 손절/트레일/순위이탈/강제매도 공시 (전량) 또는 +25% 부분익절 (1/3)
  관망 — 보유 X · 순위 ≤ watch_rank 이지만 매수 조건 중 하나라도 미충족
         → '무엇이 바뀌면 매수로 넘어가는지'를 같이 기록한다
"""
import re

import numpy as np
import pandas as pd

BUY, HOLD, SELL, WATCH = "매수", "유지", "매도", "관망"


def _f(m, t, c):
    """행렬 m의 (t, 종목 c) 값. 없으면 nan."""
    if m is None:
        return np.nan
    try:
        v = m.iat[t, m.columns.get_loc(c)]
    except KeyError:
        return np.nan
    return float(v) if v is not None and pd.notna(v) else np.nan


def trail_width(peak_gain, st):
    for lvl, w in st["trail_tiers"]:
        if peak_gain >= lvl:
            return w
    return st["trail_base"]


# ══════════════════════════════════════════════════════════════
# 시장 게이트
# ══════════════════════════════════════════════════════════════
def market_gate(market, t, st):
    """신규매수를 열지 닫을지. 반환 dict(open, regime, reason, need)."""
    row = market.iloc[t]
    regime = row["regime"] if isinstance(row["regime"], str) else "판정불가"
    info = {"regime": regime, "trend": float(row["trend"]) if pd.notna(row["trend"]) else None,
            "breadth": float(row["breadth"]) if pd.notna(row["breadth"]) else None,
            "dd": float(row["dd"]) if pd.notna(row["dd"]) else None}
    if not st.get("crash_gate", True) or regime != "급락장":
        return {**info, "open": True, "reason": f"{regime} — 신규매수 허용", "need": ""}
    win = int(st.get("crash_rebound_window", 20))
    th = float(st.get("crash_rebound_min", 0.05))
    lo = market["idx"].iloc[max(0, t - win + 1): t + 1].min()
    reb = float(row["idx"] / lo - 1) if lo else 0.0
    info["rebound"] = reb
    if reb >= th:
        return {**info, "open": True,
                "reason": f"급락장이지만 바닥 확인 (지수 {win}일 저점 대비 {reb*100:+.1f}%)", "need": ""}
    return {**info, "open": False,
            "reason": f"급락장 · 바닥 미확인 (지수 {win}일 저점 대비 {reb*100:+.1f}%)",
            "need": f"지수가 {win}일 저점 대비 +{th*100:.0f}% 이상 반등"}


# ══════════════════════════════════════════════════════════════
# 참고 지표 (결정에 쓰지 않음 — 화면 표시용)
# ══════════════════════════════════════════════════════════════
REF_SIGNALS = [
    # key, 라벨, 좋은 조건, 표시 형식, 출처
    ("trend_up", "추세정렬(50>200일선)", lambda v: v >= 0.5, lambda v: "예" if v >= 0.5 else "아니오",
     "Faber 2007 · Minervini"),
    ("hi52", "52주 고가 대비", lambda v: v >= 0.85, lambda v: f"{v*100:.0f}%", "George & Hwang 2004"),
    ("rsi14", "RSI(14)", lambda v: 30 <= v <= 70, lambda v: f"{v:.0f}", "Wilder 1978"),
    ("disp20", "20일선 이격", lambda v: v < 0.10, lambda v: f"{v*100:+.1f}%", "position_lab"),
    ("fscore", "F-score(7점)", lambda v: v >= 4, lambda v: f"{v:.0f}/7", "Piotroski 2000"),
    ("sue", "이익 서프라이즈", lambda v: v > 0, lambda v: f"{v*100:+.2f}%p", "Bernard & Thomas 1989"),
    ("op_yoy", "영업이익 TTM 증가", lambda v: v > 0, lambda v: f"{v*100:+.0f}%", "—"),
    ("sec_value_rank", "섹터 내 가치 순위", lambda v: v >= 0.5, lambda v: f"상위 {(1-v)*100:.0f}%",
     "Asness et al. 2000"),
    ("sec_mom_rank", "섹터 6개월 모멘텀", lambda v: v >= 0.5, lambda v: f"상위 {(1-v)*100:.0f}%",
     "Moskowitz & Grinblatt 1999"),
    ("sec_earn", "섹터 이익 증가율(중앙값)", lambda v: v > 0, lambda v: f"{v*100:+.0f}%", "—"),
]


def reference(feats, t, c):
    out, good, bad = [], 0, 0
    for key, label, ok, fmt, src in REF_SIGNALS:
        v = _f(feats.get(key), t, c)
        if not np.isfinite(v):
            continue
        g = bool(ok(v))
        good += g
        bad += not g
        out.append({"key": key, "label": label, "value": fmt(v), "good": g, "source": src})
    return {"items": out, "good": good, "bad": bad}


# ══════════════════════════════════════════════════════════════
# 사건(공시·기사) 거부권
# ══════════════════════════════════════════════════════════════
def events_for(news, code, asof, days):
    """news DataFrame(종목코드, 제목, 발행일) 에서 최근 days일 제목 목록."""
    if news is None or not len(news):
        return []
    sub = news[news["종목코드"] == code]
    if not len(sub):
        return []
    d = sub["_dt"]
    m = (d <= asof + pd.Timedelta(days=1)) & (d >= asof - pd.Timedelta(days=days))
    return list(sub.loc[m, "제목"].astype(str))


def load_news(path="data/news.csv"):
    try:
        n = pd.read_csv(path, dtype={"종목코드": str})
    except Exception:
        return None
    n["종목코드"] = n["종목코드"].str.zfill(6)
    raw = n["발행일"].astype(str).str.strip()
    n["_dt"] = pd.to_datetime(raw.str.slice(0, 10).str.replace("-", ""), format="%Y%m%d",
                              errors="coerce")
    return n.dropna(subset=["_dt"])


# ══════════════════════════════════════════════════════════════
# 판단
# ══════════════════════════════════════════════════════════════
def decide(t, panel, feats, positions, cfg, news=None, flags=None):
    """
    t        : 판단 기준일 인덱스 (panel['dates'][t] 장 마감 정보까지 사용)
    positions: {code: {"shares", "avg", "entry_t", "peak", "partial_done"}}
    flags    : {"flow_stale": bool, "account_halt": str|None} — 실행기가 넘기는 안전장치 상태
    반환     : (decisions list, gate dict)
    """
    st, rk = cfg["strategy"], cfg.get("risk", {})
    flags = flags or {}
    dates, names = panel["dates"], panel["names"]
    close, rank, flow = panel["close"], feats["rank"], feats.get("flow")
    asof = dates[t]
    gate = market_gate(panel["market"], t, st)
    veto_re = re.compile(rk["veto_buy_events"]) if rk.get("veto_buy_events") else None
    force_re = re.compile(rk["force_sell_events"]) if rk.get("force_sell_events") else None
    lb = int(rk.get("event_lookback_days", 14))
    out = []

    def base(c):
        r = _f(rank, t, c)
        return {"code": c, "name": names.get(c, c), "date": str(asof.date()),
                "rank": int(r) if np.isfinite(r) else None, "price": _f(close, t, c),
                "sector": panel.get("sector_of", {}).get(c, ""),
                "ref": reference(feats, t, c)}

    # ── 1) 보유 종목: 매도 / 유지 ────────────────────────────────
    n_exit = 0
    for c, q in positions.items():
        d = base(c)
        cur = d["price"]
        if not np.isfinite(cur):
            d.update(state=HOLD, action="", reason="오늘 가격 없음 — 판단 보류", checks=[])
            out.append(d)
            continue
        peak = max(q.get("peak") or cur, cur)
        ret = cur / q["avg"] - 1
        peak_gain = peak / q["avg"] - 1
        tw = trail_width(peak_gain, st)
        held = t - q["entry_t"]
        r = d["rank"]
        stop_px = q["avg"] * (1 - st["stop_loss"])
        trail_px = peak * (1 - tw)
        evs = events_for(news, c, asof, lb)
        forced = [e for e in evs if force_re and force_re.search(e)]
        checks = [
            {"label": f"손절선 -{st['stop_loss']*100:.0f}%", "ok": ret > -st["stop_loss"],
             "value": f"{ret*100:+.1f}%", "tier": "검증됨"},
            {"label": f"트레일 고점대비 -{tw*100:.0f}%", "ok": cur / peak - 1 > -tw,
             "value": f"{(cur/peak-1)*100:+.1f}%", "tier": "검증됨"},
            {"label": f"순위 {st['exit_rank']}위 이내 (또는 {st['min_hold_days']}일 미만)",
             "ok": (r is not None and r <= st["exit_rank"]) or held < st["min_hold_days"],
             "value": f"{r or '권외'}위 · {held}일", "tier": "검증됨"},
            {"label": "강제매도 공시 없음", "ok": not forced,
             "value": forced[0][:40] if forced else "없음", "tier": "안전장치"},
        ]
        d.update(held_days=held, ret=ret, peak=peak, avg=q["avg"], shares=q["shares"],
                 guard_price=max(stop_px, trail_px), trail_width=tw, checks=checks)
        # 순서 = signal.py / strategy_lab.simulate 와 동일 (부분익절 → 손절 → 트레일 → 이탈)
        tpg = st.get("take_profit_gain")                 # 없으면(null) 부분익절 끔
        tp_due = bool(tpg) and ret >= tpg and not q.get("partial_done")
        if ret <= -st["stop_loss"]:
            d.update(state=SELL, action="전량매도", fraction=1.0,
                     reason=f"손절 {ret*100:+.1f}% (기준 -{st['stop_loss']*100:.0f}%)")
        elif cur / peak - 1 <= -tw:
            d.update(state=SELL, action="전량매도", fraction=1.0,
                     reason=f"트레일 고점대비 {(cur/peak-1)*100:.1f}% (폭 -{tw*100:.0f}%, "
                            f"고점수익 {peak_gain*100:+.0f}%)")
        elif forced:
            d.update(state=SELL, action="전량매도", fraction=1.0, reason=f"중대 공시: {forced[0][:50]}")
        elif tp_due:
            d.update(state=SELL, action="부분익절", fraction=float(st["take_profit_fraction"]),
                     reason=f"{ret*100:+.1f}% 도달 — 보유의 {st['take_profit_fraction']*100:.0f}% 익절, "
                            f"나머지는 트레일 -{tw*100:.0f}%로 유지")
        elif (r is None or r > st["exit_rank"]) and held >= st["min_hold_days"]:
            d.update(state=SELL, action="전량매도", fraction=1.0,
                     reason=f"순위 {r or '권외'}위로 이탈 + {held}일 보유")
        else:
            why = (f"{r}위 · {held}일 · {ret*100:+.1f}% · 감시가 {d['guard_price']:,.0f}원"
                   if r else f"권외지만 {held}일(<{st['min_hold_days']}) · 감시가 {d['guard_price']:,.0f}원")
            d.update(state=HOLD, action="", reason=why)
        if d["state"] == SELL and d["action"] == "전량매도":
            n_exit += 1
        out.append(d)

    # ── 2) 비보유: 매수 / 관망 ──────────────────────────────────
    free = st["max_positions"] - (len(positions) - n_exit)
    cand = []
    for c in panel["codes"]:
        if c in positions:
            continue
        r = _f(rank, t, c)
        if not np.isfinite(r) or r > st["watch_rank"]:
            continue
        cand.append((int(r), c))
    cand.sort()
    for r, c in cand:
        d = base(c)
        fs = _f(flow, t, c)
        evs = events_for(news, c, asof, lb)
        vetoed = [e for e in evs if veto_re and veto_re.search(e)]
        checks = [
            {"label": f"종합순위 {st['entry_rank']}위 이내", "ok": r <= st["entry_rank"],
             "value": f"{r}위", "tier": "검증됨"},
            {"label": "시장 게이트 (급락장 바닥확인)", "ok": gate["open"],
             "value": gate["reason"], "tier": "검증됨"},
        ]
        if st.get("flow_filter", True):
            if flags.get("flow_stale") and rk.get("stale_flow_policy", "watch") == "watch":
                checks.append({"label": "외국인 수급 (데이터 신선)", "ok": False,
                               "value": "수급 데이터가 낡아 확인 불가", "tier": "안전장치"})
            elif not flags.get("flow_stale"):
                checks.append({"label": "외국인 20일 순매도 아님", "ok": not (np.isfinite(fs) and fs < 0),
                               "value": f"{fs*1e4:+.1f}bp(시총대비)" if np.isfinite(fs) else "자료없음",
                               "tier": "검증됨"})
        cd = int(st.get("cooldown_days", 0) or 0)
        ex_t = (flags.get("recent_exits") or {}).get(c)
        if cd and ex_t is not None and t - ex_t < cd:
            checks.append({"label": f"재진입 대기 {cd}거래일", "ok": False,
                           "value": f"청산 후 {t - ex_t}일", "tier": "선택 규칙"})
        checks.append({"label": "중대 악재 공시 없음", "ok": not vetoed,
                       "value": vetoed[0][:40] if vetoed else f"최근 {lb}일 없음", "tier": "안전장치"})
        if flags.get("pause_entries"):
            checks.append({"label": "신규매수 일시정지 해제", "ok": False,
                           "value": "config/PAUSE — 사용자가 신규 매수를 멈춤", "tier": "안전장치"})
        if flags.get("account_halt"):
            checks.append({"label": "계좌 서킷브레이커", "ok": False, "value": flags["account_halt"],
                           "tier": "안전장치"})
        all_ok = all(x["ok"] for x in checks)
        if all_ok and free > 0:
            d.update(state=BUY, action="신규매수", checks=checks,
                     reason=f"{r}위 · 모든 조건 통과 · 손절가 {d['price']*(1-st['stop_loss']):,.0f}원")
            free -= 1
        else:
            if all_ok:
                checks.append({"label": "빈 자리", "ok": False,
                               "value": f"보유 {st['max_positions']}/{st['max_positions']}",
                               "tier": "검증됨"})
            fails = [x for x in checks if not x["ok"]]
            need = []
            for x in fails:
                if x["label"].startswith("종합순위"):
                    need.append(f"순위 {st['entry_rank']}위 이내 진입 (현재 {r}위)")
                elif x["label"].startswith("시장"):
                    need.append(gate["need"])
                elif "수급" in x["label"] or "외국인" in x["label"]:
                    need.append("외국인 20일 순매수 전환" if "낡아" not in x["value"]
                                else "수급 데이터 갱신 (collect_flows)")
                elif "공시" in x["label"]:
                    need.append(f"악재 공시 후 {lb}일 경과")
                elif x["label"].startswith("재진입 대기"):
                    need.append(f"청산 후 {cd}거래일 경과 (현재 {t - ex_t}일)")
                elif "일시정지" in x["label"]:
                    need.append("사용자가 일시정지 해제 (autotrade-control → resume)")
                elif x["label"] == "빈 자리":
                    need.append("보유 종목 매도로 자리 생김")
                else:
                    need.append(x["label"])
            d.update(state=WATCH, action="", checks=checks,
                     reason=" · ".join(x["label"] + " ✗" for x in fails), need=need)
        out.append(d)

    order = {SELL: 0, BUY: 1, HOLD: 2, WATCH: 3}
    out.sort(key=lambda d: (order[d["state"]], d["rank"] or 999))
    return out, gate
