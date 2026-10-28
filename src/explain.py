"""
자동매매 해설 — 두 자동 계좌가 '무엇을, 어떤 기준으로, 왜' 했는지와 '무엇을 얼마나 참고했는지'를
사람이 그대로 읽으면 되는 문장으로 만든다. 분석·계산은 여기서 다 하고, 화면(docs/explain.html)은 보여 주기만 한다.

입력
  30일 자동매매 : docs/autotrade.json(오늘 판단·조건 점검) · docs/data.json(점수 구성) · 설정(settings.load)
                  docs/backtest.json(같은 규칙의 과거 거래 — 사고판 예시) · data/scores.csv(그때 순위)
  단기 모의투자 : docs/short.json(오늘 모델·주문) · short_lab(모델 재현 → 오늘 조건 점검·최근 매매 예시)
  진화          : docs/evolve.json · docs/short_evo.json
출력: docs/explain.json   (daily.yml 마지막 단계)
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings  # noqa: E402

OUT = "docs/explain.json"
FACTOR = {
    "s_value": ("가치", "PER·PBR이 낮을수록 높은 점수 — 버는 돈·자산에 비해 주가가 싼가"),
    "s_profit": ("수익성", "ROE·영업이익률이 높을수록 — 돈을 잘 버는 회사인가"),
    "s_grow": ("성장", "매출·영업이익·순이익이 1년 전보다 늘었나"),
    "s_flow": ("수급", "외국인이 최근 20일 동안 (시가총액 대비) 많이 샀나"),
    "s_rmom": ("위험조정 모멘텀", "1년 동안 덜 출렁이면서 꾸준히 올랐나 (수익 ÷ 변동성)"),
    "s_mom": ("모멘텀", "최근 1년(직전 1개월 제외) 상승 추세"),
    "x_ivol": ("저변동", "시장과 무관한 출렁임이 작나"),
    "x_max21": ("급등 적음", "최근 한 달 하루 급등이 없었나"),
    "x_secmom": ("섹터 모멘텀", "속한 업종이 최근 6개월 올랐나"),
}
LOGIC_KEY = {"s_value": "value", "s_profit": "profit", "s_grow": "growth", "s_flow": "flow", "s_rmom": "rmom", "s_mom": "momentum"}
WHY = {"트레일": "산 뒤 가장 높았던 가격에서 정해 둔 폭만큼 내려와서 (트레일링 스톱)",
       "손절": "산 가격보다 정해 둔 만큼 내려가서 (손절)",
       "순위": "순위가 정해 둔 선 밖으로 밀렸고 최소 보유 기간도 지나서",
       "부분익절": "정해 둔 수익(+25%)에 닿아 보유의 1/3을 먼저 팔아서 (부분 익절)",
       "공시": "중대 악재 공시가 나와서"}


def jload(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def won(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:,.0f}원"


def pct(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


# ══════════════════════════════════════════════════════════════
# 30일 자동매매
# ══════════════════════════════════════════════════════════════
def weights_used(cfg, Dd):
    """순위를 매길 때 쓰는 요소와 비중. 진화 챔피언이 rank_weights 를 정했으면 그것, 아니면 종합점수 가중치."""
    st = cfg["strategy"]
    rw = st.get("rank_weights")
    if rw:
        tot = sum(rw.values()) or 1
        src = f"진화 챔피언 {(cfg.get('champion') or {}).get('id', '')}이 정한 비중"
        w = {k: v / tot for k, v in rw.items() if v}
    else:
        w = (Dd or {}).get("market", {}).get("weights") or {}
        tot = sum(w.values()) or 1
        w = {k: v / tot for k, v in w.items() if v}
        src = "종합점수 가중치 (config/weights.yaml · 과거 예측력 검증으로 정함)"
    rows = [{"key": k, "name": FACTOR.get(k, (k, ""))[0], "what": FACTOR.get(k, ("", ""))[1], "w": round(v, 4)}
            for k, v in sorted(w.items(), key=lambda x: -x[1])]
    return rows, src


def rulebook(cfg):
    st, rk, ex = cfg["strategy"], cfg.get("risk", {}), cfg.get("execution", {})
    tiers = st.get("trail_tiers") or []
    tier_txt = " → ".join([f"기본 −{st['trail_base'] * 100:.0f}%"] +
                          [f"고점 수익 +{a * 100:.0f}% 넘으면 −{b * 100:.0f}%" for a, b in sorted(tiers)])
    tp = st.get("take_profit_gain")
    buy = [
        {"rule": f"종합 순위 {st['entry_rank']}위 안", "plain": f"100종목을 매일 점수로 줄 세워 {st['entry_rank']}위 안에 든 종목만 후보로 봅니다.", "tier": "검증됨"},
        {"rule": "시장 게이트", "plain": f"시장이 급락장(100종목 지수가 고점보다 15% 이상 낮음)이면, 지수가 최근 {st.get('crash_rebound_window', 20)}일 최저점보다 "
                                     f"{st.get('crash_rebound_min', 0.05) * 100:.0f}% 이상 반등하기 전까지 새로 사지 않습니다. 과거 급락장 신규매수가 평균적으로 손해였기 때문입니다.", "tier": "검증됨"},
    ]
    if st.get("flow_filter", True):
        buy.append({"rule": "외국인 수급", "plain": "외국인이 최근 20일 동안 순매도한 종목은 사지 않고 기다립니다.", "tier": "검증됨"})
    buy += [
        {"rule": "악재 공시", "plain": f"최근 {rk.get('event_lookback_days', 14)}일 안에 유상증자·전환사채·횡령 같은 공시가 있으면 사지 않습니다.", "tier": "안전장치"},
        {"rule": "빈 자리", "plain": f"계좌를 {st['max_positions']}칸으로 나눠, 빈 칸이 있을 때만 한 칸(계좌의 약 1/{st['max_positions']})씩 삽니다.", "tier": "검증됨"},
        {"rule": "주문 방법", "plain": f"판단은 장 마감 뒤 저녁, 주문은 다음 날 아침 — 매수는 전날 종가보다 {ex.get('buy_slippage', 0.01) * 100:.0f}% 높은 지정가, 매도는 시장가.", "tier": "실행"},
    ]
    sell = [
        {"rule": f"손절 −{st['stop_loss'] * 100:.0f}%", "plain": f"산 가격보다 {st['stop_loss'] * 100:.0f}% 내려가면 전부 팝니다.", "tier": "검증됨"},
        {"rule": "트레일링 스톱", "plain": f"산 뒤 가장 높았던 가격에서 일정 폭 이상 내려오면 전부 팝니다. 폭은 많이 오를수록 좁혀 수익을 지킵니다: {tier_txt}.", "tier": "검증됨"},
    ]
    if tp:
        sell.append({"rule": f"부분 익절 +{tp * 100:.0f}%", "plain": f"+{tp * 100:.0f}%에 닿으면 보유의 {st.get('take_profit_fraction', 0.3333) * 100:.0f}%를 먼저 팔고, 나머지는 트레일링 스톱으로 끌고 갑니다.", "tier": "검증됨"})
    sell += [
        {"rule": f"순위 이탈 ({st['exit_rank']}위 밖 + {st['min_hold_days']}일)", "plain": f"순위가 {st['exit_rank']}위 밖으로 밀렸고 {st['min_hold_days']}거래일 이상 들고 있었으면 팝니다. {st['min_hold_days']}일 전에는 순위가 떨어져도 기다립니다(잦은 매매 비용 방지).", "tier": "검증됨"},
        {"rule": "강제 매도 공시", "plain": "상장폐지·관리종목·횡령 같은 공시가 나오면 바로 팝니다.", "tier": "안전장치"},
    ]
    safety = [
        {"rule": "계좌 서킷브레이커", "plain": f"계좌가 고점보다 {rk.get('max_account_drawdown', 0.3) * 100:.0f}% 빠지면 새로 사지 않습니다."},
        {"rule": "주문 한도", "plain": f"한 번 주문 최대 {rk.get('max_order_krw', 0):,.0f}원, 하루 매수 최대 {rk.get('max_daily_buy_krw', 0):,.0f}원, 한 종목 최대 {rk.get('max_position_weight', 0) * 100:.0f}%."},
        {"rule": "데이터 신선도", "plain": f"점수 자료가 {rk.get('max_data_lag_days', 4)}일 넘게 낡으면 주문을 멈추고, 수급 자료가 {rk.get('stale_flow_days', 10)}거래일 넘게 비면 매수를 보류합니다."},
    ]
    return {"buy": buy, "sell": sell, "safety": safety}


def plain_check(c):
    """조건 점검 한 줄을 사람이 읽는 문장으로."""
    lab, ok, v = c["label"], c["ok"], str(c.get("value", ""))
    if lab.startswith("종합순위"):
        return f"순위 {v} — " + ("매수 후보 범위 안" if ok else "아직 매수 후보 범위 밖")
    if lab.startswith("시장"):
        return ("시장 게이트 열림 — " if ok else "시장 게이트 닫힘 — ") + v
    if "수급" in lab or "외국인" in lab:
        if "낡아" in v:
            return "수급 자료가 오래돼 외국인 매매를 확인할 수 없어 보류"
        return f"외국인 20일 순매수 {v} — " + ("팔고 있지 않음" if ok else "팔고 있음")
    if "공시" in lab:
        return ("악재 공시 없음" if ok else f"악재 공시: {v}")
    if lab.startswith("손절"):
        return f"산 가격 대비 {v} — " + ("손절선 위" if ok else "손절선 아래")
    if lab.startswith("트레일"):
        return f"보유 중 최고가 대비 {v} — " + ("트레일 폭 안" if ok else "트레일 폭을 넘음")
    if lab.startswith("순위"):
        return f"지금 {v} — " + ("보유 유지 조건 충족" if ok else "보유 유지 조건 벗어남")
    if lab == "빈 자리":
        return f"보유 칸이 꽉 참 ({v})"
    return f"{lab}: {v}"


def decision_line(d, st):
    s, r = d["state"], d.get("rank")
    if s == "매수":
        return f"{r}위이고 조건을 모두 통과해 내일 아침 삽니다. 산 뒤 −{st['stop_loss'] * 100:.0f}%면 손절합니다."
    if s == "매도":
        return f"팝니다 — {d.get('reason', '')}"
    if s == "유지":
        g = d.get("guard_price")
        return (f"{d.get('held_days', 0)}일째 보유 중 ({pct(d.get('ret'))}). "
                + (f"{won(g)} 아래로 내려가면 팝니다." if g else ""))
    fails = [c for c in d.get("checks", []) if not c["ok"]]
    labs = [c["label"] for c in fails]
    if any(x.startswith("종합순위") for x in labs):
        return f"{r}위 — 매수 후보(상위 {st['entry_rank']}위) 밖이라 지켜만 봅니다."
    parts = []
    if any(x.startswith("시장") for x in labs):
        parts.append("시장이 급락장이라")
    if any("수급" in x or "외국인" in x for x in labs):
        parts.append("수급 자료가 오래돼" if any("낡아" in str(c.get("value")) for c in fails) else "외국인이 팔고 있어")
    if any("공시" in x for x in labs):
        parts.append("악재 공시가 있어")
    if any(x == "빈 자리" for x in labs):
        parts.append("보유 칸이 꽉 차")
    return f"{r}위로 순위는 충분하지만 " + ", ".join(parts) + " 기다립니다." if parts else f"{r}위 — 기다립니다."


def engine_section(cfg, A, Dd, BT):
    st = cfg["strategy"]
    flags = (A or {}).get("flags") or {}
    items = {i["code"]: i for i in (Dd or {}).get("items", [])}
    weights, wsrc = weights_used(cfg, Dd)
    decs = (A or {}).get("decisions", [])
    gate = (A or {}).get("gate") or {}
    # 깔때기: 100종목 → 관심 → 매수 후보 → 각 조건 통과 → 매수
    non_held = [d for d in decs if d["state"] in ("매수", "관망")]
    stage = [{"label": "점수를 매긴 종목", "n": len(items) or 100}]
    stage.append({"label": f"관심 목록 (상위 {st.get('watch_rank', 20)}위, 보유 제외)", "n": len(non_held)})
    pool = [d for d in non_held if d.get("rank") and d["rank"] <= st["entry_rank"]]
    stage.append({"label": f"매수 후보 (상위 {st['entry_rank']}위)", "n": len(pool)})
    for key, lab in (("시장", "시장 게이트 통과"), ("외국인", "외국인 수급 통과"), ("공시", "악재 공시 없음")):
        pool = [d for d in pool if all(c["ok"] for c in d.get("checks", []) if (key in c["label"]) or (key == "외국인" and "수급" in c["label"]))]
        stage.append({"label": lab, "n": len(pool)})
    stage.append({"label": "실제 매수 (빈 자리만큼)", "n": sum(1 for d in decs if d["state"] == "매수")})
    # 오늘 종목별
    today = []
    for d in decs:
        it = items.get(d["code"], {})
        fac = it.get("factors") or []
        tot = sum(f.get("contrib", 0) for f in fac) or None
        today.append({
            "code": d["code"], "name": d["name"], "rank": d.get("rank"), "state": d["state"], "sector": d.get("sector"),
            "price": d.get("price"), "line": decision_line(d, st),
            "checks": [{"label": c["label"], "ok": c["ok"], "tier": c.get("tier"),
                        "plain": plain_check(c) + (f" (마지막 수급 {flags.get('flow_last')} · {flags.get('flow_gap')}거래일 비어 있음"
                                                   + (f" · 원인: {flags['collect_problem']}" if flags.get("collect_problem") else "") + ")"
                                                   if "낡아" in str(c.get("value", "")) else "")}
                       for c in d.get("checks", [])],
            "need": [x.replace("수급 데이터 갱신 (collect_flows)", "수급 자료가 다시 채워짐 (매일 오후 5시 자동 수집)") for x in (d.get("need") or [])],
            "score": it.get("score"),
            "contrib": [{"name": f["name"], "val": f["val"], "w": f["w"], "pts": f["contrib"],
                         "share": round(f["contrib"] / tot, 3) if tot else None} for f in fac],
            "ref": {"good": (d.get("ref") or {}).get("good"), "bad": (d.get("ref") or {}).get("bad"),
                    "items": [{"label": x["label"], "value": x["value"], "good": x["good"]} for x in (d.get("ref") or {}).get("items", [])]},
            "hold": {k: d.get(k) for k in ("held_days", "ret", "avg", "guard_price", "trail_width", "peak")} if d["state"] in ("유지", "매도") else None,
        })
    # 같은 규칙으로 과거에 사고판 예시 (백테스트 최근 거래)
    ex = []
    tl = (BT or {}).get("trade_list") or []
    if tl:
        sc = pd.read_csv("data/scores.csv", dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종합점수"])
        sc["종목코드"] = sc["종목코드"].str.zfill(6)
        sc["rk"] = sc.groupby("날짜")["종합점수"].rank(ascending=False, method="first")
        dates = sorted(sc["날짜"].unique())
        rk = sc.set_index(["날짜", "종목코드"])["rk"]
        for t in sorted(tl, key=lambda x: x["out"], reverse=True)[:10]:
            i = int(np.searchsorted(dates, t["in"]))
            dprev = dates[i - 1] if i > 0 else t["in"]
            r_in = rk.get((dprev, t["code"]))
            why = next((v for k, v in WHY.items() if t["why"].startswith(k)), None) or (WHY["부분익절"] if t["why"].startswith("+") else t["why"])
            ex.append({"name": t["name"], "code": t["code"], "in": t["in"], "out": t["out"], "ret": t["ret"], "held": t["held"],
                       "mae": t.get("mae"), "mfe": t.get("mfe"), "why_code": t["why"],
                       "story": (f"{t['in']}에 샀습니다 — 전날 종합 {int(r_in) if r_in == r_in and r_in is not None else '?'}위로 매수 후보였고 다른 조건도 통과. "
                                 f"{t['out']}에 팔았습니다 — {why}. {t['held']}거래일 동안 들고 {t['ret']:+.1f}% "
                                 f"(파는 날 전까지 최고 {t.get('mfe', 0):+.1f}%, 최저 {t.get('mae', 0):+.1f}%).")})
    def story(t, rk_lookup, dates):
        i = int(np.searchsorted(dates, t["in"]))
        r_in = rk_lookup.get((dates[i - 1] if i > 0 else t["in"], t["code"]))
        why = next((v for k, v in WHY.items() if t["why"].startswith(k)), None) or (WHY["부분익절"] if t["why"].startswith("+") else t["why"])
        return {"name": t["name"], "code": t["code"], "in": t["in"], "out": t["out"], "ret": t["ret"], "held": t["held"],
                "mae": t.get("mae"), "mfe": t.get("mfe"), "why_code": t["why"],
                "story": (f"{t['in']}에 샀습니다 — 전날 종합 {int(r_in) if r_in is not None and r_in == r_in else '?'}위로 매수 후보였고 다른 조건도 통과. "
                          f"{t['out']}에 팔았습니다 — {why}. {t['held']}거래일 동안 들고 {t['ret']:+.1f}% "
                          f"(파는 날 전까지 최고 {t.get('mfe', 0):+.1f}%, 최저 {t.get('mae', 0):+.1f}%).")}
    best = worst = []
    if tl:
        def pick(ts):
            seen, out = set(), []
            for t in ts:                                       # 부분 익절로 나뉜 같은 매수는 한 번만
                if (t["code"], t["in"]) not in seen:
                    seen.add((t["code"], t["in"]))
                    out.append(story(t, rk, dates))
                if len(out) == 3:
                    break
            return out
        best = pick(sorted(tl, key=lambda x: -x["ret"]))
        worst = pick(sorted(tl, key=lambda x: x["ret"]))
    tr = (BT or {}).get("trades") or {}
    stats = None
    if tr:
        ex_by = {x["why"]: x for x in tr.get("exits", [])}
        stats = {"n": tr.get("n"), "win": tr.get("win"), "avg": tr.get("avg"), "avg_win": tr.get("avg_win"), "avg_loss": tr.get("avg_loss"),
                 "avg_hold": tr.get("avg_hold"), "period": (BT or {}).get("period"),
                 "exits": [{"why": k, "n": v["n"], "avg": v["avg"], "win": v["win"], "plain": WHY.get(k, k)} for k, v in ex_by.items()],
                 "plain": (f"이 규칙을 {(BT or {}).get('period', ['', ''])[0]}~{(BT or {}).get('period', ['', ''])[1]}에 그대로 적용했다면 {tr.get('n')}번 사고팔았고, "
                           f"그중 {tr.get('win')}%가 이익이었습니다. 이익 난 거래는 평균 {tr.get('avg_win'):+.1f}%, 손해 본 거래는 평균 {tr.get('avg_loss'):+.1f}% — "
                           f"이기는 횟수보다 '이길 때 크게, 질 때 작게'로 수익을 냅니다. 평균 {tr.get('avg_hold')}거래일 보유.")}
    holds = [x for x in today if x["state"] in ("유지", "매도")]
    buys = [x for x in today if x["state"] == "매수"]
    sells = [x for x in today if x["state"] == "매도"]
    if buys or sells:
        head = f"내일 아침 {len(buys)}종목 사고 {len(sells)}종목 팝니다." if sells else f"내일 아침 {len(buys)}종목을 삽니다."
    elif not gate.get("open", True):
        head = "새로 사지 않고 기다립니다 — 시장이 급락장이라 바닥 확인을 기다리는 중입니다."
    else:
        head = "오늘은 사고팔 종목이 없습니다."
    eq = (A or {}).get("equity")
    data_issue = None
    if flags.get("collect_problem"):
        data_issue = {"problem": flags["collect_problem"], "fix": flags.get("collect_fix"), "time": flags.get("collect_time")}
    elif flags.get("flow_stale"):
        data_issue = {"problem": f"외국인·기관 수급 자료가 {flags.get('flow_gap')}거래일 비어 있습니다 (마지막 {flags.get('flow_last')})",
                      "fix": "매일 수집이 다시 채웁니다. 며칠째 그대로면 KRX 로그인 상태를 확인하세요.", "time": None}
    return {"headline": head, "equity": eq, "cash": (A or {}).get("cash"), "gate": gate, "data_issue": data_issue,
            "weights": weights, "weights_src": wsrc, "funnel": stage, "today": today,
            "n_hold": len(holds), "rules": rulebook(cfg), "examples": ex, "best": best, "worst": worst, "stats": stats,
            "journal": (A or {}).get("journal", [])[:20]}


# ══════════════════════════════════════════════════════════════
# 단기 모의투자
# ══════════════════════════════════════════════════════════════
ETF_PLAIN = {
    "hold": ("KODEX 200을 계속 들고 있습니다.", []),
    "overnight": ("매일 장 마감 때 KODEX 200을 사서 다음 날 아침 시가에 팝니다 (밤사이만 보유).", []),
    "overnight_trend": ("100종목 지수가 120일 평균보다 위에 있으면, 장 마감 때 KODEX 200을 사서 다음 날 아침에 팝니다.", ["m_trend120"]),
    "overnight_rev": ("전날 100종목 지수가 내렸으면, 장 마감 때 KODEX 200을 사서 다음 날 아침에 팝니다.", ["m_rev1"]),
    "overnight_for": ("전날 외국인이 (100종목 합계로) 순매수했으면, 장 마감 때 KODEX 200을 사서 다음 날 아침에 팝니다.", ["m_for1"]),
    "day_rev": ("전날 100종목 지수가 내렸으면 아침 시가에 KODEX 200을 사서 그날 장 마감에 팝니다.", ["m_rev1"]),
    "oo_rev": ("매일 저녁 그날 100종목 지수를 봅니다. 내렸으면 다음 날 아침 시가에 KODEX 200을 사고(이미 있으면 계속 보유), "
               "올랐으면 다음 날 아침 시가에 팔고 현금으로 기다립니다. 하락이 며칠 이어지면 그만큼 오래 들고 있게 됩니다.", ["m_rev1"]),
    "oo_revid": ("그날 장중(시가→종가)에 지수가 내렸으면 다음 날 아침 KODEX 200을 사거나 계속 들고, 아니면 아침에 팔고 현금으로 있습니다.", ["m_rev_id"]),
    "ls_rev": ("그날 지수가 내렸으면 KODEX 200을, 올랐으면 KODEX 인버스를 다음 날 아침 시가에 들고 있게 바꿉니다.", ["m_rev1"]),
    "trend_hold": ("100종목 지수가 120일 평균보다 위면 KODEX 200을 들고, 아래면 현금으로 있습니다.", ["m_trend120"]),
    "trend_overnight": ("지수가 120일 평균보다 위면 계속 보유하고, 아래면 밤사이만 보유합니다.", ["m_trend120"]),
}


def short_section(cfg, S, SX):
    if not S:
        return None
    import short_lab as L
    import short_panel as SP
    import short_evolve as SE
    D = L.Data()
    T = D.T - 1
    models = L.stock_models() + L.etf_models() + SE.models_from_registry()
    mid = S["model"]["id"]
    m = next((x for x in models if x["id"] == mid), None)
    if m is None:
        return None
    rule = S.get("rule") or {}
    board = S.get("leaderboard") or []
    rival = [b for b in board if not b.get("chosen")][:2]
    why = (f"{S.get('rule_name', '')} 규칙으로 골랐습니다. 후보 {S.get('pool_size')}개 중 최근 250거래일 위험 대비 성적(샤프) {S['model'].get('sh250')}로 1등"
           + (" — 2등 " + ", ".join(f"{b['desc']} {b['sh250']}" for b in rival) if rival else "") + "."
           + f" 다음 재선택까지 {S.get('next_reselect_days')}거래일 — 그때까지는 1등이 바뀌어도 그대로 갑니다(잦은 교체가 손해였기 때문).")
    out = {"model": {"id": mid, "desc": S["model"]["desc"], "selected_at": S["model"].get("selected_at"), "why": why,
                     "rule": rule, "rival": rival},
           "orders": [], "holdings": [], "checks": [], "signals": [], "examples": []}
    plan = S.get("plan") or {}
    dt = D.dates
    if m["kind"] == "etf":
        txt, keys = ETF_PLAIN.get(m["rule"], (m["rule"], []))
        out["model"]["plain"] = txt
        M = D.M
        lab = {"m_rev1": ("오늘 100종목 지수 등락", lambda v: -v, "내렸으면 내일 보유, 올랐으면 내일 현금"),
               "m_rev_id": ("오늘 장중 지수 등락(시가→종가)", lambda v: -v, "내렸으면 내일 보유"),
               "m_trend120": ("지수의 120일 평균 대비", lambda v: v, "위(+)면 보유"),
               "m_for1": ("오늘 외국인 순매수 (시총 대비)", lambda v: v, "샀으면(+) 보유")}
        for k in keys:
            name, f, rule_txt = lab[k]
            v = float(M[k][T]) if np.isfinite(M[k][T]) else None
            shown = f(v) if v is not None else None
            if k == "m_rev1" or k == "m_rev_id":
                ok = v is not None and v > 0
                val = pct(shown, 2)
            elif k == "m_trend120":
                ok = v is not None and v > 0
                val = pct(shown, 1)
            else:
                ok = v is not None and v > 0
                val = f"{shown * 1e4:+.1f}bp" if shown is not None else "—"
            held = bool((S.get("account") or {}).get("positions"))
            act = ("내일도 계속 보유" if held else "내일 아침 매수") if ok else ("내일 아침 매도" if held else "내일은 사지 않음")
            out["checks"].append({"label": name, "value": val, "ok": ok, "plain": f"{dt[T].date()} 값 {val} → 규칙: {rule_txt} → {act}"})
        log = []
        L.run_etf(D, m, log=log)
        import allweather as AW
        nm = [AW.NAME[c] for c in AW.CODES]
        pend = {}
        for e in log:
            if e["side"] == "buy":
                pend[e["j"]] = e
            elif e["j"] in pend:
                b = pend.pop(e["j"])
                r = e["px"] / b["px"] - 1
                out["examples"].append({"name": nm[e["j"]], "in": str(dt[b["t"]].date()), "out": str(dt[e["t"]].date()), "ret": round(r * 100, 2),
                                        "story": f"{dt[b['t']].date()} {'아침' if b['session'] == 'open' else '장 마감'}에 {nm[e['j']]} {b['px']:,.0f}원에 사서 "
                                                 f"{dt[e['t']].date()} {'아침' if e['session'] == 'open' else '장 마감'}에 {e['px']:,.0f}원에 팔았습니다 → {r * 100:+.2f}% (비용 전)"})
        out["examples"] = out["examples"][-10:][::-1]
        out["signals"] = [{"name": lab[k][0], "share": round(1 / len(keys), 3), "what": lab[k][2]} for k in keys]
    else:
        combo = m.get("combo") or L.SELECTORS[m["sel"]]
        totw = sum(w for _, w in combo) or 1
        K, X, mn, mx = L.stock_params(m)
        sig = " · ".join(f"{SP.SIGNAL_INFO.get(k, k)} {w / totw * 100:.0f}%" for k, w in combo)
        out["model"]["plain"] = (f"매일 장 마감 뒤 시가총액 상위 100종목을 [{sig}] 기준으로 줄 세워 상위 {K}종목을 다음 날 아침 시가에 삽니다. "
                                 f"산 뒤 {mn}거래일이 지나면, 순위가 {X}위 밖으로 밀렸으면 아침 시가에 팝니다 "
                                 f"(최대 {mx}거래일 보유 후에도 상위 {K}위 안이면 계속 보유). "
                                 + {"none": "", "trend": "지수가 120일 평균 아래면 새로 사지 않습니다.", "crash": "급락장(고점 −15% 이하이면서 반등 전)에는 새로 사지 않습니다."}[m["gate"]])
        out["signals"] = [{"name": SP.SIGNAL_INFO.get(k, k), "share": round(w / totw, 3), "what": SP.SIGNAL_INFO.get(k, k)} for k, w in combo]
        R = D.ranks(m["sel"], m.get("combo"))
        ok_gate = bool(D.gate[m["gate"]][T])
        out["checks"].append({"label": "시장 필터", "value": L.GATES[m["gate"]], "ok": ok_gate,
                              "plain": "새로 살 수 있는 시장" if ok_gate else "지금은 새로 사지 않는 시장 (보유분은 규칙대로 관리)"})
        top = np.argsort(R[T])[:K]
        for j in top:
            if not np.isfinite(R[T, j]):
                continue
            vals = []
            for k, w in combo:
                v = D.S[k][T, j]
                pr = D.pr(k)[T, j]
                vals.append(f"{SP.SIGNAL_INFO.get(k, k)} 상위 {max(1, round((1 - pr) * 100))}%")
            out["checks"].append({"label": f"오늘 {int(R[T, j])}위 {D.P['codes'][j]}", "value": " · ".join(vals), "ok": True,
                                  "plain": f"{int(R[T, j])}위 — " + " · ".join(vals)})
        log = []
        L.run_stock(D, m, log=log)
        names = {}
        try:
            p = pd.read_csv("data/prices.csv", dtype={"종목코드": str}, usecols=["종목코드", "종목명"]).drop_duplicates("종목코드", keep="last")
            names = dict(zip(p["종목코드"].str.zfill(6), p["종목명"]))
        except (OSError, ValueError):
            pass
        for c in out["checks"]:
            if c["label"].startswith("오늘 "):
                code = c["label"].split()[-1]
                c["label"] = c["label"].replace(code, names.get(code, code))
        pend = {}
        for e in log:
            if e["side"] == "buy":
                pend[e["j"]] = e
            elif e["j"] in pend:
                b = pend.pop(e["j"])
                r = e["px"] / b["px"] - 1
                nm = names.get(D.P["codes"][e["j"]], D.P["codes"][e["j"]])
                why_s = (f"순위가 {e['rank'] or '권외'}위로 {X}위 밖에 밀려서" if e["why"] == "rank" else f"최대 보유 {mx}일이 지났고 상위 {K}위 밖이라")
                out["examples"].append({"name": nm, "in": str(dt[b["t"]].date()), "out": str(dt[e["t"]].date()), "ret": round(r * 100, 2),
                                        "story": f"{dt[b['t']].date()} 아침 {nm}을(를) 샀습니다 — 전날 {b['rank']}위. "
                                                 f"{dt[e['t']].date()} 아침 팔았습니다 — {why_s}. {e['t'] - b['t']}거래일 {r * 100:+.2f}% (비용 전)."})
        out["examples"] = out["examples"][-10:][::-1]
    for o in plan.get("orders", []):
        who = "[전천후] " if o.get("sleeve") == "aw" else "[단기] "
        out["orders"].append({**o, "plain": who + f"{'아침 시가' if o['session'] == 'open' else '장 마감'}에 {o['name']} {o['qty']:,}주 "
                                            f"{'매수' if o['side'] == 'buy' else '매도'} (어제 종가 {o['est_price']:,.0f}원 기준 약 {o['qty'] * o['est_price']:,.0f}원)"})
    out["notes"] = plan.get("notes", [])
    acc = S.get("account") or {}
    for p in acc.get("positions", []):
        r = (p["last"] / p["avg"] - 1) if p.get("last") and p.get("avg") else None
        if p.get("sleeve") == "aw":
            txt = f"[전천후] {p['since']}에 {p['avg']:,.0f}원에 샀고 지금 {pct(r)}. 목표 비중과 5%p 넘게 벌어지거나 카드가 바뀔 때만 사고팝니다."
        else:
            txt = f"[단기] {p['since']}에 {p['avg']:,.0f}원에 샀고 지금 {pct(r)}. 위 모델 규칙대로 팔 때가 되면 아침 시가에 팝니다."
        out["holdings"].append({**p, "ret": r, "plain": txt})
    aw = S.get("aw")
    if aw:
        rg = aw["regime"]
        sl = S.get("sleeves") or {}
        out["aw"] = {"regime": rg, "share": (sl.get("aw") or {}).get("share"), "equity": (sl.get("aw") or {}).get("equity"),
                     "live_equity": (sl.get("live") or {}).get("equity"),
                     "plain": (f"계좌를 둘로 나눠 굴립니다. 절반은 위의 단기 갈아타기 모델, 절반은 '전천후 ETF 혼합'입니다. "
                               f"전천후 몫은 성격이 다른 전략 카드 {len(aw['mix'])}장(지수·미국주식·금·달러·채권 보유, 추세·모멘텀·위험균형·급락반등·하락대비)을 "
                               f"변동성 반비례로 섞어 매달 첫 거래일에 비중을 다시 맞춥니다. 분기마다 두 몫을 다시 반반으로 맞춥니다."),
                     "regime_plain": (f"지금 시장은 '{rg['name']}'입니다 — {rg['desc']} (1년 고점 대비 {pct(rg['dd'])}, 200일선 대비 {pct(rg.get('trend200'))}, "
                                      f"최근 60일 {pct(rg.get('r60'))}). 국면은 설명용입니다: 10년 자료에서 국면마다 1등 전략으로 갈아타는 방식은 지수 보유보다 못했습니다."),
                     "target": aw["target_named"][:8], "cards": aw["mix"][:6], "backtest": aw.get("backtest")}
    if plan.get("orders"):
        head = f"내일 {len(plan['orders'])}건 주문합니다."
    elif acc.get("positions"):
        head = "보유를 그대로 둡니다."
    else:
        head = "현금으로 기다립니다 — 지금 모델의 매수 조건이 맞지 않습니다."
    out["headline"] = head
    out["account"] = {k: acc.get(k) for k in ("equity", "ret", "bench_ret", "cash")}
    if SX:
        out["evolution"] = {"generations": SX.get("generations"), "candidates": len(SX.get("candidates", [])),
                            "incubate": SX.get("incubate_days"), "updated": SX.get("updated")}
    return out


# ══════════════════════════════════════════════════════════════
# 무엇을 얼마나 참고하나
# ══════════════════════════════════════════════════════════════
def info_table(cfg, eng, sh):
    st = cfg["strategy"]
    direct = [{"what": f"{w['name']} 점수", "how": w["what"], "weight": f"{w['w'] * 100:.0f}%", "role": "순위 계산"} for w in eng["weights"]]
    direct += [
        {"what": "종합 순위", "how": f"{st['entry_rank']}위 안 → 매수 후보, {st['exit_rank']}위 밖 + {st['min_hold_days']}일 → 매도", "weight": "기준선", "role": "매수·매도 결정"},
        {"what": "시장 국면 (급락장 여부·반등)", "how": "급락장이면 반등 확인 전 신규매수 중단",
         "weight": "지금 열림" if (eng.get("gate") or {}).get("open") else "지금 닫힘", "role": "매수 결정 (통과/막음)"},
        {"what": "외국인 20일 수급", "how": "순매도 중이면 매수 보류", "weight": "사용 중" if st.get("flow_filter", True) else "꺼짐", "role": "매수 결정 (통과/막음)"},
        {"what": "내 매수가 대비 손익", "how": f"−{st['stop_loss'] * 100:.0f}% 손절, +{(st.get('take_profit_gain') or 0) * 100:.0f}% 부분 익절", "weight": "기준선", "role": "매도 결정"},
        {"what": "보유 중 최고가 대비 하락", "how": f"트레일링 스톱 −{st['trail_base'] * 100:.0f}% (많이 오르면 −5%·−3%로 좁힘)", "weight": "기준선", "role": "매도 결정"},
    ]
    safety = [
        {"what": "DART 공시·뉴스 제목", "how": "유상증자·전환사채·횡령 등이면 매수 거부, 상장폐지·관리종목 등이면 즉시 매도", "weight": "거부권", "role": "안전장치"},
        {"what": "데이터 신선도", "how": "자료가 낡으면 주문 중단·매수 보류", "weight": "거부권", "role": "안전장치"},
        {"what": "계좌 낙폭·주문 한도", "how": "계좌 −30%면 신규매수 중단, 1회·하루 금액 한도", "weight": "거부권", "role": "안전장치"},
    ]
    ref = [
        {"what": "참고 지표 10개 (추세정렬·52주 고가·RSI·20일 이격·F-score·이익 서프라이즈·영업이익 증가·섹터 가치·섹터 모멘텀·섹터 이익)",
         "how": "종목마다 좋음/나쁨으로 보여 주기만", "weight": "0%", "role": "참고만 — 매수 조건에 더해도 과거 성과가 나아지지 않았음 (전략서 3-3·3-4)"},
        {"what": "판단 등급 A~D · 확률 · 시나리오", "how": "종목 화면에 요약으로", "weight": "0%", "role": "참고만 — 등급을 무시해도 결과가 같았음"},
        {"what": "뉴스 감성 점수", "how": "대부분 이미 일어난 주가 변동을 묘사", "weight": "0%", "role": "참고만"},
        {"what": "유사 차트", "how": "검증에서 예측력 없음", "weight": "0%", "role": "꺼짐"},
    ]
    short = []
    if sh:
        for s in sh.get("signals", []):
            short.append({"what": s["name"], "how": s["what"], "weight": f"{s['share'] * 100:.0f}%", "role": "단기 모델의 매수 순위/조건"})
        short.append({"what": "후보 모델들의 최근 250거래일 샤프", "how": "주 1회, 비용을 뺀 위험 대비 성적 1등 모델을 따라감", "weight": "모델 선택",
                      "role": "어떤 단기 모델을 쓸지"})
        short.append({"what": "거래 비용", "how": "주식 왕복 약 0.34% (매도세 0.20% 포함) · ETF 왕복 약 0.06%", "weight": "모든 성적에서 차감", "role": "모델 평가"})
    return {"direct": direct, "safety": safety, "ref": ref, "short": short}


GLOSSARY = [
    ("종합점수", "가치·수익성·성장·수급·모멘텀 점수(각 0~100)를 정해진 비중으로 더한 값. 높을수록 순위가 앞섭니다."),
    ("순위", "100종목을 종합점수로 줄 세운 자리. 1위가 가장 좋음."),
    ("시장 게이트", "시장 전체가 무너지는 중에는 개별 종목이 좋아 보여도 새로 사지 않게 막는 문."),
    ("급락장", "100종목 지수가 최근 고점보다 15% 넘게 빠진 상태."),
    ("반등 확인", "지수가 최근 20일 최저점보다 5% 이상 오른 것. 바닥을 지났다는 최소한의 신호로 씁니다."),
    ("손절", "산 가격보다 정해 둔 만큼 떨어지면 더 큰 손실을 막으려고 파는 것."),
    ("트레일링 스톱", "산 뒤 가장 높았던 가격을 기억해 두고, 거기서 정해 둔 폭만큼 내려오면 파는 것. 오른 만큼은 지키고 추세가 꺾이면 나옵니다."),
    ("감시가", "손절가와 트레일 가격 중 높은 쪽. 이 아래로 내려가면 팝니다."),
    ("부분 익절", "크게 올랐을 때 일부를 먼저 팔아 이익을 확정하는 것."),
    ("최소 보유 기간", "순위가 잠깐 떨어졌다고 바로 팔지 않도록 정한 기간. 잦은 매매 비용을 줄입니다."),
    ("샤프", "수익을 출렁임(변동성)으로 나눈 값. 같은 수익이면 덜 출렁인 쪽이 높습니다. 1 이상이면 좋은 편."),
    ("최대낙폭", "가장 높았던 때부터 가장 많이 빠졌던 비율."),
    ("시가 동시호가", "아침 9시 장 시작 때 한 번에 정해지는 가격. 단기 모델은 여기서 사고팝니다."),
    ("모의 계좌", "실제 돈이 아닌 가상 계좌. 실제 가격·비용으로 기록합니다."),
    ("챔피언·도전자", "지금 쓰는 30일 모델(챔피언)과, 진화로 찾은 더 나아 보이는 모델(도전자). 도전자는 등록 뒤 새 자료로 이겨야 바뀝니다."),
    ("진화 후보", "단기 모델 진화가 찾은 새 모델. 등록 뒤 125거래일 성적이 쌓여야 고를 수 있습니다."),
]


def main():
    t0 = time.time()
    cfg = settings.load()
    A, Dd, BT = jload("docs/autotrade.json"), jload("docs/data.json"), jload("docs/backtest.json")
    S, SX, EV = jload("docs/short.json"), jload("docs/short_evo.json"), jload("docs/evolve.json")
    eng = engine_section(cfg, A, Dd, BT)
    try:
        sh = short_section(cfg, S, SX)
    except Exception as e:                                  # 단기 해설이 실패해도 30일 해설은 낸다
        print("단기 해설 실패:", type(e).__name__, e)
        sh = None
    rg = ((Dd or {}).get("regime") or {}).get("current") or {}
    g = eng["gate"]
    market = {"regime": rg.get("name") or g.get("regime"), "streak": rg.get("streak"), "dd": rg.get("dd"), "trend": rg.get("trend"),
              "breadth": rg.get("breadth"), "rebound": g.get("rebound"), "open": g.get("open"),
              "plain": (f"시장은 '{rg.get('name') or g.get('regime')}' {rg.get('streak') or ''}거래일째입니다. "
                        f"100종목 지수가 고점보다 {abs(rg.get('dd') or 0):.1f}% 낮고, 200일 평균보다 {rg.get('trend') or 0:+.1f}%, "
                        f"60일 평균 위에 있는 종목은 {rg.get('breadth') or 0:.0f}%뿐입니다. "
                        + ("새로 살 수 있는 시장입니다." if g.get("open") else
                           f"30일 자동매매는 지수가 최근 20일 최저점보다 5% 이상 반등할 때까지 새로 사지 않습니다 (지금 {pct(g.get('rebound'))}).") )}
    summary = [f"30일 자동매매: {eng['headline']}"]
    if eng.get("data_issue"):
        summary.insert(0, f"자료 문제: {eng['data_issue']['problem']} — {eng['data_issue']['fix']}")
    if sh:
        summary.append(f"단기 모의투자: {sh['headline']} (모델: {sh['model']['desc']})")
    if EV:
        cont = EV.get("continuous") or {}
        summary.append(f"학습·진화: 30일 모델 {cont.get('generations', '—')}세대 · 도전자 {len(EV.get('challengers', []))}개 시험 중"
                       + (f" · 단기 진화 후보 {len((SX or {}).get('candidates', []))}개" if SX else "") + " — 등록 뒤 새 자료로 이겨야 실제로 쓰입니다.")
    out = {"updated": time.strftime("%Y-%m-%d %H:%M"), "data_date": (A or {}).get("asof"), "summary": summary, "market": market,
           "engine": eng, "short": sh, "info": info_table(cfg, eng, sh), "glossary": [{"term": a, "plain": b} for a, b in GLOSSARY],
           "champion": {"id": (EV or {}).get("champion", {}).get("id"), "since": (EV or {}).get("champion", {}).get("since")}}
    json.dump(clean(out), open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"해설 → {OUT} ({time.time() - t0:.0f}s): " + " / ".join(summary))


if __name__ == "__main__":
    main()
