"""
토스 공격 고배당 — ISA(ETF) 로는 못 하는 '고배당 개별 종목 + 추세 타이밍'을 작은 계좌로 매일 자동매매.

후보 (2026-11-14, 결과 보기 전에 적음 · div_lab 의 배당 복원 · 세후 15.4% · 매일 판단 흉내 그대로)
  T1  고배당 + 추세·눌림     점수 = 배당수익률 순위 + 추세·눌림 순위 · 3자리 · 상위 5% 안이면 삼 · 30% 밖이면 팜 · 최소 20일
  T2  고배당 + 추세 + 저변동  점수 = 배당 + 추세·눌림 + 낮은 변동성 (셋 같은 비중) · T1 규칙
  T3  고배당 + 추세·눌림     T1 점수 · 5자리
  참고 R1 고배당만 3자리 · R2 추세·눌림만 3자리

고르는 규칙: T1~T3 중 '세 시기(2003~2007 · 2008~2018 · 2019~) 모두 대형주 대비 초과 > 0' 이고
            '2019~ 20만 원 정수 주식으로 실제 산 비율 ≥ 60%' 인 것 가운데 2003~ 전체 샤프가 가장 높은 것.
            없으면 R1(고배당만 3자리).
실제 주문: 토스 실제 주문 잠금(준비도 '준비됨' · live_trading · setup.sh --live)이 풀리고, 이 전략의 모의 기록이
          60거래일 쌓이고, 사용자가 승인하면 그때 바꾼다. 그 전에는 매일 모의 기록만 (FORWARD_START 부터).

  python src/toss_div.py --research   → data/toss_div_lab.json (후보 비교 · 고른 것)
  python src/toss_div.py               → data/short/toss_div.json (매일 모의 기록 · 오늘 신호) + docs/toss.json (화면용 묶음)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LAB = "data/toss_div_lab.json"
OUT = "data/short/toss_div.json"
PAGE = "docs/toss.json"
FORWARD_START = "2026-11-16"
CASH0 = 200_000
COMPOSITE = {"r250x": 1, "r120": 1, "ma120": 1, "hi52": 1, "r60": 1, "r5": -1, "r1": -1, "ma20": -1, "clv5": -1, "clv1": -1, "r20": -1}
R3 = {"slots": 3, "q_in": 0.05, "q_out": 0.30, "min_hold": 20, "stop": 0.0, "max_hold": 0}
CANDS = {"T1": ("고배당 + 추세·눌림 · 3자리", ("y", "c"), R3),
         "T2": ("고배당 + 추세 + 저변동 · 3자리", ("y", "c", "v"), R3),
         "T3": ("고배당 + 추세·눌림 · 5자리", ("y", "c"), {**R3, "slots": 5}),
         "R1": ("참고: 고배당만 · 3자리", ("y",), R3),
         "R2": ("참고: 추세·눌림만 · 3자리", ("c",), R3)}
ERAS = {"2003~2007": ("2003-01-01", "2007-12-31"), "2008~2018": ("2008-01-01", "2018-12-31"), "2019~": ("2019-01-01", None), "전체 2003~": ("2003-01-01", None)}


def score_parts(E):
    """시대 E 의 순위 재료: y 배당수익률 · c 추세·눌림 · v 낮은 60일 변동성 (배당자료 있는 상위 100 안의 순위 −0.5~0.5)."""
    import div_lab as DL
    Uc = E.U & np.isfinite(E.ttm)
    vol = E.X[__import__("score_lab").FACTORS.index("vol60")]
    return {"y": DL.rank01(E.ttm, Uc), "c": DL.rank01(E.score(COMPOSITE), Uc), "v": DL.rank01(-vol, Uc)}


def build_scores(ER):
    parts = {e: score_parts(E) for e, E in ER.items()}
    return {k: {e: sum(parts[e][p] for p in spec[1]) for e in ER} for k, spec in CANDS.items()}


def eras_with_div():
    import div_lab as DL
    import score_lab as SC
    SC.set_factors(True)
    ER = SC.eras(lambda m: None)
    DL.attach_dividends(ER, lambda m: None)
    return ER


def research(log=print):
    import div_lab as DL
    import score_lab as SC
    t0 = time.time()
    ER = eras_with_div()
    S = build_scores(ER)
    res = {"cands": {}, "rule": "세 시기 모두 초과 > 0 · 2019~ 20만 원 산 비율 ≥ 60% · 그중 2003~ 샤프 최고 (없으면 R1)"}
    for k, (name, parts, rule) in CANDS.items():
        a = DL.run_one(ER, S[k], rule, True)
        p = DL.run_one(ER, S[k], rule, False)
        x2 = DL.run_one(ER, S[k], rule, True, cost_mult=2.0)
        row = {"name": name, "parts": list(parts), "rule": rule, "rule_text": SC.rule_label(rule),
               "trades_per_year": round(len(a["trades"]) / (len(a["r"]) / 252), 1)}
        for pn, (x0, x1) in ERAS.items():
            c = lambda s: SC.cut(s, x0, x1)                              # noqa: E731
            st = SC.stats_full(c(a["r"]), c(a["ew"]))
            sp = SC.stats_full(c(p["r"]))
            row[pn] = {**st, "price_cagr": sp["cagr"], "div_part": round(st["cagr"] - sp["cagr"], 4), "x2_cagr": SC.stats_full(c(x2["r"]))["cagr"]}
        acc = {}
        for c0 in (200_000, 1_000_000):
            q = DL.run_one(ER, S[k], rule, True, a="2019-01-01", cash0=c0)
            st = SC.stats_full(q["r"], q["ew"])
            acc[str(c0)] = {**st, "invested": round(q["invested"], 3), "final": round(float(c0 * (1 + q["r"]).prod()))}
        row["accounts_2019"] = acc
        res["cands"][k] = row
        f, h = row["전체 2003~"], row["2019~"]
        log(f"  {k} {name:24s}: 전체 연 {f['cagr']:+.1%} (배당 {f['div_part']:+.1%}) 샤프 {f['sharpe']:.2f} 낙폭 {f['mdd']:.0%} | "
            f"초과 {row['2003~2007']['excess_ann']:+.0%}/{row['2008~2018']['excess_ann']:+.0%}/{h['excess_ann']:+.0%} | 2019~ 연 {h['cagr']:+.1%} 샤프 {h['sharpe']:.2f} 낙폭 {h['mdd']:.0%} "
            f"· 비용2배 {h['x2_cagr']:+.1%} · 20만 원 연 {acc['200000']['cagr']:+.1%} (산 비율 {acc['200000']['invested']:.0%}) · 거래 {row['trades_per_year']:.0f}/년")
        SC._PREP.clear()
    ok = [k for k in ("T1", "T2", "T3") if all(res["cands"][k][pn]["excess_ann"] > 0 for pn in ("2003~2007", "2008~2018", "2019~"))
          and res["cands"][k]["accounts_2019"]["200000"]["invested"] >= 0.6]
    pick = max(ok, key=lambda k: res["cands"][k]["전체 2003~"]["sharpe"]) if ok else "R1"
    res["picked"] = pick
    res["passed"] = ok
    res["generated"] = time.strftime("%Y-%m-%d %H:%M")
    res["secs"] = round(time.time() - t0)
    json.dump(res, open(LAB, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    log(f"  기준 통과 {ok} → 고른 것 {pick} ({CANDS[pick][0]}) → {LAB}")
    return res


def picked():
    try:
        return json.load(open(LAB, encoding="utf-8"))["picked"]
    except (OSError, ValueError, KeyError):
        return "T1"


def forward(log=print):
    """매일 모의 기록 (실제 돈 아님): 고른 전략을 FORWARD_START 부터 20만 원 · 정수 주식으로 · 오늘 장 마감 신호."""
    import div_lab as DL
    import score_lab as SC
    import shadow_wf as SW
    k = picked()
    name, parts, rule = CANDS[k]
    SC.set_factors(True)
    E = SC.Era("B", lambda m: None)
    DL.attach_dividends({"B": E}, lambda m: None)
    P = score_parts(E)
    S = sum(P[p] for p in parts)
    nm = SW.names()
    last = E.dates[E.i1 - 1]
    out = {"strategy": k, "name": name, "rule": SC.rule_label(rule), "start": FORWARD_START, "data_until": str(last.date()),
           "what": "모의 기록 — 실제 돈 아님 (토스 공격 고배당)", "cash0": CASH0}
    held = []
    if last >= np.datetime64(FORWARD_START):
        lo, hi = E.rows(FORWARD_START, None)
        if min(hi, E.G.shape[0] - 2) - lo >= 1:
            saved = (E.G, E.ew)
            E.G, E.ew = E.G_div, E.ew_cov_div
            try:
                res = SC.simulate(E, S, rule, FORWARD_START, None, cash0=CASH0)
            finally:
                E.G, E.ew = saved
            eq = CASH0 * (1 + res["r"]).cumprod()
            out.update({"days": int(len(eq)), "equity": round(float(eq.iloc[-1])), "ret": round(float(eq.iloc[-1] / CASH0 - 1), 4),
                        "ew_ret": round(float((1 + res["ew"]).prod() - 1), 4), "mdd": round(float((eq / eq.cummax() - 1).min()), 4),
                        "invested": round(res["invested"], 3), "trades": int(len(res["trades"])),
                        "curve": {"date": [str(d.date()) for d in eq.index], "equity": [round(float(v)) for v in eq]}})
            held = list(res["pos"])
            out["holdings"] = [{"code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j]), "share": round(v["value"], 3), "days": v["days"]}
                               for j, v in sorted(res["pos"].items(), key=lambda x: -x[1]["value"])]
    if "days" not in out:
        out.update({"days": 0, "equity": CASH0, "ret": 0.0, "holdings": [], "note": "아직 시작 전"})
    t = E.i1 - 1
    PR, order = SC._prep(E, S)
    sell = [j for j in held if not np.isfinite(PR[t, j]) or PR[t, j] > rule["q_out"]]
    cand = [int(j) for j in order[t] if np.isfinite(PR[t, j]) and PR[t, j] <= rule["q_in"] and int(j) not in held]
    px = E.Cf[t]
    slot = out["equity"] / rule["slots"]                                   # 한 자리 금액 (정수 주식: 1주 값이 이보다 비싸면 못 삼)
    out["slot_won"] = round(slot)
    out["signal"] = {"date": str(E.dates[t].date()), "free_slots": rule["slots"] - len(held) + len(sell),
                     "sell": [{"code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j])} for j in sell],
                     "buy": [{"code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j]), "price": round(float(px[j])) if np.isfinite(px[j]) else None,
                              "yield": round(float(E.ttm[t, j]), 4) if np.isfinite(E.ttm[t, j]) else None,
                              "affordable": bool(np.isfinite(px[j]) and px[j] * 1.0007 <= slot)} for j in cand]}
    # 오늘 점수 상위 10 (배당수익률 함께)
    top = [int(j) for j in order[t] if np.isfinite(PR[t, j])][:10]
    out["top"] = [{"rank": i + 1, "code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j]), "yield": round(float(E.ttm[t, j]), 4) if np.isfinite(E.ttm[t, j]) else None,
                   "price": round(float(px[j])) if np.isfinite(px[j]) else None, "held": j in held} for i, j in enumerate(top)]
    out["generated"] = time.strftime("%Y-%m-%d %H:%M")
    log(f"  토스 공격 고배당({k}) 모의: {out['days']}일 · {out['equity']:,}원 ({out['ret']:+.1%}) · 내일 팔 것 {[x['name'] for x in out['signal']['sell']]} · "
        f"살 후보 {[x['name'] + ('' if x['affordable'] else '(1주가 한 자리보다 비쌈)') for x in out['signal']['buy']]}")
    SC._PREP.clear()
    return out


def _j(fn, default=None):
    try:
        return json.load(open(fn, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def page(fwd=None):
    """docs/toss.json — 화면용 묶음: 실제 주문(DM:2) 상태 · 준비도 · 공격 고배당 모의 기록 · 후보 비교 · 추세·눌림 도전자.
    잔고·계좌번호·금액(20만 원 모의 외)은 넣지 않는다."""
    lab = _j(LAB, {}) or {}
    rd = _j("data/readiness.json", {}) or {}
    dm = _j("data/short/size_forward.json", {}) or {}
    ts = _j("data/short/trend_shadow.json", {}) or {}
    cands = {}
    for k, v in (lab.get("cands") or {}).items():
        cands[k] = {"name": v["name"], "rule": v.get("rule_text"), "trades_per_year": v.get("trades_per_year"),
                    "eras": {pn: {x: v[pn].get(x) for x in ("cagr", "sharpe", "mdd", "excess_ann", "div_part", "x2_cagr")} for pn in ERAS if v.get(pn)},
                    "acc200k": v.get("accounts_2019", {}).get("200000"), "acc1m": v.get("accounts_2019", {}).get("1000000")}
    return {"what": "토스 종합계좌 — 시스템이 자동으로 (실제 주문은 잠금이 다 풀린 뒤에만)",
            "real": {"strategy": "DM:2", "desc": "매달 코스피200·미국 S&P500·나스닥·금·달러·장기채 ETF 중 120일 수익 상위 2개 반반 (단기채보다 못하면 단기채)",
                     "paper": {k: dm.get(k) for k in ("start", "days", "ret", "mdd")},
                     "locks": ["설정 live_trading: true", "서버 setup.sh --live", "준비도 '준비됨'"]},
            "readiness": {"verdict": rd.get("verdict"), "missing": rd.get("missing"),
                          "items": [{k: i.get(k) for k in ("name", "ok", "now", "need", "eta")} for i in (rd.get("items") or [])]},
            "aggr_div": {"picked": lab.get("picked"), "passed": lab.get("passed"), "rule": lab.get("rule"), "cands": cands,
                         "forward": fwd if fwd is not None else _j(OUT, {}),
                         "switch_rule": "토스 실제 주문을 이 전략으로 바꾸는 조건: 준비도 '준비됨' + 이 모의 기록 60거래일 이상 + 같은 기간 대형주 평균 이상 + 사용자 승인"},
            "trend": {k: ts.get(k) for k in ("start", "data_until", "note")} | {"rules": {rn: {x: r.get(x) for x in ("rule", "days", "ret", "ew_ret", "mdd", "invested", "signal", "promote_check")}
                                                                               for rn, r in (ts.get("rules") or {}).items()}},
            "generated": time.strftime("%Y-%m-%d %H:%M")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", action="store_true")
    ap.add_argument("--page", action="store_true", help="docs/toss.json 만 다시 묶기")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)                                  # noqa: E731
    if a.research:
        research(log)
        return
    res = None
    if not a.page:
        res = forward(log)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    json.dump(page(res), open(PAGE, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"→ {OUT} · {PAGE}")


if __name__ == "__main__":
    main()
