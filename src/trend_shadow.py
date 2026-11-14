"""
토스 계좌 도전자 그림자 기록 — '장기 추세 + 단기 눌림' 점수 (robust_lab 3차에서 찾은 것) 를 실제 돈 없이 앞으로 쌓는다.

  점수 = 1년(최근 20일 빼고)·120일·60일 수익, 120일선·52주 고점 근처 → +1 · 최근 1·5·20일 수익, 20일선 대비, 1·5일 종가 위치 → −1
         (그날 시총 상위 100 안의 순위, 가중치 조정 없음)
  규칙   매일 장 마감 판단 → 다음 날 시가 · 상위 10% 안이면 빈자리에 삼 · 상위 50% 밖으로 밀리면 팜 · 최소 20거래일 보유
         10자리(연구 그대로)와 5자리(20만 원 계좌용, 상위 5% 안이면 삼) 두 가지 · 20만 원 · 정수 주식 · 왕복 비용 약 0.34%

  시작일 FORWARD_START (2026-11-16) 부터 매번 처음부터 다시 계산 (상태 파일 없음 → 결과가 늘 같음). 실제 주문은 내지 않는다.

  승격 기준 (2026-11-14, 결과 보기 전에 적음) — 모두 만족하면 '토스 자금의 절반을 이 전략으로' 를 사용자에게 제안 (자동으로 바꾸지 않음)
    · 앞으로의 기록 120거래일 이상
    · 같은 기간 대형주 100 평균(비용 없음)보다 수익이 높음
    · 최대 낙폭이 −25% 보다 얕음
    · 같은 기간 DM:2 그림자(size_forward)보다 수익이 낮지 않음

  python src/trend_shadow.py → data/short/trend_shadow.json  (short_trader 가 매일 부름)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/short/trend_shadow.json"
FORWARD_START = "2026-11-16"
CASH0 = 200_000
COMPOSITE = {"r250x": 1, "r120": 1, "ma120": 1, "hi52": 1, "r60": 1, "r5": -1, "r1": -1, "ma20": -1, "clv5": -1, "clv1": -1, "r20": -1}
RULES = {"10자리": {"slots": 10, "q_in": 0.10, "q_out": 0.50, "min_hold": 20, "stop": 0.0, "max_hold": 0},
         "5자리": {"slots": 5, "q_in": 0.05, "q_out": 0.50, "min_hold": 20, "stop": 0.0, "max_hold": 0}}
PROMOTE = {"min_days": 120, "max_dd": -0.25}


def today_signal(E, S, rule, holdings, nm):
    """마지막 날 장 마감 점수로: 내일 시가에 팔 것(들고 있는데 상위 q_out 밖) · 살 수 있는 후보(상위 q_in 안, 높은 순)."""
    import score_lab as SC
    t = E.i1 - 1
    PR, order = SC._prep(E, S)
    sell = [j for j in holdings if not np.isfinite(PR[t, j]) or PR[t, j] > rule["q_out"]]
    buy = [int(j) for j in order[t] if np.isfinite(PR[t, j]) and PR[t, j] <= rule["q_in"] and int(j) not in holdings][:rule["slots"]]
    return t, sell, buy


def forward(log=print):
    import score_lab as SC
    import shadow_wf as SW
    SC.set_factors(True)
    E = SC.Era("B", lambda m: None)
    last = E.dates[E.i1 - 1]
    out = {"start": FORWARD_START, "data_until": str(last.date()), "what": "그림자 기록 — 실제 돈 아님 (토스 도전자: 장기 추세 + 단기 눌림)",
           "score": COMPOSITE, "promote_rule": PROMOTE, "rules": {}}
    if last < np.datetime64(FORWARD_START):
        out["note"] = "아직 시작 전"
    S = E.score(COMPOSITE)
    nm = SW.names()
    dm = None
    try:                                                   # 같은 시작일부터의 DM:2 그림자 (지금 토스 계획) — 같은 기간으로 비교
        import size_policy as SZ
        dm = SZ.forward({"paper_start": FORWARD_START, "paper_cash": CASH0})
    except Exception as ex:                                # noqa: BLE001
        log(f"  (DM:2 같은 기간 기록 실패: {type(ex).__name__} {ex})")
    out["dm2_same_window"] = dm
    for rn, rule in RULES.items():
        row = {"rule": SC.rule_label(rule)}
        if last >= np.datetime64(FORWARD_START):
            lo, hi = E.rows(FORWARD_START, None)
            if min(hi, E.G.shape[0] - 2) - lo >= 1:
                res = SC.simulate(E, S, rule, FORWARD_START, None, cash0=CASH0)
                eq = CASH0 * (1 + res["r"]).cumprod()
                ew = (1 + res["ew"]).cumprod()
                row.update({"days": int(len(eq)), "equity": round(float(eq.iloc[-1])), "ret": round(float(eq.iloc[-1] / CASH0 - 1), 4),
                            "ew_ret": round(float(ew.iloc[-1] - 1), 4), "mdd": round(float((eq / eq.cummax() - 1).min()), 4),
                            "invested": round(res["invested"], 3), "trades": int(len(res["trades"])),
                            "holdings": [{"code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j]), "share": round(v["value"], 3), "days": v["days"]}
                                         for j, v in sorted(res["pos"].items(), key=lambda x: -x[1]["value"])]})
                held = list(res["pos"])
            else:
                held = []
                row.update({"days": 0, "equity": CASH0, "ret": 0.0})
        else:
            held = []
            row.update({"days": 0, "equity": CASH0, "ret": 0.0})
        t, sell, buy = today_signal(E, S, rule, held, nm)
        row["signal"] = {"date": str(E.dates[t].date()), "sell": [{"code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j])} for j in sell],
                         "buy_candidates": [{"code": E.codes[j], "name": nm.get(E.codes[j], E.codes[j])} for j in buy]}
        dm_ret = (dm or {}).get("ret") if isinstance(dm, dict) else None
        ok = {"days": row.get("days", 0) >= PROMOTE["min_days"], "beats_ew": row.get("ret", 0) > row.get("ew_ret", 0) if row.get("days") else False,
              "dd": row.get("mdd", 0) > PROMOTE["max_dd"] if row.get("days") else False,
              "vs_dm2": (dm_ret is not None and row.get("ret", 0) >= dm_ret) if row.get("days") else False}
        row["promote_check"] = {**ok, "all": all(ok.values())}
        out["rules"][rn] = row
        log(f"  추세·눌림 그림자 {rn}: {row.get('days', 0)}일 · {row.get('equity', CASH0):,}원 ({row.get('ret', 0):+.1%}) · "
            f"내일 팔 것 {[x['name'] for x in row['signal']['sell']]} · 살 후보 {[x['name'] for x in row['signal']['buy_candidates'][:5]]}")
        SC._PREP.clear()
    out["generated"] = time.strftime("%Y-%m-%d %H:%M")
    return out


def main():
    res = forward(lambda m: print(m, flush=True))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
