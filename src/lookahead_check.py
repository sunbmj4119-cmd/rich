"""
미래 정보 전수 검사 (freqtrade `lookahead-analysis` 와 같은 생각)

  방법: 자료를 t일까지 잘라서 다시 계산한 값과, 전체 자료로 계산한 값의 t일까지 부분이 똑같아야 한다.
        다르면 그 계산은 t일 뒤 자료를 몰래 쓰고 있다(미래 정보). 여러 날짜에서 잘라 본다.
  대상: 단기 종목 신호 전부(short_panel.signals) · 지수 신호(market_signals) · 시장 걸러내기(market) ·
        전천후 특징·국면(allweather.features/regimes) · 전천후 카드 21장 목표 비중(targets) · 지수 ETF 규칙 목표(etf_targets)
  한계: 분할 보정 가격은 전체 자료로 만든 뒤 자른다(수익률 기반 신호엔 영향 없음).

  python src/lookahead_check.py [잘라 볼 날 수=4] → data/lookahead.json · docs/lookahead.json
"""
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allweather as AW  # noqa: E402
import short_lab as L  # noqa: E402
import short_panel as SP  # noqa: E402

OUT = "data/lookahead.json"
UI = "docs/lookahead.json"


def truncate(P, t):
    """패널을 t행(포함)까지 자른다 — 첫 차원이 날짜 수인 배열·목록은 전부."""
    T = len(P["dates"])
    Q = {}
    for k, v in P.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (T,):
            Q[k] = v[:t + 1].copy()
        elif isinstance(v, pd.DatetimeIndex) and len(v) == T:
            Q[k] = v[:t + 1]
        elif isinstance(v, dict):
            Q[k] = {kk: (vv[:t + 1].copy() if isinstance(vv, np.ndarray) and vv.shape[:1] == (T,) else vv) for kk, vv in v.items()}
        else:
            Q[k] = v
    return Q


def same(a, b, tol=1e-9):
    """a(잘라서 계산) 와 b(전체로 계산한 것의 앞부분) 가 같은가 — NaN 위치까지. 반환: (같음, 다른 칸 수)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False, -1
    na, nb = np.isnan(a), np.isnan(b)
    bad = (na != nb) | (~na & ~nb & (np.abs(a - b) > tol * (1 + np.abs(b))))
    return not bad.any(), int(bad.sum())


def check_panel(P, cuts):
    rows = []
    full_S = SP.signals(P)
    legs_f = SP.index_legs(P)
    full_M = SP.market_signals(P, {"oc": legs_f[1], "cc": legs_f[2]})
    full_mk = SP.market(P)
    for t in cuts:
        Q = truncate(P, t)
        S = SP.signals(Q)
        for k in full_S:
            ok, n = same(S[k], full_S[k][:t + 1])
            rows.append({"what": f"종목 신호 {k}", "cut": str(P["dates"][t].date()), "ok": ok, "bad": n})
        lg = SP.index_legs(Q)
        M = SP.market_signals(Q, {"oc": lg[1], "cc": lg[2]})
        for k in full_M:
            ok, n = same(M[k], full_M[k][:t + 1])
            rows.append({"what": f"지수 신호 {k}", "cut": str(P["dates"][t].date()), "ok": ok, "bad": n})
        mk = SP.market(Q)
        for k, v in full_mk.items():
            if isinstance(v, np.ndarray) and v.shape[:1] == (len(P["dates"]),):
                ok, n = same(np.asarray(mk[k], float), np.asarray(v, float)[:t + 1])
                rows.append({"what": f"시장 걸러내기 {k}", "cut": str(P["dates"][t].date()), "ok": ok, "bad": n})
        D = types.SimpleNamespace(T=t + 1, M={k: v[:t + 1] for k, v in full_M.items()})
        Df = types.SimpleNamespace(T=len(P["dates"]), M=full_M)
        for rule in L.ETF_RULES:
            wo, wc = L.etf_targets(D, rule)
            wof, wcf = L.etf_targets(Df, rule)
            ok1, n1 = same(wo, wof[:t + 1])
            ok2, n2 = same(wc, wcf[:t + 1])
            rows.append({"what": f"지수 ETF 규칙 {rule}", "cut": str(P["dates"][t].date()), "ok": ok1 and ok2, "bad": n1 + n2})
    return rows


def check_aw(cuts_frac):
    dates, Op, Cp = AW.load_prices()
    F = AW.features(Cp, dates)
    reg = AW.regimes(F)
    W = {k: AW.targets(F, Cp, k) for k in AW.RULES}
    rows = []
    for f in cuts_frac:
        t = int(len(dates) * f)
        Ft = AW.features(Cp[:t + 1], dates[:t + 1])
        for k in ("ma200", "dd", "r60", "vol20"):
            if k in F:
                ok, n = same(Ft[k], F[k][:t + 1])
                rows.append({"what": f"전천후 특징 {k}", "cut": str(dates[t].date()), "ok": ok, "bad": n})
        ok, n = same(AW.regimes(Ft), reg[:t + 1])
        rows.append({"what": "전천후 국면", "cut": str(dates[t].date()), "ok": ok, "bad": n})
        for k in AW.RULES:
            ok, n = same(AW.targets(Ft, Cp[:t + 1], k), W[k][:t + 1])
            rows.append({"what": f"전천후 카드 {k}", "cut": str(dates[t].date()), "ok": ok, "bad": n})
    return rows


def main(n_cuts=4):
    t0 = time.time()
    P = SP.load()
    T = len(P["dates"])
    rng = np.random.default_rng(int(time.strftime("%j")))           # 날마다 다른 날짜에서 잘라 본다
    cuts = sorted(set([T - 2] + list(rng.integers(300, T - 2, max(n_cuts - 1, 0)))))
    rows = check_panel(P, cuts) + check_aw([0.5, 0.8, 0.999][:max(1, min(3, n_cuts))])
    bad = [r for r in rows if not r["ok"]]
    groups = {}
    for r in rows:
        g = groups.setdefault(r["what"], {"what": r["what"], "checks": 0, "fail": 0})
        g["checks"] += 1
        g["fail"] += 0 if r["ok"] else 1
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "cuts": [str(P["dates"][t].date()) for t in cuts], "n_checks": len(rows),
           "n_items": len(groups), "n_fail": len(bad), "fails": bad[:50], "items": sorted(groups.values(), key=lambda g: (-g["fail"], g["what"]))}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(res, open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"미래 정보 검사: 항목 {len(groups)}개 × 잘라 본 날 → {len(rows)}번 비교 · 실패 {len(bad)} ({time.time() - t0:.0f}s)")
    for r in bad[:20]:
        print("  ✕", r)
    return res


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
