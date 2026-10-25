"""
예측력 개선 시도 기록 — 후보마다 판단 엔진(decision_engine)을 그대로 돌려 기준선(BASE)과 비교한다.
-> data/prediction_lab.json, docs/prediction_lab.json (성과 보고서 화면)

왜 엔진으로 재나
  alpha_lab.py 의 순위상관(IC)은 100종목 전체를 맞히는 힘이다. 실제로 사는 건 상위 10종목이고
  손절·트레일·게이트가 붙는다. 그래서 '쓸모 있는 개선'은 엔진 성과로만 판정한다.

후보 (모두 미래 정보 없이 만든다)
  순위 교체   alpha_lab 의 워크포워드 예측(EQ·ICW·RIDGE·TOP_*)으로 종합순위를 바꾼다
  매수 제외   위험 신호(복권형 급등·고유변동성 등)가 최악 10%인 종목은 사지 않는다
  BuyFit     종합 20위 안에서 z(점수)+λ·눌림목 타이밍 순으로 산다 (λ=0.25 는 기존 값, 나머지는 민감도)
  변동성 관리 시장 20일 변동성이 평소(그때까지의 중앙값)보다 크면 비중을 줄인다 (Moreira-Muir 2017)
  가중치 재현 현재 가중치(2024년 이후 성과도 보고 고름)를 매달 '그때까지의 IC'로만 다시 정한다
             → 기준선이 사후 선택으로 얼마나 부풀었는지 재는 용도 (개선 후보가 아님)

판정 (미리 정한 규칙)
  채택 = 2021-02~2023 과 2024~ 두 구간 모두 Sharpe 개선
         AND 2021-02~ 전체에서 'Sharpe 차 ≤ 0' 부트스트랩 확률 < 0.05 / 시험 수 (Bonferroni)
  부트스트랩: 일별 수익 20일 블록, 4000회. 두 전략을 같은 블록으로 뽑아 상관을 유지한다.

사용: python src/alpha_lab.py   (순위 교체 후보용 예측 저장 — data/cache/)
      python src/prediction_lab.py
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _SRC]
import signal  # noqa: E402,F401  (src/signal.py 가 표준 라이브러리를 가리는 것 방지 — 병렬 실행용)
sys.path.append(_SRC)

import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import features as F  # noqa: E402
from engine_backtest import run, COST  # noqa: E402

OUT, OUT_UI = "data/prediction_lab.json", "docs/prediction_lab.json"
CACHE = "data/cache"
A0, SPLIT = "2021-02-10", "2024-01-01"       # 워크포워드 예측이 존재하는 첫날 · 검증 구간 시작
GROUPS = ["s_value", "s_profit", "s_grow", "s_flow", "s_rmom", "s_mom"]

CANDS = [
    # key, 분류, 설명
    ("EQ", "순위 교체", "과거 IC t>2 신호 동일가중"),
    ("ICW", "순위 교체", "과거 정보비율 가중 41개 신호"),
    ("RIDGE", "순위 교체", "릿지 회귀 41개 신호"),
    ("TOP_EQ", "순위 교체", "상위10 성적 좋은 신호 + 기존 점수 반반"),
    ("TOP_TW", "순위 교체", "상위10 t값 가중 + 기존 점수 반반"),
    ("TOP_RIDGE", "순위 교체", "릿지(기존 점수 포함)"),
    ("veto:max21", "매수 제외", "최근 한 달 하루 최대 급등 상위 10% 제외"),
    ("veto:ivol", "매수 제외", "고유변동성 상위 10% 제외"),
    ("veto:lowvol", "매수 제외", "60일 변동성 상위 10% 제외"),
    ("veto:lowbeta", "매수 제외", "베타 상위 10% 제외"),
    ("veto:inst60", "매수 제외", "기관 60일 순매도 하위 10% 제외"),
    ("veto:bp_sn", "매수 제외", "업종 내 PBR 최고 10% 제외"),
    ("veto:ep_sn", "매수 제외", "업종 내 PER 최고 10% 제외"),
    ("buyfit:0.25", "BuyFit 매수 순서", "λ=0.25 (기존 대시보드 값)"),
    ("volmgmt", "변동성 관리", "시장 변동성 ↑ → 비중 ↓"),
]
SENS = [("buyfit:0.1", "BuyFit 매수 순서", "λ=0.1 (민감도)"), ("buyfit:0.5", "BuyFit 매수 순서", "λ=0.5 (민감도)")]
HIND = [("wf:FIX6", "가중치 재현", "현재 고정 가중치 (6요소 재구성)"),
        ("wf:ICW", "가중치 재현", "매달 그때까지의 IC 비례 가중")]

C = {}


def _init():
    p, f = F.build_all()
    C.update(p=p, f=f, cfg=yaml.safe_load(open("config/autotrade.yaml", encoding="utf-8")))


def _z(M):
    return M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1) + 1e-9, axis=0)


def _signals():
    if "S" not in C:
        from alpha_lab import build_signals
        C["S"] = build_signals(C["p"], C["f"])
    return C["S"]


def _components():
    s = pd.read_csv(F.SCORES, dtype={"종목코드": str})
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s["날짜"] = pd.to_datetime(s["날짜"])
    p = C["p"]
    return {k: s.pivot_table(index="날짜", columns="종목코드", values=k).reindex(index=p["dates"], columns=p["codes"])
            for k in GROUPS}


def _wf_rank(mode):
    """6개 점수 요소의 가중치를 매달 그때까지 확정된 IC 로만 다시 정한 종합순위."""
    from alpha_lab import rank_ic, FWD, STEP, REFIT, MIN_TRAIN_DAYS
    p, f = C["p"], C["f"]
    M = _components()
    if mode == "FIX6":
        w = {"s_value": .29, "s_profit": .20, "s_grow": .16, "s_flow": .18, "s_rmom": .10, "s_mom": .07}
        sc = sum(w[k] * M[k].fillna(50) for k in GROUPS)
        return sc.where(p["score"].notna()).rank(axis=1, ascending=False, method="first"), None
    close = p["close"]
    fwd = (close.shift(-FWD) / close - 1).values
    nD = len(p["dates"])
    smp = np.array(range(260, nD - FWD, STEP))
    ic = {k: np.array([rank_ic(M[k].values[t], fwd[t]) for t in smp]) for k in GROUPS}
    sc = p["score"] * np.nan
    starts = list(range(260 + MIN_TRAIN_DAYS, nD, REFIT))
    log = {}
    for i, t0 in enumerate(starts):
        ok = smp <= t0 - FWD
        m = np.array([max(np.nanmean(ic[k][ok]), 0) for k in GROUPS])
        w = m / m.sum() if m.sum() > 0 else np.ones(len(GROUPS)) / len(GROUPS)
        t1 = starts[i + 1] if i + 1 < len(starts) else nD
        sc.iloc[t0:t1] = sum(w[j] * M[k].iloc[t0:t1].fillna(50) for j, k in enumerate(GROUPS)).values
        log[str(p["dates"][t0].date())] = [round(float(x), 2) for x in w]
    r = sc.where(p["score"].notna()).rank(axis=1, ascending=False, method="first").fillna(f["rank"])
    keys = sorted(log)
    return r, {"groups": GROUPS, "first": [keys[0], log[keys[0]]], "last": [keys[-1], log[keys[-1]]]}


def _feats(key):
    """후보 key → (엔진에 넣을 feats, 사후 오버레이 함수 또는 None, 부가정보)."""
    p, f = C["p"], C["f"]
    f2 = dict(f)
    if key == "BASE" or key == "volmgmt":
        return f2, (_volmgmt if key == "volmgmt" else None), None
    if key.startswith("veto:"):
        S = _signals()[key[5:]]
        bad = (S.rank(axis=1, pct=True) <= 0.10) & S.notna()
        fl = f["flow"].copy() if "flow" in f else pd.DataFrame(np.nan, index=p["dates"], columns=p["codes"])
        f2["flow"] = fl.mask(bad, -1.0)          # 엔진은 flow<0 이면 매수만 막는다 (보유·매도엔 영향 없음)
        return f2, None, None
    if key.startswith("buyfit:"):
        lam = float(key[7:])
        timing = 0.6 * (-_z(f["dd60"])) + 0.4 * (-_z(f["ret20"]))
        bf = (_z(p["score"]) + lam * timing.fillna(0)).where(f["rank"] <= C["cfg"]["strategy"]["watch_rank"])
        er = bf.rank(axis=1, ascending=False, method="first")
        # 풀 안(관망 20위 이내)은 BuyFit 순위 1~20, 풀 밖은 원래 순위(>20) → 이탈 판단은 기준선과 같다
        f2["rank"] = er.fillna(f["rank"])
        return f2, None, None
    if key.startswith("wf:"):
        f2["rank"], info = _wf_rank(key[3:])
        return f2, None, info
    P = pd.read_pickle(os.path.join(CACHE, f"alpha_{key}.pkl"))
    r = P.rank(axis=1, ascending=False, method="first")
    f2["rank"] = r.where(P.notna().any(axis=1), axis=0).fillna(f["rank"])
    return f2, None, None


def _volmgmt(r, d0):
    """일별 수익 r(d0 부터)에 시장 변동성 관리 비중을 곱한다. t 종가 정보 → t+1 조정 → t+2 수익부터."""
    idx = C["p"]["market"]["idx"].values
    mr = np.r_[np.nan, np.diff(idx) / idx[:-1]]
    sig = pd.Series(mr).rolling(20, min_periods=15).std().values
    med = pd.Series(sig).expanding(60).median().values
    w = np.clip(med / sig, 0, 1.0)
    w = np.where(np.isfinite(w), w, 1.0)
    wl = np.r_[1.0, 1.0, w[:-2]][d0:d0 + len(r)]
    return wl * r - np.abs(np.r_[0, np.diff(wl)]) * COST


def _one(key):
    p, cfg = C["p"], C["cfg"]
    d = p["dates"]
    a = int(np.searchsorted(d.values, np.datetime64(A0)))
    s = int(np.searchsorted(d.values, np.datetime64(SPLIT)))
    try:
        f2, overlay, info = _feats(key)
    except FileNotFoundError:
        return key, None, None
    raw = run(p, f2, cfg, a, len(d), detail=True)["raw"]
    c = raw["curve"]
    r = np.r_[0.0, np.diff(c) / c[:-1]]
    if overlay:
        r = overlay(r, a)
    return key, r, {"split_i": s - a, "info": info, "first": str(d[a].date()), "last": str(d[-1].date())}


def _stats(x):
    cc = np.cumprod(1 + x)
    yrs = len(x) / 252
    return {"ann": round(float((cc[-1] ** (1 / yrs) - 1) * 100), 1),
            "sharpe": round(float(x.mean() / (x.std() + 1e-12) * np.sqrt(252)), 2),
            "mdd": round(float((cc / np.maximum.accumulate(cc) - 1).min() * 100), 1)}


def _boot(x, y, B=4000, L=20, seed=7):
    """P[Sharpe(x) − Sharpe(y) ≤ 0] 과 차이의 표준오차 (같은 블록을 뽑아 두 전략의 상관 유지)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    k = int(np.ceil(n / L))
    sh = lambda v: v.mean() / (v.std() + 1e-12) * np.sqrt(252)  # noqa: E731
    ds = np.empty(B)
    for b in range(B):
        st = rng.integers(0, n - L, k)
        ii = (st[:, None] + np.arange(L)).ravel()[:n]
        ds[b] = sh(x[ii]) - sh(y[ii])
    return round(float((ds <= 0).mean()), 4), round(float(ds.std()), 3)


def main():
    keys = ["BASE"] + [k for k, *_ in CANDS + SENS + HIND]
    missing = [k for k, g, _ in CANDS if g == "순위 교체" and not os.path.exists(os.path.join(CACHE, f"alpha_{k}.pkl"))]
    if missing:
        print(f"순위 교체 후보 예측 없음 → 건너뜀 ({', '.join(missing)}). 먼저 python src/alpha_lab.py")
    with ProcessPoolExecutor(min(4, os.cpu_count() or 1), initializer=_init) as ex:
        res = {k: (r, m) for k, r, m in ex.map(_one, keys)}
    rb, mb = res["BASE"]
    si = mb["split_i"]
    seg = {"pre": slice(0, si), "post": slice(si, None), "all": slice(0, None)}
    base = {s: _stats(rb[sl]) for s, sl in seg.items()}
    m_tests = sum(1 for k, *_ in CANDS if res.get(k, (None,))[0] is not None)
    alpha = 0.05 / max(m_tests, 1)
    rows, ses = [], []
    for k, grp, desc in CANDS + SENS + HIND:
        r, meta = res.get(k, (None, None))
        if r is None:
            continue
        st = {s: _stats(r[sl]) for s, sl in seg.items()}
        p_all, se = _boot(r, rb)
        ses.append(se)
        both = st["pre"]["sharpe"] > base["pre"]["sharpe"] and st["post"]["sharpe"] > base["post"]["sharpe"]
        kind = "민감도" if (k, grp, desc) in SENS else ("측정" if grp == "가중치 재현" else "후보")
        verdict = ("채택" if both and p_all < alpha else "두 구간 개선 · 유의성 부족" if both else "기각") if kind == "후보" else kind
        rows.append({"key": k, "group": grp, "desc": desc, "kind": kind, **st, "p": p_all, "se": se,
                     "d_sharpe": round(st["all"]["sharpe"] - base["all"]["sharpe"], 2), "verdict": verdict,
                     **({"weights": meta["info"]} if meta and meta.get("info") else {})})
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha)
    se_med = float(np.median(ses)) if ses else float("nan")
    wf = next((x for x in rows if x["key"] == "wf:ICW"), None)
    out = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "periods": {"pre": [mb["first"], "2023-12-31"], "post": [SPLIT, mb["last"]], "all": [mb["first"], mb["last"]]},
        "cost_oneway": COST, "m_tests": m_tests, "alpha": round(alpha, 4),
        "rule": "두 구간 모두 Sharpe 개선 + 전체 구간 부트스트랩 P[차≤0] < 0.05/시험 수",
        "base": base, "rows": rows,
        "adopted": [x["key"] for x in rows if x["verdict"] == "채택"],
        "power": {"se_sharpe_diff": round(se_med, 2), "min_detectable": round(z * se_med, 2),
                  "note": f"이 기간 길이로는 Sharpe 차이가 약 {z * se_med:.2f} 이상이어야 우연과 구별된다"},
        "hindsight": ({"fixed_post": base["post"]["sharpe"], "wf_post": wf["post"]["sharpe"],
                       "fixed_pre": base["pre"]["sharpe"], "wf_pre": wf["pre"]["sharpe"],
                       "note": "현재 가중치는 2024년 이후 성과도 보고 골랐다. 그 시점 정보만으로 가중치를 정하면 검증 구간 Sharpe 가 이만큼 낮다."}
                      if wf else None),
    }
    for path in (OUT, OUT_UI):
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"기준선 BASE  {base['pre']['sharpe']:.2f} / {base['post']['sharpe']:.2f} (2021-02~2023 / 2024~ Sharpe)")
    for x in rows:
        print(f"  {x['group']:<10} {x['desc']:<28} {x['pre']['sharpe']:5.2f} / {x['post']['sharpe']:5.2f}  "
              f"Δ{x['d_sharpe']:+.2f}  P {x['p']:.3f}  → {x['verdict']}")
    print(f"시험 {m_tests}건 · 문턱 P<{alpha:.4f} · 최소 검출 Sharpe 차 ≈ {out['power']['min_detectable']}")
    print(f"저장: {OUT}, {OUT_UI}")


if __name__ == "__main__":
    main()
