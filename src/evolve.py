"""
모델 진화실 — 전략을 쉬지 않고 시험하고, 증거가 쌓일 때만 바꾼다.
-> data/evolve/registry.json · data/evolve/history.csv · docs/evolve.json (성과 보고서 '모델 진화' 탭)

무엇을 바꿔 보나 (유전자)
  순위 가중치 9개 : 종합점수 요소 6개(가치·수익성·성장·수급·위험조정모멘텀·모멘텀)
                   + 위험·섹터 신호 3개(고유변동성 낮음·급등 적음·섹터 모멘텀, features.extra_signals)
  매매 규칙 11개  : 보유 종목 수 · 진입/이탈 순위 · 최소 보유일 · 손절 · 트레일 · 트레일 조임 · 부분익절
                   · 급락장 게이트 · 반등 기준 · 외국인 수급 필터

어떻게 고르나 (진화)
  인구 32개 × 10세대(주간 실행은 48개 × 14세대). 상위 8개를 남기고 변이(유전자 1~3개)와 교차로 자식을 만든다.
  적합도 = 해마다 Sharpe 의 평균 − 0.5 × 해마다 Sharpe 의 표준편차
           → 한 해만 크게 잘된 설정보다 해마다 고르게 괜찮은 설정이 이긴다. 거래 20건 미만은 탈락.
  지금 챔피언을 이기려면 적합도가 HYST(0.10) 이상 높아야 한다 (작은 차이로 설정이 흔들리지 않게).

진화가 정말 도움이 되나 — 중첩 워크포워드 (이 파일의 핵심)
  2021년부터 분기마다 '그 시점까지의 데이터로만' 진화 → 다음 분기를 그 설정으로 운용.
  이어 붙인 성과가 '진화하는 절차' 자체의 정직한 성과다. 같은 분기들에서 함께 운용한 비교 대상:
    · 지금 설정 고정 — 가중치를 2024년 이후 성과까지 보고 골랐으므로 위쪽으로 치우친 기준
    · 무학습 기준    — 6요소 동일가중 + 기본 규칙 (사후 지식이 전혀 없음)
  진화가 무학습 기준을 못 이기면 '적응'은 믿지 않고, 도전자 승격도 막는다.
  모든 분기는 현금에서 새로 시작한다(비교 대상 모두 같은 조건).

쉬지 않는 진화 — 인구(data/evolve/population.json)를 저장해 두고 2시간마다 몇 세대씩 이어서 진화한다.
  처음부터 다시 시작하지 않으므로 세대가 계속 쌓이고, 매일 새로 들어온 자료로 적합도를 다시 잰다.
  밤마다(evolve.yml) 중첩 워크포워드로 '진화가 정말 도움이 되나'를 다시 재고 더 큰 진화를 한 번 돌린다.

실제 적용 — 챔피언 · 도전자 · 전진 그림자 시험
  매주 전체 데이터로 진화해 나온 최선의 설정을 '도전자'로 등록한다. 성적은 등록일 '이후' 들어온
  데이터로만 매긴다 — 등록할 때는 존재하지 않던 데이터라 사후에 고르는 것이 불가능하다.
  승격 조건 (모두 충족)
    1) 전진 시험 60거래일 이상
    2) 같은 기간 챔피언보다 Sharpe 와 수익률이 모두 높음
    3) 전체 기간 적합도가 챔피언 − 0.1 이상 (전진 구간 운이 아님)
    4) 중첩 워크포워드에서 진화 절차가 무학습 기준보다 나음
  기본은 '승격 제안'만 한다(사용자 승인: autotrade-control → promote).
  config evolve.auto_promote: true 면 조건 통과 시 config/champion.yaml 을 자동으로 쓴다.

빠른 시뮬레이터(fastsim)는 decision_engine.decide + engine_backtest.run 과 같은 규칙·같은 현금 회계다
(뉴스 거부권·재진입 대기는 엔진 백테스트에서도 꺼져 있다). `check` 로 엔진과 대조한다.

사용
  python src/evolve.py check          # 빠른 시뮬레이터 = 엔진 대조
  python src/evolve.py meta           # 중첩 워크포워드 (수 분)
  python src/evolve.py weekly         # 진화 → 도전자 등록 → 전진 성적 → 승격 판단 → docs/evolve.json
  python src/evolve.py step           # 연속 진화 한 번 (learn.yml 이 2시간마다) — 저장된 인구에서 이어서 몇 세대 더
  python src/evolve.py promote <ID>   # 도전자를 챔피언으로 (config/champion.yaml)
"""
import os
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _SRC]
import signal  # noqa: E402,F401  (src/signal.py 가 표준 라이브러리를 가리는 것 방지 — 병렬 실행용)
sys.path.append(_SRC)

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import features as F  # noqa: E402
import settings  # noqa: E402

COST = 0.0015
WARMUP = 260
MIN_TRAIN = 504            # 진화에 쓰는 최소 학습 기간 (2년)
STEP = 63                  # 분기마다 다시 진화
ROLL = 756                 # '최근 3년' 변형의 학습 기간
HYST = 0.10                # 챔피언을 바꾸려면 적합도가 이만큼 높아야 한다
POP, GENS, ELITE = 32, 10, 8
SPLIT = "2024-01-01"
OUT_DIR, OUT_UI = "data/evolve", "docs/evolve.json"
REG = os.path.join(OUT_DIR, "registry.json")
HIST = os.path.join(OUT_DIR, "history.csv")
META = os.path.join(OUT_DIR, "meta.json")
POPF = os.path.join(OUT_DIR, "population.json")      # 연속 진화: 세대를 넘겨 이어 가는 인구

KEYS = F.RANK_KEYS
BASE_W = {"s_value": .29, "s_profit": .20, "s_grow": .16, "s_flow": .18, "s_rmom": .10, "s_mom": .07}
LABEL = {"s_value": "가치", "s_profit": "수익성", "s_grow": "성장", "s_flow": "수급", "s_rmom": "위험조정모멘텀",
         "s_mom": "모멘텀", "x_ivol": "저변동", "x_max21": "급등 적음", "x_secmom": "섹터 모멘텀"}
CHOICES = {
    "max_positions": [5, 7, 10, 12, 15],
    "entry_rank": [5, 7, 10, 12, 15],
    "exit_rank": [15, 20, 25, 30, 40],
    "min_hold": [0, 10, 20, 30, 45, 60],
    "stop": [0.07, 0.10, 0.12, 0.15, 0.20],
    "trail": [0.06, 0.08, 0.10, 0.12, 0.15],
    "tiers": [True, False],
    "tp": [None, 0.15, 0.25, 0.35],
    "crash_gate": [True, False],
    "rebound": [0.03, 0.05, 0.08],
    "flow_filter": [True, False],
}
GLABEL = {"max_positions": "보유 종목 수", "entry_rank": "진입 순위", "exit_rank": "이탈 순위", "min_hold": "최소 보유일",
          "stop": "손절", "trail": "트레일", "tiers": "트레일 조임", "tp": "부분익절", "crash_gate": "급락장 게이트",
          "rebound": "반등 기준", "flow_filter": "수급 필터"}
TIERS = [(0.30, 0.03), (0.15, 0.05)]
NAIVE = {"w": {k: round(1 / 6, 4) for k in BASE_W}, "max_positions": 10, "entry_rank": 10, "exit_rank": 20,
         "min_hold": 30, "stop": 0.10, "trail": 0.08, "tiers": True, "tp": 0.25, "crash_gate": True,
         "rebound": 0.05, "flow_filter": True}


# ══════════════════════════════════════════════════════════════
# 데이터 (한 번 읽고, 병렬 작업자는 fork 로 그대로 물려받는다)
# ══════════════════════════════════════════════════════════════
D = None


def load():
    p, f = F.build_all()
    ex = F.extra_signals(p, f)
    d = SimpleNamespace()
    d.dates = p["dates"]
    d.nD = len(d.dates)
    d.px = p["close"].values.astype(float)
    sc = p["score"].values.astype(float)
    d.mask = np.isfinite(sc)
    d.comps = {}
    for k in KEYS:
        m = p.get(k) if k.startswith("s_") else ex.get(k)
        if m is not None:
            d.comps[k] = np.nan_to_num(m.reindex(index=p["dates"], columns=p["codes"]).values.astype(float), nan=50.0)
    fl = f.get("flow")
    d.flow_bad = (fl.values < 0) if fl is not None else np.zeros_like(d.px, bool)
    d.flow_bad = np.where(np.isfinite(fl.values) if fl is not None else False, d.flow_bad, False)
    mk = p["market"]
    d.crash = (mk["regime"] == "급락장").values
    idx = mk["idx"].values.astype(float)
    lo = pd.Series(idx).rolling(20, min_periods=1).min().values      # decision_engine.market_gate 와 같은 창
    d.reb = idx / lo - 1
    d.idx = idx
    d.years = d.dates.year.values
    d.base_rank = f["rank"].values.astype(float)
    d.split = int(np.searchsorted(d.dates.values, np.datetime64(SPLIT)))
    d.codes = p["codes"]
    return d


_RCACHE = {}


def _order(rk):
    o = np.argsort(np.where(np.isfinite(rk), rk, np.inf), axis=1, kind="stable")[:, :20]
    return o


def ranks(w):
    """가중치 dict(없으면 종합점수) → (순위 행렬, 날짜별 상위 20 순서). 프로세스별 캐시."""
    key = None if not w else tuple(round(float(w.get(k, 0)), 4) for k in KEYS)
    if key in _RCACHE:
        return _RCACHE[key]
    if key is None:
        rk = D.base_rank
    else:
        sc = None
        for k, v in zip(KEYS, key):
            if v and k in D.comps:
                sc = D.comps[k] * v if sc is None else sc + D.comps[k] * v
        sc = np.where(D.mask, sc, np.nan)
        o = np.argsort(np.where(np.isfinite(sc), -sc, np.inf), axis=1, kind="stable")
        rk = np.empty_like(sc)
        np.put_along_axis(rk, o, np.arange(1, sc.shape[1] + 1, dtype=float)[None, :].repeat(sc.shape[0], 0), axis=1)
        rk[~np.isfinite(sc)] = np.nan
    if len(_RCACHE) > 400:
        _RCACHE.clear()
    _RCACHE[key] = (rk, _order(rk))
    return _RCACHE[key]


# ══════════════════════════════════════════════════════════════
# 빠른 시뮬레이터 — engine_backtest.run 과 같은 규칙·회계
# ══════════════════════════════════════════════════════════════
def fastsim(g, d0, d1):
    """반환: (일별 수익률 배열 길이 d1-d0, 매도 체결 수)"""
    rk, order = ranks(g.get("w"))
    px = D.px
    maxpos, er, xr, mh = g["max_positions"], g["entry_rank"], g["exit_rank"], g["min_hold"]
    sl, tb, tp, tpf = g["stop"], g["trail"], g.get("tp"), 0.3333
    tiers = TIERS if g.get("tiers", True) else ()
    gate = ~D.crash | (D.reb >= g["rebound"]) if g.get("crash_gate", True) else np.ones(D.nD, bool)
    fb = D.flow_bad if g.get("flow_filter", True) else None
    cash, pos, sells, buys = 1.0, {}, [], []
    n = d1 - d0
    rets = np.zeros(n)
    eq_prev, ntr = 1.0, 0
    for k in range(n):
        t = d0 + k
        pt = px[t]
        if sells or buys:
            eq_now = cash
            for i, q in pos.items():
                c = pt[i]
                if c == c:
                    eq_now += q[0] * c
            for i, fr in sells:
                q = pos.get(i)
                c = pt[i]
                if q is None or c != c:
                    continue
                sh = q[0] * fr
                cash += sh * c * (1 - COST)
                q[0] -= sh
                ntr += 1
                if fr < 1:
                    q[4] = True
                if q[0] <= 1e-12 or fr >= 1:
                    del pos[i]
            for i in buys:
                c = pt[i]
                if c != c or i in pos:
                    continue
                amt = min(eq_now / maxpos, cash)
                if amt <= 1e-9:
                    continue
                pos[i] = [amt / (c * (1 + COST)), c, t, c, False]
                cash -= amt
            sells, buys = [], []
        eq = cash
        for i, q in pos.items():
            c = pt[i]
            if c == c:
                if c > q[3]:
                    q[3] = c
                eq += q[0] * c
            else:
                eq += q[0] * q[1]
        rets[k] = eq / eq_prev - 1
        eq_prev = eq
        if t >= d1 - 1:
            break
        rkt = rk[t]
        n_exit = 0
        for i, q in pos.items():
            c = pt[i]
            if c != c:
                continue
            peak = q[3] if q[3] > c else c
            ret = c / q[1] - 1
            pg = peak / q[1] - 1
            tw = tb
            for lvl, w_ in tiers:
                if pg >= lvl:
                    tw = w_
                    break
            r = rkt[i]
            if ret <= -sl or c / peak - 1 <= -tw:
                sells.append((i, 1.0))
                n_exit += 1
            elif tp and ret >= tp and not q[4]:
                sells.append((i, tpf))
            elif (r != r or r > xr) and t - q[2] >= mh:
                sells.append((i, 1.0))
                n_exit += 1
        free = maxpos - (len(pos) - n_exit)
        if free > 0 and gate[t]:
            for i in order[t]:
                r = rkt[i]
                if r != r or r > er:
                    break
                if i in pos or (fb is not None and fb[t, i]):
                    continue
                buys.append(i)
                free -= 1
                if free == 0:
                    break
    return rets, ntr


def stats(r):
    if len(r) < 2:
        return {"ann": None, "sharpe": None, "mdd": None, "total": None}
    c = np.cumprod(1 + r)
    yrs = len(r) / 252
    return {"ann": round(float((c[-1] ** (1 / max(yrs, 1e-9)) - 1) * 100), 1) if yrs >= 0.25 else None,
            "total": round(float((c[-1] - 1) * 100), 1),
            "sharpe": round(float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)), 2),
            "mdd": round(float((c / np.maximum.accumulate(c) - 1).min() * 100), 1)}


def fitness(g, a, b):
    """해마다 Sharpe 평균 − 0.5 × 표준편차. 반환 dict."""
    r, ntr = fastsim(g, a, b)
    yrs = D.years[a:b]
    sh = []
    for y in np.unique(yrs):
        m = yrs == y
        if m.sum() >= 120:
            x = r[m]
            sh.append(x.mean() / (x.std() + 1e-12) * np.sqrt(252))
    if len(sh) < 2:
        h = len(r) // 2
        sh = [x.mean() / (x.std() + 1e-12) * np.sqrt(252) for x in (r[:h], r[h:])]
    fit = float(np.mean(sh) - 0.5 * np.std(sh)) if ntr >= 20 else -9.0
    s = stats(r)
    return {"fit": round(fit, 4), "sharpe": s["sharpe"], "ann": s["ann"], "mdd": s["mdd"], "trades": ntr}


# ══════════════════════════════════════════════════════════════
# 유전자 연산
# ══════════════════════════════════════════════════════════════
def gkey(g):
    w = g.get("w")
    wk = None if not w else tuple(round(float(w.get(k, 0)), 2) for k in KEYS)
    return (wk,) + tuple(g[k] for k in CHOICES)


def norm_w(w):
    w = {k: max(0.0, float(v)) for k, v in w.items() if k in KEYS}
    s = sum(w.values()) or 1.0
    w = {k: round(v / s, 2) for k, v in w.items() if v / s >= 0.005}
    return w


def mutate(g, rng, n=None, only=None):
    """only: None(전부) · "w"(순위 가중치만) · "rules"(매매 규칙만)"""
    c = {**g, "w": dict(g["w"]) if g.get("w") else dict(BASE_W)}
    genes = (["w", "w", "w"] if only != "rules" else []) + (list(CHOICES) if only != "w" else [])
    n = min(n or int(rng.integers(1, 4)), len(genes))
    for gene in rng.choice(genes, size=n, replace=False):
        if gene == "w":
            w = c["w"]
            k = KEYS[int(rng.integers(len(KEYS)))]
            if w.get(k, 0) < 0.01:
                w[k] = float(rng.choice([0.05, 0.10, 0.15, 0.20]))
            else:
                w[k] = w[k] * float(np.exp(rng.normal(0, 0.6)))
                if rng.random() < 0.15:
                    w[k] = 0.0
            c["w"] = norm_w(w)
        else:
            opts = [o for o in CHOICES[gene] if o != c[gene]]
            c[gene] = opts[int(rng.integers(len(opts)))]
    if c["entry_rank"] > c["exit_rank"]:
        c["exit_rank"] = min(x for x in CHOICES["exit_rank"] if x >= c["entry_rank"])
    return c


def crossover(a, b, rng, only=None):
    c = {k: (a[k] if (only == "w" or rng.random() < 0.5) else b[k]) for k in CHOICES}
    wa, wb = a.get("w") or BASE_W, b.get("w") or BASE_W
    c["w"] = dict(wa) if only == "rules" else norm_w({k: (wa.get(k, 0) + wb.get(k, 0)) / 2 for k in KEYS})
    if c["entry_rank"] > c["exit_rank"]:
        c["exit_rank"] = min(x for x in CHOICES["exit_rank"] if x >= c["entry_rank"])
    return c


def dist(g, ref):
    """두 설정이 얼마나 다른가: 가중치 L1 거리(0~2) + 규칙 하나 다를 때마다 0.1"""
    wa, wb = g.get("w") or BASE_W, ref.get("w") or BASE_W
    return sum(abs(wa.get(k, 0) - wb.get(k, 0)) for k in KEYS) + 0.1 * sum(g[k] != ref[k] for k in CHOICES)


def consensus(gs):
    """상위 설정들의 합의: 가중치는 평균, 규칙은 다수결 (같으면 앞선 것)."""
    w = norm_w({k: float(np.mean([(g.get("w") or BASE_W).get(k, 0) for g in gs])) for k in KEYS})
    c = {"w": w}
    for k in CHOICES:
        vals = [g[k] for g in gs]
        c[k] = max(vals, key=lambda v: (vals.count(v), -vals.index(v)))
    if c["entry_rank"] > c["exit_rank"]:
        c["exit_rank"] = min(x for x in CHOICES["exit_rank"] if x >= c["entry_rank"])
    return c


# 미리 정한 진화 방식들 (중첩 워크포워드에서 함께 시험 → 시험 수만큼 문턱을 높인다)
MODES = {
    "free":       {"label": "제약 없는 진화",            "win": "expand"},
    "free_roll":  {"label": "제약 없는 진화 (최근 3년)", "win": "roll"},
    "shrink":     {"label": "수축 (기본값에서 멀수록 감점)", "win": "expand", "lam": 0.5},
    "consensus":  {"label": "합의 (상위 8개 평균)",      "win": "expand", "k": 8},
    "w_only":     {"label": "순위 가중치만 학습",        "win": "expand", "only": "w"},
    "rules_only": {"label": "매매 규칙만 학습",          "win": "expand", "only": "rules"},
}


def _eval(args):
    g, a, b = args
    return fitness(g, a, b)


def evolve(a, b, seeds, rng, pool, pop=POP, gens=GENS, log=None, mode=None, out_pop=None):
    """[a, b) 데이터로만 진화. 반환: (최선 유전자, 그 성적, 평가한 수).
    mode: MODES 항목 — lam(수축 강도) · k(합의할 상위 개수) · only(학습할 유전자)
    out_pop: 목록을 넘기면 마지막 인구(점수 순 상위 pop 개)를 채워 준다 — 다음 실행이 여기서 이어 간다."""
    mode = mode or {}
    lam, k_cons, only = float(mode.get("lam", 0)), int(mode.get("k", 0)), mode.get("only")
    adj = lambda g, r: r["fit"] - lam * dist(g, NAIVE)  # noqa: E731
    cache = {}

    def evaluate(gs):
        todo = [g for g in gs if gkey(g) not in cache]
        uniq = list({gkey(g): g for g in todo}.values())
        res = pool.map(_eval, [(g, a, b) for g in uniq]) if pool else [_eval((g, a, b)) for g in uniq]
        for g, r in zip(uniq, res):
            cache[gkey(g)] = (g, r)

    popl = list({gkey(s): s for s in seeds}.values())
    while len(popl) < pop:
        popl.append(mutate(seeds[int(rng.integers(len(seeds)))], rng, only=only))
    for gen in range(gens):
        evaluate(popl)
        ranked = sorted({gkey(g): cache[gkey(g)] for g in popl}.values(), key=lambda x: -adj(*x))
        elites = [x[0] for x in ranked[:ELITE]]
        if log is not None:
            log.append({"gen": gen, "best": ranked[0][1]["fit"], "median": float(np.median([x[1]["fit"] for x in ranked]))})
        if gen == gens - 1:
            break
        kids, tries = [], 0
        while len(kids) < pop - len(elites) and tries < pop * 20:
            tries += 1
            if rng.random() < 0.3 and len(elites) > 1:
                i, j = rng.choice(len(elites), 2, replace=False)
                kid = crossover(elites[i], elites[j], rng, only=only)
            else:
                kid = mutate(elites[int(rng.integers(len(elites)))], rng, only=only)
            if gkey(kid) not in cache and all(gkey(kid) != gkey(x) for x in kids):
                kids.append(kid)
        popl = elites + kids
    allr = sorted(cache.values(), key=lambda x: -adj(*x))
    if out_pop is not None:
        out_pop.extend(x[0] for x in allr[:pop])
    if k_cons:
        g = consensus([x[0] for x in allr[:k_cons]])
        r = cache[gkey(g)][1] if gkey(g) in cache else fitness(g, a, b)
        return g, {**r, "adj": round(adj(g, r), 4)}, len(cache) + 1
    g, r = allr[0]
    return g, {**r, "adj": round(adj(g, r), 4)}, len(cache)


def choose(champ, champ_fit, best, best_fit):
    """히스테리시스: 새 설정이 HYST 이상 나아야 바꾼다 (수축 방식이면 감점 반영한 점수로)."""
    if champ is None or best_fit.get("adj", best_fit["fit"]) > champ_fit.get("adj", champ_fit["fit"]) + HYST:
        return best, best_fit, True
    return champ, champ_fit, False


# ══════════════════════════════════════════════════════════════
# 설정 ↔ 유전자
# ══════════════════════════════════════════════════════════════
def genome_from_cfg(st):
    tp = st.get("take_profit_gain")
    return {"w": st.get("rank_weights") or None, "max_positions": int(st["max_positions"]),
            "entry_rank": int(st["entry_rank"]), "exit_rank": int(st["exit_rank"]),
            "min_hold": int(st["min_hold_days"]), "stop": float(st["stop_loss"]), "trail": float(st["trail_base"]),
            "tiers": bool(st.get("trail_tiers")), "tp": (float(tp) if tp and tp < 5 else None),
            "crash_gate": bool(st.get("crash_gate", True)), "rebound": float(st.get("crash_rebound_min", 0.05)),
            "flow_filter": bool(st.get("flow_filter", True))}


def strategy_from_genome(g):
    return {"rank_weights": g.get("w") or None, "max_positions": g["max_positions"], "entry_rank": g["entry_rank"],
            "watch_rank": max(20, g["entry_rank"]), "exit_rank": g["exit_rank"], "min_hold_days": g["min_hold"],
            "stop_loss": g["stop"], "trail_base": g["trail"],
            "trail_tiers": [list(x) for x in TIERS] if g["tiers"] else [],
            "take_profit_gain": g["tp"], "crash_gate": g["crash_gate"], "crash_rebound_min": g["rebound"],
            "flow_filter": g["flow_filter"]}


def describe(g, ref=None):
    """사람이 읽는 설명. ref 가 있으면 다른 점만."""
    out = []
    w, rw = g.get("w") or BASE_W, (ref or {}).get("w") or BASE_W
    if ref is None or gkey({**g, "w": w}) [0] != gkey({**ref, "w": rw})[0]:
        top = sorted(w.items(), key=lambda x: -x[1])
        out.append("순위 " + " · ".join(f"{LABEL[k]} {v:.2f}" for k, v in top if v >= 0.01) + ("" if g.get("w") else " (종합점수)"))
    fmt = {"stop": lambda v: f"-{v*100:.0f}%", "trail": lambda v: f"-{v*100:.0f}%", "tp": lambda v: "끔" if not v else f"+{v*100:.0f}%",
           "rebound": lambda v: f"+{v*100:.0f}%", "tiers": lambda v: "켬" if v else "끔", "crash_gate": lambda v: "켬" if v else "끔",
           "flow_filter": lambda v: "켬" if v else "끔", "min_hold": lambda v: f"{v}일"}
    for k in CHOICES:
        if ref is None or g[k] != ref[k]:
            f = fmt.get(k, lambda v: f"{v}")
            out.append(f"{GLABEL[k]} {f(ref[k])}→{f(g[k])}" if ref is not None else f"{GLABEL[k]} {f(g[k])}")
    return out


# ══════════════════════════════════════════════════════════════
# 대조 · 중첩 워크포워드
# ══════════════════════════════════════════════════════════════
def boot_p(x, y, B=3000, L=20, seed=11):
    """P[Sharpe(x) − Sharpe(y) ≤ 0] — 같은 블록을 뽑아 두 전략의 상관을 유지."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2 * L:
        return None
    k = int(np.ceil(n / L))
    sh = lambda v: v.mean() / (v.std() + 1e-12)  # noqa: E731
    cnt = 0
    for _ in range(B):
        st = rng.integers(0, n - L, k)
        ii = (st[:, None] + np.arange(L)).ravel()[:n]
        cnt += sh(x[ii]) - sh(y[ii]) <= 0
    return round(cnt / B, 4)


def check(cfg):
    from engine_backtest import run
    p, f = F.build_all()
    f = F.apply_strategy(p, f, cfg["strategy"])
    g = genome_from_cfg(cfg["strategy"])
    rows = []
    for nm, a, b in [("2019~2023", WARMUP, D.split), ("2024~", D.split, D.nD)]:
        e = run(p, f, cfg, a, b)
        r, _ = fastsim(g, a, b)
        s = stats(r)
        rows.append((nm, e, s))
        print(f"  {nm:<10} 엔진 연{e['ann']:+6.1f}% Sh{e['sharpe']:5.2f} MDD{e['mdd']:6.1f}% │ 빠른 연{s['ann']:+6.1f}% Sh{s['sharpe']:5.2f} MDD{s['mdd']:6.1f}%")
    return rows


def meta(cfg, pool, seed=7):
    """분기마다 그 시점까지의 데이터로만 진화 → 다음 분기 운용. 미리 정한 방식(MODES) 모두를 같은 분기에서.
    반환 dict (화면용)."""
    static = genome_from_cfg(cfg["strategy"])
    refits = list(range(WARMUP + MIN_TRAIN, D.nD - 10, STEP))
    series = {k: [] for k in list(MODES) + ["static", "naive"]}
    champs = {m: [None, None] for m in MODES}
    log, n_eval = [], 0
    for T in refits:
        T1 = min(T + STEP, D.nD)
        row = {"date": str(D.dates[T].date())}
        for m, md in MODES.items():
            rng = np.random.default_rng([seed, T, list(MODES).index(m)])
            a = WARMUP if md["win"] == "expand" else max(WARMUP, T - ROLL)
            ch, chf = champs[m]
            seeds = [NAIVE] + ([ch] if ch else [])
            best, bf, ne = evolve(a, T, seeds, rng, pool, mode=md)
            n_eval += ne
            if ch is not None:                                  # 같은 학습 구간으로 다시 재서 비교
                r0 = fitness(ch, a, T)
                chf = {**r0, "adj": round(r0["fit"] - float(md.get("lam", 0)) * dist(ch, NAIVE), 4)}
            ch, chf, changed = choose(ch, chf, best, bf)
            champs[m] = [ch, chf]
            r, _ = fastsim(ch, T, T1)
            series[m].append(r)
            row[m] = {"changed": changed, "fit": chf["fit"], "oos": stats(r)["total"], "desc": describe(ch, NAIVE)[:4]}
        for v, g in [("static", static), ("naive", NAIVE)]:
            r, _ = fastsim(g, T, T1)
            series[v].append(r)
            row[v] = {"oos": stats(r)["total"]}
        log.append(row)
        print(f"  {row['date']} " + " ".join(f"{m[:6]} {row[m]['oos']:+6.1f}%{'*' if row[m]['changed'] else ' '}" for m in MODES)
              + f" │ 고정 {row['static']['oos']:+6.1f}% 무학습 {row['naive']['oos']:+6.1f}%", flush=True)
    full = {k: np.concatenate(v) for k, v in series.items()}
    dates = [str(D.dates[i].date()) for T in refits for i in range(T, min(T + STEP, D.nD))]
    cut = sum(1 for d in dates if d < SPLIT)
    res = {k: {"all": stats(r), "pre": stats(r[:cut]), "post": stats(r[cut:])} for k, r in full.items()}
    alpha = round(0.05 / len(MODES), 4)
    tests = {m: {"vs_naive_p": boot_p(full[m], full["naive"]), "vs_static_p": boot_p(full[m], full["static"])} for m in MODES}
    passed = [m for m in MODES
              if all(res[m][p]["sharpe"] > res["naive"][p]["sharpe"] for p in ("all", "pre", "post"))
              and tests[m]["vs_naive_p"] is not None and tests[m]["vs_naive_p"] < alpha]
    best = max(MODES, key=lambda m: res[m]["all"]["sharpe"])
    curve = {k: [round(float(x), 4) for x in np.cumprod(1 + v)[::5]] for k, v in full.items()}
    return {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "refits": len(refits), "step": STEP,
            "pop": POP, "gens": GENS, "n_eval": n_eval, "period": [dates[0], dates[-1]],
            "modes": {m: md["label"] for m, md in MODES.items()}, "alpha": alpha,
            "results": res, "tests": tests, "best": best, "passed": passed, "trusted": bool(passed),
            "trusted_mode": (max(passed, key=lambda m: res[m]["all"]["sharpe"]) if passed else None),
            "rule": f"진화 방식의 바깥 구간 Sharpe 가 무학습 기준보다 전체·2024년 전·2024년 후 모두 높고, "
                    f"부트스트랩 P < 0.05/{len(MODES)} (방식 {len(MODES)}개를 시험했으므로) 이면 '적응을 신뢰'",
            "log": log, "curve": {"d": dates[::5], **curve}}


def meta_multi(cfg, pool, seeds=(7, 8, 9)):
    """진화는 난수에 따라 결과가 흔들린다 → 시드 여러 개로 반복하고, 모든 시드에서 통과해야 '신뢰'."""
    runs = []
    for sd in seeds:
        print(f"── 시드 {sd}", flush=True)
        runs.append(meta(cfg, pool, seed=sd))
    m0 = runs[0]
    keys = list(MODES) + ["static", "naive"]
    spread = {}
    for k in keys:
        sh = {p: [r["results"][k][p]["sharpe"] for r in runs] for p in ("all", "pre", "post")}
        spread[k] = {p: {"mean": round(float(np.mean(v)), 2), "min": round(float(np.min(v)), 2), "max": round(float(np.max(v)), 2)}
                     for p, v in sh.items()}
        if k in MODES:
            spread[k]["p_mean"] = round(float(np.mean([r["tests"][k]["vs_naive_p"] for r in runs])), 3)
    passed = [m for m in MODES if all(m in r["passed"] for r in runs)]
    best = max(MODES, key=lambda m: spread[m]["all"]["mean"])
    return {**m0, "seeds": list(seeds), "spread": spread, "passed": passed, "trusted": bool(passed),
            "trusted_mode": (max(passed, key=lambda m: spread[m]["all"]["mean"]) if passed else None), "best": best,
            "n_eval": int(sum(r["n_eval"] for r in runs)),
            "rule": m0["rule"] + f" — 시드 {len(seeds)}개 모두에서"}


# ══════════════════════════════════════════════════════════════
# 챔피언 · 도전자 등록부
# ══════════════════════════════════════════════════════════════
def load_reg(cfg):
    if os.path.exists(REG):
        return json.load(open(REG, encoding="utf-8"))
    g = genome_from_cfg(cfg["strategy"])
    today = str(D.dates[-1].date())
    # 첫 도전자: 무학습 기준(6요소 동일가중 + 기본 규칙). 중첩 워크포워드에서 지금 설정과 Sharpe 가 같고
    # 최대낙폭은 더 얕았다 — 사후 지식 없이 정한 설정이라 전진 시험으로 맞붙일 가치가 있다.
    return {"champion": {"id": "C0", "since": today, "genome": g, "source": "초기 설정 (config/autotrade.yaml)"},
            "challengers": [{"id": "N0", "born": today, "genome": NAIVE, "status": "관찰 중",
                             "source": "무학습 기준 (6요소 동일가중 + 기본 규칙)"}],
            "retired": [], "n_evaluated": 0}


def forward(g, born):
    """등록일 다음 날부터 지금까지 (현금에서 시작). 반환 (일별수익, 일수)."""
    ib = int(np.searchsorted(D.dates.values, np.datetime64(born), side="right"))
    if ib >= D.nD - 1:
        return np.zeros(0), 0
    r, _ = fastsim(g, ib, D.nD)
    return r, len(r)


def load_pop():
    try:
        return json.load(open(POPF, encoding="utf-8"))
    except (OSError, ValueError):
        return {"generations": 0, "runs": 0, "pop": []}


def weekly(cfg, pool, seed=None, pop=48, gens=14, label="주간 진화"):
    """진화 → 도전자 등록 → 전진 성적 → 승격. 저장된 인구에서 이어 간다(연속 진화).
    step 은 같은 일을 2시간마다(인구 48 × 20세대) 이어서 한다."""
    os.makedirs(OUT_DIR, exist_ok=True)
    ev = cfg.get("evolve", {}) or {}
    min_fwd = int(ev.get("min_forward_days", 60))
    long_fwd = int(ev.get("long_forward_days", 250))
    max_ch = int(ev.get("max_challengers", 6))
    reg = load_reg(cfg)
    champ = reg["champion"]
    today = str(D.dates[-1].date())
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % (2 ** 32))
    meta_res = json.load(open(META, encoding="utf-8")) if os.path.exists(META) else None
    trusted = bool(meta_res and meta_res.get("trusted"))
    # 1) 전체 데이터로 진화. 방식은 중첩 워크포워드가 신뢰한 것, 없으면 가장 보수적인 '수축'
    mname = (meta_res or {}).get("trusted_mode") or "shrink"
    md = MODES[mname]
    lam = float(md.get("lam", 0))
    popst = load_pop()
    seeds = [champ["genome"], NAIVE] + [c["genome"] for c in reg["challengers"]] + popst["pop"][:pop]
    glog = []
    final = []
    best, bf, ne = evolve(WARMUP, D.nD, seeds, rng, pool, pop=pop, gens=gens, log=glog, mode=md, out_pop=final)
    reg["n_evaluated"] = int(reg.get("n_evaluated", 0)) + ne
    popst = {"generations": int(popst.get("generations", 0)) + gens, "runs": int(popst.get("runs", 0)) + 1,
             "updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "data_date": today, "mode": mname,
             "best_adj": bf["adj"], "pop": final}
    json.dump(popst, open(POPF, "w", encoding="utf-8"), ensure_ascii=False, default=_js)
    cf = fitness(champ["genome"], WARMUP, D.nD)
    cf["adj"] = round(cf["fit"] - lam * dist(champ["genome"], NAIVE), 4)
    champ["fit"] = cf
    print(f"{label}({MODES[mname]['label']}): {ne}개 평가 · 누적 {popst['generations']}세대 · 최선 {bf['adj']:.3f} (Sharpe {bf['sharpe']}) · 챔피언 {cf['adj']:.3f}")
    # 2) 도전자 등록 — 챔피언보다 HYST 이상 낫고, 이미 있는 도전자와 다를 때
    #    전진 시험은 끝까지 가야 의미가 있으므로, 자리가 차면 새 도전자를 받지 않는다(기존 도전자를 밀어내지 않음).
    #    챔피언·도전자·물러난 설정과 거의 같은 설정(거리 0.15 미만)도 받지 않는다.
    others = [champ["genome"]] + [c["genome"] for c in reg["challengers"]] + [c["genome"] for c in reg["retired"]]
    near = min(dist(best, o) for o in others)
    full = len(reg["challengers"]) >= max_ch
    new_id = None
    if full:
        print(f"도전자 자리 없음 ({len(reg['challengers'])}/{max_ch}) — 기존 도전자의 전진 시험을 계속합니다")
    elif label == "연속 진화" and any(c["born"] == today and c["id"].startswith("E") for c in reg["challengers"]):
        print("오늘 자료로 이미 새 도전자를 등록했습니다 — 자료 하루에 하나만 (자리를 아껴 전진 시험을 끝까지)")
    elif bf["adj"] > cf["adj"] + HYST and near >= 0.15:
        used = {c["id"] for c in reg["challengers"] + reg["retired"]} | {champ["id"]}
        new_id = "E" + today.replace("-", "")
        k = 1
        while new_id in used:                                   # 하루에 여러 번 돌 수 있다 (연속 진화)
            k += 1
            new_id = "E" + today.replace("-", "") + chr(ord("a") + k - 2)
        reg["challengers"].append({"id": new_id, "born": today, "genome": best, "fit_born": bf, "status": "관찰 중",
                                   "source": f"{label} · {MODES[mname]['label']}"})
        print(f"새 도전자 {new_id}: " + " · ".join(describe(best, champ["genome"])[:5]))
    # 3) 전진 그림자 시험 + 승격 판단
    #    길 A (적응을 신뢰할 때): 전진 60일 + 챔피언보다 Sharpe·수익 모두 높음 + 전체 적합도 유지
    #    길 B (신뢰 못 해도):     전진 250일(약 1년) + 챔피언보다 높음 + 부트스트랩 P < 0.10
    proposals = []
    for c in reg["challengers"]:
        rc, n = forward(c["genome"], c["born"])
        rh, _ = forward(champ["genome"], c["born"])
        sc, sh = stats(rc), stats(rh)
        p_fwd = boot_p(rc, rh) if n >= 60 else None
        c["forward"] = {"days": n, "me": sc, "champ": sh, "p": p_fwd}
        c["fit_now"] = fitness(c["genome"], WARMUP, D.nD)
        beat = bool(n >= 2 and sc["sharpe"] is not None and sc["sharpe"] > sh["sharpe"] and sc["total"] > sh["total"])
        ok_a = {"days": n >= min_fwd, "beat": beat, "fit": c["fit_now"]["fit"] >= cf["fit"] - 0.10, "meta": trusted}
        ok_b = {"days": n >= long_fwd, "beat": beat, "p": p_fwd is not None and p_fwd < 0.10}
        c["checks"] = {"A": ok_a, "B": ok_b}
        if all(ok_a.values()) or all(ok_b.values()):
            c["status"] = "승격 후보"
            proposals.append(c)
        elif (n >= long_fwd and not beat) or (n >= 120 and sc["sharpe"] is not None and sc["sharpe"] < sh["sharpe"] - 0.5):
            c["status"] = "탈락"
        else:
            c["status"] = "관찰 중"
    for c in [c for c in reg["challengers"] if c["status"] == "탈락"]:
        reg["retired"].append({**c, "retired": today})
    reg["challengers"] = [c for c in reg["challengers"] if c["status"] != "탈락"]
    reg["retired"] = reg["retired"][-20:]
    promoted = None
    if proposals:
        top = max(proposals, key=lambda c: c["forward"]["me"]["sharpe"])
        if ev.get("auto_promote"):
            promoted = promote(reg, top["id"], why="자동 승격 (전진 시험 통과)")
        else:
            print(f"승격 제안: {top['id']} — autotrade-control 에서 promote {top['id']} 를 실행하면 적용됩니다.")
        try:
            import notify
            f = top["forward"]
            notify.send(cfg, "approval", f"{top['id']} 전진 {f['days']}거래일 · 수익 {f['me']['total']:+.1f}% (챔피언 {f['champ']['total']:+.1f}%)\n"
                        + ("자동으로 승격했습니다." if promoted else f"승인: autotrade-control → promote {top['id']}"),
                        "모델 진화: " + ("승격" if promoted else "승격 제안"))
        except Exception as e:                                   # 알림 실패는 진화를 막지 않는다
            print("알림 실패:", e)
    # 4) 챔피언 자신의 전진 성적 · 주간 기록
    rc, n = forward(champ["genome"], champ["since"])
    champ["forward"] = {"days": n, "me": stats(rc)}
    wl = os.path.join(OUT_DIR, "weekly.csv")                 # 같은 날 다시 돌리면 그 날 줄을 바꾼다
    row = {"날짜": today, "방식": mname, "평가 수": ne, "최선 점수": bf["adj"], "챔피언 점수": cf["adj"],
           "도전자 수": len(reg["challengers"]), "새 도전자": new_id or "", "적응 신뢰": trusted}
    old = pd.read_csv(wl, dtype={"새 도전자": str}).fillna("") if os.path.exists(wl) else pd.DataFrame(columns=list(row))
    old = old[old["날짜"].astype(str) != today]
    if not new_id and len(old) == 0 and any(c["born"] == today for c in reg["challengers"] if c["id"].startswith("E")):
        row["새 도전자"] = next(c["id"] for c in reg["challengers"] if c["id"].startswith("E") and c["born"] == today)
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(wl, index=False)
    json.dump(reg, open(REG, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=_js)
    write_ui(reg, meta_res, glog, proposals, promoted, cfg, mname, popst)
    return reg


def promote(reg, cid, why="사용자 승인"):
    c = next((c for c in reg["challengers"] if c["id"] == cid), None)
    if c is None:
        raise SystemExit(f"도전자 {cid} 없음")
    old = reg["champion"]
    today = str(D.dates[-1].date())
    reg["retired"].append({**old, "retired": today, "status": "이전 챔피언"})
    reg["champion"] = {"id": c["id"], "since": today, "genome": c["genome"], "source": why, "born": c["born"],
                       "forward_at_promotion": c.get("forward")}
    reg["challengers"] = [x for x in reg["challengers"] if x["id"] != cid]
    st = strategy_from_genome(c["genome"])
    with open(settings.CHAMPION, "w", encoding="utf-8") as fh:
        fh.write(f"# 진화 모델 챔피언 {c['id']} — src/evolve.py promote 가 씀. 지우면 config/autotrade.yaml 원래 설정으로 돌아감\n")
        yaml.safe_dump({"id": c["id"], "since": today, "why": why, "strategy": st}, fh, allow_unicode=True, sort_keys=False)
    new = not os.path.exists(HIST)
    with open(HIST, "a", encoding="utf-8") as fh:
        if new:
            fh.write("날짜,새 챔피언,이전 챔피언,사유,바뀐 점\n")
        fh.write(f"{today},{c['id']},{old['id']},{why},\"{' / '.join(describe(c['genome'], old['genome']))}\"\n")
    print(f"승격: {old['id']} → {c['id']} ({why})")
    return c["id"]


def _js(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def write_ui(reg, meta_res, glog, proposals, promoted, cfg, mname="shrink", popst=None):
    ch = reg["champion"]
    out = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "asof": str(D.dates[-1].date()),
        "auto_promote": bool((cfg.get("evolve") or {}).get("auto_promote")),
        "min_forward_days": int((cfg.get("evolve") or {}).get("min_forward_days", 60)),
        "n_evaluated": reg.get("n_evaluated", 0),
        "champion": {"id": ch["id"], "since": ch["since"], "source": ch.get("source"), "desc": describe(ch["genome"]),
                     "fit": ch.get("fit"), "forward": ch.get("forward")},
        "challengers": [{"id": c["id"], "born": c["born"], "status": c["status"], "checks": c.get("checks"), "source": c.get("source"),
                         "forward": c.get("forward"), "fit_born": c.get("fit_born"), "fit_now": c.get("fit_now"),
                         "diff": describe(c["genome"], ch["genome"])} for c in reg["challengers"]],
        "retired": [{"id": c["id"], "retired": c.get("retired"), "status": c.get("status")} for c in reg["retired"][-8:]],
        "proposal": proposals[0]["id"] if proposals and not promoted else None,
        "promoted": promoted, "generations": glog,
        "history": (pd.read_csv(HIST).to_dict("records") if os.path.exists(HIST) else []),
        "meta": ({k: meta_res.get(k) for k in ["generated", "refits", "step", "n_eval", "period", "results", "tests", "best",
                                              "trusted", "trusted_mode", "passed", "alpha", "modes", "seeds", "spread", "rule", "curve"]}
                 | {"log": meta_res["log"][-12:]}) if meta_res else None,
        "mode": {"key": mname, "label": MODES[mname]["label"]},
        "continuous": {k: (popst or {}).get(k) for k in ("generations", "runs", "updated", "best_adj")}
                      | {"pop": len((popst or {}).get("pop") or []), "every": "2시간"},
        "weekly": (pd.read_csv(os.path.join(OUT_DIR, "weekly.csv")).tail(26).to_dict("records")
                   if os.path.exists(os.path.join(OUT_DIR, "weekly.csv")) else []),
        "rules": {"hyst": HYST, "pop": 48, "gens": 14, "fitness": "해마다 Sharpe 평균 − 0.5 × 표준편차 (거래 20건 미만 제외)",
                  "long_forward_days": int((cfg.get("evolve") or {}).get("long_forward_days", 250))},
    }
    json.dump(out, open(OUT_UI, "w", encoding="utf-8"), ensure_ascii=False, default=_js)
    print(f"저장: {REG}, {OUT_UI}")


# ══════════════════════════════════════════════════════════════
def main():
    global D
    cmd = sys.argv[1] if len(sys.argv) > 1 else "weekly"
    cfg = settings.load()
    D = load()
    if cmd == "check":
        check(cfg)
        return
    if cmd == "promote":
        reg = load_reg(cfg)
        promote(reg, sys.argv[2])
        json.dump(reg, open(REG, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=_js)
        return
    ctx = mp.get_context("fork")
    with ctx.Pool(max(1, (os.cpu_count() or 2))) as pool:
        if cmd in ("meta", "all"):
            os.makedirs(OUT_DIR, exist_ok=True)
            m = meta_multi(cfg, pool)
            json.dump(m, open(META, "w", encoding="utf-8"), ensure_ascii=False, default=_js)
            sp = m["spread"]
            print(f"시드 {m['seeds']} — Sharpe 평균 [최소~최대]")
            for k, nm in list(m["modes"].items()) + [("static", "지금 설정 고정"), ("naive", "무학습 기준")]:
                x = sp[k]
                print(f"  {nm:<22} 전체 {x['all']['mean']:5.2f} [{x['all']['min']:5.2f}~{x['all']['max']:5.2f}] │ "
                      f"~2023 {x['pre']['mean']:5.2f} │ 2024~ {x['post']['mean']:5.2f}"
                      + (f" │ P 평균 {x['p_mean']}" if "p_mean" in x else ""))
            print(f"  적응 신뢰: {m['trusted']} {m['passed']} · 문턱 P<{m['alpha']} · 평가 {m['n_eval']}개")
        if cmd in ("weekly", "all"):
            weekly(cfg, pool)
        if cmd == "step":
            # 한 번 실행에 쓰는 시간 (LEARN_MINUTES, 기본 0 = 20세대 한 번).
            # GitHub 예약은 2시간마다 걸어도 실제로는 하루 4~6번만 돈다 → 한 번 돌 때 시간만큼 세대를 더 돌린다.
            budget = float(os.environ.get("LEARN_MINUTES", "0") or 0) * 60
            t0 = time.time()
            weekly(cfg, pool, pop=48, gens=20, label="연속 진화")
            spent = time.time() - t0
            left = budget - spent
            if left > spent * 1.5:
                gens = int(min(400, 20 * left / max(spent, 1) * 0.8))
                print(f"남은 시간 {left / 60:.0f}분 → {gens}세대 더")
                weekly(cfg, pool, pop=48, gens=gens, label="연속 진화")


if __name__ == "__main__":
    main()
