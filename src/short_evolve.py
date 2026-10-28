"""
단기 모델 진화 — 쉬지 않고 새 단기 모델을 만들고, '태어난 뒤' 성적으로만 시험한다.
learn.yml 이 2시간마다(실제로는 GitHub 예약 사정으로 하루 4~6번) `python src/short_evolve.py step` 을 돌린다.
LEARN_MINUTES(기본 15분)만큼 세대를 이어 간다.

유전자 (short_lab 의 종목 모델과 같은 시뮬레이터)
  신호 가중치  : 13개 신호(short_panel.SIGNAL_INFO) 중 1~4개, 가중치 0.5·1·1.5·2
  종목 수 5·10 · 보유 5·10일 · 시장 필터(항상·120일선 위·급락장 제외)
  — 1~3일 보유는 검증에서 비용 때문에 거의 모두 손실이라 진화 대상에서 뺐다 (전략서 3-6-5)

적합도 = 해마다 샤프 평균 − 0.5 × 해마다 샤프 표준편차 − 0.02 × (신호 수 − 1)   (매수 50회 미만 탈락)
         + 0.3 × min(0, 가장 나빴던 시장 국면의 샤프)   ← 국면 견고성 (2026-10-28 추가: '어느 시장에서도 크게 잃지 않게')
         국면 = 강세·횡보·약세·급락 (allweather.py, KODEX 200 종가로 그날 저녁 판정)

인구(data/short/evo_population.json)를 저장해 두고 매번 이어서 몇 세대씩 진화한다 → 세대가 계속 쌓인다.

등록 · 시험 (과최적화 막기)
  고정 후보(5·10일 보유 종목 모델 132개) 중 최고보다 적합도가 0.05 이상 높고, 기존 진화 후보와 다르면
  '진화 후보'로 등록한다(data/short/evolved.json, 등록일 = 그날 자료 날짜).
  단기 모의투자(short_trader)는 진화 후보의 성적을 **등록일 다음 날부터만** 센다
  → 선택 규칙(최근 250일 샤프)이 요구하는 125거래일(약 6개월)이 쌓여야 고를 수 있다. 사후에 고른 성적은 끼어들 수 없다.
  등록 뒤 250거래일이 지났는데 그 뒤 샤프가 0 이하면 물러난다. 동시에 시험 중인 후보는 최대 20개,
  새 후보는 자료 하루에 하나(가중치만 다른 것은 같은 모델). 자리가 차면 60일 이상 시험해 샤프가 0 미만인 가장 나쁜 후보를 내보낸다.

  python src/short_evolve.py step     # 한 번 (인구 24 × 6세대)
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import short_lab as L  # noqa: E402
import short_panel as SP  # noqa: E402

DIR = "data/short"
POPF = os.path.join(DIR, "evo_population.json")
EVOF = os.path.join(DIR, "evolved.json")
UI = "docs/short_evo.json"
SIGS = list(SP.SIGNAL_INFO)
WEIGHTS = [0.5, 1.0, 1.5, 2.0]
CH = {"K": [5, 10], "hold": ["H5", "H10"], "gate": list(L.GATES)}
POP, GENS, ELITE = 24, 6, 6
MARGIN = 0.05
MAX_ACTIVE = 20
INCUBATE = 125           # 선택 규칙(250일 샤프)이 요구하는 최소 유효일 = L/2
RETIRE_AFTER = 250


def gkey(g):
    return json.dumps({"w": sorted(g["w"].items()), "K": g["K"], "hold": g["hold"], "gate": g["gate"]})


def random_genome(rng):
    n = int(rng.integers(1, 5))
    sig = rng.choice(SIGS, n, replace=False)
    return {"w": {str(k): float(rng.choice(WEIGHTS)) for k in sig},
            "K": int(rng.choice(CH["K"])), "hold": str(rng.choice(CH["hold"])), "gate": str(rng.choice(CH["gate"]))}


def mutate(g, rng):
    c = {"w": dict(g["w"]), "K": g["K"], "hold": g["hold"], "gate": g["gate"]}
    for _ in range(int(rng.integers(1, 3))):
        r = rng.random()
        if r < 0.45:                                          # 가중치 바꾸기
            k = str(rng.choice(list(c["w"])))
            c["w"][k] = float(rng.choice(WEIGHTS))
        elif r < 0.65 and len(c["w"]) < 4:                    # 신호 더하기
            k = str(rng.choice([s for s in SIGS if s not in c["w"]]))
            c["w"][k] = float(rng.choice(WEIGHTS))
        elif r < 0.8 and len(c["w"]) > 1:                     # 신호 빼기
            c["w"].pop(str(rng.choice(list(c["w"]))))
        else:                                                 # 매매 규칙 하나
            f = str(rng.choice(list(CH)))
            c[f] = type(c[f])(rng.choice(CH[f]))
    return c


def crossover(a, b, rng):
    keys = list(set(a["w"]) | set(b["w"]))
    rng.shuffle(keys)
    w = {}
    for k in keys[:4]:
        src = a["w"] if (k in a["w"] and (k not in b["w"] or rng.random() < 0.5)) else b["w"]
        w[k] = src[k]
    return {"w": w or dict(a["w"]), **{f: (a if rng.random() < 0.5 else b)[f] for f in CH}}


def to_model(g, mid="X?", born=None):
    m = {"id": mid, "kind": "stock", "sel": "EVO", "K": int(g["K"]), "hold": g["hold"], "gate": g["gate"],
         "combo": [(k, float(v)) for k, v in sorted(g["w"].items())]}
    if born:
        m["born"] = born
    return m


FIT_VER = 2              # 2: 국면 견고성 감점 추가 (2026-10-28) — 바뀌면 문턱(고정 후보 최고)을 다시 잰다
REG_PEN = 0.3            # 네 국면 중 가장 나빴던 국면의 샤프가 마이너스면 그 0.3배를 뺀다


def regimes(D):
    """패널 날짜별 시장 국면 (allweather 정의, KODEX 200 종가 기준). 한 번만 계산해 D 에 붙여 둔다."""
    if getattr(D, "_reg", None) is None:
        try:
            import allweather as AW
            dates, Op, Cp = AW.load_prices()
            F = AW.features(Cp, dates)
            D._reg = pd.Series(F["reg"], index=dates).reindex(D.dates).ffill().fillna(-1).to_numpy().astype(int)
        except Exception:                                 # ETF 자료가 없으면 감점 없이
            D._reg = np.full(D.T, -1)
    return D._reg


def regime_sharpes(D, r):
    reg = regimes(D)
    prev = np.r_[-1, reg[:-1]]
    out = {}
    for k in range(4):
        x = r[prev == k]
        if len(x) >= 60 and x.std() > 0:
            out[k] = float(x.mean() / x.std() * np.sqrt(252))
    return out


def fitness(D, g):
    """해마다 샤프 평균 − 0.5 × 표준편차 − 신호 수 벌점 + 국면 견고성(가장 나쁜 국면 샤프가 마이너스면 감점)."""
    res = L.run_stock(D, to_model(g))
    E = res["E"]
    rr = L.rets(E)
    r = pd.Series(rr, index=D.dates)
    yrs = [x for _, x in r.groupby(D.dates.year) if len(x) >= 60]
    sh = [x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0 for x in yrs]
    st = L.stats(E, D.dates)
    if res["buys"] < 50 or not sh:
        return {"fit": -9.0, "sharpe": st["sharpe"] if st else None, "buys": res["buys"]}
    rs = regime_sharpes(D, rr)
    worst = min(rs.values()) if rs else 0.0
    fit = float(np.mean(sh) - 0.5 * np.std(sh) - 0.02 * (len(g["w"]) - 1) + REG_PEN * min(0.0, worst))
    return {"fit": round(fit, 4), "sharpe": st["sharpe"], "cagr": st["cagr"], "mdd": st["mdd"], "buys": int(res["buys"]),
            "worst_regime": round(worst, 3)}


def fixed_best(D):
    """고정 후보(5·10일 보유 종목 모델)의 최고 적합도 — 진화 후보가 넘어야 할 문턱."""
    best, who = -9.0, None
    for m in L.stock_models():
        if m["hold"] not in ("H5", "H10"):
            continue
        g = {"w": {k: float(w) for k, w in L.SELECTORS[m["sel"]]}, "K": m["K"], "hold": m["hold"], "gate": m["gate"]}
        f = fitness(D, g)["fit"]
        if f > best:
            best, who = f, m["id"]
    return best, who


def load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def models_from_registry(reg=None):
    """short_trader 가 후보 목록에 더하는 진화 모델 (시험 중인 것만)."""
    reg = reg if reg is not None else load(EVOF, {"candidates": []})
    return [to_model(c["genome"], c["id"], c["born"]) for c in reg.get("candidates", [])]


def forward_stats(D, m):
    """등록일 다음 날부터의 성적 (진화 후보 모델의 '태어난 뒤' 성적)."""
    E = L.run_stock(D, m)["E"]
    b = int(np.searchsorted(D.dates, pd.Timestamp(m["born"]), side="right"))
    n = D.T - b
    if n < 2:
        return {"days": max(0, n), "ret": 0.0, "sharpe": None}
    r = L.rets(E)[b:]
    return {"days": int(n), "ret": round(float(E[-1] / E[b - 1] - 1), 4),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3) if r.std() > 0 else 0.0}


def step(pop=POP, gens=GENS, seed=None):
    t0 = time.time()
    os.makedirs(DIR, exist_ok=True)
    D = L.Data()
    today = str(D.dates[-1].date())
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % (2 ** 32))
    st = load(POPF, {"generations": 0, "runs": 0, "pop": []})
    reg = load(EVOF, {"candidates": [], "retired": []})
    if st.get("bench_date") != today or st.get("fit_ver") != FIT_VER:   # 문턱은 자료나 적합도 정의가 바뀔 때만 다시 잰다
        st["bench_fit"], st["bench_id"] = fixed_best(D)
        st["bench_date"] = today
        st["fit_ver"] = FIT_VER
    cache = {}

    def ev(g):
        k = gkey(g)
        if k not in cache:
            cache[k] = (g, fitness(D, g))
        return cache[k][1]

    popl = [g for g in st.get("pop", [])][:pop] + [c["genome"] for c in reg["candidates"]]
    while len(popl) < pop:
        popl.append(random_genome(rng))
    popl = list({gkey(g): g for g in popl}.values())
    log = []
    for gen in range(gens):
        for g in popl:
            ev(g)
        ranked = sorted(popl, key=lambda g: -ev(g)["fit"])
        elites = ranked[:ELITE]
        log.append({"gen": int(st.get("generations", 0)) + gen + 1, "best": ev(ranked[0])["fit"],
                    "median": float(np.median([ev(g)["fit"] for g in popl]))})
        kids, tries = [], 0
        while len(kids) < pop - len(elites) and tries < pop * 20:
            tries += 1
            if rng.random() < 0.3:
                i, j = rng.choice(len(elites), 2, replace=False)
                kid = crossover(elites[i], elites[j], rng)
            elif rng.random() < 0.15:
                kid = random_genome(rng)                     # 가끔 완전히 새로운 것 (다양성)
            else:
                kid = mutate(elites[int(rng.integers(len(elites)))], rng)
            if gkey(kid) not in cache and all(gkey(kid) != gkey(x) for x in kids):
                kids.append(kid)
        popl = elites + kids
    for g in popl:
        ev(g)
    ranked = sorted(cache.values(), key=lambda x: -x[1]["fit"])
    best, bf = ranked[0]
    st.update({"generations": int(st.get("generations", 0)) + gens, "runs": int(st.get("runs", 0)) + 1,
               "updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "data_date": today,
               "best_fit": bf["fit"], "pop": [g for g, _ in ranked[:pop]], "log": (st.get("log", []) + log)[-120:]})
    # 등록
    new_id = None
    shape = lambda g: (tuple(sorted(g["w"])), g["K"], g["hold"], g["gate"])   # noqa: E731 — 가중치만 다른 것은 같은 모델로 본다
    seen = {shape(c["genome"]) for c in reg["candidates"] + reg.get("retired", [])}
    born_today = any(c["born"] == today for c in reg["candidates"])            # 자료 하루에 새 후보는 하나만
    if len(reg["candidates"]) >= MAX_ACTIVE:                                    # 자리가 차면 증거로 내보낸다
        bad = [c for c in reg["candidates"] if (c.get("forward") or {}).get("days", 0) >= 60
               and ((c["forward"].get("sharpe") or 0) < 0)]
        if bad:
            out = min(bad, key=lambda c: c["forward"]["sharpe"])
            reg.setdefault("retired", []).append({**out, "retired": today, "why": "자리 비우기 (태어난 뒤 샤프 < 0)"})
            reg["candidates"] = [c for c in reg["candidates"] if c["id"] != out["id"]]
    if (bf["fit"] > st["bench_fit"] + MARGIN and shape(best) not in seen and not born_today
            and len(reg["candidates"]) < MAX_ACTIVE):
        used = {c["id"] for c in reg["candidates"] + reg.get("retired", [])}
        base = "X" + today.replace("-", "")[2:]
        new_id = base + "a"
        while new_id in used:
            new_id = base + chr(ord(new_id[-1]) + 1)
        reg["candidates"].append({"id": new_id, "born": today, "genome": best, "fit_born": bf,
                                  "beat": {"id": st["bench_id"], "fit": st["bench_fit"]}})
    # 시험 중인 후보의 '태어난 뒤' 성적 · 물러나기
    keep = []
    for c in reg["candidates"]:
        c["forward"] = forward_stats(D, to_model(c["genome"], c["id"], c["born"]))
        f = c["forward"]
        c["eligible"] = f["days"] >= INCUBATE
        if f["days"] >= RETIRE_AFTER and (f["sharpe"] is None or f["sharpe"] <= 0):
            reg.setdefault("retired", []).append({**c, "retired": today})
        else:
            keep.append(c)
    reg["candidates"] = keep
    reg["retired"] = reg.get("retired", [])[-40:]
    json.dump(st, open(POPF, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(reg, open(EVOF, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ui = {"updated": st["updated"], "data_date": today, "generations": st["generations"], "runs": st["runs"],
          "best_fit": bf["fit"], "best_desc": L.describe(to_model(best, "최선")), "bench": {"id": st["bench_id"], "fit": st["bench_fit"]},
          "incubate_days": INCUBATE, "retire_after": RETIRE_AFTER, "new": new_id, "log": st["log"][-40:],
          "candidates": [{"id": c["id"], "born": c["born"], "desc": L.describe(to_model(c["genome"], c["id"])),
                          "fit_born": c["fit_born"], "forward": c["forward"], "eligible": c["eligible"]} for c in reg["candidates"]],
          "retired": [{"id": c["id"], "born": c["born"], "retired": c.get("retired"), "forward": c.get("forward")}
                      for c in reg.get("retired", [])[-10:]]}
    json.dump(ui, open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"단기 진화: {len(cache)}개 평가 · 누적 {st['generations']}세대 · 최선 {bf['fit']:.3f} (샤프 {bf['sharpe']}) "
          f"· 고정 후보 최고 {st['bench_fit']:.3f} ({st['bench_id']}) · 시험 중 {len(reg['candidates'])}개"
          + (f" · 새 후보 {new_id}" if new_id else "") + f" · {time.time() - t0:.0f}s")
    return ui


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "step"
    if cmd == "step":
        # LEARN_MINUTES 만큼 세대를 이어서 (기본 0 = 6세대 한 번) — learn.yml 이 정한다
        budget = float(os.environ.get("LEARN_MINUTES", "0") or 0) * 60
        t0 = time.time()
        step()
        spent = time.time() - t0
        if budget - spent > spent * 1.5:
            g = int(min(200, GENS * (budget - spent) / max(spent, 1) * 0.8))
            print(f"남은 시간 {(budget - spent) / 60:.0f}분 → {g}세대 더")
            step(gens=g)
