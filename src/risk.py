"""
리스크 매니저 (거부권) — 모델이 낸 내일 주문을 '저장하기 직전'에 검사해 줄이거나 막는다. 모델은 이 판정을 바꿀 수 없다.
  근거: ai-hedge-fund VISION("analysts cannot override risk limits") · TradingAgents Portfolio Manager · nautilus_trader RiskEngine
        (ACTIVE/REDUCING/HALTED) · zipline TradingControl · pysystemtrade risk overlay(≤1배) · Knight Capital 2012(경보만 보내지 말고 자동 정지)

  계좌 상태 (전체)
    HALTED   : config/KILL 이 있거나 주문 수가 비정상(60건 초과) → 주문 전부 취소
    REDUCING : config/PAUSE 가 있거나 자료가 2영업일 넘게 묵음 → 새로 사는 주문만 취소 (파는 주문은 그대로)
  몫(단기·전천후·미국)별 감시 — 실제 일수익을 '백테스트 기대치의 절반'(SR*, Carver·Quantopian 888 연구)과 비교
    감쇠 감시 CUSUM (Page 1954, Philips·Yashchin·Stein 2003): 7년에 한 번꼴 오경보로 문턱을 정한다
    낙폭 감시: 지금 낙폭을 'SR*·σ* 로 같은 기간 굴렸을 때 낙폭 분포'(꼬리 두꺼운 t분포 2000번)와 비교 (95%·99% 밖)
    처분 (config risk_manager.decay_action / drawdown_action) — 기본 '경고만':
        과거 재현(replay)에서 경보 뒤 자동 축소·정지는 전천후엔 중립, 단기 규칙·지수 보유엔 손해였다
        (단기 규칙 샤프 0.155→0.042, 지수 보유 0.457→0.358 — 손실 뒤 줄이면 되돌아오는 구간을 놓친다; Carver 2015, Lo 2017 과 같은 결론).
        그래서 경보는 투자위원회 기록에 '검토 필요'로 올리고, 크기는 사람이 정한다. half = 경보 뒤 2년간 ×0.5 · stop = 99% 낙폭이면 새 매수 중단
  주문별 검사
    한 종목 매수 ≤ 계좌의 25%(주식) / 60%(ETF, 단기채 ETF 는 예외) · 매수 수량 ≤ 최근 20일 평균 거래량의 2.5% (FINSABER)
    체결 때(settle) 가격 보호: 시가가 계획 가격보다 12% 넘게 높으면 그 매수는 하지 않는다
  쓰지 않은 것: 변동성 맞추기(가설 검증에서 샤프를 못 올림) · 손절 촘촘히(되돌림 전략엔 해로움, Kaminski·Lo 2014) · 연속 손실 잠금

  python src/risk.py   → 지금 상태 점검만 출력
"""
import json
import math
import os

import numpy as np
import pandas as pd

KILL, PAUSE = "config/KILL", "config/PAUSE"
MAX_ORDERS = 60
CAP_STOCK, CAP_ETF = 0.25, 0.60
CASH_LIKE = {"153130"}
ADV_CAP = 0.025
COLLAR = 0.12
ARL0_DAYS = 252 * 7               # 오경보 평균 7년에 한 번
HAIRCUT = 0.5                     # 실제 기대 샤프 = 백테스트 × 0.5


def jload(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════
# 감쇠 감시 (CUSUM)
# ══════════════════════════════════════════════════════════════
def cusum_h(m, arl0=ARL0_DAYS):
    """Siegmund 근사로 문턱 h: ARL0 = (e^b − b − 1)/(m²/2), b = h + 1.166·m (z 단위 증분 m(m/2 − z), H0 에서 z~N(m,1))."""
    target = arl0 * m * m / 2
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        b = (lo + hi) / 2
        if math.exp(b) - b - 1 < target:
            lo = b
        else:
            hi = b
    b = (lo + hi) / 2
    return max(b - 1.166 * m, 1e-6), b


def cusum(r, sr_star, vol_star):
    """r: 실제 일수익. 반환: 지금 누적값·문턱·경보 횟수(경보 뒤 0 으로 다시)·기대가 0 이면 알아채는 데 걸리는 평균 일수."""
    m = sr_star / math.sqrt(252)
    sd = vol_star / math.sqrt(252)
    h, b = cusum_h(m)
    S, alarms, last = 0.0, 0, None
    for i, x in enumerate(np.asarray(r, float)):
        if not np.isfinite(x):
            continue
        z = x / sd
        S = max(0.0, S + m * (m / 2 - z))
        if S > h:
            alarms += 1
            last = i
            S = 0.0
    arl1 = (math.exp(-b) + b - 1) / (m * m / 2)
    return {"S": round(float(S), 4), "h": round(float(h), 4), "level": round(float(S / h), 3), "alarms": alarms, "last_alarm_idx": last,
            "detect_days_if_zero": int(arl1), "false_alarm_years": round(ARL0_DAYS / 252, 1)}


# ══════════════════════════════════════════════════════════════
# 낙폭 감시
# ══════════════════════════════════════════════════════════════
def mdd(r):
    E = np.cumprod(1 + np.nan_to_num(np.asarray(r, float)))
    return float((E / np.maximum.accumulate(np.r_[1.0, E])[1:] - 1).min()) if len(E) else 0.0


def dd_band(n, sr_star, vol_star, sims=2000, seed=7, df=4):
    """기대 성적(SR*, σ*)으로 n일 굴렸을 때 최대 낙폭 분포의 95%·99% 지점 (t분포 꼬리)."""
    if n < 2:
        return {"p95": 0.0, "p99": 0.0}
    rng = np.random.default_rng(seed)
    sd = vol_star / math.sqrt(252)
    mu = sr_star / math.sqrt(252) * sd
    e = rng.standard_t(df, (sims, n)) * math.sqrt((df - 2) / df)
    R = mu + sd * e
    E = np.cumprod(1 + R, 1)
    peak = np.maximum.accumulate(np.c_[np.ones(sims), E], 1)[:, 1:]
    m = (E / peak - 1).min(1)
    return {"p95": round(float(np.percentile(m, 5)), 4), "p99": round(float(np.percentile(m, 1)), 4)}


def monitor(r, sr_star, vol_star, decay_action="alert", drawdown_action="alert", memory=504):
    """한 몫의 감시 결과와 처분 (크기 배수·상태). 경보는 최근 memory 거래일 안의 것만 센다."""
    r = pd.Series(r).dropna().to_numpy(float)
    out = {"days": int(len(r)), "sr_star": round(sr_star, 3), "vol_star": round(vol_star, 4)}
    if len(r) < 5 or not (sr_star > 0 and vol_star > 0):
        out.update({"action": "ACTIVE", "size": 1.0, "why": ["실제 기록이 아직 짧음 — 감시 대기"], "cusum": None, "dd": None})
        return out
    c = cusum(r, sr_star, vol_star)
    recent = c["alarms"] > 0 and c["last_alarm_idx"] is not None and len(r) - c["last_alarm_idx"] <= memory
    d = {"now": round(mdd(r), 4), **dd_band(len(r), sr_star, vol_star)}
    act, size, why, review = "ACTIVE", 1.0, [], False
    if recent:
        review = True
        if decay_action == "half":
            size = 0.5
            why.append("감쇠 경보 — 기대보다 꾸준히 못함 → 새 매수 크기 절반 (설정 half)")
        else:
            why.append("감쇠 경보 — 기대보다 꾸준히 못함 → 검토 필요 (경고만: 과거 재현에서 자동 축소가 손해)")
    if d["now"] < d["p99"]:
        review = True
        if drawdown_action == "stop":
            act, size = "REDUCING", 0.0
            why.append(f"낙폭 {d['now'] * 100:.1f}% — 기대 성적이면 100번 중 1번도 안 나올 수준(<{d['p99'] * 100:.1f}%) → 새 매수 중단 (설정 stop)")
        else:
            why.append(f"낙폭 {d['now'] * 100:.1f}% — 기대 범위 99% 밖(<{d['p99'] * 100:.1f}%) → 검토 필요 (경고만)")
    elif d["now"] < d["p95"]:
        why.append(f"낙폭 {d['now'] * 100:.1f}% — 기대 범위 95% 밖(<{d['p95'] * 100:.1f}%) · 경고만")
    if not why:
        why.append(f"정상 — 감쇠 누적 {c['level'] * 100:.0f}% (100% 면 경보) · 낙폭 {d['now'] * 100:.1f}% (경고선 {d['p95'] * 100:.1f}%)")
    out["review"] = review
    out.update({"action": act, "size": size, "why": why, "cusum": c, "dd": d})
    return out


# ══════════════════════════════════════════════════════════════
# 기대치 (백테스트) · 실제 일수익
# ══════════════════════════════════════════════════════════════
def expectations():
    """몫별 백테스트 샤프·변동성 → 기대 SR* = 백테스트 × 0.5 (긴 역사 표본 외가 더 낮으면 그것)."""
    ex = {}
    lab = jload("docs/short_lab.json", {}) or {}
    a = ((lab.get("rules") or {}).get("live") or {}).get("all") or {}
    if a.get("sharpe"):
        ex["live"] = {"sr_bt": a["sharpe"], "vol": a["vol"], "src": "short_lab 채택 규칙 2019~"}
    s = jload("docs/short.json", {}) or {}
    b = ((s.get("aw") or {}).get("backtest") or {}).get("stats") or {}
    if b.get("sharpe"):
        ex["aw"] = {"sr_bt": b["sharpe"], "vol": b["vol"], "src": "전천후 혼합 2016~"}
    h = jload("docs/history.json", {}) or {}
    oos = ((h.get("aw") or {}).get("oos") or {}).get("sharpe")
    if "aw" in ex and oos:
        ex["aw"]["sr_oos"] = oos
    u = jload("docs/us_lab.json", {}) or {}
    for m in u.get("top_models") or []:
        if m.get("id") == "S:MOM:10:BAND:none" and (m.get("full") or {}).get("sharpe"):
            ex["us"] = {"sr_bt": m["full"]["sharpe"], "vol": m["full"]["vol"], "src": "미국 추세 상위 10 2016~"}
    if "us" not in ex:
        sc = (jload("docs/science.json", {}) or {}).get("overfit", {}).get("us") or {}
        if sc.get("sharpe"):
            ex["us"] = {"sr_bt": sc["sharpe"], "vol": 0.2, "src": "과최적화 진단 (변동성 20% 가정)"}
    for v in ex.values():
        v["sr_star"] = round(min(HAIRCUT * v["sr_bt"], v.get("sr_oos", 9)), 3)
    return ex


def flow_adjusted(eq, flows=None):
    """평가액 열(날짜 index) → 일수익. flows: [{after: 날짜, amount: +들어옴/−나감}] — 몫 사이 옮긴 돈은 수익에서 뺀다."""
    eq = eq.dropna()
    r = eq.pct_change()
    for f in flows or []:
        nxt = eq.index[eq.index > str(f["after"])]
        if len(nxt):
            d = nxt[0]
            i = eq.index.get_loc(d)
            if i > 0:
                r.iloc[i] = (eq.iloc[i] - f["amount"]) / eq.iloc[i - 1] - 1
    return r.iloc[1:]


# ══════════════════════════════════════════════════════════════
# 계좌 상태 · 주문 검사
# ══════════════════════════════════════════════════════════════
def conf(c=None):
    """config/autotrade.yaml 의 risk_manager: (없으면 기본값)."""
    c = dict(c or {})
    return {"decay_action": c.get("decay_action", "alert"), "drawdown_action": c.get("drawdown_action", "alert"),
            "max_orders": int(c.get("max_orders", MAX_ORDERS)), "cap_stock": float(c.get("cap_stock", CAP_STOCK)),
            "cap_etf": float(c.get("cap_etf", CAP_ETF)), "adv_cap": float(c.get("adv_cap", ADV_CAP)), "collar": float(c.get("collar", COLLAR))}


def global_state(data_date, n_orders=0, today=None, max_orders=MAX_ORDERS):
    why = []
    state = "ACTIVE"
    if os.path.exists(KILL):
        return "HALTED", ["config/KILL — 모든 주문 중단"]
    if n_orders > max_orders:
        return "HALTED", [f"주문 {n_orders}건 — 비정상(한도 {max_orders}) → 모두 취소하고 점검"]
    if os.path.exists(PAUSE):
        state = "REDUCING"
        why.append("config/PAUSE — 새 매수 중단 (매도는 그대로)")
    today = pd.Timestamp(today or (pd.Timestamp.now(tz="Asia/Seoul") - pd.Timedelta(hours=9)).date())
    lag = len(pd.bdate_range(pd.Timestamp(data_date) + pd.Timedelta(days=1), today)) if pd.Timestamp(data_date) < today else 0
    if lag > 2:
        state = "REDUCING"
        why.append(f"자료가 {lag}영업일 묵음 ({pd.Timestamp(data_date).date()}) — 새 매수 중단")
    return state, why or ["정상"]


def review(orders, state, sleeves, equity_total, adv=None, kind_of=None, value_of=None, cf=None):
    """주문 목록 검사. sleeves: {몫 이름: monitor 결과}. adv(code)→20일 평균 거래량(주), value_of(order)→매수 금액.
    반환: (통과 주문, 기록 [{code, side, act, why}])."""
    cf = cf or conf()
    log = []
    out = []
    for o in orders:
        sl = o.get("sleeve") or "live"
        mon = sleeves.get(sl) or {"action": "ACTIVE", "size": 1.0}
        if state == "HALTED":
            log.append({"code": o["code"], "side": o["side"], "sleeve": sl, "act": "취소", "why": "계좌 정지(HALTED)"})
            continue
        if o["side"] == "sell":
            out.append(o)
            continue
        if state == "REDUCING" or mon["action"] == "REDUCING":
            log.append({"code": o["code"], "side": "buy", "sleeve": sl, "act": "취소",
                        "why": "계좌 매도만(REDUCING)" if state == "REDUCING" else "몫 매도만 — " + "; ".join(mon.get("why", []))})
            continue
        o = dict(o)
        k = (kind_of or (lambda x: x.get("kind", "stock")))(o)
        val = (value_of or (lambda x: x.get("qty", 0) * x.get("est_price", 0)))(o)
        scale = float(mon.get("size", 1.0))
        notes = []
        if scale < 1:
            notes.append(f"크기 ×{scale:g} (감쇠 경보)")
        lim = cf["cap_stock"] if k == "stock" else cf["cap_etf"]
        cap = None if o["code"] in CASH_LIKE else lim * equity_total
        if cap and val * scale > cap:
            scale = cap / val
            notes.append(f"한 종목 한도 {int(lim * 100)}% 로 축소")
        if adv and k == "stock":
            a = adv(o["code"])
            px = o.get("est_price") or 0
            q = val * scale / px if px else 0
            if a and a > 0 and q > cf["adv_cap"] * a:
                scale *= cf["adv_cap"] * a / q
                notes.append(f"거래량 한도(20일 평균의 {cf['adv_cap'] * 100:g}%) 로 축소")
        if scale < 1:
            if "qty" in o:
                o["qty"] = int(math.floor(o["qty"] * scale))
            if "usd" in o:
                o["usd"] = round(o["usd"] * scale, 2)
            if o.get("qty", 1) <= 0 or o.get("usd", 1) <= 0:
                log.append({"code": o["code"], "side": "buy", "sleeve": sl, "act": "취소", "why": " · ".join(notes)})
                continue
            log.append({"code": o["code"], "side": "buy", "sleeve": sl, "act": "축소", "why": " · ".join(notes)})
        out.append(o)
    return out, log


def collar_ok(o, px, collar=COLLAR):
    """체결 때 가격 보호 — 매수만. 시가가 계획 가격보다 12% 넘게 높으면 하지 않는다."""
    if o.get("side") != "buy" or not o.get("est_price") or not px:
        return True
    return px <= float(o["est_price"]) * (1 + collar)


def apply(plan, data_date, sleeve_returns, equity_total, adv=None, kind_of=None, value_of=None, today=None, cfg=None):
    """계획(plan['orders'])을 검사해 고치고 plan['risk'] 에 기록. sleeve_returns: {몫: 실제 일수익 Series}."""
    cf = conf(cfg)
    ex = expectations()
    sleeves = {}
    for sl, r in sleeve_returns.items():
        e = ex.get(sl)
        sleeves[sl] = (monitor(r, e["sr_star"], e["vol"], cf["decay_action"], cf["drawdown_action"]) if e
                       else {"action": "ACTIVE", "size": 1.0, "why": ["기대치 없음"]})
        if e:
            sleeves[sl]["expect"] = e
    state, why = global_state(data_date, len(plan.get("orders") or []), today, cf["max_orders"])
    before = len(plan.get("orders") or [])
    plan["orders"], log = review(plan.get("orders") or [], state, sleeves, equity_total, adv, kind_of, value_of, cf)
    plan["risk"] = {"state": state, "why": why, "sleeves": sleeves, "log": log, "orders_before": before,
                    "orders_after": len(plan["orders"]), "limits": {**cf, "haircut": HAIRCUT, "false_alarm_years": ARL0_DAYS / 252}}
    return plan


def fail_closed(plan, ex):
    """검사 자체가 실패하면 새 매수는 막는다 (nautilus: reconciliation fails closed)."""
    plan["orders"] = [o for o in plan.get("orders") or [] if o.get("side") == "sell"]
    plan["risk"] = {"state": "REDUCING", "why": [f"리스크 검사 실패({type(ex).__name__}: {ex}) — 새 매수 보류"], "sleeves": {}, "log": []}
    return plan


def replay(r, sr_star, vol_star, mode="half", memory=504, dd_rule=False, every=21):
    """과거 일수익에 감시를 '그날까지 자료로만' 적용해 본다. 경보는 최근 memory 일 안의 것만 센다.
    mode: 'alert' = 크기 안 바꿈 · 'half' = 최근 경보 있으면 ×0.5 · 'stop' = 1번 ×0.5, 2번 0.
    dd_rule: 낙폭 99% 밖이면 0 (회복하면 다시). 반환: 관리한 일수익, 크기 열, 경보 날짜."""
    r = pd.Series(r).dropna()
    x = r.to_numpy(float)
    m = sr_star / math.sqrt(252)
    sd = vol_star / math.sqrt(252)
    h, _ = cusum_h(m)
    S, al, out, sizes, when = 0.0, [], np.zeros(len(x)), np.ones(len(x)), []
    dd_off = False
    E, peak = 1.0, 1.0
    for i, v in enumerate(x):
        n = sum(1 for a in al if i - a <= memory)
        size = 1.0
        if mode == "half" and n >= 1:
            size = 0.5
        elif mode == "stop":
            size = 0.0 if n >= 2 else (0.5 if n == 1 else 1.0)
        if dd_off:
            size = 0.0
        sizes[i] = size
        out[i] = size * v
        S = max(0.0, S + m * (m / 2 - v / sd))           # 감시는 '원래 전략'의 성적으로 (그림자)
        if S > h:
            al.append(i)
            S = 0.0
            when.append(str(r.index[i].date()) if hasattr(r.index[i], "date") else i)
        E *= 1 + v
        peak = max(peak, E)
        if dd_rule and i % every == 0 and i > 20:
            dd_off = (E / peak - 1) < dd_band(min(i + 1, 2520), sr_star, vol_star, sims=400)["p99"]
    return pd.Series(out, index=r.index), sizes, when


if __name__ == "__main__":
    print(json.dumps(expectations(), ensure_ascii=False, indent=1))
    print(global_state("2026-10-30"))
    for sr in (0.5, 0.6):
        print(sr, cusum(np.zeros(10), sr, 0.2))
