"""
투자위원회 — 월가 'AI 트레이딩 팀' 구조(TradingAgents: 분석가 → 강세·약세 연구원 → 트레이더 → 리스크 → 펀드매니저)를
            LLM 없이 '정해진 규칙'으로 옮긴 매일 저녁 결정 기록. 같은 자료면 언제 돌려도 같은 결론이 나온다.

  왜 LLM 이 아닌가: LLM 팀은 학습 자료에 시험 기간이 들어 있어(Profit Mirage 2025: 마감일 뒤 성과 약 절반) 백테스트가 부풀고,
                   같은 입력에도 결론이 바뀌며(AlphaForgeBench 2026), 20년·100종목 재검증(FINSABER 2025)에서 우위가 사라졌다.
                   그래서 구조(역할 분담·반대 논거·거부권·기록·사후 검토)만 가져온다.
  역할
    분석가 4명   시장(추세·낙폭·변동성·국면) · 수급(외국인 순매수) · 거시(환율·미국·채권·금) · 모델(고른 모델의 성적·순위·관문)
    강세·약세    분석가 보고에서 규칙으로 뽑은 찬성·반대 근거 — '확신도' = 찬성 − 반대 (참고용: 이 점수로 크기를 바꾸지 않는다, 검증 전)
    트레이더     short_trader · us_trader 가 낸 내일 주문
    리스크 매니저 risk.py 의 판정 (거부권 — 이미 주문에 반영됨)
    펀드매니저   최종: 승인 / 축소 승인 / 매도만 / 정지 + '검토 필요' 표시
  기록        data/committee/journal.jsonl — 한 줄에 하루, 앞 줄의 해시를 이어 붙여(해시 사슬) 나중에 고칠 수 없게
  사후 검토    5·20거래일이 지난 결정의 실제 결과(몫 수익 − 지수)를 확신도·모델별로 통계만 (글로 된 '반성'은 하지 않음 —
              자유 형식 기억은 끝없는 표본 내 조정이 된다)

  python src/committee.py   → data/committee/journal.jsonl · docs/committee.json
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allweather as AW  # noqa: E402
import risk as RK  # noqa: E402

DIR = "data/committee"
JOURNAL = os.path.join(DIR, "journal.jsonl")
UI = "docs/committee.json"


def jload(p, default=None):
    return RK.jload(p, default)


def pct_rank(x, hist):
    hist = np.asarray(hist, float)
    hist = hist[np.isfinite(hist)]
    return None if not len(hist) or not np.isfinite(x) else round(float((hist <= x).mean()), 3)


# ══════════════════════════════════════════════════════════════
# 분석가
# ══════════════════════════════════════════════════════════════
def analyst_market():
    dates, Op, Cp = AW.load_prices()
    F = AW.features(Cp, dates)
    c = F["c"]
    T = int(np.where(np.isfinite(c))[0][-1])
    vol = F["vol20"]
    out = {"date": str(dates[T].date()), "price": round(float(c[T]), 1),
           "vs_ma120": round(float(c[T] / F["ma120"][T] - 1), 4) if np.isfinite(F["ma120"][T]) else None,
           "vs_ma200": round(float(c[T] / F["ma200"][T] - 1), 4) if np.isfinite(F["ma200"][T]) else None,
           "dd252": round(float(F["dd"][T]), 4), "r20": round(float(F["r20"][T]), 4), "r60": round(float(F["r60"][T]), 4),
           "vol20": round(float(vol[T]), 4), "vol20_pct3y": pct_rank(vol[T], vol[max(0, T - 756):T])}
    try:
        out["regime"] = AW.regime_now(F)
    except Exception:
        out["regime"] = None
    extra = {}
    for code, key in (("148070", "bond10_r60"), ("132030", "gold_r60"), ("261240", "usd_r60")):
        j = AW.CODES.index(code)
        x = pd.Series(Cp[:T + 1, j]).dropna()
        if len(x) > 61:
            extra[key] = round(float(x.iloc[-1] / x.iloc[-61] - 1), 4)
    out.update(extra)
    return out


def analyst_flows():
    try:
        f = pd.read_csv("data/flows.csv", usecols=["날짜", "외국인순매수", "기관순매수"])
    except (OSError, ValueError):
        return None
    g = f.groupby("날짜")[["외국인순매수", "기관순매수"]].sum().sort_index()
    g = g.iloc[-300:]
    f5 = g["외국인순매수"].rolling(5).sum()
    i5 = g["기관순매수"].rolling(5).sum()
    return {"date": str(g.index[-1]), "foreign_5d": float(f5.iloc[-1]), "foreign_20d": float(g["외국인순매수"].iloc[-20:].sum()),
            "inst_5d": float(i5.iloc[-1]), "foreign_5d_pct1y": pct_rank(f5.iloc[-1], f5.iloc[-251:-1]),
            "inst_5d_pct1y": pct_rank(i5.iloc[-1], i5.iloc[-251:-1]), "note": "금액은 네이버 대체 수집이면 수량×종가 근사"}


def analyst_macro():
    out = {}
    try:
        fx = pd.read_csv("data/fx_usdkrw.csv").set_index("날짜")["환율"].astype(float).dropna()
        out["usdkrw"] = round(float(fx.iloc[-1]), 1)
        out["usdkrw_r20"] = round(float(fx.iloc[-1] / fx.iloc[-21] - 1), 4)
    except (OSError, ValueError, KeyError, IndexError):
        pass
    try:
        u = pd.read_csv("data/us_ohlcv.csv", usecols=["날짜", "종목코드", "종가"])
        spy = u[u["종목코드"] == "SPY"].set_index("날짜")["종가"].astype(float).sort_index()
        out["spy_vs_ma200"] = round(float(spy.iloc[-1] / spy.iloc[-200:].mean() - 1), 4)
        out["spy_r20"] = round(float(spy.iloc[-1] / spy.iloc[-21] - 1), 4)
        out["spy_date"] = str(spy.index[-1])
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return out


def analyst_models(short_ui, us_ui):
    out = {}
    m = (short_ui or {}).get("model") or {}
    board = (short_ui or {}).get("leaderboard") or []
    if m:
        top = board[:10]
        kind = next((b["kind"] for b in board if b["id"] == m.get("id")), None)
        same = sum(1 for b in top if b.get("kind") == kind) if kind else None
        gate = jload("data/history/gate.json", {}) or {}
        osh = (gate.get("sharpe") or {}).get(m.get("id"))
        lab = jload("docs/short_lab.json", {}) or {}
        sc = (jload("docs/science.json", {}) or {}).get("overfit", {}).get("kr_short") or {}
        out["short"] = {"id": m.get("id"), "desc": m.get("desc"), "sh250": m.get("sh250"), "selected_at": m.get("selected_at"),
                        "changed_today": m.get("changed_today"), "rank": next((i + 1 for i, b in enumerate(board) if b["id"] == m.get("id")), None),
                        "pool": (short_ui or {}).get("pool_size"), "top10_same_kind": same, "kind": kind, "old_era_sharpe": osh,
                        "pbo": sc.get("pbo"), "dsr": sc.get("dsr"), "rule_test_sharpe": ((lab.get("rules") or {}).get("live") or {}).get("test", {}).get("sharpe")}
    if us_ui:
        sc = (jload("docs/science.json", {}) or {}).get("overfit", {}).get("us") or {}
        out["us"] = {"id": us_ui.get("model"), "desc": us_ui.get("model_desc"), "pbo": sc.get("pbo"), "dsr": sc.get("dsr")}
    return out


# ══════════════════════════════════════════════════════════════
# 강세·약세 연구원 (규칙)
# ══════════════════════════════════════════════════════════════
def debate(mk, fl, mc, md):
    bull, bear = [], []

    def b(cond, txt, tag, lst):
        if cond:
            lst.append({"why": txt, "from": tag})
    if mk:
        b(mk.get("vs_ma200") is not None and mk["vs_ma200"] > 0, f"지수가 200일선 위 ({mk['vs_ma200'] * 100:+.1f}%)", "시장", bull)
        b(mk.get("vs_ma200") is not None and mk["vs_ma200"] < 0, f"지수가 200일선 아래 ({mk['vs_ma200'] * 100:+.1f}%)", "시장", bear)
        b(mk["dd252"] <= -0.15, f"1년 고점 대비 {mk['dd252'] * 100:.0f}% — 급락 국면", "시장", bear)
        b(mk.get("vol20_pct3y") is not None and mk["vol20_pct3y"] >= 0.9, f"20일 변동성이 3년 중 상위 {100 - mk['vol20_pct3y'] * 100:.0f}% — 흔들림 큼", "시장", bear)
        b(mk.get("r20", 0) > 0.05, f"최근 20일 지수 {mk['r20'] * 100:+.1f}% — 반등 중", "시장", bull)
    if fl:
        p = fl.get("foreign_5d_pct1y")
        amt = f"{fl['foreign_5d'] / 1e8:+,.0f}억"
        b(p is not None and p >= 0.7, f"외국인 5일 수급 {amt} — 1년 중 위쪽 {100 - p * 100:.0f}% (평소보다 덜 팔거나 더 삼)", "수급", bull)
        b(p is not None and p <= 0.3, f"외국인 5일 수급 {amt} — 1년 중 아래쪽 {p * 100:.0f}% (평소보다 많이 팖)", "수급", bear)
    if mc:
        b(mc.get("usdkrw_r20", 0) >= 0.03, f"원/달러 20일 {mc['usdkrw_r20'] * 100:+.1f}% — 원화 약세(외국인 이탈 위험)", "거시", bear)
        b(mc.get("usdkrw_r20", 0) <= -0.03, f"원/달러 20일 {mc['usdkrw_r20'] * 100:+.1f}% — 원화 강세", "거시", bull)
        b(mc.get("spy_vs_ma200") is not None and mc["spy_vs_ma200"] > 0, "미국 S&P 500 이 200일선 위", "거시", bull)
        b(mc.get("spy_vs_ma200") is not None and mc["spy_vs_ma200"] < 0, "미국 S&P 500 이 200일선 아래", "거시", bear)
    s = (md or {}).get("short") or {}
    if s:
        b((s.get("sh250") or 0) >= 1, f"고른 모델 최근 250일 샤프 {s['sh250']:.2f}", "모델", bull)
        b((s.get("sh250") or 0) < 0.3, f"고른 모델 최근 250일 샤프 {s.get('sh250') or 0:.2f} — 약함", "모델", bear)
        b(s.get("old_era_sharpe") is not None and s["old_era_sharpe"] > 0.3, f"설계에 안 쓴 옛 시대에도 샤프 {s.get('old_era_sharpe') or 0:.2f}", "모델", bull)
        b(s.get("old_era_sharpe") is None, "옛 시대 자료가 없어 긴 역사 검증을 못 한 모델 (수급·진화)", "모델", bear)
        b((s.get("top10_same_kind") or 0) >= 7, f"상위 10개 중 {s.get('top10_same_kind')}개가 같은 종류 — 의견 일치", "모델", bull)
        b(s.get("pbo") is not None and s["pbo"] >= 0.2, f"과최적화 진단 PBO {s['pbo'] * 100:.0f}% — 백테스트의 절반 정도 기대", "모델", bear)
    return bull, bear


# ══════════════════════════════════════════════════════════════
# 펀드매니저 · 기록 · 사후 검토
# ══════════════════════════════════════════════════════════════
def decide(plan, sleeve_names):
    rk = (plan or {}).get("risk") or {}
    state = rk.get("state", "—")
    out = []
    for sl, name in sleeve_names.items():
        orders = [o for o in (plan or {}).get("orders") or [] if (o.get("sleeve") or "live") == sl]
        log = [x for x in rk.get("log") or [] if x.get("sleeve") == sl]
        mon = (rk.get("sleeves") or {}).get(sl) or {}
        if state == "HALTED":
            d = "정지"
        elif state == "REDUCING" or mon.get("action") == "REDUCING":
            d = "매도만"
        elif any(x["act"] in ("축소", "취소") for x in log):
            d = "축소 승인"
        else:
            d = "승인" if orders else "할 일 없음"
        out.append({"sleeve": sl, "name": name, "decision": d, "orders": len(orders),
                    "buys": sum(1 for o in orders if o["side"] == "buy"), "sells": sum(1 for o in orders if o["side"] == "sell"),
                    "risk_log": log, "monitor": {k: mon.get(k) for k in ("why", "review", "days", "sr_star")}, "review": bool(mon.get("review"))})
    return state, rk.get("why"), out


def last_entry():
    if not os.path.exists(JOURNAL):
        return None
    last = None
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


def append_journal(entry):
    """해시 사슬: hash = sha256(앞 hash + 이 항목(해시 제외) JSON). 같은 날·같은 내용이면 다시 쓰지 않는다."""
    os.makedirs(DIR, exist_ok=True)
    prev = last_entry()
    body = json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
    if prev and prev.get("date") == entry["date"] and prev.get("body_hash") == hashlib.sha256(body.encode()).hexdigest():
        return prev, False
    e = dict(entry)
    e["body_hash"] = hashlib.sha256(body.encode()).hexdigest()
    e["prev"] = prev["hash"] if prev else "0" * 64
    e["hash"] = hashlib.sha256((e["prev"] + e["body_hash"]).encode()).hexdigest()
    e["rerun"] = bool(prev and prev.get("date") == entry["date"])
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    return e, True


def verify_chain():
    if not os.path.exists(JOURNAL):
        return {"ok": True, "n": 0}
    prev, n = "0" * 64, 0
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            n += 1
            body = {k: v for k, v in e.items() if k not in ("body_hash", "prev", "hash", "rerun")}
            bh = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
            if e.get("prev") != prev or e.get("body_hash") != bh or e.get("hash") != hashlib.sha256((prev + bh).encode()).hexdigest():
                return {"ok": False, "n": n, "broken_at": e.get("date")}
            prev = e["hash"]
    return {"ok": True, "n": n}


def reflection():
    """지난 결정의 실제 결과 (통계만): 결정 뒤 5·20거래일 단기 몫 수익 − 지수 ETF, 확신도·모델별."""
    try:
        eq = pd.read_csv("data/short/equity.csv").set_index("날짜")
    except (OSError, ValueError):
        return None
    acc = jload("data/short/account.json", {}) or {}
    fl = acc.get("flows") or []
    r_live = RK.flow_adjusted(eq["단기몫"].astype(float), [{"after": f["after"], "amount": f["live"]} for f in fl]) if "단기몫" in eq else None
    r_idx = eq["지수ETF"].astype(float).pct_change().iloc[1:] if "지수ETF" in eq else None
    if r_live is None or r_idx is None or not os.path.exists(JOURNAL):
        return None
    rows = []
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if not e.get("rerun"):
                    rows.append(e)
    out = {}
    for hz in (5, 20):
        res = []
        for e in rows:
            d = e["date"]
            fut = r_live[r_live.index > d].iloc[:hz]
            fi = r_idx[r_idx.index > d].iloc[:hz]
            if len(fut) < hz:
                continue
            ex = float(np.prod(1 + fut) - np.prod(1 + fi))
            res.append({"date": d, "conv": e.get("conviction", 0), "model": (e.get("models") or {}).get("short", {}).get("id"), "excess": ex})
        if not res:
            out[f"{hz}d"] = {"n": 0}
            continue
        df = pd.DataFrame(res)
        out[f"{hz}d"] = {"n": int(len(df)), "mean_excess": round(float(df.excess.mean()), 4), "hit": round(float((df.excess > 0).mean()), 3),
                         "by_conv": {s: {"n": int(len(g)), "mean_excess": round(float(g.excess.mean()), 4)}
                                     for s, g in df.groupby(np.sign(df.conv).map({-1: "약세 우세", 0: "팽팽", 1: "강세 우세"}))},
                         "by_model": {m: {"n": int(len(g)), "mean_excess": round(float(g.excess.mean()), 4)} for m, g in df.groupby("model")}}
    return out


def shortfall(short_ui):
    """실행 괴리: 단기 몫의 실제(모의) 수익 − 같은 규칙을 시뮬레이터로 따라간 수익 (Lean 의 '실거래 옆 백테스트')."""
    if not short_ui:
        return None
    sh = {x["key"]: x for x in short_ui.get("shadow") or []}
    live = (short_ui.get("sleeves") or {}).get("live") or {}
    acc = jload("data/short/account.json", {}) or {}
    start_live = acc.get("initial_cash", 0) * (live.get("share") or 0.5)
    net_in = sum(f["live"] for f in acc.get("flows") or [])
    if not start_live or "live" not in sh:
        return None
    real = (live["equity"] - net_in) / start_live - 1
    sim = sh["live"]["ret"]
    return {"real": round(real, 4), "sim": round(sim, 4), "gap": round(real - sim, 4), "days": sh["live"].get("days"),
            "note": "모의는 같은 가격으로 체결하므로 차이 = 정수 주식·현금 여유·체결 시점 차이. 실계좌를 붙이면 미끄러짐이 여기 나타난다"}


def main():
    t0 = time.time()
    short_ui = jload("docs/short.json", {}) or {}
    us_ui = jload("docs/us.json", {}) or {}
    plan_s = jload("data/short/plan.json", {}) or {}
    plan_u = jload("data/us/plan.json", {}) or {}
    mk = analyst_market()
    fl = analyst_flows()
    mc = analyst_macro()
    md = analyst_models(short_ui, us_ui)
    bull, bear = debate(mk, fl, mc, md)
    st_s, why_s, dec_s = decide(plan_s, {"live": "단기 갈아타기", "aw": "전천후 ETF 혼합"})
    st_u, why_u, dec_u = decide(plan_u, {"us": "미국 추세 상위 10"})
    review = [f"{d['name']}: " + "; ".join(d["monitor"].get("why") or []) for d in dec_s + dec_u if d["review"]]
    entry = {"date": short_ui.get("data_date") or mk["date"], "us_date": us_ui.get("data_date"),
             "code": os.environ.get("GITHUB_SHA", "")[:12] or None,
             "conviction": len(bull) - len(bear), "bull": [x["why"] for x in bull], "bear": [x["why"] for x in bear],
             "state": {"kr": st_s, "us": st_u}, "decisions": [{k: d[k] for k in ("sleeve", "decision", "buys", "sells")} for d in dec_s + dec_u],
             "models": {"short": {"id": (md.get("short") or {}).get("id"), "sh250": (md.get("short") or {}).get("sh250")},
                        "us": {"id": (md.get("us") or {}).get("id")}},
             "regime": (mk.get("regime") or {}).get("name") if isinstance(mk.get("regime"), dict) else mk.get("regime"),
             "orders": {"kr": [{k: o.get(k) for k in ("side", "code", "qty", "sleeve")} for o in plan_s.get("orders") or []],
                        "us": [{k: o.get(k) for k in ("side", "code", "usd")} for o in plan_u.get("orders") or []]},
             "review": review}
    e, wrote = append_journal(entry)
    chain = verify_chain()
    tail = []
    if os.path.exists(JOURNAL):
        with open(JOURNAL, encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        tail = [{k: x.get(k) for k in ("date", "conviction", "regime", "decisions", "models", "hash", "rerun")} for x in lines[-30:]][::-1]
    ui = {"updated": time.strftime("%Y-%m-%d %H:%M"), "date": entry["date"],
          "analysts": {"market": mk, "flows": fl, "macro": mc, "models": md},
          "bull": bull, "bear": bear, "conviction": entry["conviction"],
          "risk": {"kr": {"state": st_s, "why": why_s, "limits": (plan_s.get("risk") or {}).get("limits")},
                   "us": {"state": st_u, "why": why_u}},
          "decisions": dec_s + dec_u, "review": review, "journal": {"chain": chain, "tail": tail, "wrote": wrote, "hash": e.get("hash")},
          "reflection": reflection(), "shortfall": shortfall(short_ui)}
    from short_trader import clean
    json.dump(clean(ui), open(UI, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"투자위원회 {entry['date']}: 확신도 {entry['conviction']:+d} (강세 {len(bull)} · 약세 {len(bear)}) · "
          + " · ".join(f"{d['name']} {d['decision']}" for d in dec_s + dec_u)
          + (f" · 검토 필요 {len(review)}" if review else "") + f" · 기록 {'추가' if wrote else '그대로'} · 사슬 {'정상' if chain['ok'] else '깨짐'} ({time.time() - t0:.0f}s)")
    return ui


if __name__ == "__main__":
    main()
