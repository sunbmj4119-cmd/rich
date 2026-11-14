"""
그림자 기록 거꾸로 따라가기 — '나무 모델 · 5일마다 · 상위 3종목 (상위 10 안이면 계속 보유)'을 2008 년부터 매주 실제로 했다면.
예측은 해마다 그때까지의 자료로만 다시 배운 것 (wf_lab 과 같음) · 세 씨앗(0 · +1000 · +2000)의 예측 순위 평균.
매주 무엇을 샀는지 · 얼마가 됐는지를 DM:2(지금 공격 엔진) · 전천후 16장 · 코스피 · 대형주 평균과 나란히.

  ⚠ 5일마다 사고파는 규칙은 '어느 날부터 세기 시작했나'(시작 요일)에 따라 2019~ 결과가 연 0% ~ +38% 로 갈린다.
    그래서 한 날짜에 다 거는 대신 '5개 묶음' — 종목 몫을 5 등분해 묶음마다 다른 날(1~5번째 거래일)에 5일 주기로 바꾼다 —
    을 본 결과로 쓰고, 다섯 시작 요일 각각의 결과도 따로 남긴다 (운 좋은 하루를 고르지 않도록).

  python src/shadow_wf.py --seed 0|1000|2000     # 씨앗별 예측 (뒤에서 나눠 돌림)
  python src/shadow_wf.py                         # 합쳐서 분석 → data/shadow_wf.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_lab as EL  # noqa: E402
import wf_lab as W  # noqa: E402

OUT = "data/shadow_wf.json"
DOCS = "docs/shadow_wf.json"
H, K, KEEP, SHARE = 5, 3, 10, 0.3
DEV_END = "2018-12-31"


def tag():
    return "v" + time.strftime("%Y%m%d")


def seed_pred(D, off, log):
    """walk_forward 와 같은 방식 · 나무 모델 씨앗 = 해 + off."""
    keys = list(EL.FEAT)
    if off == 0:
        return W.walk_forward(D, keys, H, "gbm", every=2, log=log)
    return W.walk_forward_seed(D, keys, H, off, log)


def keyframe(D):
    import setup_lab as SL
    codes = {era: [str(c).zfill(6) for c in SL.panel(era)[0]["codes"]] for era in ("A", "B")}
    return pd.DataFrame({"date": D["date"].to_numpy(), "code": [codes[e][j] for e, j in zip(D["era"], D["j"])]})


def names():
    nm = {}
    try:
        h = pd.read_csv("data/history/kr_stocks.csv", usecols=["종목코드", "종목명"], dtype=str).drop_duplicates("종목코드", keep="last")
        nm.update(dict(zip(h["종목코드"].str.zfill(6), h["종목명"])))
    except (OSError, ValueError):
        pass
    s = pd.read_csv("data/sectors.csv", dtype=str)
    nm.update(dict(zip(s["종목코드"].str.zfill(6), s["종목명"])))
    return nm


def grid(D, start="2008-01-01"):
    return np.sort(D.loc[D["date"] >= pd.Timestamp(start), "date"].unique())


def detailed_book(D, p, codes, days, cost_mult=1.0):
    """book_buffer 와 같은 규칙으로, 주마다 산 종목·판 종목·종목별 수익까지 남긴다 · days = 이 묶음이 바꾸는 날들."""
    cb, cs = W.SL.COMM + W.SL.SLIP * cost_mult, W.SL.COMM + W.SL.SLIP * cost_mult + W.SL.TAX
    d = D[["date", "y5"]].assign(p=p, code=codes)
    d = d[d["date"].isin(days)]
    held, rows = [], []
    for day, g in d.groupby("date", sort=True):
        lab = g[np.isfinite(g["y5"])]
        ew = float(lab["y5"].mean()) if len(lab) else np.nan
        g = g[np.isfinite(g["p"])].sort_values("p", ascending=False)
        rank = {c: i for i, c in enumerate(g["code"])}
        ret = dict(zip(g["code"], g["y5"]))
        stay = [c for c in held if c in rank and rank[c] < KEEP][:K]
        new = [c for c in g["code"] if c not in stay][: K - len(stay)]
        sold = [c for c in held if c not in stay]
        port = stay + new
        w = 1.0 / len(port) if port else 0.0
        r = sum(w * ret.get(c, np.nan) for c in port) if port else 0.0
        cost = len(new) * w * cb + (len(sold) * (1.0 / max(len(held), 1)) * cs if held else 0.0)
        rows.append({"date": day, "s": r - cost if np.isfinite(r) else np.nan, "ew": ew, "hold": port, "new": new, "sold": sold,
                     "rets": {c: (None if not np.isfinite(ret.get(c, np.nan)) else round(float(ret[c]), 4)) for c in port}})
        held = port
    return pd.DataFrame(rows).set_index("date")


def integer_curve(D, kf, bk, aw_week, cash0, w=SHARE, a="2019-01-01"):
    """실제 시세·정수 주식 (2019~) 한 묶음: 주마다 자산의 w 를 그 주 보유 종목에 똑같이 · 1주도 못 사는 종목 몫은 현금 · 나머지는 전천후.
    비용: 새로 산 종목에만 왕복 비용을 산 날에 (계속 보유는 비용 없음). 반환: 실현일 기준 자산 (결정일 색인, 그 주 수익은 다음 결정일에 반영) · 산 비율."""
    import setup_lab as SL
    P = SL.panel("B")[0]
    O = P["O"]
    sel = (D["era"] == "B").to_numpy()
    look = {(d, c): (int(t), int(j), y) for d, c, t, j, y in zip(D["date"][sel], kf["code"][sel], D["t"][sel], D["j"][sel], D["y5"][sel])}
    cb, cs = W.CB, W.CS
    weeks = bk[bk.index >= pd.Timestamp(a)]
    E, curve, used = float(cash0), [], []
    for day, r in weeks.iterrows():
        curve.append((day, E))
        if not np.isfinite(r["s"]):
            continue
        stock = w * E
        per = stock / max(len(r["hold"]), 1)
        spent = got = 0.0
        for c in r["hold"]:
            t, j, y = look.get((day, c), (None, None, np.nan))
            if t is None or not np.isfinite(y):
                continue
            po = O[t + 1, j]
            q = int(per // (po * (1 + cb))) if np.isfinite(po) and po > 0 else 0
            if q <= 0:
                continue
            fee = (cb + cs) if c in r["new"] else 0.0
            spent += q * po
            got += q * po * (1 + y) - q * po * fee
        used.append(spent / max(stock, 1e-9))
        E = (stock - spent + got) + (1 - w) * E * (1 + aw_week.get(day, 0.0))
    return pd.Series(dict(curve)), float(np.mean(used)) if used else 0.0


def integer_accounts(D, kf, books, aws, sample, sizes=(200_000, 1_000_000, 3_000_000, 10_000_000)):
    """5개 묶음 계좌: 처음 돈을 5 등분해 묶음마다 정수 주식 계좌 하나씩 → 합친 자산을 5거래일 간격으로 읽어 성적."""
    out = {}
    days = pd.DatetimeIndex(sample)
    for cash0 in sizes:
        parts, used = [], []
        for bk, aw in zip(books, aws):
            c, u = integer_curve(D, kf, bk, aw, cash0 / len(books))
            parts.append(c.reindex(c.index.union(days)).ffill().reindex(days).fillna(cash0 / len(books)))
            used.append(u)
        tot = sum(parts)
        tot = tot[tot.index >= pd.Timestamp("2019-01-01")]
        rr = tot.pct_change().dropna()
        st = stats(rr, 252 / H)
        out[cash0] = {**st, "final": round(float(tot.iloc[-1])), "invested": round(float(np.mean(used)), 3)}
    return out


def realized(s, all_days):
    """묶음 하나의 기간 수익 (결정일 색인) → 실현일 기준 자산을 모든 거래일로 펼침 (그 주 수익은 다음 결정일에 반영)."""
    e = (1 + s.fillna(0.0)).cumprod().shift(1).fillna(1.0)
    return e.reindex(e.index.union(all_days)).ffill().reindex(all_days).fillna(1.0)


def stats(r, ppy):
    r = r.dropna()
    if len(r) < 10:
        return None
    E = (1 + r).cumprod()
    yrs = len(r) / ppy
    return {"cagr": round(float(E.iloc[-1] ** (1 / yrs) - 1), 4), "sharpe": round(float(r.mean() / r.std() * np.sqrt(ppy)), 3) if r.std() > 0 else 0.0,
            "mdd": round(float((E / E.cummax() - 1).min()), 4), "total": round(float(E.iloc[-1] - 1), 4),
            "win_weeks": round(float((r > 0).mean()), 3)}


def load_pred(D):
    ps = []
    for off in (0, 1000, 2000):
        fn = os.path.join(W.CACHE, f"shadow_gbm_h5_seed{off}_{tag()}.npy")
        if not os.path.exists(fn):
            cands = sorted(f for f in os.listdir(W.CACHE) if f.startswith(f"shadow_gbm_h5_seed{off}_"))
            fn = os.path.join(W.CACHE, cands[-1]) if cands else fn
        p = np.load(fn)
        assert len(p) == len(D), f"예측 길이가 다름 {fn}"
        ps.append(pd.Series(p).groupby(D["date"].to_numpy()).rank(pct=True).to_numpy())
    return np.nanmean(np.c_[ps], axis=0)


def kospi_index(M):
    """실제 코스피 지수(^KS11 종가, data/history/hist.csv) · 그 뒤 날짜는 KODEX 200 종가 수익으로 이어 붙임.
    파일이 없으면 패널의 시총가중 지수 (두 시대 이음매 하루 수익은 0 으로)."""
    try:
        h = pd.read_csv("data/history/hist.csv", dtype={"종목코드": str})
        k = h[h["종목코드"] == "^KS11"].assign(날짜=lambda x: pd.to_datetime(x["날짜"])).set_index("날짜")["종가"].sort_index()
        try:
            e = pd.read_csv("data/etf_ohlcv.csv", dtype={"종목코드": str})
            e = e[e["종목코드"] == "069500"].assign(날짜=lambda x: pd.to_datetime(x["날짜"])).set_index("날짜")["종가"].sort_index()
            r = e.pct_change()
            r = r[r.index > k.index[-1]].dropna()
            if len(r):
                k = pd.concat([k, k.iloc[-1] * (1 + r).cumprod()])
        except (OSError, KeyError, ValueError):
            pass
        return k / k.iloc[0]
    except (OSError, KeyError, ValueError, IndexError):
        r = M["idx"].pct_change().fillna(0.0)
        r[r.abs() > 0.5] = 0.0
        return (1 + r).cumprod()


def cut(x, a, b):
    if a:
        x = x[x.index >= pd.Timestamp(a)]
    if b:
        x = x[x.index <= pd.Timestamp(b)]
    return x


def analyze(log=print):
    t0 = time.time()
    D, M = W.build(log, hs=(H,), sector=False)
    kf = keyframe(D)
    p = load_pred(D)
    nm = names()
    codes = kf["code"].to_numpy()
    all_days = grid(D)
    aw_ix, dm_ix = W.aw_index(), W.card_index("DM:2")
    ks_ix = kospi_index(M)
    # 묶음 5개: 묶음 g 는 g번째 거래일부터 5일마다 바꾼다
    books, books3, aws = [], [], []
    for g in range(H):
        dg = all_days[g::H]
        bk = detailed_book(D, p, codes, dg)
        bk3 = detailed_book(D, p, codes, dg, cost_mult=3.0).reindex(bk.index)
        books.append(bk)
        books3.append(bk3)
        aws.append(W.period_ret(aw_ix, bk.index, H))
    last_real = min(bk.index[np.isfinite(bk["s"].to_numpy())].max() for bk in books)
    sample = all_days[::H]
    sample = sample[sample <= np.datetime64(last_real)]
    S = pd.DatetimeIndex(sample)

    def combine(series_list):
        e = sum(realized(s, pd.DatetimeIndex(all_days)) for s in series_list) / len(series_list)
        return e.reindex(S)

    eq = {"stock": combine([bk["s"] for bk in books]),
          "blend": combine([SHARE * bk["s"] + (1 - SHARE) * aw for bk, aw in zip(books, aws)]),
          "blend_x3": combine([SHARE * b3["s"] + (1 - SHARE) * aw for b3, aw in zip(books3, aws)]),
          "ew": combine([bk["ew"] for bk in books])}
    for k, ix in (("dm2", dm_ix), ("aw16", aw_ix), ("kospi", ks_ix)):
        e = ix.reindex(ix.index.union(S)).ffill().reindex(S)
        eq[k] = e / e.iloc[0]
    order = ["stock", "blend", "blend_x3", "dm2", "aw16", "kospi", "ew"]
    E = pd.DataFrame({k: eq[k] for k in order})
    ser = E.pct_change().iloc[1:]
    ppy = 252 / H
    periods = {"전체 2008~": (None, None), "개발 2008~2018": (None, DEV_END), "최종 확인 2019~": ("2019-01-01", None),
               "최근 1년": ((ser.index[-1] - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), None)}
    names_ser = {"stock": "종목 3개 (모델 · 5개 묶음)", "blend": "종목 30% + 전천후 70%", "blend_x3": "같은 섞기 · 비용 3배", "dm2": "DM:2 (지금 공격 엔진 · 20만 원 계획)",
                 "aw16": "전천후 16장", "kospi": "코스피", "ew": "대형주 100 평균"}
    tab = {pn: {k: stats(cut(ser, a, b)[k], ppy) for k in order} for pn, (a, b) in periods.items()}
    # 시작 요일 하나에만 걸었을 때 (묶음 하나 = 자산 전부)
    phase_tab, phase_eq = [], {}
    for g, (bk, aw) in enumerate(zip(books, aws)):
        s = bk["s"][bk.index <= pd.Timestamp(last_real)]
        bl = SHARE * s + (1 - SHARE) * aw.reindex(s.index)
        for pn, (a, b) in periods.items():
            if pn == "전체 2008~":
                continue
            st, sb = stats(cut(s, a, b), ppy), stats(cut(bl, a, b), ppy)
            phase_tab.append({"period": pn, "phase": g + 1, "start": str(pd.Timestamp(all_days[g]).date()),
                              **{f"stock_{x}": st[x] for x in ("cagr", "sharpe", "mdd")}, **{f"blend_{x}": sb[x] for x in ("cagr", "sharpe", "mdd")}})
        phase_eq[f"p{g + 1}"] = realized(s, pd.DatetimeIndex(all_days)).reindex(S)
    yr = E.groupby(E.index.year).last()
    yret = (yr / yr.shift(1).fillna(1.0) - 1).round(4)
    dd = (E / E.cummax() - 1).round(4)
    # 최근 바꾼 기록 (묶음마다 하루씩 돌아가며 · 최근 30번)
    ev = sorted(((d, g, r) for g, bk in enumerate(books) for d, r in bk.iterrows()), key=lambda x: x[0])[-30:]
    log_rows = []
    for d, g, r in ev:
        log_rows.append({"date": str(d.date()), "group": g + 1,
                         "hold": [{"code": c, "name": nm.get(c, c), "ret": r["rets"].get(c), "new": c in r["new"]} for c in r["hold"]],
                         "sold": [{"code": c, "name": nm.get(c, c)} for c in r["sold"]],
                         "stock": None if not np.isfinite(r["s"]) else round(float(r["s"]), 4)})
    allh = pd.Series([c for bk in books for h in bk["hold"] for c in h])
    top = allh.value_counts().head(15)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "data_until": str(D["date"].max().date()), "measured_until": str(pd.Timestamp(last_real).date()),
           "rule": {"every_days": H, "k": K, "keep_top": KEEP, "share": SHARE, "seeds": [0, 1000, 2000], "groups": H},
           "names": names_ser, "table": tab, "phases": phase_tab,
           "curve": {"date": [str(d.date()) for d in E.index], **{k: [round(float(v), 4) for v in E[k]] for k in E.columns}},
           "phase_curve": {"date": [str(d.date()) for d in S], **{k: [round(float(v), 4) for v in s_] for k, s_ in phase_eq.items()}},
           "drawdown": {"date": [str(d.date()) for d in dd.index], **{k: [float(v) for v in dd[k]] for k in ("blend", "dm2", "kospi", "stock")}},
           "years": {str(y): {k: (None if not np.isfinite(v) else float(v)) for k, v in row.items()} for y, row in yret.iterrows()},
           "weeks": log_rows,
           "favorites": [{"code": c, "name": nm.get(c, c), "weeks": int(n)} for c, n in top.items()],
           "turnover_new_per_week": round(float(np.mean([bk["new"].apply(len).mean() for bk in books])), 2),
           "secs": round(time.time() - t0)}
    try:
        res["discipline"] = json.load(open("data/discipline_lab.json", encoding="utf-8"))["results"]
    except (OSError, ValueError, KeyError):
        pass
    res["accounts"] = {str(k): v for k, v in integer_accounts(D, kf, books, aws, sample).items()}
    last = D["date"].max()
    lm = (D["date"] == last).to_numpy()
    now = pd.DataFrame({"code": codes[lm], "p": p[lm]}).sort_values("p", ascending=False).head(10)
    weight = {}
    for bk in books:
        h = bk["hold"].iloc[-1]
        for c in h:
            weight[c] = weight.get(c, 0.0) + 1.0 / H / max(len(h), 1)
    nxt = int(np.argmin([bk.index[-1] for bk in books]))
    res["now"] = {"date": str(last.date()), "next_group": nxt + 1, "next_group_last": str(books[nxt].index[-1].date()),
                  "held": [{"code": c, "name": nm.get(c, c), "weight": round(w_ * SHARE, 4)} for c, w_ in sorted(weight.items(), key=lambda x: -x[1])],
                  "top": [{"rank": i + 1, "code": c, "name": nm.get(c, c), "score": round(float(v), 3), "weight": round(weight.get(c, 0.0) * SHARE, 4)}
                          for i, (c, v) in enumerate(zip(now["code"], now["p"]))]}
    for fn in (OUT, DOCS):
        json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for k, v in res["accounts"].items():
        log(f"  실제 주식 수 {int(k):>11,}원: 연 {v['cagr']:+.1%} · 샤프 {v['sharpe']:.2f} · 낙폭 {v['mdd']:.0%} → {v['final']:,}원 · 종목 몫 실제로 산 비율 {v['invested']:.0%}")
    log(f"  오늘({res['now']['date']}) 상위: " + ", ".join(f"{x['name']}" for x in res["now"]["top"][:5]) + f" · 지금 보유 {[x['name'] for x in res['now']['held']]}")
    log(f"끝 ({res['secs']}초) · 자료 {res['data_until']} 까지 (성적은 {res['measured_until']} 결정분까지) → {OUT}")
    for pn, v in tab.items():
        log(f"■ {pn}")
        for k, st in v.items():
            if st:
                log(f"  {names_ser[k]:28s} 연 {st['cagr']:+.1%} · 샤프 {st['sharpe']:.2f} · 최대낙폭 {st['mdd']:.0%} · 오른 주 {st['win_weeks']:.0%}")
    log("■ 시작 요일 하나에만 걸었을 때")
    for r in phase_tab:
        log(f"  {r['period']:12s} {r['phase']}번째 날 시작: 종목 연 {r['stock_cagr']:+.1%} 샤프 {r['stock_sharpe']:.2f} · 섞기 연 {r['blend_cagr']:+.1%} 샤프 {r['blend_sharpe']:.2f} 낙폭 {r['blend_mdd']:.0%}")
    return res


def export(res, outdir):
    """화면(대시보드)에 붙일 자료 파일 — 행 + 만든 코드 (읽는 사람이 어떻게 계산했는지 볼 수 있게)."""
    os.makedirs(outdir, exist_ok=True)
    code = open(os.path.abspath(__file__), encoding="utf-8").read()
    meta = [{"label": "shadow_wf.py", "value": code, "type": "code", "lang": "python"},
            {"label": "입력", "value": f"그때의 시총 상위 100 일봉·수급 (2002~2018 KRX 상장폐지 포함 · 2019~ 단기 패널), 전천후 16장·DM:2 (allweather), 자료 {res['data_until']} 까지"},
            {"label": "규칙", "value": "5거래일마다 나무 모델(세 씨앗 평균, 해마다 다시 배움) 상위 3종목 · 상위 10 안이면 계속 보유 · 종목 30% + 전천후 70% · 왕복 비용 0.34% · "
                                     "종목 몫을 5개 묶음으로 나눠 묶음마다 다른 날(1~5번째 거래일)부터 5일 주기 · 성적은 합친 자산을 5거래일 간격으로 읽음"}]
    nmap = res["names"]
    files = {}
    c = res["curve"]
    files["curve"] = [{"date": d, **{k: c[k][i] for k in nmap}} for i, d in enumerate(c["date"])]
    files["summary"] = [{"period": pn, "strategy": nmap[k], "key": k, **(st or {})} for pn, v in res["table"].items() for k, st in v.items() if st]
    files["years"] = [{"year": int(y), **{k: (None if v.get(k) is None else round(v[k], 4)) for k in nmap}} for y, v in res["years"].items()]
    files["phases"] = res["phases"]
    pc = res["phase_curve"]
    i0 = next(i for i, d in enumerate(pc["date"]) if d >= "2019-01-01")
    base = {k: pc[k][i0] for k in pc if k != "date"}
    kb, db_ = c["kospi"][i0], c["dm2"][i0]
    files["phase_curve"] = [{"date": d, **{k: round(pc[k][i] / base[k], 4) for k in base}, "kospi": round(c["kospi"][i] / kb, 4), "dm2": round(c["dm2"][i] / db_, 4),
                             "stock": round(c["stock"][i] / c["stock"][i0], 4)} for i, d in enumerate(pc["date"]) if i >= i0]
    files["weeks"] = [{"date": w["date"], "group": w["group"], "name": h["name"], "code": h["code"], "ret": h["ret"], "new": bool(h["new"]), "week_stock": w["stock"]}
                      for w in res["weeks"] for h in w["hold"]]
    files["discipline"] = [{"period": pn, "method": m, **st} for pn, v in (res.get("discipline") or {}).items() for m, st in v.items()]
    files["accounts"] = [{"cash0": int(k), **v} for k, v in res["accounts"].items()]
    files["now"] = [{**x, "date": res["now"]["date"], "next_group": res["now"]["next_group"]} for x in res["now"]["top"]]
    files["held"] = [{**x, "date": res["now"]["date"]} for x in res["now"]["held"]]
    files["favorites"] = res["favorites"]
    for k, rows in files.items():
        json.dump({"rows": rows, "meta": meta}, open(os.path.join(outdir, f"{k}.json"), "w", encoding="utf-8"), ensure_ascii=False, default=float)
    return list(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--export", default=None, help="화면용 자료 파일을 쓸 폴더")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)                            # noqa: E731
    if a.seed is None:
        res = analyze(log)
        if a.export:
            log(f"  화면 자료: {export(res, a.export)}")
        return
    D, _ = W.build(log, hs=(H,), sector=False)
    p = seed_pred(D, a.seed, log)
    os.makedirs(W.CACHE, exist_ok=True)
    np.save(os.path.join(W.CACHE, f"shadow_gbm_h5_seed{a.seed}_{tag()}.npy"), p)
    log(f"씨앗 {a.seed} 끝")


if __name__ == "__main__":
    main()
