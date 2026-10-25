"""
토스 자료 예측력 검증 (서버에서 실행) — toss_collect.py 가 모은 원본(~/.rich/toss/)으로
'이 숫자가 다음 날~10일 주가를 미리 알려 주는가'를 short_lab 과 같은 방식(순위 IC · 다음날 시가 체결)으로 잰다.

  필드 이름을 미리 알 수 없어(명세에 예시 없음) 숫자 필드를 전부 후보로 삼는다:
    · 금액/수량 같은 흐름 값 → 시가총액(또는 상장주식수)으로 나눔, 1일·5일 합
    · 잔고(balance/remain/outstanding) 같은 쌓이는 값 → 5일 변화율
    · 비율(ratio/rate/percent) → 그대로
  다중검정: 후보가 많으므로 |t| ≥ 3 이고 9년 중 7년 이상 같은 부호일 때만 '쓸 만함'으로 표시.

GitHub 에는 통계만 올린다 (data/toss/lab.json — 필드 이름 · IC · t · 해마다 부호, 원자료 없음).
  python toss_lab.py
"""
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)
import short_panel as SP  # noqa: E402

RAW = os.path.join(os.path.expanduser("~"), ".rich", "toss")
OUT = os.path.join(REPO, "data", "toss", "lab.json")
ENDPOINTS = ["investor-trading", "program-trades", "short-selling", "credit-trades", "securities-lending"]
STOCK_LIKE = re.compile(r"(balance|remain|outstanding|잔고|holding)", re.I)
RATIO_LIKE = re.compile(r"(ratio|rate|percent|weight|비중|비율)", re.I)
QTY_LIKE = re.compile(r"(volume|quantity|qty|shares|수량)", re.I)
SKIP = re.compile(r"(date|time|symbol|currency|code|name)$", re.I)


def num(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_endpoint(ep, dates, codes):
    """원본 jsonl → {필드: (T×N) 행렬}."""
    mats = {}
    d_ix = {d: i for i, d in enumerate(dates.strftime("%Y-%m-%d"))}
    c_ix = {c: j for j, c in enumerate(codes)}
    folder = os.path.join(RAW, ep)
    if not os.path.isdir(folder):
        return mats
    for fn in os.listdir(folder):
        code = fn.split(".")[0]
        j = c_ix.get(code)
        if j is None:
            continue
        for line in open(os.path.join(folder, fn), encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            i = d_ix.get(r.get("_date"))
            if i is None:
                continue
            for k, v in r.items():
                if k.startswith("_") or SKIP.search(k):
                    continue
                x = num(v)
                if np.isnan(x):
                    continue
                if k not in mats:
                    mats[k] = np.full((len(dates), len(codes)), np.nan)
                mats[k][i, j] = x
    return mats


def variants(name, X, P):
    """필드 하나 → 비교 가능한 신호 몇 개 (클수록 '산다' 방향은 IC 부호로 판단)."""
    MC, C = P["MC"], P["C"]
    shares = MC / C
    out = {}
    with np.errstate(all="ignore"):
        if RATIO_LIKE.search(name):
            out[name] = X
            out[name + " Δ5"] = X - SP._lag(X, 5)
        elif STOCK_LIKE.search(name):
            base = shares if QTY_LIKE.search(name) else MC
            out[name + " /시총"] = X / base
            out[name + " 5일 변화"] = (X - SP._lag(X, 5)) / base
        else:
            base = shares if QTY_LIKE.search(name) else MC
            out[name + " /시총"] = X / base
            out[name + " 5일합/시총"] = SP._roll(X, 5, "sum") / base
    for k in out:
        out[k][~np.isfinite(out[k])] = np.nan
    return out


def ic_stats(x, O, U, dates, h):
    fr = np.full_like(O, np.nan)
    fr[:-(1 + h)] = O[1 + h:] / O[1:-h] - 1
    a = np.where(U, x, np.nan)
    b = np.where(U, fr, np.nan)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 2000:
        return None
    ra = pd.DataFrame(np.where(ok, a, np.nan)).rank(axis=1, pct=True).to_numpy()
    rb = pd.DataFrame(np.where(ok, b, np.nan)).rank(axis=1, pct=True).to_numpy()
    with np.errstate(all="ignore"):
        am, bm = ra - np.nanmean(ra, 1, keepdims=True), rb - np.nanmean(rb, 1, keepdims=True)
        ic = np.nansum(am * bm, 1) / np.sqrt(np.nansum(am ** 2, 1) * np.nansum(bm ** 2, 1))
        top = np.nanmean(np.where(ra >= 0.9, b, np.nan), 1) - np.nanmean(np.where(ok, b, np.nan), 1)
    ic[ok.sum(1) < 30] = np.nan
    sub = np.arange(0, len(ic), h)
    v = ic[sub][~np.isnan(ic[sub])]
    if len(v) < 60:
        return None
    sp = top[sub][~np.isnan(top[sub])]
    yrs = pd.Series(ic, index=dates).dropna().groupby(lambda d: d.year).mean()
    sign = np.sign(v.mean())
    return {"ic": round(float(v.mean()), 4), "t": round(float(v.mean() / v.std() * np.sqrt(len(v))), 2),
            "top10_bp": round(float(np.mean(sp) * 1e4), 1), "n_days": int(len(v)),
            "same_sign_years": int((np.sign(yrs) == sign).sum()), "years": int(len(yrs)),
            "first": str(pd.Series(ic, index=dates).dropna().index.min().date())}


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    P = SP.load()
    dates, codes, O, U = P["dates"], P["codes"], P["O"], P["U"]
    res = {"time": time.strftime("%Y-%m-%d %H:%M"), "endpoints": {}, "useful": []}
    for ep in ENDPOINTS:
        mats = load_endpoint(ep, dates, codes)
        rows = []
        for name, X in mats.items():
            if np.isfinite(X).sum() < 5000:
                continue
            for vname, x in variants(name, X, P).items():
                row = {"signal": vname}
                for h in (1, 5, 10):
                    row[f"h{h}"] = ic_stats(x, O, U, dates, h)
                if any(row[f"h{h}"] for h in (1, 5, 10)):
                    rows.append(row)
                    for h in (1, 5, 10):
                        s = row[f"h{h}"]
                        if s and abs(s["t"]) >= 3 and s["same_sign_years"] >= max(5, int(0.75 * s["years"])):
                            res["useful"].append({"endpoint": ep, "signal": vname, "h": h, **s})
        res["endpoints"][ep] = {"fields": sorted(mats), "tested": rows}
        print(f"✓ {ep}: 필드 {len(mats)}개 · 신호 {len(rows)}개 시험", flush=True)
    res["seconds"] = round(time.time() - t0)
    res["note"] = "|t|≥3 이고 해마다 같은 부호(≥75%)인 것만 useful. 비용(주식 왕복 약 34bp)을 넘는지는 top10_bp 로 판단."
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"끝 · 쓸 만한 후보 {len(res['useful'])}개 → data/toss/lab.json ({res['seconds']}초)")
    for u in sorted(res["useful"], key=lambda x: -abs(x["t"]))[:10]:
        print(f"  {u['endpoint']} · {u['signal']} · {u['h']}일: IC {u['ic']:+.3f} (t {u['t']}) · 상위10% {u['top10_bp']}bp · {u['same_sign_years']}/{u['years']}년")


if __name__ == "__main__":
    main()
