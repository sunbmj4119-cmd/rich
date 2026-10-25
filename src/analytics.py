"""
종목 분석값 — 시중 프로그램들이 보여주는 '예측·분석 화면'을 이 데이터로 정직하게 만든다.
-> docs/analytics.json  (docs/stock.html · docs/autotrade.html · docs/features.html 가 읽는다)

만드는 것 (괄호 = 비슷한 것을 보여주는 시중 서비스)
  1) 팩터 성적표 A+~F, 섹터 내 상대평가      (Seeking Alpha Quant Ratings · 초이스스탁 스마트스코어)
  2) 점수 기여도 분해 — 왜 이 순위인가        (Danelfin '점수를 움직인 요인')
  3) 30거래일 예측 밴드 10/25/50/75/90%       (Tickeron 예측 범위 · 중앙은행 fan chart)
     + 밴드가 실제로 맞았는지 test 구간 적중률
  4) 순위 구간별 과거 성적표                  (Zacks Rank · Portfolio123 버킷 성과)
  5) 유사 차트 패턴 — 비슷했던 과거 구간과 그 뒤 30일   (알파스퀘어 · 딥서치 유사차트)
     + 이 방식에 예측력이 있는지 검증 결과 (없으면 '미검증'으로 표시)
  6) 섹터 히트맵 · 모델 포트폴리오 섹터/베타 노출  (Finviz map · Stock Rover)

원칙
  · 모든 확률·밴드는 '이 순위·이 국면에 있던 과거 전체 종목'에서 나온 빈도다 (한 종목의 과거 아님).
  · 표본 수와 독립 표본 수(겹치지 않는 30일 묶음)를 같이 내보낸다.
  · 검증에서 예측력이 없던 값은 숫자를 지우지 않고 '미검증' 꼬리표를 붙인다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
import features as F            # noqa: E402

OUT = "docs/analytics.json"
FWD = 30
SPLIT = pd.Timestamp("2024-01-01")
WARMUP = 260
HIST_DAYS = 160                 # 차트용 가격 이력 길이

# (weights.yaml 키, scores.csv 열, 화면 이름)
FACTORS = [("value", "s_value", "가치"), ("profit", "s_profit", "수익성"),
           ("flow", "s_flow", "외국인 수급"), ("growth", "s_grow", "성장"),
           ("rmom", "s_rmom", "위험조정 모멘텀"), ("momentum", "s_mom", "모멘텀"),
           ("stability", "s_stab", "재무 안정성"), ("short", "s_short", "공매도 적음")]
RANK_BANDS = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
QS = [0.10, 0.25, 0.50, 0.75, 0.90]
GRADES = [(90, "A+"), (80, "A"), (70, "B+"), (60, "B"), (50, "C+"), (40, "C"), (30, "D+"),
          (20, "D"), (0, "F")]


def grade(p):
    for lim, g in GRADES:
        if p >= lim:
            return g
    return "F"


def band_of(r):
    for i, (a, b) in enumerate(RANK_BANDS):
        if a <= r <= b:
            return i
    return len(RANK_BANDS) - 1


def spearman(a, b):
    """순위상관 (scipy 없이 — Actions 환경에 scipy가 없다)."""
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return a.corr(b)


def r1(x, k=1):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


# ══════════════════════════════════════════════════════════════
# 1·2) 팩터 성적표 · 기여도
# ══════════════════════════════════════════════════════════════
def factor_cards(s_last, weights, sector_of):
    """s_last: 최신일 scores 행들. 섹터 내 백분위로 등급, 가중치×(점수−50)으로 기여도."""
    df = s_last.set_index("종목코드")
    df["섹터"] = [sector_of.get(c, "기타") for c in df.index]
    out = {}
    for c in df.index:
        out[c] = {"grades": {}, "contrib": []}
    for key, col, label in FACTORS:
        if col not in df.columns:
            continue
        mkt = df[col].rank(pct=True) * 100
        sec = df.groupby("섹터")[col].rank(pct=True) * 100
        n_sec = df.groupby("섹터")[col].transform("count")
        sec = sec.where(n_sec >= 3, mkt)                    # 섹터 종목이 3개 미만이면 시장 전체와 비교
        flat = df[col].nunique() <= 1                       # 전 종목 같은 값 = 데이터 없음(중립 채움)
        w = float(weights.get(key, 0) or 0)
        for c in df.index:
            v = float(df.at[c, col])
            out[c]["grades"][key] = {
                "label": label, "score": r1(v), "weight": w,
                "sector_pct": r1(sec[c], 0), "market_pct": r1(mkt[c], 0),
                "grade": "—" if flat else grade(sec[c]), "no_data": bool(flat)}
            if w > 0:
                out[c]["contrib"].append({"key": key, "label": label, "weight": w, "score": r1(v),
                                          "contrib": round(w * (v - 50), 2), "no_data": bool(flat)})
    return out


# ══════════════════════════════════════════════════════════════
# 3·4) 예측 밴드 · 순위 성적표
# ══════════════════════════════════════════════════════════════
def outcome_table(panel, feats):
    """(날짜, 종목)별 순위밴드·국면·30일 수익·시장 대비 초과·변동성 표준화 수익."""
    close = panel["close"]
    mk = panel["market"]
    fwd = close.shift(-FWD) / close - 1
    mfwd = mk["idx"].shift(-FWD) / mk["idx"] - 1
    vol = feats["vol20"] * np.sqrt(FWD)
    rank = feats["rank"]
    t = pd.DataFrame({
        "date": np.repeat(close.index.values, close.shape[1]),
        "code": np.tile(close.columns.values, close.shape[0]),
        "rank": rank.values.ravel(), "fwd": fwd.values.ravel(), "vol": vol.values.ravel(),
    })
    t["mfwd"] = np.repeat(mfwd.values, close.shape[1])
    t["regime"] = np.repeat(mk["regime"].values, close.shape[1])
    t = t.dropna(subset=["rank", "fwd", "vol", "regime"])
    t = t[(t["vol"] > 0) & (t["date"] >= close.index[WARMUP])]
    t["band"] = t["rank"].astype(int).map(band_of)
    t["z"] = t["fwd"] / t["vol"]
    t["excess"] = t["fwd"] - t["mfwd"]
    return t


def eff_n(dates):
    """겹치지 않는 30일 묶음 수 — 같은 날 여러 종목은 시장 요인으로 같이 움직인다."""
    d = np.sort(pd.unique(dates))
    n, last = 0, None
    for x in d:
        if last is None or (x - last) >= np.timedelta64(42, "D"):
            n += 1
            last = x
    return n


def band_model(tab):
    """train 구간으로 (순위밴드 × 국면) 칸의 표준화 수익 분위수. 칸이 얇으면 밴드 전체로 대체."""
    tr = tab[tab["date"] < SPLIT]
    cells = {}
    for (b, g), sub in tr.groupby(["band", "regime"]):
        cells[(b, g)] = {"q": sub["z"].quantile(QS).values, "n": len(sub)}
    for b, sub in tr.groupby("band"):
        cells[(b, "*")] = {"q": sub["z"].quantile(QS).values, "n": len(sub)}
    return cells


def pick_cell(cells, b, g):
    c = cells.get((b, g))
    if c and c["n"] >= 2000:
        return c, f"{RANK_BANDS[b][0]}-{RANK_BANDS[b][1]}위 × {g}"
    return cells[(b, "*")], f"{RANK_BANDS[b][0]}-{RANK_BANDS[b][1]}위 (전 국면)"


def band_calibration(tab, cells):
    """train 분위수로 만든 밴드가 test 구간에서 실제로 몇 %를 담았나 (80% 밴드면 80% 근처여야 정상)."""
    te = tab[tab["date"] >= SPLIT].dropna(subset=["z"])
    lo10, lo25, hi75, hi90 = [], [], [], []
    for (b, g), sub in te.groupby(["band", "regime"]):
        c, _ = pick_cell(cells, b, g)
        q = c["q"]
        lo10.append((sub["z"] >= q[0]).sum())
        hi90.append((sub["z"] <= q[4]).sum())
        lo25.append(((sub["z"] >= q[1]) & (sub["z"] <= q[3])).sum())
        hi75.append(((sub["z"] >= q[0]) & (sub["z"] <= q[4])).sum())
    n = len(te)
    return {"n": int(n), "cover80": r1(sum(hi75) / n * 100), "cover50": r1(sum(lo25) / n * 100),
            "below10": r1(100 - sum(lo10) / n * 100), "above90": r1(100 - sum(hi90) / n * 100)}


def rank_track(tab):
    """순위 10구간별 30일 수익·시장 초과·초과확률. train/test 따로 — 두 구간에서 순서가 유지되면 믿을 만하다."""
    out = {}
    tab = tab.assign(dec=((tab["rank"] - 1) // 10).clip(0, 9).astype(int))
    for nm, sub in [("train", tab[tab["date"] < SPLIT]), ("test", tab[tab["date"] >= SPLIT])]:
        rows = []
        for d, g in sub.groupby("dec"):
            rows.append({"bucket": f"{d*10+1}-{d*10+10}위", "avg": r1(g["fwd"].mean() * 100, 2),
                         "excess": r1(g["excess"].mean() * 100, 2),
                         "beat": r1((g["excess"] > 0).mean() * 100),
                         "up": r1((g["fwd"] > 0).mean() * 100), "n": int(len(g)),
                         "n_eff": eff_n(g["date"].values)})
        out[nm] = rows
        a = sub.groupby("dec")["excess"].mean()
        out[nm + "_spearman"] = r1(spearman(a.values, -a.index.values), 2)
    return out


# ══════════════════════════════════════════════════════════════
# 5) 유사 차트 패턴
# ══════════════════════════════════════════════════════════════
WIN, STEP_LIB, NPTS = 60, 3, 30


def _shape(seg):
    """60일 종가 → 로그가격 30점 → 평균0·표준편차1. 모양만 비교(수준·변동폭 무시)."""
    x = np.log(seg[np.linspace(0, len(seg) - 1, NPTS).astype(int)])
    x = x - x.mean()
    s = x.std()
    return x / s if s > 0 else None


def pattern_library(close, end_limit):
    """end_limit(인덱스) 이전에 결과(30일 뒤)까지 확인된 구간만 도서관에 넣는다."""
    vals = close.values
    shapes, meta = [], []
    for j in range(close.shape[1]):
        col = vals[:, j]
        for e in range(WIN, min(end_limit, len(col) - FWD), STEP_LIB):
            seg = col[e - WIN:e]
            nxt = col[e - 1 + FWD]
            if not (np.isfinite(seg).all() and np.isfinite(nxt)):
                continue
            sh = _shape(seg)
            if sh is None:
                continue
            shapes.append(sh)
            meta.append((j, e - 1, nxt / seg[-1] - 1))
    return np.array(shapes), np.array(meta, dtype=float)


def analogs(lib, meta, query, k=20, exclude=None):
    sims = lib @ query / NPTS                       # 표준화된 벡터의 내적/길이 = 상관계수
    if exclude is not None:
        sims = np.where(exclude, -2, sims)
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def pattern_validation(close, dates):
    """train 도서관으로 test 구간 질의 → 유사구간 평균 30일 수익이 실제 30일 수익을 맞히나."""
    split = int(np.searchsorted(dates.values, np.datetime64(SPLIT)))
    lib, meta = pattern_library(close, split - FWD)
    rng = np.random.default_rng(7)
    vals = close.values
    preds, acts = [], []
    for _ in range(1500):
        j = rng.integers(close.shape[1])
        e = rng.integers(split + WIN, len(dates) - FWD)
        seg = vals[e - WIN:e, j]
        nxt = vals[e - 1 + FWD, j]
        if not (np.isfinite(seg).all() and np.isfinite(nxt)):
            continue
        q = _shape(seg)
        if q is None:
            continue
        idx, _ = analogs(lib, meta, q)
        preds.append(np.median(meta[idx, 2]))
        acts.append(nxt / seg[-1] - 1)
    p, a = np.array(preds), np.array(acts)
    ic = spearman(p, a)
    hit = ((p > 0) == (a > 0)).mean()
    base = (a > 0).mean()
    return {"n": int(len(p)), "ic": r1(ic, 3), "hit": r1(hit * 100), "base_up": r1(base * 100),
            "verdict": "검증됨" if (ic is not None and ic > 0.03) else "미검증"}


# ══════════════════════════════════════════════════════════════
def main():
    panel, feats = F.build_all()
    dates, codes, close = panel["dates"], panel["codes"], panel["close"]
    t = len(dates) - 1
    asof = dates[t]
    mk = panel["market"]
    regime = mk["regime"].iloc[t] if isinstance(mk["regime"].iloc[t], str) else "판정불가"
    weights = yaml.safe_load(open("config/weights.yaml", encoding="utf-8"))["logic"]
    sector_of = panel["sector_of"]

    s = pd.read_csv(F.SCORES, dtype={"종목코드": str})
    s["종목코드"] = s["종목코드"].str.zfill(6)
    s_last = s[s["날짜"] == s["날짜"].max()].copy()
    cards = factor_cards(s_last, weights, sector_of)

    print("예측 밴드·순위 성적표 계산…")
    tab = outcome_table(panel, feats)
    cells = band_model(tab)
    calib = band_calibration(tab, cells)
    track = rank_track(tab)

    print("유사 패턴 검증…")
    pval = pattern_validation(close, dates)
    lib, meta = pattern_library(close, t + 1)

    betas = json.load(open("data/regime.json", encoding="utf-8")).get("betas", {}) \
        if os.path.exists("data/regime.json") else {}
    rank_t = feats["rank"].iloc[t]
    vol_t = feats["vol20"].iloc[t]
    stocks = {}
    for c in codes:
        r = rank_t.get(c)
        px = close[c].iloc[t]
        if not (np.isfinite(r) and np.isfinite(px)):
            continue
        r = int(r)
        b = band_of(r)
        cell, cell_name = pick_cell(cells, b, regime)
        sig = vol_t.get(c) * np.sqrt(FWD) if np.isfinite(vol_t.get(c, np.nan)) else None
        band = None
        if sig:
            rets = [float(z * sig) for z in cell["q"]]
            band = {"ret": [round(x * 100, 1) for x in rets],
                    "price": [round(px * (1 + x)) for x in rets],
                    "cell": cell_name, "n": int(cell["n"]), "sigma30": round(sig * 100, 1)}
        hist = close[c].iloc[max(0, t - HIST_DAYS + 1): t + 1]
        item = {"name": panel["names"].get(c, c), "sector": sector_of.get(c, "기타"),
                "rank": r, "score": r1(panel["score"][c].iloc[t]), "close": float(px),
                "ret20": r1(feats["ret20"][c].iloc[t] * 100),
                "beta": betas.get(c, {}).get("beta"), "vol": betas.get(c, {}).get("vol"),
                "band": band, **cards.get(c, {}),
                "hist": {"d": [str(x.date()) for x in hist.index[::2]],
                         "c": [r1(x, 0) for x in hist.values[::2]],
                         "ma20": [r1(x, 0) for x in feats["ma20"][c].iloc[max(0, t - HIST_DAYS + 1):t + 1].values[::2]],
                         "ma200": [r1(x, 0) for x in feats["ma200"][c].iloc[max(0, t - HIST_DAYS + 1):t + 1].values[::2]]}}
        # 유사 패턴 — 상위 30위만 (계산량·화면 길이)
        if r <= 30:
            seg = close[c].values[t - WIN + 1: t + 1]
            q = _shape(seg) if np.isfinite(seg).all() else None
            if q is not None:
                j = codes.index(c)
                excl = (meta[:, 0] == j) & (meta[:, 1] > t - WIN - FWD)   # 자기 자신의 최근 구간 제외
                idx, sims = analogs(lib, meta, q, k=20, exclude=excl)
                outs = meta[idx, 2]
                item["analog"] = {
                    "sim": r1(float(sims.mean()), 2), "median": r1(float(np.median(outs)) * 100),
                    "up": r1(float((outs > 0).mean()) * 100),
                    "p25": r1(float(np.percentile(outs, 25)) * 100), "p75": r1(float(np.percentile(outs, 75)) * 100),
                    "cases": [{"name": panel["names"].get(codes[int(meta[i, 0])], ""),
                               "date": str(dates[int(meta[i, 1])].date()),
                               "sim": r1(float(sims[k]), 2), "ret": r1(float(meta[i, 2]) * 100)}
                              for k, i in enumerate(idx[:8])]}
        stocks[c] = item

    # 섹터 히트맵
    sec_rows = []
    sec = pd.Series(sector_of)
    for nm, cs in sec.groupby(sec):
        cs = [x for x in cs.index if x in stocks]
        if not cs:
            continue
        sec_rows.append({"sector": nm, "n": len(cs),
                         "ret1": r1(np.nanmean([close[x].pct_change().iloc[t] * 100 for x in cs])),
                         "ret20": r1(np.nanmean([stocks[x]["ret20"] or np.nan for x in cs])),
                         "score": r1(np.nanmean([stocks[x]["score"] for x in cs])),
                         "top10": sum(1 for x in cs if stocks[x]["rank"] <= 10)})
    sec_rows.sort(key=lambda x: -x["n"])

    out = {"asof": str(asof.date()), "regime": regime, "fwd": FWD,
           "factors": [{"key": k, "label": l, "weight": float(weights.get(k, 0) or 0)} for k, _, l in FACTORS],
           "flow_no_data": bool(s_last["s_flow"].nunique() <= 1),
           "band_calibration": calib, "rank_track": track, "pattern_validation": pval,
           "sectors": sec_rows, "stocks": stocks}
    os.makedirs("docs", exist_ok=True)
    # 화면 기능 기본값 (config/features.yaml) → 페이지들이 읽는다
    fy = yaml.safe_load(open("config/features.yaml", encoding="utf-8")) if os.path.exists("config/features.yaml") else {}
    json.dump(fy or {}, open("docs/features.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"),
              default=lambda o: o.item() if hasattr(o, "item") else None)
    print(f"저장: {OUT} ({os.path.getsize(OUT)//1024}KB) · 종목 {len(stocks)}")
    print(f"  밴드 적중(test): 80%밴드 {calib['cover80']}% · 50%밴드 {calib['cover50']}% (n={calib['n']:,})")
    print(f"  순위 성적표 순서 일관성(Spearman): train {track['train_spearman']} · test {track['test_spearman']}")
    print(f"  유사패턴 검증: IC {pval['ic']} · 방향적중 {pval['hit']}% (기본 상승률 {pval['base_up']}%) → {pval['verdict']}")


if __name__ == "__main__":
    main()
