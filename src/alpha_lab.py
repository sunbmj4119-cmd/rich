"""
예측력 연구실 — 지금 있는 데이터만으로 '30거래일 뒤 상대수익'을 더 잘 맞히는 순위를 찾는다.
-> data/alpha_lab.json

왜 따로 만드나
  지금 종합점수의 가중치(config/weights.yaml)는 2018~2026 전체를 보고 정했다. 즉 기준선 자체가
  미래를 조금 본 상태다. 새 방법은 **워크포워드**로만 만든다 — 매달 그 시점에 결과가 확정된
  과거(30거래일 이전까지)로만 학습하고 다음 달을 예측한다. 그래도 이기면 진짜 개선이다.

평가 (같은 날짜·같은 종목에서 비교)
  · 순위 IC: 예측 순위와 30일 뒤 시장대비 수익 순위의 상관 (5거래일마다 표본)
  · t값: 30일 창이 겹치므로 Newey-West(시차 6)로 보정
  · 상위 10종목 30일 초과수익, 상위−하위 10분위 차이
  · 기준선 대비 차이(ΔIC)의 t값 — 이게 '유의미하게 올랐나'의 답
  · 여러 방법을 시험했으므로 Bonferroni 문턱(시험 수 m에 대해 양측 0.05/m)도 같이 본다

결합 방법 (모두 워크포워드)
  BASE    현재 종합점수
  EQ      과거 IC t값 > 2 인 신호만 골라 동일가중
  ICW     과거 IC 평균 ÷ 표준편차(정보비율)로 가중, 음수는 0
  RIDGE   순위 특성 → 30일 뒤 순위 회귀 (릿지), 매달 다시 학습
사용: python src/alpha_lab.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import features as F          # noqa: E402

OUT = "data/alpha_lab.json"
CACHE = "data/cache"            # 워크포워드 예측 (prediction_lab.py 가 읽음 · 저장소엔 안 올림)
FWD = 30
STEP = 5                       # 5거래일마다 표본
REFIT = 21                     # 한 달마다 다시 학습
MIN_TRAIN_DAYS = 504           # 학습에 최소 2년치 표본
SPLIT = pd.Timestamp("2024-01-01")
NW_LAG = math.ceil(FWD / STEP)


# ══════════════════════════════════════════════════════════════
# 통계 도구
# ══════════════════════════════════════════════════════════════
def nw_t(x, lag=NW_LAG):
    """Newey-West 보정 t값 — 겹치는 표본의 자기상관을 반영."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    s = (e @ e) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def xs_rank(df):
    """그날 종목 간 순위를 −0.5~+0.5 로. 결측은 0(중립)."""
    r = df.rank(axis=1, pct=True) - 0.5
    return r.where(df.notna(), 0.0)


def rank_ic(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    x, y = pd.Series(a[m]).rank(), pd.Series(b[m]).rank()
    return float(x.corr(y))


# ══════════════════════════════════════════════════════════════
# 후보 신호 — 모두 '클수록 좋다' 방향으로 맞춘다
# ══════════════════════════════════════════════════════════════
def build_signals(p, f):
    close, vol = p["close"], p["volume"]
    dates, codes = p["dates"], p["codes"]
    r = close.pct_change(fill_method=None)
    rm = r.mean(axis=1)
    S = {}
    # ── 가격·위험 ──
    S["mom12_1"] = close.shift(21) / close.shift(252) - 1
    v60 = r.rolling(60, min_periods=40).std()
    S["rmom"] = S["mom12_1"] / (v60 * np.sqrt(252))
    S["mom6_1"] = (close.shift(21) / close.shift(126) - 1) / (v60 * np.sqrt(252))
    cov = (r.mul(rm, axis=0)).rolling(252, min_periods=150).mean() - r.rolling(252, min_periods=150).mean().mul(rm.rolling(252, min_periods=150).mean(), axis=0)
    beta = cov.div(rm.rolling(252, min_periods=150).var(), axis=0)
    resid = r - beta.shift(1).mul(rm, axis=0)
    rs = resid.rolling(231, min_periods=150).sum().shift(21)
    S["resid_mom"] = rs / (resid.rolling(252, min_periods=150).std() * np.sqrt(231))      # Blitz 외 2011
    S["rev20"] = -close.pct_change(20, fill_method=None)                                    # 단기 반전
    S["lowvol"] = -v60
    S["lowbeta"] = -beta                                                                    # Frazzini-Pedersen
    S["ivol"] = -resid.rolling(60, min_periods=40).std()                                    # Ang 외 2006
    S["max21"] = -r.rolling(21, min_periods=15).max()                                       # Bali 외 2011 (복권주 회피)
    S["hi52"] = f["hi52"]
    S["ma200gap"] = close / f["ma200"] - 1
    S["vol_trend"] = -(vol.rolling(20, min_periods=10).mean() / vol.rolling(120, min_periods=60).mean())
    S["disp20"] = -f["disp20"]
    # ── 수급 ──
    fl = pd.read_csv(F.FLOWS, dtype={"종목코드": str})
    fl["종목코드"] = fl["종목코드"].str.zfill(6)
    fl["날짜"] = pd.to_datetime(fl["날짜"])
    shares = None
    fin = pd.read_csv(F.FIN, dtype={"종목코드": str})
    fin["종목코드"] = fin["종목코드"].str.zfill(6)
    fin["사용가능일"] = pd.to_datetime(fin["사용가능일"])
    shares = F.asof_matrix(fin.assign(sh=fin["주식수"]), "sh", dates, codes)
    mcap = close * shares                                   # 시총 = 주가 × 주식수 (시총 파일이 8/7에서 멈춰 직접 계산)
    for col, nm in [("외국인순매수", "frgn"), ("기관순매수", "inst"), ("개인순매수", "indv")]:
        m = fl.pivot_table(index="날짜", columns="종목코드", values=col).reindex(index=dates, columns=codes)
        S[f"{nm}20"] = m.rolling(20, min_periods=10).sum() / mcap
        S[f"{nm}60"] = m.rolling(60, min_periods=30).sum() / mcap
        if nm == "frgn":
            S["frgn_persist"] = (m > 0).astype(float).where(m.notna()).rolling(20, min_periods=10).mean()
    S["indv20"] = -S["indv20"]                              # 개인 순매수는 역지표 가설
    S["indv60"] = -S["indv60"]
    # ── 재무 (공시 후 사용가능일 기준) ──
    q = F.quarterly_standalone(fin).sort_values(["종목코드", "연도", "q"])
    g = q.groupby("종목코드")
    for col in ["매출액", "영업이익", "당기순이익"]:
        q[col + "_ttm"] = g[col + "_q"].transform(lambda x: x.rolling(4, min_periods=4).sum())
    q["asset_g"] = g["자산총계"].pct_change(4, fill_method=None)
    tbl = q[["종목코드", "사용가능일", "매출액_ttm", "영업이익_ttm", "당기순이익_ttm", "자본총계", "자산총계", "부채총계", "asset_g"]]
    A = {c: F.asof_matrix(tbl, c, dates, codes) for c in tbl.columns[2:]}
    S["ep"] = A["당기순이익_ttm"] / mcap                       # 이익수익률 (1/PER)
    S["bp"] = A["자본총계"] / mcap                              # 1/PBR
    S["sp"] = A["매출액_ttm"] / mcap                            # 1/PSR
    S["op_y"] = A["영업이익_ttm"] / mcap                        # 영업이익수익률
    S["roe"] = A["당기순이익_ttm"] / A["자본총계"].where(A["자본총계"] > 0)
    S["opm"] = A["영업이익_ttm"] / A["매출액_ttm"].where(A["매출액_ttm"] > 0)
    S["roa"] = A["당기순이익_ttm"] / A["자산총계"]
    S["lowlev"] = -(A["부채총계"] / A["자산총계"])
    S["low_asset_g"] = -A["asset_g"]                            # 투자 팩터 (Fama-French CMA)
    for k in ["sue", "op_yoy", "fscore"]:
        if k in f:
            S[k] = f[k]
    # ── 섹터 ──
    S["sec_mom"] = f["sec_mom_rank"]
    sec = pd.Series(p["sector_of"]).reindex(codes)
    groups = [list(v.index) for _, v in sec.groupby(sec) if len(v) >= 3]

    def neutral(m):
        out = pd.DataFrame(np.nan, index=m.index, columns=m.columns)
        for cols in groups:
            out[cols] = m[cols].rank(axis=1, pct=True)
        return out.fillna(m.rank(axis=1, pct=True))
    S["ep_sn"] = neutral(S["ep"])
    S["bp_sn"] = neutral(S["bp"])
    S["rmom_sn"] = neutral(S["rmom"])
    # ── 기존 점수 구성요소 ──
    for k in ["s_value", "s_profit", "s_grow", "s_flow", "s_rmom"]:
        if k in p:
            S[k] = p[k]
    S["BASE"] = p["score"]
    return {k: v.reindex(index=dates, columns=codes).astype(float) for k, v in S.items()}


# ══════════════════════════════════════════════════════════════
def main():
    p, f = F.build_all()
    dates, codes = p["dates"], p["codes"]
    close = p["close"]
    fwd = close.shift(-FWD) / close - 1
    fwd_x = fwd.sub(fwd.mean(axis=1), axis=0)              # 시장(등가중) 대비
    S = build_signals(p, f)
    names = [k for k in S if k != "BASE"]
    print(f"후보 신호 {len(names)}개 · 기간 {dates[0].date()} ~ {dates[-1].date()}")

    idx = list(range(260, len(dates) - FWD, STEP))
    R = {k: xs_rank(S[k]).values for k in S}
    Y = xs_rank(fwd_x).values
    YV = fwd_x.values

    # ── 1) 단일 신호 IC ─────────────────────────────────────
    ic = {k: np.array([rank_ic(S[k].values[t], YV[t]) for t in idx]) for k in S}
    dts = dates[idx]
    single = []
    for k in S:
        x = ic[k]
        tr, te = x[dts < SPLIT], x[dts >= SPLIT]
        yrs = pd.Series(x, index=dts).groupby(dts.year).mean()
        single.append({"name": k, "ic": round(float(np.nanmean(x)), 4), "t": round(nw_t(x), 2),
                       "ic_train": round(float(np.nanmean(tr)), 4), "ic_test": round(float(np.nanmean(te)), 4),
                       "pos_years": f"{int((yrs > 0).sum())}/{len(yrs)}",
                       "coverage": round(float(np.isfinite(S[k].values[idx]).mean()), 2)})
    single.sort(key=lambda d: -d["t"] if np.isfinite(d["t"]) else 0)
    print("\n■ 단일 신호 (전체 기간 IC · NW t값 · 2019~23 / 2024~ · 양수 연도)")
    for d in single:
        print(f"  {d['name']:<13} IC {d['ic']:+.4f}  t {d['t']:+5.2f}  │ {d['ic_train']:+.4f} / {d['ic_test']:+.4f}  │ {d['pos_years']}  표본 {d['coverage']:.0%}")

    # ── 2) 워크포워드 결합 ───────────────────────────────────
    K = len(names)
    X_all = np.stack([R[k] for k in names], axis=2)        # (날짜, 종목, 신호)
    preds = {m: np.full(close.shape, np.nan) for m in ["EQ", "ICW", "RIDGE"]}
    weights_log = []
    t0 = idx[0] + MIN_TRAIN_DAYS
    for start in range(t0, len(dates), REFIT):
        cut = start - FWD                                   # 결과가 확정된 마지막 날
        tr_idx = [t for t in idx if t <= cut]
        if len(tr_idx) < 60:
            continue
        # 과거 IC (학습 구간만)
        icm = np.array([[ic[k][i] for i, t in enumerate(idx) if t <= cut] for k in names])
        mu = np.nanmean(icm, axis=1)
        sd = np.nanstd(icm, axis=1)
        tv = np.array([nw_t(row) for row in icm])
        w_eq = np.where(tv > 2, 1.0, 0.0)
        w_icw = np.clip(mu / np.where(sd > 0, sd, np.nan), 0, None)
        w_icw = np.nan_to_num(w_icw)
        # 릿지: 쌓은 표본으로 순위 → 순위 회귀
        Xt = X_all[tr_idx].reshape(-1, K)
        yt = Y[tr_idx].reshape(-1)
        ok = np.isfinite(yt) & (np.abs(yt) > 0)
        Xt, yt = Xt[ok], yt[ok]
        lam = 0.05 * len(yt)                                # 강한 규제 — 과적합 방지
        w_r = np.linalg.solve(Xt.T @ Xt + lam * np.eye(K), Xt.T @ yt)
        end = min(start + REFIT, len(dates))
        for m, w in [("EQ", w_eq), ("ICW", w_icw), ("RIDGE", w_r)]:
            if np.abs(w).sum() == 0:
                continue
            preds[m][start:end] = X_all[start:end] @ w
        weights_log.append({"date": str(dates[start].date()),
                            "ridge": {k: round(float(v), 4) for k, v in zip(names, w_r)},
                            "icw": {k: round(float(v), 3) for k, v in zip(names, w_icw) if v > 0},
                            "eq": [k for k, v in zip(names, w_eq) if v > 0]})

    # ── 3) 같은 날짜로 비교 ─────────────────────────────────
    eval_idx = [t for t in idx if t >= t0]
    ed = dates[eval_idx]
    M = {"BASE": S["BASE"].values, **preds}
    res, ics = {}, {}
    for m, P in M.items():
        x = np.array([rank_ic(P[t], YV[t]) for t in eval_idx])
        ics[m] = x
        top, spread = [], []
        for t in eval_idx:
            a, b = P[t], YV[t]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 30:
                continue
            o = np.argsort(-a[ok])
            yb = b[ok][o]
            top.append(yb[:10].mean())
            n10 = max(1, len(yb) // 10)
            spread.append(yb[:n10].mean() - yb[-n10:].mean())
        res[m] = {"ic": round(float(np.nanmean(x)), 4), "t": round(nw_t(x), 2),
                  "ic_pre": round(float(np.nanmean(x[ed < SPLIT])), 4), "ic_post": round(float(np.nanmean(x[ed >= SPLIT])), 4),
                  "top10_excess": round(float(np.mean(top)) * 100, 2), "decile_spread": round(float(np.mean(spread)) * 100, 2),
                  "top10_t": round(nw_t(top), 2)}
    crit = 2.39                                              # 결합 방법 3개 → 양측 0.05/3 의 정규 임계값
    print(f"\n■ 워크포워드 결합 — 평가 {ed[0].date()} ~ {ed[-1].date()} ({len(eval_idx)}개 날짜, 모든 방법 같은 날짜)")
    for m, d in res.items():
        dlt = ics[m] - ics["BASE"]
        d["delta_ic"] = round(float(np.nanmean(dlt)), 4)
        d["delta_t"] = round(nw_t(dlt), 2) if m != "BASE" else None
        d["significant"] = bool(m != "BASE" and d["delta_t"] is not None and d["delta_t"] > crit)
        print(f"  {m:<6} IC {d['ic']:+.4f} (t {d['t']:+.2f}) │ ~2023 {d['ic_pre']:+.4f} · 2024~ {d['ic_post']:+.4f} │ "
              f"상위10 초과 {d['top10_excess']:+.2f}% (t {d['top10_t']:+.2f}) │ 10분위 차 {d['decile_spread']:+.2f}%"
              + (f" │ ΔIC {d['delta_ic']:+.4f} t {d['delta_t']:+.2f} {'✅ 유의' if d['significant'] else '—'}" if m != "BASE" else ""))

    # ── 4) 시도 3: 상위 종목 기준 워크포워드 ─────────────────
    # 시도 2의 교훈: IC가 높은 저위험 신호는 '떨어질 종목'을 맞힐 뿐 상위 10을 개선하지 못했다.
    # 이 전략은 상위 10만 사므로, 학습 목표를 '상위 10 초과수익'으로 바꾼다. 신호 선택도 워크포워드 안에서.
    def top10_series(P):
        out = np.full(len(idx), np.nan)
        for i, t in enumerate(idx):
            a, y = P[t], YV[t]
            ok = np.isfinite(a) & np.isfinite(y)
            if ok.sum() >= 30:
                out[i] = y[ok][np.argsort(-a[ok])][:10].mean()
        return out
    T10 = {k: top10_series(S[k].values) for k in S}
    idx_arr = np.array(idx)
    top_preds = {m: np.full(close.shape, np.nan) for m in ["TOP_EQ", "TOP_TW", "TOP_RIDGE"]}
    top_log = []
    for start in range(t0, len(dates), REFIT):
        cut = start - FWD
        msk = idx_arr <= cut
        if msk.sum() < 60:
            continue
        tv = {k: nw_t(T10[k][msk]) for k in names}
        sel = [k for k in names if np.isfinite(tv[k]) and tv[k] > 2.0]
        end = min(start + REFIT, len(dates))
        base_r = R["BASE"][start:end]
        if sel:
            eq = np.mean([R[k][start:end] for k in sel], axis=0)
            top_preds["TOP_EQ"][start:end] = 0.5 * base_r + 0.5 * eq
            tw = np.array([tv[k] for k in sel])
            tws = np.tensordot(tw / tw.sum(), np.stack([R[k][start:end] for k in sel]), axes=1)
            top_preds["TOP_TW"][start:end] = 0.5 * base_r + 0.5 * tws
        else:
            top_preds["TOP_EQ"][start:end] = base_r
            top_preds["TOP_TW"][start:end] = base_r
        # 상위 20%에 드는지를 맞히는 릿지 (BASE 포함 모든 신호)
        tr_idx = [t for t in idx if t <= cut]
        Xt = np.concatenate([X_all[tr_idx], R["BASE"][tr_idx][..., None]], axis=2).reshape(-1, K + 1)
        yt = (Y[tr_idx] > 0.3).astype(float).reshape(-1) - 0.2
        ok = np.isfinite(Y[tr_idx].reshape(-1)) & (np.abs(Y[tr_idx].reshape(-1)) > 0)
        w = np.linalg.solve(Xt[ok].T @ Xt[ok] + 0.05 * ok.sum() * np.eye(K + 1), Xt[ok].T @ yt[ok])
        top_preds["TOP_RIDGE"][start:end] = np.concatenate([X_all[start:end], R["BASE"][start:end][..., None]], axis=2) @ w
        top_log.append({"date": str(dates[start].date()), "selected": sel})
    print("\n■ 시도 3 — 상위10 기준 워크포워드 (신호 선택도 매달 과거 자료로만)")
    b10 = top10_series(S["BASE"].values)[[i for i, t in enumerate(idx) if t >= t0]]
    for m, P in top_preds.items():
        s10 = top10_series(P)[[i for i, t in enumerate(idx) if t >= t0]]
        d = s10 - b10
        x = np.array([rank_ic(P[t], YV[t]) for t in eval_idx])
        res[m] = {"ic": round(float(np.nanmean(x)), 4), "t": round(nw_t(x), 2),
                  "top10_excess": round(float(np.nanmean(s10)) * 100, 2), "top10_t": round(nw_t(s10), 2),
                  "delta_top10": round(float(np.nanmean(d)) * 100, 2), "delta_top10_t": round(nw_t(d), 2),
                  "pre": round(float(np.nanmean(d[ed < SPLIT])) * 100, 2), "post": round(float(np.nanmean(d[ed >= SPLIT])) * 100, 2)}
        r_ = res[m]
        print(f"  {m:<10} 상위10 초과 {r_['top10_excess']:+.2f}% (t {r_['top10_t']:+.2f}) │ BASE 대비 Δ {r_['delta_top10']:+.2f}%p "
              f"(t {r_['delta_top10_t']:+.2f}) │ ~2023 {r_['pre']:+.2f} · 2024~ {r_['post']:+.2f} │ IC {r_['ic']:+.4f}")
    if top_log:
        from collections import Counter
        cnt = Counter(k for e in top_log for k in e["selected"])
        print("  자주 선택된 신호: " + " · ".join(f"{k} {v}/{len(top_log)}" for k, v in cnt.most_common(10)))
    preds.update(top_preds)

    # 마지막 릿지 가중치 (해석용)
    last = weights_log[-1]["ridge"] if weights_log else {}
    top_w = sorted(last.items(), key=lambda kv: -abs(kv[1]))[:12]
    print("\n■ 마지막 릿지 가중치 (절댓값 상위)")
    print("  " + " · ".join(f"{k} {v:+.3f}" for k, v in top_w))

    out = {"fwd": FWD, "step": STEP, "refit": REFIT, "split": str(SPLIT.date()),
           "eval": [str(ed[0].date()), str(ed[-1].date())], "n_dates": len(eval_idx),
           "bonferroni_t": crit, "single": single, "combo": res, "weights": weights_log[-1] if weights_log else {}}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 엔진 백테스트에 쓸 수 있게 워크포워드 예측을 저장
    os.makedirs(CACHE, exist_ok=True)
    for m in preds:
        pd.DataFrame(preds[m], index=dates, columns=codes).to_pickle(os.path.join(CACHE, f"alpha_{m}.pkl"))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
