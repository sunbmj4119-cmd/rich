"""
보완 백테스트 — A·B 약점 정량화
  A. 생존편향: 시점별 '그날 실제 데이터가 있던 종목'만 유니버스로 사용(point-in-time).
              + 2018년부터 존재한 종목만 쓰는 보수 모드 비교.
  B. 비용 현실화: 왕복 비용을 0.15%→0.35%→0.50%로 바꿔가며 수익 감소 측정.
출력: 콘솔 + docs/robust.json (대시보드 표시용)
"""
import os, json
import pandas as pd
import numpy as np

SCORES = "data/scores.csv"
HOLD = 30
TOPN_POOL = 20      # 점수 상위 풀
PICK = 5            # 매수 종목수
LAMBDA = 0.25       # BuyFit 타이밍 비중 (워크포워드 검증값)


def load():
    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    s["날짜"] = pd.to_datetime(s["날짜"])
    s = s.sort_values(["종목코드", "날짜"])
    s["fwd"] = s.groupby("종목코드")["종가"].shift(-HOLD) / s["종가"] - 1
    s["ret20"] = s.groupby("종목코드")["종가"].pct_change(20)
    s["dd"] = s["종가"] / s.groupby("종목코드")["종가"].transform(
        lambda x: x.rolling(60, min_periods=20).max()) - 1
    return s


def zx(df, col):
    return df.groupby("날짜")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))


def run_bt(s, cost_roundtrip, pit=True, use_buyfit=True, born2018=None):
    """
    cost_roundtrip: 왕복 비용 비율 (예 0.0035)
    pit: point-in-time True면 그날 데이터 있는 종목만(생존편향 완화)
    born2018: set of codes existing at 2018-01-02 (보수모드용); None이면 미적용
    """
    s = s.copy()
    s["timing"] = LAMBDA * (0.6 * (-zx(s, "dd")) + 0.4 * (-zx(s, "ret20")))
    s["zs"] = zx(s, "종합점수")
    rows = []
    for d, gd in s[s["fwd"].notna()].groupby("날짜"):
        if born2018 is not None:
            gd = gd[gd["종목코드"].isin(born2018)]
        # pit: groupby 날짜 자체가 그날 존재 종목만 포함하므로 자동 충족
        gd = gd.sort_values("종합점수", ascending=False).head(TOPN_POOL)
        if len(gd) < PICK:
            continue
        if use_buyfit:
            gd = gd.copy()
            gd["bf"] = gd["zs"] + gd["timing"]
            picks = gd.sort_values("bf", ascending=False).head(PICK)
        else:
            picks = gd.head(PICK)
        # 30일 보유 수익 - 왕복비용
        net = picks["fwd"].mean() - cost_roundtrip
        rows.append((d, net))
    r = pd.DataFrame(rows, columns=["날짜", "ret"]).set_index("날짜")
    if len(r) == 0:
        return None
    # 30일 비중첩 가정의 단순 평균 + 연율화 근사 (월 1회 회전 ≈ 연 12회)
    mean30 = r["ret"].mean()
    win = (r["ret"] > 0).mean()
    std30 = r["ret"].std()
    sharpe = mean30 / (std30 + 1e-9) * np.sqrt(12)  # 월간 → 연율 근사
    ann = (1 + mean30) ** 12 - 1                      # 복리 연율 근사
    return dict(n=int(len(r)), mean30=round(mean30 * 100, 2), win=round(win * 100, 1),
                ann=round(ann * 100, 1), sharpe=round(sharpe, 2))


def load_pit_universe():
    """universe.csv가 있으면 {스냅샷날짜: set(종목코드)} 반환, 없으면 None"""
    if not os.path.exists("data/universe.csv"):
        return None
    u = pd.read_csv("data/universe.csv", dtype={"종목코드": str})
    u["날짜"] = pd.to_datetime(u["날짜"])
    snaps = {}
    for d, g in u.groupby("날짜"):
        snaps[d] = set(g["종목코드"].str.zfill(6))
    return dict(sorted(snaps.items()))


def pit_universe_for(snaps, date):
    """해당 거래일에 유효한 가장 최근 스냅샷의 종목집합"""
    valid = [s for s in snaps if s <= date]
    return snaps[max(valid)] if valid else None


def main():
    s = load()
    born2018 = set(s[s["날짜"] <= pd.Timestamp("2018-01-02")]["종목코드"].unique())
    pit = load_pit_universe()
    # 진짜 point-in-time 보정은 '백테스트 구간을 실제로 덮는' 스냅샷이 여럿 있어야 성립.
    # 스냅샷이 1개(또는 최근 몇 개)뿐이면 과거 날짜엔 유효 유니버스가 없어 사실상 무효.
    pit_ok = bool(pit and len(pit) >= 8 and min(pit) <= pd.Timestamp("2020-01-01"))
    delisted_n = 0
    if os.path.exists("data/delisted.csv"):
        try:
            delisted_n = len(pd.read_csv("data/delisted.csv"))
        except Exception:
            pass
    if pit_ok:
        print(f"[point-in-time 유니버스 로드: {len(pit)}개 스냅샷] — 진짜 생존편향 보정 가능\n")
    elif pit:
        print(f"[⚠️ universe.csv 스냅샷 {len(pit)}개뿐 · 최초 {min(pit).date()}] "
              f"— 과거 구간을 못 덮어 point-in-time 보정은 건너뜀 (아래 수치는 생존자만 대상)\n")

    print("=" * 64)
    print("  보완 백테스트: 생존편향 + 비용 현실화")
    print("=" * 64)

    scenarios = []
    grid = [
        ("기존가정 (전종목·비용0.15%)", 0.0015, None, True),
        ("비용 현실화 0.35%",          0.0035, None, True),
        ("비용 현실화 0.50%",          0.0050, None, True),
        ("생존편향완화(2018존재 87종목)·0.35%", 0.0035, born2018, True),
        ("생존편향완화 87종목·비용0.50%",        0.0050, born2018, True),
        ("점수만(타이밍OFF)·87종목·0.35%",       0.0035, born2018, False),
    ]
    for name, cost, b18, bf in grid:
        r = run_bt(s, cost, born2018=b18, use_buyfit=bf)
        if r:
            scenarios.append(dict(name=name, **r))
            print(f"\n■ {name}")
            print(f"   30일평균 {r['mean30']:+.2f}% · 승률 {r['win']}% · "
                  f"연율근사 {r['ann']:+.1f}% · Sharpe(연) {r['sharpe']}  (N={r['n']})")

    # point-in-time 유니버스가 '실제로 과거를 덮을' 때만 진짜 보정 시나리오 추가
    if pit_ok:
        s["_pit_ok"] = False
        mask = []
        for d, g in s.groupby("날짜"):
            u = pit_universe_for(pit, d)
            mask.append(g["종목코드"].isin(u) if u else pd.Series(False, index=g.index))
        s["_pit_ok"] = pd.concat(mask).reindex(s.index).fillna(False)
        s_pit = s[s["_pit_ok"]].copy()
        r = run_bt(s_pit, 0.0050, use_buyfit=True)
        if r:
            scenarios.append(dict(name="★진짜 PIT유니버스·비용0.50%", **r))
            print(f"\n■ ★진짜 point-in-time 유니버스·0.50%")
            print(f"   30일평균 {r['mean30']:+.2f}% · 승률 {r['win']}% · "
                  f"연율근사 {r['ann']:+.1f}% · Sharpe {r['sharpe']}  (N={r['n']})")

    base = scenarios[0]
    concl = min(scenarios, key=lambda x: x["ann"])  # 진짜 최저(가장 보수적) 연율
    erosion = base["ann"] - concl["ann"]
    print("\n" + "=" * 64)
    print(f"  기존 연율 {base['ann']:+.1f}%  →  보수가정 연율 {concl['ann']:+.1f}%")
    print(f"  편향+비용으로 사라지는 수익: 약 {erosion:.1f}%p")
    print(f"  현실적 기대 연수익 (보수): {concl['ann']:+.1f}%  승률 {concl['win']}%")
    print("=" * 64)

    if delisted_n:
        print(f"\n[생존편향 참고] 상장폐지 종목 {delisted_n}건이 데이터에서 누락됨.")
        print(f"  → 이들은 백테스트에 처음부터 없으므로, 실제 수익은 보정치보다 더 낮을 수 있음.")

    # 생존편향 정량화: 백테스트 구간의 KOSPI 상장폐지 수/연율 + 가정 기반 haircut
    surv = None
    try:
        if os.path.exists("data/delisted.csv"):
            dl = pd.read_csv("data/delisted.csv")
            if "DelistingDate" in dl.columns:
                dl["DelistingDate"] = pd.to_datetime(dl["DelistingDate"], errors="coerce")
                start, end = s["날짜"].min(), s["날짜"].max()
                yrs = max(1.0, (end - start).days / 365.25)
                win = dl[(dl["DelistingDate"] >= start) & (dl["DelistingDate"] <= end)]
                n_kospi = int(win["Market"].astype(str).str.contains("KOSPI", case=False, na=False).sum()) \
                    if "Market" in win.columns else 0
                rate = n_kospi / yrs
                # 가정(명시): 상폐 종목 평균 -60% 손실, KOSPI 상장수 ~800 → 연 drag ≈ (rate/800)*60%p, 상한 4%p
                haircut = round(min(4.0, (rate / 800.0) * 60.0), 1)
                surv = dict(n_kospi=n_kospi, rate=round(rate, 1), years=round(yrs, 1),
                            haircut=haircut, ann_after=round(concl["ann"] - haircut, 1))
                print(f"\n[생존편향 정량화] 구간({surv['years']}년) KOSPI 상폐 {n_kospi}건 (연 {rate:.1f}건).")
                print(f"  가정: 상폐 평균 -60% → 추가 haircut ≈ {haircut:.1f}%p/yr → 초보수 기대 {surv['ann_after']:+.1f}%")
    except Exception as e:
        print(f"생존편향 정량화 스킵: {e}")

    os.makedirs("docs", exist_ok=True)
    json.dump({"scenarios": scenarios,
               "base_ann": base["ann"], "concl_ann": concl["ann"],
               "concl_name": concl["name"],
               "erosion": round(erosion, 1),
               "pit": pit_ok,                       # 진짜 PIT 보정이 실행됐을 때만 True
               "pit_snapshots": len(pit) if pit else 0,
               "delisted_n": delisted_n,
               "surv": surv},
              open("docs/robust.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()

