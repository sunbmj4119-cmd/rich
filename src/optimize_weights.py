"""
가중치 전수 탐색 최적화기 v2 — 순수 numpy 고속판

모든 팩터 가중치 조합을 빠짐없이 탐색해, 과최적화가 아닌
(In-Sample·Out-of-Sample 둘 다에서 통하는) 가장 강건한 가중치를 찾는다.

평가: 일별 횡단면 IC의 t값. IS(2018~2022)/OOS(2023~) 분리.
강건성 = min(IS_t, OOS_t).

사용:
  python src/optimize_weights.py                  # step 0.1
  python src/optimize_weights.py --step 0.05      # 더 정밀(느림)
  python src/optimize_weights.py --max-factors 5
"""
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

SCORES = "data/scores.csv"
OHLCV = "data/ohlcv.csv"
FACTORS = ["s_value", "s_flow", "s_profit", "s_grow", "s_mom", "s_stab", "s_vtrend"]
FAC_NAME = {"s_value": "가치", "s_flow": "외국인", "s_profit": "수익성",
            "s_grow": "성장", "s_mom": "모멘텀", "s_stab": "안정성", "s_vtrend": "거래대금"}
KEYMAP = {"s_value": "value", "s_flow": "flow", "s_profit": "profit",
          "s_grow": "growth", "s_mom": "momentum", "s_stab": "stability",
          "s_vtrend": "vtrend"}
FWD = 20
SPLIT = "2023-01-01"


def load():
    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    o = pd.read_csv(OHLCV, dtype={"종목코드": str})
    df = s.merge(o[["날짜", "종목코드", "종가"]].rename(columns={"종가": "_p"}),
                 on=["날짜", "종목코드"], how="left")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df = df.sort_values(["종목코드", "날짜"])
    df["fwd"] = df.groupby("종목코드")["종가"].shift(-FWD) / df["종가"] - 1
    df = df.dropna(subset=["fwd"] + FACTORS)
    df["fwd_rank"] = df.groupby("날짜")["fwd"].rank()
    return df


class ICEngine:
    """순수 numpy 일별 횡단면 IC/t 계산기."""
    def __init__(self, df):
        df = df.sort_values("날짜").reset_index(drop=True)
        codes, grp = np.unique(df["날짜"].values, return_inverse=True)
        F = df[FACTORS].values
        fr = df["fwd_rank"].values
        order = np.argsort(grp, kind="stable")
        gs = grp[order]
        self.F = F[order]
        self.fr = fr[order]
        bounds = np.searchsorted(gs, np.arange(len(codes) + 1))
        self.slices = [(bounds[i], bounds[i + 1]) for i in range(len(codes))
                       if bounds[i + 1] - bounds[i] >= 10]

    def ic_t(self, w):
        sc = self.F @ w
        ics = []
        for a, b in self.slices:
            xr = sc[a:b].argsort().argsort().astype(float)
            ic = np.corrcoef(xr, self.fr[a:b])[0, 1]
            if not np.isnan(ic):
                ics.append(ic)
        if len(ics) < 30:
            return 0.0, 0.0
        ics = np.array(ics)
        sd = ics.std()
        return ics.mean(), (ics.mean() / sd * np.sqrt(len(ics)) if sd > 0 else 0.0)


def gen_grid(n, step, max_factors):
    levels = int(round(1.0 / step))
    out = []
    def rec(rem, slots, cur):
        if slots == 1:
            cur.append(rem)
            if sum(1 for v in cur if v > 0) <= max_factors:
                out.append(np.array(cur) * step)
            cur.pop()
            return
        for v in range(rem + 1):
            cur.append(v)
            rec(rem - v, slots - 1, cur)
            cur.pop()
    rec(levels, n, [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max-factors", type=int, default=4)
    ap.add_argument("--mode", choices=["is_oos", "walkforward"], default="walkforward",
                    help="is_oos: 2분할 / walkforward: 다구간 강건성(권장)")
    args = ap.parse_args()

    df = load()
    print(f"\ub370\uc774\ud130: {df['\ub0a0\uc9dc'].min().date()} ~ {df['\ub0a0\uc9dc'].max().date()}, "
          f"{df['\uc885\ubaa9\ucf54\ub4dc'].nunique()}\uc885\ubaa9, {len(df):,}\ud589")

    if args.mode == "walkforward":
        # 2년 단위 구간으로 쪼개 모든 구간에서의 t값을 본다 → 진짜 강건성
        periods = [(2018, 2019), (2020, 2021), (2022, 2023), (2024, 2026)]
        engines = []
        for y0, y1 in periods:
            m = (df["\ub0a0\uc9dc"].dt.year >= y0) & (df["\ub0a0\uc9dc"].dt.year <= y1)
            engines.append(ICEngine(df[m]))
        print(f"Walk-forward \uad6c\uac04: {periods}")
        print("\uac15\uac74\uc131 = \ubaa8\ub4e0 \uad6c\uac04 t\uac12\uc758 \ucd5c\uc18c\uac12 (\ud55c \uad6c\uac04\ub3c4 \uc548 \ub9dd\uac00\uc838\uc57c \ucc44\ud0dd)\n")

        grid = gen_grid(len(FACTORS), args.step, args.max_factors)
        print(f"\ud0d0\uc0c9 \uc870\ud569: {len(grid):,}\uac1c\n")
        res = []
        for i, w in enumerate(grid):
            ts = [e.ic_t(w)[1] for e in engines]
            res.append((min(ts), ts, w))
            if (i + 1) % 2000 == 0:
                print(f"  \uc9c4\ud589 {i+1}/{len(grid)}...")
        res.sort(key=lambda r: -r[0])

        print(f"\n{'='*82}")
        print(f"\uac15\uac74 \uc0c1\uc704 {args.top}\uac1c (\ubaa8\ub4e0 \uad6c\uac04 t\uac12 \ucd5c\uc18c\uac12 \uc21c)")
        print(f"{'='*82}")
        hdr = "\uad6c\uac04t\ucd5c\uc18c " + " ".join(f"{y0}-{y1}" for y0, y1 in periods)
        print(f"{'#':>2} {hdr}  \uac00\uc911\uce58")
        for rank, (mn, ts, w) in enumerate(res[:args.top], 1):
            ws = ", ".join(f"{FAC_NAME[FACTORS[j]]}{w[j]:.2f}"
                           for j in range(len(FACTORS)) if w[j] > 0)
            tstr = " ".join(f"{t:6.1f}" for t in ts)
            print(f"{rank:>2} {mn:>6.1f}  {tstr}  {ws}")

        best = res[0][2]
        print(f"\n{'='*82}")
        print("\ucd94\ucc9c weights.yaml (\ubaa8\ub4e0 \uad6c\uac04 \uac15\uac74 1\uc704)")
        print(f"{'='*82}")
        print("logic:")
        for j, f in enumerate(FACTORS):
            print(f"  {KEYMAP[f]:10s}: {best[j]:.2f}")
        print(f"\n\uad6c\uac04\ubcc4 t: {[round(t,1) for t in res[0][1]]}")
        print("\u2192 \ubaa8\ub4e0 \uad6c\uac04\uc774 \uc591\uc218(\uac00\ub2a5\ud558\uba74 2\u2191)\uc774\uba74 \uc2dc\uae30 \ubb34\uad00\ud558\uac8c \uc791\ub3d9\ud558\ub294 \uc9c4\uc9dc \uac15\uac74 \uc804\ub7b5.")
        print("\ud55c \uad6c\uac04\ub9cc \ub192\uace0 \ub098\uba38\uc9c0 \ub0ae\uc73c\uba74 \uadf8 \uc2dc\uae30 \uacfc\uc801\ud569 \uc758\uc2ec.")
        return

    # is_oos 모드 (기존)
    is_eng = ICEngine(df[df["\ub0a0\uc9dc"] < SPLIT])
    oos_eng = ICEngine(df[df["\ub0a0\uc9dc"] >= SPLIT])
    print("IS: ~2022 | OOS: 2023~")
    grid = gen_grid(len(FACTORS), args.step, args.max_factors)
    print(f"\ud0d0\uc0c9 \uc870\ud569: {len(grid):,}\uac1c\n")
    res = []
    for w in grid:
        _, ist = is_eng.ic_t(w)
        if ist < 1.0:
            continue
        _, oost = oos_eng.ic_t(w)
        res.append((min(ist, oost), ist, oost, w))
    res.sort(key=lambda r: -r[0])
    print(f"{'#':>2} {'\uac15\uac74t':>6} {'IS_t':>6} {'OOS_t':>6}  \uac00\uc911\uce58")
    for rank, (rob, ist, oost, w) in enumerate(res[:args.top], 1):
        ws = ", ".join(f"{FAC_NAME[FACTORS[j]]}{w[j]:.2f}"
                       for j in range(len(FACTORS)) if w[j] > 0)
        print(f"{rank:>2} {rob:>6.1f} {ist:>6.1f} {oost:>6.1f}  {ws}")


if __name__ == "__main__":
    main()

