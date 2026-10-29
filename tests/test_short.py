"""
단기매매 회귀 테스트 — 분할 보정 · 미래 정보 차단 · 연구=모의투자 일치 · 체결(정수 주식·비용) · 토스 수집 넘김.
사용: python tests/test_short.py   (pytest 로도 돈다)
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "deploy/oracle")
import short_panel as SP   # noqa: E402
import short_lab as L      # noqa: E402
import short_trader as ST  # noqa: E402

_D = None


def data():
    global _D
    if _D is None:
        _D = L.Data()
    return _D


def test_split_adjustment():
    """50:1 분할 → 과거 가격이 1/50 로 보정된다. 근거가 없으면 '끊김'으로 표시."""
    close = np.array([[100.0, 10.0], [102.0, 10.5], [2.04, 2.0], [2.1, 2.1]])
    adj = np.array([[2.0, np.nan], [2.04, np.nan], [2.04, np.nan], [2.1, np.nan]])
    mc = np.full_like(close, np.nan)
    fac, br = SP.adjust_factors(close, adj, mc)
    a = close * fac
    assert abs(a[1, 0] - 2.04) < 1e-9 and abs(a[2, 0] / a[1, 0] - 1) < 1e-9
    assert br[2, 1] and not br[2, 0]


def test_panel_has_no_impossible_moves():
    D = data()
    r = D.P["C"][1:] / D.P["C"][:-1] - 1
    r[D.P["breaks"][1:]] = np.nan
    assert np.nanmax(np.abs(r)) <= SP.LIMIT + 1e-9


def test_choose_uses_only_past():
    """t일 선택은 t일까지의 성적만 본다: t+1일 이후 수익을 바꿔도 t일 선택은 같다."""
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (400, 5))
    R[:, 2] += 0.002
    a = L.choose(R, 300, every=1, L=250, how="sharpe", k=1)
    R2 = R.copy()
    R2[351:, 4] += 0.05
    b = L.choose(R2, 300, every=1, L=250, how="sharpe", k=1)
    assert a[:351] == b[:351] and a[300] == 2


def test_follow_reproduces_model():
    """'모델 하나를 계속 따라가기'는 그 모델 단독 시뮬레이션과 같은 결과 (연구와 모의투자가 같은 코드).
    현금이 모자랄 때 나누는 방식만 달라 8년 누적 0.1% 안쪽 차이는 허용."""
    D = data()
    ms = [{"id": "S:MOM:10:H5:crash", "kind": "stock", "sel": "MOM", "K": 10, "hold": "H5", "gate": "crash"},
          {"id": "E:overnight_trend", "kind": "etf", "rule": "overnight_trend"}]
    recs = [L.run(D, m, record=True) for m in ms]
    for i in range(2):
        E, _ = L.follow(D, ms, [i] * D.T, recs)
        assert abs(E[-1] / recs[i]["E"][-1] - 1) < 1e-3


def test_daily_roundtrip_costs_more_than_reversal_edge():
    """검증의 핵심 사실: 1일 반전 상위 10%의 하루 초과수익 < 주식 왕복 비용."""
    ic = {r["signal"]: r for r in L.ic_table(data(), horizons=(1,))}
    assert ic["rev1"]["h1"]["ic"] > 0.03
    assert ic["rev1"]["h1"]["top10_bp"] < sum(L.costs("stock")) * 1e4


def fake_D():
    dates = pd.DatetimeIndex(["2026-10-23", "2026-10-26"])
    O = np.array([[100.0, 1_000_000.0], [101.0, 1_010_000.0]])
    return SimpleNamespace(dates=dates, P={"codes": ["000001", "000002"]}, O=O, Cf=O.copy(),
                           Op=np.array([[50.0, 5.0], [51.0, 4.9]]), Cp=np.array([[50.0, 5.0], [52.0, 4.8]]),
                           legs={"source": "etf"}, T=2, N=2)


def test_settle_integer_shares_costs_and_cash():
    D = fake_D()
    acc = {"cash": 1_000_000.0, "initial_cash": 1_000_000.0, "positions": {}, "etf": {}}
    plan = {"made": "2026-10-23", "model": "S:test", "orders": [
        {"session": "open", "kind": "stock", "code": "000001", "name": "A", "side": "buy", "qty": 5000, "est_price": 100},
        {"session": "open", "kind": "stock", "code": "000002", "name": "B", "side": "buy", "qty": 1, "est_price": 1_000_000},
        {"session": "close", "kind": "etf", "code": "069500", "name": "KODEX 200", "side": "buy", "qty": 10, "est_price": 50}]}
    fills, t = ST.settle(D, acc, plan, {})
    assert t == 1
    cb = L.costs("stock")[0]
    a = next(f for f in fills if f["종목명"] == "A")
    assert a["수량"] == 5000 and a["가격"] == 101.0                # 다음 날 시가
    assert all(f["종목명"] != "B" for f in fills)                  # 현금이 모자라면 정수로 줄이고, 0주면 체결 없음
    assert acc["cash"] >= 0 and abs(acc["positions"]["000001"]["avg"] - 101 * (1 + cb)) < 1e-9
    e = next(f for f in fills if f["종목명"] == "KODEX 200")
    assert e["가격"] == 52.0                                        # 종가 주문은 그날 종가
    # 다시 체결해도 같은 계획을 두 번 체결하지 않도록: 계획의 다음 거래일이 없으면 아무것도 안 함
    acc2 = json.loads(json.dumps(acc))
    f2, t2 = ST.settle(D, acc2, {"made": "2026-10-26", "orders": plan["orders"]}, {})
    assert f2 == [] and t2 is None


def test_sell_pnl_includes_tax():
    D = fake_D()
    acc = {"cash": 0.0, "positions": {"000001": {"qty": 10, "avg": 100.0, "since": "2026-10-20"}}, "etf": {}}
    plan = {"made": "2026-10-23", "orders": [{"session": "open", "kind": "stock", "code": "000001", "side": "sell", "qty": 10}]}
    fills, _ = ST.settle(D, acc, plan, {})
    cs = L.costs("stock")[1]
    assert abs(acc["cash"] - 1010 * (1 - cs)) < 1e-6 and "000001" not in acc["positions"]
    assert fills[0]["손익"] == round(1010 * (1 - cs) - 1000)


def test_short_json_has_no_nan():
    assert ST.clean({"a": float("nan"), "b": [1.0, float("inf")], "c": np.float64(2.0)}) == {"a": None, "b": [1.0, None], "c": 2.0}


def test_toss_collect_pagination():
    """최신→과거로 넘기며 모으고, 매일 모드에서는 이미 가진 날짜를 만나면 멈춘다."""
    import toss_collect as TC
    days = pd.bdate_range("2025-01-01", periods=250)[::-1].strftime("%Y-%m-%d").tolist()

    class Api:
        n = 0

        def get(self, path, params):
            self.n += 1
            start = days.index(params["until"]) if params.get("until") else 0
            page = days[start:start + params["count"]]
            nxt = days[start + params["count"]] if start + params["count"] < len(days) else None
            return {"result": {"records": [{"baseDate": d, "foreign": {"netBuyAmount": str(i)}} for i, d in enumerate(page)],
                               "nextUntil": nxt}}

    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "x.jsonl")
        api = Api()
        r = TC.collect_records(api, "/p", {"count": 100}, f, backfill=True)
        assert r["rows"] == 250 and api.n == 3 and "foreign.netBuyAmount" in r["fields"]
        api2 = Api()
        r2 = TC.collect_records(api2, "/p", {"count": 100}, f, backfill=False)
        assert r2["added"] == 0 and api2.n == 1


def test_trailing_score_ignores_unborn_days():
    """진화 후보는 등록 전 날들이 NaN — 유효한 날이 L/2 보다 적으면 점수가 없다(선택 불가), 차면 생긴다."""
    rng = np.random.default_rng(1)
    R = rng.normal(0.001, 0.01, (600, 2))
    R[:500, 1] = np.nan                          # 100일 전에 '태어난' 후보
    sc = L.trailing_score(R, 599, 250, "sharpe")
    assert np.isfinite(sc[0]) and np.isnan(sc[1])
    R2 = R.copy()
    R2[470:500, 1] = 0.001                        # 130일째 → 선택 가능
    assert np.isfinite(L.trailing_score(R2, 599, 250, "sharpe")[1])
    ch = L.choose(R, 560, every=1, L=250, how="sharpe", k=1)
    assert all(c == 0 for c in ch[560:])


def test_short_evolve_genomes_valid():
    import short_evolve as SE
    rng = np.random.default_rng(3)
    g = SE.random_genome(rng)
    for _ in range(200):
        g = SE.mutate(g, rng) if rng.random() < 0.7 else SE.crossover(g, SE.random_genome(rng), rng)
        assert 1 <= len(g["w"]) <= 4 and all(k in SP.SIGNAL_INFO for k in g["w"])
        assert g["K"] in (5, 10) and g["hold"] in ("H5", "H10") and g["gate"] in L.GATES
    m = SE.to_model(g, "X1", "2026-10-23")
    assert m["combo"] and m["born"] == "2026-10-23" and L.describe(m).startswith("진화 X1")


def test_evolved_model_runs_like_fixed():
    """진화 모델 경로(조합 순위)는 같은 조합의 고정 모델과 똑같이 움직인다."""
    D = data()
    fixed = {"id": "S:REVC:5:H5:crash", "kind": "stock", "sel": "REVC", "K": 5, "hold": "H5", "gate": "crash"}
    evo = {**fixed, "id": "X", "sel": "EVO", "combo": [(k, float(w)) for k, w in L.SELECTORS["REVC"]]}
    assert abs(L.run_stock(D, fixed)["E"][-1] - L.run_stock(D, evo)["E"][-1]) < 1e-12


def test_explain_has_plain_answers():
    """해설: 오늘 판단마다 사람이 읽는 한 줄 · 조건 점검 문장 · 규칙집 · 참고 정보 비중이 있다."""
    import explain as X
    cfg = X.settings.load()
    A = {"asof": "2026-10-23", "gate": {"open": False, "regime": "급락장", "rebound": 0.01, "reason": "급락장 · 바닥 미확인", "need": "반등"},
         "decisions": [{"code": "000001", "name": "가", "rank": 3, "state": "관망", "price": 1000.0, "checks": [
             {"label": "종합순위 10위 이내", "ok": True, "value": "3위"}, {"label": "시장 게이트 (급락장 바닥확인)", "ok": False, "value": "급락장"}],
             "need": ["지수 반등"]},
             {"code": "000002", "name": "나", "rank": 15, "state": "관망", "price": 1000.0, "checks": [
             {"label": "종합순위 10위 이내", "ok": False, "value": "15위"}]}]}
    e = X.engine_section(cfg, A, {"items": [], "market": {"weights": {"s_value": 0.5, "s_flow": 0.5}}}, None)
    assert "급락장" in e["today"][0]["line"] and "후보" in e["today"][1]["line"]
    assert e["funnel"][2]["n"] == 1 and e["funnel"][3]["n"] == 0
    assert e["rules"]["buy"] and e["rules"]["sell"] and abs(sum(w["w"] for w in e["weights"]) - 1) < 1e-9
    info = X.info_table(cfg, e, None)
    assert any(r["weight"] == "0%" for r in info["ref"]) and info["direct"][0]["weight"] == "50%"


def test_aw_tax_and_costs():
    """기타 ETF(달러) 이익 매도 → 이익의 15.4% 세금, 국내 주식형은 세금 없음. 비용은 호가 반 칸."""
    import allweather as AW
    T = 3
    Op = np.full((T, AW.A), np.nan)
    Cp = np.full((T, AW.A), np.nan)
    j, k = AW.IX["USD"], AW.IX["IDX"]
    for col in (j, k):
        Op[:, col] = [10000.0, 10000.0, 12000.0]
        Cp[:, col] = [10000.0, 10000.0, 12000.0]
    for col, taxed in ((j, True), (k, False)):
        W = np.full((T, AW.A), np.nan)
        W[1] = 0.0
        W[1, col] = 1.0
        W[2] = 0.0
        res = AW.simulate(Op, Cp, W, band=0.0)
        cb, cs = AW.costs(np.array([10000.0]))[0][0], AW.costs(np.array([12000.0]))[1][0]
        units = 1 / (1 + cb) / 10000
        gain = units * 12000 * (1 - cs) - 1.0
        want = 1.0 + gain - (0.154 * gain if taxed else 0)
        assert abs(res["E"][-1] - want) < 1e-9, (col, res["E"][-1], want)


def test_aw_targets_use_only_past():
    """t행 목표 비중은 t-1일까지의 가격으로만: t일 이후 가격을 바꿔도 t행까지 같다."""
    import allweather as AW
    dates, Op, Cp = AW.load_prices()
    F = AW.features(Cp, dates)
    t0 = len(dates) - 300
    Cp2 = Cp.copy()
    Cp2[t0:] *= 0.5
    F2 = AW.features(Cp2, dates)
    for rule in ("T:ls", "T:def", "DM:1", "RP", "VT", "CR:lev", "BR:usd"):
        a, b = AW.targets(F, Cp, rule), AW.targets(F2, Cp2, rule)
        assert np.array_equal(np.nan_to_num(a[:t0 + 1], nan=-1), np.nan_to_num(b[:t0 + 1], nan=-1)), rule


def test_aw_live_mix_target_is_full_weight():
    import allweather as AW
    m = AW.live_mix()
    assert abs(m["target"].sum() - 1) < 0.02 and (m["target"] >= -1e-12).all()
    assert m["regime"]["name"] in AW.REGIMES and len(m["mix"]) >= 10


def test_regime_labels():
    import allweather as AW
    c = np.r_[np.linspace(100, 200, 300), np.linspace(200, 150, 40)]
    Cp = np.full((len(c), AW.A), np.nan)
    Cp[:, AW.IX["IDX"]] = c
    reg = AW.features(Cp)["reg"]
    assert reg[299] == 0 and reg[-1] == 3                    # 꾸준히 오르면 강세, 고점 대비 -25% 면 급락


def test_quarter_turn_and_aw_sleeve_settle():
    import allweather as AW
    assert ST.quarter_turn(pd.Timestamp("2026-12-31")) and ST.quarter_turn(pd.Timestamp("2026-09-30"))
    assert not ST.quarter_turn(pd.Timestamp("2026-10-28"))
    D = fake_D()
    Op = np.full((2, AW.A), np.nan)
    Op[:, :2] = D.Op
    Op[:, AW.IX["GOLD"]] = [20000.0, 21000.0]
    D.Op, D.Cp = Op, Op.copy()
    acc = {"cash": 100.0, "positions": {}, "etf": {}, "aw": {"cash": 0.0, "etf": {"132030": {"qty": 10, "avg": 20000.0, "since": "x"}}}}
    plan = {"made": "2026-10-23", "orders": [{"session": "open", "kind": "etf", "sleeve": "aw", "code": "132030", "side": "sell", "qty": 10}],
            "transfer": {"amount": 50000}}
    fills, _ = ST.settle(D, acc, plan, {})
    cs = AW.costs(np.array([21000.0]))[1][0]
    proceeds = 210000 * (1 - cs)
    tax = 0.154 * (proceeds - 200000)
    assert abs(acc["aw"]["cash"] + acc["cash"] - 100 - (proceeds - tax)) < 1e-6
    assert abs(acc["cash"] - 100 - 50000) < 1e-6 and "132030" not in acc["aw"]["etf"] and acc["positions"] == {}
    assert fills[0]["손익"] == round(proceeds - 200000 - tax)


def test_evolve_fitness_penalizes_bad_regime():
    import short_evolve as SE
    D = data()
    g = {"w": {k: float(w) for k, w in L.SELECTORS["REVC"]}, "K": 5, "hold": "H10", "gate": "none"}
    f = SE.fitness(D, g)
    assert "worst_regime" in f and (f["worst_regime"] >= 0 or f["fit"] < f["sharpe"])


def test_us_settle_fractional_and_costs():
    """미국 모의: 다음 미국 장 시가에 소수점 4자리로 사고, 매도 땐 수수료·SEC 를 떼고 그해 실현손익에 쌓는다."""
    import us_trader as UT
    D = SimpleNamespace(dates=pd.DatetimeIndex(["2026-10-28", "2026-10-29"]), P={"codes": ["AAA", "BBB"]},
                        O=np.array([[100.0, 50.0], [101.0, 55.0]]))
    acc = {"cash": 1000.0, "positions": {"BBB": {"qty": 2.0, "avg": 50.0}}}
    plan = {"made": "2026-10-28", "orders": [{"side": "buy", "code": "AAA", "usd": 300.0}, {"side": "sell", "code": "BBB", "qty": 2.0}]}
    fills, t = UT.settle(D, acc, plan)
    cb, cs = L.costs("us")
    assert t == 1 and "BBB" not in acc["positions"]
    q = acc["positions"]["AAA"]["qty"]
    assert q == np.floor(300 / 101 * 10000) / 10000
    assert abs(acc["cash"] - (1000 + 110 * (1 - cs) - q * 101 * (1 + cb))) < 1e-9
    assert abs(acc["realized"]["2026"] - (110 * (1 - cs) - 100)) < 1e-9


def test_us_after_tax_only_taxes_gains():
    import us_lab as UL
    dates = pd.DatetimeIndex(["2024-01-02", "2024-12-31", "2025-06-30", "2025-12-31"])
    E = np.array([1.0, 1.2, 1.1, 1.0])                       # 2024 +20%, 2025 -16.7%
    A = UL.after_tax(E, dates)
    assert abs(A[1] - 1.2) < 1e-12 and abs(A[-1] - 1.0 * (1 + 0.2 * 0.78) / 1.2) < 1e-12


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
