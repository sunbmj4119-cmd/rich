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


def test_overfit_tools():
    """통계 도구: 무작위 전략들 중 1등의 PBO 는 약 0.5, 진짜 우위가 있는 하나가 끼면 낮다. Holm 은 단조·1 이하. 순차검정은 차이 0 에서 거의 안 터진다."""
    import science_lab as SL
    import evolve as EV
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (1200, 40))
    p_noise = SL.pbo(R, S=10)["pbo"]
    R2 = R.copy()
    R2[:, 0] += 0.003
    p_edge = SL.pbo(R2, S=10)["pbo"]
    assert 0.3 < p_noise < 0.7 and p_edge < 0.1
    d, _ = SL.dsr(R2[:, 0], [np.mean(R2[:, i]) / np.std(R2[:, i]) for i in range(40)])
    assert d > 0.9
    adj = SL.holm([0.01, 0.04, None, 0.03])
    assert adj[2] is None and adj[0] == 0.03 and all(a <= 1 for a in adj if a is not None)
    assert abs(SL._nppf(0.975) - 1.959964) < 1e-4 and abs(SL._ncdf(1.959964) - 0.975) < 1e-4
    hits = sum(1 for k in range(100) if (lambda r: r and r["lambda"] >= 20)(EV.msprt(rng.normal(0, 0.006, 300))))
    assert hits <= 5


def test_new_research_signals_dont_grow_fixed_pool():
    """연구에서 넣은 재료(fip·for5 신호, hivol 필터)는 진화 전용 — 고정 후보 330개는 그대로 (고르는 데서 생기는 과최적화를 키우지 않게)."""
    assert len(L.stock_models()) == 330
    D = data()
    assert "fip" in D.S and "for5" in D.S and "hivol" in D.gate and "hivol" in L.GATES
    import short_evolve as SE
    assert "hivol" in SE.CH["gate"] and "fip" in SE.SIGS


def test_long_gate_removes_old_failures_from_pool():
    """긴 역사 관문: 옛 시대에 손해였던 모델만 후보에서 빠지고, 다 빠지면 관문 없이."""
    ms = L.stock_models() + L.etf_models()
    full = L.pool_index(ms, "SLOW", set())
    drop = {"E:oo_rev", ms[full[0]]["id"]}
    kept = L.pool_index(ms, "SLOW", drop)
    assert len(kept) == len(full) - 2 and not drop & {ms[i]["id"] for i in kept}
    assert L.pool_index(ms, "SLOW", {ms[i]["id"] for i in full}) == full
    with tempfile.TemporaryDirectory() as d:
        old = L.GATE_FILE
        try:
            L.GATE_FILE = os.path.join(d, "gate.json")
            assert L.long_gate() == set()                      # 파일 없으면 빈 관문
            json.dump({"fail": ["E:oo_rev"]}, open(L.GATE_FILE, "w"))
            assert L.long_gate() == {"E:oo_rev"}
            assert len(L.pool_index(ms, "SLOW")) == len(full) - 1
        finally:
            L.GATE_FILE = old


def test_history_alignment_and_rolling():
    """미국 장·환율은 한국 날짜 '앞선 날' 값만 (미래 정보 없음) · 시작일별 분포 · 순위 상관."""
    import history_lab as HL
    kr = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    us = pd.Series([10.0, 11.0, 12.0, 13.0], index=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]))
    b = HL.before(us, kr)
    assert list(b) == [10.0, 11.0, 12.0]                       # 같은 날짜 미국 종가는 아직 모른다
    r = pd.Series(np.full(600, 0.001))
    ro = HL.rolling_origin(r, (252,))["1y"]
    assert ro["p_loss"] == 0 and abs(ro["median"] - (1.001 ** 252 - 1)) < 1e-4
    assert HL.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and HL.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_risk_manager_rules():
    """리스크 매니저: 정지/매도만/한도 축소/거래량/가격 보호/검사 실패 시 새 매수 보류."""
    import risk as RK
    orders = [{"side": "sell", "code": "A", "qty": 10, "est_price": 100.0, "kind": "stock"},
              {"side": "buy", "code": "B", "qty": 100, "est_price": 100.0, "kind": "stock"},
              {"side": "buy", "code": "153130", "qty": 50, "est_price": 100.0, "kind": "etf", "sleeve": "aw"}]
    ok = {"live": {"action": "ACTIVE", "size": 1.0}, "aw": {"action": "ACTIVE", "size": 1.0}}
    out, log = RK.review(orders, "HALTED", ok, 1e6)
    assert out == [] and len(log) == 3
    out, log = RK.review(orders, "REDUCING", ok, 1e6)
    assert [o["side"] for o in out] == ["sell"]
    out, log = RK.review(orders, "ACTIVE", ok, 20000.0)               # 한 종목 25% = 5000원 → B 50주로 축소, 단기채 ETF 는 예외
    assert next(o for o in out if o["code"] == "B")["qty"] == 50 and next(o for o in out if o["code"] == "153130")["qty"] == 50
    out, _ = RK.review(orders, "ACTIVE", ok, 1e9, adv=lambda c: 1000.0)  # 거래량 1000주의 2.5% = 25주
    assert next(o for o in out if o["code"] == "B")["qty"] == 25
    out, _ = RK.review(orders, "ACTIVE", {"live": {"action": "REDUCING", "size": 0.0}, "aw": ok["aw"]}, 1e9)
    assert {o["code"] for o in out} == {"A", "153130"}
    assert RK.collar_ok(orders[1], 111.0) and not RK.collar_ok(orders[1], 113.0) and RK.collar_ok(orders[0], 500.0)
    p = RK.fail_closed({"orders": list(orders)}, ValueError("x"))
    assert [o["side"] for o in p["orders"]] == ["sell"] and p["risk"]["state"] == "REDUCING"
    with tempfile.TemporaryDirectory() as d:
        old = RK.KILL
        try:
            RK.KILL = os.path.join(d, "KILL")
            open(RK.KILL, "w").close()
            assert RK.global_state("2026-10-30", 1, "2026-10-30")[0] == "HALTED"
        finally:
            RK.KILL = old
    assert RK.global_state("2026-10-20", 1, "2026-10-30")[0] == "REDUCING"        # 자료가 묵음
    assert RK.global_state("2026-10-30", 61, "2026-10-30")[0] == "HALTED"         # 주문 수 비정상


def test_cusum_monitor():
    """기대대로 벌면 경보 없음 · 꾸준히 잃으면 경보 · 문턱은 오경보 7년 기준."""
    import risk as RK
    sr, vol = 0.5, 0.2
    mu = sr / np.sqrt(252) * vol / np.sqrt(252)
    assert RK.cusum(np.full(2000, mu), sr, vol)["alarms"] == 0
    assert RK.cusum(np.full(400, -0.01), sr, vol)["alarms"] >= 1
    m = RK.monitor(np.full(300, -0.003), sr, vol)
    assert m["review"] and m["size"] == 1.0                    # 기본은 경고만 (재현에서 자동 축소가 손해)
    m = RK.monitor(np.full(300, -0.003), sr, vol, decay_action="half")
    assert m["size"] == 0.5
    e = pd.Series([100.0, 110.0, 160.0, 176.0], index=["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"])
    r = RK.flow_adjusted(e, [{"after": "2026-01-02", "amount": 39.0}])   # 1/5 에 39 가 들어옴 → 수익은 (160-39)/110-1
    assert abs(r.loc["2026-01-05"] - (121 / 110 - 1)) < 1e-12 and abs(r.loc["2026-01-02"] - 0.1) < 1e-12


def test_committee_journal_chain():
    """결정 기록은 해시 사슬 — 한 글자라도 고치면 깨진 것으로 나온다."""
    import committee as CMT
    with tempfile.TemporaryDirectory() as d:
        old = (CMT.DIR, CMT.JOURNAL)
        try:
            CMT.DIR, CMT.JOURNAL = d, os.path.join(d, "j.jsonl")
            CMT.append_journal({"date": "2026-10-29", "conviction": 1})
            e2, w2 = CMT.append_journal({"date": "2026-10-30", "conviction": -1})
            _, w3 = CMT.append_journal({"date": "2026-10-30", "conviction": -1})     # 같은 날 같은 내용 = 다시 안 씀
            assert w2 and not w3 and CMT.verify_chain() == {"ok": True, "n": 2}
            txt = open(CMT.JOURNAL).read().replace('"conviction": 1', '"conviction": 2')
            open(CMT.JOURNAL, "w").write(txt)
            assert not CMT.verify_chain()["ok"]
        finally:
            CMT.DIR, CMT.JOURNAL = old


def test_lookahead_check_catches_leaks():
    """미래 정보 검사: 진짜 신호는 통과, 일부러 넣은 미래 정보(전체 평균·내일 수익)는 걸린다."""
    import lookahead_check as LC
    P = SP.load()
    t = len(P["dates"]) // 2
    assert not [r for r in LC.check_panel(P, [t]) if not r["ok"]]
    a = np.array([1.0, 2.0, np.nan])
    assert LC.same(a, a.copy())[0] and not LC.same(a, np.array([1.0, 2.5, np.nan]))[0] and not LC.same(a, np.array([1.0, 2.0, 3.0]))[0]
    C = P["C"]
    full = (C - np.nanmean(C, 0))
    part = (C[:t + 1] - np.nanmean(C[:t + 1], 0))
    assert not LC.same(part, full[:t + 1])[0]


def test_community_weights_engine():
    """남들 전략 시뮬레이터: 한 자산 100% 를 계속 들면 = 그 자산 수익 − 처음 한 번 비용."""
    import community_lab as CL
    d = pd.bdate_range("2020-01-01", periods=60)
    c = pd.DataFrame({"X": np.linspace(100, 130, 60), "Y": np.full(60, 50.0)}, index=d)
    o = c.shift(1).fillna(100.0)
    dec = np.full((60, 2), np.nan)
    dec[0] = [1.0, 0.0]
    r, turn = CL.run_weights(dec, o, c, cost=0.001)
    E = np.prod(1 + r[1:])
    assert abs(E - (130 / 100) * (1 - 0.001)) < 1e-9 and abs(turn - 1.0) < 1e-12


def test_mirror_executor_and_reconcile():
    """거울 계좌: 모의 몫 × 배율을 목표로 차이만 주문(매도 먼저·가격 보호 지정가·늦으면 매수 안 함) · 대사 1주·2% 허용."""
    import short_exec as SX
    acc = {"cash": 100.0, "positions": {"A": {"qty": 10}}, "etf": {}, "aw": {"cash": 50.0, "etf": {"E1": {"qty": 100}, "E2": {"qty": 40}}}}
    plan = {"orders": [{"sleeve": "aw", "side": "buy", "code": "E1", "qty": 20, "session": "open"},
                       {"sleeve": "aw", "side": "sell", "code": "E2", "qty": 40, "session": "open"},
                       {"side": "buy", "code": "B", "qty": 5, "session": "open"}]}
    assert SX.after_plan(acc, plan, ["aw"]) == {"E1": 120}
    tgt = SX.targets(acc, plan, {"scale": 0.5}, ["aw"])
    assert tgt == {"E1": 60}
    orders = SX.diff_orders(tgt, {"E1": {"shares": 50}, "E2": {"shares": 20}}, {"E1": 10000.0}, 0.12)
    assert [(o["side"], o["code"], o["qty"]) for o, _ in orders] == [("sell", "E2", 20), ("buy", "E1", 10)]
    assert orders[1][0]["kind"] == "limit" and orders[1][0]["price"] == 11200
    late = SX.diff_orders(tgt, {"E1": {"shares": 50}}, {"E1": 10000.0}, 0.12, late=True)
    assert late[0][0]["kind"] == "skip"
    ok = SX.compare({"E1": 100}, {"E1": {"shares": 101}})
    bad = SX.compare({"E1": 100, "E2": 10, "E3": 5}, {"E1": {"shares": 90}, "E2": {"shares": 0}, "E3": {"shares": 9}})
    assert ok["state"] == "ACTIVE" and bad["state"] == "HALTED" and set(bad["blocked"]) == {"E1", "E2", "E3"}
    own = SX.own_positions({"E1": {"shares": 70}, "X": {"shares": 5}}, {"external": {"E1": 20, "X": 5}})
    assert own == {"E1": {"shares": 50.0}}                                       # 직접 보유분은 빼고 본다
    assert SX.diff_orders({}, own, {}, 0.12) == [(SX.diff_orders({}, own, {}, 0.12)[0][0], "시장가 매도")]
    assert SX.diff_orders({}, own, {}, 0.12)[0][0]["qty"] == 50                  # 직접 산 20주·X 는 팔지 않음
    b, name, why = SX.make({"short": {"broker": "paper"}})
    assert b is None and name == "paper"
    assert SX.execute({"short": {"broker": "kis_demo", "exec_on": "oracle"}}, where="github") == []   # 한 곳에서만 보낸다


def test_risk_blocks_on_reconcile():
    """대사가 어긋난 종목은 새 매수 금지 · 대사 HALTED 면 계좌 정지 · 모의(paper) 대사는 영향 없음."""
    import risk as RK
    plan = {"orders": [{"side": "buy", "code": "X", "qty": 1, "est_price": 100.0}, {"side": "buy", "code": "Y", "qty": 1, "est_price": 100.0},
                       {"side": "sell", "code": "Z", "qty": 1, "est_price": 100.0}]}
    p = RK.apply(json.loads(json.dumps(plan)), "2026-10-30", {}, 1e6, today="2026-10-30",
                 recon={"mode": "kis_demo", "state": "REDUCING", "blocked": ["X"], "why": ["x"]})
    assert {o["code"] for o in p["orders"]} == {"Y", "Z"}
    p = RK.apply(json.loads(json.dumps(plan)), "2026-10-30", {}, 1e6, today="2026-10-30",
                 recon={"mode": "kis_demo", "state": "HALTED", "blocked": [], "why": ["잔고 조회 실패"]})
    assert p["orders"] == [] and p["risk"]["state"] == "HALTED"
    p = RK.apply(json.loads(json.dumps(plan)), "2026-10-30", {}, 1e6, today="2026-10-30",
                 recon={"mode": "paper", "state": "HALTED", "blocked": ["X"]})
    assert len(p["orders"]) == 3


def test_dsr_counts_all_trials():
    """시도를 더 많이 했다고 적으면 진짜 실력일 확률(DSR)은 내려가고 운의 기준선은 올라간다."""
    import science_lab as SL
    rng = np.random.default_rng(0)
    r = rng.normal(0.0008, 0.01, 1500)
    srs = list(rng.normal(0.0, 0.03, 150))
    p1, s1 = SL.dsr(r, srs)
    p2, s2 = SL.dsr(r, srs, 180000)
    assert p2 < p1 and s2 > s1


def test_toss_broker_readonly_and_order_lock():
    """토스(명세 1.2.28): 보유·매수 가능 금액 해석 · accountSeq 0 도 헤더로 · 주문 본문 형식 · 명세가 바뀌면 안 보냄 · 체결 조회."""
    import types
    import brokers
    fake_tp = types.SimpleNamespace(BASE="https://x", TOKEN="/nonexistent/tok", load_env=lambda: {},
                                    get_token=lambda env: ("TOKEN", {"status": "cached"}))
    sys.modules["toss_probe"] = fake_tp
    calls = []

    class R:
        def __init__(self, st, j):
            self.status_code, self._j, self.text = st, j, ""

        def json(self):
            return self._j

    class S:
        def request(self, method, url, params=None, json=None, headers=None, timeout=None):
            calls.append((method, url, headers.get("X-Tossinvest-Account"), json, params))
            if url.endswith("/api/v1/accounts"):
                return R(200, {"result": [{"accountNo": "111", "accountSeq": 0, "accountType": "BROKERAGE"}]})
            if url.endswith("/api/v1/holdings"):
                return R(200, {"result": {"items": [
                    {"symbol": "069500", "marketCountry": "KR", "quantity": "3", "averagePurchasePrice": "90000"},
                    {"symbol": "AAPL", "marketCountry": "US", "quantity": "2", "averagePurchasePrice": "180.5"},
                    {"symbol": "114800", "marketCountry": "KR", "quantity": "0", "averagePurchasePrice": "1000"}]}})
            if url.endswith("/api/v1/buying-power"):
                return R(200, {"result": {"currency": "KRW", "cashBuyingPower": "1234567"}})
            if url.endswith("/api/v1/orders") and method == "POST":
                return R(200, {"result": {"orderId": "OID1", "clientOrderId": json.get("clientOrderId")}})
            if url.endswith("/api/v1/orders") and method == "GET":
                return R(200, {"result": {"orders": [{"orderId": "OID1", "symbol": "069500", "side": "BUY", "status": "FILLED",
                                                      "quantity": "1", "orderedAt": "2026-11-06T08:41:00+09:00",
                                                      "execution": {"filledQuantity": "1", "averageFilledPrice": "90100"}}],
                                           "nextCursor": None, "hasNext": False}})
            return R(404, {})
    old_spec = brokers.TossBroker.spec_version
    try:
        tb = brokers.TossBroker(env={"TOSS_CLIENT_ID": "a", "TOSS_CLIENT_SECRET": "b", "TOSS_ACCOUNT": "111"}, session=S())
        tb.spec_version = lambda: "1.2.28"
        a = tb.account()
        assert a == {"cash": 1234567.0, "positions": {"069500": {"shares": 3.0, "avg": 90000.0}}}      # 미국 보유는 국내 대사에서 뺀다
        assert tb.seq == 0 and all(h == "0" for m, u, h, b, p in calls if not u.endswith("/accounts"))  # accountSeq 0 도 헤더로
        r = tb.order("buy", "069500", 2, "limit", 101700, tif="OPG", client_id="m20261106b069500")
        assert not r["ok"] and r["body"] == {"symbol": "069500", "side": "BUY", "orderType": "LIMIT", "timeInForce": "OPG",
                                             "quantity": "2", "price": "101700", "clientOrderId": "m20261106b069500"}  # 잠김 = 안 보냄
        assert not any(m == "POST" for m, *_ in calls)
        tb.live_orders = True
        r = tb.order("sell", "069500", 1, "market", tif="OPG")
        posted = [b for m, u, h, b, p in calls if m == "POST"]
        assert r["ok"] and r["order_no"] == "OID1" and "price" not in posted[-1] and posted[-1]["orderType"] == "MARKET"
        tb.spec_version = lambda: "1.3.0"                                                             # 토스가 명세를 바꾸면 멈춤
        n = len(calls)
        assert not tb.order("buy", "069500", 1, "limit", 1000)["ok"] and len(calls) == n
        assert not tb.order("buy", "069500", 1.5, "limit", 1000)["ok"]                               # 정수 수량만
        f = tb.daily_fills("2026-11-06", "2026-11-06")
        assert f == [{"order_no": "OID1", "code": "069500", "date": "2026-11-06", "side": "buy", "qty": 1.0, "filled": 1.0,
                      "avg": 90100.0, "status": "FILLED"}]
    finally:
        brokers.TossBroker.spec_version = old_spec
        sys.modules.pop("toss_probe", None)


def test_toss_reconcile_off_server_reads_file():
    """토스는 GitHub 에서 접속이 안 됨 → 서버가 남긴 대사 기록을 읽고, 1영업일 넘게 묵으면 새 매수 금지."""
    import short_exec as SX
    with tempfile.TemporaryDirectory() as d:
        old = (SX.RECON, SX.MIRROR, SX.MIRROR_STATUS, SX.on_server)
        try:
            SX.RECON, SX.MIRROR = os.path.join(d, "recon.json"), os.path.join(d, "mirror.json")
            SX.MIRROR_STATUS = os.path.join(d, "mirror_status.json")                       # 저장소의 실제 연결 기록과 분리
            SX.on_server = lambda: False
            cfg = {"short": {"broker": "toss"}}
            r = SX.reconcile(cfg, acc={})
            assert r["mode"] == "toss" and r["state"] == "ACTIVE" and not r["ok"]          # 연결 전 = 모의만
            json.dump({"broker": "toss", "scale": 1.0}, open(SX.MIRROR, "w"))
            json.dump({"date": "2020-01-02", "mode": "toss", "state": "ACTIVE", "ok": True}, open(SX.RECON, "w"))
            r = SX.reconcile(cfg, acc={})
            assert r["state"] == "REDUCING" and "묵음" in r["why"][0]
            today = SX.kst_now().strftime("%Y-%m-%d")
            json.dump({"date": today, "mode": "toss", "state": "ACTIVE", "ok": True, "blocked": []}, open(SX.RECON, "w"))
            assert SX.reconcile(cfg, acc={})["state"] == "ACTIVE"
            assert SX.execute(cfg, where="github") == []                                   # 토스 주문은 서버에서만
            pub = SX.public_recon({"date": today, "mode": "toss", "state": "REDUCING", "broker_cash": 123456789,
                                   "mismatch": [{"code": "069500", "expect": 10, "broker": 7}],
                                   "slippage": [{"code": "069500", "broker_px": 1.0, "book_px": 1.0, "cost_bp": 3.2}]})
            txt = json.dumps(pub)
            assert "123456789" not in txt and '"expect"' not in txt and "broker_px" not in txt     # 공개 파일엔 금액·수량 없음
            assert pub["mismatch_codes"] == ["069500"] and pub["slippage_bp"] == [3.2]
        finally:
            SX.RECON, SX.MIRROR, SX.MIRROR_STATUS, SX.on_server = old


def test_krx_history_collector_resume_and_block():
    """KRX 긴 역사 수집기: 일봉은 KRX 직접(상장폐지 포함), 이어 받기, 빈 응답이 이어지면 그 단계만 멈춤, 접속 안 되면 바로 끝."""
    import types
    mode = {"m": "ok", "n": 0}

    def mcap(d, market="KOSPI"):
        if mode["m"] == "dead":
            return pd.DataFrame()
        y, n = int(d[:4]), 130
        return pd.DataFrame({"시가총액": np.arange(n, 0, -1) * 1e9}, index=pd.Index([f"{(i + y) % 300:06d}" for i in range(n)], name="티커"))

    def bars(s, e, *a, **k):
        idx = pd.bdate_range(s, e)[:3]
        return pd.DataFrame({"시가": 1, "고가": 2, "저가": 1, "종가": 2, "거래량": 10, "거래대금": 20}, index=pd.Index(idx, name="날짜"))

    def ohlcv(s, e, c, adj=True):
        mode["n"] += 1
        return pd.DataFrame() if mode["m"] == "block" else bars(s, e)

    def flows(s, e, c, detail=False):
        idx = pd.bdate_range(s, e)[:3]
        return pd.DataFrame({"기관합계": 1, "개인": -2, "외국인합계": 1}, index=pd.Index(idx, name="날짜"))
    stock = SimpleNamespace(get_market_cap=mcap, get_market_ticker_name=lambda c: "", get_market_trading_value_by_date=flows,
                            get_etf_ohlcv_by_date=lambda s, e, c: bars(s, e) if int(s[:4]) >= 2010 else pd.DataFrame())
    pk, web, kw = types.ModuleType("pykrx"), types.ModuleType("pykrx.website"), SimpleNamespace(get_market_ohlcv_by_date=ohlcv)
    pk.stock, web.krx, pk.website = stock, kw, web
    saved = {k: sys.modules.get(k) for k in ("pykrx", "pykrx.website")}
    sys.modules.update({"pykrx": pk, "pykrx.website": web})
    import collect_krx_history as C
    old = (C.DIR, C.CACHE, C.time.sleep, sys.argv)
    try:
        with tempfile.TemporaryDirectory() as d:
            C.DIR, C.CACHE, C.time.sleep = d, os.path.join(d, "krx_cache"), (lambda x: None)
            sys.argv = ["x", "--years", "2016", "2018", "--top", "20"]
            C.main()
            o = pd.read_csv(os.path.join(d, "krx_ohlcv.csv.gz"), dtype={"종목코드": str})
            f = pd.read_csv(os.path.join(d, "krx_flows.csv.gz"), dtype={"종목코드": str})
            assert "거래대금" in o.columns and list(f.columns) == ["날짜", "종목코드", "외국인순매수", "기관순매수", "개인순매수"]
            n1, mode["n"] = mode["n"], 0
            C.main()                                                                # 이어 받기: 다시 받지 않는다
            assert n1 > 0 and mode["n"] == 0
            u = pd.read_csv(os.path.join(d, "krx_universe.csv.gz"), dtype={"종목코드": str})
            need = C.needed_years(u, 2016, 2018)
            assert all(v == list(range(v[0], v[-1] + 1)) and 2016 <= v[0] and v[-1] <= 2018 for v in need.values())
        with tempfile.TemporaryDirectory() as d:
            C.DIR, C.CACHE = d, os.path.join(d, "krx_cache")
            mode.update(m="block", n=0)
            sys.argv = ["x", "--years", "2000", "2018", "--top", "20"]
            C.main()                                                                # 일봉만 멈추고 수급·ETF 는 계속
            assert mode["n"] == 60 and not os.path.exists(os.path.join(d, "krx_ohlcv.csv.gz"))
            assert os.path.exists(os.path.join(d, "krx_flows.csv.gz")) and not glob_any(os.path.join(d, "krx_cache"), "ohlcvk_")
            mode["m"] = "dead"
            try:
                C.main()
                raise AssertionError("접속이 안 되면 바로 끝나야 함")
            except SystemExit as ex:
                assert ex.code == 2
    finally:
        C.DIR, C.CACHE, C.time.sleep, sys.argv = old
        for k, v in saved.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)


def glob_any(d, prefix):
    return os.path.isdir(d) and any(n.startswith(prefix) for n in os.listdir(d))


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
