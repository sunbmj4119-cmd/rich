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
            assert len(L.pool_index(ms, "SLOW")) == len(full) - len({"E:oo_rev"} | L.geared_gate())   # 기본 관문 = 긴 역사 + 레버리지·인버스
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


def test_krx_adjust_splits_and_rights():
    """KRX 원본가 보정: 등락률(기준가 대비)로 50:1 분할·유상증자 정확히, 등락률 없으면 분할 비율로 추정하고 그날은 끊김."""
    import history_lab as H
    d = pd.bdate_range("2018-01-02", periods=8)
    C = np.array([[100., 1000], [102, 1000], [104, 1000], [2.1, 1000], [2.12, 900], [2.1, 880], [np.nan, 880], [2.2, 880]])
    FR = np.array([[0, 0], [2, 0], [1.96, 0], [0.96, 0], [0.95, -5.26], [-0.94, -2.22], [np.nan, 0], [4.76, 0]])
    fac, b = H.krx_adjust(C, FR, d)
    A = C * fac
    assert abs(A[3, 0] / A[2, 0] - 1.0096) < 1e-3 and abs(A[0, 0] - 2.0) < 1e-3      # 분할 날도 실제 등락(+0.96%) 그대로
    assert abs(A[4, 1] / A[3, 1] - (1 - 0.0526)) < 1e-3 and not b.any()             # 증자 권리락: 기준가 대비 -5.26%
    fac2, b2 = H.krx_adjust(C, None, d)
    assert abs((C * fac2)[2, 0] - 2.08) < 1e-6 and b2[3, 0] and b2.sum() == 1         # 1/50 로 추정, 분할 날만 끊김


def test_no_geared_excludes_leverage_inverse():
    """short.no_geared: 전천후 카드 5장·단기 인버스 모델을 후보에서 빼고, 증권사 매수도 막는다 (매도는 허용)."""
    import allweather as AW
    import short_exec as SX
    on, off = AW.allowed_rules(False), AW.allowed_rules(True)
    assert set(off) - set(on) == AW.GEARED_RULES and "H:CASH" in on and len(off) == len(AW.RULES)
    assert AW.GEARED_CODES == {"114800", "122630", "252670"}
    assert AW.no_geared({"short": {}}) and not AW.no_geared({"short": {"no_geared": False}})
    rng = np.random.default_rng(1)
    T = 400
    dates = pd.bdate_range("2015-01-02", periods=T)
    Cp = np.cumprod(1 + rng.normal(0, 0.01, (T, AW.A)), axis=0) * 1000
    F = AW.features(Cp, dates)
    gj = [AW.CODES.index(c) for c in AW.GEARED_CODES]
    for k in on:                                                        # 남은 카드는 레버리지·인버스를 한 번도 들지 않는다
        assert np.nan_to_num(AW.targets(F, Cp, k)[:, gj]).max() == 0, k
    models = [{"id": "E:ls_rev", "kind": "etf"}, {"id": "E:hold", "kind": "etf"}]
    old = L.long_gate
    try:
        L.long_gate = lambda: set()
        assert [models[i]["id"] for i in L.pool_index(models, "SLOW")] == ["E:hold"]
    finally:
        L.long_gate = old
    pairs = [({"side": "buy", "code": "122630", "qty": 1, "kind": "limit"}, "x"),
             ({"side": "sell", "code": "114800", "qty": 3, "kind": "market"}, "y"),
             ({"side": "buy", "code": "069500", "qty": 1, "kind": "limit"}, "z")]
    g = SX.guard_geared(pairs, {"short": {"no_geared": True}})
    assert g[0][0]["kind"] == "skip" and "레버리지" in g[0][1] and g[1] == pairs[1] and g[2] == pairs[2]
    assert SX.guard_geared(pairs, {"short": {"no_geared": False}}) == pairs


def test_stock_stop_loss_take_profit_option():
    """run_stock 의 손절·익절 선택: 전날 종가가 매입가 대비 -sl/+tp 를 넘으면 오늘 시가에 판다 (없으면 예전과 같음)."""
    T, N = 8, 2
    O = np.array([[100, 50]] * T, float)
    C = np.array([[100, 50], [100, 50], [95, 50], [92, 50], [92, 50], [120, 50], [120, 50], [120, 50]], float)
    O[1:, 0] = C[:-1, 0]                                           # 시가 = 전날 종가
    R = np.tile([1.0, 2.0], (T, 1))
    D = SimpleNamespace(T=T, N=N, O=O, Cf=C, gate={"none": np.ones(T, bool)}, ranks=lambda sel, combo=None: R)
    base = {"id": "x", "kind": "stock", "sel": "REV1", "K": 1, "hold": "H10", "gate": "none"}
    log = []
    L.run_stock(D, base, log=log)
    assert [x["side"] for x in log] == ["buy"]                      # 손절 없으면 순위 1등이라 계속 보유
    log = []
    L.run_stock(D, {**base, "sl": 0.07}, log=log)
    sells = [x["t"] for x in log if x["side"] == "sell"]
    assert sells and sells[0] == 4                                 # 3일 종가 92 ≤ 100×0.93 → 4일 시가에 손절
    log = []
    L.run_stock(D, {**base, "tp": 0.10}, log=log)
    assert [x for x in log if x["side"] == "buy"][0]["t"] == 1
    assert [x["t"] for x in log if x["side"] == "sell"][0] == 6     # 100 에 사서 5일 종가 120 ≥ 110 → 6일 시가에 익절
    O2, C2 = O.copy(), C.copy()
    C2[2:, 0] = 112.0
    O2[1:, 0] = C2[:-1, 0]
    D2 = SimpleNamespace(T=T, N=N, O=O2, Cf=C2, gate={"none": np.ones(T, bool)}, ranks=lambda sel, combo=None: R)
    log = []
    L.run_stock(D2, {**base, "tp": 0.10}, log=log)
    assert [x["t"] for x in log if x["side"] == "sell"][0] == 3     # 2일 종가 112 ≥ 110 → 3일 시가에 익절


def test_size_policy_rule_allocation_and_execution():
    """자산 규모별 전략: 작을수록 공격 몫↑, 정수 주식 배분, 시뮬레이션, 서버 실행(계좌 금액 → 주식 수)·연결(내 현금·종목 제외)."""
    import size_policy as SZ
    import short_exec as SX
    import allweather as AW
    sp = SZ.DEFAULT
    sh = [SZ.aggr_share(e, sp) for e in (200_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000, 50_000_000)]
    assert sh == [1.0, 1.0, 0.5, 0.2, 0.0, 0.0] and all(a >= b for a, b in zip(sh, sh[1:]))
    assert SZ.aggr_share(5_000_000, {**sp, "loss_budget": 2_000_000}) == 1.0           # 감당 손실을 키우면 더 오래 공격적
    q = SZ.allocate([0.5, 0.5], [161_300, 73_675], 200_000)
    assert list(q) == [0, 2]                                                            # 1주가 비싸면 살 수 있는 쪽으로 채워 현금으로 놀지 않게
    assert list(SZ.allocate([0.05, 0.95], [100_000, 10_000], 200_000)) == [0, 19]       # 5% 자리에 10만 원짜리 1주를 몰아넣지 않음
    q = SZ.allocate([0.5, 0.5, 0.0], [22_590, 14_095, 10_000], 200_000)
    assert list(q) == [4, 7, 0] and (q * np.array([22_590, 14_095, 10_000])).sum() <= 200_000
    rng = np.random.default_rng(3)
    T, A = 300, AW.A
    dates = pd.bdate_range("2020-01-02", periods=T)
    Cp = np.cumprod(1 + rng.normal(0.0003, 0.01, (T, A)), axis=0) * rng.uniform(10_000, 120_000, A)
    Op = Cp * (1 + rng.normal(0, 0.002, (T, A)))
    W1 = np.zeros((T, A)); W1[:, 0] = 1.0
    W2 = np.zeros((T, A)); W2[:, 5] = 0.5; W2[:, 8] = 0.5
    r = SZ.simulate(dates, Op, Cp, W1, W2, 1, 200_000, sp)
    assert np.isfinite(r["E"][1:]).all() and r["cash"] >= -1e-6 and r["a"][-1] == 1.0
    big = SZ.simulate(dates, Op, Cp, W1, W2, 1, 20_000_000, sp)
    assert big["a"][-1] == 0.0                                                          # 큰 돈 = 전천후 쪽
    cfg = {"short": {"broker_sleeves": ["size"], "size_policy": {}}}
    plan = {"made": "2026-11-09", "size": {"aw": {"114260": 0.6, "261240": 0.4}, "aggr": {"132030": 0.5, "261240": 0.5},
                                           "rebalance": True, "px": {"114260": 62_755, "261240": 14_095, "132030": 22_590}}}
    mirror = {"external_cash": 10_026, "hwm": 0, "last_target": None}
    acct = {"cash": 210_026, "positions": {}}
    tgt, info = SX.size_targets(cfg, plan, mirror, acct, {})
    spent = sum(q * plan["size"]["px"][c] for c, q in tgt.items())
    assert info["aggr"] == 1.0 and info["last_eq"] == 200_000 and spent <= 200_000 and set(tgt) <= {"132030", "261240"}
    plan["size"]["rebalance"] = False
    keep, info2 = SX.size_targets(cfg, plan, {**mirror, "last_target": tgt}, acct, {})
    assert keep == tgt and not info2["rebalanced"]                                     # 다시 맞추는 날이 아니면 목표 유지
    assert SX.size_targets(cfg, {"made": "x"}, mirror, acct, {})[0] is None            # 계획에 목표가 없으면 보내지 않음

    class FakeB:
        def price(self, c):
            return 20_000.0
    with tempfile.TemporaryDirectory() as d:
        old = (SX.MIRROR, SX.MIRROR_STATUS)
        try:
            SX.MIRROR, SX.MIRROR_STATUS = os.path.join(d, "m.json"), os.path.join(d, "ms.json")
            cfg2 = {"short": {"broker_sleeves": ["size"], "broker_capital": 200_000}}
            m = SX.link_size(cfg2, FakeB(), "toss", {"cash": 210_026, "positions": {"005930": {"shares": 3, "avg": 50_000}}})
            assert m["external"] == {"005930": 3.0} and m["external_cash"] == 10_026 and m["hwm"] == 200_000
            pub = json.load(open(SX.MIRROR_STATUS))
            assert "capital" not in pub and "external_cash" not in pub                 # 금액은 공개 파일에 없음
            m2 = SX.link_size(cfg2, FakeB(), "toss", {"cash": 110_000, "positions": {"005930": {"shares": 3, "avg": 50_000},
                                                                                    "261240": {"shares": 5, "avg": 14_000}}})
            assert m2["external"] == {"005930": 3.0} and m2["external_cash"] == 10_000      # 다시 연결해도 자동매매가 산 종목은 자동매매 것
        finally:
            SX.MIRROR, SX.MIRROR_STATUS = old


def test_overnight_stream_timing_and_costs():
    """밤사이 보유: t-2일 저녁 신호로 t-1일 종가에 사서 t일 시가에 판다 (미래 정보 없음) · 120일선 아래면 쉼 · 비용 차감."""
    import optimize_lab as O
    import allweather as AW
    T, A = 200, AW.A
    Cp = np.tile(np.linspace(100, 200, T)[:, None], (1, A))           # 꾸준히 오름 → 120일선 위
    Op = np.r_[Cp[:1], Cp[:-1] * 1.001]                               # 시가 = 전날 종가 +0.1%
    dates = pd.bdate_range("2020-01-02", periods=T)
    r = O.overnight_stream(dates, Op, Cp, slip_fixed=0.0)
    on = r[r != 0]
    assert len(on) > 0 and np.allclose(on, 1.001 * (1 - AW.COMM) / (1 + AW.COMM) - 1)
    assert (r[:97] == 0).all()                                       # 120일선이 생기기 전(80%=96일 + 2일 지연)엔 안 함
    Cd = Cp[::-1].copy()                                             # 꾸준히 내림 → 120일선 아래 → 안 함
    Od = np.r_[Cd[:1], Cd[:-1] * 1.001]
    assert (O.overnight_stream(dates, Od, Cd, slip_fixed=0.0)[120:] == 0).all()
    st = O.steadiness(r, dates)
    assert st["day_win"] == 1.0 and 0 < st["days_in"] < 1


def test_setup_lab_trades_no_overlap_and_costs():
    """조건 매매: 신호 다음 날 시가에 사서 h일 뒤 시가에 판다 · 보유 중 같은 종목 신호는 세지 않음 · 비용(거래세 포함) 차감."""
    import setup_lab as SL
    T, N = 30, 2
    O = np.full((T, N), 100.0)
    O[:, 0] = 100 + np.arange(T)
    S = np.zeros((T, N), bool)
    S[5, 0] = S[6, 0] = S[12, 0] = True                              # 6일 신호는 5일 거래(6~11일 보유) 중이라 무시
    P = {"O": O, "dates": pd.bdate_range("2020-01-01", periods=T), "breaks": np.zeros((T, N), bool)}
    tr = SL.trades(P, np.full(T, 100.0), S, 5, "2020-01-01", None)
    assert len(tr) == 2 and list(tr["j"]) == [0, 0]
    cb, cs = SL.COMM + SL.SLIP, SL.COMM + SL.SLIP + SL.TAX
    assert abs(tr["net"].iloc[0] - (O[11, 0] * (1 - cs) / (O[6, 0] * (1 + cb)) - 1)) < 1e-12
    assert abs(tr["gross"].iloc[0] - (O[11, 0] / O[6, 0] - 1)) < 1e-12


def test_bdw_rules_and_withdraw_hint():
    """불장단타왕 규칙: 근거 이탈 청산(다음 날 시가)·시간 손절 · 수익 출금 안내(넣은 돈 2배)."""
    import bdw_lab as B
    import short_exec as SX
    T, N = 40, 1
    O = np.full((T, N), 100.0)
    C = np.full((T, N), 101.0)
    S = np.zeros((T, N), bool)
    S[5, 0] = True
    X = np.zeros((T, N), bool)
    X[9, 0] = True                                                   # 9일 종가 이탈 → 10일 시가 청산
    P = {"O": O, "Cf": C, "dates": pd.bdate_range("2020-01-01", periods=T), "breaks": np.zeros((T, N), bool)}
    tr = B.trades_dyn(P, np.full(T, 100.0), S, X, 20, "2020-01-01", None)
    assert len(tr) == 1 and int(tr["e"].iloc[0]) == 6 and int(tr["x"].iloc[0]) == 10
    C2 = C.copy()
    C2[10, 0] = 99.0                                                 # 5일째(6+4=10일) 종가가 매입가 아래 → 11일 시가
    P2 = {**P, "Cf": C2}
    tr2 = B.trades_dyn(P2, np.full(T, 100.0), S, np.zeros((T, N), bool), 20, "2020-01-01", None, time_stop=5)
    assert int(tr2["x"].iloc[0]) == 11
    assert SX.withdraw_hint(400_000, 200_000) and not SX.withdraw_hint(399_999, 200_000) and not SX.withdraw_hint(1, 0)


def test_bdw_intraday_exits_and_portfolio():
    """불장단타왕 분봉 엔진: -1% 손절 · +2% 절반 익절 뒤 본절 · +6% 목표 · 선 이탈 · 시간 손절 · 동시 2종목·하루 손실 한도."""
    import bdw_intraday as BI
    R = BI.RULES
    V = BI.VENUES["upbit"]
    cin = lambda e: e * (1 + V["fee"])                                           # noqa: E731
    sell = 1 - V["fee"] - V["slip"]

    def A(o, h, lo, c, vw):
        n = len(o)
        ts = pd.date_range("2024-01-01", periods=n, freq="5min").to_numpy()
        return {"o": np.array(o, float), "h": np.array(h, float), "lo": np.array(lo, float), "c": np.array(c, float),
                "vw": np.array(vw, float), "last": np.zeros(n, bool), "day": np.zeros(n), "ts": ts}
    e = 100 * (1 + V["slip"])
    k, net, ent, why = BI.simulate_trade(A([100, 100, 100], [100, 100.5, 100], [100, 98.5, 100], [100, 99.5, 100], [90, 90, 90]), 0, "upbit", R)
    assert why == "손절" and abs(net - (e * 0.99 * sell / cin(e) - 1)) < 1e-9
    k, net, ent, why = BI.simulate_trade(A([100, 100, 101, 101], [100, 102.5, 101, 101], [100, 99.5, 99.9, 99], [100, 101, 100.5, 100], [90] * 4), 0, "upbit", R)
    assert why == "본전 손절" and abs(net - ((0.5 * e * 1.02 + 0.5 * e) * sell / cin(e) - 1)) < 1e-9
    k, net, ent, why = BI.simulate_trade(A([100, 100, 103], [100, 102.5, 107], [100, 99.5, 102], [100, 101, 106], [90] * 3), 0, "upbit", R)
    assert why == "목표 +6%" and abs(net - ((0.5 * e * 1.02 + 0.5 * e * 1.06) * sell / cin(e) - 1)) < 1e-9
    k, net, ent, why = BI.simulate_trade(A([100, 100, 100.4, 100.2], [100, 100.5, 100.5, 100.3], [100, 99.6, 99.8, 100], [100, 100.4, 100.2, 100], [90, 90, 101, 101]), 0, "upbit", R)
    assert why == "선 이탈" and k == 3                                          # 2봉 종가가 선 아래 → 3봉 시가
    o = [100] + [100] * 8
    k, net, ent, why = BI.simulate_trade(A(o, [100.3] * 9, [99.5] * 9, [99.8] * 9, [90] * 9), 0, "upbit", R)
    assert why == "시간 손절" and k == 1 + R["time_bars"]                       # 6봉째 종가가 매입가 아래 → 다음 봉
    # 포트폴리오: 같은 시각 후보 3개 → 동시 2종목까지 · 하루 손실 한도
    t0 = pd.Timestamp("2024-01-02 10:00")
    c = pd.DataFrame([{"sym": f"S{i}", "t_sig": t0, "t_in": t0, "t_out": t0 + pd.Timedelta(minutes=30), "net": -0.03, "why": "손절",
                       "value_day": 100 - i, "ret_day": 0.01} for i in range(3)])
    tk, curve = BI.portfolio(c)
    assert len(tk) == 2 and set(tk["sym"]) == {"S0", "S1"}
    later = pd.DataFrame([{"sym": "S9", "t_sig": t0 + pd.Timedelta(hours=1), "t_in": t0 + pd.Timedelta(hours=1), "t_out": t0 + pd.Timedelta(hours=2),
                           "net": 0.02, "why": "x", "value_day": 1, "ret_day": 0.0}])
    tk2, _ = BI.portfolio(pd.concat([c, later], ignore_index=True), {"day_limit": 0.02})
    assert "S9" not in set(tk2["sym"])                                          # 오전에 -3%×50%×2 = -3% → 하루 한도 -2% 넘어 멈춤
    # 전체 흐름: 무작위 걸음 자료 두 종목 + 기준
    rng = np.random.default_rng(5)
    data = {}
    for sym in ("BTC", "AAA", "BBB"):
        n = 4000
        ts = pd.date_range("2024-01-01", periods=n, freq="5min")
        cl = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, n)))
        op = np.r_[cl[0], cl[:-1]]
        data[sym] = pd.DataFrame({"ts": ts, "open": op, "high": np.maximum(op, cl) * 1.002, "low": np.minimum(op, cl) * 0.998,
                                  "close": cl, "volume": rng.lognormal(10, 1, n)})
    cand = BI.candidates(data, "upbit", bench="BTC")
    tk, curve = BI.portfolio(cand)
    sm = BI.summary(tk, curve)
    assert sm["trades"] == len(tk) and (len(tk) == 0 or (tk["t_out"] >= tk["t_in"]).all())


def test_intraday_regular_session_5m():
    """토스 1분봉(넥스트레이드 08~20시 포함) → 정규장(09:00~15:30)만 5분봉으로 · 미국은 뉴욕 09:30~16:00."""
    import intraday_lab as IL
    ts = pd.date_range("2026-11-06 08:00", "2026-11-06 20:00", freq="1min")
    d = pd.DataFrame({"ts": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": np.arange(len(ts), dtype=float), "volume": 1.0})
    r = IL.regular_5m(d, "kr")
    assert len(r) == 78 and str(r["ts"].iloc[0]) == "2026-11-06 09:00:00" and str(r["ts"].iloc[-1]) == "2026-11-06 15:25:00"
    assert r["close"].iloc[0] == 64 and r["volume"].iloc[0] == 5                     # 09:00~09:04 다섯 봉, 종가 = 09:04
    u = IL.regular_5m(d.assign(ts=pd.date_range("2026-11-06 04:00", periods=len(ts), freq="1min")), "us")
    assert str(u["ts"].iloc[0]) == "2026-11-06 09:30:00" and u["ts"].dt.strftime("%H:%M").max() == "15:55"


def test_intraday_sweep_matches_engine():
    """격자 탐색의 빠른 계산 = bdw_intraday 한 건씩 계산 (청산 봉·순수익) · 장부 = portfolio (거래 수·총수익) · 끝까지 돌아감."""
    import itertools
    import bdw_intraday as BI
    import intraday_sweep as S
    rng = np.random.default_rng(3)
    bd = pd.bdate_range("2025-11-10", periods=100)
    tod = pd.timedelta_range("09:00:00", "15:25:00", freq="5min")
    ts = pd.DatetimeIndex((bd.values[:, None] + tod.values[None, :]).ravel())
    data = {}
    for k in range(7):
        r = rng.normal(0, 0.003, len(ts))
        cl = 10000 * np.exp(np.cumsum(r))
        op = np.r_[cl[0], cl[:-1]] * np.exp(rng.normal(0, 0.0005, len(ts)))
        data["069500" if k == 0 else f"S{k}"] = pd.DataFrame({
            "ts": ts, "open": op, "high": np.maximum(op, cl) * np.exp(np.abs(rng.normal(0, 0.002, len(ts)))),
            "low": np.minimum(op, cl) * np.exp(-np.abs(rng.normal(0, 0.002, len(ts)))), "close": cl,
            "volume": rng.lognormal(10, 0.8, len(ts)) * (1 + 3 * (rng.random(len(ts)) < 0.05))})
    E = {"touch": [0.005], "chase": [0.02], "vol_mult": [1.5], "window": [300], "top_n": [0, 3]}
    X = {"stop": [0.005, 0.01, 0.03], "tp1": [0.01, 0.02], "frac": [0.5, 1.0], "tp2": [0.04], "time_bars": [6, 0], "line_exit": [1, 0]}
    g = S.build(data, "kr", "069500", E)
    assert g["H"] == 78 and S.n_unique(E, X) == 2 * 3 * 2 * 2 * 2 * 2
    sig = S.signals(0.005, 0.02, 1.5, 300, 0)
    assert len(sig) > 20 and (np.diff(g["ts"][sig + 1]) >= 0).all()
    start = {si: int(np.flatnonzero(g["sym"] == si)[0]) for si in range(len(g["syms"]))}
    A = {si: BI.arrays(BI.prepare(data[s], "kr")) for si, s in enumerate(g["syms"])}
    P = S.Paths(sig, exit_=X)
    for s, a, tb in itertools.product(X["stop"], X["tp1"], X["time_bars"]):
        net, xo, wy = P.exit(s, a, 0.5, 0.04, tb, 1, why=True)
        R = {**BI.RULES, "stop": s, "tp1": a, "tp1_frac": 0.5, "tp2": 0.04, "time_bars": tb}
        for q, i in enumerate(sig):
            si = int(g["sym"][i])
            k, nt, _, _ = BI.simulate_trade(A[si], int(i - start[si]), "kr", R)
            assert abs(nt - net[q]) < 1e-9 and k == xo[q] - start[si]
    net1, xo1, wy1 = P.exit(0.01, 0.02, 1.0, 0.04, 0, 0, why=True)                # 1차에서 전부 · 선 이탈 끔 · 시간 손절 끔
    assert set(np.unique(wy1)) <= {0, 2, 6}
    net, xo = P.exit(0.01, 0.02, 0.5, 0.04, 6, 1)
    c = pd.DataFrame({"sym": [g["syms"][k] for k in g["sym"][sig]], "t_sig": pd.to_datetime(g["ts"][sig]), "t_in": pd.to_datetime(g["ts"][sig + 1]),
                      "t_out": pd.to_datetime(g["ts"][xo]), "net": net, "why": "x", "value_day": g["val_day"][sig], "ret_day": 0.0})
    tk, curve = BI.portfolio(c, {"top_n": 10 ** 9})
    _, L = S._lists(sig)
    n, mean, tot, mdd, _, _ = S.book(L[0], g["ts"][xo].tolist(), L[1], L[2], L[3], net.tolist(), S.weight(0.01))
    assert n == len(tk) and abs(tot - (curve.iloc[-1] - 1)) < 1e-9
    assert len(S.signals(0.005, 0.02, 1.5, 300, 3)) < len(sig)                     # 거래대금 상위 3종목만
    T = np.arange(12, dtype=float).reshape(3, 4)
    assert S.neighbor_median(T)[1, 1] == 5.0 and S.neighbor_median(T)[0, 0] == 1.0  # 자기+이웃 (격자 밖 제외)
    old = dict(S.CRIT)
    S.CRIT.update({"min_trades": 5, "n_random": 3})
    try:
        res = S.run("kr", None, "069500", workers=1, entry=E, exit_=X, data=data, log=lambda m: None)
    finally:
        S.CRIT.clear()
        S.CRIT.update(old)
    assert set(res["picks"]) == {"A→B", "B→A"} and isinstance(res["pass"], bool) and res["cells_unique"] == S.n_unique(E, X)
    assert set(res["marginals"]["stop"]) == {"0.005", "0.01", "0.03"} and "persistence" in res


def test_intraday_sweep_round2_multiday_exits():
    """2차 (며칠 보유·추세 이탈·같은 비중): 빠른 계산 = 한 건씩 따라가는 계산 (청산 봉·순수익) · 밤사이 갭은 시가에 손절."""
    import itertools
    import bdw_intraday as BI
    import intraday_sweep as S
    rng = np.random.default_rng(5)
    bd = pd.bdate_range("2025-11-10", periods=80)
    tod = pd.timedelta_range("09:00:00", "15:25:00", freq="5min")
    ts = pd.DatetimeIndex((bd.values[:, None] + tod.values[None, :]).ravel())
    data = {}
    for k in range(6):
        cl = 10000 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, len(ts))))
        gap = rng.normal(0, 0.003, len(ts)) * (np.arange(len(ts)) % 78 == 0)
        op = np.r_[cl[0], cl[:-1]] * np.exp(gap + rng.normal(0, 0.0005, len(ts)))
        data["069500" if k == 0 else f"S{k}"] = pd.DataFrame({
            "ts": ts, "open": op, "high": np.maximum(op, cl) * np.exp(np.abs(rng.normal(0, 0.003, len(ts)))),
            "low": np.minimum(op, cl) * np.exp(-np.abs(rng.normal(0, 0.003, len(ts)))), "close": cl,
            "volume": rng.lognormal(10, 0.8, len(ts)) * (1 + 3 * (rng.random(len(ts)) < 0.05))})
    g = S.build(data, "kr", "069500", S.ENTRY2)
    sig = S.signals(0.01, 0.04, 1.5, 390, 0)
    assert len(sig) > 20
    V = BI.VENUES["kr"]

    def ref(i, s, a, f, b, ln, md):
        o, h, lo, c, vw, tr, sym, ds = g["o"], g["h"], g["l"], g["c"], g["vw"], g["trend"], g["sym"], g["dayseq"]
        e = i + 1
        E = o[e] * (1 + V["slip"])
        cin, sell = E * (1 + V["fee"]), 1 - (V["fee"] + V["slip"] + V["tax"])
        stop, half, left, got, k = E * (1 - s), False, 1.0, 0.0, e
        while True:
            last = k + 1 >= len(o) or sym[k + 1] != sym[e] or ds[k + 1] - ds[e] > md
            if lo[k] <= stop:
                return k, (got + left * (stop if k == e else min(stop, o[k])) * sell) / cin - 1
            if not half and h[k] >= E * (1 + a):
                got, left, half, stop = got + f * E * (1 + a) * sell, left - f, True, max(stop, E)
                if f >= 1:
                    return k, got / cin - 1
            if half and h[k] >= E * (1 + b):
                return k, (got + left * E * (1 + b) * sell) / cin - 1
            if last:
                return k, (got + left * c[k] * sell) / cin - 1
            if (ln == 1 and c[k] < vw[k]) or (ln == 2 and not tr[k]):
                return k + 1, (got + left * o[k + 1] * sell) / cin - 1
            k += 1
    X = {"stop": [0.02, 0.10], "tp1": [0.03, 0.08], "frac": [0.5, 1.0], "tp2": [0.12], "time_bars": [0], "line_exit": [0, 1, 2], "max_days": [0, 3]}
    P = S.Paths(sig, exit_=X)
    assert P.H == 4 * g["H"]
    seen = set()
    for s, a, f, ln, md in itertools.product(X["stop"], X["tp1"], X["frac"], X["line_exit"], X["max_days"]):
        net, xo, wy = P.exit(s, a, f, 0.12, 0, ln, md, why=True)
        seen |= {S.WHY[w] for w in wy}
        for q, i in enumerate(sig):
            k, nt = ref(int(i), s, a, f, 0.12, ln, md)
            assert k == xo[q] and abs(nt - net[q]) < 2e-6
    assert {"보유 기한 정리", "추세 이탈", "손절"} <= seen
    old = dict(S.CFG)
    try:
        S.CFG.update({"sizing": "equal", "max_pos": 5})
        assert S.weight(0.10) == S.weight(0.02) == 0.2                              # 손절 폭과 상관없이 같은 돈
        S.CFG.update({"sizing": "risk", "max_pos": 2})
        assert S.weight(0.02) == 0.25 and S.weight(0.005) == 0.5
    finally:
        S.CFG.update(old)
    assert S.n_unique(S.ENTRY2, S.EXIT2) == 142_560
    assert S.n_trials({"kr": {"venue": "kr", "cells_unique": 3}, "round2": {"kr": {"venue": "kr", "cells_unique": 4}}, "n_trials": 9}) == 7


def test_dm2_evolved_cards_and_challenger_shadow():
    """5회차 진화 카드: 비중 합 1 · 레버리지·인버스 없음 · 변동성 반비례/조절/주간 확인의 성질 · 전천후 카드 목록은 그대로 · 공격 자리에 꽂아도 돈다."""
    import allweather as AW
    import size_policy as SP
    rng = np.random.default_rng(9)
    T = 700
    dates = pd.bdate_range("2023-01-02", periods=T)
    vol = np.linspace(0.004, 0.03, AW.A)                                        # 자산마다 변동성이 다름
    Cp = 10000 * np.exp(np.cumsum(rng.normal(0.0004, vol, (T, AW.A)), axis=0))
    Op = Cp * np.exp(rng.normal(0, 0.001, (T, AW.A)))
    F = AW.features(Cp, dates)
    geared = [AW.IX[k] for k in AW.GEARED_KEYS]
    W2 = AW.targets(F, Cp, "DM:2")
    for k in AW.DM_EVOLVED:
        W = AW.targets(F, Cp, k)
        rows = np.flatnonzero(~np.isnan(W).all(1))
        assert len(rows) > 5 and np.allclose(W[rows].sum(1), 1) and (W[rows] >= -1e-12).all(), k
        assert np.nansum(W[rows][:, geared]) == 0, k
        assert k not in AW.RULES and k not in AW.allowed_rules()                 # 전천후 16장 혼합은 바뀌지 않음
    Wv, Wt, Ww = AW.targets(F, Cp, "DM:2v"), AW.targets(F, Cp, "DM:2t"), AW.targets(F, Cp, "DM:2w")
    r2 = np.flatnonzero(~np.isnan(W2).all(1))
    risky = [j for j in range(AW.A) if j != AW.IX["CASH"]]
    assert any(not np.allclose(Wv[t], W2[t]) for t in r2)                          # 반반이 아닌 날이 있다
    assert all(set(np.flatnonzero(Wv[t] > 0)) <= set(np.flatnonzero(W2[t] > 0)) for t in r2)   # 같은 자산만
    assert all(Wt[t][risky].sum() <= W2[t][risky].sum() + 1e-9 for t in r2)      # 변동성 조절은 줄이기만
    cash = AW.IX["CASH"]
    rw = np.flatnonzero(~np.isnan(Ww).all(1))
    assert set(r2) <= set(rw)
    for t in sorted(set(rw) - set(r2)):                                           # 주간 확인으로 바뀐 날 = 단기채로만 옮김
        prev = Ww[[x for x in rw if x < t][-1]]
        assert Ww[t][cash] > prev[cash] and (Ww[t][risky] <= prev[risky] + 1e-12).all()
    sp = {**SP.DEFAULT, "aggr_card": "DM:2t", "paper_start": str(dates[600].date())}
    W_aw, W_ag, s = SP.weight_paths(dates, Op, Cp, sp, start=str(dates[300].date()))
    assert np.allclose(W_ag, AW.desired(Wt))
    ch = SP.challenger_forward(sp, (dates, Op, Cp))
    assert set(ch["cards"]) == {"DM:2", *SP.CHALLENGERS} and all(v["days"] == T - 600 for v in ch["cards"].values())


def test_chart_lab_stop_target_order_and_baseline():
    """차트 자리 실험실: 같은 날 손절·익절이면 손절 · 갭 하락은 시가에 손절 · 익절 · 기한 종가 · 겹침 없음 · 같은 날 다른 종목 기준."""
    import chart_lab as CL
    T, N = 40, 3
    O = np.full((T, N), 100.0)
    H, L, C = O * 1.01, O * 0.99, O.copy()
    # 종목 0: 진입(1일) 다음 날 고가 +6%·저가 −6% 같은 날 → 손절이 먼저
    H[2, 0], L[2, 0] = 106.0, 94.0
    # 종목 1: 2일째 시가가 −12% 로 갭 하락 → 그 시가에 손절
    O[2, 1], L[2, 1], H[2, 1], C[2, 1] = 88.0, 87.0, 89.0, 88.0
    # 종목 2: 3일째 고가 +6% → 익절
    H[3, 2] = 106.0
    P = {"O": O, "H": H, "L": L, "C": C, "Cf": C, "U": np.ones((T, N), bool), "dates": pd.bdate_range("2020-01-01", periods=T), "breaks": None}
    PA = CL.Paths(P, [0, 0, 0], [0, 1, 2])
    net, kind, days = PA.exit(0.05, 0.05, 5, detail=True)
    cb, cs = CL.COMM + CL.SLIP, CL.COMM + CL.SLIP + CL.TAX
    f = lambda px: px * (1 - cs) / (1 + cb) - 1                                    # noqa: E731
    assert list(kind) == [0, 0, 1] and abs(net[0] - f(0.95)) < 1e-12 and abs(net[1] - f(0.88)) < 1e-12 and abs(net[2] - f(1.05)) < 1e-12
    n2, k2, d2 = PA.exit(0.30, 0.30, 5, detail=True)
    assert list(k2) == [2, 2, 2] and list(d2) == [5, 5, 5]                         # 안 닿으면 5일째 종가
    S = np.zeros((T, N), bool)
    S[[3, 5, 12], 0] = True
    ts, js = CL.entries(P, S, "2020-01-01", None, 5)
    assert list(ts) == [3, 12]                                                     # 5일 동안 다시 안 삼
    rt, rj, src = CL.random_entries(P, ts, js, 4, seed=1)
    assert (rt == np.repeat(ts, 4)).all() and (rj != 0).all() and list(np.bincount(src)) == [4, 4]
    sig = CL.setup_signals({**P, "V": np.ones((T, N)), "MC": np.ones((T, N))})
    assert set(sig) == set(CL.SETUPS) and all(v.shape == (T, N) for v in sig.values())


def test_edge_lab_features_calendar_and_model():
    """새 자리 실험실: 재료 36개 · 그날 순위 0~1 · 옵션 만기(둘째 목요일)·월말월초 · 달별 t · 로지스틱이 방향을 찾음 · 날마다 상위 k."""
    import edge_lab as EL
    rng = np.random.default_rng(4)
    T, N = 320, 12
    dates = pd.bdate_range("2021-01-04", periods=T)
    C = 10000 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    O = C * np.exp(rng.normal(0, 0.005, (T, N)))
    H, L = np.maximum(O, C) * 1.01, np.minimum(O, C) * 0.99
    V = rng.lognormal(12, 0.5, (T, N))
    P = {"U": np.ones((T, N), bool), "C": C, "Cf": C, "O": O, "H": H, "L": L, "V": V, "VAL": V * C, "MC": C * 1e7,
         "FOR": rng.normal(0, 1e8, (T, N)), "INS": rng.normal(0, 1e8, (T, N)), "IND": rng.normal(0, 1e8, (T, N)), "dates": dates}
    F = EL.features(P, C.mean(1))
    assert set(F) == set(EL.FEAT)
    for k in set(EL.FEAT) - set(EL.MARKET) - set(EL.BINARY):
        x = F[k][~np.isnan(F[k])]
        assert len(x) and x.min() > 0 and x.max() <= 1, k                           # 그날 순위
    exp_days = dates[F["expiry"][:, 0] > 0.5]
    assert len(exp_days) >= 12 and all(d.weekday() == 3 and 8 <= d.day <= 14 for d in exp_days)
    assert all(d.weekday() == 4 for d in dates[F["friday"][:, 0] > 0.5])
    tom = dates[F["tom"][:, 0] > 0.5]
    assert pd.Timestamp("2021-01-04") in tom and pd.Timestamp("2021-01-29") in tom and pd.Timestamp("2021-01-15") not in tom
    mo = np.repeat(np.arange(24), 10)
    assert EL._tm(np.ones(240) + rng.normal(0, 0.1, 240), mo) > 10 and EL._tm(np.ones(30), np.repeat(np.arange(3), 10)) is None
    X = rng.normal(size=(4000, 3))
    y = (X[:, 0] * 2 - X[:, 2] + rng.normal(size=4000) > 0).astype(float)
    m = EL.fit_logit(X, y)
    assert m["w"][0] > 0.5 and m["w"][2] < -0.2 and abs(m["w"][1]) < 0.2
    class E:                                                                        # noqa: E306
        pass
    E.P, E.rows = P, np.nonzero(np.ones((T, N), bool))
    tm = EL.top_mask(E, rng.random(T * N), 3)
    assert (tm.sum(1) == 3).all()


def test_exit_lab_partial_breakeven_trailing_gap():
    """매도 규칙: 1차 익절 절반 → 본전 손절 · 고점 대비 따라 올라가는 손절 · 갭 하락은 시가 · 안 닿으면 기한 날 종가."""
    import exit_lab as XL
    H = XL.HMAX

    class PA:                                                                       # noqa: E306
        n = 4
        o = np.full((4, H), 1.002)                                                  # 기본 가격은 본전(1.0)보다 조금 위
        h = np.full((4, H), 1.01)
        l = np.full((4, H), 1.001)                                                  # noqa: E741
        c = np.full((4, H), 1.005)
    PA.h[0, 1] = 1.11                                                               # 0: 1일째 +10% 닿음 → 절반 팔고
    PA.l[0, 3] = 0.99                                                               #    3일째 본전(1.0) 깨짐
    PA.c[1, 1], PA.h[1, 1] = 1.2, 1.2                                               # 1: 종가 1.2 까지 → 고점 대비 −10% = 1.08
    PA.o[1, 2], PA.l[1, 2] = 1.15, 1.07
    PA.o[2, 1], PA.l[2, 1] = 0.85, 0.84                                             # 2: 1일째 시가가 손절선(0.9) 아래 → 0.85
    PA.c[3, 19] = 1.03                                                              # 3: 아무것도 안 닿음 → 20일째 종가
    got, xd = XL.simulate(PA, (0.10, 0.10, 0.5, 9.0, 1, 0.0, 20), detail=True)
    assert abs(got[0] - (0.5 * 1.10 + 0.5 * 1.0)) < 1e-12 and xd[0] == 3
    assert abs(got[2] - 0.85) < 1e-12 and xd[2] == 1 and abs(got[3] - 1.03) < 1e-12 and xd[3] == 19
    g2, x2 = XL.simulate(PA, (0.10, 0.50, 1.0, 9.0, 0, 0.10, 20), detail=True)
    assert abs(g2[1] - 1.08) < 1e-12 and x2[1] == 2                                 # 고점 1.2 × 0.9
    g3 = XL.simulate(PA, (0.10, 0.10, 0.5, 9.0, 0, 0.0, 20))
    assert g3[0] > got[0]                                                           # 본전 손절을 안 하면 0번은 끝까지 감
    assert XL.combo_key((0.1, 0.1, 1.0, 0.3, 1, 0.0, 20)) == XL.combo_key((0.1, 0.1, 1.0, 0.5, 0, 0.0, 20))
    assert XL.combo_key((0.1, 0.3, 0.5, 0.3, 1, 0.0, 20)) is None
    assert abs(XL.nets(np.array([1.0]))[0] - ((1 - (CL_cs := 0.000186 + 0.0005 + 0.0020)) / (1 + 0.000186 + 0.0005) - 1)) < 1e-12 and CL_cs > 0


def test_wf_lab_gbm_book_and_discipline():
    """해마다 다시 배우는 실험실: 나무 부스팅이 섞인 효과를 찾음 · 덮개로 쉬는 기간은 현금 0% · 규칙 대 습관 계좌 장부."""
    import wf_lab as W
    import discipline_lab as DL
    rng = np.random.default_rng(1)
    n = 40_000
    X = rng.random((n, 5)).astype(np.float32)
    y = 0.03 * ((X[:, 0] > 0.7) & (X[:, 1] < 0.3)) + rng.normal(0, 0.02, n)
    g = W.fit_gbm(X[:30_000], y[:30_000], n_trees=30, min_leaf=200)
    r = W.fit_ridge(X[:30_000].astype(float), y[:30_000])
    cg = np.corrcoef(W.predict(g, X[30_000:]), y[30_000:])[0, 1]
    cr = np.corrcoef(W.predict(r, X[30_000:]), y[30_000:])[0, 1]
    assert cg > cr and cg > 0.3                                                    # 선형으로는 못 잡는 '둘 다' 조건
    dates = pd.bdate_range("2019-01-01", periods=60)
    D = pd.DataFrame({"date": np.repeat(dates, 4), "y5": np.tile([0.10, 0.0, 0.0, -0.10], 60)})
    p = np.tile([3.0, 2.0, 1.0, 0.0], 60).astype(float)
    p[:40] = np.nan                                                                # 앞 10일은 덮개로 쉼
    bk = W.book(D, p, 5, 1, "2019-01-01", None)
    assert len(bk) == 12 and (bk["s"].iloc[:2] == 0).all()                         # 쉬는 기간도 기간으로 남고 0%
    assert abs(bk["s"].iloc[3] - (1.10 * (1 - W.CS) / (1 + W.CB) - 1)) < 1e-12 and abs(bk["ew"].iloc[3]) < 1e-12
    d0 = pd.Timestamp("2020-01-01")
    rows = [(d0 + pd.Timedelta(days=i), 0) for i in range(4)]
    st = DL.book_slots(rows, [0.1, 0.1, 0.1, 0.1], [d0 + pd.Timedelta(days=10)] * 4, slots=3)
    assert st["trades"] == 3 and abs(st["total"] - 0.1) < 1e-12                    # 자리 3개가 차면 네 번째는 못 산다


def glob_any(d, prefix):
    return os.path.isdir(d) and any(n.startswith(prefix) for n in os.listdir(d))


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
