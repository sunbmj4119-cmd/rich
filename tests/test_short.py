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


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
