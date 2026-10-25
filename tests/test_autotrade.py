"""
자동매매 회귀 테스트 — 실제 데이터 위에서 판단·계획·모의체결 경로를 점검한다.
사용: python tests/test_autotrade.py   (pytest 로도 돈다)
"""
import copy
import os
import sys
import tempfile

import yaml

sys.path.insert(0, "src")
import features as F                                    # noqa: E402
from decision_engine import decide, BUY, SELL, HOLD, WATCH   # noqa: E402
from brokers import PaperBroker, round_tick, tick_size  # noqa: E402
import autotrader as A                                  # noqa: E402

CFG = yaml.safe_load(open("config/autotrade.yaml", encoding="utf-8"))
PANEL, FEATS = F.build_all()
T = len(PANEL["dates"]) - 1


def open_cfg():
    c = copy.deepcopy(CFG)
    c["strategy"]["crash_gate"] = False
    c["risk"]["stale_flow_policy"] = "ignore"
    return c


def top(n):
    r = FEATS["rank"].iloc[T].sort_values()
    return list(r.index[:n])


def test_tick():
    assert tick_size(1500) == 1 and tick_size(45_000) == 50 and tick_size(600_000) == 1000
    assert round_tick(45_123) == 45_100 and round_tick(45_123, up=True) == 45_150


def test_sell_rules():
    c1, c2, c3 = top(3)
    px = PANEL["close"].iloc[T]
    pos = {
        c1: {"shares": 10, "avg": px[c1] / 0.85, "entry_t": T - 5, "peak": px[c1] / 0.85, "partial_done": False},
        c2: {"shares": 10, "avg": px[c2] / 1.30, "entry_t": T - 5, "peak": px[c2], "partial_done": False},
        c3: {"shares": 10, "avg": px[c3], "entry_t": T - 5, "peak": px[c3], "partial_done": False},
    }
    dec, _ = decide(T, PANEL, FEATS, pos, CFG, flags={"flow_stale": False})
    by = {d["code"]: d for d in dec}
    assert by[c1]["state"] == SELL and "손절" in by[c1]["reason"]
    assert by[c2]["state"] == SELL and by[c2]["action"] == "부분익절" and abs(by[c2]["fraction"] - 1/3) < 1e-3
    assert by[c3]["state"] == HOLD
    # 부분익절을 이미 했으면 다시 권하지 않는다
    pos[c2]["partial_done"] = True
    dec, _ = decide(T, PANEL, FEATS, pos, CFG, flags={"flow_stale": False})
    assert {d["code"]: d for d in dec}[c2]["state"] == HOLD


def test_gate_blocks_buys():
    cfg = copy.deepcopy(CFG)
    cfg["risk"]["stale_flow_policy"] = "ignore"
    dec, gate = decide(T, PANEL, FEATS, {}, cfg, flags={"flow_stale": True})
    if gate["regime"] == "급락장" and not gate["open"]:
        assert not any(d["state"] == BUY for d in dec)
        assert all(d["state"] == WATCH for d in dec)


def test_buys_fill_free_slots_only():
    cfg = open_cfg()
    held = top(1)[0]
    pos = {held: {"shares": 1, "avg": float(PANEL["close"].iloc[T][held]), "entry_t": T - 1,
                  "peak": float(PANEL["close"].iloc[T][held]), "partial_done": False}}
    dec, _ = decide(T, PANEL, FEATS, pos, cfg, flags={"flow_stale": True})
    buys = [d for d in dec if d["state"] == BUY]
    assert len(buys) <= cfg["strategy"]["max_positions"] - 1
    assert all(d["rank"] <= cfg["strategy"]["entry_rank"] for d in buys)
    assert held not in [d["code"] for d in buys]
    # 관망 종목엔 '무엇이 바뀌면 매수인지'가 붙는다
    assert all(d.get("need") for d in dec if d["state"] == WATCH)


def test_plan_limits():
    cfg = open_cfg()
    cfg["risk"]["max_order_krw"] = 500_000
    dec, _ = decide(T, PANEL, FEATS, {}, cfg, flags={"flow_stale": True})
    acct = {"cash": 10_000_000, "positions": {}}
    plan = A.build_plan(dec, acct, 10_000_000, cfg, PANEL["dates"][T])
    for o in plan["orders"]:
        assert o["side"] == "buy" and isinstance(o["qty"], int) and o["qty"] > 0
        assert o["qty"] * o["price"] <= 500_000
        assert o["price"] % tick_size(o["price"]) == 0


def test_paper_settle():
    with tempfile.TemporaryDirectory() as d:
        b = PaperBroker(1_000_000, 0.0015, path=os.path.join(d, "p.json"))
        b.order("buy", "005930", 10, "limit", 60_000, date="2026-01-02")
        f = b.settle(lambda c: 50_000, "2026-01-02")             # 같은 날 → 체결 안 됨
        assert f == [] and len(b.s["pending"]) == 1
        f = b.settle(lambda c: 50_000, "2026-01-05")
        assert f[0]["status"] == "체결" and b.s["positions"]["005930"]["shares"] == 10
        assert abs(b.s["cash"] - (1_000_000 - 10 * 50_000 * 1.0015)) < 1
        b.order("sell", "005930", 4, "market", None, date="2026-01-05")
        b.settle(lambda c: 55_000, "2026-01-06")
        assert b.s["positions"]["005930"]["shares"] == 6
        b.order("buy", "000660", 1, "limit", 100, date="2026-01-06")
        f = b.settle(lambda c: 200, "2026-01-07")                # 지정가 초과 → 미체결
        assert f[0]["status"].startswith("미체결")


def test_reconcile_partial():
    led = {"005930": {"entry_date": "2026-01-02", "partial_done": False, "last_shares": 9}}
    new = A.reconcile(led, {"cash": 0, "positions": {"005930": {"shares": 6, "avg": 1},
                                                      "000660": {"shares": 3, "avg": 1}}}, "2026-02-01")
    assert new["005930"]["partial_done"] is True and new["005930"]["entry_date"] == "2026-01-02"
    assert new["000660"]["entry_date"] == "2026-02-01"


def test_real_account_needs_three_keys():
    """실계좌는 broker:kis_real + live_trading:true + AUTOTRADE_CONFIRM 셋 다 있어야 전송된다."""
    from brokers import make_broker
    env = {"KIS_APP_KEY": "x", "KIS_APP_SECRET": "y", "KIS_ACCOUNT": "12345678"}
    old = {k: os.environ.get(k) for k in [*env, "AUTOTRADE_CONFIRM"]}
    os.environ.update(env)
    os.environ.pop("AUTOTRADE_CONFIRM", None)
    try:
        b = make_broker({"broker": "kis_real", "live_trading": True})
        assert not b.is_live
        r = b.order("buy", "005930", 1, "limit", 50_000)       # 네트워크 호출 없이 차단돼야 한다
        assert not r["ok"] and "DRY-RUN" in r["msg"]
        os.environ["AUTOTRADE_CONFIRM"] = "I_UNDERSTAND"
        assert not make_broker({"broker": "kis_real", "live_trading": False}).is_live
        assert make_broker({"broker": "kis_real", "live_trading": True}).is_live
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_pause_blocks_buys_not_sells():
    cfg = open_cfg()
    c1 = top(1)[0]
    px = PANEL["close"].iloc[T]
    pos = {c1: {"shares": 10, "avg": px[c1] / 0.85, "entry_t": T - 5, "peak": px[c1] / 0.85, "partial_done": False}}
    dec, _ = decide(T, PANEL, FEATS, pos, cfg, flags={"flow_stale": True, "pause_entries": True})
    assert not any(d["state"] == BUY for d in dec)
    assert {d["code"]: d for d in dec}[c1]["state"] == SELL          # 손절은 일시정지와 무관
    w = [d for d in dec if d["state"] == WATCH and d["rank"] <= 10]
    assert w and all(any("일시정지" in n for n in d["need"]) for d in w)


def test_cooldown_rule():
    cfg = open_cfg()
    cfg["strategy"]["cooldown_days"] = 10
    c1 = top(1)[0]
    dec, _ = decide(T, PANEL, FEATS, {}, cfg, flags={"flow_stale": True, "recent_exits": {c1: T - 3}})
    d = {x["code"]: x for x in dec}[c1]
    assert d["state"] == WATCH and any("재진입" in n or "청산 후" in n for n in d["need"])
    cfg["strategy"]["cooldown_days"] = 0
    dec, _ = decide(T, PANEL, FEATS, {}, cfg, flags={"flow_stale": True, "recent_exits": {c1: T - 3}})
    assert {x["code"]: x for x in dec}[c1]["state"] == BUY


def test_manual_approval_holds_orders(tmp=None):
    """승인 모드: 승인 없이 execute 하면 아무 주문도 나가지 않는다."""
    import json
    with tempfile.TemporaryDirectory() as d:
        old = (A.PLAN, A.ORDERS)
        A.PLAN, A.ORDERS = os.path.join(d, "plan.json"), os.path.join(d, "orders.csv")
        json.dump({"date": "2026-10-23", "orders": [{"side": "buy", "code": "005930", "name": "삼성전자",
                                                      "qty": 1, "kind": "limit", "price": 1000}]},
                  open(A.PLAN, "w"))
        try:
            cfg = copy.deepcopy(CFG)
            cfg["execution"]["approval"] = "manual"
            A.phase_execute(cfg, dry=False, approve=False)
            assert not os.path.exists(A.ORDERS)
        finally:
            A.PLAN, A.ORDERS = old


def test_notify_disabled_is_silent():
    import notify
    assert notify.send({"alerts": {"enabled": False}}, "sell", "x") == []
    assert notify.send({"alerts": {"enabled": True, "on": ["buy"]}}, "sell", "x") == []
    # 키가 없으면 채널이 있어도 보내지 않는다 (네트워크 호출 없음)
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "NTFY_TOPIC"):
        os.environ.pop(k, None)
    assert notify.send({"alerts": {"enabled": True, "on": ["sell"], "channels": ["telegram", "ntfy"]}},
                       "sell", "x") == []


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
