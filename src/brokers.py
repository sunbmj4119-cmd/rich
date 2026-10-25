"""
브로커 어댑터 — 모의(paper) / 한국투자증권 Open API (모의투자 · 실계좌)

공통 인터페이스
  account()                    -> {"cash": float, "positions": {code: {"shares", "avg"}}}
  order(side, code, qty, kind, price) -> {"ok", "order_no", "msg"}
  is_live                      -> 실제 돈이 움직이는가

한국투자증권 API 스펙 (공식 예제 koreainvestment/open-trading-api 에서 확인, 2026-10)
  토큰   POST /oauth2/tokenP                     (1분에 1회만 재발급 가능)
  주문   POST /uapi/domestic-stock/v1/trading/order-cash
         실전 매수 TTTC0012U · 매도 TTTC0011U / 모의 VTTC0012U · VTTC0011U
         ORD_DVSN 00=지정가 01=시장가 · 수량·단가는 문자열
  잔고   GET  /uapi/domestic-stock/v1/trading/inquire-balance  실전 TTTC8434R / 모의 VTTC8434R
  현재가 GET  /uapi/domestic-stock/v1/quotations/inquire-price FHKST01010100
  도메인 실전 https://openapi.koreainvestment.com:9443 · 모의 https://openapivts.koreainvestment.com:29443

필요한 환경변수 (GitHub Secrets 로 넣는다 — 저장소에 절대 커밋하지 않는다)
  KIS_APP_KEY · KIS_APP_SECRET · KIS_ACCOUNT (8자리) · KIS_ACCOUNT_PRODUCT (기본 01)
"""
import json
import math
import os
import time

PAPER_STATE = "data/autotrade/paper_account.json"


def tick_size(price):
    """KRX 유가증권 호가단위 (2023-01 개편 기준)."""
    for lim, tick in [(2000, 1), (5000, 5), (20000, 10), (50000, 50), (200000, 100), (500000, 500)]:
        if price < lim:
            return tick
    return 1000


def round_tick(price, up=False):
    t = tick_size(price)
    return int((math.ceil if up else math.floor)(price / t) * t)


# ══════════════════════════════════════════════════════════════
class PaperBroker:
    """모의 계좌. 주문은 '대기'로 쌓였다가 다음 실행 때 그날 종가로 체결된다
    (= 백테스트의 't일 판단 → t+1일 종가 체결'과 정확히 같다)."""
    is_live = False
    name = "paper"

    def __init__(self, initial_cash=10_000_000, cost=0.0015, path=PAPER_STATE):
        self.path, self.cost = path, cost
        if os.path.exists(path):
            self.s = json.load(open(path, encoding="utf-8"))
        else:
            self.s = {"cash": float(initial_cash), "positions": {}, "pending": [], "fills": []}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.s, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def account(self):
        return {"cash": self.s["cash"],
                "positions": {c: dict(p) for c, p in self.s["positions"].items()}}

    def order(self, side, code, qty, kind="market", price=None, date=None):
        if qty <= 0:
            return {"ok": False, "order_no": "", "msg": "수량 0"}
        self.s["pending"].append({"side": side, "code": code, "qty": int(qty), "kind": kind,
                                  "limit": price, "date": date})
        return {"ok": True, "order_no": f"P{len(self.s['fills']) + len(self.s['pending'])}",
                "msg": "모의 주문 접수 (다음 거래일 종가 체결)"}

    def settle(self, close_of, fill_date):
        """대기 주문을 fill_date 종가로 체결. 지정가 매수는 종가가 지정가 이하일 때만."""
        done, keep = [], []
        for o in self.s["pending"]:
            if o.get("date") and o["date"] >= fill_date:
                keep.append(o)             # 같은 날 낸 주문은 다음 날 체결
                continue
            px = close_of(o["code"])
            if px is None:
                keep.append(o)
                continue
            pos = self.s["positions"]
            if o["side"] == "buy":
                if o["kind"] == "limit" and o.get("limit") and px > o["limit"]:
                    done.append({**o, "status": "미체결(지정가 초과)", "fill": None})
                    continue
                qty = o["qty"]
                need = qty * px * (1 + self.cost)
                if need > self.s["cash"]:
                    qty = int(self.s["cash"] // (px * (1 + self.cost)))
                    need = qty * px * (1 + self.cost)
                if qty <= 0:
                    done.append({**o, "status": "미체결(현금부족)", "fill": None})
                    continue
                self.s["cash"] -= need
                p = pos.get(o["code"], {"shares": 0, "avg": 0.0})
                tot = p["shares"] + qty
                p["avg"] = (p["shares"] * p["avg"] + qty * px * (1 + self.cost)) / tot
                p["shares"] = tot
                pos[o["code"]] = p
                done.append({**o, "qty": qty, "status": "체결", "fill": px, "fill_date": fill_date})
            else:
                p = pos.get(o["code"])
                if not p:
                    done.append({**o, "status": "보유없음", "fill": None})
                    continue
                qty = min(o["qty"], p["shares"])
                self.s["cash"] += qty * px * (1 - self.cost)
                p["shares"] -= qty
                if p["shares"] <= 0:
                    pos.pop(o["code"])
                done.append({**o, "qty": qty, "status": "체결", "fill": px, "fill_date": fill_date,
                             "pnl": qty * (px * (1 - self.cost) - p["avg"])})
        self.s["pending"] = keep
        self.s["fills"].extend(done)
        self.s["fills"] = self.s["fills"][-500:]
        return done


# ══════════════════════════════════════════════════════════════
class KISBroker:
    """한국투자증권 Open API. demo=True 면 모의투자 서버."""
    name = "kis"

    def __init__(self, demo=True, live_orders=False):
        import requests                       # 실계좌 모드에서만 필요
        self.rq = requests
        self.demo = demo
        self.is_live = (not demo) and live_orders
        self.live_orders = live_orders
        self.base = ("https://openapivts.koreainvestment.com:29443" if demo
                     else "https://openapi.koreainvestment.com:9443")
        self.key = os.environ.get("KIS_APP_KEY", "")
        self.secret = os.environ.get("KIS_APP_SECRET", "")
        self.cano = os.environ.get("KIS_ACCOUNT", "")
        self.prdt = os.environ.get("KIS_ACCOUNT_PRODUCT", "01")
        if not (self.key and self.secret and self.cano):
            raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT 환경변수가 없습니다.")
        self._token = None

    def _tr(self, real, demo):
        return demo if self.demo else real

    def token(self):
        if self._token:
            return self._token
        r = self.rq.post(f"{self.base}/oauth2/tokenP", timeout=15,
                         json={"grant_type": "client_credentials",
                               "appkey": self.key, "appsecret": self.secret})
        r.raise_for_status()
        self._token = r.json()["access_token"]
        return self._token

    def _headers(self, tr_id):
        return {"content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {self.token()}",
                "appkey": self.key, "appsecret": self.secret,
                "tr_id": tr_id, "custtype": "P", "tr_cont": ""}

    def _get(self, path, tr_id, params):
        for k in range(3):
            r = self.rq.get(f"{self.base}{path}", headers=self._headers(tr_id),
                            params=params, timeout=15)
            j = r.json()
            if j.get("msg_cd") == "EGW00201":       # 초당 호출 제한 — 잠시 쉬고 재시도
                time.sleep(1.0 + k)
                continue
            return j
        return j

    def price(self, code):
        j = self._get("/uapi/domestic-stock/v1/quotations/inquire-price", "FHKST01010100",
                      {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})
        try:
            return float(j["output"]["stck_prpr"])
        except (KeyError, TypeError, ValueError):
            return None

    def account(self):
        params = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "AFHR_FLPR_YN": "N",
                  "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01",
                  "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00",
                  "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        j = self._get("/uapi/domestic-stock/v1/trading/inquire-balance",
                      self._tr("TTTC8434R", "VTTC8434R"), params)
        if j.get("rt_cd") != "0":
            raise RuntimeError(f"잔고 조회 실패: {j.get('msg_cd')} {j.get('msg1')}")
        pos = {}
        for r in j.get("output1", []):
            q = int(float(r.get("hldg_qty", 0) or 0))
            if q > 0:
                pos[r["pdno"]] = {"shares": q, "avg": float(r.get("pchs_avg_pric", 0) or 0)}
        o2 = (j.get("output2") or [{}])[0]
        # 주문가능 현금 근사: D+2 예수금 (매도대금 정산 반영)
        cash = float(o2.get("prvs_rcdl_excc_amt") or o2.get("dnca_tot_amt") or 0)
        return {"cash": cash, "positions": pos}

    def order(self, side, code, qty, kind="market", price=None, date=None):
        if qty <= 0:
            return {"ok": False, "order_no": "", "msg": "수량 0"}
        body = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "PDNO": code,
                "ORD_DVSN": "01" if kind == "market" else "00",
                "ORD_QTY": str(int(qty)),
                "ORD_UNPR": "0" if kind == "market" else str(int(price)),
                "EXCG_ID_DVSN_CD": "KRX", "SLL_TYPE": "01" if side == "sell" else "",
                "CNDT_PRIC": ""}
        if not self.demo and not self.live_orders:
            return {"ok": False, "order_no": "", "msg": "DRY-RUN (live_trading=false) — 전송 안 함",
                    "body": body}
        tr = (self._tr("TTTC0012U", "VTTC0012U") if side == "buy"
              else self._tr("TTTC0011U", "VTTC0011U"))
        r = self.rq.post(f"{self.base}/uapi/domestic-stock/v1/trading/order-cash",
                         headers=self._headers(tr), data=json.dumps(body), timeout=15)
        j = r.json()
        ok = j.get("rt_cd") == "0"
        return {"ok": ok, "order_no": (j.get("output") or {}).get("ODNO", ""),
                "msg": f"{j.get('msg_cd', '')} {j.get('msg1', '')}".strip()}


def make_broker(cfg):
    b = cfg.get("broker", "paper")
    if b == "paper":
        pc = cfg.get("paper", {})
        return PaperBroker(pc.get("initial_cash", 10_000_000), pc.get("cost", 0.0015))
    if b == "kis_demo":
        return KISBroker(demo=True, live_orders=True)
    if b == "kis_real":
        live = bool(cfg.get("live_trading")) and os.environ.get("AUTOTRADE_CONFIRM") == "I_UNDERSTAND"
        return KISBroker(demo=False, live_orders=live)
    raise ValueError(f"알 수 없는 broker: {b}")
