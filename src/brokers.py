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
import re
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
    q = round(price / t, 6)                    # 11200.000000000002 같은 부동소수 잡음이 한 칸 올림을 만들지 않게
    return int((math.ceil if up else math.floor)(q) * t)


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

    def __init__(self, demo=True, live_orders=False, account=None, product=None):
        import requests                       # 실계좌 모드에서만 필요
        self.rq = requests
        self.demo = demo
        self.is_live = (not demo) and live_orders
        self.live_orders = live_orders
        self.base = ("https://openapivts.koreainvestment.com:29443" if demo
                     else "https://openapi.koreainvestment.com:9443")
        self.key = os.environ.get("KIS_APP_KEY", "")
        self.secret = os.environ.get("KIS_APP_SECRET", "")
        self.cano = account or os.environ.get("KIS_ACCOUNT", "")
        self.prdt = product or os.environ.get("KIS_ACCOUNT_PRODUCT", "01")
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


    def daily_fills(self, start, end):
        """주식일별주문체결조회 (3개월 이내) — 체결된 주문만. 반환 [{order_no, code, side, qty, filled, avg}].
        TR: 실전 TTTC8001R / 모의 VTTC8001R (공식 예제 inquire_daily_ccld 기준 — 모의투자에서 처음 돌릴 때 응답 필드 확인)."""
        params = {"CANO": self.cano, "ACNT_PRDT_CD": self.prdt, "INQR_STRT_DT": start.replace("-", ""),
                  "INQR_END_DT": end.replace("-", ""), "SLL_BUY_DVSN_CD": "00", "INQR_DVSN": "00", "PDNO": "",
                  "CCLD_DVSN": "01", "ORD_GNO_BRNO": "", "ODNO": "", "INQR_DVSN_3": "00", "INQR_DVSN_1": "",
                  "EXCG_ID_DVSN_CD": "KRX", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        j = self._get("/uapi/domestic-stock/v1/trading/inquire-daily-ccld", self._tr("TTTC8001R", "VTTC8001R"), params)
        if j.get("rt_cd") != "0":
            raise RuntimeError(f"체결 조회 실패: {j.get('msg_cd')} {j.get('msg1')}")
        out = []
        for r in j.get("output1") or []:
            q = int(float(r.get("tot_ccld_qty", 0) or 0))
            out.append({"order_no": r.get("odno", ""), "code": r.get("pdno", ""), "date": r.get("ord_dt", ""),
                        "side": "sell" if r.get("sll_buy_dvsn_cd") == "01" else "buy", "qty": int(float(r.get("ord_qty", 0) or 0)),
                        "filled": q, "avg": float(r.get("avg_prvs", 0) or 0)})
        return out


# ══════════════════════════════════════════════════════════════
class TossBroker:
    """토스증권 오픈API (https://openapi.tossinvest.com) — 미리 등록한 IP(오라클 서버)에서만 된다. 모의투자 서버는 없다.
    명세 1.2.28 (data/toss/api_detail.json) 기준으로 확인한 형식:
      헤더  X-Tossinvest-Account: accountSeq (GET /api/v1/accounts 의 accountSeq, 정수)
      주문  POST /api/v1/orders  {symbol, side BUY|SELL, orderType LIMIT|MARKET, timeInForce DAY|OPG(국내 시가단일가)|CLS(미국),
                                 quantity "정수 문자열", price "호가 단위 정수 문자열"(LIMIT 만), clientOrderId(멱등성 키 10분)}
            1억원 이상은 confirmHighValueOrder 가 없으면 거절 — 착오 주문 방지로 그대로 둔다
      보유  GET /api/v1/holdings → result.items[] {symbol, marketCountry KR|US, quantity, averagePurchasePrice, …}
      현금  GET /api/v1/buying-power?currency=KRW → result.cashBuyingPower
      체결  GET /api/v1/orders?status=CLOSED&from&to (cursor) → orders[].execution {filledQuantity, averageFilledPrice, …}
      취소  POST /api/v1/orders/{orderId}/cancel
    토큰은 클라이언트당 하나만 유효 → 서버의 점검·수집기와 같은 토큰 파일(~/.rich/toss_token.json)을 같이 쓴다.
    주문은 VERIFIED(형식을 확인한 명세 버전)와 서버가 받은 명세 버전이 같을 때만 — 토스가 명세를 바꾸면 다시 확인할 때까지 멈춘다."""
    name = "toss"
    VERIFIED = "1.2.28"

    def __init__(self, live_orders=False, env=None, session=None):
        import requests
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "oracle"))
        import toss_probe as TP
        self.TP = TP
        self.rq = session or requests
        self.env = env if env is not None else {**TP.load_env(), **{k: v for k, v in os.environ.items() if k.startswith("TOSS_")}}
        if not (self.env.get("TOSS_CLIENT_ID") and self.env.get("TOSS_CLIENT_SECRET")):
            raise RuntimeError("TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 가 없습니다 (서버 ~/.rich/secrets.env — setup.sh --keys)")
        self.base = TP.BASE
        self.live_orders = live_orders
        self.is_live = bool(live_orders)
        self._tok = None
        v = self.env.get("TOSS_ACCOUNT_SEQ")
        self.seq = int(v) if v not in (None, "") else None

    def spec_version(self):
        try:
            return json.load(open("data/toss/api_detail.json", encoding="utf-8")).get("version")
        except (OSError, ValueError):
            return None

    def verified(self):
        return bool(self.VERIFIED) and self.spec_version() == self.VERIFIED

    def token(self, fresh=False):
        if fresh:
            try:
                os.remove(self.TP.TOKEN)
            except OSError:
                pass
            self._tok = None
        if not self._tok:
            self._tok, info = self.TP.get_token(self.env)
            if not self._tok:
                raise RuntimeError(f"토스 토큰 실패: {info}")
        return self._tok

    def _req(self, method, path, params=None, body=None, account=True):
        st, j = 0, {}
        for i in range(4):
            h = {"Authorization": f"Bearer {self.token()}", "Accept": "application/json"}
            if account:
                h["X-Tossinvest-Account"] = str(self.account_seq())
            r = self.rq.request(method, self.base + path, params=params, json=body, headers=h, timeout=20)
            st = r.status_code
            if st == 401 and i == 0:
                self.token(fresh=True)
                continue
            if st in (429, 500, 502, 503, 504) and method == "GET":        # 주문(POST)은 다시 보내지 않는다 — 멱등성 키로만
                time.sleep(1.5 * (i + 1))
                continue
            try:
                j = r.json()
            except ValueError:
                j = {"raw": r.text[:300]}
            return st, j
        return st, j

    def account_seq(self):
        if self.seq is None:
            st, j = self._req("GET", "/api/v1/accounts", account=False)
            items = (j or {}).get("result") or []
            want = str(self.env.get("TOSS_ACCOUNT") or "")
            pick = next((x for x in items if str(x.get("accountNo")) == want), None) or \
                next((x for x in items if x.get("accountType") == "BROKERAGE"), None)
            if not pick or pick.get("accountSeq") is None:
                raise RuntimeError(f"토스 계좌를 찾지 못함 (상태 {st})")
            self.seq = int(pick["accountSeq"])
        return self.seq

    @staticmethod
    def _num(v):
        if isinstance(v, dict):
            v = v.get("amount", v.get("krw", v.get("value")))
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _err(j):
        e = (j or {}).get("error") or {}
        return f"{e.get('code', '')} {e.get('message', '')}".strip() or str(j)[:200]

    def account(self, market="KR"):
        st, j = self._req("GET", "/api/v1/holdings")
        if st != 200:
            raise RuntimeError(f"보유 조회 실패 {st}: {self._err(j)}")
        pos = {}
        for it in ((j or {}).get("result") or {}).get("items") or []:
            if market and it.get("marketCountry") not in (None, market):
                continue
            q = self._num(it.get("quantity"))
            if not it.get("symbol") or q is None:
                raise RuntimeError(f"보유 항목 형식이 명세와 다름 — 키 {sorted(it.keys())}")
            if q > 0:
                pos[str(it["symbol"])] = {"shares": q, "avg": self._num(it.get("averagePurchasePrice")) or 0.0}
        st, j = self._req("GET", "/api/v1/buying-power", {"currency": "KRW"})
        if st != 200:
            raise RuntimeError(f"매수 가능 금액 조회 실패 {st}: {self._err(j)}")
        cash = self._num(((j or {}).get("result") or {}).get("cashBuyingPower"))
        return {"cash": cash or 0.0, "positions": pos}

    def price(self, code):
        st, j = self._req("GET", "/api/v1/prices", {"symbols": code})
        res = (j or {}).get("result")
        it = res[0] if isinstance(res, list) and res else {}
        return self._num(it.get("lastPrice"))

    def order(self, side, code, qty, kind="market", price=None, date=None, tif="DAY", client_id=None):
        if int(qty) <= 0 or int(qty) != qty:
            return {"ok": False, "order_no": "", "msg": "수량은 1 이상 정수만"}
        if not self.verified():
            return {"ok": False, "order_no": "", "msg": f"토스 명세가 바뀜(서버 {self.spec_version()}, 확인 {self.VERIFIED}) — 다시 확인할 때까지 보내지 않음"}
        body = {"symbol": str(code), "side": "BUY" if side == "buy" else "SELL",
                "orderType": "MARKET" if kind == "market" else "LIMIT", "timeInForce": tif, "quantity": str(int(qty))}
        if kind != "market":
            body["price"] = str(int(price))
        if client_id:
            body["clientOrderId"] = re.sub(r"[^A-Za-z0-9_-]", "", client_id)[:36]
        if not self.live_orders:
            return {"ok": False, "order_no": "", "msg": "DRY-RUN (실제 주문 잠김) — 보내지 않음", "body": body}
        st, j = self._req("POST", "/api/v1/orders", body=body)
        ok = st == 200 and bool(((j or {}).get("result") or {}).get("orderId"))
        return {"ok": ok, "order_no": ((j or {}).get("result") or {}).get("orderId", ""), "msg": "접수" if ok else f"{st} {self._err(j)}"}

    def order_status(self, order_id):
        st, j = self._req("GET", f"/api/v1/orders/{order_id}")
        return (j or {}).get("result") if st == 200 else None

    def cancel(self, order_id):
        st, j = self._req("POST", f"/api/v1/orders/{order_id}/cancel", body={})
        return {"ok": st == 200, "msg": "취소" if st == 200 else f"{st} {self._err(j)}"}

    def daily_fills(self, start, end):
        out, cursor = [], None
        for _ in range(20):
            params = {"status": "CLOSED", "from": start, "to": end, "limit": 100, **({"cursor": cursor} if cursor else {})}
            st, j = self._req("GET", "/api/v1/orders", params)
            if st != 200:
                raise RuntimeError(f"주문 목록 조회 실패 {st}: {self._err(j)}")
            res = (j or {}).get("result") or {}
            for o in res.get("orders") or []:
                ex = o.get("execution") or {}
                f = self._num(ex.get("filledQuantity")) or 0
                out.append({"order_no": o.get("orderId", ""), "code": o.get("symbol", ""), "date": (o.get("orderedAt") or "")[:10],
                            "side": (o.get("side") or "").lower(), "qty": self._num(o.get("quantity")) or 0, "filled": f,
                            "avg": self._num(ex.get("averageFilledPrice")) or 0.0, "status": o.get("status")})
            if not res.get("hasNext"):
                break
            cursor = res.get("nextCursor")
        return out


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
