"""
토스증권 오픈API 연결 점검 (서버에서 실행) — 돈이 움직이는 기능은 쓰지 않는다.

  1) 공식 OpenAPI 명세를 받아 '주소 · 방식 · 입력 이름 · 출력 이름'만 추린다 → data/toss/api_summary.json
     + 주문·잔고·체결 API 는 끝까지 풀어서 → data/toss/api_detail.json (주문 요청 형식을 정확히 맞추려고)
  2) 키로 토큰을 발급받고(POST /oauth2/token) 계좌 목록만 조회한다(GET) → 상태 코드·개수만 data/toss/probe.json
  3) 계좌 식별값을 찾으면 서버의 ~/.rich/secrets.env 에만 TOSS_ACCOUNT 로 저장한다 (비어 있을 때만) · 주문용 accountSeq 는 TOSS_ACCOUNT_SEQ
  4) 읽기 전용 점검: 보유·매수 가능 금액·주문 목록·장 운영·현재가·수수료 — 상태 코드와 응답의 '모양'(키 이름)만 남긴다

저장소가 공개이므로 GitHub 에 올리는 파일에는 잔고·보유 종목·키·토큰을 넣지 않고, 숫자 6자리 이상은 끝 4자리만 남긴다.
토큰은 클라이언트당 하나만 유효하다(새로 받으면 이전 것이 무효) → ~/.rich/toss_token.json(600)에 저장해 다시 쓴다.

사용: bash ~/rich/deploy/oracle/setup.sh --probe
"""
import json
import os
import re
import sys
import time

import requests

BASE = "https://openapi.tossinvest.com"
SPEC_URLS = [BASE + "/openapi-docs/latest/openapi.json",
             "https://developers.tossinvest.com/openapi.json",
             "https://developers.tossinvest.com/openapi-docs/latest/openapi.json"]
HOME = os.path.expanduser("~")
STATE = os.path.join(HOME, ".rich")
SECRETS = os.path.join(STATE, "secrets.env")
TOKEN = os.path.join(STATE, "toss_token.json")
REPO = os.path.join(HOME, "rich")
OUT = os.path.join(REPO, "data", "toss")


def load_env():
    env = {}
    if os.path.exists(SECRETS):
        for line in open(SECRETS, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def mask(x):
    """숫자 6자리 이상은 끝 4자리만 남긴다."""
    return re.sub(r"\d{6,}", lambda m: "***" + m.group(0)[-4:], str(x))


def resolve(spec, node, depth=0):
    if depth > 6 or not isinstance(node, dict):
        return node
    if "$ref" in node:
        ref = node["$ref"].lstrip("#/").split("/")
        cur = spec
        for r in ref:
            cur = cur.get(r, {}) if isinstance(cur, dict) else {}
        return resolve(spec, cur, depth + 1)
    return node


def fields(spec, schema, depth=0):
    """스키마의 속성 이름(한 단계 아래까지)."""
    s = resolve(spec, schema or {})
    if not isinstance(s, dict) or depth > 2:
        return None
    if s.get("type") == "array" or "items" in s:
        return {"[]": fields(spec, s.get("items"), depth + 1)}
    props = s.get("properties") or {}
    if not props and "allOf" in s:
        props = {}
        for part in s["allOf"]:
            p = resolve(spec, part)
            props.update((p or {}).get("properties") or {})
    out = {}
    for k, v in props.items():
        v2 = resolve(spec, v)
        sub = fields(spec, v2, depth + 1) if isinstance(v2, dict) and (v2.get("properties") or v2.get("items")) else None
        out[k] = {"type": (v2 or {}).get("type"), "desc": ((v2 or {}).get("description") or "")[:80],
                  **({"enum": v2["enum"][:12]} if isinstance(v2, dict) and v2.get("enum") else {}),
                  **({"fields": sub} if sub else {})}
    return {"required": s.get("required", []), "props": out}


DETAIL_OPS = [("POST", "/api/v1/orders"), ("GET", "/api/v1/orders"), ("GET", "/api/v1/orders/{orderId}"),
              ("POST", "/api/v1/orders/{orderId}/cancel"), ("POST", "/api/v1/orders/{orderId}/modify"),
              ("GET", "/api/v1/holdings"), ("GET", "/api/v1/buying-power"), ("GET", "/api/v1/sellable-quantity"),
              ("GET", "/api/v1/prices"), ("GET", "/api/v1/price-limits"), ("GET", "/api/v1/market-calendar/KR"),
              ("GET", "/api/v1/market-calendar/US"), ("GET", "/api/v1/commissions"), ("GET", "/api/v1/accounts"),
              ("GET", "/api/v1/stocks/{symbol}/warnings"), ("GET", "/api/v1/exchange-rate")]


def deep(spec, node, depth=0):
    """스키마를 끝까지 풀어서 (oneOf·anyOf·allOf·enum·예시 포함) — 주문 요청 형식을 정확히 맞추려고."""
    node = resolve(spec, node)
    if not isinstance(node, dict):
        return node
    if depth > 8:
        return {"...": "깊이 제한"}
    out = {}
    for k in ("type", "format", "enum", "required", "example", "examples", "default", "minimum", "maximum",
              "pattern", "nullable", "const", "minLength", "maxLength"):
        if k in node:
            out[k] = node[k]
    if node.get("description"):
        out["description"] = node["description"][:600]
    if isinstance(node.get("discriminator"), dict):
        out["discriminator"] = {"propertyName": node["discriminator"].get("propertyName"),
                                "mapping": list((node["discriminator"].get("mapping") or {}).keys())}
    for comb in ("oneOf", "anyOf", "allOf"):
        if comb in node:
            out[comb] = [deep(spec, x, depth + 1) for x in node[comb]]
    if "properties" in node:
        out["properties"] = {k: deep(spec, v, depth + 1) for k, v in node["properties"].items()}
    if "items" in node:
        out["items"] = deep(spec, node["items"], depth + 1)
    return out


def detail(spec):
    """주문·잔고·체결에 쓰는 API 만 끝까지 풀어서 → data/toss/api_detail.json (공개 명세 — 계좌 정보 없음)."""
    ops = []
    for m, path in DETAIL_OPS:
        item = (spec.get("paths") or {}).get(path) or {}
        op = item.get(m.lower())
        if not op:
            continue
        params = [resolve(spec, x) for x in (op.get("parameters") or []) + (item.get("parameters") or [])]
        rb = resolve(spec, op.get("requestBody") or {})
        body = {ct: {"schema": deep(spec, c.get("schema")), "examples": c.get("examples") or c.get("example")}
                for ct, c in ((rb or {}).get("content") or {}).items()}
        res = {}
        for code, r in (op.get("responses") or {}).items():
            r = resolve(spec, r)
            res[code] = {"description": ((r or {}).get("description") or "")[:300],
                         **{ct: deep(spec, c.get("schema")) for ct, c in ((r or {}).get("content") or {}).items() if code in ("200", "201")}}
        ops.append({"method": m, "path": path, "summary": op.get("summary"), "description": (op.get("description") or "")[:4000],
                    "params": [{"name": x.get("name"), "in": x.get("in"), "required": x.get("required", False),
                                "description": (x.get("description") or "")[:400], "schema": deep(spec, x.get("schema"))}
                               for x in params if isinstance(x, dict)],
                    "body": body, "responses": res})
    info = spec.get("info") or {}
    return {"version": info.get("version"), "info_description": (info.get("description") or "")[:8000], "ops": ops}


def summarize(spec):
    ops = []
    for path, item in (spec.get("paths") or {}).items():
        for method, op in item.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            params = [resolve(spec, p) for p in (op.get("parameters") or []) + (item.get("parameters") or [])]
            body = None
            rb = resolve(spec, op.get("requestBody") or {})
            for ct, c in ((rb or {}).get("content") or {}).items():
                body = {"content_type": ct, "schema": fields(spec, c.get("schema"))}
                break
            res = {}
            for code, r in (op.get("responses") or {}).items():
                r = resolve(spec, r)
                for ct, c in ((r or {}).get("content") or {}).items():
                    res[code] = fields(spec, c.get("schema"))
                    break
                else:
                    res[code] = None
            ops.append({"method": method.upper(), "path": path, "summary": op.get("summary") or op.get("operationId"),
                        "tags": op.get("tags"),
                        "params": [{"name": p.get("name"), "in": p.get("in"), "required": p.get("required", False),
                                    "desc": (p.get("description") or "")[:80]} for p in params if isinstance(p, dict)],
                        "body": body, "responses": res})
    return {"title": (spec.get("info") or {}).get("title"), "version": (spec.get("info") or {}).get("version"),
            "servers": spec.get("servers"), "security": list(((spec.get("components") or {}).get("securitySchemes") or {}).keys()),
            "n_ops": len(ops), "ops": ops}


def shape(j, depth=0):
    """응답의 '모양'만 (키 이름·형식·배열 길이) — 값은 남기지 않는다."""
    if depth > 4:
        return "…"
    if isinstance(j, dict):
        return {k: shape(v, depth + 1) for k, v in list(j.items())[:40]}
    if isinstance(j, list):
        return [shape(j[0], depth + 1)] if j else []          # 개수도 남기지 않는다 (보유·주문 건수도 계좌 정보)
    return type(j).__name__


def seq_from_accounts(tok):
    try:
        r = requests.get(BASE + "/api/v1/accounts", timeout=15, headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"})
        items = (r.json() or {}).get("result") or []
        env = load_env()
        for it in items:
            if str(it.get("accountNo")) == str(env.get("TOSS_ACCOUNT")) or it.get("accountType") == "BROKERAGE":
                return it.get("accountSeq")
    except Exception:
        return None
    return None


def get_spec():
    for u in SPEC_URLS:
        try:
            r = requests.get(u, timeout=15)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith(("application/json", "text/plain")):
                s = r.json()
                if isinstance(s, dict) and s.get("paths"):
                    return u, s
            print(f"  명세 {u} → {r.status_code}")
        except Exception as e:
            print(f"  명세 {u} → 실패 ({type(e).__name__})")
    return None, None


def get_token(env):
    if os.path.exists(TOKEN):
        try:
            t = json.load(open(TOKEN))
            if t.get("expires_at", 0) > time.time() + 300:
                return t["access_token"], {"status": "cached", "expires_in": int(t["expires_at"] - time.time())}
        except Exception:
            pass
    r = requests.post(BASE + "/oauth2/token", timeout=15,
                      data={"grant_type": "client_credentials", "client_id": env.get("TOSS_CLIENT_ID", ""),
                            "client_secret": env.get("TOSS_CLIENT_SECRET", "")},
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    info = {"status": r.status_code}
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code == 200 and j.get("access_token"):
        exp = int(j.get("expires_in") or 3600)
        with open(os.open(TOKEN, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as fh:
            json.dump({"access_token": j["access_token"], "expires_at": time.time() + exp}, fh)
        info.update(expires_in=exp, token_type=j.get("token_type"), fields=sorted(j.keys()))
        return j["access_token"], info
    info["error"] = mask({k: v for k, v in j.items() if k in ("error", "error_description", "code", "message")} or r.text[:200])
    return None, info


def main():
    os.makedirs(OUT, exist_ok=True)
    env = load_env()
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "base": BASE}
    try:
        res["egress_ip"] = requests.get("https://ifconfig.me/ip", timeout=8).text.strip()
    except Exception:
        res["egress_ip"] = None
    print("1) 공식 명세 받기")
    url, spec = get_spec()
    summ = None
    if spec:
        summ = summarize(spec)
        summ["source"] = url
        json.dump(summ, open(os.path.join(OUT, "api_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  ✓ {url} · 버전 {summ['version']} · API {summ['n_ops']}개 → data/toss/api_summary.json")
        res["spec"] = {"url": url, "version": summ["version"], "n_ops": summ["n_ops"]}
        det = detail(spec)
        json.dump(det, open(os.path.join(OUT, "api_detail.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  ✓ 주문·잔고·체결 API {len(det['ops'])}개 상세 → data/toss/api_detail.json")
    else:
        res["spec"] = None
        print("  ✗ 명세를 받지 못함 (주소가 바뀌었을 수 있음)")
    print("2) 토큰 발급 (POST /oauth2/token)")
    if not env.get("TOSS_CLIENT_ID") or not env.get("TOSS_CLIENT_SECRET"):
        print("  ✗ 키가 없습니다 — bash ~/rich/deploy/oracle/setup.sh --keys")
        res["token"] = {"status": "no_keys"}
        tok = None
    else:
        tok, info = get_token(env)
        res["token"] = info
        print(f"  {'✓' if tok else '✗'} 상태 {info.get('status')}" + (f" · 유효 {info.get('expires_in')}초" if tok else f" · {info.get('error')}"))
        if not tok and info.get("status") == 403:
            print(f"  → 403 은 보통 허용 IP 문제입니다. 토스 허용 IP 에 {res['egress_ip']} 이 등록됐는지 확인하세요.")
    print("3) 계좌 목록 조회 (GET, 잔고는 보지 않음)")
    res["accounts"] = None
    if tok:
        paths = []
        if summ:
            paths = [o["path"] for o in summ["ops"] if o["method"] == "GET" and re.search(r"/accounts/?$", o["path"])]
        paths = paths or ["/v1/accounts", "/api/v1/accounts"]
        for p in paths[:3]:
            try:
                r = requests.get(BASE + p, timeout=15, headers={"Authorization": f"Bearer {tok}", "Accept": "application/json"})
            except Exception as e:
                print(f"  ✗ {p} → {type(e).__name__}")
                continue
            entry = {"path": p, "status": r.status_code}
            try:
                j = r.json()
            except Exception:
                j = None
            items = j if isinstance(j, list) else next((v for v in (j or {}).values() if isinstance(v, list)), None) if isinstance(j, dict) else None
            if r.status_code == 200 and items is not None:
                entry["count"] = len(items)
                entry["item_keys"] = sorted(items[0].keys()) if items and isinstance(items[0], dict) else None
                entry["top_keys"] = sorted(j.keys()) if isinstance(j, dict) else None
                ident = None
                if items and isinstance(items[0], dict):
                    for k in items[0]:
                        if re.search(r"account", k, re.I) and re.search(r"(no|num|id|key|seq)", k, re.I):
                            ident = str(items[0][k])
                            entry["id_field"] = k
                            entry["id_masked"] = mask(ident)
                            break
                if ident and not env.get("TOSS_ACCOUNT"):
                    lines = [ln for ln in open(SECRETS, encoding="utf-8") if not ln.startswith("TOSS_ACCOUNT=")]
                    lines.append(f"TOSS_ACCOUNT={ident}\n")
                    with open(os.open(SECRETS, os.O_WRONLY | os.O_TRUNC, 0o600), "w", encoding="utf-8") as fh:
                        fh.writelines(lines)
                    entry["saved_to_secrets"] = True
                print(f"  ✓ {p} → 계좌 {entry['count']}개" + (f" · {entry.get('id_field')} {entry.get('id_masked')} (서버 키 파일에 저장)" if entry.get("saved_to_secrets") else ""))
                res["accounts"] = entry
                break
            entry["error"] = mask({k: j[k] for k in list(j)[:4]} if isinstance(j, dict) else r.text[:200])
            print(f"  ✗ {p} → {r.status_code}")
            res.setdefault("tried", []).append(entry)
    # 4) 읽기 전용 점검 — 상태 코드와 '항목 이름'만 남긴다 (금액·종목·수량은 저장소에 올리지 않음)
    if tok and res.get("accounts"):
        seq = env.get("TOSS_ACCOUNT_SEQ") or seq_from_accounts(tok)
        if seq and not env.get("TOSS_ACCOUNT_SEQ"):
            lines = [ln for ln in open(SECRETS, encoding="utf-8") if not ln.startswith("TOSS_ACCOUNT_SEQ=")]
            lines.append(f"TOSS_ACCOUNT_SEQ={seq}\n")
            with open(os.open(SECRETS, os.O_WRONLY | os.O_TRUNC, 0o600), "w", encoding="utf-8") as fh:
                fh.writelines(lines)
        checks = {}
        for name, path, params in (("holdings", "/api/v1/holdings", None), ("buying_power", "/api/v1/buying-power", {"currency": "KRW"}),
                                   ("orders", "/api/v1/orders", None), ("calendar_kr", "/api/v1/market-calendar/KR", None),
                                   ("prices", "/api/v1/prices", {"symbols": "069500"}), ("commissions", "/api/v1/commissions", None)):
            try:
                r = requests.get(BASE + path, params=params, timeout=15, headers={"Authorization": f"Bearer {tok}", "Accept": "application/json",
                                                                                 **({"X-Tossinvest-Account": str(seq)} if seq else {})})
                j = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
            except Exception as e:
                checks[name] = {"error": type(e).__name__}
                continue
            checks[name] = {"status": r.status_code, "shape": shape(j)}
            if r.status_code != 200:
                checks[name]["error"] = mask({k: j[k] for k in list(j)[:3]} if isinstance(j, dict) else "")
            print(f"  {'✓' if r.status_code == 200 else '✗'} {path} → {r.status_code}")
            time.sleep(0.3)
        res["readonly"] = checks
    json.dump(res, open(os.path.join(OUT, "probe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = bool(res.get("accounts"))
    print(f"\n결과: {'연결 성공' if ok else '아직 연결 안 됨'} — data/toss/probe.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
