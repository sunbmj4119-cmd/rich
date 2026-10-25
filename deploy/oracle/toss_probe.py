"""
토스증권 오픈API 연결 점검 (서버에서 실행) — 돈이 움직이는 기능은 쓰지 않는다.

  1) 공식 OpenAPI 명세를 받아 '주소 · 방식 · 입력 이름 · 출력 이름'만 추린다 → data/toss/api_summary.json
  2) 키로 토큰을 발급받고(POST /oauth2/token) 계좌 목록만 조회한다(GET) → 상태 코드·개수만 data/toss/probe.json
  3) 계좌 식별값을 찾으면 서버의 ~/.rich/secrets.env 에만 TOSS_ACCOUNT 로 저장한다 (비어 있을 때만)

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
    json.dump(res, open(os.path.join(OUT, "probe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = bool(res.get("accounts"))
    print(f"\n결과: {'연결 성공' if ok else '아직 연결 안 됨'} — data/toss/probe.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
