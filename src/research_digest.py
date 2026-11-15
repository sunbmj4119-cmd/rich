"""
지금까지의 연구 모음 → docs/research_all.json ('내 계좌' 화면의 '연구 결과' 탭)

  예전 연구 (docs/STRATEGY.md 3-1 ~ 3-6-13 · 3-7): data/research_digest_base.json — 문서의 숫자 그대로 한 줄씩 옮겨 둔 것
  최근 연구 (2026-11-14 ~): 각 실험실 결과 파일에서 숫자를 바로 읽어 만든다 (결과 파일이 바뀌면 같이 바뀜)
  판정: 채택 = 지금 계좌에 씀 · 참고 = 일부만 쓰거나 확신 부족 · 기각 = 검증에서 떨어짐

  python src/research_digest.py
"""
import json
import os
import time

BASE = "data/research_digest_base.json"
OUT = "docs/research_all.json"


def _j(fn):
    try:
        return json.load(open(fn, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def p(v, d=1, sign=True):
    if v is None:
        return "—"
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def recent():
    out = []
    if os.path.exists("data/shadow_wf.json"):                # 숫자는 STRATEGY 3-6-10-16 표 그대로
        out.append({"id": "3-6-10-16", "date": "2026-11-14", "title": "그림자 기록 거꾸로 따라가기", "src": "src/shadow_wf.py", "verdict": "기각",
                    "q": "매주 고르던 종목 모델(5일마다 3종목)을 2008 년부터 실제로 했다면 정말 시장을 이겼나?",
                    "result": "같은 규칙인데 5일 주기를 며칠째부터 세느냐에 따라 2019~ 성적이 연 +38% 에서 +4% 까지 갈렸습니다. 앞서 준 '통과'는 날짜 운이어서 거둬들였고, 5묶음으로 고르게 재면 DM:2 보다 낮았습니다.",
                    "nums": [{"k": "1번째 날 2019~", "v": "+38.1%"}, {"k": "5번째 날 2019~", "v": "+3.6%"},
                             {"k": "5묶음 섞기 2019~", "v": "+11.5%"}, {"k": "DM:2 같은 기간", "v": "+16.2%"}],
                    "used_now": "h일마다 바꾸는 규칙은 'h개 묶음'으로 재기 · 20만 원 계획은 DM:2 유지"})
    s = _j("data/score_lab.json")
    if s:
        f = g(s, "final", "hold") or {}
        out.append({"id": "3-6-10-17", "date": "2026-11-14", "title": "점수 매매 실험실 (1·2차)", "src": "src/score_lab.py", "verdict": "기각",
                    "q": "종목 점수를 숫자로 매기고, 공식·가중치·요소와 매수·매도 규칙을 바꿔 가며 찾은 규칙이 새 기간에도 통할까?",
                    "result": "1차는 2019~ 최종 확인 한 번에서 떨어졌습니다(연 +0.2% · 샤프 0.16) — 비용 전에는 나았지만 1년에 150~450번 사고팔아 비용이 다 먹었습니다. 2차(해마다 과거로만 다시 고르기)는 미리 정한 기준을 넘었지만(2019~ 연 +35.8%) 고르는 방법을 조금만 바꿔도 +1%~+20% 로 흔들려 실제 돈에는 넣지 않았습니다.",
                    "nums": [{"k": "최종 2019~ 연", "v": p(f.get("cagr"))}, {"k": "대형주 평균", "v": p(f.get("ew_cagr"))},
                             {"k": "거래/년", "v": f"{f.get('trades_per_year', 0):.0f}번"}, {"k": "낙폭", "v": p(f.get("mdd"), 0)}],
                    "used_now": "비용 0.34% 를 늘 넣는 같은 흉내 장치 · 사전 등록 규칙"})
    r = _j("data/robust_lab.json")
    if r:
        sl = g(r, "composite", "rules", "느림") or {}
        out.append({"id": "3-6-10-18", "date": "2026-11-14", "title": "견고한 점수 찾기 (3차)", "src": "src/robust_lab.py", "verdict": "참고",
                    "q": f"단순한 점수 공식 {r.get('n_candidates', 0):,}개를 세 시기 모두에서 시험해 운이 아닌 것을 고를 수 있나?",
                    "result": "'장기 추세 + 단기 눌림' 묶음이 세 시기 모두 시장을 이겼고 주변 값에도 넓게 버텼습니다. 다만 2019~ 만 따로 보면 운과 구분이 안 됐습니다(SPA p 0.45).",
                    "nums": [{"k": "느린 규칙 2019~ 연", "v": p(g(sl, "2019~", "strategy", "cagr"))}, {"k": "샤프", "v": f"{g(sl, '2019~', 'strategy', 'sharpe') or 0:.2f}"},
                             {"k": "낙폭", "v": p(g(sl, "2019~", "strategy", "mdd"), 0)}, {"k": "세 시기 일관", "v": f"{r.get('n_consistent', 0)}개"}],
                    "used_now": "토스 도전자 모의 기록 (trend_shadow) · 토스 공격 고배당 점수의 절반"})
    d = _j("data/div_lab.json")
    if d:
        st = d.get("strategies") or {}
        out.append({"id": "3-6-10-19", "date": "2026-11-14", "title": "고배당주로 가면 — 이익 구조", "src": "src/div_lab.py", "verdict": "참고",
                    "q": "고배당주를 사서 배당을 받으며 가면 이익은 가격과 배당 중 어디서 나오고, 시장을 이기나?",
                    "result": "고배당만 담으면 배당(연 4~5%)은 확실히 쌓였지만 가격이 덜 올라 대형주 평균과 비슷했습니다. 배당철만 노리는 방법은 크게 졌습니다.",
                    "nums": [{"k": "고배당 꾸준히 2003~ 연", "v": p(g(st, "D1", "전체 2003~", "cagr"))}, {"k": "그중 배당", "v": p(g(st, "D1", "전체 2003~", "div_part"))},
                             {"k": "배당철 받기", "v": p(g(st, "D5", "전체 2003~", "cagr"))}, {"k": "추세+고배당 반반 2019~", "v": p(g(d, "mix", "2019~", "cagr"))}],
                    "used_now": "ISA 핵심을 고배당 ETF 로 · 배당 재투자 · 주식배당·증자 거르기(2026-11-15 고침)"})
    dp = _j("data/div_project.json")
    if dp:
        sp = g(dp, "speed", "ETF대용", "2003~ 전체") or {}
        imp = {x.get("change", ""): x for x in (dp.get("improve") or [])}
        spend = next((v for k, v in imp.items() if "꺼내" in k), {})
        out.append({"id": "3-6-10-20", "date": "2026-11-14", "title": "고배당 ETF 로 자산 키우는 속도", "src": "src/div_project.py", "verdict": "채택",
                    "q": "고배당 ETF·주식으로 가면 자산이 얼마나 빨리 불고, 무엇을 바꾸면 더 빨라지나?",
                    "result": "가장 큰 차이는 배당을 꺼내 쓰지 않고 다시 사는 것이었습니다. 보수 0.3%→0.1% 나 세금보다 컸습니다.",
                    "nums": [{"k": "고배당 ETF 대용 연", "v": p(sp.get("cagr"))}, {"k": "두 배까지", "v": f"{sp.get('double_years', '—')}년"},
                             {"k": "배당 꺼내 쓰면", "v": p(spend.get("cagr"))}, {"k": "최악 1년", "v": p(sp.get("worst_year"), 0)}],
                    "used_now": "ISA 핵심 50% (PLUS 고배당주) · 분배금 재투자"})
    il = _j("data/isa_lab.json")
    if il:
        e = g(il, "eras", "전체 2005~") or {}
        out.append({"id": "새로", "date": "2026-11-15", "title": "ISA 계획 — 고배당 50 + 매달 회전 50", "src": "src/isa_daily.py", "verdict": "채택",
                    "q": "ISA 에서 고배당 ETF 와 짝을 지을 나머지 반을 코스피200 으로 둘까, 매달 1·2등 ETF 로 갈아탈까?",
                    "result": f"세 시기 중 {il.get('p1_wins', 0)}곳에서 매달 갈아타는 쪽이 위험 대비 더 나았고, 최대 낙폭이 크게 얕았습니다. 고배당 ETF 와 덜 같이 움직였기 때문입니다.",
                    "nums": [{"k": "회전 안 2005~ 연", "v": p(g(e, "P1", "cagr"))}, {"k": "샤프 (코스피200 안)", "v": f"{g(e, 'P1', 'sharpe') or 0:.2f} ({g(e, 'P0', 'sharpe') or 0:.2f})"},
                             {"k": "낙폭 (코스피200 안)", "v": f"{p(g(e, 'P1', 'mdd'), 0)} ({p(g(e, 'P0', 'mdd'), 0)})"}, {"k": "상관", "v": str(g(il, "corr", "고배당-DM:2"))}],
                    "used_now": "ISA 매일 안내 (docs/isa.json · '내 계좌' 화면)"})
    td = _j("data/toss_div_lab.json")
    if td:
        k = td.get("picked")
        c = g(td, "cands", k) or {}
        out.append({"id": "새로", "date": "2026-11-15", "title": "토스 공격 고배당 — 후보 5개", "src": "src/toss_div.py", "verdict": "채택",
                    "q": "ISA 의 ETF 로는 못 하는 '고배당 개별 종목 몇 개에 몰아 담고 추세로 갈아타기'가 작은 토스 계좌에서 통할까?",
                    "result": f"미리 정한 기준(세 시기 모두 초과 > 0 · 20만 원으로 60% 이상 실제로 삼)을 {', '.join(td.get('passed') or []) or '아무것도'} 통과 → {c.get('name', '')}. 낙폭이 매우 커 작은 돈에만 쓰고, 지금은 모의 기록만 쌓습니다.",
                    "nums": [{"k": "2003~ 연", "v": p(g(c, "전체 2003~", "cagr"))}, {"k": "낙폭", "v": p(g(c, "전체 2003~", "mdd"), 0)},
                             {"k": "2019~ 20만 원 연", "v": p(g(c, "accounts_2019", "200000", "cagr"))}, {"k": "거래/년", "v": f"{c.get('trades_per_year', 0):.0f}번"}],
                    "used_now": "토스 공격 고배당 모의 기록 (2026-11-16 ~) — 실제 주문은 조건이 다 맞고 승인한 뒤"})
    ig = _j("data/income_goal.json")
    if ig:
        n = g(ig, "need", "고배당·저변동") or {}
        ip = g(ig, "income_path", "고배당·저변동", "2003-01-01") or {}
        row = next((r for r in ig.get("years_table") or [] if r["base"] == "2008~2018 (보수적)" and r["monthly"] == 500_000 and r["infl"] == 0), {})
        na = g(ig, "need_by_acct", "고배당·저변동") or {}
        d90 = g(ig, "draw", "need90") or {}
        out.append({"id": "새로", "date": "2026-11-15", "title": f"배당으로 매달 {ig.get('goal_month', 0) // 10000}만 원", "src": "src/income_goal.py", "verdict": "채택",
                    "q": f"세후 배당 매달 {ig.get('goal_month', 0) // 10000}만 원을 받으려면 계좌별로 얼마가 필요하고, 몇 년 걸리고, 받는 배당은 얼마나 흔들리나? 일부를 팔아 쓰면?",
                    "result": f"고배당·덜 흔들림 10종목의 지금 배당률(세전 {p(n.get('yield_now_gross'), 1, False)})이면 일반 계좌 {na.get('일반', 0) / 1e8:.2f}억 · ISA {na.get('ISA', 0) / 1e8:.2f}억 · 연금 {na.get('연금', 0) / 1e8:.2f}억. "
                              f"일부를 팔아 쓰면 30년 10번 중 9번 버티는 데 보통 시기 {d90.get('2005~ 전체', 0) / 1e8:.2f}억 · 약한 시기 {d90.get('2008~2018 (보수적)', 0) / 1e8:.2f}억. "
                              f"2003 에 사 두고 배당만 꺼내 썼다면 받은 배당은 해마다 {p(ip.get('growth'))} 늘었지만 {ip.get('worst_year')} 년에는 {p(ip.get('worst_drop'), 0)} 줄었다 → 1년치 비상금과 배당 통장으로 받는다.",
                    "nums": [{"k": "필요한 돈 (10종목 · ISA)", "v": f"{na.get('ISA', 0) / 1e8:.2f}억"}, {"k": "매달 50만 넣으면 (약한 시기)", "v": f"{row.get('p50', '—')}년"},
                             {"k": "배당 가장 크게 준 해", "v": p(ip.get("worst_drop"), 0)}, {"k": "12월 배당락 몫", "v": p(g(ig, "current", "ex_month_share", "12"), 0, False)}],
                    "used_now": "'내 계좌' 화면의 '배당 100만' 탭 (계산기 · 세 단계 계획)"})
    return out


def build():
    base = _j(BASE) or []
    for x in base:
        x.setdefault("nums", [])
    items = recent()[::-1] + base[::-1]                     # 최근 연구가 위로
    order = {"채택": 0, "참고": 1, "기각": 2}
    n = {v: sum(1 for x in items if x.get("verdict") == v) for v in order}
    return {"lead": f"모든 연구는 같은 잣대(그날 저녁까지의 자료로 판단 → 다음 거래일 체결 · 비용 포함 · 미리 정한 기준)로 쟀습니다. "
                    f"채택 {n['채택']} · 참고 {n['참고']} · 기각 {n['기각']} — 떨어진 것이 더 많은 것이 정상입니다.",
            "items": items, "counts": n, "generated": time.strftime("%Y-%m-%d %H:%M")}


def main():
    res = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"연구 모음 {len(res['items'])}개 ({res['counts']}) → {OUT}")


if __name__ == "__main__":
    main()
