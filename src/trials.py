"""
시도 횟수 장부 — '몇 개를 시도해서 1등을 골랐나'를 빠짐없이 센다 (Bailey·López de Prado: 시도 수를 숨기면 과최적화가 숨는다).

  국내 단기 몫의 1등이 이기려고 겨룬 상대 = 고정 후보 모델 × 고르는 방식 격자 + 진화로 만들어 평가한 유전자 전부
  다른 연구(가설·남들 전략·긴 역사·미국·30일 엔진)도 따로 센다 — 화면에 그대로 보여 준다.
  science_lab 의 DSR(진짜 실력일 확률)은 이 '전체 시도 수'를 N 으로 쓴다 (예전엔 후보 수만 썼다 → 너무 너그러웠다).

  python src/trials.py → data/trials.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/trials.json"


def jload(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def count():
    import short_lab as L
    kr_models = len(L.stock_models()) + len(L.etf_models())
    grid = 2 * 3 * 6 * 2 * 2 * 2                             # short_lab.rule_grid: 후보군 2 × 주기 3 × 기간 6 × 기준 2 × k 2 × 끈기 2
    evo = jload("data/short/evo_population.json", {}) or {}
    genomes = int(evo.get("generations", 0)) * max(len(evo.get("pop") or []), 1)
    reg30 = jload("data/evolve/registry.json", {}) or {}
    sci = jload("data/science/lab.json", {}) or {}
    comm = jload("data/community/lab.json", {}) or {}
    hist = jload("data/history/lab.json", {}) or {}
    us = jload("docs/us_lab.json", {}) or {}
    opt = jload("data/optimize/lab.json", {}) or {}
    rows = {
        "kr_short_models": {"n": kr_models, "what": "국내 단기 고정 후보 (종목 모델 + 지수 ETF 규칙)"},
        "kr_rule_grid": {"n": grid, "what": "고르는 방식 격자 (얼마나 자주·몇 일 성적·무엇으로·몇 개·끈기)"},
        "kr_evolved": {"n": genomes, "what": f"단기 진화로 만들어 평가한 유전자 (세대 {evo.get('generations', 0):,} × 인구 {len(evo.get('pop') or [])})"},
        "aw_cards": {"n": 21 + 11, "what": "전천후 카드 21장 + 섞는 방식 11가지"},
        "science": {"n": len(sci.get("hypotheses") or []), "what": "다분야 가설"},
        "community": {"n": len((comm.get("rows") or {})), "what": "남들 전략 재검증"},
        "history": {"n": len(((hist.get("sector_kr") or {}).get("variants") or {})) + len(((hist.get("sector_us") or {}).get("variants") or {})) + 10,
                    "what": "긴 역사: 업종 회전 변형 + 지수 ETF 규칙"},
        "us_models": {"n": int(us.get("n_models") or 270), "what": "미국 종목 모델"},
        "thirty_day": {"n": int(reg30.get("n_evaluated") or 0), "what": "30일 엔진 진화에서 평가한 설정"},
        "optimize": {"n": int(opt.get("n_trials") or 0), "what": "최적화 실험실 (앙상블·덮개·ETF 위성·손절익절, 회차 합계)"},
    }
    kr_n = kr_models * grid + genomes
    return {"generated": time.strftime("%Y-%m-%d %H:%M"), "rows": rows, "kr_short_effective": kr_n,
            "total": int(sum(v["n"] for v in rows.values())),
            "note": "국내 단기 몫 1등의 '유효 시도 수' = 고정 후보 × 고르는 방식 + 진화 유전자. DSR 의 N 으로 쓴다."}


def main():
    res = count()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"시도 장부: 전체 {res['total']:,}번 · 국내 단기 몫 유효 시도 {res['kr_short_effective']:,}번 → {OUT}")
    return res


if __name__ == "__main__":
    main()
