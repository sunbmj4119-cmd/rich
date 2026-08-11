"""
투자 논리 엔진 — 상승 시나리오 / 하락·반대 논리 / 확률 계산
build_data.py가 import해서 종목마다 호출한다(별도 실행 불필요).

이 파일이 하는 일은 딱 셋이다.
  1) bull_case()  — 왜 오를 수 있나: 어떤 팩터가 이끄는지, 그게 실현되려면 무엇이 참이어야 하는지
  2) bear_case()  — 왜 틀릴 수 있나: 자동 반대논리 체크리스트 + 논리가 깨졌다고 인정할 조건(무효화)
  3) probability() — 얼마나 확실한가: 겹침 보정 유효표본 · Wilson 신뢰구간 · 손절 반영 기대값 · 켈리 비중

설계 철학
  - '오를 것 같다'는 말을 쓰지 않는다. 과거 유사 국면의 **분포**만 말한다.
  - 표본이 겹치면(30일 창을 매일 표집) 실제 독립 관측치는 N/30이다. 이 보정을 반드시 한다.
    보정 없이 계산한 신뢰구간은 실제보다 5배 이상 좁아 '확실해 보이는 착시'를 만든다.
  - 반대논리는 사람이 기억해서 떠올리는 게 아니라 규칙으로 자동 제시한다(확증편향 차단).
"""
import math

FWD = 30            # 예측 지평(거래일) = 표본 겹침 길이
ROUND_COST = 0.003  # 왕복 거래비용+슬리피지 가정 0.3%
STOP = -0.10        # 손절선 (signal.py와 동일)
CONCURRENT = 10     # 이 전략이 동시에 드는 종목 수 (켈리 비중을 나누는 분모)

# ── 팩터별 상승 촉매 / 그 논리가 성립하려면 무엇이 참이어야 하나 ─────────────
CATALYST = {
    "s_value": ("저평가 해소(리레이팅)",
                "PER·PBR이 100종목 중 하위권. 시장이 정상 밸류로 되돌리면 오른다.",
                "이익이 훼손되지 않아야 한다 — 싸진 게 실적 악화 때문이면 더 싸질 수 있다."),
    "s_profit": ("이익 체력",
                 "ROE·영업이익률이 상위권이고 개선 중. 버는 힘이 주가를 끌어올린다.",
                 "다음 분기에도 마진이 유지돼야 한다 — 일회성 이익이면 되돌아간다."),
    "s_grow": ("성장 지속",
               "매출·영업이익·순이익이 전년 대비 성장. 성장이 이어지면 재평가된다.",
               "성장률이 꺾이지 않아야 한다 — 기저효과였다면 다음 분기에 드러난다."),
    "s_flow": ("외국인 매수 지속",
               "최근 20일 외국인이 시총 대비 순매수. 수급이 뒤를 받친다.",
               "외국인이 순매도로 돌아서지 않아야 한다 — 수급은 가장 빨리 뒤집힌다."),
    "s_mom": ("추세 지속",
              "12-1개월 상승 추세가 살아있다.",
              "추세가 꺾이지 않아야 한다 — 대형주는 평균회귀가 강해 오래 못 간다."),
}


# ══════════════════════════════════════════════════════════════
# 확률 계산
# ══════════════════════════════════════════════════════════════
def wilson(p, n, z=1.96):
    """Wilson score 신뢰구간. p=비율(0~1), n=유효표본수. 표본이 작을수록 정직하게 넓어진다."""
    if n <= 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def probability(rets, eff_n=None, base_win=None):
    """
    과거 유사국면 수익률 배열(list[float], 소수) → 확률 블록.

    핵심 두 가지 (이걸 빼면 숫자가 실제보다 확실해 보인다):
      · eff_n — 30일 창을 매일 표집한 표본은 서로 겹친다. 신뢰구간은 '겹치지 않는
        관측치 수'로 계산해야 한다. build_data가 실측해 넘긴다(없으면 n/30 근사).
      · base_win — 아무 종목이나 아무 날 샀을 때의 승률. 엣지는 '승률'이 아니라
        '승률 − 기준선'이다.
    """
    r = [x for x in rets if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(r)
    if n < 10:
        return None
    eff_n = float(eff_n) if eff_n else max(1.0, n / FWD)
    eff_n = max(1.0, eff_n)

    # 실제 전략은 -10%에서 손절한다 → 손절을 적용한 수익분포로 기대값을 낸다(근사).
    rs = [max(x, STOP) for x in r]

    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    p_win = len(wins) / n
    lo, hi = wilson(p_win, eff_n)

    g = sum(wins) / len(wins) if wins else 0.0            # 이길 때 평균 상승폭
    l = abs(sum(losses) / len(losses)) if losses else 0.0  # 질 때 평균 하락폭

    ev_raw = sum(r) / n
    ev = sum(rs) / n - ROUND_COST                          # 손절·비용 반영 기대수익

    payoff = (g / l) if l > 1e-9 else None                 # 손익비

    # ── 켈리 비중 (계좌의 몇 %를 이 종목에 넣을 것인가) ─────────
    # 연속수익 근사 f* ≈ μ/σ² (E[log(1+fR)] 최대화). 2결과 공식은 손절로 l이 작아지면
    # f*가 수백 %로 튀어 무의미해지므로 쓰지 않는다.
    # 교과서 켈리를 그대로 쓰면 반드시 과대베팅이 된다. 세 겹으로 깎는다:
    #   1) 추정오차 — μ는 얇은 표본의 추정치다. 표준오차 1개만큼 깎아 보수적 μ를 쓴다.
    #   2) 동시보유 — 이 전략은 10종목을 함께 든다. 종목들은 같이 움직이므로(시장 베타)
    #      한 종목이 전체 리스크예산을 독점할 수 없다 → CONCURRENT로 나눈다.
    #   3) 상한·하프켈리 — 캡 20%, 실제 권장은 그 절반.
    # 결과가 0%면 "이 표본으로는 비중을 키울 근거가 없다"는 뜻이다(음수 베팅은 안 한다).
    mu = ev
    var = (sum((x - sum(rs) / n) ** 2 for x in rs) / (n - 1)) if n > 1 else 0.0
    sd = math.sqrt(var) if var > 0 else 0.0
    se = sd / math.sqrt(eff_n) if eff_n > 0 else 0.0
    mu_lo = mu - se                                        # 추정오차 1σ 차감
    kelly = _clamp((mu_lo / var) / CONCURRENT, 0.0, 0.20) if var > 1e-9 else None

    def frac(cond):
        return round(sum(1 for x in r if cond(x)) / n * 100, 1)

    return {
        "n": n, "eff_n": round(eff_n, 1),
        "win": round(p_win * 100, 1),
        "win_lo": round(lo * 100, 1), "win_hi": round(hi * 100, 1),
        "base_win": base_win,
        "edge_pp": round(p_win * 100 - base_win, 1) if base_win is not None else None,
        # 신뢰구간 하한이 기준선보다 위인가 = 통계적으로 우위를 단정할 수 있는가
        "proven": bool(base_win is not None and lo * 100 > base_win),
        "ev": round(ev * 100, 2), "ev_raw": round(ev_raw * 100, 2),
        "avg_win": round(g * 100, 1), "avg_loss": round(l * 100, 1),
        "payoff": round(payoff, 2) if payoff else None,
        "sd": round(sd * 100, 1), "se": round(se * 100, 2),
        "mu_lo": round(mu_lo * 100, 2),
        "kelly": round(kelly * 100, 1) if kelly is not None else None,
        # 실제 권장은 하프켈리 — 과거≠미래(비정상성)를 감안한 관행적 안전계수
        "kelly_use": round(kelly / 2 * 100, 1) if kelly is not None else None,
        "concurrent": CONCURRENT,
        # 결과 구간별 실측 빈도 (손절 미적용 원분포 = 실제로 겪는 가격 경로)
        "p_stop": frac(lambda x: x <= STOP),      # 손절 맞을 확률
        "p_down5": frac(lambda x: x <= -0.05),
        "p_up10": frac(lambda x: x >= 0.10),
        "p_up20": frac(lambda x: x >= 0.20),
        "cost": round(ROUND_COST * 100, 2),
    }


# ══════════════════════════════════════════════════════════════
# 상승 시나리오
# ══════════════════════════════════════════════════════════════
# 무효화 조건 템플릿. {stop}=체결가 -10% 근방, {bull}=상승논리의 핵심.
# 값이 없는 항목은 UI에서 건너뛴다(첫 항목은 stop, 마지막은 bull이 있을 때만).
INVALIDATION = [
    "체결가 대비 -10%({stop}원 근방) 이탈 → 논리 불문 손절",
    "진입 후 고점 대비 -8% 하락(트레일링 스탑) → 청산",
    "종합점수 순위가 20위 밖으로 이탈(최소 30거래일 보유 후) → 매도",
    "외국인 20일 순매수가 순매도로 전환 → 비중 축소 검토",
    "상승 논리의 핵심({bull})이 다음 분기 실적에서 깨짐 → 즉시 재검토",
]

TIMING_CATALYST = ("과매도 반등",
                   "60일 고점 대비 크게 눌린 상태. 대형주는 단기 평균회귀가 관찰된다.",
                   "눌림이 '일시적 조정'이어야 한다 — 실적 훼손으로 인한 하락이면 계속 빠진다.")
NO_DRIVER_MUST = "뚜렷한 강점이 없으므로, 이 종목을 사야 할 적극적 이유를 따로 확인하라."


def bull_case(ctx, meta=None):
    """
    ctx: build_data가 만든 종목 문맥 dict → 상승 논리.
    meta가 주어지면 촉매 설명(why/must)은 meta['catalysts'][title]에 한 번만 담고
    종목에는 (팩터명·점수·촉매제목)만 남긴다. 전제조건은 촉매 제목으로 되찾는다.
    """
    factors = {f["key"]: f for f in ctx["factors"]}
    drivers, extra = [], []

    def drv(name, val, title, why, must):
        if meta is not None:
            meta.setdefault("catalysts", {}).setdefault(title, {"why": why, "must": must})
            drivers.append({"name": name, "val": val, "title": title})
        else:
            drivers.append({"name": name, "val": val, "title": title,
                            "why": why, "must": must})

    for k, f in sorted(factors.items(), key=lambda kv: -kv[1]["contrib"]):
        if f["w"] <= 0 or f["val"] < 60:
            continue
        title, why, must = CATALYST.get(k, (f["name"], "", ""))
        drv(f["name"], f["val"], title, why, must)
        if len(drivers) >= 3:
            break

    timing = ctx.get("timing") or 0
    if timing >= 60:
        drv("타이밍", round(timing), *TIMING_CATALYST)

    pr = ctx.get("prob")
    target = None
    if pr and ctx.get("tgt_hi"):
        target = {"price": ctx["tgt_hi"], "pct": ctx["exp"]["p75"] if ctx.get("exp") else None,
                  "p_up10": pr["p_up10"], "p_up20": pr["p_up20"]}

    if drivers:
        head = " · ".join(d["title"] for d in drivers[:2])
        summary = (f"{ctx['name']}의 상승 논리는 <b>{head}</b>다. "
                   f"종합 {ctx['score']:.0f}점(100종목 중 {ctx['rank']}위).")
    else:
        summary = (f"{ctx['name']}은 뚜렷하게 앞서는 팩터가 없다. "
                   f"종합 {ctx['score']:.0f}점({ctx['rank']}위)은 특정 강점보다 평균적 균형에 가깝다.")
        extra.append(NO_DRIVER_MUST)

    if pr:
        summary += (f" 과거 비슷한 국면 {pr['n']}회(독립 ≈{pr['eff_n']:.0f}회)에서 30일 뒤 "
                    f"상승 {pr['win']}%, +10% 이상 {pr['p_up10']}%.")

    return {"summary": summary, "drivers": drivers, "extra_musts": extra, "target": target}


# ══════════════════════════════════════════════════════════════
# 하락 시나리오 / 반대 논리 (확증편향 차단용 자동 체크리스트)
# ══════════════════════════════════════════════════════════════
def bear_case(ctx, meta=None):
    """meta가 주어지면 반복되는 설명문은 meta['risks'][tag]에 한 번만 저장하고
    종목별로는 (심각도·태그·증거)만 남긴다 — data.json이 100배로 부푸는 것을 막는다."""
    v = {f["key"]: f["val"] for f in ctx["factors"]}
    risks = []

    def add(sev, tag, text, ev):
        r = {"sev": sev, "tag": tag, "evidence": ev}
        if meta is None:
            r["text"] = text
        else:
            prev = meta.setdefault("risks", {}).get(tag)
            if prev is None:
                meta["risks"][tag] = text
            elif prev != text:          # 같은 태그인데 문구가 다르면 그 종목에만 저장
                r["text"] = text
        risks.append(r)

    def rtext(r):
        return r.get("text") or (meta or {}).get("risks", {}).get(r["tag"], "")

    # 1) 밸류트랩 — 싼 데는 이유가 있다
    if v.get("s_value", 50) >= 70 and v.get("s_grow", 50) <= 40 and v.get("s_profit", 50) <= 50:
        add("high", "밸류트랩",
            "싸 보이는 이유가 '성장·수익성이 꺾여서'일 수 있다. 저PER이 함정인 전형적 조합.",
            f"가치 {v['s_value']:.0f} vs 성장 {v['s_grow']:.0f} · 수익성 {v['s_profit']:.0f}")
    # 2) 이익 체력 부실
    if v.get("s_profit", 50) <= 30:
        add("high", "수익성 취약",
            "ROE·영업이익률이 하위권. 업황이 나빠지면 가장 먼저 적자로 밀린다.",
            f"수익성 {v['s_profit']:.0f}점(100종목 중 하위)")
    # 3) 성장 정체
    if v.get("s_grow", 50) <= 30:
        add("med", "성장 정체",
            "매출·이익 성장률이 하위권. 재평가를 이끌 동력이 약하다.",
            f"성장 {v['s_grow']:.0f}점")
    # 4) 수급 이탈
    if v.get("s_flow", 50) <= 35:
        add("high", "외국인 이탈",
            "외국인이 최근 20일 순매도 쪽. 한국 대형주에서 외국인 방향은 가장 빠른 경고다.",
            f"수급 {v['s_flow']:.0f}점")
    # 5) 추격매수 위험
    if (ctx.get("timing") or 50) <= 30 and (ctx.get("dd") is None or ctx["dd"] >= -5):
        add("med", "고점권 진입",
            "이미 많이 올라 신고가권. 지금 사면 조정 시작점에 물릴 수 있다.",
            f"타이밍 {ctx.get('timing', 0):.0f}/100 · 60일고점대비 {ctx.get('dd')}%")
    # 6) 떨어지는 칼날
    if (ctx.get("dd") is not None and ctx["dd"] <= -20) and v.get("s_mom", 50) <= 35:
        add("high", "하락추세 진행",
            "많이 빠졌지만 추세도 같이 무너졌다. '싸졌다'가 아니라 '계속 빠지는 중'일 수 있다.",
            f"60일고점대비 {ctx['dd']}% · 모멘텀 {v['s_mom']:.0f}점")
    # 7) 공매도 부담
    if (ctx.get("short_rank") or 0) >= 80:
        add("med", "공매도 집중",
            "하락에 베팅한 물량이 100종목 중 상위권. 반대편이 강하게 보고 있다.",
            f"공매도 잔고비중 {ctx.get('short_pct')}% (백분위 {ctx.get('short_rank')})")
    # 8) 변동성 과대 — 손절이 먼저 맞을 위험
    if (ctx.get("vol_rank") or 0) >= 80:
        add("med", "변동성 과대",
            "출렁임이 커서 논리가 맞아도 -10% 손절에 먼저 걸려 털릴 수 있다.",
            f"연변동성 {ctx.get('vol')}% (백분위 {ctx.get('vol_rank')})")
    # 9) 하락장 취약 (하락베타)
    if (ctx.get("down_beta") or 0) >= 1.3:
        add("med", "하락장 취약",
            "시장이 내릴 때 더 많이 내리는 종목. 시장 리스크가 그대로 증폭된다.",
            f"하락베타 {ctx.get('down_beta')} (시장 -10%면 대략 {ctx['down_beta']*-10:.0f}%)")
    # 10) 통계적 엣지 부재
    pr = ctx.get("prob")
    if pr is None:
        add("high", "표본 부족",
            "과거 유사 국면 표본이 부족해 확률을 말할 수 없다. 이 종목은 '모른다'가 정답.",
            "유사사례 10회 미만")
    else:
        bwin = pr.get("base_win")
        if pr.get("edge_pp") is not None and pr["edge_pp"] <= 0:
            add("high", "엣지 없음",
                "이 종목의 유사국면 승률이 '아무거나 산 경우'의 기준선보다 낮다. "
                "점수가 높다고 승률까지 높은 건 아니다.",
                f"승률 {pr['win']}% vs 기준선 {bwin}% ({pr['edge_pp']:+.1f}%p)")
        elif not pr.get("proven"):
            add("med", "승률 미확정",
                "표본이 얇아 신뢰구간 하한이 기준선 아래다 — 우위가 있다고 통계적으로 "
                "단정할 수는 없다(이 데이터로는 어느 종목도 대부분 단정 못 한다).",
                f"승률 {pr['win']}% (95% 구간 {pr['win_lo']}~{pr['win_hi']}%, "
                f"독립표본 {pr['eff_n']:.0f}회, 기준선 {bwin}%)")
        if pr["ev"] <= 0:
            add("high", "기대값 음수",
                "비용·손절을 반영하면 기대수익이 0 이하. 반복하면 잃는 베팅이다.",
                f"기대 {pr['ev']:+.2f}% (비용 {pr['cost']}% 차감 후)")
        if pr["p_stop"] >= 25:
            add("med", "손절 빈발",
                "과거 유사 국면에서 30일 안에 -10%를 맞은 비율이 높다. 잦은 손절을 각오해야 한다.",
                f"손절 발동 {pr['p_stop']}%")
    # 11) 뉴스 악재
    for nw in (ctx.get("news_flags") or [])[:2]:
        add("high", "뉴스 악재", nw["text"], nw["title"][:60])
    # 12) 섹터 쏠림
    if ctx.get("sector_conc"):
        add("med", "포트폴리오 쏠림",
            "이미 같은 업종 비중이 높다. 한 방향으로 같이 움직여 분산 효과가 사라진다.",
            ctx["sector_conc"])

    # 반대논리 강도 0~100
    W = {"high": 22, "med": 11, "low": 5}
    bear_score = min(100, sum(W[r["sev"]] for r in risks))

    # 무효화 조건 — '내가 틀렸다'고 인정할 선을 미리 못박는다.
    # 공통 문구는 meta['invalidation']에 한 번만 두고, 종목별로는 값(손절가·핵심논리)만 넘긴다.
    if meta is not None:
        meta.setdefault("invalidation", INVALIDATION)
    inval = {"stop": ctx.get("stop_buy"), "bull_first": ctx.get("bull_first")}
    if meta is None:
        inval = [t.format(stop=f"{ctx.get('stop_buy') or 0:,}", bull=ctx.get("bull_first") or "")
                 for t in INVALIDATION]

    # 하락 경로
    downside = None
    if pr:
        downside = {"stop_price": ctx.get("stop_buy"), "p_stop": pr["p_stop"],
                    "p_down5": pr["p_down5"],
                    "worst": ctx["analog"]["worst"] if ctx.get("analog") else None,
                    "p25_price": ctx.get("tgt_lo"),
                    "p25": ctx["exp"]["p25"] if ctx.get("exp") else None}

    if risks:
        top = [r for r in risks if r["sev"] == "high"][:2] or risks[:2]
        summary = ("반대 논리: " + " / ".join(f"<b>{r['tag']}</b> — {rtext(r)}" for r in top))
    else:
        summary = ("자동 점검에서 걸린 반대 논리가 없다. 다만 '체크리스트에 없는 위험'"
                   "(규제·경영진·업황 전환)은 뉴스로 직접 확인해야 한다.")

    return {"summary": summary, "risks": risks, "bear_score": bear_score,
            "invalidation": inval, "downside": downside}


# ══════════════════════════════════════════════════════════════
# 시장 상황별 시나리오 (베타 기반)
# ══════════════════════════════════════════════════════════════
MSCEN_LABELS = ["시장 급락 (-10%↓)", "시장 조정 (-10~-5%)", "시장 약보합 (-5~0%)",
                "시장 강보합 (0~+5%)", "시장 상승 (+5~+10%)", "시장 급등 (+10%↑)"]
MSCEN_MOVES = [-15, -7.5, -2.5, 2.5, 7.5, 15]


def market_scenarios(ctx, regime, meta=None):
    """
    시장이 30일 뒤 m% 움직였을 때 이 종목의 기대수익.
      기대 = (유사사례 중앙값) + b × (m − 시장 중앙수익)
      b = 시장이 중앙값보다 나쁠 때는 하락베타, 좋을 때는 일반 베타
    확률은 '현재 국면에서 실제로 그 구간이 나온 빈도'를 붙인다.
    """
    if not ctx.get("analog") or not regime:
        return None
    beta = ctx.get("beta")
    if beta is None:
        return None
    dbeta = ctx.get("down_beta") or beta
    base = ctx["analog"]["med"]                    # 유사사례 30일 중앙수익(%)
    m_med = regime.get("mkt_med_all", 1.0)         # 시장 30일 중앙수익(%)
    probs = regime.get("mkt_probs_now") or regime.get("mkt_probs_all")
    # 버킷 중앙값과 시나리오를 맞춘다: -99~-10, -10~-5, -5~0, 0~5, 5~10, 10~99
    if meta is not None:
        meta.setdefault("mscen_labels", MSCEN_LABELS)
        meta.setdefault("mscen_moves", MSCEN_MOVES)
    out = []
    for k, m in enumerate(MSCEN_MOVES):
        b = dbeta if m < m_med else beta
        exp = base + b * (m - m_med)
        price = int(round(ctx["price"] * (1 + exp / 100)))
        row = {"beta": round(b, 2), "exp": round(exp, 1), "price": price,
               "prob": (probs[k] if probs and k < len(probs) else None),
               "stop_hit": bool(exp <= -10)}
        if meta is None:
            row["label"], row["move"] = MSCEN_LABELS[k], m
        out.append(row)
    return {"rows": out, "mkt_med": m_med,
            "regime": regime.get("current", {}).get("name", "")}


# ══════════════════════════════════════════════════════════════
# 최종 판단 (신뢰도 등급)
# ══════════════════════════════════════════════════════════════
def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# 확신도 구성요소 설명 (100종목 공통 → meta에 한 번만 실린다)
VERDICT_NOTES = {
    "초과 승률": "이 종목 유사국면 승률 − 전체 기준선(아무 종목이나 아무 날 샀을 때의 승률). "
                 "기준선보다 높아야 비로소 엣지다.",
    "기대수익": "손절(-10%)·왕복비용 0.3%를 반영한 30일 기대값.",
    "표본 두께": "겹치는 표본 중 실제로 겹치지 않는 관측 횟수. 얇으면 숫자를 믿을 수 없다.",
    "반대 논리": "자동 체크리스트에 걸린 하락 논리의 가중합(치명 22 · 주의 11점).",
    "시장 국면": "지금 국면에서 상위10 전략의 과거 30일 승률.",
}


def verdict(ctx, bull, bear, regime, base_win=None, meta=None):
    """
    확률·반대논리·시장국면을 합쳐 0~100 확신도와 등급을 낸다.

    등급이 뜻하는 것 / 뜻하지 않는 것 (중요)
      뜻함   : 같은 날 100종목을 같은 잣대로 줄 세웠을 때 근거가 두꺼운 쪽인가.
      뜻 안함: '통계적으로 오른다고 입증됐다'. 개별 종목 30일 승률은 이 데이터로
               유의하게 입증되지 않는다(prob.proven 필드가 그 사실을 그대로 표시한다).
    """
    pr = ctx.get("prob")
    conf = 50.0
    parts = []
    bw = base_win if base_win is not None else 55.0

    if meta is not None:
        meta.setdefault("verdict_notes", VERDICT_NOTES)

    def part(k, v, d):
        p = {"k": k, "v": v, "d": round(d, 1)}
        if meta is None:
            p["note"] = VERDICT_NOTES.get(k, "")
        parts.append(p)

    if pr:
        edge = (pr.get("edge_pp") if pr.get("edge_pp") is not None else pr["win"] - bw)
        d = _clamp(edge * 1.8, -22, 22)
        conf += d
        part("초과 승률", f"{edge:+.1f}%p", d)
        d = _clamp(pr["ev"] * 2.5, -15, 15)
        conf += d
        part("기대수익", f"{pr['ev']:+.2f}%", d)
        d = _clamp((pr["eff_n"] - 8) * 1.0, -12, 12)
        conf += d
        part("표본 두께", f"독립 {pr['eff_n']:.0f}회", d)
    else:
        conf -= 22
        part("표본 두께", "부족", -22.0)

    d = -bear["bear_score"] * 0.32
    conf += d
    n_high = sum(1 for r in bear["risks"] if r["sev"] == "high")
    part("반대 논리", f"{len(bear['risks'])}건(치명 {n_high})", d)

    rw = None
    if regime:
        cur = regime.get("current", {}).get("name")
        rw = next((h.get("win") for h in regime.get("history", []) if h["name"] == cur), None)
    if rw is not None:
        d = _clamp((rw - 55) * 0.6, -8, 8)
        conf += d
        part("시장 국면", f"{regime['current']['name']} 승률 {rw:.0f}%", d)

    conf = _clamp(conf, 0, 100)

    if conf >= 68:
        grade, gtxt, act = "A", "근거가 두껍다", "규칙대로 정상 비중"
    elif conf >= 56:
        grade, gtxt, act = "B", "근거가 있으나 확정적이진 않다", "정상 비중, 손절 엄수"
    elif conf >= 44:
        grade, gtxt, act = "C", "근거가 얇다", "소액(정상의 1/2 이하) 또는 관망"
    else:
        grade, gtxt, act = "D", "근거가 약하거나 반대논리가 크다", "보류 권장"

    # ── 매매 규칙이 등급을 이긴다 ────────────────────────────
    # 손절/매도 신호가 난 종목에 '정상 비중'을 권하면 규칙이 무너진다.
    # 등급은 어디까지나 '근거의 두께'이고, 행동은 규칙이 정한다.
    sig = ctx.get("signal") or ""
    if "손절" in sig:
        mode, act = "sell", "규칙상 손절 — 논리와 무관하게 청산. 등급은 참고용."
    elif "매도" in sig:
        mode, act = "sell", "매도 신호(순위 이탈 + 최소보유 충족) — 청산."
    elif "보류" in sig:
        mode, act = "wait", "외국인 순매도 중 — 신규매수 보류. 수급 전환 시 재검토."
    elif sig in ("🟢유지", "⏳보유"):
        mode = "hold"
        act = ("보유 유지 — 감시가 이탈 시에만 청산." if grade in ("A", "B")
               else f"보유 중이나 근거가 얇아졌다({grade}등급). 추가매수는 하지 말 것.")
    else:
        mode = "buy"

    # 한 줄 결론
    if pr:
        line = (f"승률 {pr['win']}%(기준선 {bw}% 대비 {pr['edge_pp']:+.1f}%p) · "
                f"기대 {pr['ev']:+.2f}% · 손절확률 {pr['p_stop']}% · "
                f"반대논리 {len(bear['risks'])}건 → <b>{grade}등급</b>, {act}")
    else:
        line = f"확률을 계산할 표본이 없다 → <b>{grade}등급</b>, {act}"

    return {"conf": round(conf), "grade": grade, "grade_text": gtxt,
            "action": act, "mode": mode, "line": line, "parts": parts,
            "regime_win": rw, "base_win": bw,
            "proven": bool(pr and pr.get("proven"))}
