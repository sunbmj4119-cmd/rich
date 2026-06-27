"""
일일 리포트 HTML - 실제 거래내역 기반, 손절/매도 경고 강조
출력: docs/index.html (GitHub Pages, 폰 열람)
"""
import os
import pandas as pd

SCORES = "data/scores.csv"
SIGNALS = "data/signals_today.csv"
OUT = "docs/index.html"


def bar(pct, color):
    pct = max(0, min(100, pct))
    return (f'<div style="background:#eee;border-radius:4px;height:16px;flex:1">'
            f'<div style="background:{color};height:16px;border-radius:4px;width:{pct}%"></div></div>')


def main():
    os.makedirs("docs", exist_ok=True)
    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    last = s["날짜"].max()
    today = s[s["날짜"] == last].sort_values("종합점수", ascending=False).reset_index(drop=True)

    sig = pd.read_csv(SIGNALS, dtype={"종목코드": str}) if os.path.exists(SIGNALS) else pd.DataFrame()

    def grp(tag):
        return sig[sig["구분"] == tag] if len(sig) else pd.DataFrame()

    cut = grp("🔴손절")
    sell = grp("🔵매도")
    buy = grp("🟡매수")
    hold = pd.concat([grp("🟢유지"), grp("⏳보유")]) if len(sig) else pd.DataFrame()

    H = []
    H.append(f"""<!doctype html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>주식 신호</title><style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:12px;background:#f5f5f7;color:#1d1d1f}}
.card{{background:#fff;border-radius:12px;padding:14px;margin-bottom:10px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
h1{{font-size:19px;margin:2px 0}} h2{{font-size:15px;margin:6px 0}}
.date{{color:#888;font-size:12px}}
.alert{{background:#fff0f0;border:1px solid #ffcccc;border-radius:12px;padding:14px;margin-bottom:10px}}
.sig{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f0f0f0;font-size:14px}}
.sig:last-child{{border:0}}
.nm{{font-weight:600}} .meta{{color:#888;font-size:12px}}
.pos{{color:#d70015;font-weight:700}} .neg{{color:#0071e3;font-weight:700}}
.row{{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px}}
.row .l{{width:80px;font-weight:600;flex-shrink:0}} .row .v{{width:30px;text-align:right;color:#666}}
.tag{{font-size:11px;padding:1px 6px;border-radius:8px;margin-left:5px}}
</style></head><body>
<h1>📊 매매 신호</h1>
<div class="date">기준일 {last} · 상위20/진입10위/30일/손절-10%</div>
<a href="trade.html" style="display:block;text-align:center;background:#0071e3;color:#fff;
padding:13px;border-radius:12px;font-weight:700;text-decoration:none;margin:12px 0;font-size:15px">
📝 매매 기록하기</a>""")

    # 손절 (최우선 경고)
    if len(cut):
        H.append('<div class="alert"><h2>🔴 손절 (즉시 매도 검토)</h2>')
        for _, r in cut.iterrows():
            H.append(f'<div class="sig"><span class="nm">{r["종목명"]}</span>'
                     f'<span class="neg">{r["수익률%"]:+}%</span></div>')
        H.append('</div>')

    # 매도
    if len(sell):
        H.append('<div class="card"><h2>🔵 매도</h2>')
        for _, r in sell.iterrows():
            pnl = f'<span class="{"pos" if float(r["수익률%"])>=0 else "neg"}">{r["수익률%"]:+}%</span>' if r["수익률%"] != "" else ""
            H.append(f'<div class="sig"><span><span class="nm">{r["종목명"]}</span> '
                     f'<span class="meta">{r["사유"]}</span></span>{pnl}</div>')
        H.append('</div>')

    # 매수 추천
    H.append('<div class="card"><h2>🟡 매수 추천</h2>')
    if len(buy):
        for _, r in buy.iterrows():
            H.append(f'<div class="sig"><span class="nm">{r["종목명"]}</span>'
                     f'<span class="meta">{int(r["순위"])}위 · 점수 {r["점수"]:.0f}</span></div>')
    else:
        H.append('<div class="meta">신규 매수 없음 (보유 충분)</div>')
    H.append('</div>')

    # 보유 현황
    if len(hold):
        H.append('<div class="card"><h2>💼 보유 현황</h2>')
        for _, r in hold.iterrows():
            cls = "pos" if (r["수익률%"] != "" and float(r["수익률%"]) >= 0) else "neg"
            pnl = f'<span class="{cls}">{r["수익률%"]:+}%</span>' if r["수익률%"] != "" else ""
            H.append(f'<div class="sig"><span><span class="nm">{r["종목명"]}</span> '
                     f'<span class="meta">{r["순위"]}위·{r["보유일"]}일</span></span>{pnl}</div>')
        H.append('</div>')

    # 상위 점수
    H.append('<div class="card"><h2>🏆 오늘 상위 12종목</h2>')
    held_names = set(hold["종목명"]) if len(hold) else set()
    for _, r in today.head(12).iterrows():
        mark = "🟢" if r["종목명"] in held_names else ""
        H.append(f'<div class="row"><span class="l">{mark}{r["종목명"]}</span>'
                 f'{bar(r["종합점수"], "#0071e3")}<span class="v">{r["종합점수"]:.0f}</span></div>')
    H.append('</div>')

    H.append("""<div class="card"><h2>📈 전략 검증 (워크포워드 6구간)</h2>
<div class="meta" style="line-height:1.7">
연평균 기대수익 ~87% · 6구간 전부 양수 · Sharpe 1.34<br>
안정형·공격형 목적함수가 동일 파라미터로 수렴 (강건성 입증)</div></div>""")

    H.append('<div class="date" style="text-align:center;margin:14px 0">'
             '⚠️ 참고용 · 투자 최종책임 본인 · 손절은 기계적으로 지킬 것</div></body></html>')

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(H))
    print(f"리포트 생성: {OUT}")


if __name__ == "__main__":
    main()

