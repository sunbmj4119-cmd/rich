"""
일일 리포트 HTML 생성 - 폰에서 보기 좋은 단일 파일
- 오늘의 매수/매도/유지 신호 (signal.py 결과)
- 상위 점수 종목 막대그래프
- 백테스트 요약 (validation/live 핵심지표)
- 보유 포트폴리오 현황
출력: docs/index.html  (GitHub Pages로 폰에서 바로 열람)
외부 라이브러리 없음(순수 HTML+CSS+인라인 차트)
"""
import os
import json
import pandas as pd

SCORES = "data/scores.csv"
SIGNALS = "data/signals_today.csv"
PORT = "data/portfolio.json"
OUTDIR = "docs"
OUT = "docs/index.html"

TOP_N = int(os.environ.get("TOP_N", "20"))


def bar(pct, color):
    pct = max(0, min(100, pct))
    return (f'<div style="background:#eee;border-radius:4px;height:18px;width:100%">'
            f'<div style="background:{color};height:18px;border-radius:4px;'
            f'width:{pct}%"></div></div>')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    last = s["날짜"].max()
    today = s[s["날짜"] == last].sort_values("종합점수", ascending=False).reset_index(drop=True)

    sig = pd.read_csv(SIGNALS, dtype={"종목코드": str}) if os.path.exists(SIGNALS) else pd.DataFrame()
    port = json.load(open(PORT, encoding="utf-8")) if os.path.exists(PORT) else {}

    buys = sig[sig["구분"] == "매수"] if len(sig) else pd.DataFrame()
    sells = sig[sig["구분"] == "매도"] if len(sig) else pd.DataFrame()
    holds = sig[sig["구분"].str.startswith("유지")] if len(sig) else pd.DataFrame()

    html = []
    html.append(f"""<!doctype html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>주식 점수 리포트</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:12px;background:#f5f5f7;color:#1d1d1f}}
.card{{background:#fff;border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
h1{{font-size:20px;margin:4px 0}} h2{{font-size:16px;margin:8px 0;border-left:4px solid #0071e3;padding-left:8px}}
.date{{color:#888;font-size:13px}}
.buy{{color:#d70015;font-weight:700}} .sell{{color:#0071e3;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{padding:6px 4px;border-bottom:1px solid #eee;text-align:left}}
th{{color:#888;font-weight:600}}
.tag{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600}}
.tbuy{{background:#ffe5e5;color:#d70015}} .tsell{{background:#e5f0ff;color:#0071e3}}
.thold{{background:#eee;color:#666}}
.row{{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px}}
.row .nm{{width:88px;flex-shrink:0;font-weight:600}}
.row .sc{{width:34px;text-align:right;color:#666}}
.pos{{color:#d70015}} .neg{{color:#0071e3}}
</style></head><body>
<h1>📊 주식 점수 리포트</h1>
<div class="date">기준일 {last} · 상위{TOP_N} / 40일 보유 전략</div>
""")

    # 신호 요약
    html.append('<div class="card"><h2>🔔 내일 매매 신호</h2>')
    if len(buys):
        html.append('<div style="margin-bottom:8px"><b class="buy">매수</b> ')
        html.append(" · ".join(f"{r['종목명']}({int(r['현재순위'])}위)" for _, r in buys.iterrows()))
        html.append('</div>')
    if len(sells):
        html.append('<div style="margin-bottom:8px"><b class="sell">매도</b> ')
        html.append(" · ".join(f"{r['종목명']}" for _, r in sells.iterrows()))
        html.append('</div>')
    if not len(buys) and not len(sells):
        html.append('<div style="color:#888">신규 매수/매도 신호 없음 — 보유 유지</div>')
    html.append('</div>')

    # 상위 종목 막대그래프
    html.append('<div class="card"><h2>🏆 상위 15 종목 점수</h2>')
    for _, r in today.head(15).iterrows():
        held = "🟢" if r["종목코드"] in port else ""
        html.append(f'<div class="row"><span class="nm">{held}{r["종목명"]}</span>'
                    f'{bar(r["종합점수"], "#0071e3")}'
                    f'<span class="sc">{r["종합점수"]:.0f}</span></div>')
    html.append('<div style="font-size:12px;color:#888;margin-top:6px">🟢=현재 보유중</div></div>')

    # 보유 현황
    if len(holds) or len(sells):
        html.append('<div class="card"><h2>💼 보유 포트폴리오</h2><table>')
        html.append('<tr><th>종목</th><th>순위</th><th>보유일</th><th>수익률</th><th></th></tr>')
        allhold = pd.concat([holds, sells]) if len(sells) else holds
        for _, r in allhold.iterrows():
            pnl = r["수익률%"]
            cls = "pos" if (pnl != "" and float(pnl) >= 0) else "neg"
            pnls = f'<span class="{cls}">{float(pnl):+.1f}%</span>' if pnl != "" else "-"
            tag = '<span class="tag tsell">매도</span>' if r["구분"] == "매도" else ''
            html.append(f'<tr><td>{r["종목명"]}</td><td>{r["현재순위"]}</td>'
                        f'<td>{r["보유일"]}일</td><td>{pnls}</td><td>{tag}</td></tr>')
        html.append('</table></div>')

    # 모델 성과 (고정 — 백테스트 검증 결과)
    html.append("""<div class="card"><h2>📈 모델 검증 성과</h2>
<table>
<tr><th>구간</th><th>IC(N20)</th><th>5분위 스프레드</th><th>롱숏 Sharpe</th></tr>
<tr><td>검증 24~25</td><td>0.063</td><td>5.43%</td><td>1.47</td></tr>
<tr><td>실전 26~</td><td>0.072</td><td>10.86%</td><td>1.59</td></tr>
</table>
<div style="font-size:12px;color:#888;margin-top:6px">
IC 0.05↑·스프레드 양수 = 점수 높을수록 실제 더 오름. 시장 Sharpe 1.46 대비 우위.</div></div>""")

    html.append('<div class="date" style="text-align:center;margin:16px 0">'
                '⚠️ 투자 판단 참고용 · 최종 결정은 본인 책임</div>')
    html.append('</body></html>')

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("".join(html))
    print(f"리포트 생성: {OUT}")


if __name__ == "__main__":
    main()

