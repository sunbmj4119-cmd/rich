"""
일일 리포트 (매수적합도 BuyFit 중심) -> docs/index.html
- 핵심: '지금 사기 좋은 순서'(BuyFit)로 추천 재정렬, 점수순위와 나란히 표시
- 종목 터치 -> 큰 그래프 + 팩터 + 타이밍 분해 + 과거 유사사례 승률 + 예상 매도
docs/data.json 선행 필요 (build_data.py)
"""
import os, json

DATA="docs/data.json"; OUT="docs/index.html"


def main():
    os.makedirs("docs",exist_ok=True)
    d=json.load(open(DATA,encoding="utf-8"))
    m=d["market"]; items=d["items"]; payload=json.dumps(d,ensure_ascii=False)
    rb=None
    if os.path.exists("docs/robust.json"):
        try: rb=json.load(open("docs/robust.json",encoding="utf-8"))
        except Exception: rb=None
    bymap={i["code"]:i for i in items}
    pf=d.get("portfolio")
    buylist=[bymap[c] for c in m["buylist"] if c in bymap]
    cuts=[i for i in items if i["signal"]=="🔴손절"]
    sells=[i for i in items if i["signal"]=="🔵매도"]
    holds=[i for i in items if i["signal"] in ("🟢유지","⏳보유")]

    H=f"""<!doctype html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>매수적합도 분석</title><style>
:root{{--blue:#0071e3;--green:#34c759;--red:#ff3b30;--org:#ff9500;--pur:#af52de}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:14px;background:#f2f2f7;color:#1d1d1f;-webkit-text-size-adjust:100%}}
h1{{font-size:22px;margin:2px 0}} h2{{font-size:17px;margin:4px 0 10px}}
.date{{color:#888;font-size:13px;line-height:1.6}}
.card{{background:#fff;border-radius:16px;padding:16px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.alert{{background:#fff0f0;border:1.5px solid #ffcccc}}
.btn{{display:block;text-align:center;background:var(--blue);color:#fff;padding:15px;border-radius:14px;font-weight:700;text-decoration:none;margin:14px 0;font-size:16px}}
.bi{{display:flex;align-items:center;gap:11px;padding:14px 6px;border-bottom:1px solid #f0f0f0;cursor:pointer}}
.bi:last-child{{border:0}} .bi:active{{background:#f5f5f7}}
.brk{{width:34px;height:34px;border-radius:9px;color:#fff;font-weight:800;font-size:15px;display:flex;align-items:center;justify-content:center;flex-shrink:0;background:var(--green)}}
.brk.r2{{background:#5ac85f}}.brk.r3{{background:#7dd47f}}.brk.rx{{background:#c7c7cc}}
.bmid{{flex:1;min-width:0}}
.bn{{font-weight:700;font-size:16px}} .bsub{{font-size:12px;color:#888;margin-top:2px}}
.fit{{text-align:right;flex-shrink:0}}
.fitv{{font-size:21px;font-weight:800;color:var(--green)}} .fitk{{font-size:10px;color:#aaa}}
.chev{{color:#c8c8cd;font-size:18px}}
.pp{{color:#d70015;font-weight:700;font-size:14px}} .pn{{color:var(--blue);font-weight:700;font-size:14px}}
.item{{display:flex;align-items:center;gap:10px;padding:12px 6px;border-bottom:1px solid #f0f0f0;cursor:pointer}}
.item:last-child{{border:0}} .item:active{{background:#f5f5f7}}
.rk{{width:28px;height:28px;border-radius:50%;background:#eef;color:var(--blue);font-weight:800;font-size:12px;display:flex;align-items:center;justify-content:center;flex-shrink:0}}
.nm{{font-weight:700;font-size:15px;flex:1}} .sc{{font-size:19px;font-weight:800;color:var(--blue)}}
.hist{{display:flex;align-items:flex-end;gap:4px;height:110px;margin:12px 0 4px}}
.hcol{{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center}}
.hbar{{width:100%;border-radius:4px 4px 0 0;min-height:3px}}
.hnum{{font-size:10px;color:#666;margin-bottom:2px}} .hlab{{font-size:9px;color:#999;margin-top:3px}}
.note{{font-size:13px;color:#444;line-height:1.7;background:#fff;border-radius:12px;padding:12px;margin:8px 0}}
.leg{{font-size:12px;color:#888;line-height:1.7;margin-top:8px}}
.modal{{position:fixed;inset:0;background:rgba(0,0,0,.45);display:none;z-index:99;align-items:flex-end}}
.modal.on{{display:flex}}
.sheet{{background:#f2f2f7;width:100%;max-height:93vh;overflow-y:auto;border-radius:20px 20px 0 0;padding:18px 16px 44px;animation:up .25s}}
@keyframes up{{from{{transform:translateY(100%)}}to{{transform:translateY(0)}}}}
.shandle{{width:40px;height:5px;background:#ccc;border-radius:3px;margin:0 auto 14px}}
.dh{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px}}
.x{{background:#e5e5ea;border:0;border-radius:50%;width:32px;height:32px;font-size:16px;color:#666;flex-shrink:0}}
.stat{{display:flex;gap:8px;margin:10px 0}}
.stat>div{{flex:1;background:#fff;border-radius:12px;padding:12px 6px;text-align:center}}
.stat .v{{font-size:21px;font-weight:800}} .stat .k{{font-size:10px;color:#888;margin-top:2px}}
.bigbar-wrap{{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:14px}}
.bl{{width:64px;font-weight:600;flex-shrink:0;font-size:13px}}
.bigbar{{flex:1;background:#eee;border-radius:6px;height:22px;overflow:hidden}}
.bigbar>div{{height:22px}}
.bv{{width:30px;text-align:right;font-weight:700}} .bw{{width:40px;text-align:right;color:#bbb;font-size:12px}}
.range{{position:relative;height:50px;margin:16px 6px 6px}}
.rtrack{{position:absolute;top:20px;left:0;right:0;height:7px;background:linear-gradient(90deg,#ff3b30,#ffcc00,#34c759);border-radius:4px}}
.rdot{{position:absolute;top:12px;width:22px;height:22px;border-radius:50%;background:#fff;border:3px solid #1d1d1f;transform:translateX(-50%)}}
.rlab{{position:absolute;top:32px;font-size:10px;color:#666;transform:translateX(-50%);white-space:nowrap}}
.warn{{font-size:11px;color:#8a7;background:#fffbe6;border-radius:8px;padding:9px;margin-top:8px;line-height:1.6}}
svg{{display:block;background:#fafafa;border-radius:8px}}
</style></head><body>
<h1>🎯 매수적합도 분석</h1>
<div class="date">기준일 {m['date']} · <b>BuyFit</b> = 종합점수 + 진입타이밍(평균회귀)을 합친 '지금 사기 좋은 순서'<br>종목을 터치하면 근거·그래프·승률을 봅니다</div>
<a href="trade.html" class="btn">📝 매매 기록하기</a>
"""

    def small_row(i,pnl=False):
        p=""
        if pnl and i["pnl"] not in("",None):
            cls="pp" if float(i["pnl"])>=0 else "pn"; p=f'<span class="{cls}">{float(i["pnl"]):+}%</span>'
        return (f'<div class="item" onclick="openD(\'{i["code"]}\')"><div class="rk">{i["rank"]}</div>'
                f'<span class="nm">{i["name"]}</span>{p}<span class="sc">{i["score"]:.0f}</span><span class="chev">›</span></div>')

    def hold_row(i):
        p=""
        if i["pnl"] not in ("",None):
            cls="pp" if float(i["pnl"])>=0 else "pn"; p=f'<span class="{cls}">{float(i["pnl"]):+}%</span>'
        bits=[]
        if i.get("guard_gap") is not None:
            bits.append(f'손절까지 {abs(i["guard_gap"]):.1f}%')
        if i.get("dmin_remain") is not None:
            bits.append("매도가능" if i["dmin_remain"]<=0 else f'매도가능 D-{i["dmin_remain"]}({i.get("dmin_date","")})')
        sub=" · ".join(bits)
        return (f'<div class="item" onclick="openD(\'{i["code"]}\')"><div class="rk">{i["rank"]}</div>'
                f'<div style="flex:1;min-width:0"><div style="font-weight:700;font-size:15px">{i["name"]}</div>'
                f'<div style="font-size:11px;color:#999;margin-top:1px">{sub}</div></div>'
                f'{p}<span class="sc">{i["score"]:.0f}</span><span class="chev">›</span></div>')

    if cuts:
        H+='<div class="card alert"><h2>🔴 손절 (즉시 검토)</h2>'+"".join(small_row(i,True) for i in cuts)+'</div>'
    if sells:
        H+='<div class="card"><h2>🔵 매도</h2>'+"".join(small_row(i,True) for i in sells)+'</div>'

    # 💼 내 계좌 요약 + 최악의 날 리스크
    if pf:
        up=pf["upnl"]; upc=pf["upnl_pct"]
        upcol="#d70015" if up>=0 else "#0071e3"   # 한국식: 빨강=이익
        eqdiv='<div id="eqchart" style="margin-top:10px"></div>' if len(pf.get("equity",[]))>=2 else ''
        real=pf.get("realized",0); tp=pf.get("total_pnl",up)
        real_html=""
        if real:
            rcol="#d70015" if real>=0 else "#0071e3"; tcol="#d70015" if tp>=0 else "#0071e3"
            real_html=(f'<div class="stat" style="margin-top:6px">'
                f'<div><div class="v" style="color:{rcol}">{"+" if real>=0 else ""}{real:,}</div><div class="k">실현손익(원)</div></div>'
                f'<div><div class="v" style="color:{tcol}">{"+" if tp>=0 else ""}{tp:,}</div><div class="k">총손익(실현+평가)</div></div>'
                f'<div></div></div>')
        sec_html=""
        if pf.get("sectors"):
            chips="".join(f'<span style="display:inline-block;background:#eef;color:#0071e3;border-radius:8px;padding:3px 9px;margin:2px 3px 2px 0;font-size:12px;font-weight:600">{sc["name"]} {sc["pct"]:.0f}%</span>' for sc in pf["sectors"])
            approx="업종은 KRX 분류 기반 자동 근사(참고용)."
            top=pf.get("top_sector_pct",0); nn=pf.get("n",0)
            if top>=40 and nn>=3:
                cwarn=f'<div class="warn" style="margin-top:6px">⚠ <b>{pf["top_sector"]}</b> 비중 {top:.0f}% — 한 업종에 쏠려 있습니다. 같은 업종은 함께 움직이기 쉬우니 <b>분산</b>을 확인하세요. ({approx})</div>'
            elif top>=50:
                cwarn=f'<div class="warn" style="margin-top:6px">보유 종목수가 적어 특정 업종(<b>{pf["top_sector"]}</b>) 비중이 높게 잡힙니다. 늘릴 때 분산 고려. ({approx})</div>'
            else:
                cwarn=f'<div class="leg" style="margin-top:4px;color:#aaa">{approx}</div>'
            sec_html=f'<div style="margin-top:10px"><div class="leg" style="margin:0 0 4px">📦 업종 편중 (평가액 기준)</div>{chips}{cwarn}</div>'
        H+=f"""<div class="card"><h2>💼 내 계좌</h2>
<div class="stat">
  <div><div class="v" style="color:{upcol}">{'+' if up>=0 else ''}{up:,}</div><div class="k">평가손익 ({'+' if upc>=0 else ''}{upc}%)</div></div>
  <div><div class="v">{pf['value']:,}</div><div class="k">평가액(원)</div></div>
  <div><div class="v" style="color:#999">{pf['invested']:,}</div><div class="k">원금(원)</div></div>
</div>
<div class="stat" style="margin-top:6px">
  <div style="background:#fff0f0"><div class="v" style="color:#ff3b30">{pf['worst']:,}</div><div class="k">모든 손절 동시발동 시(원)</div></div>
  <div style="background:#fff0f0"><div class="v" style="color:#ff3b30">{pf['worst_pct']}%</div><div class="k">최악의 날 손실률</div></div>
  <div><div class="v">{pf['n']}</div><div class="k">보유 종목</div></div>
</div>{real_html}{sec_html}{eqdiv}
<div class="leg"><b>최악의 날</b> = 지금 걸어둔 감시가(손절·트레일 중 먼저 닿는 값)가 전 종목 동시 체결될 때의 추가손실. 실제 동시발동은 드물지만 <b>감내 가능한 최대 손실</b>을 미리 확인하세요.</div></div>"""

    # ★ 핵심: 매수적합도 추천
    H+='<div class="card"><h2>🟢 지금 사기 좋은 순서 (BuyFit)</h2>'
    H+='<div class="leg" style="margin-top:0;margin-bottom:6px">점수 상위 풀에서, 최근 조정받아 반등 여지가 큰 종목을 앞으로. (백테스트에서 <b>관찰된</b> 단기 평균회귀 · 생존편향 미보정 → 아래 한계 참고)</div>'
    for k,i in enumerate(buylist):
        cls="brk" if k==0 else ("brk r2" if k==1 else ("brk r3" if k==2 else "brk rx"))
        tcol = "#34c759" if i["timing"]>=66 else ("#ff9500" if i["timing"]>=33 else "#ff3b30")
        ddtxt = f'{i["dd"]}%' if i["dd"] is not None else "-"
        H+=(f'<div class="bi" onclick="openD(\'{i["code"]}\')">'
            f'<div class="{cls}">{k+1}</div>'
            f'<div class="bmid"><div class="bn">{i["name"]}</div>'
            f'<div class="bsub">점수 {i["score"]:.0f} (전체 {i["rank"]}위) · 60일고점대비 {ddtxt} · '
            f'<span style="color:{tcol};font-weight:700">타이밍 {i["timing"]:.0f}</span></div></div>'
            f'<div class="fit"><div class="fitv">{i["buyfit"]:.0f}</div><div class="fitk">BuyFit</div></div>'
            f'<span class="chev">›</span></div>')
    H+='<div class="leg">BuyFit(0~100): 펀더멘털 점수 + 진입타이밍 종합. 높을수록 "지금" 매수 적합.</div></div>'

    if holds:
        H+='<div class="card"><h2>💼 보유 현황</h2>'+"".join(hold_row(i) for i in holds)+'</div>'

    H+='''<div class="card"><h2>🔎 전체 종목 (검색·정렬)</h2>
<input id="q" placeholder="종목명 검색 (예: 삼성)" oninput="renderStocks()" style="width:100%;padding:11px;font-size:15px;border:1px solid #ddd;border-radius:10px;margin-bottom:8px">
<div id="sortBtns" style="display:flex;gap:6px;margin-bottom:6px"></div>
<div id="stockList"></div></div>'''

    H+=f"""<div class="card"><h2>📊 시장 점수 분포</h2>
<div class="date">평균 {m['avg']} · 중앙값 {m['median']} · 최고 {m['max']}</div>
<div class="hist" id="mhist"></div>
<div class="leg">100종목 점수 분포. 상위 추천권은 보통 65점+.</div></div>"""

    if rb:
        scen_rows="".join(
          f'<div class="bigbar-wrap"><span class="bl" style="width:auto;flex:1;font-size:12px">{x["name"]}</span>'
          f'<span class="bv" style="width:auto;color:{"#34c759" if x["ann"]>0 else "#ff3b30"}">{x["ann"]:+.0f}%</span></div>'
          for x in rb["scenarios"])
        pit_line = ("point-in-time 유니버스로 생존편향을 실제 보정했습니다."
                    if rb.get("pit")
                    else "⚠️ 생존편향은 <b>비용·최근상장 제외까지만</b> 보정됐고, 진짜 상장폐지 종목은 데이터에 없어 <b>반영 못 했습니다</b>. 실제 기대는 아래 수치보다 더 낮습니다.")
        surv = rb.get("surv")
        surv_html = ""
        if surv:
            surv_html = (f'<div class="note" style="background:#fff6f6;margin-top:6px">🪦 <b>생존편향 정량화</b>: 이 구간({surv["years"]:.0f}년) KOSPI에서 '
                         f'<b>{surv["n_kospi"]}건</b>(연 {surv["rate"]:.1f}건) 상장폐지됐지만 백테스트엔 <b>없습니다</b>. '
                         f'가정(상폐 평균 -60%) 반영 시 추가 <b>-{surv["haircut"]:.1f}%p/년</b> → <b style="color:#d70015">초보수 기대 연 {surv["ann_after"]:+.0f}%</b>. '
                         f'<span style="color:#999">(haircut은 가정에 따른 근사치)</span></div>')
        H+=f"""<div class="card"><h2>🔬 현실적 기대수익 (비용 보정·생존편향 일부)</h2>
<div class="note" style="background:#f0f7ff">백테스트 숫자를 그대로 믿지 않기 위해 <b>거래비용을 현실화하고 생존편향을 일부</b> 보정한 결과:<br><br>
이상적 가정 연 <b>{rb['base_ann']:+.0f}%</b> → 보수적 가정 연 <b style="color:#0071e3">{rb['concl_ann']:+.0f}%</b> (비용·편향으로 <b>{rb['erosion']:.0f}%p</b> 증발)</div>
<div class="date" style="margin:8px 0 4px">가정별 연율 수익(보수일수록 아래):</div>
{scen_rows}
{surv_html}
<div class="warn">※ {pit_line}<br>이 표본은 <b>강세장 비중이 큰 2018~2026</b> 구간이니 <b>실전 기대는 더 보수적으로</b> 잡으세요. 상장폐지 {rb.get('delisted_n',0):,}종목은 애초에 데이터에 없습니다.</div></div>"""
    else:
        H+='<div class="card"><h2>🔬 전략 검증</h2><div class="note">보완 백테스트 미실행. Actions의 robust-backtest를 한 번 돌리면 생존편향·비용 보정 기대수익이 여기 표시됩니다.</div></div>'
    H+="""<div class="card"><h2>📚 방법론 · 이 추천의 근거</h2>
<details><summary style="font-size:14px;color:#0071e3;cursor:pointer;font-weight:600">계산 방식 전체 보기 (전문가용)</summary>
<div class="note" style="margin-top:8px;line-height:1.9">
<b>1) 미래참조 차단</b> — 재무는 공시 '사용가능일' 기준 as-of(backward) 조인으로 결합해, 그 시점에 알 수 없는 정보로 과거를 채점하지 않습니다. 신호는 t일 종가로 만들어 <b>t+1일 체결</b>을 가정(실전과 동일).<br><br>
<b>2) 팩터 점수</b> — 매 거래일 100종목을 <b>횡단면 백분위(0~100)</b>로 표준화(학술 표준 cross-sectional rank) 후 가중합: 가치0.32·수익성0.22·수급0.20·성장0.18·모멘텀0.08. 가치(Fama-French HML)·수익성(RMW/q팩터 ROE)·외국인수급(정보우위)·모멘텀(Jegadeesh-Titman)은 <b>학술적으로 초과수익이 보고된</b> 팩터이고, 성장(이익모멘텀)은 보조로 소가중합니다. ※IC 등 검증치는 이 저장소가 <b>과거(2024-25) 표본에서 추정</b>한 값으로 매일 재검증되지는 않습니다.<br><br>
<b>3) 매수 타이밍(BuyFit)</b> — z(종합점수) + 0.25×타이밍, 타이밍 = 0.6·(-z 60일낙폭) + 0.4·(-z 20일수익). 점수가 높으면서 <b>최근 눌린</b> 종목을 앞세우는 단기 평균회귀. λ=0.25는 워크포워드로 과최적화를 피해 선택.<br><br>
<b>4) 기대손익</b> — 미래를 점치지 않고, <b>과거에 점수·타이밍이 비슷했던 실제 사례</b>들의 30일 뒤 수익 분포(승률·중앙값·사분위)를 경험적 사전분포로 제시합니다. ※이 표본은 <b>겹치는 기간</b>(인접일 30일 창이 29/30 중복)이라 표시된 '표본 N'보다 <b>독립 관측치는 적고</b>, <b>생존종목만</b> 포함되며 진입은 당일 종가 기준이라 실제 t+1 체결보다 다소 낙관적일 수 있습니다. <b>점추정이 아닌 분포·불확실성</b>으로 보세요.<br><br>
<b>5) 리스크 규칙</b> — 손절 -10%, 트레일 -8%(고점 대비), 최소 30거래일 보유, 외국인 순매도 종목 매수보류.<br><br>
<b>⚠ 한계(정직 고지)</b> — 표본은 현재 생존 100종목(상장폐지 미포함)의 2018~2026 강세장 편향. 백테스트 수익·Sharpe는 <b>낙관 상한</b>이며, 실전 기대는 위 '현실적 기대수익' 카드의 보수 수치보다도 더 낮게 잡는 것이 안전합니다.
</div></details></div>
<div class="date" style="text-align:center;margin:16px 0">⚠️ 과거 통계는 미래 보장 아님 · 분산 큰 개별종목 · 손절 기계적 준수 · 최종책임 본인</div>

<div class="modal" id="modal" onclick="if(event.target.id=='modal')closeD()"><div class="sheet" id="sheet"></div></div>
<script>
const DATA="""+payload+""";
const FCOLOR={가치:'#0071e3',수익성:'#34c759',성장:'#ff9500',수급:'#af52de',모멘텀:'#ff2d55'};
const map={};DATA.items.forEach(i=>map[i.code]=i);
(function(){const m=DATA.market,mx=Math.max(...m.hist,1),el=document.getElementById('mhist');
 m.hist.forEach((c,k)=>{const h=Math.round(c/mx*92)+4,lab=m.hist_labels[k],col=(+lab>=65)?'#ff9500':'#0071e3';
 el.innerHTML+=`<div class="hcol"><div class="hnum">${c}</div><div class="hbar" style="height:${h}px;background:${col}"></div><div class="hlab">${lab}</div></div>`;});})();
// 계좌 손익 추이 차트 (보유가 2거래일 이상 쌓이면 표시)
(function(){const pf=DATA.portfolio;const el=document.getElementById('eqchart');
 if(!pf||!el||!pf.equity||pf.equity.length<2)return;
 const W=Math.min(window.innerWidth-64,470);
 el.innerHTML='<div class="date" style="margin-bottom:2px">내 계좌 손익률 추이(%)</div>'+
   lineChart(pf.equity.map(p=>p.date),pf.equity.map(p=>p.pnl),'#0071e3',W,140,{refs:[{v:0,color:'#c7c7cc',label:'0%'}]});})();

// 🔎 전체 종목 검색·정렬 (앱처럼 자유 탐색)
let SORT='score';
(function(){const sb=document.getElementById('sortBtns');if(!sb)return;
 [['score','종합점수'],['buyfit','BuyFit'],['timing','타이밍']].forEach((o,k)=>{
  const b=document.createElement('button');b.textContent=o[1];
  b.style.cssText='flex:1;padding:8px;border:0;border-radius:9px;background:'+(k==0?'#0071e3':'#e5e5ea')+';color:'+(k==0?'#fff':'#333')+';font-weight:700;font-size:13px';
  b.onclick=()=>{SORT=o[0];sb.querySelectorAll('button').forEach(x=>{x.style.background='#e5e5ea';x.style.color='#333'});b.style.background='#0071e3';b.style.color='#fff';renderStocks();};
  sb.appendChild(b);});})();
function renderStocks(){
 const el=document.getElementById('stockList');if(!el)return;
 const q=(document.getElementById('q').value||'').trim();
 let arr=DATA.items.slice();
 if(q)arr=arr.filter(i=>i.name.indexOf(q)>=0||(i.sector&&i.sector.indexOf(q)>=0)||(i.industry&&i.industry.indexOf(q)>=0));
 arr.sort((a,b)=>(b[SORT]||0)-(a[SORT]||0));
 el.innerHTML=arr.slice(0,60).map(i=>{
   const sig=i.signal?`<span style="font-size:11px;margin-left:4px">${i.signal}</span>`:'';
   const pnl=(i.pnl!==''&&i.pnl!=null)?`<span class="${i.pnl>=0?'pp':'pn'}" style="font-size:13px;margin-right:6px">${i.pnl>=0?'+':''}${i.pnl}%</span>`:'';
   const sec=i.sector?`<div style="font-size:11px;color:#999;margin-top:1px">${i.sector}</div>`:'';
   return `<div class="item" onclick="openD('${i.code}')"><div class="rk">${i.rank}</div>`+
     `<div style="flex:1;min-width:0"><div style="font-weight:700;font-size:15px">${i.name}${sig}</div>${sec}</div>${pnl}`+
     `<span style="font-size:12px;color:#888;margin-right:6px">BF ${(i.buyfit||0).toFixed(0)}</span>`+
     `<span class="sc">${i.score.toFixed(0)}</span><span class="chev">›</span></div>`;
 }).join('')||'<div class="empty" style="color:#aaa;text-align:center;padding:16px">검색 결과 없음</div>';
 if(arr.length>60)el.innerHTML+='<div class="leg" style="text-align:center">상위 60개 표시 · 검색(종목·업종)으로 좁히세요</div>';
}
renderStocks();

function lineChart(dates,vals,color,w,h,opt){
 opt=opt||{};const refs=opt.refs||[],shade=opt.shade;
 let all=vals.slice();
 refs.forEach(r=>{if(r.v!=null&&isFinite(r.v))all.push(+r.v)});
 if(shade){if(shade.lo!=null)all.push(+shade.lo);if(shade.hi!=null)all.push(+shade.hi);}
 const lo=Math.min(...all),hi=Math.max(...all),rng=(hi-lo)||1,n=vals.length;
 const X=i=>26+i/(n-1)*(w-36),Y=v=>h-20-(v-lo)/rng*(h-34);
 let pts=vals.map((v,i)=>`${X(i)},${Y(v)}`).join(' ');
 let area=`26,${h-20} `+pts+` ${X(n-1)},${h-20}`,yl='';
 [hi,(hi+lo)/2,lo].forEach((v,k)=>{const yy=20+k*(h-54)/2;yl+=`<text x="2" y="${yy+4}" font-size="9" fill="#aaa">${v.toFixed(0)}</text>`;});
 let xl='';[0,Math.floor(n/2),n-1].forEach(i=>{xl+=`<text x="${X(i)}" y="${h-5}" font-size="9" fill="#aaa" text-anchor="middle">${dates[i]}</text>`;});
 let sh='';
 if(shade&&shade.lo!=null&&shade.hi!=null){const y1=Y(+shade.hi),y2=Y(+shade.lo);sh=`<rect x="26" y="${Math.min(y1,y2)}" width="${w-36}" height="${Math.abs(y2-y1)}" fill="${shade.color||'#34c759'}18"/>`;}
 let rf='';
 refs.forEach(r=>{if(r.v==null||!isFinite(r.v))return;const y=Y(+r.v);
  rf+=`<line x1="26" y1="${y}" x2="${w-8}" y2="${y}" stroke="${r.color}" stroke-width="1.1" stroke-dasharray="4 3" opacity="0.85"/>`+
      `<text x="${w-9}" y="${y-2.5}" font-size="8.5" fill="${r.color}" text-anchor="end">${r.label}</text>`;});
 return `<svg width="${w}" height="${h}">${sh}<polygon points="${area}" fill="${color}22"/>
  <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.5"/>
  <circle cx="${X(n-1)}" cy="${Y(vals[n-1])}" r="4.5" fill="${color}"/>${rf}${yl}${xl}</svg>`;
}
function openD(code){
 const i=map[code];if(!i)return;const W=Math.min(window.innerWidth-32,520);
 // 매수 대상 판정 (보유/보유중/보류 종목엔 매수 안내·밴드를 숨김)
 const isBuy=i.signal&&i.signal.includes("매수");
 const blocked=i.signal==='🟢유지'||i.signal==='⏳보유'||i.signal==='⚪보류';
 const showBuy=isBuy||(i.rank<=20&&!blocked);
 let h='<div class="shandle"></div>';
 let pnl='';if(i.pnl!==''&&i.pnl!=null){const c=i.pnl>=0?'pp':'pn';pnl=` <span class="${c}">${i.pnl>=0?'+':''}${i.pnl}%</span>`;}
 h+=`<div class="dh"><div><div style="font-size:24px;font-weight:800">${i.name}${i.sector?` <span style="font-size:12px;font-weight:600;color:#0071e3;background:#eef;border-radius:7px;padding:2px 7px;vertical-align:middle">${i.sector}</span>`:''}</div>
   <div class="date">점수 ${i.score.toFixed(0)} (전체 ${i.rank}위) · 매수적합 ${i.buyrank}위 · ${i.price.toLocaleString()}원${pnl}${i.industry?' · '+i.industry:''}</div></div>
   <button class="x" onclick="closeD()">✕</button></div>`;
 if(i.signal)h+=`<div class="note"><b>${i.signal}</b> · ${i.reason}</div>`;

 // BuyFit 요약 카드
 const tcol=i.timing>=66?'#34c759':(i.timing>=33?'#ff9500':'#ff3b30');
 h+=`<div class="card"><h2>🎯 지금 매수 적합도</h2><div class="stat">
   <div><div class="v" style="color:#34c759">${i.buyfit.toFixed(0)}</div><div class="k">BuyFit(0~100)</div></div>
   <div><div class="v" style="color:${tcol}">${i.timing.toFixed(0)}</div><div class="k">진입타이밍</div></div>
   <div><div class="v">${i.dd!=null?i.dd:'-'}%</div><div class="k">60일고점대비</div></div></div>
   <div class="leg">타이밍↑ = 최근 조정으로 반등여지 큼(평균회귀). 점수가 높아도 신고가권이면 타이밍은 낮게 나옵니다.</div></div>`;

 // 🎓 왜 추천하나 — 근거 서술 (전문가 설득용)
 const tnote=i.timing>=66?'최근 조정으로 반등 여지가 큰 국면(평균회귀)':(i.timing>=33?'중립적 진입 구간':'신고가권 — 눌림을 기다릴 여지');
 let recTxt=`<b>${i.name}</b>는 ${i.basis||'복합 지표'}가 이끌어 종합 <b>${i.score.toFixed(0)}점</b>(100종목 중 ${i.rank}위). 진입타이밍 ${i.timing.toFixed(0)}/100 → ${tnote}.`;
 if(i.analog){const a=i.analog;recTxt+=` 과거 <b>점수·타이밍이 비슷했던 ${a.n}회</b>에서 30일 뒤 상승확률 <b>${a.win}%</b>, 중앙 <b>${a.med>=0?'+':''}${a.med}%</b>(통상 ${a.p25}~${a.p75}%).`;}
 else recTxt+=` (과거 유사표본이 적어 통계적 기대는 참고만).`;
 recTxt+=` 이는 개별 예측이 아니라 <b>규칙을 반복 적용해 분포의 평균</b>을 취하는 통계적 접근입니다.`;
 if(i.edge_weak)recTxt+=` <span style="color:#946200"><b>단, 이 국면은 승률/기대가 약해 확신은 낮습니다.</b></span>`;
 h+=`<div class="card" style="background:#f7fbff"><h2>🎓 왜 추천하나 (근거)</h2><div class="note" style="background:transparent;padding:0;line-height:1.8">${recTxt}</div></div>`;

 // 🩳 공매도 부담 (참고 지표 · 점수 미반영)
 if(i.short_pct!=null){
   const shHi=i.short_rank>=80;const shc=shHi?'#ff3b30':'#1d1d1f';
   h+=`<div class="card"><h2>🩳 공매도 부담 (참고)</h2><div class="stat">
     <div><div class="v" style="color:${shc}">${i.short_pct}%</div><div class="k">잔고비중 (${i.short_asof} 기준)</div></div>
     <div><div class="v" style="color:${shc}">${i.short_rank}</div><div class="k">공매도 백분위(100=최다)</div></div>
     <div><div class="v">${shHi?'⚠ 높음':'보통'}</div><div class="k">부담도</div></div></div>
     <div class="leg">공매도 잔고비중↑ = 하락에 베팅한 물량이 많다는 약세 신호(Boehmer·Jones·Zhang 2008). <b>KRX 잔고는 T+2 지연</b> 공시이고 표본이 최근뿐이라 <b>아직 종합점수엔 반영하지 않은</b> 참고 지표입니다.</div></div>`;
 }
 h+=`<div class="card"><h2>📈 최근 60일 종합점수</h2>${lineChart(i.dates,i.scores,'#0071e3',W,180)}</div>`;
 // 주가 차트에 매매 기준선(밴드) 오버레이
 let pb={refs:[],shade:null};
 if(i.avg_price!==''&&i.avg_price!=null)pb.refs.push({v:+i.avg_price,color:'#8e8e93',label:'평단'});
 if(i.guard_price!==''&&i.guard_price!=null)pb.refs.push({v:+i.guard_price,color:'#ff3b30',label:'감시가'});
 if(showBuy){
   if(i.buy_limit!=null)pb.refs.push({v:+i.buy_limit,color:'#0071e3',label:'지정가'});
   if(i.stop_buy!=null)pb.refs.push({v:+i.stop_buy,color:'#ff3b30',label:'손절'});
 }
 if(i.tgt_med){const tc=i.edge_weak?'#c7c7cc':'#34c759';pb.refs.push({v:i.tgt_med,color:tc,label:'목표中'});pb.shade={lo:i.tgt_lo,hi:i.tgt_hi,color:tc};}
 const legParts=['<span style="color:#34c759">초록</span>=과거통계 목표(중앙값)·25~75%밴드'];
 if(showBuy)legParts.unshift('<span style="color:#0071e3">파랑</span>=권장지정가·<span style="color:#ff3b30">빨강</span>=손절선');
 if(i.avg_price!==''&&i.avg_price!=null)legParts.push('<span style="color:#8e8e93">회색</span>=내 평단·<span style="color:#ff3b30">빨강</span>=감시가');
 h+=`<div class="card"><h2>💰 최근 60일 주가 + 매매 밴드</h2>${lineChart(i.dates,i.prices,'#34c759',W,196,pb)}<div class="leg">${legParts.join(' · ')}</div></div>`;

 h+='<div class="card"><h2>🧩 점수 구성 · 근거</h2>';
 h+=`<div class="leg" style="margin-top:0">종합점수 = 각 팩터의 <b>당일 100종목 중 백분위(0~100)</b> × 가중치. 오른쪽 숫자는 <b>기여점수</b>(백분위×가중).</div>`;
 i.factors.forEach(f=>{const col=FCOLOR[f.name]||'#888',wcol=f.w>0?'#1d1d1f':'#c7c7cc';
  h+=`<div class="bigbar-wrap"><span class="bl">${f.name}</span>
   <div class="bigbar"><div style="width:${Math.max(0,Math.min(100,f.val))}%;background:${col}"></div></div>
   <span class="bv">${f.val.toFixed(0)}</span><span class="bw" style="color:${wcol}">${f.w>0?'+'+f.contrib.toFixed(1):'미사용'}</span></div>`;});
 const stg=i.factors.filter(f=>f.val>=70&&f.w>0).map(f=>f.name),wk=i.factors.filter(f=>f.val<=35&&f.w>0).map(f=>f.name);
 let why=[];if(stg.length)why.push('강점 '+stg.join('·'));if(wk.length)why.push('약점 '+wk.join('·'));
 if(why.length)h+=`<div class="note">📌 ${why.join(' / ')}</div>`;
 h+='<details style="margin-top:6px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">각 팩터의 학술적 근거 보기</summary>';
 i.factors.filter(f=>f.w>0).forEach(f=>{h+=`<div class="leg" style="margin:8px 0 0"><b>${f.name}</b> — ${f.why}<br><span style="color:#0071e3">📚 ${f.academic}</span><br><span style="color:#946200">📊 ${f.evidence}</span></div>`;});
 h+='</details></div>';


 // 💵 매수 가격 안내 + 기대손익 계산기 (이미 보유/보류 종목엔 숨김)
 if(showBuy){
   h+=`<div class="card"><h2>💵 이 가격에 매수 (지정가)</h2>
     <div class="stat">
       <div><div class="v" style="color:#0071e3">${(+i.buy_limit).toLocaleString()}</div><div class="k">권장 지정가</div></div>
       <div><div class="v">${(+i.price).toLocaleString()}</div><div class="k">현재가</div></div>
       <div><div class="v" style="color:#ff3b30">${Math.round(i.price*0.9).toLocaleString()}</div><div class="k">손절가(체결가 -10%)</div></div>
     </div>
     <div class="leg">권장 지정가 = 현재가와 최근 5일 저가 사이(평균회귀 전략상 살짝 눌렀을 때 매수). 이 가격에 안 닿으면 미체결될 수 있으니, 확실히 사려면 현재가로.<br><b>손절가는 실제 체결가의 -10%</b>로 거세요 (표시값은 현재가 체결 기준). 지정가 ${(+i.buy_limit).toLocaleString()}원에 체결되면 손절가는 ${Math.round(i.buy_limit*0.9).toLocaleString()}원.</div></div>`;
   // 손실예산 기반 수량 계산 (지정가 체결·손절 -10% → 잃는 돈 ≈ 매수금액의 10%)
   h+=`<div class="card"><h2>🎯 손실예산으로 수량 정하기</h2>
     <div class="leg" style="margin-top:0">"이 종목에서 최대 얼마까지 잃어도 되나"를 고르면, 지정가 ${(+i.buy_limit).toLocaleString()}원·손절 -10% 기준 <b>살 수량과 매수금액</b>을 계산합니다.</div>
     <div id="riskBtns" style="display:flex;gap:6px;margin:8px 0"></div>
     <div id="riskOut"></div>
     <div class="leg">한 종목에서 감수할 손실을 <b>원금의 1~2%</b>로 두면 자연스럽게 분산됩니다. (슬리피지·수수료 제외 근사)</div></div>`;
   // 기대손익 계산기
   if(i.exp){
     const e=i.exp;
     h+=`<div class="card"><h2>💰 이 금액으로 사면? (과거 통계 기반)</h2>
       <div class="leg" style="margin-top:0">투자금 선택 → 30일 뒤 예상 손익 (점수·타이밍 유사했던 과거 사례 분포)</div>
       <div id="invBtns" style="display:flex;gap:6px;margin:8px 0"></div>
       <div id="expOut"></div>
       <div class="leg">중앙값=가장 흔한 결과. 좋을때/나쁠때는 상·하위 25% 지점. 승률 ${e.win}%.</div></div>`;
   }
 }

 if(i.analog){const a=i.analog,wc=a.win>=53?'#34c759':(a.win>=45?'#ff9500':'#ff3b30');
  h+=`<div class="card"><h2>🎯 과거 비슷할 때 (30일 후 실제)</h2>`;
  if(i.edge_weak)h+=`<div class="warn" style="color:#946200;background:#fff8e1">⚠ <b>엣지 약함</b> — 승률 ${a.win}% · 평균 ${a.avg>=0?'+':''}${a.avg}%. 과거 유사국면에서 뚜렷한 우위가 없었습니다. 매수는 소액·보류를 고려하세요.</div>`;
  h+=`<div class="date">점수 ±5${a.fine?' + 타이밍 동일분위':''} 였던 <b>${a.n}회</b>의 30일 뒤 수익 분포</div>
   <div class="stat">
    <div><div class="v" style="color:${wc}">${a.win}%</div><div class="k">상승확률(승률)</div></div>
    <div><div class="v">${a.med>=0?'+':''}${a.med}%</div><div class="k">중앙값</div></div>
    <div><div class="v">${a.avg>=0?'+':''}${a.avg}%</div><div class="k">평균</div></div></div>`;
  const lo=a.worst,hi=a.best,rng=(hi-lo)||1,pos=v=>((v-lo)/rng*100);
  h+=`<div class="range"><div class="rtrack"></div><div class="rdot" style="left:${pos(a.med)}%"></div>
   <div class="rlab" style="left:${pos(a.worst)}%">${a.worst}%</div>
   <div class="rlab" style="left:${pos(a.best)}%">${a.best}%</div></div>
   <div class="leg">검정점=중앙값. 50% 구간 ${a.p25}%~${a.p75}%. 최악 ${a.worst}% · 최선 ${a.best}% (표본 ${a.n}).<br>※ 표본은 <b>겹치는 기간·생존종목 한정</b>이라 실제 독립 관측치는 적고 불확실성은 더 큽니다.</div></div>`;
 } else h+=`<div class="card"><h2>🎯 과거 유사사례</h2><div class="date">유사 표본 부족</div></div>`;

  let priceBox="";
  if(i.signal&&(i.signal.includes("손절")||i.signal.includes("매도"))){
    priceBox=`<div class="stat"><div><div class="v" style="color:#ff3b30">${(+i.price).toLocaleString()}</div><div class="k">지정가매도 참고</div></div>`+
      (i.guard_price!==""&&i.guard_price!=null?`<div><div class="v">${(+i.guard_price).toLocaleString()}</div><div class="k">손절감시가</div></div>`:"")+`</div>`;
  } else if(i.avg_price!==""&&i.avg_price!=null&&i.signal&&i.signal.includes("유지")){
    const sp=Math.round(i.avg_price*0.9);
    priceBox=`<div class="stat"><div><div class="v">${(+i.avg_price).toLocaleString()}</div><div class="k">내 평단가</div></div>`+
      `<div><div class="v" style="color:#ff3b30">${sp.toLocaleString()}</div><div class="k">손절가(-10%)</div></div>`+
      (i.guard_price!==""&&i.guard_price!=null?`<div><div class="v">${(+i.guard_price).toLocaleString()}</div><div class="k">감시가</div></div>`:"")+`</div>`;
  } else if(i.stop_price!==""&&i.stop_price!=null){
    priceBox=`<div class="stat"><div><div class="v">${(+i.price).toLocaleString()}</div><div class="k">현재가(매수참고)</div></div>`+
      `<div><div class="v" style="color:#ff3b30">${(+i.stop_price).toLocaleString()}</div><div class="k">매수후 손절가</div></div></div>`;
  }
 let holdInfo="";
 if(i.guard_gap!=null||i.dmin_remain!=null){
   holdInfo=`<div class="stat" style="margin-top:6px">`+
     (i.guard_gap!=null?`<div><div class="v" style="color:#ff3b30">${Math.abs(i.guard_gap).toFixed(1)}%</div><div class="k">손절까지 여유(현재가↓)</div></div>`:``)+
     (i.dmin_remain!=null?`<div><div class="v">${i.dmin_remain<=0?'가능':'D-'+i.dmin_remain}</div><div class="k">${i.dmin_remain<=0?'최소보유 충족':'매도가능('+(i.dmin_date||'')+')'}</div></div>`:``)+
     `</div>`;
 }
 h+=`<div class="card"><h2>🚪 매도 시점 / 주문 가격</h2>${priceBox}${holdInfo}<div class="note">${i.sell_hint}</div>
  <div class="warn">※ 개별 가격 예측이 아니라 위 통계 분포로 해석하세요. 이 시스템은 규칙대로 사고팔아 분포의 평균을 취하는 전략입니다.</div></div>`;
 document.getElementById('sheet').innerHTML=h;

 // 기대손익 계산기 활성화 (매수 카드가 렌더된 경우에만 — invBtns/expOut이 그 안에 있음)
 if(showBuy && i.exp){
   const e=i.exp, amounts=[500000,1000000,3000000,5000000];
   const btns=document.getElementById('invBtns');
   const out=document.getElementById('expOut');
   function render(amt){
     const med=Math.round(amt*e.med/100), avg=Math.round(amt*e.avg/100);
     const good=Math.round(amt*e.p75/100), bad=Math.round(amt*e.p25/100);
     const sign=v=>(v>=0?'+':'')+v.toLocaleString()+'원';
     const col=v=>v>=0?'#d70015':'#0071e3';
     out.innerHTML=`<div class="stat">
       <div><div class="v" style="color:${col(med)};font-size:18px">${sign(med)}</div><div class="k">기대(중앙값) ${e.med>=0?'+':''}${e.med}%</div></div>
       <div><div class="v" style="color:#34c759;font-size:16px">${sign(good)}</div><div class="k">좋을때 +${e.p75}%</div></div>
       <div><div class="v" style="color:#ff3b30;font-size:16px">${sign(bad)}</div><div class="k">나쁠때 ${e.p25}%</div></div></div>`;
   }
   btns.innerHTML='';
   amounts.forEach((a,k)=>{
     const b=document.createElement('button');
     b.textContent=(a/10000)+'만';
     b.style.cssText='flex:1;padding:9px;border:0;border-radius:9px;background:'+(k==1?'#0071e3':'#e5e5ea')+';color:'+(k==1?'#fff':'#333')+';font-weight:700;font-size:13px';
     b.onclick=()=>{btns.querySelectorAll('button').forEach(x=>{x.style.background='#e5e5ea';x.style.color='#333'});b.style.background='#0071e3';b.style.color='#fff';render(a);};
     btns.appendChild(b);
   });
   render(1000000);
 }

 // 손실예산 기반 수량 계산기 활성화 (매수 카드가 렌더된 경우에만)
 if(showBuy){
   const bl=+i.buy_limit,riskB=document.getElementById('riskBtns'),riskO=document.getElementById('riskOut');
   if(bl&&riskB&&riskO){
     const budgets=[20000,50000,100000,200000];
     function rrender(bud){
       const sh=Math.floor((bud*10)/bl);          // 손절 -10% → 매수금액=예산×10, 수량=금액/지정가
       if(sh<=0){
         const need=Math.ceil(bl/10);            // 1주 손절손실이 예산 이하가 되는 최소 예산
         riskO.innerHTML=`<div class="warn">이 예산으론 1주도 못 삽니다 (지정가 ${bl.toLocaleString()}원). 1주 사려면 예산을 <b>${need.toLocaleString()}원</b> 이상으로 (또는 지정가로 1주 ${bl.toLocaleString()}원 매수 시 손절손실 ${Math.round(bl*0.1).toLocaleString()}원).</div>`;
         return;
       }
       const realInvest=sh*bl,realLoss=Math.round(realInvest*0.10);
       riskO.innerHTML=`<div class="stat">
         <div><div class="v" style="color:#0071e3">${sh.toLocaleString()}주</div><div class="k">살 수량</div></div>
         <div><div class="v">${realInvest.toLocaleString()}</div><div class="k">매수금액(원)</div></div>
         <div><div class="v" style="color:#ff3b30">-${realLoss.toLocaleString()}</div><div class="k">손절 시 손실(원)</div></div></div>`;
     }
     riskB.innerHTML='';
     budgets.forEach((bg,k)=>{
       const b=document.createElement('button');
       b.textContent=(bg/10000)+'만';
       b.style.cssText='flex:1;padding:9px;border:0;border-radius:9px;background:'+(k==1?'#ff3b30':'#e5e5ea')+';color:'+(k==1?'#fff':'#333')+';font-weight:700;font-size:13px';
       b.onclick=()=>{riskB.querySelectorAll('button').forEach(x=>{x.style.background='#e5e5ea';x.style.color='#333'});b.style.background='#ff3b30';b.style.color='#fff';rrender(bg);};
       riskB.appendChild(b);
     });
     rrender(50000);
   }
 }

 document.getElementById('modal').classList.add('on');document.body.style.overflow='hidden';
}
function closeD(){document.getElementById('modal').classList.remove('on');document.body.style.overflow='';}
</script></body></html>"""
    open(OUT,"w",encoding="utf-8").write(H)
    print(f"리포트: {OUT} · BuyFit추천 {len(buylist)} · 보유 {len(holds)}")


if __name__=="__main__":
    main()


