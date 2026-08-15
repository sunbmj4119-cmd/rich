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
    holds=[i for i in items if i["signal"] in ("🟢유지","⏳보유","🟠부분익절")]
    tps=[i for i in items if i["signal"]=="🟠부분익절"]

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
/* 판단 등급 배지 */
.gr{{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:7px;
     color:#fff;font-weight:800;font-size:12px;flex-shrink:0}}
.gA{{background:#34c759}}.gB{{background:#30a2e8}}.gC{{background:#ff9500}}.gD{{background:#c7c7cc}}
/* 반대논리 / 상승논리 행 */
.rk2{{display:flex;gap:9px;padding:10px 0;border-bottom:1px solid #f2f2f5;align-items:flex-start}}
.rk2:last-child{{border:0}}
.sev{{font-size:10px;font-weight:800;color:#fff;border-radius:6px;padding:3px 6px;flex-shrink:0;margin-top:1px}}
.sHigh{{background:#ff3b30}}.sMed{{background:#ff9500}}.sLow{{background:#c7c7cc}}
.rtx{{flex:1;min-width:0;font-size:13px;line-height:1.65}}
.rev{{font-size:11px;color:#999;margin-top:3px}}
/* 시나리오 표 — 좁은 화면(390px)에서도 잘리지 않게. 넘치면 가로 스크롤. */
.tw{{overflow-x:auto;-webkit-overflow-scrolling:touch;margin-top:6px}}
.stab{{width:100%;border-collapse:collapse;font-size:12px}}
.stab th{{font-size:10px;color:#999;font-weight:600;text-align:right;padding:5px 2px;border-bottom:1px solid #eee;white-space:nowrap}}
.stab th:first-child{{text-align:left}}
.stab td{{padding:8px 2px;text-align:right;border-bottom:1px solid #f5f5f7;white-space:nowrap}}
.stab td:first-child{{text-align:left;font-weight:600;white-space:normal}}
.stab tr:last-child td{{border:0}}
.stab tr.now td{{background:#eef5ff}}
.stab tr.now td:first-child{{border-left:3px solid #0071e3;padding-left:5px}}
/* 확률 막대 */
.pbar{{display:flex;height:26px;border-radius:7px;overflow:hidden;margin:8px 0 4px}}
.pbar>div{{display:flex;align-items:center;justify-content:center;font-size:10px;color:#fff;font-weight:700;min-width:0}}
.ci{{position:relative;height:34px;margin:10px 4px 4px}}
.citrack{{position:absolute;top:13px;left:0;right:0;height:8px;background:#eee;border-radius:4px}}
.cirange{{position:absolute;top:13px;height:8px;background:#0071e3;opacity:.32;border-radius:4px}}
.cidot{{position:absolute;top:8px;width:18px;height:18px;border-radius:50%;background:#0071e3;border:3px solid #fff;
        box-shadow:0 0 0 1px #0071e3;transform:translateX(-50%)}}
.cibase{{position:absolute;top:9px;width:2px;height:16px;background:#ff3b30}}
.cilab{{position:absolute;top:26px;font-size:9.5px;color:#888;transform:translateX(-50%);white-space:nowrap}}
/* 뉴스 */
.nw{{display:block;padding:10px 0;border-bottom:1px solid #f2f2f5;text-decoration:none;color:inherit}}
.nw:last-child{{border:0}}
.nwt{{font-size:13.5px;line-height:1.5;font-weight:600}}
.nws{{font-size:11px;color:#999;margin-top:3px}}
.chip{{display:inline-block;border-radius:7px;padding:2px 7px;font-size:11px;font-weight:700;margin-right:4px}}
.cpos{{background:#e8f8ec;color:#1a8b38}}.cneg{{background:#ffecec;color:#d70015}}.cneu{{background:#f0f0f3;color:#888}}
.regbox{{display:flex;align-items:center;gap:12px;padding:4px 0 10px}}
.regemo{{font-size:34px;line-height:1}}
/* 한눈 요약 띠 — 스크롤 전에 핵심 3개를 먼저 */
.strip{{display:flex;gap:8px;margin:12px 0 14px}}
.sbox{{flex:1;background:#fff;border-radius:14px;padding:12px 6px;text-align:center;
       box-shadow:0 1px 4px rgba(0,0,0,.06);min-width:0}}
.sbig{{font-size:22px;font-weight:800;line-height:1.15;white-space:nowrap}}
.ssub{{font-size:10.5px;color:#8e8e93;margin-top:4px;line-height:1.35}}
/* 신뢰도 배지 — 이 숫자를 얼마나 믿을지 카드마다 한눈에 */
.tb{{display:inline-block;font-size:10px;font-weight:800;border-radius:6px;padding:2px 6px;
     margin-left:6px;vertical-align:middle;white-space:nowrap}}
.t1{{background:#e3f7e8;color:#137a33}}   /* 검증됨 */
.t2{{background:#fff2e0;color:#9a5800}}   /* 참고 */
.t3{{background:#ffe9e9;color:#c00}}      /* 미검증 */
/* 종목 상세 탭 — 17개 카드를 3묶음으로 */
.tabs{{display:flex;gap:6px;position:sticky;top:0;background:#f2f2f7;padding:6px 0 10px;z-index:5}}
.tabs button{{flex:1;padding:11px 3px;border:0;border-radius:11px;background:#e5e5ea;color:#555;
              font-weight:800;font-size:13.5px}}
.tabs button.on{{background:#0071e3;color:#fff}}
.pane{{display:none}} .pane.on{{display:block}}
/* 오늘 할 일 */
.todo{{display:flex;gap:10px;align-items:flex-start;padding:12px 2px;border-bottom:1px solid #f0f0f0;cursor:pointer}}
.todo:last-child{{border:0}} .todo:active{{background:#f5f5f7}}
.tnum{{width:26px;height:26px;border-radius:8px;color:#fff;font-weight:800;font-size:13px;
       display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}}
.ttx{{flex:1;min-width:0;font-size:14px;line-height:1.6}}
.tsub{{font-size:12px;color:#888;margin-top:3px;line-height:1.55}}
/* 쉽게 보기 — 초보자에겐 카드 17개가 곧 '판단 불가'다. 기본은 핵심만 보여준다. */
body.simple .adv{{display:none}}
body:not(.simple) .simponly{{display:none}}
.lvbar{{display:flex;gap:6px;background:#e9e9ee;border-radius:12px;padding:4px;margin:12px 0}}
.lvbar button{{flex:1;border:0;border-radius:9px;padding:9px 4px;font-size:14px;font-weight:800;
  background:transparent;color:#666;cursor:pointer;font-family:inherit}}
.lvbar button.on{{background:#fff;color:#0071e3;box-shadow:0 1px 3px rgba(0,0,0,.12)}}
.advtag{{display:inline-block;background:#f2f2f7;color:#8e8e93;border-radius:6px;
  padding:1px 6px;font-size:10px;font-weight:700;vertical-align:middle;margin-left:5px}}
/* 3줄 요약 — 스크롤 전에 결론부터 */
.tl{{background:#fff;border-radius:16px;padding:15px 16px;margin-bottom:12px;
  box-shadow:0 1px 3px rgba(0,0,0,.07)}}
.tl b{{font-size:15px}}
.tl .ln{{display:flex;gap:9px;align-items:flex-start;margin-top:9px;font-size:14px;line-height:1.5}}
.tl .no{{flex:none;width:21px;height:21px;border-radius:50%;background:#0071e3;color:#fff;
  font-size:12px;font-weight:800;display:flex;align-items:center;justify-content:center;margin-top:1px}}
</style></head><body>
<h1>🎯 매수적합도 분석</h1>
<div class="date simponly">기준일 {m['date']} · 코스피 100종목을 <b>지금 사기 좋은 순서</b>로 줄 세웠습니다.<br>종목을 누르면 <b>사라·들고 있어라·팔아라</b>를 퍼센트와 이유로 알려줍니다.</div>
<div class="date adv">기준일 {m['date']} · <b>BuyFit</b> = 종합점수 + 진입타이밍(평균회귀)을 합친 '지금 사기 좋은 순서'<br>종목을 터치하면 상승·하락 시나리오와 확률을 봅니다</div>
<div style="display:flex;gap:8px;margin:14px 0">
  <a href="trade.html" class="btn" style="flex:1;margin:0">📝 매매 기록</a>
  <a href="journal.html" class="btn" style="flex:1;margin:0;background:#5856d6">📓 투자논리</a>
</div>
<div class="lvbar"><button id="lv0" class="on" onclick="setLv(1)">🟢 쉽게 보기</button>
  <button id="lv1" onclick="setLv(0)">🔬 자세히 보기</button></div>
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

    # ── 한눈 요약 띠 — 스크롤하기 전에 '시장 / 내 계좌 / 오늘 주문' 셋만 ──
    rg0=d.get("regime") or {}
    rc=(rg0.get("current") or {})
    n_act=len(cuts)+len(sells)
    n_buy=sum(1 for i in items if i["signal"]=="🟡매수")
    if pf:
        up=pf["upnl"]; ucol="#d70015" if up>=0 else "#0071e3"
        acct=(f'<div class="sbig" style="color:{ucol}">{"+" if up>=0 else ""}{up:,}</div>'
              f'<div class="ssub">평가손익 ({pf["upnl_pct"]:+.1f}%)<br>{pf["n"]}종목 보유</div>')
    else:
        acct='<div class="sbig" style="color:#c7c7cc">—</div><div class="ssub">보유 없음</div>'
    acol="#ff3b30" if n_act else ("#34c759" if n_buy else "#8e8e93")
    H+=f"""<div class="strip">
  <div class="sbox"><div class="sbig" style="color:{rc.get('color','#8e8e93')};font-size:19px">
    {rc.get('emoji','')} {rc.get('name','국면 미계산')}</div>
    <div class="ssub">{rc.get('streak',0)}거래일째<br>breadth {rc.get('breadth',0):.0f}%</div></div>
  <div class="sbox">{acct}</div>
  <div class="sbox"><div class="sbig" style="color:{acol}">{n_act + min(n_buy,3)}</div>
    <div class="ssub">오늘 검토할 주문<br>청산 {n_act} · 매수후보 {min(n_buy,3)}</div></div>
</div>"""

    # ── 세 줄 요약 — 스크롤하기 전에 결론부터 ────────────────
    # 초보자가 가장 먼저 막히는 건 '숫자가 없어서'가 아니라 '어디부터 봐야 할지 몰라서'다.
    # 그래서 오늘 할 일 위에 결론 세 줄을 먼저 놓는다.
    tl=[]
    if cuts or sells:
        nm=", ".join(i["name"] for i in (cuts+sells)[:2])
        tl.append(f'<b>먼저 팔 것이 있습니다</b> — {nm}{" 외" if len(cuts+sells)>2 else ""}. '
                  f'규칙이 정한 것이니 고민하지 말고 실행하세요.')
    elif tps:
        tl.append(f'<b>{tps[0]["name"]}이(가) 1차 익절선에 닿았습니다</b> — 3분의 1만 팔고 나머지는 그대로 둡니다.')
    else:
        tl.append('<b>오늘 꼭 팔아야 할 것은 없습니다.</b> 보유 종목의 손절가만 확인하세요.')
    if n_buy:
        top=next((i for i in buylist if i["signal"]=="🟡매수"), None)
        tl.append(f'<b>살 만한 것은 {n_buy}개</b>' +
                  (f' — 1순위는 {top["name"]}, 지정가 {top["buy_limit"]:,}원입니다.' if top else '.'))
    else:
        tl.append('<b>오늘은 새로 살 종목이 없습니다.</b> 안 사는 것도 결정입니다.')
    nb=((m.get("poslab") or {}).get("new") or {}).get("1-10")
    if nb:
        tl.append(f'<b>사면 5번 중 1번은 손절입니다</b>(손절 {nb["hit_stop"]}% vs 익절 {nb["hit_tp"]}%). '
                  f'그래서 <b>한 종목에 크게 걸지 않는 것</b>이 이 전략의 핵심입니다.')
    else:
        tl.append('<b>한 종목에 크게 걸지 마세요.</b> 손절은 익절보다 자주 옵니다.')
    H+=('<div class="tl"><b>📌 오늘 이것만 알면 됩니다</b>'
        +"".join(f'<div class="ln"><div class="no">{k+1}</div><div>{t}</div></div>'
                 for k,t in enumerate(tl))
        +'<div class="leg" style="margin-top:10px">더 깊이 보려면 위의 <b>🔬 자세히 보기</b>를 누르세요.</div></div>')

    # ── 📋 오늘 할 일 — 화면 맨 위. 스크롤 없이 '무엇을 하면 되는가'만. ──
    SPC={"buy":"#0071e3","hold":"#34c759","sell":"#ff3b30"}

    def sp_line(i):
        """근거를 합산한 포지션 배분 한 줄 — 반대편이 몇 %인지도 같이 보여준다"""
        sp=i.get("split")
        if not sp: return ""
        return ('<div style="margin-top:3px;font-size:11px">'
                +' · '.join(f'<b style="color:{SPC[k]}">{sp["labels"][k]} {sp["pct"][k]}%</b>'
                            for k in sorted(sp["pct"],key=lambda k:-sp["pct"][k]))
                +'</div>')

    todos=[]
    for i in cuts:
        todos.append(("#ff3b30", f'<b>{i["name"]}</b> 손절 — {i["reason"]}',
                      f'지정가 매도 {i["price"]:,}원 · 규칙이 정한 것이니 논리와 무관하게 실행'+sp_line(i)))
    for i in sells:
        todos.append(("#0071e3", f'<b>{i["name"]}</b> 매도 — {i["reason"]}',
                      f'지정가 매도 {i["price"]:,}원'+sp_line(i)))
    for i in tps:
        q=i.get("tp_qty") or ""
        todos.append(("#ff9500", f'<b>{i["name"]}</b> 부분 익절 — {i["pnl"]:+}%',
                      f'보유의 <b>1/3</b>{f" (약 {q}주)" if q else ""} 지정가 매도 {i["price"]:,}원 · '
                      f'나머지는 트레일 -{i.get("trail_w") or 8}%로 계속 보유'+sp_line(i)))
    # 매수 후보: 실제로 '지금 살 수 있는' 것만.
    #  · 이미 보유(🟢유지·⏳보유)나 청산 예정(🔴손절·🔵매도)은 매수 대상이 아니다
    #  · ⚪보류는 외국인 순매도라 시스템이 보류시킨 것
    #  · D등급 제외 — 검증에서 시장 수익률을 밑돌았다
    BUYABLE=("🟡매수","")
    cands=[i for i in buylist
           if i["signal"] in BUYABLE and i["verdict"]["grade"]!="D"][:3]
    for i in cands:
        pr=i.get("prob") or {}
        kel=pr.get("kelly_use") or 0
        amt=(f'권장비중 계좌의 <b>{kel}%</b>까지' if kel > 0
             else '<b>비중을 키울 근거 없음</b> → 손실예산(원금 1~2%)으로만')
        todos.append(("#34c759",
            f'<b>{i["name"]}</b> 매수 검토 <span class="gr g{i["verdict"]["grade"]}">{i["verdict"]["grade"]}</span>',
            f'지정가 {i["buy_limit"]:,}원 · 손절 {int(i["buy_limit"]*0.9):,}원<br>'
            f'{amt} · 이 구간 손절확률 {pr.get("p_stop","-")}%'+sp_line(i)))
    if not todos:
        todos.append(("#8e8e93","오늘은 실행할 주문이 없습니다","보유 종목의 감시가만 확인하세요"))
    trisk=""
    if pf and pf.get("stress"):
        s10=next((x for x in pf["stress"] if x["move"]==-10),None)
        if s10:
            trisk=(f'<div class="warn" style="margin-top:8px">시장이 30일 안에 <b>-10%</b>면(과거 빈도 '
                   f'{s10["prob"]:.0f}%) 내 계좌는 <b>{s10["pnl"]:+,}원</b>, 손절 <b>{s10["stops"]}종목</b> 발동. '
                   f'감당 가능한지 먼저 보세요.</div>')
    todo_codes=([i["code"] for i in cuts]+[i["code"] for i in sells]+[i["code"] for i in tps]
                +[i["code"] for i in cands]+[""]*3)

    def todo_row(n, item, code):
        col, txt, sub = item
        click = f''' onclick="openD('{code}')"''' if code else ""
        return (f'<div class="todo"{click}><div class="tnum" style="background:{col}">{n+1}</div>'
                f'<div class="ttx">{txt}<div class="tsub">{sub}</div></div></div>')

    H+=('<div class="card" style="border:2px solid #0071e3"><h2>📋 오늘 할 일</h2>'
        +"".join(todo_row(n,t,c) for n,(t,c) in enumerate(zip(todos,todo_codes)))
        +trisk
        +'<div class="leg">매수 후보는 <b>BuyFit 상위</b>에서 D등급을 뺀 것입니다. '
         'D등급은 검증에서 시장 수익률을 밑돌았습니다.</div></div>')

    # 데이터가 낡았으면 조용히 넘어가지 않는다 — 사용자가 그 값으로 주문을 넣기 때문
    st=m.get("stale")
    if st:
        H+=(f'<div class="card alert"><h2>⚠️ 데이터 지연 알림</h2>'
            f'<div class="note" style="background:transparent;padding:0;line-height:1.8">'
            f'<b>{st["file"]}</b>가 <b>{st["last"]}</b>에서 멈춰 있습니다({st["lag"]}거래일 지연).<br>'
            f'그래서 {st["effect"]}. 낡은 저가로 엉뚱한 지정가를 제시하지 않기 위한 안전장치입니다.'
            f'</div><div class="leg">주가·점수·수급은 정상 갱신 중이라 <b>순위와 판단은 유효</b>합니다. '
            f'다만 <b>권장 지정가는 현재가 대비 -1%</b>로만 제안되니, 실제 주문가는 호가창을 보고 정하세요.</div></div>')

    # 신뢰도 범례 — 아래 모든 카드의 배지를 읽는 법 (한 번만)
    H+=('<div class="card adv"><h2>🏷 숫자 읽는 법<span class="advtag">자세히</span></h2>'
        '<div class="leg" style="margin-top:0;line-height:2">'
        '<span class="tb t1">검증됨</span> 과거 데이터로 실제 맞았는지 확인함 → <b>판단 근거로 쓰세요</b><br>'
        '<span class="tb t2">참고</span> 논리적 근거는 있으나 이 저장소가 검증하지 않음 → <b>확인할 거리</b><br>'
        '<span class="tb t3">미검증</span> 검증에서 예측력이 없었음 → <b>기록으로만 보세요</b>'
        '</div><div class="warn">가장 흔한 실수는 <b>승률 숫자를 예보로 읽는 것</b>입니다. '
        '35,163건 검증에서 개별 종목 승률의 예측력은 0이었습니다. 승률은 "과거에 이랬다"는 기록입니다.</div></div>')

    if cuts:
        H+='<div class="card alert"><h2>🔴 손절 (즉시 검토)</h2>'+"".join(small_row(i,True) for i in cuts)+'</div>'
    if sells:
        H+='<div class="card"><h2>🔵 매도</h2>'+"".join(small_row(i,True) for i in sells)+'</div>'

    # 🌡 지금 어떤 시장인가 — 국면 판정 + 국면별 이 전략의 과거 성적
    rg=d.get("regime")
    if rg:
        c=rg["current"]
        rows=""
        for h in rg["history"]:
            now=' class="now"' if h["name"]==c["name"] else ""
            if "win" in h:
                wc="#34c759" if h["win"]>=60 else ("#ff9500" if h["win"]>=52 else "#ff3b30")
                rows+=(f'<tr{now}><td>{h["emoji"]} {h["name"]}</td>'
                       f'<td style="color:{wc};font-weight:800">{h["win"]:.0f}%</td>'
                       f'<td>{h["med"]:+.1f}%</td>'
                       f'<td style="color:#999">{h["edge"]:+.1f}%p</td>'
                       f'<td style="color:#bbb">{h["eff_n"]:.0f}</td></tr>')
            else:
                rows+=(f'<tr{now}><td>{h["emoji"]} {h["name"]}</td>'
                       f'<td colspan="4" style="color:#bbb;text-align:left">표본 부족 ({h["days"]}일)</td></tr>')
        tr=rg.get("transition",{}).get(c["name"],{})
        trtxt=" · ".join(f'{k} {v:.0f}%' for k,v in sorted(tr.items(),key=lambda x:-x[1])) if tr else "표본 부족"
        H+=f"""<div class="card"><h2>🌡 지금 어떤 시장인가<span class="tb t1">검증됨</span></h2>
<div class="regbox"><div class="regemo">{c['emoji']}</div>
  <div style="flex:1;min-width:0">
    <div style="font-size:19px;font-weight:800;color:{c['color']}">{c['name']}</div>
    <div class="date" style="margin-top:2px">{c['streak']}거래일째 ({c['since']}~) · {c['desc']}</div></div></div>
<div class="stat">
  <div><div class="v" style="font-size:17px">{c['trend']:+.1f}%</div><div class="k">200일선 대비</div></div>
  <div><div class="v" style="font-size:17px">{c['breadth']:.0f}%</div><div class="k">60일선 위 종목비율</div></div>
  <div><div class="v" style="font-size:17px">{c['vol']:.0f}%</div><div class="k">연환산 변동성</div></div>
</div>
<div id="ridx" style="margin-top:10px"></div>
<div class="leg" style="margin-top:10px"><b>국면별로 이 전략(상위10종목·30일보유)이 과거에 어땠나</b> <span style="color:#0071e3">파란 줄 = 지금</span></div>
<div class="tw"><table class="stab"><tr><th>국면</th><th>승률</th><th>중앙</th><th>vs시장</th><th>표본</th></tr>{rows}</table></div>
<div class="note" style="margin-top:8px;background:#f7f7fa">🔮 <b>{c['name']}</b>에서 30거래일 뒤 국면 (과거 실측 빈도)<br>{trtxt}</div>
<div class="warn">국면 판정은 <b>그날 알 수 있는 정보</b>(200일선·상승종목비율·고점대비낙폭)만 씁니다. 급락장은 표본이 적어 그 줄의 통계는 특히 불확실합니다. 30일 수익을 매일 겹쳐 표집하므로 '독립표본'만 실질 관측치입니다.</div></div>"""

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
        # 🌐 시장이 이렇게 되면 내 계좌는 — 베타 기반 스트레스 테스트
        stress_html=""
        if pf.get("stress"):
            srows=""
            for st in pf["stress"]:
                col="#d70015" if st["pnl"]>=0 else "#0071e3"
                mcol="#ff3b30" if st["move"]<0 else ("#8e8e93" if st["move"]==0 else "#34c759")
                pb=f'{st["prob"]:.0f}%' if st.get("prob") is not None else "-"
                stp=f'<span style="color:#ff3b30;font-weight:700">{st["stops"]}종목</span>' if st["stops"] else '<span style="color:#c7c7cc">0</span>'
                srows+=(f'<tr><td style="color:{mcol}">시장 {st["move"]:+d}%</td>'
                        f'<td style="color:#999">{pb}</td>'
                        f'<td>{st["value"]:,}</td>'
                        f'<td style="color:{col};font-weight:700">{st["pnl"]:+,}</td>'
                        f'<td>{stp}</td></tr>')
            bt=f' · 내 계좌 베타 <b>{pf["beta"]}</b>' if pf.get("beta") else ""
            stress_html=(f'<div style="margin-top:12px"><div class="leg" style="margin:0 0 2px">'
                f'🌐 <b>시장이 이렇게 되면 내 계좌는</b>{bt}</div>'
                f'<div class="tw"><table class="stab"><tr><th>시장</th><th>확률</th><th>평가액</th><th>손익</th><th>손절발동</th></tr>{srows}</table></div>'
                f'<div class="leg">각 종목의 <b>베타</b>(시장 1% 움직일 때 이 종목이 몇 % 움직였나, 최근 250일)로 추정. '
                f'내릴 때는 <b>하락베타</b>를 씁니다. 확률은 현재 국면에서 30일 뒤 그 구간이 나온 과거 빈도.<br>'
                f'※ 베타는 과거 평균일 뿐 급락장에서는 상관이 1에 수렴해 <b>더 나쁘게</b> 나오는 경향이 있습니다.</div></div>')
        H+=f"""<div class="card adv"><h2>💼 내 계좌<span class="advtag">자세히</span></h2>
<div class="stat">
  <div><div class="v" style="color:{upcol}">{'+' if up>=0 else ''}{up:,}</div><div class="k">평가손익 ({'+' if upc>=0 else ''}{upc}%)</div></div>
  <div><div class="v">{pf['value']:,}</div><div class="k">평가액(원)</div></div>
  <div><div class="v" style="color:#999">{pf['invested']:,}</div><div class="k">원금(원)</div></div>
</div>
<div class="stat" style="margin-top:6px">
  <div style="background:#fff0f0"><div class="v" style="color:#ff3b30">{pf['worst']:,}</div><div class="k">모든 손절 동시발동 시(원)</div></div>
  <div style="background:#fff0f0"><div class="v" style="color:#ff3b30">{pf['worst_pct']}%</div><div class="k">최악의 날 손실률</div></div>
  <div><div class="v">{pf['n']}</div><div class="k">보유 종목</div></div>
</div>{real_html}{stress_html}{sec_html}{eqdiv}
<div class="leg"><b>최악의 날</b> = 지금 걸어둔 감시가(손절·트레일 중 먼저 닿는 값)가 전 종목 동시 체결될 때의 추가손실. 실제 동시발동은 드물지만 <b>감내 가능한 최대 손실</b>을 미리 확인하세요.</div></div>"""

    # ★ 핵심: 매수적합도 추천
    H+='<div class="card"><h2>🟢 지금 사기 좋은 순서 (BuyFit)<span class="tb t1">검증됨</span></h2>'
    H+='<div class="leg" style="margin-top:0;margin-bottom:6px">점수 상위 풀에서, 최근 조정받아 반등 여지가 큰 종목을 앞으로. (백테스트에서 <b>관찰된</b> 단기 평균회귀 · 생존편향 미보정 → 아래 한계 참고)</div>'
    for k,i in enumerate(buylist):
        cls="brk" if k==0 else ("brk r2" if k==1 else ("brk r3" if k==2 else "brk rx"))
        tcol = "#34c759" if i["timing"]>=66 else ("#ff9500" if i["timing"]>=33 else "#ff3b30")
        ddtxt = f'{i["dd"]}%' if i["dd"] is not None else "-"
        v=i["verdict"]; pr=i.get("prob")
        # 두 번째 줄은 '왜 이 순위인가'가 아니라 '얼마나 믿을 만한가'를 보여준다
        bits=[f'점수 {i["score"]:.0f}({i["rank"]}위)']
        if pr: bits.append(f'승률 {pr["win"]:.0f}%<span style="color:#bbb">(기준 {pr["base_win"]:.0f})</span>')
        if pr: bits.append(f'손절 {pr["p_stop"]:.0f}%')
        nrisk=len(i["bear"]["risks"])
        if nrisk: bits.append(f'<span style="color:#ff9500">반대 {nrisk}</span>')
        H+=(f'<div class="bi" onclick="openD(\'{i["code"]}\')">'
            f'<div class="{cls}">{k+1}</div>'
            f'<div class="bmid"><div class="bn">{i["name"]} '
            f'<span class="gr g{v["grade"]}" style="vertical-align:middle">{v["grade"]}</span></div>'
            f'<div class="bsub">{" · ".join(bits)}</div>'
            f'<div class="bsub" style="color:#aaa">60일고점대비 {ddtxt} · '
            f'<span style="color:{tcol};font-weight:700">타이밍 {i["timing"]:.0f}</span></div></div>'
            f'<div class="fit"><div class="fitv">{i["buyfit"]:.0f}</div><div class="fitk">BuyFit</div></div>'
            f'<span class="chev">›</span></div>')
    H+=('<div class="leg">BuyFit(0~100): 펀더멘털 점수 + 진입타이밍 종합. 높을수록 "지금" 매수 적합.<br>'
        '<b>A~D 등급</b>은 초과승률·기대값·표본두께·반대논리·시장국면을 합친 <b>근거의 두께</b>입니다. '
        '"오른다"는 예측이 아니라 "같은 잣대로 줄 세웠을 때 어디쯤인가"입니다.</div></div>')

    # 📰 시장 뉴스
    mnews=m.get("news") or []
    if mnews:
        nrows=""
        for a in mnews[:5]:
            cc="cpos" if a["sent"]>0 else ("cneg" if a["sent"]<0 else "cneu")
            lbl="호재" if a["sent"]>0 else ("악재" if a["sent"]<0 else "중립")
            nrows+=(f'<a class="nw" href="{a["link"]}" target="_blank" rel="noopener">'
                    f'<div class="nwt"><span class="chip {cc}">{lbl}</span>{a["title"]}</div>'
                    f'<div class="nws">{a["src"]} · {a["pub"]}</div></a>')
        H+=(f'<div class="card adv"><h2>📰 시장 뉴스<span class="tb t2">참고</span></h2>{nrows}'
            f'<div class="leg">호재/악재 표시는 <b>제목의 키워드만</b> 본 자동 분류입니다. 점수에는 반영하지 않으며, '
            f'"내 판단을 뒤집을 사건이 있나" 확인용입니다.</div></div>')

    if holds:
        H+='<div class="card"><h2>💼 보유 현황</h2>'+"".join(hold_row(i) for i in holds)+'</div>'

    H+='''<div class="card"><h2>🔎 전체 종목 (검색·정렬)</h2>
<input id="q" placeholder="종목명 검색 (예: 삼성)" oninput="renderStocks()" style="width:100%;padding:11px;font-size:15px;border:1px solid #ddd;border-radius:10px;margin-bottom:8px">
<div id="sortBtns" style="display:flex;gap:6px;margin-bottom:6px"></div>
<div id="stockList"></div></div>'''

    H+=f"""<div class="card adv"><details><summary style="font-size:17px;font-weight:700;cursor:pointer">📊 시장 점수 분포</summary>
<div class="date" style="margin-top:8px">평균 {m['avg']} · 중앙값 {m['median']} · 최고 {m['max']}</div>
<div class="hist" id="mhist"></div>
<div class="leg">100종목 점수 분포. 상위 추천권은 보통 65점+.</div></details></div>"""

    T1X='<span class="tb t1">검증됨</span>'
    T2X='<span class="tb t2">참고</span>'
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
        H+=f"""<div class="card adv"><h2>🔬 현실적 기대수익{T2X}</h2>
<div class="note" style="background:#f0f7ff">백테스트 숫자를 그대로 믿지 않기 위해 <b>거래비용을 현실화하고 생존편향을 일부</b> 보정한 결과:<br><br>
이상적 가정 연 <b>{rb['base_ann']:+.0f}%</b> → 보수적 가정 연 <b style="color:#0071e3">{rb['concl_ann']:+.0f}%</b> (비용·편향으로 <b>{rb['erosion']:.0f}%p</b> 증발)</div>
<details style="margin-top:6px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">가정별 연율 수익 자세히</summary>
<div class="date" style="margin:8px 0 4px">보수적일수록 아래:</div>
{scen_rows}
{surv_html}</details>
<div class="warn">※ {pit_line}<br>이 표본은 <b>강세장 비중이 큰 2018~2026</b> 구간이니 <b>실전 기대는 더 보수적으로</b> 잡으세요. 상장폐지 {rb.get('delisted_n',0):,}종목은 애초에 데이터에 없습니다.</div></div>"""
    else:
        H+='<div class="card adv"><h2>🔬 전략 검증</h2><div class="note">보완 백테스트 미실행. Actions의 robust-backtest를 한 번 돌리면 생존편향·비용 보정 기대수익이 여기 표시됩니다.</div></div>'
    # 🔍 이 숫자를 믿어도 되나 — 등급·확률을 과거 데이터로 실제 검증한 결과
    vf=d.get("verify")
    if vf:
        NOW_CLS=' class="now"'
        grows="".join(
          f'<tr{NOW_CLS if g["grade"]=="A" else ""}><td>{g["grade"]}등급</td>'
          f'<td>{g["win"]:.1f}%</td><td>{g["avg"]:+.2f}%</td><td>{g["p_stop"]:.1f}%</td>'
          f'<td style="color:#bbb">{g["indep"]:,}</td></tr>' for g in vf.get("grades",[]))
        srows=""
        for r in vf.get("sims",[]):
            hl=("background:#eef5ff;font-weight:700" if "대조군" in r["name"] else "")
            c="#34c759" if r["ann"]>=20 else ("#ff9500" if r["ann"]>=10 else "#ff3b30")
            srows+=(f'<tr style="{hl}"><td style="font-size:11px">{r["name"]}</td>'
                    f'<td style="color:{c};font-weight:700">{r["ann"]:+.1f}%</td>'
                    f'<td>{r["sharpe"]:.2f}</td><td>{r["worst"]:+.0f}%</td></tr>')
        cal_ok=sum(1 for c in vf.get("calibration",[]) if c["ok"])
        cal_n=len(vf.get("calibration",[]))
        H+=f"""<div class="card adv" style="border:2px solid #5856d6"><h2>🔍 이 숫자를 믿어도 되나<span class="tb t1">검증됨</span></h2>
<div class="leg" style="margin-top:0">대시보드가 말하는 등급·확률을 <b>과거 시점 정보만으로 다시 계산해</b> 실제 결과와 대조했습니다.
({vf['period'][0]}~{vf['period'][1]} · {vf['n_eval']:,}건 · 미래참조 차단)</div>
<div class="note" style="background:#fff6f6;margin-top:8px">❌ <b>승률 예측은 실패했습니다.</b>
예측 승률과 실제 결과의 상관은 <b>{vf['cal_corr']:+.4f}</b>(사실상 0), 캘리브레이션은 {cal_n}구간 중 <b>{cal_ok}개만</b> 맞았습니다.
개별 종목의 30일 승률은 <b>이 데이터로 예측되지 않습니다</b> — 그래서 화면의 승률은 참고용 기록으로만 두세요.</div>
<div class="note" style="background:#f2fbf4;margin-top:6px">✅ <b>기대수익 방향은 살아있습니다.</b>
등급별 실제 30일 평균수익이 A {vf['grades'][0]['avg']:+.2f}% → D {vf['grades'][-1]['avg']:+.2f}%로 갈렸고,
손절 발동률도 A {vf['grades'][0]['p_stop']:.0f}% vs D {vf['grades'][-1]['p_stop']:.0f}%로 차이가 납니다.</div>
<details style="margin-top:10px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">검증 표 자세히 보기 (등급별 결과 · 전략 시뮬레이션)</summary>
<div class="leg" style="margin-top:8px"><b>등급별 실제 결과</b></div>
<div class="tw"><table class="stab"><tr><th>등급</th><th>승률</th><th>평균수익</th><th>손절률</th><th>독립표본</th></tr>{grows}</table></div>
<div class="leg" style="margin-top:10px"><b>이 방법으로 투자했다면 (30일 리밸런싱·비용 0.3%)</b></div>
<div class="tw"><table class="stab"><tr><th>전략</th><th>연율</th><th>Sharpe</th><th>최악</th></tr>{srows}</table></div>
<div class="warn">파란 줄이 <b>대조군</b>입니다. 등급을 무시하고 BuyFit 상위5만 사도 결과가 같거나 더 낫습니다 →
<b>등급은 추천을 요약할 뿐 추가 수익을 만들지 못합니다.</b> 등급은 "얼마나 확신할지"를 보는 용도로만 쓰세요.<br>
리밸런싱 {vf['sims'][0]['n_rebal']}회는 통계적으로 적은 표본입니다. 연율 차이는 우연일 수 있습니다.</div></details></div>"""

    # 🎯 익절·손절이 실제로 얼마나 자주 오는가 — 380만 건 전수 조사
    # 사용자가 "익절을 못 해서 손절만 한다"고 한 문제를 숫자로 확인하고 답하는 카드
    pl=m.get("poslab")
    if pl:
        nb=pl.get("new",{}).get("1-10") or pl.get("new",{}).get("21-100")
        rows=""
        for lab_ in pl["ret_labels"]:
            st=pl["by_ret"].get(lab_)
            if not st: continue
            # 지금 가격에서 5% 더 밀릴 확률 vs 5% 더 오를 확률 — 들고 갈지 팔지의 핵심
            w=st["give5"]+st["gain5"]
            gw=st["gain5"]/w*100 if w else 50
            hot=lab_ in ("+25~40%","+40%↑")
            rows+=(f'<div class="bigbar-wrap"><span class="bl"{" style=font-weight:800" if hot else ""}>{lab_}</span>'
                   f'<div class="bigbar" style="background:#ffd9d6"><div style="width:{gw:.0f}%;background:#34c759"></div></div>'
                   f'<span class="bv" style="font-size:11px">{st["gain5"]:.0f}/{st["give5"]:.0f}</span></div>')
        ov=pl["overall"]
        # 생존편향을 얼마나 메웠는지는 사실대로 적는다 — 상장폐지 표본이 들어왔나 아닌가
        dd_=pl.get("dead")
        rev=pl['by_ret'].get('-10%↓',{}).get('up','-')
        if dd_:
            rd=dd_["by_ret"].get('-10%↓',{}).get('up')
            surv=(f'· 위 표는 <b>살아남은 {pl.get("n_live") or 100}종목</b> 기준입니다. '
                  f'2018년 이후 <b>상장폐지된 {dd_["n_stocks"]}종목</b>을 따로 재보니 '
                  f'상승확률 <b>{dd_["overall"]["up"]}%</b>로 {abs(dd_["up_gap"]):.1f}%p 낮았습니다 — '
                  f'<b>이 차이가 곧 생존편향의 크기</b>입니다. '
                  f'둘을 섞지 않은 이유는 사라진 종목이 대부분 소형주·우선주라, '
                  f'표본 수대로 합치면 대형주 판단에 소형주 부도율을 밀어넣는 꼴이 되기 때문입니다. '
                  f'<b>실제 값은 두 숫자 사이 어딘가입니다.</b>')
        else:
            surv=(f'· 표본은 <b>지금까지 살아남은 {pl.get("n_live") or 100}종목</b>입니다. 사라진 종목이 빠져 '
                  f'<b>하락 꼬리가 실제보다 얇습니다</b>.')
        # 생존편향을 숨기지 않는다 — 특히 '많이 빠진 뒤 반등' 칸이 이 편향을 가장 크게 받는다
        deadbox=""
        if dd_ and dd_["by_ret"].get('-10%↓'):
            rd=dd_["by_ret"]['-10%↓']['up']
            deadbox=(f'<div class="warn" style="background:#f4f0ff;color:#4b3f8f">'
                     f'⚠ <b>이 표가 낙관적인 만큼은 알고 보세요.</b> 위 표에서 “-10% 아래”의 상승확률이 '
                     f'<b>{rev}%</b>로 가장 높게 나오지만, 2018년 이후 <b>상장폐지된 {dd_["n_stocks"]}종목</b>의 '
                     f'같은 칸은 <b>{rd}%</b>였습니다({rd - pl["by_ret"]["-10%↓"]["up"]:+.1f}%p). '
                     f'끝내 회복 못 한 종목은 살아있는 표본에 남아있지 않기 때문입니다.<br>'
                     f'👉 그래서 <b>“많이 빠졌으니 곧 오른다”로 손절을 미루지 마세요.</b> '
                     f'종목 판단에서도 이 칸의 “버텨라” 무게를 절반만 반영합니다.</div>')
        tip=""
        if nb:
            odds=nb["hit_stop"]/nb["hit_tp"] if nb["hit_tp"] else 0
            tip=(f'<div class="warn">👉 갓 산 자리에서 <b>30거래일 안에 -10% 손절선을 밟을 확률 '
                 f'{nb["hit_stop"]}%</b>, <b>+25% 익절선을 밟을 확률 {nb["hit_tp"]}%</b>. '
                 f'손절이 익절보다 <b>{odds:.1f}배</b> 자주 옵니다. '
                 f'“오를 땐 못 팔고 손절만 한다”는 느낌은 착각이 아니라 <b>구조</b>입니다.<br>'
                 f'그래서 규칙은 두 가지로 답합니다 — ① <b>+25%에서 1/3 익절</b>로 이긴 거래를 확정하고, '
                 f'② <b>고점수익 +15%↑면 트레일을 -5%, +30%↑면 -3%로 조여</b> 남은 이익을 지킵니다. '
                 f'손절 횟수는 못 줄여도, <b>한 번 이길 때 손에 남는 크기</b>는 늘릴 수 있습니다.</div>')
        H+=f"""<div class="card" style="border:2px solid #ff9500"><h2>🎯 익절과 손절, 실제 확률{T1X}</h2>
<div class="leg" style="margin-top:0">2018년 이후 모든 (종목 × 진입일 × 보유일) <b>{ov['n']:,}건</b>을 전부 세어
“지금 이 수익률 자리에서 앞으로 30거래일에 무슨 일이 있었나”를 만든 표입니다. 전략과 무관하게 가격만 씁니다.</div>
{tip}
<div class="leg" style="margin-top:10px"><b>지금 수익률별 — 여기서 5% 더 오를 확률 vs 5% 더 밀릴 확률</b></div>
{rows}
<div class="leg">초록=더 오름 / 빨강=더 밀림. 숫자는 <b>오름/밀림</b>(%). 둘 다 일어난 경우가 많아 합이 100을 넘습니다.</div>
<div class="warn" style="background:#fff8e1;color:#946200">많이 오를수록 <b>양쪽이 같이 커집니다</b>.
+40% 이상에서는 5% 더 밀릴 확률이 {pl['by_ret'].get('+40%↑',{}).get('give5','-')}%까지 올라갑니다.
큰 수익을 들고 있다면 “더 갈까”가 아니라 <b>“얼마를 지킬까”</b>가 맞는 질문입니다.</div>
{deadbox}
<details style="margin-top:6px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">이 표의 한계</summary>
<div class="leg" style="margin-top:8px;line-height:1.85">
{surv}<br>
· 관측이 겹칩니다(하루씩 밀며 세므로 30일 창이 29/30 동일). 겹치지 않는 날짜 묶음은 최대 {ov['indep']}개뿐이라
표본 수 {ov['n']:,}건은 <b>독립 표본이 아닙니다</b>.<br>
· 전략이 실제로 고른 종목이 아니라 <b>모든 종목</b>이 대상입니다. 순위 보정은 따로 붙였습니다
(추천 1-10위 밴드 상승확률 {pl['by_rank'].get('1-10',{}).get('up','-')}% vs 전체 {ov['up']}%).
</div></details></div>"""

    # 🧪 전략 점검 — 규칙 하나하나가 값어치를 하는지 따로 잰 결과
    lab=m.get("lab")
    if lab and lab.get("why"):
        w=lab["why"]; tot=max(1,sum(w.values()))
        bars="".join(
          f'<div class="bigbar-wrap"><span class="bl">{k}</span>'
          f'<div class="bigbar"><div style="width:{w[k]/tot*100:.0f}%;background:{c}"></div></div>'
          f'<span class="bv">{w[k]/tot*100:.0f}%</span></div>'
          for k,c in [("트레일","#ff9500"),("순위이탈","#0071e3"),("손절","#ff3b30"),("익절","#c7c7cc")])
        H+=f"""<div class="card adv"><h2>🧪 전략 점검{T1X}</h2>
<div class="leg" style="margin-top:0">규칙을 하나씩 바꿔가며 <b>train({lab['train'][0]}~{lab['train'][1]})에서 고르고
test({lab['test'][0]}~{lab['test'][1]})로 확인</b>했습니다. train에서 좋아도 test에서 나빠지면 채택하지 않았습니다.</div>
<div class="leg" style="margin-top:10px"><b>실제로 무엇이 포지션을 끝냈나</b> (거래 {lab['n']}건 · 평균보유 {lab['avg_hold']:.0f}일)</div>
{bars}
<div class="warn">👉 <b>고정 손절 -10%는 거의 작동하지 않습니다.</b> 매수 후 +2.2%만 올라도
트레일링(고점 -8%)이 손절선을 넘어서기 때문입니다. 증권사 앱에 -10% 하나만 걸어두면
이 전략을 실행하는 게 아니니, <b>주가가 오르면 스탑을 고점 기준으로 올리세요</b> (고점×0.92 → 고점수익 +15%↑면 ×0.95 → +30%↑면 ×0.97).</div>
<details style="margin-top:6px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">바꾸려다 되돌린 것들 (왜 안 바꿨나)</summary>
<div class="leg" style="margin-top:8px;line-height:1.85">
<b>· 팩터 가중치 — 그대로 둠</b><br>
백테스트만 보면 "모멘텀만" 쓰는 게 최고(test 연 +84.8%)였지만, <b>연도별로 나눠보니 모멘텀의
전체 IC는 −0.010으로 오히려 음수</b>였습니다(2018년 −0.121, 2021년 −0.129). 2024~25년에만 통한 것입니다.
반면 <b>가치는 IC +0.029로 가장 강하고 9년 중 6년 양수</b> — 지금의 가중 0.32가 맞습니다.<br><br>
<b>· 트레일링 폭 — 그대로 둠</b><br>
-12%가 기본 설정에선 가장 좋았지만, 종목수·이탈순위·보유기간을 바꿔가며 7개 조합으로 교차하니
<b>train·test 동시 1위는 7번 중 1번뿐</b>이었습니다. 차이가 표본오차 안입니다.
다만 <b>트레일링을 아예 빼면 양쪽 모두 나빠져</b>, 있는 것 자체는 정당합니다.<br><br>
<b>· 전량 익절 — 여전히 안 씀 (부분 익절만 채택)</b><br>
+15%/+20%/+30%/+50%에서 <b>다 파는</b> 방식은 train·test 어디서도 나아지지 않았습니다.
수익이 소수의 큰 상승에서 나오는데 윗쪽을 잘라버리기 때문입니다.
그래서 <b>+25%에서 1/3만 팔고 나머지는 조인 트레일로 끌고 가는</b> 방식만 넣었습니다.<br><br>
<b>· 변동성 역가중 — 기각</b> (train Sharpe 0.67→0.35). 이 종목군에선 잘 오른 게 변동성 큰 종목이었습니다.<br>
<b>· 약세장 신규매수 중단 — 기각</b> (train은 개선되나 test 연 +61.8%→+46.2%). 약세장 뒤 반등을 놓칩니다.
</div></details></div>"""

    H+="""<div class="card adv"><h2>📚 방법론 · 이 추천의 근거</h2>
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
const META=DATA.meta||{};
// data.json은 100종목이 공유하는 설명문을 meta에 한 번만 담는다 → 여기서 되돌려 붙인다
const catOf=t=>(META.catalysts||{})[t]||{};
const riskTxt=r=>r.text||(META.risks||{})[r.tag]||'';
const noteOf=k=>(META.verdict_notes||{})[k]||(META.action_notes||{})[k]||'';
// 이격률 구간별 과거 통계 — 모달에서 '지금 이 구간은 어땠나'를 데이터로 말하려고
const POSD=((DATA.market||{}).poslab||{}).by_disp||null;
const DSPB=((DATA.market||{}).poslab||{}).dsp_bins||[-99,-5,-2,2,5,10,99];
const DSPL=((DATA.market||{}).poslab||{}).dsp_labels||[];
function bandOf(v){
  for(let k=0;k<DSPL.length;k++){ if(v>DSPB[k]&&v<=DSPB[k+1]) return DSPL[k]; }
  return DSPL[v<=DSPB[0]?0:DSPL.length-1];
}
const factMeta=k=>(META.factors||{})[k]||{};
function invalList(b){
  const tpl=META.invalidation||[], iv=b.invalidation;
  if(Array.isArray(iv))return iv;                       // meta 없이 만든 구버전 호환
  const stop=iv&&iv.stop, bull=iv&&iv.bull_first;
  return tpl.map(t=>{
    if(t.indexOf('{stop}')>=0){return stop?t.replace('{stop}',(+stop).toLocaleString()):null;}
    if(t.indexOf('{bull}')>=0){return bull?t.replace('{bull}',bull):null;}
    return t;
  }).filter(Boolean);
}
function mustList(b){
  const out=(b.drivers||[]).map(d=>d.must||catOf(d.title).must).filter(Boolean);
  return out.concat(b.extra_musts||[]);
}
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
(function(){const m=DATA.market,mx=Math.max(...m.hist,1),el=document.getElementById('mhist');
 m.hist.forEach((c,k)=>{const h=Math.round(c/mx*92)+4,lab=m.hist_labels[k],col=(+lab>=65)?'#ff9500':'#0071e3';
 el.innerHTML+=`<div class="hcol"><div class="hnum">${c}</div><div class="hbar" style="height:${h}px;background:${col}"></div><div class="hlab">${lab}</div></div>`;});})();
// 🌡 시장 지수(등가중 100종목) 최근 120거래일
(function(){const rg=DATA.regime,el=document.getElementById('ridx');
 if(!rg||!el||!rg.index||rg.index.length<2)return;
 const W=Math.min(window.innerWidth-64,470);
 el.innerHTML='<div class="date" style="margin-bottom:2px">KOSPI100 등가중 지수 (120거래일, 시작=100)</div>'+
   lineChart(rg.index.map(p=>p.date),rg.index.map(p=>p.v),rg.current.color,W,130,
             {refs:[{v:100,color:'#c7c7cc',label:'시작'}]});})();
// 계좌 손익 추이 차트 (보유가 2거래일 이상 쌓이면 표시)
(function(){const pf=DATA.portfolio;const el=document.getElementById('eqchart');
 if(!pf||!el||!pf.equity||pf.equity.length<2)return;
 const W=Math.min(window.innerWidth-64,470);
 el.innerHTML='<div class="date" style="margin-bottom:2px">내 계좌 손익률 추이(%)</div>'+
   lineChart(pf.equity.map(p=>p.date),pf.equity.map(p=>p.pnl),'#0071e3',W,140,{refs:[{v:0,color:'#c7c7cc',label:'0%'}]});})();

// 🔎 전체 종목 검색·정렬 (앱처럼 자유 탐색)
let SORT='score';
(function(){const sb=document.getElementById('sortBtns');if(!sb)return;
 [['score','종합점수'],['buyfit','BuyFit'],['conf','판단등급'],['timing','타이밍']].forEach((o,k)=>{
  const b=document.createElement('button');b.textContent=o[1];
  b.style.cssText='flex:1;padding:8px;border:0;border-radius:9px;background:'+(k==0?'#0071e3':'#e5e5ea')+';color:'+(k==0?'#fff':'#333')+';font-weight:700;font-size:13px';
  b.onclick=()=>{SORT=o[0];sb.querySelectorAll('button').forEach(x=>{x.style.background='#e5e5ea';x.style.color='#333'});b.style.background='#0071e3';b.style.color='#fff';renderStocks();};
  sb.appendChild(b);});})();
function renderStocks(){
 const el=document.getElementById('stockList');if(!el)return;
 const q=(document.getElementById('q').value||'').trim();
 let arr=DATA.items.slice();
 if(q)arr=arr.filter(i=>i.name.indexOf(q)>=0||(i.sector&&i.sector.indexOf(q)>=0)||(i.industry&&i.industry.indexOf(q)>=0));
 const key=i=>SORT==='conf'?(i.verdict?i.verdict.conf:0):(i[SORT]||0);
 arr.sort((a,b)=>key(b)-key(a));
 el.innerHTML=arr.slice(0,60).map(i=>{
   const sig=i.signal?`<span style="font-size:11px;margin-left:4px">${i.signal}</span>`:'';
   const pnl=(i.pnl!==''&&i.pnl!=null)?`<span class="${i.pnl>=0?'pp':'pn'}" style="font-size:13px;margin-right:6px">${i.pnl>=0?'+':''}${i.pnl}%</span>`:'';
   const sec=i.sector?`<div style="font-size:11px;color:#999;margin-top:1px">${i.sector}</div>`:'';
   const g=i.verdict?`<span class="gr g${i.verdict.grade}" style="margin-right:7px">${i.verdict.grade}</span>`:'';
   return `<div class="item" onclick="openD('${i.code}')"><div class="rk">${i.rank}</div>`+
     `<div style="flex:1;min-width:0"><div style="font-weight:700;font-size:15px">${i.name}${sig}</div>${sec}</div>${pnl}${g}`+
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
 // 카드가 17개라 한 화면에 다 쌓으면 9번을 스크롤해야 한다 → 3묶음으로 나눈다
 // A 판단(지금 뭘 할까) · B 근거(왜 그런가) · C 숫자(얼마나 확실한가)
 let h='<div class="shandle"></div>';
 let A='',B='',C='';
 let pnl='';if(i.pnl!==''&&i.pnl!=null){const c=i.pnl>=0?'pp':'pn';pnl=` <span class="${c}">${i.pnl>=0?'+':''}${i.pnl}%</span>`;}
 h+=`<div class="dh"><div><div style="font-size:24px;font-weight:800">${i.name}${i.sector?` <span style="font-size:12px;font-weight:600;color:#0071e3;background:#eef;border-radius:7px;padding:2px 7px;vertical-align:middle">${i.sector}</span>`:''}</div>
   <div class="date">점수 ${i.score.toFixed(0)} (전체 ${i.rank}위) · 매수적합 ${i.buyrank}위 · ${i.price.toLocaleString()}원${pnl}${i.industry?' · '+i.industry:''}</div></div>
   <button class="x" onclick="closeD()">✕</button></div>`;
 if(i.signal)h+=`<div class="note"><b>${i.signal}</b> · ${i.reason}</div>`;

 // 🧭 포지션 배분 — 이 화면에서 제일 먼저 봐야 할 것
 // "매수/유지/매도 각각 몇 %이고, 그 퍼센트가 어느 숫자에서 나왔나"
 const sp=i.split;
 if(sp){
   const SC={buy:'#0071e3',hold:'#34c759',sell:'#ff3b30'};
   const ord=['buy','hold','sell'].sort((a,b)=>sp.pct[b]-sp.pct[a]);
   // 가로 막대 하나에 셋을 붙여 비율이 한눈에 보이게
   let bar='';
   ['sell','hold','buy'].forEach(k=>{
     if(sp.pct[k]<=0)return;
     const lb=sp.pct[k]>=30?`${sp.labels[k]} ${sp.pct[k]}%`:(sp.pct[k]>=12?sp.pct[k]+'%':'');
     bar+=`<div style="width:${sp.pct[k]}%;background:${SC[k]};display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#fff;white-space:nowrap;overflow:hidden">${lb}</div>`;
   });
   let chips='';
   ord.forEach(k=>{
     const on=k===sp.top;
     chips+=`<div style="flex:1;text-align:center;padding:7px 2px;border-radius:11px;background:${on?SC[k]:'#f2f2f7'};color:${on?'#fff':'#555'}">
       <div style="font-size:20px;font-weight:800;line-height:1.1">${sp.pct[k]}%</div>
       <div style="font-size:11px;font-weight:700">${sp.labels[k]}</div></div>`;
   });
   // 항목별 기여 — 어느 근거가 어느 쪽으로 밀었나
   let rows='';
   sp.parts.forEach(p=>{
     const dir=Math.abs(p.sell)>=Math.abs(p.buy)&&Math.abs(p.sell)>=Math.abs(p.hold)?'sell':
               (Math.abs(p.buy)>=Math.abs(p.hold)?'buy':'hold');
     const tag=[['buy',p.buy],['hold',p.hold],['sell',p.sell]]
       .filter(x=>Math.abs(x[1])>=0.5)
       .map(x=>`<span style="color:${SC[x[0]]};font-weight:800">${sp.labels[x[0]]} ${x[1]>0?'+':''}${x[1]}</span>`).join(' · ');
     rows+=`<div class="rk2" style="align-items:flex-start"><span class="sev" style="background:${SC[dir]};flex:none">${p.k.startsWith('규칙')?'규칙':(p.k.startsWith('출발')?'기본':'근거')}</span>
       <div class="rtx"><b>${p.k}</b><div style="font-size:11px;margin:2px 0">${tag||'영향 없음'}</div>
       <div class="rev">${p.why}</div></div></div>`;
   });
   // 익절·손절 자리 — 사용자가 가장 중요하다고 한 부분이라 접지 않고 펼쳐 둔다
   let lv='';
   (sp.levels||[]).forEach(L=>{
     const c=L.k.includes('익절')?'#34c759':'#ff3b30';
     lv+=`<div class="rk2"><span class="sev" style="background:${c}">${L.p}%</span>
       <div class="rtx"><b>${L.k}</b>${L.v?` · <b style="color:${c}">${L.v.toLocaleString()}원</b>`:''}
       <div class="rev">${L.why}</div></div></div>`;
   });
   // 이격률 — "평균가격에서 얼마나 벌어졌나". 문구를 임의로 정하지 않고
   // 판단 엔진이 실제로 쓴 숫자(과거 같은 구간의 '5% 더 밀릴 확률')로 말한다.
   let dsp='';
   if(i.disp20!=null){
     const dp=(sp.parts||[]).find(x=>x.k.indexOf('이격률')>=0);
     const v=i.disp20;
     // 이 항목이 매도 쪽에 몇 점을 줬는지가 곧 위험 신호의 세기다
     const push=dp?dp.sell:0;
     const c=push>=6?'#ff3b30':(push>=2.5?'#ff9500':'#8e8e93');
     const pb=(POSD||{})[bandOf(v)], base=(POSD||{})['-2~+2%'];
     let say;
     if(pb&&base) say=`과거 이 구간에서 <b>5% 더 밀린 경우 ${pb.give5}%</b>`
                     +` (평균 근처였을 땐 ${base.give5}%)`
                     +(push>=2.5?' — <b>지킬 준비를 하세요</b>':' — 특별한 신호 없음');
     else say=(v>=5?'평소보다 떠 있습니다':(v<=-5?'평소보다 눌려 있습니다':'평균 근처입니다'));
     dsp=`<div class="rk2"><span class="sev" style="background:${c}">${v>=0?'+':''}${v}%</span>
       <div class="rtx"><b>20일 평균가격 대비</b><div class="rev">${say}</div></div></div>`;
   }
   A+=`<div class="card" style="border:2px solid ${SC[sp.top]}"><h2>🧭 지금 뭘 할까<span class="tb t2">근거 합산</span></h2>
     ${sp.act?`<div class="note" style="background:${SC[sp.top]}12;border-left:3px solid ${SC[sp.top]};font-weight:700;margin:0 0 8px">👉 ${sp.act}</div>`:''}
     <div style="display:flex;gap:6px;margin:4px 0 8px">${chips}</div>
     <div style="display:flex;height:22px;border-radius:7px;overflow:hidden;margin-bottom:6px">${bar}</div>
     <div class="leg" style="margin-top:0">${noteOf('split')||''}</div>
     ${lv?`<div style="margin-top:10px"><div style="font-size:13px;font-weight:800;margin-bottom:4px">🎯 익절 · 손절 자리와 닿을 확률</div>${lv}</div>`:''}
     ${dsp?`<div style="margin-top:8px">${dsp}</div>`:''}
     <details style="margin-top:8px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">이 퍼센트는 이렇게 나왔습니다 (항목별 근거)</summary>
       <div style="margin-top:4px">${rows}</div>
       ${sp.cell_label?`<div class="leg">지금 자리 = <b>${sp.cell_label}</b>. ${noteOf('cell')||''}</div>`:''}
       <div class="leg">${noteOf('survivor')||''}</div></details></div>`;
 }

 // ⚖️ 판단 요약 — 등급 · 확신도 · 지금 할 행동
 const v=i.verdict, pr=i.prob;
 const MODE={buy:['#0071e3','신규 매수 관점'],hold:['#34c759','보유 관점'],
             sell:['#ff3b30','청산 관점'],wait:['#ff9500','대기 관점']};
 const md=MODE[v.mode]||MODE.buy;
 let vparts='';
 v.parts.forEach(p=>{
   const c=p.d>=0?'#34c759':'#ff3b30';
   vparts+=`<div class="rk2"><span class="sev" style="background:${c}">${p.d>=0?'+':''}${p.d}</span>`+
     `<div class="rtx"><b>${p.k}</b> · ${p.v}<div class="rev">${noteOf(p.k)}</div></div></div>`;
 });
 A+=`<div class="card adv" style="border:2px solid ${md[0]}"><h2>⚖️ 판단 요약<span class="advtag">자세히</span><span class="tb t2">참고</span></h2>
   <div style="display:flex;align-items:center;gap:12px;margin:6px 0 10px">
     <div class="gr g${v.grade}" style="width:46px;height:46px;border-radius:13px;font-size:24px">${v.grade}</div>
     <div style="flex:1;min-width:0">
       <div style="font-size:15px;font-weight:800;color:${md[0]}">${md[1]} · ${v.action}</div>
       <div class="date" style="margin-top:2px">확신도 ${v.conf}/100 — ${v.grade_text}</div></div></div>
   <div class="note" style="line-height:1.8;background:#f7f7fa">${v.line}</div>
   <details style="margin-top:6px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">확신도는 이렇게 계산됐습니다</summary>
     <div style="margin-top:4px">${vparts}</div>
     <div class="leg">기본 50점에서 시작해 위 항목을 더하고 뺍니다. A≥68 · B≥56 · C≥44 · D&lt;44.</div></details>
   ${v.proven?'':'<div class="warn">⚠ 이 종목의 우위는 <b>통계적으로 입증되지 않았습니다</b>(신뢰구간 하한이 기준선 아래). 이 데이터로는 거의 모든 종목이 그렇습니다. 등급은 <b>근거의 두께 순위</b>일 뿐 "오른다"는 뜻이 아닙니다.</div>'}</div>`;

 // BuyFit 요약 카드
 const tcol=i.timing>=66?'#34c759':(i.timing>=33?'#ff9500':'#ff3b30');
 A+=`<div class="card adv"><h2>🎯 지금 매수 적합도<span class="advtag">자세히</span><span class="tb t1">검증됨</span></h2><div class="stat">
   <div><div class="v" style="color:#34c759">${i.buyfit.toFixed(0)}</div><div class="k">BuyFit(0~100)</div></div>
   <div><div class="v" style="color:${tcol}">${i.timing.toFixed(0)}</div><div class="k">진입타이밍</div></div>
   <div><div class="v">${i.dd!=null?i.dd:'-'}%</div><div class="k">60일고점대비</div></div></div>
   <div class="leg">타이밍↑ = 최근 조정으로 반등여지 큼(평균회귀). 점수가 높아도 신고가권이면 타이밍은 낮게 나옵니다.</div></div>`;

 // 📈 상승 시나리오
 const bl=i.bull;
 let drv='';
 (bl.drivers||[]).forEach(dv=>{
   const why=dv.why||catOf(dv.title).why||'';
   drv+=`<div class="rk2"><span class="sev" style="background:#34c759">${Math.round(dv.val)}</span>`+
     `<div class="rtx"><b>${dv.title}</b> <span style="color:#999">(${dv.name})</span><div class="rev">${why}</div></div></div>`;
 });
 const musts=mustList(bl);
 let mustHtml='';
 if(musts.length)mustHtml='<div class="note" style="background:#fffdf0;margin-top:8px"><b>이 논리가 성립하려면 (전제조건)</b><br>'+
   musts.map(x=>'· '+x).join('<br>')+'</div>';
 let tgtHtml='';
 if(bl.target&&bl.target.price){
   tgtHtml=`<div class="stat" style="margin-top:8px">
     <div><div class="v" style="color:#34c759;font-size:18px">${bl.target.price.toLocaleString()}</div><div class="k">강세 목표(상위25%)</div></div>
     <div><div class="v" style="font-size:18px">${bl.target.p_up10}%</div><div class="k">+10% 이상 확률</div></div>
     <div><div class="v" style="font-size:18px">${bl.target.p_up20}%</div><div class="k">+20% 이상 확률</div></div></div>`;
 }
 B+=`<div class="card" style="background:#f5fbf6"><h2>📈 상승 시나리오<span class="tb t2">참고</span></h2>
   <div class="note" style="background:transparent;padding:0;line-height:1.8">${bl.summary}</div>
   ${drv?'<div style="margin-top:6px">'+drv+'</div>':''}${tgtHtml}${mustHtml}
   <div class="leg">전제조건이 깨지면 상승 논리도 같이 깨집니다. 다음 실적 발표 때 여기부터 확인하세요.</div></div>`;

 // 📉 하락 시나리오 · 반대 논리
 const be=i.bear;
 const SEVN={high:['sHigh','치명'],med:['sMed','주의'],low:['sLow','참고']};
 let rsk='';
 (be.risks||[]).forEach(r=>{
   const sv=SEVN[r.sev]||SEVN.low;
   rsk+=`<div class="rk2"><span class="sev ${sv[0]}">${sv[1]}</span>`+
     `<div class="rtx"><b>${r.tag}</b> — ${riskTxt(r)}<div class="rev">근거: ${esc(r.evidence)}</div></div></div>`;
 });
 if(!rsk)rsk='<div class="leg">자동 점검에 걸린 항목 없음.</div>';
 const bs=be.bear_score, bcol=bs>=45?'#ff3b30':(bs>=22?'#ff9500':'#34c759');
 let dnHtml='';
 if(be.downside){
   const dn=be.downside;
   dnHtml=`<div class="stat" style="margin-top:8px">
     <div><div class="v" style="color:#ff3b30;font-size:18px">${dn.p_stop}%</div><div class="k">-10% 손절 맞을 확률</div></div>
     <div><div class="v" style="font-size:18px">${dn.p25_price?dn.p25_price.toLocaleString():'-'}</div><div class="k">약세 시(하위25%)</div></div>
     <div><div class="v" style="color:#8e0000;font-size:18px">${dn.worst!=null?dn.worst+'%':'-'}</div><div class="k">과거 최악</div></div></div>`;
 }
 B+=`<div class="card" style="background:#fff8f8"><h2>📉 하락 시나리오 · 반대 논리<span class="tb t2">참고</span></h2>
   <div class="note" style="background:transparent;padding:0;line-height:1.8">${be.summary}</div>
   <div class="bigbar-wrap" style="margin-top:10px"><span class="bl">반대논리</span>
     <div class="bigbar"><div style="width:${bs}%;background:${bcol}"></div></div>
     <span class="bv" style="color:${bcol}">${bs}</span></div>
   <div style="margin-top:4px">${rsk}</div>${dnHtml}
   <div class="note" style="background:#fff;margin-top:8px"><b>🚨 무효화 조건 — 여기 닿으면 논리와 무관하게 실행</b><br>
     ${invalList(be).map(x=>'· '+x).join('<br>')}</div>
   <div class="leg">사람은 산 종목의 좋은 점만 찾게 됩니다(확증편향). 이 목록은 <b>규칙이 자동으로 제시하는 반대 의견</b>이라 기분에 좌우되지 않습니다.</div></div>`;

 // 🎲 확률 계산 — 기준집단(순위밴드 × 국면)
 if(pr){
   const lo=Math.max(0,Math.min(100,pr.win_lo)), hi=Math.max(0,Math.min(100,pr.win_hi));
   const ec=pr.edge_pp>=0?'#34c759':'#ff3b30';
   const kc=pr.kelly_use>0?'#0071e3':'#c7c7cc';
   const rgn=(DATA.regime&&DATA.regime.current)?DATA.regime.current.name:'';
   // 검증에서 살아남은 숫자(손절률·구간 평균수익)를 먼저, 죽은 숫자(승률)는 접어서 뒤로.
   C+=`<div class="card"><h2>🎲 확률 계산</h2>
   <div class="note" style="background:#f0f7ff;margin-top:0">기준집단: <b>종합 ${pr.band}위</b>${rgn?` · <b>${rgn}</b>`:''}
   <div class="leg" style="margin-top:4px">이 종목의 과거가 아니라 <b>같은 순위·같은 국면이었던 100종목 전체 ${pr.n.toLocaleString()}건</b>(겹치지 않는 기간 ${pr.eff_n.toFixed(0)}개)의 30일 후 결과입니다.</div></div>

   <div class="leg" style="margin-top:10px"><span class="tb t1">검증됨</span> <b>이 두 개는 판단에 쓰세요</b></div>
   <div class="stat">
     <div><div class="v" style="color:#ff3b30;font-size:21px">${pr.p_stop}%</div><div class="k">-10% 손절 맞을 확률</div></div>
     <div><div class="v" style="font-size:21px">${pr.ev>=0?'+':''}${pr.ev}%</div><div class="k">30일 기대수익(비용·손절 반영)</div></div>
   </div>
   <div class="leg">구간이 올라갈수록 기대수익이 실제로 갈렸습니다(1-5위 <b>+3.5%</b> vs 51-100위 <b>+1.7%</b>). 손절 발동률도 15~19%로 구간마다 다릅니다. 이 둘은 <b>표본 18만건</b>으로 뒷받침됩니다.</div>

   <div class="leg" style="margin-top:12px"><span class="tb t3">미검증</span> <b>승률은 기록이지 예보가 아닙니다</b></div>
   <div class="ci">
     <div class="citrack"></div><div class="cirange" style="left:${lo}%;width:${hi-lo}%"></div>
     <div class="cidot" style="left:${pr.win}%"></div>
     <div class="cibase" style="left:${pr.base_win}%"></div>
     <div class="cilab" style="left:${lo}%">${lo}%</div>
     <div class="cilab" style="left:${pr.win}%;color:#0071e3;font-weight:700">${pr.win}%</div>
     <div class="cilab" style="left:${hi}%">${hi}%</div>
   </div>
   <div class="warn">35,163건 검증에서 예측 승률과 실제 결과의 상관은 <b>≈0</b>이었습니다. 파란 띠(95% 구간)가 <span style="color:#ff3b30">빨간 선</span>(기준선 ${pr.base_win}%)을 확실히 넘어야 우위인데 넘는 칸이 거의 없습니다. <b>승률은 50% 근처로 보세요.</b></div>
   <details style="margin-top:4px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">그래도 자세히 보기 (초과승률·손익비·켈리)</summary>
   <div class="stat" style="margin-top:6px">
     <div><div class="v" style="color:${ec};font-size:17px">${pr.edge_pp>=0?'+':''}${pr.edge_pp}%p</div><div class="k">기준선 대비 초과승률</div></div>
     <div><div class="v" style="font-size:17px">${pr.payoff||'-'}</div><div class="k">손익비(이익÷손실)</div></div>
     <div><div class="v" style="font-size:17px">${pr.sd}%</div><div class="k">수익 표준편차</div></div>
   </div>
   <div class="leg">표시 승률은 관측값 ${pr.win_raw}%를 표본 두께에 맞춰 전체 평균 쪽으로 당긴 값입니다(얇은 칸의 극단값을 그대로 믿지 않기 위해).</div></details>
   <div class="leg" style="margin-top:6px"><b>30일 뒤 결과 분포</b></div>
   <div class="pbar">
     <div style="width:${pr.p_stop}%;background:#ff3b30">${pr.p_stop>=8?pr.p_stop+'%':''}</div>
     <div style="width:${Math.max(0,pr.p_down5-pr.p_stop)}%;background:#ff9500">${(pr.p_down5-pr.p_stop)>=8?(pr.p_down5-pr.p_stop).toFixed(0)+'%':''}</div>
     <div style="width:${Math.max(0,100-pr.p_down5-pr.p_up10)}%;background:#c7c7cc">${(100-pr.p_down5-pr.p_up10)>=8?(100-pr.p_down5-pr.p_up10).toFixed(0)+'%':''}</div>
     <div style="width:${pr.p_up10}%;background:#34c759">${pr.p_up10>=8?pr.p_up10+'%':''}</div>
   </div>
   <div class="leg"><span style="color:#ff3b30">■</span> -10%↓(손절) ${pr.p_stop}% · <span style="color:#ff9500">■</span> -10~-5% · <span style="color:#8e8e93">■</span> -5~+10% · <span style="color:#34c759">■</span> +10%↑ ${pr.p_up10}%</div>
   <div class="leg" style="margin-top:12px"><span class="tb t2">참고</span> <b>비중 계산</b></div>
   <div class="stat">
     <div><div class="v" style="color:${kc};font-size:21px">${pr.kelly_use}%</div><div class="k">권장 계좌비중(하프켈리)</div></div>
     <div><div class="v" style="color:#999;font-size:19px">${pr.kelly}%</div><div class="k">켈리 상한</div></div>
   </div>
   <div class="leg"><b>켈리 공식</b>은 "장기 자산 성장률을 최대화하는 베팅 비중"입니다. 여기선 세 겹으로 깎았습니다:
     ① 기대수익에서 <b>추정오차 1σ</b>(${pr.se}%p) 차감 → 보수적 기대 ${pr.mu_lo}%,
     ② 이 전략은 <b>${pr.concurrent}종목을 동시에</b> 들므로 나눔,
     ③ 상한 20% + <b>절반만</b> 사용.
     <b>0%면 "이 표본으로는 비중을 키울 근거가 없다"</b>는 뜻입니다.
     아래 '손실예산으로 수량 정하기'와 함께 보고 <b>더 작은 쪽</b>을 택하세요.</div>
   </div>`;
 } else {
   C+=`<div class="card"><h2>🎲 확률 계산</h2><div class="warn">과거 유사 국면이 <b>10회 미만</b>이라 확률을 계산하지 않았습니다. 숫자를 지어내는 대신 <b>"모른다"</b>로 둡니다. 이런 종목은 비중을 키우지 마세요.</div></div>`;
 }

 // 🌐 시장 상황별 시나리오
 if(i.mscen&&i.mscen.rows){
   const LB=META.mscen_labels||[], MV=META.mscen_moves||[];
   let srow='';
   i.mscen.rows.forEach((r,k)=>{
     const lab=r.label||LB[k]||'', col=r.exp>=0?'#d70015':'#0071e3';
     const st=r.stop_hit?'<span style="color:#ff3b30;font-weight:700">손절</span>':'<span style="color:#c7c7cc">-</span>';
     srow+=`<tr><td style="font-size:11.5px">${lab}</td><td style="color:#999">${r.prob!=null?r.prob+'%':'-'}</td>`+
       `<td style="color:${col};font-weight:700">${r.exp>=0?'+':''}${r.exp}%</td><td>${r.price.toLocaleString()}</td><td>${st}</td></tr>`;
   });
   C+=`<div class="card"><h2>🌐 시장이 이렇게 되면<span class="tb t2">참고</span></h2>
   <div class="leg" style="margin-top:0">현재 <b>${i.mscen.regime}</b> 기준. 이 종목 베타 <b>${i.beta}</b>(내릴 때 <b>${i.down_beta}</b>) 적용.</div>
   <div class="tw"><table class="stab"><tr><th>30일 뒤 시장</th><th>확률</th><th>이 종목</th><th>주가</th><th>손절</th></tr>${srow}</table></div>
   <div class="leg">기대 = (유사사례 중앙수익) + 베타 × (시장수익 − 시장 중앙수익 ${i.mscen.mkt_med}%). 확률은 현재 국면에서 그 구간이 나온 과거 빈도.<br>
   ※ 베타는 과거 평균입니다. <b>진짜 급락장에서는 상관이 1로 수렴</b>해 표보다 더 나빠지는 경향이 있습니다.</div></div>`;
 }

 // 📰 최신 뉴스
 if(i.news&&i.news.length){
   let nr='';
   i.news.forEach(a=>{
     const cc=a.sent>0?'cpos':(a.sent<0?'cneg':'cneu'), lbl=a.sent>0?'호재':(a.sent<0?'악재':'중립');
     nr+=`<a class="nw" href="${a.link}" target="_blank" rel="noopener">
       <div class="nwt"><span class="chip ${cc}">${lbl}</span>${esc(a.title)}</div>
       <div class="nws">${esc(a.src)} · ${a.pub}${a.kw?' · '+esc(a.kw):''}</div></a>`;
   });
   B+=`<div class="card"><h2>📰 최신 뉴스 (14일)<span class="tb t2">참고</span></h2>${nr}
   <div class="leg">제목 키워드만 본 자동 분류이며 <b>점수에는 반영하지 않습니다</b>. 위 상승·하락 논리를 <b>뒤집을 사건</b>이 있는지 직접 확인하세요.</div></div>`;
 }

 // 🩳 공매도 부담 (참고 지표 · 점수 미반영)
 if(i.short_pct!=null){
   const shHi=i.short_rank>=80;const shc=shHi?'#ff3b30':'#1d1d1f';
   B+=`<div class="card"><h2>🩳 공매도 부담<span class="tb t2">참고</span></h2><div class="stat">
     <div><div class="v" style="color:${shc}">${i.short_pct}%</div><div class="k">잔고비중 (${i.short_asof} 기준)</div></div>
     <div><div class="v" style="color:${shc}">${i.short_rank}</div><div class="k">공매도 백분위(100=최다)</div></div>
     <div><div class="v">${shHi?'⚠ 높음':'보통'}</div><div class="k">부담도</div></div></div>
     <div class="leg">공매도 잔고비중↑ = 하락에 베팅한 물량이 많다는 약세 신호(Boehmer·Jones·Zhang 2008). <b>KRX 잔고는 T+2 지연</b> 공시이고 표본이 최근뿐이라 <b>아직 종합점수엔 반영하지 않은</b> 참고 지표입니다.</div></div>`;
 }
 C+=`<div class="card"><h2>📈 최근 60일 종합점수</h2>${lineChart(i.dates,i.scores,'#0071e3',W,180)}</div>`;
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
 C+=`<div class="card"><h2>💰 최근 60일 주가 + 매매 밴드</h2>${lineChart(i.dates,i.prices,'#34c759',W,196,pb)}<div class="leg">${legParts.join(' · ')}</div></div>`;

 B+='<div class="card"><h2>🧩 점수 구성 · 근거<span class="tb t2">참고</span></h2>';
 B+=`<div class="leg" style="margin-top:0">종합점수 = 각 팩터의 <b>당일 100종목 중 백분위(0~100)</b> × 가중치. 오른쪽 숫자는 <b>기여점수</b>(백분위×가중).</div>`;
 i.factors.forEach(f=>{const col=FCOLOR[f.name]||'#888',wcol=f.w>0?'#1d1d1f':'#c7c7cc';
  B+=`<div class="bigbar-wrap"><span class="bl">${f.name}</span>
   <div class="bigbar"><div style="width:${Math.max(0,Math.min(100,f.val))}%;background:${col}"></div></div>
   <span class="bv">${f.val.toFixed(0)}</span><span class="bw" style="color:${wcol}">${f.w>0?'+'+f.contrib.toFixed(1):'미사용'}</span></div>`;});
 const stg=i.factors.filter(f=>f.val>=70&&f.w>0).map(f=>f.name),wk=i.factors.filter(f=>f.val<=35&&f.w>0).map(f=>f.name);
 let why=[];if(stg.length)why.push('강점 '+stg.join('·'));if(wk.length)why.push('약점 '+wk.join('·'));
 if(why.length)B+=`<div class="note">📌 ${why.join(' / ')}</div>`;
 B+='<details style="margin-top:6px"><summary style="font-size:13px;color:#0071e3;cursor:pointer;font-weight:600">각 팩터의 학술적 근거 보기</summary>';
 i.factors.filter(f=>f.w>0).forEach(f=>{const fm=factMeta(f.key);
  B+=`<div class="leg" style="margin:8px 0 0"><b>${f.name}</b> — ${f.why||fm.why||''}<br><span style="color:#0071e3">📚 ${f.academic||fm.academic||''}</span><br><span style="color:#946200">📊 ${f.evidence||fm.evidence||''}</span></div>`;});
 B+='</details></div>';


 // 💵 매수 가격 안내 + 기대손익 계산기 (이미 보유/보류 종목엔 숨김)
 if(showBuy){
   A+=`<div class="card"><h2>💵 이 가격에 매수 (지정가)</h2>
     <div class="stat">
       <div><div class="v" style="color:#0071e3">${(+i.buy_limit).toLocaleString()}</div><div class="k">권장 지정가</div></div>
       <div><div class="v">${(+i.price).toLocaleString()}</div><div class="k">현재가</div></div>
       <div><div class="v" style="color:#ff3b30">${Math.round(i.price*0.9).toLocaleString()}</div><div class="k">손절가(체결가 -10%)</div></div>
     </div>
     <div class="leg">권장 지정가 = 현재가와 최근 5일 저가 사이(평균회귀 전략상 살짝 눌렀을 때 매수). 이 가격에 안 닿으면 미체결될 수 있으니, 확실히 사려면 현재가로.</div>
     <div class="note" style="background:#fff8e1;margin-top:8px"><b>🔺 스탑은 고정이 아니라 올려야 합니다</b>
       <div class="leg" style="margin-top:5px;line-height:1.75">
       처음엔 <b>체결가 -10%</b>(지정가 체결 시 ${Math.round(i.buy_limit*0.9).toLocaleString()}원)로 겁니다.
       그런데 주가가 <b>${Math.round(i.buy_limit*1.022).toLocaleString()}원</b>(+2.2%)만 넘어도
       <b>트레일링</b>이 그보다 높아집니다. 그때부터는 스탑을 <b>고점 기준</b>으로 올리세요.<br>
       <b>익절 사다리</b> — 고점×0.92가 기본, 고점수익 <b>+15%</b> 넘으면 <b>×0.95</b>,
       <b>+30%</b> 넘으면 <b>×0.97</b>로 조입니다. 수익 <b>+25%</b>에서는 보유의 <b>1/3</b>을 익절합니다
       (${Math.round(i.buy_limit*1.25).toLocaleString()}원 근방).<br>
       실측상 청산의 <b>77%가 트레일링</b>이고 고정 손절은 7%뿐입니다.
       고정 -10%만 걸어두면 이 전략을 실행하는 게 아닙니다.</div></div></div>`;
   // 손실예산 기반 수량 계산 (지정가 체결·손절 -10% → 잃는 돈 ≈ 매수금액의 10%)
   A+=`<div class="card"><h2>🎯 손실예산으로 수량 정하기</h2>
     <div class="leg" style="margin-top:0">"이 종목에서 최대 얼마까지 잃어도 되나"를 고르면, 지정가 ${(+i.buy_limit).toLocaleString()}원·손절 -10% 기준 <b>살 수량과 매수금액</b>을 계산합니다.</div>
     <div id="riskBtns" style="display:flex;gap:6px;margin:8px 0"></div>
     <div id="riskOut"></div>
     <div class="leg">한 종목에서 감수할 손실을 <b>원금의 1~2%</b>로 두면 자연스럽게 분산됩니다. (슬리피지·수수료 제외 근사)</div></div>`;
   // 기대손익 계산기
   if(i.exp){
     const e=i.exp;
     A+=`<div class="card adv"><h2>💰 이 금액으로 사면? (과거 통계 기반)<span class="advtag">자세히</span></h2>
       <div class="leg" style="margin-top:0">투자금 선택 → 30일 뒤 예상 손익 (점수·타이밍 유사했던 과거 사례 분포)</div>
       <div id="invBtns" style="display:flex;gap:6px;margin:8px 0"></div>
       <div id="expOut"></div>
       <div class="leg">중앙값=가장 흔한 결과. 좋을때/나쁠때는 상·하위 25% 지점. 승률 ${e.win}%.</div></div>`;
   }
 }

 if(i.analog){const a=i.analog,wc=a.win>=53?'#34c759':(a.win>=45?'#ff9500':'#ff3b30');
  C+=`<div class="card"><h2>🎯 과거 비슷할 때<span class="tb t3">미검증</span></h2>`;
  if(i.edge_weak)C+=`<div class="warn" style="color:#946200;background:#fff8e1">이 표본에서는 승률 ${a.win}% · 평균 ${a.avg>=0?'+':''}${a.avg}%로 뚜렷한 우위가 없었습니다. <b>다만 이 지표(종목별 유사사례)는 검증에서 예측력이 0이었으므로 매수 판단의 근거로 쓰지 마세요.</b></div>`;
  C+=`<div class="date">점수 ±5${a.fine?' + 타이밍 동일분위':''} 였던 <b>${a.n}회</b>의 30일 뒤 수익 분포</div>
   <div class="stat">
    <div><div class="v" style="color:${wc}">${a.win}%</div><div class="k">상승확률(승률)</div></div>
    <div><div class="v">${a.med>=0?'+':''}${a.med}%</div><div class="k">중앙값</div></div>
    <div><div class="v">${a.avg>=0?'+':''}${a.avg}%</div><div class="k">평균</div></div></div>`;
  const lo=a.worst,hi=a.best,rng=(hi-lo)||1,pos=v=>((v-lo)/rng*100);
  C+=`<div class="range"><div class="rtrack"></div><div class="rdot" style="left:${pos(a.med)}%"></div>
   <div class="rlab" style="left:${pos(a.worst)}%">${a.worst}%</div>
   <div class="rlab" style="left:${pos(a.best)}%">${a.best}%</div></div>
   <div class="leg">검정점=중앙값. 50% 구간 ${a.p25}%~${a.p75}%. 최악 ${a.worst}% · 최선 ${a.best}% (표본 ${a.n}).<br>※ 표본은 <b>겹치는 기간·생존종목 한정</b>이라 실제 독립 관측치는 적고 불확실성은 더 큽니다.</div></div>`;
 } else C+=`<div class="card"><h2>🎯 과거 유사사례<span class="tb t3">미검증</span></h2><div class="date">유사 표본 부족</div></div>`;

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
 A+=`<div class="card"><h2>🚪 매도 시점 / 주문 가격</h2>${priceBox}${holdInfo}<div class="note">${i.sell_hint}</div>
  <div class="warn">※ 개별 가격 예측이 아니라 위 통계 분포로 해석하세요. 이 시스템은 규칙대로 사고팔아 분포의 평균을 취하는 전략입니다.</div></div>`;

 // 📓 내 투자논리 기록 — 판단한 이유를 남기고, 나중에 그 논리가 맞았는지 대조한다
 let jr='';
 (i.journal||[]).forEach(e=>{
   const px=parseFloat(e.px)||0;
   const chg=px?((i.price/px-1)*100):null;
   const cc=chg==null?'#999':(chg>=0?'#d70015':'#0071e3');
   const since=chg==null?'':` · 기록 후 <span style="color:${cc};font-weight:700">${chg>=0?'+':''}${chg.toFixed(1)}%</span>`;
   jr+=`<div class="rk2"><span class="sev" style="background:#5856d6">${esc(e.kind||'기록')}</span>
     <div class="rtx"><b>${e.date}</b>${e.conv?' · 확신 '+esc(e.conv)+'/5':''}${since}
       ${e.bull?'<div class="rev" style="color:#1a8b38">▲ '+esc(e.bull)+'</div>':''}
       ${e.bear?'<div class="rev" style="color:#d70015">▼ '+esc(e.bear)+'</div>':''}
       ${e.inval?'<div class="rev">🚨 무효화: '+esc(e.inval)+'</div>':''}
       ${e.memo?'<div class="rev">'+esc(e.memo)+'</div>':''}</div></div>`;
 });
 if(!jr)jr='<div class="leg">아직 기록이 없습니다. 지금 사려는 이유를 남겨두면, 한 달 뒤 <b>내 판단이 맞았는지</b> 대조할 수 있습니다.</div>';
 B+=`<div class="card"><h2>📓 내 투자논리 기록</h2>${jr}
   <a class="btn" style="background:#5856d6;margin-top:10px" href="journal.html?code=${i.code}">✍️ ${i.name} 논리 기록하기</a>
   <div class="leg">위 상승·하락 논리가 자동으로 채워집니다. <b>기억은 결과에 맞춰 왜곡됩니다</b> — 사기 전에 적어두는 것이 유일한 방어입니다.</div></div>`;

 // 탭 조립 — 기본은 '판단'(지금 뭘 할까). 근거·숫자는 필요할 때만 편다.
 const nR=(i.bear&&i.bear.risks?i.bear.risks.length:0);
 const nN=(i.news?i.news.length:0);
 h+=`<div class="tabs">
   <button class="on" onclick="tabD(0)">판단</button>
   <button onclick="tabD(1)">근거${nR?` <span style="display:inline-block;background:#ff9500;color:#fff;border-radius:9px;min-width:17px;padding:1px 4px;font-size:11px;vertical-align:top">${nR}</span>`:''}</button>
   <button onclick="tabD(2)">숫자</button></div>
  <div class="pane on">${A}</div><div class="pane">${B}</div><div class="pane">${C}</div>`;
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
function tabD(k){
 const sh=document.getElementById('sheet');
 sh.querySelectorAll('.tabs button').forEach((b,j)=>b.className=(j===k?'on':''));
 sh.querySelectorAll('.pane').forEach((p,j)=>p.className='pane'+(j===k?' on':''));
 sh.scrollTop=0;
}
function closeD(){document.getElementById('modal').classList.remove('on');document.body.style.overflow='';}

// ── 쉽게 보기 / 자세히 보기 ──────────────────────────────
// 카드가 17개면 초보자는 '어디부터 봐야 하나'에서 멈춘다. 기본은 핵심만 띄우고,
// 숨긴 것들은 지우는 게 아니라 한 번의 터치로 되돌린다.
function setLv(simple){
  document.body.classList.toggle('simple', !!simple);
  document.getElementById('lv0').className = simple?'on':'';
  document.getElementById('lv1').className = simple?'':'on';
  try{ localStorage.setItem('rich_lv', simple?'1':'0'); }catch(e){}
}
setLv((()=>{ try{ return localStorage.getItem('rich_lv')!=='0'; }catch(e){ return true; } })());
</script></body></html>"""
    open(OUT,"w",encoding="utf-8").write(H)
    print(f"리포트: {OUT} · BuyFit추천 {len(buylist)} · 보유 {len(holds)}")


if __name__=="__main__":
    main()


