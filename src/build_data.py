"""
대시보드 데이터 생성 -> docs/data.json
핵심: '점수 높은 종목 중 지금 사기 좋은 것'을 골라주는 매수적합도(BuyFit) 계산
  - 종합점수(z) : 펀더멘털 우열  (검증: 단조, rho+0.027)
  - 타이밍점수(z): 단기 평균회귀  (검증: 낙폭 큰 종목이 30일후↑, Q1승률56% vs Q5 46%)
      timing = 0.6*(-z(60일고점대비낙폭)) + 0.4*(-z(최근20일수익))
  - BuyFit = z(종합점수) + 0.25*timing  (λ 워크포워드 검증: 0.5는 과최적화, 0.25가 안정)
모든 z는 '당일 100종목 횡단면' (cross-sectional, 학술 표준)

여기에 더해 종목마다 아래를 붙인다 (thesis.py / regime.py / news.py 결과):
  · 상승 시나리오 · 하락(반대) 논리 · 무효화 조건
  · 확률 블록(겹침 보정 유효표본 · Wilson 신뢰구간 · 손절 반영 기대값 · 켈리)
  · 시장 상황별 시나리오(베타 기반) · 최신 뉴스 · 내가 기록한 투자 논리
"""
import os, json
import pandas as pd
import numpy as np
import yaml

import thesis

SCORES="data/scores.csv"; SIGNALS="data/signals_today.csv"; WEIGHTS="config/weights.yaml"
ACCOUNT="data/account.json"; REGIME="data/regime.json"; NEWS="data/news.csv"; JOURNAL="data/journal.csv"
REFCLASS="data/refclass.json"; VERIFY="data/verify_thesis.json"; LAB="data/strategy_lab.json"
OUT="docs/data.json"; HOLD=30
NEWS_FRESH_DAYS=14      # 대시보드에 띄울 뉴스 기간
NEWS_PER_STOCK=4
FACTORS=[("s_value","가치"),("s_profit","수익성"),("s_grow","성장"),("s_flow","수급"),("s_mom","모멘텀")]
WKEY={"s_value":"value","s_profit":"profit","s_grow":"growth","s_flow":"flow","s_mom":"momentum"}

# 팩터별 '왜 이것이 수익을 예측하는가' — 학술 근거 + 이 시스템의 검증치 (전문가 설명용)
# evidence의 IC/Sharpe는 '이 저장소가 과거(주로 2024-25) 표본에서 추정한 값'이며 매일 재검증되지 않음.
FACTOR_META={
 "s_value":{"why":"PER·PBR이 낮을수록 시장가 대비 저평가. 평균회귀로 되돌아오는 경향.",
            "academic":"가치 프리미엄 — Fama·French(1992) HML, 저PBR 초과수익 Lakonishok·Shleifer·Vishny(1994).",
            "evidence":"이 표본에서 t-통계 최강(중심 팩터). 가중 0.32."},
 "s_profit":{"why":"ROE·영업이익률이 높고 개선 중인 '돈 잘 버는' 기업.",
            "academic":"수익성 프리미엄 — Fama·French 5요인 RMW(영업수익성) · Hou·Xue·Zhang(2015) q팩터 ROE. (Novy-Marx는 '총이익률'이 ROE보다 낫다고 주장한 점 유의)",
            "evidence":"과거 표본 추정 IC≈0.06 (2024-25, 매일 재검증 아님). 가중 0.22."},
 "s_grow":{"why":"매출·영업이익·순이익이 전년동기보다 성장.",
            "academic":"이익·매출 성장 모멘텀. ※단순 성장률 순위는 예측력이 약할 수 있어 소가중.",
            "evidence":"과거 표본 추정 IC≈0.06 (2024-25, 매일 재검증 아님). 가중 0.18."},
 "s_flow":{"why":"외국인이 최근 20일 순매수(시총 대비)한 종목 = 스마트머니 유입.",
            "academic":"외국인 정보우위 — 한국 대형주 방향 예측력 (Kim·Yi 등 2014).",
            "evidence":"과거 표본 추정 valid IC≈+0.055(모멘텀과 독립), 순매도 제외 시 Sharpe↑(2024-25 표본). 가중 0.20."},
 "s_mom":{"why":"최근 1년(직전 1개월 제외) 상승 추세가 이어지는 경향.",
            "academic":"모멘텀 효과 — Jegadeesh·Titman(1993), Carhart(1997) UMD.",
            "evidence":"대형주는 평균회귀가 강해 소액만 반영(가중 0.08)."},
}


def main():
    os.makedirs("docs",exist_ok=True)
    s=pd.read_csv(SCORES,dtype={"종목코드":str}); s["날짜"]=pd.to_datetime(s["날짜"])
    s=s.sort_values(["종목코드","날짜"])
    s["fwd"]=s.groupby("종목코드")["종가"].shift(-HOLD)/s["종가"]-1
    s["ret20"]=s.groupby("종목코드")["종가"].pct_change(20)
    s["roll60max"]=s.groupby("종목코드")["종가"].transform(lambda x:x.rolling(60,min_periods=20).max())
    s["dd"]=s["종가"]/s["roll60max"]-1  # 60일 고점 대비 낙폭(음수)
    # 기준선(base rate): 아무 종목이나 아무 날 샀을 때 30일 뒤 오를 확률.
    # 승률 60%가 '높은' 게 아니라 '기준보다 얼마나 높은가'가 엣지다 — 이 착시를 막는 기준값.
    base_win=round(float((s["fwd"]>0).mean()*100),1)
    base_med=round(float(s["fwd"].median()*100),1)
    weights={k:yaml.safe_load(open(WEIGHTS,encoding="utf-8")).get("logic",{}).get(v,0) for k,v in WKEY.items()}
    s["dd_q"]=s.groupby("날짜")["dd"].transform(lambda x:pd.qcut(x.rank(method="first"),5,labels=False,duplicates="drop") if x.notna().sum()>=5 else np.nan)
    # 최근 5일 저가 (매수 지정가 제안용)
    #  ※ ohlcv.csv가 낡으면 '최근 5일 저가'가 사실은 2주 전 저가일 수 있다.
    #    그 값으로 지정가를 계산하면 사용자가 엉뚱한 가격에 주문을 넣는다.
    #    그래서 주가 데이터보다 뒤처진 정도를 재고, 많이 낡았으면 아예 쓰지 않는다.
    #  '최근 5일'이 의미를 가지려면 창이 실제 최근과 겹쳐야 한다. 1거래일 지연까지만 허용.
    LOW5_MAX_LAG = 1          # 거래일 기준 이보다 더 뒤처지면 폐기
    low5 = {}; stale = None
    try:
        ox = pd.read_csv("data/ohlcv.csv", dtype={"종목코드": str})
        ox["날짜"] = pd.to_datetime(ox["날짜"])
        ox_last = ox["날짜"].max()
        px_last = s["날짜"].max()
        # 두 날짜 사이에 실제로 몇 거래일이 있었나 (주말·휴장 제외)
        lag = int((s["날짜"] > ox_last).sum() and
                  len([d for d in sorted(s["날짜"].unique()) if ox_last < d <= px_last]))
        if lag > LOW5_MAX_LAG:
            stale = {"file": "ohlcv.csv", "last": ox_last.strftime("%Y-%m-%d"),
                     "lag": lag, "effect": "매수 지정가를 현재가 기준으로만 계산합니다"}
            print(f"⚠ ohlcv.csv가 {lag}거래일 뒤처짐({ox_last.date()}) → 5일저가 미사용")
        else:
            for code, g2 in ox.sort_values("날짜").groupby("종목코드"):
                tail = g2.tail(5)
                low5[code] = int(tail["저가"].min())
    except Exception:
        pass

    # 업종/섹터 (collect_sectors.py 결과)
    sec_map = {}
    if os.path.exists("data/sectors.csv"):
        try:
            sc2 = pd.read_csv("data/sectors.csv", dtype={"종목코드": str})
            sc2["종목코드"] = sc2["종목코드"].str.zfill(6)
            sec_map = {r["종목코드"]: {"sector": r["섹터"], "industry": r["업종"]}
                       for _, r in sc2.iterrows()}
        except Exception:
            pass

    # 공매도 잔고비중 (표시용 리스크; 각 종목 최신 가용값 + 100종목 중 백분위). 점수 미반영(weight 0).
    short_map = {}
    if os.path.exists("data/shorts.csv"):
        try:
            shd = pd.read_csv("data/shorts.csv", dtype={"종목코드": str})
            shd["종목코드"] = shd["종목코드"].str.zfill(6)
            shd["날짜"] = pd.to_datetime(shd["날짜"], errors="coerce")
            shd = shd.dropna(subset=["날짜", "공매도잔고비중"]).sort_values("날짜")
            latest = shd.groupby("종목코드").tail(1)
            vals = latest.set_index("종목코드")["공매도잔고비중"]
            rnk = vals.rank(pct=True) * 100
            asof = latest.set_index("종목코드")["날짜"]
            for c in vals.index:
                short_map[c] = {"pct": round(float(vals[c]), 2),
                                "rank": int(round(float(rnk[c]))),
                                "asof": asof[c].strftime("%m/%d")}
        except Exception:
            pass

    # ── 시장국면 (regime.py) ──────────────────────────────────
    regime=None
    if os.path.exists(REGIME):
        try: regime=json.load(open(REGIME,encoding="utf-8"))
        except Exception: regime=None
    betas=(regime or {}).get("betas",{})

    # ── 기준집단 (refclass.py) · 검증결과 (verify_thesis.py) ────
    # 확률은 '그 종목의 과거'가 아니라 '순위밴드×국면' 칸에서 가져온다.
    # 이유: verify_thesis가 종목별 유사국면 승률의 예측력을 0으로 판정했다.
    refc=None
    if os.path.exists(REFCLASS):
        try: refc=json.load(open(REFCLASS,encoding="utf-8"))
        except Exception: refc=None
    verify=None
    if os.path.exists(VERIFY):
        try: verify=json.load(open(VERIFY,encoding="utf-8"))
        except Exception: verify=None
    # 전략 점검 결과 — 어떤 규칙이 실제로 청산을 일으키는지 등
    lab=None
    if os.path.exists(LAB):
        try:
            L=json.load(open(LAB,encoding="utf-8"))
            b=L.get("baseline") or {}
            tr=b.get("train") or {}
            lab={"why":tr.get("why"),"n":tr.get("n_trades"),
                 "train":L.get("train"),"test":L.get("test"),
                 "ann_tr":tr.get("ann"),"ann_te":(b.get("test") or {}).get("ann"),
                 "avg_hold":tr.get("avg_hold")}
        except Exception: lab=None

    def ref_cell(rk):
        """종합순위 → 기준집단 통계 (국면칸 우선, 얇으면 밴드 전체로 폴백)"""
        if not refc: return None,None
        band=None
        for b in refc.get("bands",[]):
            lo,hi=(int(x) for x in b.split("-"))
            if lo<=rk<=hi: band=b; break
        if not band: return None,None
        rg=(regime or {}).get("current",{}).get("name")
        cell=refc.get("cells",{}).get(f"{band}|{rg}") if rg else None
        return (cell or refc.get("by_band",{}).get(band)), band

    # ── 최신 뉴스 (news.py) ───────────────────────────────────
    news_map={}; market_news=[]
    if os.path.exists(NEWS):
        try:
            nw=pd.read_csv(NEWS,dtype={"종목코드":str}).fillna("")
            nw["감성"]=pd.to_numeric(nw["감성"],errors="coerce").fillna(0)
            cutoff=(pd.Timestamp.now(tz="Asia/Seoul")-pd.Timedelta(days=NEWS_FRESH_DAYS)).strftime("%Y-%m-%d")
            nw=nw[nw["발행일"].astype(str).str[:10]>=cutoff]
            nw=nw.sort_values("발행일",ascending=False)
            for code,g3 in nw.groupby("종목코드"):
                arts=[{"title":str(r["제목"]),"link":str(r["링크"]),"pub":str(r["발행일"])[:16],
                       "src":str(r["출처"]),"sent":int(r["감성"]),"kw":str(r["키워드"])}
                      for _,r in g3.head(NEWS_PER_STOCK).iterrows()]
                if code=="MARKET": market_news=arts
                else: news_map[code]=arts
        except Exception:
            pass

    # ── 내가 기록한 투자 논리 (journal.html → data/journal.csv) ─
    journal_map={}
    if os.path.exists(JOURNAL):
        try:
            jr=pd.read_csv(JOURNAL,dtype=str).fillna("")
            code_of=dict(zip(s["종목명"],s["종목코드"]))
            for _,r in jr.iterrows():
                code=code_of.get(str(r.get("종목명","")).strip())
                if not code: continue
                journal_map.setdefault(code,[]).append({
                    "date":str(r.get("날짜","")),"kind":str(r.get("구분","")),
                    "conv":str(r.get("확신도","")),"bull":str(r.get("상승논리","")),
                    "bear":str(r.get("반대논리","")),"target":str(r.get("목표가","")),
                    "stop":str(r.get("손절가","")),"inval":str(r.get("무효화조건","")),
                    "px":str(r.get("기록시점가","")),"memo":str(r.get("메모",""))})
            for k in journal_map:
                journal_map[k]=sorted(journal_map[k],key=lambda x:x["date"],reverse=True)[:5]
        except Exception:
            pass

    last=s["날짜"].max()
    cur=s[s["날짜"]==last].copy()
    def z(col):
        x=cur[col]; return (x-x.mean())/(x.std()+1e-9)
    cur["z_score"]=z("종합점수")
    cur["timing"]=0.6*(-z("dd"))+0.4*(-z("ret20"))
    cur["buyfit"]=cur["z_score"]+0.25*cur["timing"]  # λ=0.25 (워크포워드 검증: 0.5는 과함)
    # 0~100 표시용 환산
    def to100(col):
        x=cur[col]; lo,hi=x.min(),x.max()
        return ((x-lo)/((hi-lo)+1e-9)*100)
    cur["buyfit100"]=to100("buyfit")
    cur["timing100"]=to100("timing")
    cur=cur.sort_values("종합점수",ascending=False).reset_index(drop=True)
    cur["순위"]=cur.index+1
    cur["buyrank"]=cur["buyfit"].rank(ascending=False).astype(int)
    cur["buyfit_pct"]=cur["buyfit"].rank(pct=True)*100   # 100=가장 추천 (등급의 뼈대)

    sig=pd.read_csv(SIGNALS,dtype={"종목코드":str}) if os.path.exists(SIGNALS) else pd.DataFrame()
    def _s(v):  # 빈칸/NaN → "" (data.json이 유효 JSON이 되도록; NaN 금지)
        return "" if (v is None or (isinstance(v,float) and pd.isna(v))) else v
    sigmap={r["종목코드"]:dict(구분=r["구분"],사유=r["사유"],보유일=_s(r.get("보유일",0)),수익률=_s(r.get("수익률%")),손절가=_s(r.get("손절가")),감시가=_s(r.get("감시가")),평단가=_s(r.get("평단가")),투자금액=_s(r.get("투자금액")),수량=_s(r.get("수량")),실현손익=_s(r.get("실현손익"))) for _,r in sig.iterrows()} if len(sig) else {}

    def _num(v):
        """빈칸/NaN 안전 정수 변환 (없으면 None)"""
        s=str(v).strip()
        if s in ("","nan","None"): return None
        try: return int(round(float(s)))
        except Exception: return None

    # 거래일 → 일련번호 (표본 겹침 보정용)
    _dord={d:k for k,d in enumerate(sorted(s["날짜"].unique()))}

    def independent_n(dates):
        """겹치지 않는 관측치가 실제 몇 개인가.
        30거래일 창을 매일 표집하면 인접 표본은 29/30이 같은 구간이라 사실상 1개다.
        정렬 후 앞에서부터 30거래일 간격을 두고 고르는 그리디 선택 = 최대 비중첩 표본수.
        (n/30으로 일괄 나누는 것보다 정확 — 유사국면이 여러 시기에 흩어져 있으면 그만큼 인정)"""
        pos=sorted(_dord[d] for d in dates if d in _dord)
        cnt=0; nxt=-1
        for p in pos:
            if p>=nxt:
                cnt+=1; nxt=p+HOLD
        return cnt

    # 과거 유사사례: '종합점수 ±5 AND 타이밍분위 같음' 2D 매칭 (더 정밀)
    # 반환 시 원수익 배열(rets)·독립표본수도 함께 주어 thesis.probability()가 신뢰구간·켈리를 계산한다.
    def analog(code,score,ddq):
        g=s[s["종목코드"]==code].dropna(subset=["fwd"])
        if pd.isna(score) or len(g)==0: return None,None,None
        sim=g[(g["종합점수"]>=score-5)&(g["종합점수"]<=score+5)]
        # 타이밍까지 맞춘 정밀 표본
        fine=sim[sim["dd_q"]==ddq] if not pd.isna(ddq) else sim
        use=fine if len(fine)>=15 else sim
        if len(use)<10: return None,None,None
        r=use["fwd"]
        st=dict(n=int(len(use)),fine=bool(len(fine)>=15),
            win=round((r>0).mean()*100,1),avg=round(r.mean()*100,1),med=round(r.median()*100,1),
            p25=round(r.quantile(.25)*100,1),p75=round(r.quantile(.75)*100,1),
            best=round(r.max()*100,1),worst=round(r.min()*100,1))
        return st,[float(x) for x in r],independent_n(use["날짜"])

    # 보유 종목의 업종 비중 (반대논리 '포트폴리오 쏠림' 판정용 — 평가액 대신 원금 기준 근사)
    held_sec={}; held_cost=0.0
    for _,r0 in (sig.iterrows() if len(sig) else []):
        if str(r0.get("구분","")) not in ("🟢유지","⏳보유"): continue
        try: amt=float(str(r0.get("투자금액","")).replace(",","") or 0)
        except ValueError: amt=0.0
        if amt<=0: continue
        sc0=sec_map.get(str(r0["종목코드"]).zfill(6),{}).get("sector") or "기타"
        held_sec[sc0]=held_sec.get(sc0,0.0)+amt; held_cost+=amt

    # 100종목이 공유하는 설명문 사전 (종목마다 복사하면 data.json이 2배로 부푼다)
    meta={"factors":{k:FACTOR_META.get(k,{}) for k,_ in FACTORS},
          "catalysts":{},"risks":{}}

    items=[]
    for _,r in cur.iterrows():
        code=r["종목코드"]
        g=s[s["종목코드"]==code].sort_values("날짜").tail(60)
        factors=[]
        for k,nm in FACTORS:
            # 팩터의 학술근거 설명문은 100종목 모두 같다 → meta["factors"]에 한 번만 담는다.
            val=round(float(r.get(k,50)),1); wv=weights.get(k,0)
            factors.append({"key":k,"name":nm,"val":val,"w":wv,"contrib":round(val*wv,1)})
        # 추천 근거: 이 종목 점수에 가장 크게 기여한 팩터 상위 2
        _top=sorted(factors,key=lambda f:f["contrib"],reverse=True)[:2]
        basis=" · ".join(f'{f["name"]} {f["val"]:.0f}점' for f in _top if f["w"]>0)
        st,rets,eff_n=analog(code,r["종합점수"],r["dd_q"])
        # 매수 지정가 제안: 현재가와 최근5일 저가 사이 (평균회귀 — 약간 눌렀을 때 매수)
        px_now=int(r["종가"])
        lo5=low5.get(code)
        if lo5 is None:
            # 5일저가를 못 쓰면(데이터 낡음) 현재가에서 소폭만 눌러 제안 — 지어내지 않는다
            lo5 = px_now
            buy_limit = int(round(px_now * 0.99))
        else:
            buy_limit=int(round((px_now*0.65 + lo5*0.35)))  # 현재가 쪽에 가중, 저가도 반영
        buy_limit=min(buy_limit, px_now)  # 현재가 넘지 않게
        # 추천 투자금 기준 기대손익 (기본 100만원, 화면에서 조정)
        exp=None
        if st:
            exp=dict(med=st["med"], avg=st["avg"], p25=st["p25"], p75=st["p75"], win=st["win"])
        sg=sigmap.get(code,{})
        held=int(sg.get("보유일",0) or 0)
        is_held = sg.get("구분") in ("🟢유지","⏳보유","🟠부분익절")
        if is_held:
            remain=max(0,HOLD-held)
            sell_hint=(f"보유 {held}일 경과. "+(f"{remain}일 더 보유 후 20위 이탈 시 매도." if remain>0 else "최소보유 충족 — 20위 밖 이탈 시 매도.")+" 손절 -10% · 트레일 -8%.")
        else:
            remain=None
            sell_hint=f"매수 시 최소 {HOLD}일 보유 → 이후 종합 20위 이탈 시 매도. 손절 -10%, 트레일 -8%."

        # ── 추가 판단정보 ──────────────────────────────
        invest=_num(sg.get("투자금액"))            # 내가 넣은 원금(보유 종목)
        avgp=_num(sg.get("평단가"))
        guard=_num(sg.get("감시가"))               # 손절 감시가(둘 중 먼저 닿는 쪽)
        # 보유 수량: signal.py가 내보낸 정확한 값 우선, 없으면 원가/평단
        qty=None
        try:
            qs=str(sg.get("수량","")).strip().replace(",","")
            if qs not in ("","nan","None"): qty=float(qs)
        except Exception: qty=None
        shares=qty if qty is not None else (round(invest/avgp,2) if (invest and avgp) else None)
        realized_pos=_num(sg.get("실현손익"))       # 이 종목 실현손익(원)
        # 손절까지 남은 여유 (현재가 대비 %, 음수 = 아래로 이만큼 남음)
        guard_gap=round((guard/px_now-1)*100,1) if (guard and px_now) else None
        # 매도 가능까지 남은 거래일 + 근사 달력일 (주말만 스킵)
        dmin_date=None
        if remain is not None:
            dmin_date=(last if remain<=0 else (last+pd.offsets.BDay(remain))).strftime("%m/%d")
        # 목표가 밴드 (과거 유사사례 30일 후 분포) + 매수후 손절가
        tgt_med=int(round(px_now*(1+st["med"]/100))) if st else None
        tgt_lo =int(round(px_now*(1+st["p25"]/100))) if st else None
        tgt_hi =int(round(px_now*(1+st["p75"]/100))) if st else None
        stop_buy=int(round(px_now*0.9))            # 신규매수 시 실제 손절선(체결가 -10% 근사)
        # 엣지 약함: 승률<53(동전던지기 근방) 또는 기대≤0 또는 중앙값이 ±1% 이내(사실상 무변동)
        edge_weak=bool(st and (st["win"]<53 or st["avg"]<=0 or abs(st["med"])<1))

        # ── 상승/하락 시나리오 · 확률 · 시장 시나리오 (thesis.py) ──
        bt=betas.get(code,{})
        nflags=[{"title":a["title"],"text":f'최근 부정 기사: "{a["title"][:40]}…" ({a["src"]})'}
                for a in news_map.get(code,[]) if a["sent"]<=-2]
        conc=None
        if held_cost>0:
            _sc=sec_map.get(code,{}).get("sector") or "기타"
            _w=held_sec.get(_sc,0.0)/held_cost*100
            if _w>=35: conc=f"보유 중 '{_sc}' 업종이 이미 {_w:.0f}%"
        ctx={"name":r["종목명"],"code":code,"score":float(r["종합점수"]),"rank":int(r["순위"]),
             "price":px_now,"factors":factors,"timing":float(r["timing100"]),
             "dd":round(float(r["dd"])*100,1) if pd.notna(r["dd"]) else None,
             "analog":st,"exp":exp,"tgt_lo":tgt_lo,"tgt_hi":tgt_hi,"stop_buy":stop_buy,
             "short_pct":short_map.get(code,{}).get("pct"),
             "short_rank":short_map.get(code,{}).get("rank"),
             "beta":bt.get("beta"),"down_beta":bt.get("down"),
             "vol":bt.get("vol"),"vol_rank":bt.get("vol_rank"),
             "news_flags":nflags,"sector_conc":conc,"signal":sg.get("구분","")}
        cell,band=ref_cell(int(r["순위"]))
        prob=dict(cell) if cell else None      # 기준집단 = 판단에 쓰는 확률
        if prob: prob["band"]=band
        ctx["prob"]=prob
        ctx["buyfit_pct"]=float(r["buyfit_pct"]); ctx["buyrank"]=int(r["buyrank"])
        # 종목별 유사사례 통계는 '참고'로만 남긴다 (예측력 검증 실패 — 아래 note)
        analog_prob=thesis.probability(rets,eff_n,base_win) if rets else None
        bull=thesis.bull_case(ctx,meta)
        ctx["bull_first"]=bull["drivers"][0]["title"] if bull["drivers"] else None
        bear=thesis.bear_case(ctx,meta)
        mscen=thesis.market_scenarios(ctx,regime,meta)
        vd=thesis.verdict(ctx,bull,bear,regime,base_win,meta)

        items.append({
            "code":code,"name":r["종목명"],"rank":int(r["순위"]),
            "score":round(float(r["종합점수"]),1),"price":int(r["종가"]),
            "buyrank":int(r["buyrank"]),"buyfit":round(float(r["buyfit100"]),0),
            "timing":round(float(r["timing100"]),0),
            "dd":round(float(r["dd"])*100,1) if pd.notna(r["dd"]) else None,"ret20":round(float(r["ret20"])*100,1) if pd.notna(r["ret20"]) else None,
            "dates":[d.strftime("%m/%d") for d in g["날짜"]],
            "scores":[round(float(x),1) for x in g["종합점수"]],
            "prices":[int(x) for x in g["종가"]],
            "factors":factors,"signal":sg.get("구분",""),"reason":sg.get("사유",""),
            "pnl":sg.get("수익률",""),"held":held,"analog":st,"sell_hint":sell_hint,
            "buy_limit":buy_limit,"low5":lo5,"exp":exp,
            "stop_price":sg.get("손절가",""),"guard_price":sg.get("감시가",""),"avg_price":sg.get("평단가",""),
            "invest":invest,"avgp":avgp,"shares":shares,"realized":realized_pos,"guard_gap":guard_gap,
            "trail_w":_num(sg.get("트레일폭%")) if sg.get("트레일폭%") not in ("",None) else None,
            "tp_price":_num(sg.get("익절가")),"tp_qty":sg.get("익절수량",""),
            "dmin_remain":remain,"dmin_date":dmin_date,
            "tgt_med":tgt_med,"tgt_lo":tgt_lo,"tgt_hi":tgt_hi,
            "stop_buy":stop_buy,"edge_weak":edge_weak,"basis":basis,
            "sector":sec_map.get(code,{}).get("sector",""),
            "industry":sec_map.get(code,{}).get("industry",""),
            "short_pct":short_map.get(code,{}).get("pct"),
            "short_rank":short_map.get(code,{}).get("rank"),
            "short_asof":short_map.get(code,{}).get("asof"),
            # ── 판단 근거 블록 ──
            "bull":bull,"bear":bear,"prob":prob,"aprob":analog_prob,"mscen":mscen,"verdict":vd,
            "buyfit_pct":round(float(r["buyfit_pct"]),0),
            "beta":bt.get("beta"),"down_beta":bt.get("down"),
            "vol":bt.get("vol"),"vol_rank":bt.get("vol_rank"),
            "news":news_map.get(code,[]),"journal":journal_map.get(code,[])})

    # ── 계좌 요약 + 손익 추이(equity curve) ─────────────
    realized_total=0
    if os.path.exists(ACCOUNT):
        try: realized_total=int(json.load(open(ACCOUNT,encoding="utf-8")).get("realized_total",0))
        except Exception: realized_total=0
    # 오늘 밤 실제 보유 = 유지/보유 + 내일 팔 예정(손절/매도)까지 포함 (아직 소유 중)
    held_items=[i for i in items if i["signal"] in ("🟢유지","⏳보유","🟠부분익절","🔴손절","🔵매도") and i.get("shares") and i.get("invest")]
    portfolio=None
    if held_items:
        trading_dates=sorted(s["날짜"].unique())
        # 각 보유 종목 진입 거래일 = last 기준 보유일 만큼 이전 거래일
        for i in held_items:
            idx=max(0,len(trading_dates)-1-int(i["held"]))
            i["_entry"]=trading_dates[idx]
        start=min(i["_entry"] for i in held_items)
        codes=[i["code"] for i in held_items]
        sub=s[(s["종목코드"].isin(codes))&(s["날짜"]>=start)]
        piv=sub.pivot_table(index="날짜",columns="종목코드",values="종가")
        # 평가액 = 원금×(현재가/평단가) → signal.py 수익률%와 정확히 일치(정수주 반올림 왜곡 제거)
        def _val(i, px): return i["invest"]*(px/i["avgp"]) if i.get("avgp") else (i["shares"] or 0)*px
        curve=[]
        for d,row in piv.iterrows():
            val=cost=0.0
            for i in held_items:
                c=i["code"]
                if d>=i["_entry"] and c in row.index and pd.notna(row[c]):
                    val+=_val(i,float(row[c])); cost+=i["invest"]
            if cost>0:
                curve.append({"date":d.strftime("%m/%d"),
                              "pnl":round((val/cost-1)*100,2),"value":int(round(val))})
        invested=sum(i["invest"] for i in held_items)
        value=sum(_val(i,i["price"]) for i in held_items)
        # 오늘 전 종목 손절 동시 발동 시 현 시점 대비 추가손실(원): 감시가 없으면 -10% 근사
        wc=sum(_val(i,i["price"])*((i["guard_gap"]/100) if i.get("guard_gap") is not None else -0.10)
               for i in held_items)
        # 섹터 편중 (평가액 기준)
        sec_val={}
        for i in held_items:
            sec=i.get("sector") or "기타"
            sec_val[sec]=sec_val.get(sec,0)+_val(i,i["price"])
        sectors=sorted(({"name":k,"pct":round(v/value*100,1)} for k,v in sec_val.items()),
                       key=lambda x:-x["pct"]) if value else []
        # ── 시장 상황별 계좌 스트레스 테스트 (베타 기반) ──────────
        # 시장이 m% 움직이면 각 종목은 대략 beta×m% (내릴 땐 하락베타). 손절 발동 여부까지 센다.
        stress=[]
        if value:
            probs=(regime or {}).get("mkt_probs_now") or (regime or {}).get("mkt_probs_all")
            for k2,m in enumerate([-20,-10,-5,0,5,10]):
                pv=0.0; hits=0
                for i in held_items:
                    b0=betas.get(i["code"],{})
                    b=(b0.get("down") or b0.get("beta") or 1.0) if m<0 else (b0.get("beta") or 1.0)
                    mv=b*m/100.0
                    pv+=_val(i,i["price"])*(1+mv)
                    gap=(i.get("guard_gap") if i.get("guard_gap") is not None else -10.0)/100.0
                    if mv<=gap: hits+=1
                # 확률: mkt_probs 버킷(-99~-10,-10~-5,-5~0,0~5,5~10,10~99)의 대표값에 맞춰 매핑
                pmap={-20:0,-10:1,-5:2,0:3,5:4,10:5}
                stress.append({"move":m,"value":int(round(pv)),
                               "pnl":int(round(pv-invested)),
                               "pnl_pct":round((pv/invested-1)*100,1) if invested else 0.0,
                               "chg_pct":round((pv/value-1)*100,1),
                               "stops":hits,
                               "prob":(probs[pmap[m]] if probs else None)})
        portfolio=dict(
            n=len(held_items),invested=int(round(invested)),value=int(round(value)),
            upnl=int(round(value-invested)),
            upnl_pct=round((value/invested-1)*100,2) if invested else 0.0,
            worst=int(round(wc)),worst_pct=round(wc/value*100,2) if value else 0.0,
            realized=realized_total,
            total_pnl=int(round(value-invested))+realized_total,
            sectors=sectors,
            top_sector=sectors[0]["name"] if sectors else "",
            top_sector_pct=sectors[0]["pct"] if sectors else 0.0,
            equity=curve,stress=stress,
            beta=round(sum(_val(i,i["price"])*(betas.get(i["code"],{}).get("beta") or 1.0)
                           for i in held_items)/value,2) if value else None)
        for i in held_items:
            i.pop("_entry",None)
    elif realized_total:   # 보유는 없지만 과거 실현손익이 있는 경우
        portfolio=dict(n=0,invested=0,value=0,upnl=0,upnl_pct=0.0,worst=0,worst_pct=0.0,
                       realized=realized_total,total_pnl=realized_total,equity=[],
                       stress=[],beta=None)

    # BuyFit 순 추천: 점수 상위20 풀 안에서 buyfit 높은 순
    pool=cur.head(20)
    buylist=[c for c in pool.sort_values("buyfit",ascending=False)["종목코드"].tolist()]
    market={"date":last.strftime("%Y-%m-%d"),
        "avg":round(float(cur["종합점수"].mean()),1),"median":round(float(cur["종합점수"].median()),1),
        "max":round(float(cur["종합점수"].max()),1),"min":round(float(cur["종합점수"].min()),1),
        # 점수를 빈 범위로 clip → >85(최상위)·<20(최하위)도 양끝 빈에 집계되어 합계 100 유지
        "hist":[int(x) for x in np.histogram(cur["종합점수"].clip(20,84.9),bins=np.arange(20,90,5))[0]],
        "hist_labels":[f"{b}" for b in np.arange(20,85,5)],
        "weights":weights,"buylist":buylist[:10],"news":market_news,"stale":stale,"lab":lab}
    # allow_nan=False → 유효 JSON 보장(trade.html의 JSON.parse 및 표준 준수). NaN 있으면 즉시 실패.
    json.dump({"market":market,"items":items,"portfolio":portfolio,"regime":regime,
               "refclass":refc,"verify":verify,"meta":meta},
              open(OUT,"w",encoding="utf-8"),ensure_ascii=False,allow_nan=False)
    pf=(f" · 계좌 원금 {portfolio['invested']:,}→평가 {portfolio['value']:,} ({portfolio['upnl_pct']:+.1f}%)"
        if portfolio else " · 보유없음")
    gr={}
    for i in items: gr[i["verdict"]["grade"]]=gr.get(i["verdict"]["grade"],0)+1
    rg=f" · 국면 {regime['current']['name']}" if regime else " · 국면 미계산"
    nn=sum(len(i["news"]) for i in items)
    print(f"data.json: {len(items)}종목 · BuyFit 추천순 상위3 {buylist[:3]} · "
          f"통계 {sum(1 for i in items if i['analog'])}건{pf}{rg}")
    print(f"  판단등급 " + " ".join(f"{k}:{gr.get(k,0)}" for k in "ABCD") +
          f" · 뉴스 {nn}건 · 논리기록 {sum(len(i['journal']) for i in items)}건")


if __name__=="__main__":
    main()


