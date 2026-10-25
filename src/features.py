"""
판단 엔진용 공용 지표 — 라이브(decision_engine)와 백테스트(engine_lab)가 **같은 함수**를 쓴다.
(백테스트에서 검증한 숫자와 실제 주문을 낸 숫자가 다른 코드에서 나오면 검증이 무의미해진다.)

모든 지표는 (날짜 × 종목) 행렬. t일 값은 t일 장 마감까지 알 수 있는 정보만 쓴다.

[차트]  ma50/ma200 · 52주 고가 근접도 · RSI14 · 거래량 추세 · 20일 변동성 · 이격도
[재무]  분기 단독값 환산 → TTM → 수정 F-score(7점) · 이익 서프라이즈(SUE 근사)
[동종]  섹터 내 가치 순위 · 섹터 이익모멘텀 · 섹터 가격모멘텀
[시장]  100종목 등가중 지수 · 200일선 추세 · 폭(breadth) · 고점대비 낙폭 · 국면

학술 근거 (docs/STRATEGY.md에 정리)
  Faber(2007) 200일선 · George&Hwang(2004) 52주 고가 · Piotroski(2000) F-score
  Bernard&Thomas(1989) PEAD · Moskowitz&Grinblatt(1999) 산업 모멘텀
  Asness·Porter·Stevens(2000) 산업 내 상대가치 · Wilder(1978) RSI
"""
import os

import numpy as np
import pandas as pd

PRICES = "data/prices.csv"
SCORES = "data/scores.csv"
FIN = "data/financials.csv"
SECTORS = "data/sectors.csv"
FLOWS = "data/flows.csv"
MCAP = "data/marketcap.csv"


def _code(s):
    return s.astype(str).str.zfill(6)


# ══════════════════════════════════════════════════════════════
# 로드
# ══════════════════════════════════════════════════════════════
def load_panel():
    """scores.csv의 (날짜, 종목) 격자 위에 가격·점수를 깐다. 반환 dict."""
    s = pd.read_csv(SCORES, dtype={"종목코드": str})
    s["종목코드"] = _code(s["종목코드"])
    s["날짜"] = pd.to_datetime(s["날짜"])
    dates = pd.DatetimeIndex(sorted(s["날짜"].unique()))
    codes = sorted(s["종목코드"].unique())

    def piv(df, col):
        return (df.pivot_table(index="날짜", columns="종목코드", values=col)
                  .reindex(index=dates, columns=codes))

    names = s.drop_duplicates("종목코드", keep="last").set_index("종목코드")["종목명"].to_dict()
    px = pd.read_csv(PRICES, dtype={"종목코드": str}, usecols=["날짜", "종목코드", "종가", "거래량"])
    px["종목코드"] = _code(px["종목코드"])
    px["날짜"] = pd.to_datetime(px["날짜"])
    px = px[px["종목코드"].isin(codes)]
    close = piv(px, "종가")
    volume = piv(px, "거래량")
    score = piv(s, "종합점수")
    panel = {"dates": dates, "codes": codes, "names": names,
             "close": close, "volume": volume, "score": score}
    for k in ["s_value", "s_profit", "s_grow", "s_flow", "s_rmom", "이격도20"]:
        if k in s.columns:
            panel[k] = piv(s, k)
    return panel


# ══════════════════════════════════════════════════════════════
# 차트 지표
# ══════════════════════════════════════════════════════════════
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def chart_features(close, volume):
    f = {}
    f["ma20"] = close.rolling(20, min_periods=15).mean()
    f["ma50"] = close.rolling(50, min_periods=40).mean()
    f["ma200"] = close.rolling(200, min_periods=150).mean()
    f["ma200_slope"] = f["ma200"] / f["ma200"].shift(20) - 1
    # 추세 정렬 (Faber 200일선 + Minervini 추세 템플릿의 단순형)
    f["trend_up"] = (close > f["ma200"]) & (f["ma50"] > f["ma200"])
    f["hi52"] = close / close.rolling(252, min_periods=120).max()        # 1.0 = 신고가
    f["rsi14"] = rsi(close, 14)
    f["vol20"] = close.pct_change().rolling(20, min_periods=10).std()
    f["vol_ratio"] = (volume.rolling(20, min_periods=10).mean()
                      / volume.rolling(120, min_periods=60).mean())
    f["disp20"] = close / f["ma20"] - 1
    f["ret20"] = close.pct_change(20)
    f["ret120"] = close.pct_change(120)
    f["dd60"] = close / close.rolling(60, min_periods=30).max() - 1
    return f


# ══════════════════════════════════════════════════════════════
# 재무 지표 (분기 단독 → TTM → F-score · 서프라이즈)
# ══════════════════════════════════════════════════════════════
FLOW_ITEMS = ["매출액", "영업이익", "당기순이익"]      # 손익 = 기간값 (4Q는 연간 누계로 들어온다)
STOCK_ITEMS = ["자산총계", "부채총계", "자본총계", "주식수"]  # 재무상태 = 시점값


def quarterly_standalone(fin):
    """4Q 행의 손익은 연간 누계다 → 4Q 단독 = 연간 − (1Q+2Q+3Q).
    1~3분기가 하나라도 빠진 해의 4Q는 단독값을 믿을 수 없어 NaN."""
    f = fin.copy()
    f["q"] = f["분기"].str[0].astype(int)
    f = f.sort_values(["종목코드", "연도", "q"])
    for col in FLOW_ITEMS:
        f[col + "_q"] = f[col]
        g = f.groupby(["종목코드", "연도"])
        s123 = g[col].transform(lambda x: x[f.loc[x.index, "q"] < 4].sum(min_count=3))
        n123 = g["q"].transform(lambda x: (x < 4).sum())
        m4 = f["q"] == 4
        f.loc[m4, col + "_q"] = np.where(n123[m4] == 3, f.loc[m4, col] - s123[m4], np.nan)
    return f


def fundamental_table(fin=None):
    """종목·분기별 TTM 지표와 수정 F-score. 사용가능일 기준으로 반환."""
    if fin is None:
        fin = pd.read_csv(FIN, dtype={"종목코드": str})
    fin["종목코드"] = _code(fin["종목코드"])
    fin["사용가능일"] = pd.to_datetime(fin["사용가능일"])
    f = quarterly_standalone(fin)
    f = f.sort_values(["종목코드", "연도", "q"]).reset_index(drop=True)
    g = f.groupby("종목코드")
    for col in FLOW_ITEMS:
        f[col + "_ttm"] = g[col + "_q"].transform(lambda x: x.rolling(4, min_periods=4).sum())
    f["roa"] = f["당기순이익_ttm"] / f["자산총계"]
    f["opm"] = f["영업이익_ttm"] / f["매출액_ttm"].where(f["매출액_ttm"] > 0)
    f["turn"] = f["매출액_ttm"] / f["자산총계"]
    f["lev"] = f["부채총계"] / f["자산총계"]
    lag4 = lambda c: g[c].shift(4)   # 1년 전 같은 분기
    # 수정 Piotroski F-score — 현금흐름·유동비율 데이터가 없어 9개 중 7개만 쓴다.
    items = {
        "f_roa": f["roa"] > 0,                                 # 수익성: 순이익 흑자
        "f_op": f["영업이익_ttm"] > 0,                          # 수익성: 영업이익 흑자 (CFO 대용)
        "f_droa": f["roa"] > lag4("roa"),                      # 수익성 개선
        "f_dlev": f["lev"] < lag4("lev"),                      # 부채비율 하락
        "f_eq": f["주식수"] <= lag4("주식수") * 1.001,           # 신주 발행 없음 (희석 없음)
        "f_dopm": f["opm"] > lag4("opm"),                      # 마진 개선 (매출총이익률 대용)
        "f_dturn": f["turn"] > lag4("turn"),                   # 자산회전율 개선
    }
    valid = lag4("roa").notna() & f["roa"].notna()
    f["fscore"] = sum(v.astype(int) for v in items.values()).where(valid)
    # 이익 서프라이즈 근사 (SUE): 분기 영업이익의 전년동기 대비 변화 ÷ 자산
    #   애널리스트 컨센서스가 없으므로 계절 랜덤워크 기대치(Foster 1977 / Bernard&Thomas 1989)
    f["sue"] = (f["영업이익_q"] - g["영업이익_q"].shift(4)) / f["자산총계"]
    f["op_yoy"] = (f["영업이익_ttm"] / g["영업이익_ttm"].shift(4) - 1).where(
        g["영업이익_ttm"].shift(4) > 0)
    return f[["종목코드", "사용가능일", "연도", "분기", "fscore", "sue", "op_yoy",
              "roa", "opm", "lev"]]


def asof_matrix(tbl, col, dates, codes):
    """사용가능일 기준 as-of(backward) — 공시 전 정보는 쓰지 않는다."""
    t = tbl.dropna(subset=[col]).sort_values("사용가능일")
    t = t.drop_duplicates(["종목코드", "사용가능일"], keep="last")
    t = t.assign(_stamp=(t["사용가능일"] - pd.Timestamp("1970-01-01")).dt.days)   # 일 단위

    def ff(v):
        m = (t.pivot_table(index="사용가능일", columns="종목코드", values=v, aggfunc="last")
               .reindex(columns=codes))
        return m.reindex(m.index.union(dates)).sort_index().ffill().reindex(dates)

    m, stamp = ff(col), ff("_stamp")
    today = np.asarray((dates - pd.Timestamp("1970-01-01")).days)[:, None]
    # 18개월 넘게 갱신이 없으면 낡은 값 — 버린다
    return m.where((today - stamp.values) <= 550)


def fundamental_features(dates, codes):
    if not os.path.exists(FIN):
        return {}
    tbl = fundamental_table()
    return {k: asof_matrix(tbl, k, dates, codes) for k in ["fscore", "sue", "op_yoy"]}


# ══════════════════════════════════════════════════════════════
# 동종업계(섹터) 지표
# ══════════════════════════════════════════════════════════════
def load_sectors(codes):
    if not os.path.exists(SECTORS):
        return {c: "기타" for c in codes}
    s = pd.read_csv(SECTORS, dtype={"종목코드": str})
    m = dict(zip(_code(s["종목코드"]), s["섹터"].fillna("기타")))
    return {c: m.get(c, "기타") for c in codes}


def sector_features(panel, chart, fund, sector_of):
    """같은 섹터 안에서 비교한 값. 섹터 종목이 3개 미만이면 전체와 비교(NaN→중립)."""
    codes = panel["codes"]
    sec = pd.Series(sector_of).reindex(codes)
    groups = {k: list(v.index) for k, v in sec.groupby(sec)}
    out = {}

    def within_rank(mat):
        r = pd.DataFrame(np.nan, index=mat.index, columns=mat.columns)
        for _, cols in groups.items():
            if len(cols) >= 3:
                r[cols] = mat[cols].rank(axis=1, pct=True)
        return r.fillna(0.5)

    def sector_mean(mat):
        r = pd.DataFrame(np.nan, index=mat.index, columns=mat.columns)
        for _, cols in groups.items():
            m = mat[cols].median(axis=1)
            for c in cols:
                r[c] = m
        return r

    if "s_value" in panel:
        out["sec_value_rank"] = within_rank(panel["s_value"])        # 1 = 섹터 내 가장 쌈
    out["sec_mom"] = sector_mean(chart["ret120"])                    # 섹터 6개월 수익 (산업 모멘텀)
    out["sec_mom_rank"] = out["sec_mom"].rank(axis=1, pct=True)
    if "op_yoy" in fund:
        out["sec_earn"] = sector_mean(fund["op_yoy"].clip(-1, 3))    # 섹터 이익 증가율 중앙값
    return out


# ══════════════════════════════════════════════════════════════
# 시장 지표 (regime.py의 정의를 그대로 따른다)
# ══════════════════════════════════════════════════════════════
def market_features(close):
    ret = close.pct_change()
    mret = ret.mean(axis=1, skipna=True)
    idx = (1 + mret.fillna(0)).cumprod()
    ma200 = idx.rolling(200, min_periods=100).mean()
    breadth = (close > close.rolling(60, min_periods=30).mean()).sum(axis=1) \
        / close.notna().sum(axis=1).replace(0, np.nan)
    dd = idx / idx.cummax() - 1
    vol = mret.rolling(20, min_periods=10).std() * np.sqrt(252)
    m = pd.DataFrame({"idx": idx, "ma200": ma200, "trend": idx / ma200 - 1,
                      "breadth": breadth, "dd": dd, "vol": vol})
    reg = pd.Series(None, index=m.index, dtype=object)
    ok = m["ma200"].notna()
    reg[ok & (m["trend"] >= 0) & (m["breadth"] >= 0.55)] = "강세장"
    reg[ok & (m["trend"] >= 0) & (m["breadth"] < 0.55)] = "횡보장"
    reg[ok & (m["trend"] < 0)] = "약세장"
    reg[ok & (m["dd"] <= -0.15)] = "급락장"
    m["regime"] = reg
    # 급락장 안에서도 '추가 하락 중'과 '바닥 확인 후 반등 중'은 다르다
    m["rebound"] = m["idx"] / m["idx"].rolling(20, min_periods=10).min() - 1
    return m


# ══════════════════════════════════════════════════════════════
# 수급
# ══════════════════════════════════════════════════════════════
def flow_strength(dates, codes):
    """외국인 20일 순매수 ÷ 시가총액. 파일이 없으면 None."""
    if not (os.path.exists(FLOWS) and os.path.exists(MCAP)):
        return None
    fl = pd.read_csv(FLOWS, dtype={"종목코드": str})
    fl["종목코드"] = _code(fl["종목코드"])
    fl["날짜"] = pd.to_datetime(fl["날짜"])
    f20 = (fl.pivot_table(index="날짜", columns="종목코드", values="외국인순매수")
             .reindex(index=dates, columns=codes).rolling(20, min_periods=10).sum())
    mc = pd.read_csv(MCAP, dtype={"종목코드": str})
    mc["종목코드"] = _code(mc["종목코드"])
    mc["날짜"] = pd.to_datetime(mc["날짜"])
    cap = (mc.pivot_table(index="날짜", columns="종목코드", values="시가총액")
             .reindex(index=dates, columns=codes).ffill())
    return f20 / cap


def build_all():
    """판단 엔진이 쓰는 모든 지표를 한 번에. 반환: (panel, feats dict)"""
    p = load_panel()
    ch = chart_features(p["close"], p["volume"])
    fu = fundamental_features(p["dates"], p["codes"])
    sector_of = load_sectors(p["codes"])
    se = sector_features(p, ch, fu, sector_of)
    mk = market_features(p["close"])
    fs = flow_strength(p["dates"], p["codes"])
    feats = {**ch, **fu, **se}
    if fs is not None:
        feats["flow"] = fs
    feats["rank"] = p["score"].rank(axis=1, ascending=False, method="first")
    p["sector_of"] = sector_of
    p["market"] = mk
    return p, feats
