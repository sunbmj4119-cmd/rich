# 시중 자동매매·투자보조 프로그램 비교 분석 (2026-10-25)

이 시스템에 무엇을 더할지 고르기 위해, 국내외 32곳의 서비스가 **예측값·분석값을 어떻게 보여 주는지**와
**어떤 편의·안전 기능을 넣었는지** 조사했습니다. 화면으로 보고 고르는 곳은 [`docs/features.html`](features.html)입니다.

> **조사 방법과 한계** — 웹 검색 결과의 요약·발췌로 확인했고, 이 환경에서는 원문 페이지를 직접 열지 못했습니다
> (일부 오픈소스는 PyPI 소스코드를 받아 지표 이름을 직접 확인). 수치 대부분은 **회사가 스스로 발표한 것**이며,
> 확인하지 못한 항목은 적지 않았습니다. 투자 판단의 근거가 아니라 **화면·기능 설계의 참고 자료**입니다.

---

## 1. 결론

1. **잘하는 곳은 숫자 하나가 아니라 근거와 범위를 같이 보여 준다.**
   등급별 과거 성적표(Zacks·Portfolio123·Stockopedia), "N건 중 M건" 빈도(알파스퀘어·Tickeron),
   기준선 비교(Danelfin "59% 대 평균 52%"), 가격 범위(TipRanks 최고/평균/최저, 중앙은행 fan chart).
2. **검증은 서비스 발표와 자주 엇갈린다.** Zacks Rank 실제 운용 계좌는 SPY와 같은 수익(CXO Advisory),
   Tickeron '고신뢰' 패턴의 실제 목표 도달 약 18%, 라씨 "적중률 83%"는 산출 기준 비공개.
3. **편의 기능의 중심은 '멈추는 버튼'과 '확인 후 주문'.** Freqtrade `/pause`(신규만 중단)와 `/stop`의 구분,
   Trade Ideas·Portfolio123·Webull의 신호 + 확인, 키움·토스의 서버 감시주문(손절·트레일링·OCO).
4. **한국에서는 규제가 강해지고 있다.** 2024-08 자본시장법 개정으로 유사투자자문의 양방향 채널·수익률 과장 금지,
   종목 자동선택 프로그램 영업은 미등록 투자일임에 해당할 수 있다는 금융위 해석, 2026년 AI 자동매매 사칭 경보.

## 2. 이 시스템에 반영한 것과 그 검증 결과

| 기능 | 가져온 곳 | 이 데이터로 확인한 것 | 기본값 |
|---|---|---|---|
| 30일 가격 범위(10/25/50/75/90%) | Tickeron · TipRanks · BoE fan chart | 2019~23으로 만든 80% 범위가 2024~ **64,863건 중 79.8%**를 담음 (50% 범위 51.6%) → **검증됨** | 켬 |
| 순위 구간별 과거 성적 | Zacks · Portfolio123 · Stockopedia | **1~10위만** 두 구간 모두 시장 초과(+0.36%p / +3.25%p). 순위상관 train 0.15 · test 0.81 | 켬 |
| 팩터 성적표 A+~F (섹터 상대) | Seeking Alpha · 초이스스탁 · Simply Wall St | 순위의 분해 표시 (예측 주장 없음) | 켬 |
| 점수 기여도 | Danelfin | 종합점수 = 팩터 가중합과 오차 0.07 이내로 일치 | 켬 |
| 종목별 엔진 매매 기록 | 라씨 매매비서 | 비용 반영 가설 성과, 표본 수 표시 | 켬 |
| 유사 차트 패턴 | 알파스퀘어 · 딥서치 | 방향 적중 **47.6% < '무조건 상승' 54.2%**, 순위상관 −0.015 → **검증 탈락** | **끔** |
| 시장 점수·상태 분포 | 초이스스탁 마켓스코어 | 현황 표시 | 켬 |
| 섹터 히트맵 | Finviz | 현황 표시 | 켬 |
| 성과 보고서 | TradingView · QuantStats · pyfolio · 인텔리퀀트 | 월별 히트맵, 수중 차트, 최악 낙폭 5개, 청산 사유별, MAE, PSR | 켬 |
| 설정 바꿔보기 72개 | 젠포트 · 인텔리퀀트 · Composer | 현재보다 두 구간 모두 나은 조합 4/72, 운으로 생길 수 있는 폭 Sharpe ≈0.29 함께 표시 | 켬 |
| 실전(모의) 추적 + 기대 범위 | pyfolio cone · Composer OOS · QuantConnect | 모의 계좌를 백테스트 ±1σ/±2σ 위에 겹침 | 켬 |
| 자동 매매일지 | Tradervue · 삼성 mPOP | 주문마다 이유·순위·국면·가격 범위·승인 여부 | 켬 |
| 숫자 읽는 법 | 퀀터스 | 지표마다 기준 해설 | 켬 |
| 승인 후 주문 | Trade Ideas · Portfolio123 · Webull | 승인 전 주문 0건, skip으로 종목 제외 (테스트) | 자동 |
| 알림 (텔레그램·ntfy) | 라씨 · 핀트 · TradingView · Freqtrade | 키 없으면 네트워크 호출 없음 (테스트) | 끔 |
| 신규매수만 일시정지 | Freqtrade `/pause` | 매도는 그대로, 매수만 관망 (테스트) | 버튼 |
| 원격 조종 버튼 | Freqtrade 텔레그램 명령 | GitHub Actions `autotrade-control`: approve·pause·resume·kill·unkill·status | — |
| 날짜별 계획 보관 | Collective2 (신호 사후 수정 금지) | `data/autotrade/plans/YYYY-MM-DD.json` | 항상 |
| 재진입 대기 | Freqtrade CooldownPeriod | 10종목·10일: train Sharpe **0.76→0.27** → **검증 탈락** | **0일(끔)** |
| 가설 성과 고지 · 라벨 | CFTC 4.41 · SEC Marketing Rule · Collective2 | BACKTEST / PAPER 라벨, 고지문 | 항상 |

## 3. 예측값을 보여 주는 원칙

**따른 것**
- 확률은 빈도로: "10번 중 8번", "6,520건 중 3,195건" (Gigerenzer; 알파스퀘어·Tickeron 표기)
- 한 점이 아니라 범위로, 그 범위가 실제로 맞았는지까지 (TipRanks, BoE fan chart)
- 등급은 같은 섹터 안 백분위임을 밝힘 (Seeking Alpha, Stock Rover)
- "왜"를 팩터별 기여로 분해 (Danelfin)
- 순위 버킷별 성적을 표본 수·독립 표본 수와 함께 (Zacks, Portfolio123, Stockopedia; Portfolio123의 "상위 버킷이 중간보다 낮으면 실패" 기준)
- 백테스트·모의·실전 라벨 분리, 실전 시작일 이후 따로 추적 (Composer OOS, pyfolio `live_start_date`)
- 여러 설정을 시험했으면 운의 몫을 함께 표시 (Deflated Sharpe, Bailey & López de Prado; Harvey·Liu·Zhu t>3)

**피한 것**
- 표본오차보다 정밀한 숫자 (예: 59.48%), 반올림이 맞지 않는 odds (Tickeron 92/99를 90%로)
- 보정 안 된 '신뢰도'를 확률처럼 표기 (Tickeron 고신뢰 패턴 실제 도달 약 18%)
- 목표가 하나 — 애널리스트 12개월 목표가는 기간 말 도달 26% (Bradshaw·Brown), FINRA 2241은 근거·위험 동반 요구
- 기간·비용·생존편향 없는 승률, 설명 없는 시작일·수치 변경 (WallStreetZen, TipRanks 사례)
- 최근 수익률 순 정렬 — 과최적화 전략이 위로 (젠마켓 기본 정렬)

## 4. 서비스별 요약

### 국내 증권사
- **키움 영웅문·REST API** — 서버 자동감시주문(최대 30일): 이익실현·이익보존·손실제한·트레일링스탑, 매수 전에 거는 '잔고편입 스탑로스', 감시 기준가 선택(평균매입가/현재가). 키우GO 로보는 테스트베드 수익률을 1주~2년·연환산·누적 표와 표준편차·샤프·최대손실로 표시. REST 조건검색은 WebSocket 실시간 편입·이탈.
  [0624](https://download.kiwoom.com/hero4_help_new/0624.htm) · [0621](https://download.kiwoom.com/hero4_help_new/0621.htm) · [키우GO](https://www.kiwoom.com/wm/gbi/rm/gbiRoboMain) · [REST 조건검색](https://algolab.co.kr/blog/kiwoom-rest-condition-search-websocket-2026) · [매일신문](https://www.imaeil.com/page/view/2026052110031715229)
- **토스증권** — 익절·손절 '한번에 설정' OCO 조건주문(가격/수익률 지정), Open API의 SINGLE·OCO·OTO, 주식 모으기(무료), AI 시그널(가격 변동 원인 요약). 종목 점수·매매신호는 확인되지 않음.
  [공지](https://corp.tossinvest.com/en/post?type=notice&id=5929&category=129) · [OCO/OTO](https://algolab.co.kr/blog/toss-open-api-conditional-order-oco-oto-2026) · [헤럴드](https://www.heraldk.com/article/2025111117325540433)
- **한국투자증권** — Open API, 자연어 MCP 서비스(2026-09), AI 종목분석 '지금 종목은'. 주문 API 스펙은 [공식 예제](https://github.com/koreainvestment/open-trading-api)로 확인해 `src/brokers.py`에 구현.
  [이코노미스트](https://economist.co.kr/article/view/ecn202610140048)
- **미래에셋 MAPS** — '마켓Now' 2시간마다 한 문장 시황, AI 이슈체크·사업요약. AI 자동주문은 단계적 도입 예정. [머니투데이](https://www.mt.co.kr/stock/2026/10/01/2026100109153897204)
- **삼성 mPOP** — 홈 '마이픽'에 매매일지·보유 리포트, 통합 주문 버튼. [아시아경제](https://core.asiae.co.kr/article/2025121910140338403)
- **신한 SOL** — 해외 자동감시주문 90일, '시가 대비' 조건 추가 후 이용률 4배. [스마트투데이](https://www.smarttoday.co.kr/ko-kr/articles/84180)

### 퀀트 플랫폼
- **젠포트(뉴지스탁)** — 약 120팩터, 백테스트→가상매매→실전 3단계, 젠트레이더로 증권사 계좌 완전자동. 젠마켓은 최근 1개월 수익률 순 기본 정렬. [매뉴얼](https://genport.newsystock.com/GenTrader/Down/genport_manual_v1.7_ver.pdf) · [경북매일](https://kbmaeil.com/article/20260223500582)
- **인텔리퀀트** — 블록 조립 전략, 거래비용 반영 백테스트, 총수익·MDD·샤프·승률·손익비 한 화면. 무료 5년·50회/일, 유료 월 3만·10만원. [멤버십](https://www.intelliquant.ai/membership) · [블로그](https://blog.intelliquant.ai/post/quant-backtest-guide)
- **퀀터스** — 해석 가이드(CAGR 15%↑ 우수, MDD −20% 이내 안정, 샤프 1↑ 우수, 3년 이상·여러 장세 포함). [가이드](https://www.quantus.kr/en/guide/backtest)
- **콴텍 · 핀트** — 투자일임. 핀트는 AI가 사고팔 때마다 푸시. [콴텍](https://apps.apple.com/kr/app/id1622337971) · [핀트](https://apps.apple.com/kr/app/id1434650965)

### 분석·신호 앱
- **알파스퀘어** — 유사 차트를 유사도 순으로, "10개 중 7개 상승 = 70%" 빈도 표기, 약 2개월 뒤 예상주가, 5단계 의견 + 목표가·손절가, 600봉 기준 전략 시뮬레이션. 스탠다드 월 19,800원·프리미엄 69,900원(앱스토어 표기). [AI예측](https://alphasquare.oopy.io/tutorial/ai-predict) · [매매신호](https://alphasquare.oopy.io/tutorial/trading-signal)
- **초이스스탁(US)** — 스마트스코어 5축(22팩터), 스마트스코어·밸류에이션·수급 세 조건을 모두 통과해야 매수, **매수→보유→매도→관망 4단계 신호**, 매수·보유 비율로 '마켓스코어'. [시그널](https://www.choicestock.co.kr/stock/signal)
- **라씨 매매비서(씽크풀)** — 매수 10·매도 17 알고리즘, 매수가 기준 맞춤 매도 신호, 종목별 "50회·평균 26일·적중 83%·평균 +10.9%" 카드. 산출 기준 비공개, "신호 종목 다수가 거래량이 적다"는 리뷰. [뉴스핌](https://www.newspim.com/news/view/20260506000166) · [씽크풀](https://www.thinkpool.com/signal)
- **증권플러스** — 장전·장중·장마감 하루 3회 이슈 요약 푸시. [앱스토어](https://apps.apple.com/us/app/id913934976)

### 해외 점수 서비스
- **Danelfin** — AI 점수 1~10 = 3개월 시장 초과 확률, "59% vs 평균 52%(+7%p)" 기준선, 신호별 ±%p 기여. [Kiplinger](https://kiplinger.com/investing/stocks/stocks-to-buy/603536/can-ai-beat-the-market-10-stocks-to-watch)
- **Zacks Rank** — 1~5, 등급별 연수익표(1988~2025 가설: #1 23.8% … #5 2.6%, 비용 0). 실제 운용 계좌(1995~2009)는 SPY와 같은 6.5%. [공시](https://www.zacks.com/performance_disclosure) · [CXO](https://www.cxoadvisory.com/fundamental-valuation/are-zacks-rankings-exploitable/)
- **Seeking Alpha Quant** — 팩터 등급 A+~F를 섹터 대비로, Quant·SA 애널리스트·월가 의견을 나란히. 스스로 "상당 부분 백테스트, 엄밀히 투자 가능하지 않음". [FAQ](https://help.seekingalpha.com/premium/quant-ratings-and-factor-grades-faq) · [PRO Quant](https://seekingalpha.com/article/4824424-inside-the-pro-quant-portfolio)
- **Tickeron** — confidence %, "N건 중 M건" odds. 고신뢰 패턴 20,755건 중 목표 도달 3,744건(약 18%). [패턴 스캐너](https://tickeron.com/stock-pattern-scanner/bullish-broadening-bottom-to-buy)
- **TipRanks** — Smart Score 1~10(상대 분포, 8요인 각 12.5%), 목표가 최고/평균/최저 + 애널리스트 수, 애널리스트 성공률에 Z검정. [Smart Score](https://www.tipranks.com/news/labs/get-a-full-stock-analysis-with-tipranks-smart-score)
- **Morningstar** — 적정가치 + 별 1~5, 불확실성이 클수록 별 기준 할인·프리미엄 폭을 넓힘. [불확실성 등급](https://morningstar.co.uk/uk/news/227180/what-is-the-morningstar-uncertainty-rating.aspx)
- **Simply Wall St** — Snowflake 5축 레이더, 축마다 체크 6개. [도움말](https://support.simplywall.st/hc/en-us/articles/360001740916-How-does-the-Snowflake-work)
- **Portfolio123 · Stockopedia · WallStreetZen** — 버킷별 성과 막대, 출시 후 실성과 매일 갱신. WallStreetZen은 기간·수치를 설명 없이 바꿈. [P123](https://portfolio123.customerly.help/en/articles/14052-dataminer-operation-ranks-performance) · [Stockopedia](https://learn.stockopedia.com/en/articles/3513123-tracking-the-stockranks-performance) · [Zen](https://wallstreetzen.com/zen-ratings)
- **Finviz** — 섹터 히트맵(크기=시총, 색=등락). [블로그](https://finviz.com/blog/the-finviz-matrix-market-breadth-visualized/)

### 해외 자동매매·보고서 도구
- **TradingView** — Strategy Tester(Overview·Performance·Trades analysis·List of Trades, 거래별 run-up/drawdown), 알림 앱·이메일·웹훅. [Strategy Report](https://www.tradingview.com/support/solutions/43000764138-tradingview-strategy-report-how-to-start/)
- **Trade Ideas Holly** — 진입·손절·목표·예상 보유시간, 수동(사람이 주문) ↔ 자동(Brokerage Plus). [AI 전략](https://www.trade-ideas.com/ai-strategies/)
- **Composer** — OOS 시작일 이후 성과 분리, 마감 전 일괄 실행, 임계치 리밸런싱. [백테스트](https://help.composer.trade/article/215-7-how-do-backtests-work-in-composer)
- **QuantConnect** — 월별 히트맵·롤링 Sharpe·위기구간, `--live-results`로 실거래를 백테스트 위에 겹침, 텔레그램 알림. [리포트](https://www.quantconnect.com/docs/v2/cloud-platform/backtesting/report)
- **Freqtrade** — dry-run, `max_open_trades`, Protections(StoplossGuard·MaxDrawdown·CooldownPeriod), 텔레그램 `/pause`(신규만 중단)·`/stop`·`/forceexit`. [PyPI](https://pypi.org/project/freqtrade/)
- **Hummingbot · eToro** — 손실률 도달 시 자동 정지, 복사 전체 손절. [Kill switch](https://hummingbot.org/client/global-configs/kill-switch/) · [CopyTrader](https://www.etoro.com/copytrader/how-it-works/)
- **Collective2** — 모든 성과 '가설' 표기, 비용 포함 평가, 신호 사후 수정 금지. [설명](https://elitetrader.com/et/threads/how-does-collective2-work.229089/)
- **QuantStats · pyfolio** — 티어시트(Worst Drawdowns, 수중 차트, 월별 히트맵, PSR), `live_start_date` 기대 범위(cone). [QuantStats](https://pypi.org/project/quantstats/) · [pyfolio-reloaded](https://pypi.org/project/pyfolio-reloaded/)
- **Interactive Brokers** — 주문 금액·수량·가격 이탈 확인(precautionary settings), 조건부 주문. [Precautionary](https://interactivebrokers.co.uk/en/software/tws.bak/usersguidebook/configuretws/define_precautionary_settings.htm)
- **3Commas 교훈** — 2022년 API 키 약 10만 개 유출. 출금 권한 없는 최소권한 키, 정기 교체. [Halborn](https://www.halborn.com/blog/post/explained-the-3commas-breach-december-2022)

## 5. 규제·위험 (한국)

- 2024-08-14 개정 자본시장법: 유사투자자문의 유료 양방향 채널(오픈채팅 등) 금지, 사실과 다른 수익률·이익보장 광고 금지. [금융위](https://fsc.go.kr/edu/news/83077?curPage=1)
- 2025 점검(2026-04 발표): 105개사 위법 133건, 수익률 과장·필수 고지 누락 증가. [헤럴드](https://www.heraldk.com/article/2026041919001151230)
- 종목이 자동 선택되는 자동매매 프로그램 영업 → 미등록 투자일임 해당 가능(금융위 해석). [금융위 회신](https://better.fsc.go.kr/fsc_new/replyCase/LawreqDetail.do?stNo=11&muNo=85&muGpNo=75&lawreqIdx=3505)
- 2026-03 가짜 AI 자동매매 프로그램 경보 ("호가창·차트까지 갖춰 겉모습으로 구별 어려움"). [헤럴드](https://www.heraldk.com/article/2026030814271729715)
- **이 저장소에 주는 의미**: 개인 계좌용으로만 쓰고, 대시보드를 공개하더라도 유료화·개별 추천을 하지 마세요.
  적중률·수익률을 보여 줄 때는 기간·비용·표본 수·가설 성과 고지를 함께 둡니다(이미 반영).

## 6. 성과 보고서 표준 구성 (적용 현황)

| 항목 | 출처 | 적용 |
|---|---|---|
| 가정 공개(기간·유니버스·체결·비용), BACKTEST/PAPER/LIVE 라벨 | NFA 2-29, SEC Marketing Rule | ✅ |
| CAGR·Sharpe·Sortino·Calmar·MDD·최장 회복기간·노출 | QuantStats, LEAN | ✅ |
| 승률·Profit factor·손익비·청산 사유별 | TradingView, Freqtrade | ✅ |
| 자산곡선 vs 벤치마크, 수중 차트, 월별 히트맵, 롤링 수익 | QuantStats, pyfolio | ✅ |
| Worst 5 Drawdowns | QuantStats | ✅ |
| MAE/MFE | TradingView run-up/drawdown, LEAN TradeBuilder | ✅ |
| PSR, 여러 설정 시험 시 운 보정(Deflated Sharpe 기준선) | Bailey & López de Prado | ✅ (근사) |
| 실전 시작일 이후 분리 + 기대 범위 cone | pyfolio, Composer, LEAN | ✅ |
| VaR·CVaR, 위기구간 비교 | QuantStats, LEAN | 아직 없음 |
