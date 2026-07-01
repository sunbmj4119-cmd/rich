# 📈 주식 투자 매니저 (KOSPI 100)

매 평일 장 마감 후 자동으로 데이터를 수집·채점하고, **다음 날 예약매매(지정가 매수 / 손절·매도)를 결정할 수 있는 대시보드**를 생성하는 개인용 시스템입니다.

- 대시보드: `docs/index.html` (GitHub Pages로 서빙 → 저녁에 휴대폰으로 확인)
- 매매 기록: `docs/trade.html` (매수/매도 입력 → CSV 복사 → `data/my_trades.csv`에 붙여넣기)

---

## 🗓 하루 사용 흐름

1. **17:00** GitHub Actions(`daily.yml`)가 자동 실행 → 데이터 갱신·채점·리포트 생성·커밋.
2. **저녁** 대시보드(`index.html`)를 열어 확인:
   - 🔴 손절 / 🔵 매도 / 🟢 지금 사기 좋은 순서(BuyFit) / 💼 내 계좌
   - 종목을 터치 → 그래프·점수근거·과거 유사사례 승률·매수밴드·기대손익
3. 결정한 주문을 **증권사 앱에 예약**(지정가 매수, 손절가 매도).
4. 실제 체결되면 `trade.html`로 기록 → CSV를 `data/my_trades.csv`에 반영(커밋).

---

## 🔧 파이프라인 (`.github/workflows/daily.yml` 순서)

| 단계 | 스크립트 | 입력 → 출력 | 역할 |
|---|---|---|---|
| 1 | `src/update_prices.py` | KRX → `data/prices.csv` | 100종목 종가 append |
| 2 | `src/collect_flows.py` | KRX → `flows/ohlcv/marketcap/shorts.csv` | 수급·시세·공매도(최근 7일) |
| 3 | `src/score.py` | prices+financials+flows → `data/scores.csv` | 팩터 채점 → 종합점수 |
| 4 | `src/signal.py` | scores+`my_trades.csv` → `signals_today.csv`+`account.json` | 매수/매도/보유 판정 · 주식수 회계(부분매도·실현손익) |
| 5 | `src/build_data.py` | scores+signals+sectors → `docs/data.json` | BuyFit·타이밍·유사사례·기대손익·계좌·섹터편중·근거 |
| 6 | `src/report.py` | data.json(+robust.json) → `docs/index.html` | 대시보드 렌더 |

보조 수집: `src/collect_sectors.py` (FinanceDataReader → `data/sectors.csv`, 업종/섹터. 주기적 1회).

**검증 스크립트(수동/별도 워크플로우)**
- `src/backtest.py` — 비중첩 리밸런싱 백테스트(정석)
- `src/backtest_signal.py` — signal.py 로직 재현 백테스트(t점수→t+1체결)
- `src/robust_backtest.py` — 비용/생존편향 보정 시나리오 → `docs/robust.json`
- `src/optimize_weights.py` — 워크포워드 가중치 탐색
- `src/verify_factor.py` — 신규 팩터 IC/단조성/독립성 검증 (score.py 반영 **전** 필수)
- `src/dart_financials.py` / `src/backfill.py` / `src/collect_universe.py` — 재무·과거·유니버스 수집

---

## 🧮 점수 엔진 (`src/score.py` + `config/weights.yaml`)

매 거래일 종목 간 **횡단면 백분위(0~100)** 로 각 팩터를 표준화한 뒤 가중합.

| 팩터 | 가중치 | 내용 |
|---|---|---|
| 가치(value) | 0.32 | PER·PBR (낮을수록 高) — 종가/EPS, 종가/BPS |
| 수익성(profit) | 0.22 | ROE + 영업이익률 + ROE개선추세 |
| 수급(flow) | 0.20 | 외국인 20일 순매수 / 시가총액 (횡단면 z) |
| 성장(growth) | 0.18 | 매출·영업이익·순이익 전년동기 성장률 |
| 모멘텀(momentum) | 0.08 | 20·60일 + 12-1개월(단기반전 제거) |
| 안정성/거래추세/감성 | 0.00 | 계산은 되나 현재 미사용 |

- **미래참조 차단**: 재무는 `사용가능일` 기준 as-of(backward) join.
- 결측은 중립 50점으로 채움.

## 📊 매매 전략 (`src/signal.py`)

- 종합점수 **상위 20 풀**, **10위 이내** 진입, 보유가 **20위 밖**으로 이탈 + **최소 30거래일** 보유 시 매도.
- **손절 -10%**, **트레일링 스탑 -8%**(진입 후 고점 대비) — 둘 중 먼저 닿는 값이 감시가.
- **외국인 순매도** 종목은 매수 보류(⚪).
- **BuyFit** = z(종합점수) + 0.25 × 타이밍, 타이밍 = 0.6·(-z 60일낙폭) + 0.4·(-z 20일수익) → 단기 평균회귀. "점수는 좋은데 지금 눌린" 종목을 앞으로.

---

## 🖥 로컬 실행

```bash
pip install pandas numpy pyyaml finance-datareader
# (수급/공매도 수집은 pykrx + KRX 로그인 필요: KRX_ID / KRX_PW)

python src/collect_sectors.py  # 업종/섹터 → data/sectors.csv (주기적 1회)
python src/score.py          # 점수 재계산
python src/signal.py         # 신호 재생성 (+ account.json)
python src/build_data.py     # 대시보드 데이터
python src/report.py         # docs/index.html
python src/robust_backtest.py  # 현실적 기대수익(느림) → docs/robust.json
```
> Windows 콘솔에서 이모지 출력 오류가 나면 `set PYTHONIOENCODING=utf-8` 후 실행.

---

## ⚠️ 한계 (반드시 인지)

- **생존편향**: `prices.csv`는 2026년까지 살아남은 현재 KOSPI 100종목을 2018년까지 백필한 것. 실제 상장폐지 종목(약 4,000건)은 데이터에 없어, 백테스트 수익은 **실전보다 높게** 나옵니다.
- `data/universe.csv` 스냅샷이 1개뿐이라 point-in-time 생존편향 보정은 아직 **미작동**(대시보드는 이를 정직하게 "일부만 보정"으로 표시). 장기적으로 월간 스냅샷을 쌓으면 개선됩니다.
- `robust_backtest.py`의 Sharpe/연율은 30일 수익을 **매일 중첩 표집**해 정밀도를 과대평가합니다. 표본은 강세장 비중이 큰 2018~2026 구간 → **실전 기대는 더 보수적으로**.
- `data/shorts.csv`(공매도)는 2026-06-22부터 수집만 되고 아직 점수에 미반영. `verify_factor.py` 검증 후 팩터화 예정.

---

## 📁 데이터 스키마 요약

| 파일 | 컬럼 |
|---|---|
| `prices.csv` | 날짜, 종목코드, 종목명, 종가, 거래량 (시가총액/PER/PBR은 미사용·빈값) |
| `ohlcv.csv` | 날짜, 종목코드, 시/고/저/종가, 거래량, 거래대금 |
| `flows.csv` | 날짜, 종목코드, 외국인·기관·개인 순매수 |
| `marketcap.csv` | 날짜, 종목코드, 시가총액, 상장주식수 |
| `financials.csv` | 종목코드, 연도, 분기, 사용가능일, 매출/영업이익/순이익, ROE, EPS, BPS … |
| `my_trades.csv` | 날짜, 종목명, 구분(매수/매도), 금액, **수량**(선택) ← **사용자가 기록**. 매수는 수량·금액 중 하나 이상, 매도는 수량=부분매도·공란=전량 |
| `signals_today.csv` | 구분, 종목코드/명, 순위, 점수, 보유일, 평단가, 현재가, 수익률%, 사유, 손절가, 트레일가, 감시가, 투자금액, 수량, 실현손익 |
| `sectors.csv` | 종목코드, 종목명, 업종(KRX), 섹터(그룹) — `collect_sectors.py` 생성 |
| `account.json` | 실현손익 합계, 투자원금, 보유종목수 — `signal.py` 생성 |
