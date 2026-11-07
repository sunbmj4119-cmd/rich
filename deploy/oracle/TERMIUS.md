# 터미우스(Termius) 앱으로 오라클 서버에서 할 일

휴대폰의 터미우스 앱으로 오라클 서버에 접속해서 명령을 한 줄씩 붙여넣는 방식입니다.
비밀값(키·비밀번호)은 화면에 보이지 않게 입력되고, 서버 안 `~/.rich/secrets.env` 에만 저장됩니다. **채팅에는 절대 붙여넣지 마세요.**

---

## 0. 처음 한 번: 서버 등록 (이미 접속해 본 적이 있으면 건너뛰세요)

1. 터미우스 앱 → 아래 **Vaults**(또는 Hosts) → 오른쪽 위 **＋** → **New Host**
2. **Address**: 오라클 콘솔의 서버 Public IP (Compute → Instances → 내 서버)
3. **Username**: Ubuntu 이미지면 `ubuntu`, Oracle Linux 면 `opc`
4. **Key**: Keys → 서버 만들 때 받은 개인 키 파일(.key)을 가져오기 (파일 앱에서 선택)
5. 저장 → 목록에서 그 서버를 누르면 검은 화면(터미널)이 열립니다

## 1. 붙여넣기·특수키 쓰는 법

- **붙여넣기**: 이 문서의 명령 한 줄을 길게 눌러 복사 → 터미우스 검은 화면을 길게 누르기 → **Paste** → 키보드의 **Enter**
- **Ctrl+C** (보고 있던 화면 빠져나오기): 키보드 위 줄의 **Ctrl** 을 누른 다음 **c**
- 화면에 아무 반응이 없어 보여도 비밀값 입력 중이면 정상입니다 (글자를 숨김)
- 끝낼 때: `exit` 입력 또는 앱 닫기 (뒤에서 돌리는 작업은 계속됩니다)

---

## 2. 지금 할 일 (순서대로, 각 줄을 붙여넣고 Enter)

### ① 최신 코드 받기
```
cd ~/rich && git pull
```
`Already up to date.` 또는 바뀐 파일 목록이 나오면 됩니다.

### ② 서버 상태 확인
```
bash ~/rich/deploy/oracle/setup.sh --check
```
표에서 확인할 것:
- **GitHub 쓰기: 가능** — `아직` 이면 아래 '문제 해결 ①'
- **바깥으로 나가는 IP** — 토스 개발자 센터의 '허용 IP' 에 등록된 값과 같아야 합니다 (지난번 값: 129.225.178.181)

### ③ 한국거래소 아이디·비밀번호 넣기
```
bash ~/rich/deploy/oracle/setup.sh --keys
```
차례로 묻습니다. **이미 넣은 토스 값(TOSS_…)은 그냥 Enter 로 건너뛰고**, `KRX_ID:` 차례에 한국거래소(KRX 정보데이터시스템) 아이디를 붙여넣고 Enter,
`KRX_PW:` 에 비밀번호를 붙여넣고 Enter. (GitHub Secrets 에 넣은 것과 같은 값)

### ④ 토스 주문 명세 받기 (조회만 — 주문·돈은 건드리지 않음)
```
bash ~/rich/deploy/oracle/setup.sh --probe
```
`결과를 GitHub 에 올렸습니다` 가 나오면 채팅으로 **"점검 끝"** 이라고만 알려 주세요.
→ 제가 받은 명세로 토스 주문 형식을 맞추고 확인 표시를 합니다.

### ⑤ 한국거래소 긴 역사 자료 받기 (2~3시간, 뒤에서 돎)
```
cd ~/rich && git pull && bash ~/rich/deploy/oracle/setup.sh --history
```
20초쯤 뒤 처음 몇 줄이 나옵니다. `0) KRX 접속 확인 ✓` 이면 정상, `✗` 이면 화면을 캡처해 보내 주세요.
이미 받던 게 있으면 멈추고 최신 코드로 다시 시작합니다 (받아 둔 것은 이어서 씀).
진행을 보고 싶으면 (몇 번이든):
```
tail -n 5 ~/.rich/logs/krx_history.log
```
`ohlcv 종목 40/420 · … · 남은 시간 약 70분` 같은 줄이 늘어나면 정상입니다. 앱을 닫아도 됩니다.
끝나면 저절로 GitHub 에 올라갑니다 → 채팅으로 **"수집 끝"**.

### ⑥ 자동 실행 등록 (지금 해 둬도 안전 — 모의 상태면 아무것도 안 함)
```
bash ~/rich/deploy/oracle/setup.sh --cron
```
평일 08:40 아침 주문 · 16:10 잔고 대사를 등록합니다. 표의 '자동 실행' 줄에 둘 다 '등록' 이 보이면 됩니다.

---

## 3. 다음 할 일 (2026-11-06 점검 결과를 보고)

④ 점검에서 토큰·계좌·현재가·장 운영 시간은 정상이었고, 보유·매수 가능 금액·주문 목록은 '계좌 헤더 없음'으로 실패했습니다
(점검 코드가 계좌 식별값 0 을 '없음'으로 잘못 본 문제 — 고쳤습니다). 주문 형식은 명세 1.2.28 로 확인을 마쳤습니다.

### ⑦ 최신 코드 받고 점검 한 번 더
```
cd ~/rich && git pull && bash ~/rich/deploy/oracle/setup.sh --probe
```
"결과를 GitHub 에 올렸습니다" → 채팅으로 **"점검 끝"** (이번엔 보유·매수 가능 금액까지 200 이 나와야 합니다)

### ⑧ 토스 계좌를 거울 계좌로 연결 (주문은 아직 잠김 — 연결 점검 20거래일이 이때부터 시작)
```
bash ~/rich/deploy/oracle/setup.sh --link
```
- 계좌에 이미 있던 종목은 '직접 보유분'으로 기록되고 자동매매는 건드리지 않습니다.
- 자동매매에 쓸 금액(short.broker_capital)을 아직 안 정했으면 지금 현금 전부로 잡아 둡니다. 실제 돈 주문을 켜기 전까지는
  금액을 정한 뒤 이 명령을 다시 하면 됩니다.

### ⑨ 자동 실행 등록 (⑥을 아직 안 했으면)
```
bash ~/rich/deploy/oracle/setup.sh --cron
```

### 그 뒤 (제가 "이제 하세요" 라고 알려 드릴 때)

| 단계 | 명령 | 언제 |
|---|---|---|
| 1주 실주문 시험 | `bash ~/rich/deploy/oracle/setup.sh --smoke` | 연결 뒤 며칠, 평일 09:05~15:10 (KODEX 미국달러선물 1주를 사고 바로 팖 — 수십~수백 원 듦). **계좌에 현금 2만 원 이상** 필요 |
| 실제 돈 주문 최종 확인 | `bash ~/rich/deploy/oracle/setup.sh --live` | 준비도 점검표가 모두 ✓ 일 때만 |
| 실제 돈 주문 끄기 | `bash ~/rich/deploy/oracle/setup.sh --unlive` | 언제든 — 끄면 주문을 만들기만 하고 보내지 않음 |

- **입금하거나 자동매매 금액을 정하면 `--link` 를 다시** 하세요 (연결할 때의 현금으로 배율을 잡습니다. 지금은 현금이 적어 주문이 거의 0주로 계산됨).
- 연결 뒤 같은 계좌 현금을 직접 쓰면 자동매매 주문이 현금 부족으로 빠질 수 있습니다 (그러면 대사가 알려 줍니다).
- **레버리지·인버스 ETF는 하지 않습니다** (2026-11-07 결정, 설정 `short.no_geared: true`). 전략에서 빼 두었고, 혹시 계획에 섞여도 서버가 매수를 보내지 않습니다.

## 4. 문제 해결

1. **GitHub 쓰기: 아직** → 토큰 방식:
   GitHub 웹 → 프로필 → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token →
   Only select repositories: `rich` → Repository permissions → **Contents: Read and write** → Generate → Copy, 그다음
   ```
   bash ~/rich/deploy/oracle/setup.sh --token
   ```
   토큰을 붙여넣고 Enter (화면에 안 보임).
2. **토스 토큰 실패 403** → 토스 '허용 IP' 에 ② 표의 IP 가 들어 있는지 확인.
3. **`git pull` 이 충돌** → `cd ~/rich && git stash && git pull` 후 다시.
4. 무엇이든 막히면 화면을 캡처해서 보내 주세요 (비밀값은 화면에 나오지 않습니다).
5. **급히 모든 주문 멈추기**: 휴대폰 GitHub 앱 → 저장소 rich → Actions → autotrade-control → Run workflow → `kill`
   (서버 쪽 실제 돈 주문만 끄려면 `setup.sh --unlive`)
