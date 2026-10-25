# Oracle 서버 설정 — 고정 IP에서 증권사 주문 보내기

판단(매수·매도 결정)은 지금처럼 GitHub Actions가 매일 17:00에 합니다.
이 서버는 **다음 날 08:40에 그 계획을 받아 증권사로 보내는 일만** 합니다.
토스증권 오픈API는 미리 등록한 IP에서 온 요청만 받기 때문입니다.

## 1. 서버 IP가 고정인지 확인 (Oracle 콘솔)

Compute → Instances → 내 서버 → Attached VNICs → VNIC 이름 → IPv4 Addresses → Public IP 옆 표시

- **Reserved**: 좋습니다. 서버를 지우지 않는 한 그대로입니다.
- **Ephemeral**: 서버를 껐다 켜도 유지되지만 서버를 지우면 바뀝니다. 바꾸려면 연필(Edit) → No public IP 저장 →
  다시 Edit → Reserved public IP → Create new. 이때 IP가 바뀌므로 토스에는 바뀐 IP를 등록합니다.

## 2. 서버 접속

콘솔의 Public IP, 사용자 이름(Ubuntu 이미지는 `ubuntu`, Oracle Linux는 `opc`), 서버를 만들 때 받은 개인 키 파일(.key)이 필요합니다.

```bash
ssh -i 키파일경로.key ubuntu@서버IP        # Mac은 먼저: chmod 600 키파일경로.key
```

휴대폰이라면 Termius 앱에 같은 정보를 넣으면 됩니다. 개인 키 파일은 누구에게도 보내지 마세요.

## 3. 설치 한 번

```bash
curl -fsSL https://raw.githubusercontent.com/sunbmj4119-cmd/rich/main/deploy/oracle/setup.sh -o setup.sh
less setup.sh     # 내용 확인 (q 로 나가기)
bash setup.sh
```

휴대폰에서 붙여넣기가 안 되면 짧은 두 줄을 직접 입력해도 됩니다 (키보드를 영어로):

```bash
git clone https://github.com/sunbmj4119-cmd/rich
bash rich/deploy/oracle/setup.sh
```

설치가 하는 일: git·Python 설치 → (메모리 2GB 미만이고 스왑이 없으면) 스왑 2GB → 코드 받기 →
GitHub 쓰기용 배포 키 만들기 → 비밀 키 파일 `~/.rich/secrets.env` 만들기 → 점검 결과 출력.
여러 번 실행해도 안전합니다. 서버의 시간대나 이미 돌고 있는 다른 프로그램은 건드리지 않습니다(설치 위치는 `~/rich`, `~/.rich` 뿐).

## 4. GitHub 쓰기 권한 — 둘 중 하나

**휴대폰(Cloud Shell)이라 화면 글자를 복사하기 어려우면 → 토큰 방식**
GitHub(브라우저) → 프로필 → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token
→ Repository access: Only select repositories → `rich` → Repository permissions → **Contents: Read and write** → Generate → **Copy**.
서버에서 `bash ~/rich/deploy/oracle/setup.sh --token` → 키보드의 클립보드로 붙여넣고 Enter (화면에 안 보임).
토큰에는 만료일이 있어, 만료되면 같은 방법으로 새로 넣습니다.

**PC라면 → 배포 키 방식 (만료 없음)**

설치 화면에 나온 `ssh-ed25519 …` 한 줄(공개 키)을 복사합니다.
GitHub → 저장소 → Settings → Deploy keys → Add deploy key → Title `oracle` → Key에 붙여넣기 →
**Allow write access 체크** → Add key. 그다음 `bash ~/rich/deploy/oracle/setup.sh --check` 에서 `GitHub 쓰기: 가능`이면 됩니다.

## 5. 토스증권

- WTS → 설정 → Open API → 키 발급 (client_id, client_secret)
- 같은 화면 허용 IP 관리에 점검 결과의 **바깥으로 나가는 IP** 등록
- 키는 서버 파일에만: `bash ~/rich/deploy/oracle/setup.sh --keys` → 물어보는 항목마다 붙여넣고 Enter (화면에 안 보임)
  (또는 `nano ~/.rich/secrets.env` 로 직접 편집 — 저장 Ctrl+O, Enter, 나가기 Ctrl+X)

## 6. 토스 자료 수집 (지금 켜도 안전 — 조회만)

```bash
bash ~/rich/deploy/oracle/setup.sh --collect
```

- 종목별 투자자별·프로그램매매·공매도·신용·대차 일별 이력과 KOSPI 투자자별 매매대금, KODEX 200·인버스 일봉을 받습니다.
- 처음 한 번은 2018년까지 거슬러 받느라 **1시간 안팎** 걸립니다. 뒤에서 돌기 때문에 앱을 닫아도 됩니다.
  진행 보기: `tail -f ~/.rich/logs/collect_backfill.log` (그만 보기 Ctrl+C — 수집은 계속됨)
- 그 뒤로는 **한국 시간 평일 18:50**에 그날 것만 덧붙입니다 (cron, 2~3분).
- 원본은 서버의 `~/.rich/toss/`에만 둡니다 (저장소가 공개라 토스 자료를 그대로 올리지 않음).
  GitHub 에는 몇 종목·며칠치를 받았는지(`data/toss/collect_status.json`)와 예측력 통계(`data/toss/lab.json`)만 올라갑니다.
- 주문·잔고·계좌 API는 쓰지 않습니다. 디스크는 1년에 수십 MB 정도입니다.

## 7. 아직 하지 않을 것

`bash setup.sh --cron` (한국 시간 평일 08:40 자동 실행 — 서버 시간대와 무관)은 **토스 연결 코드가 들어간 뒤에** 켭니다.
지금 켜도 설정이 모의 계좌라 아무 주문도 보내지 않고 끝나지만, 순서를 지키는 편이 안전합니다.

## 주의

- Oracle은 7일 동안 CPU·네트워크·메모리 사용이 모두 낮은 무료 서버를 '유휴'로 보고 회수할 수 있습니다.
  하루 두 번 도는 작업은 여기에 걸릴 수 있습니다. 2차 자료에 따르면, 회수 경고 메일이 오면 Billing에서 Pay As You Go로 올리는 방법이
  알려져 있습니다(무료 한도 안의 자원은 계속 0원이지만, 한도를 넘는 자원을 만들면 과금되니 주의). 공식 문서로는 확인하지 못했습니다.
- 이 서버에는 GitHub 실행기(self-hosted runner)를 설치하지 않습니다. 저장소가 공개라 다른 사람의 코드가 서버에서 돌 위험이 있기 때문입니다.
  서버는 정해진 시각에 코드를 받아 주문만 보내고 결과만 올립니다.
