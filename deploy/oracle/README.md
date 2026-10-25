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

설치가 하는 일: git·Python 설치 → 시간대 서울 → (메모리 2GB 미만이면) 스왑 2GB → 코드 받기 →
GitHub 쓰기용 배포 키 만들기 → 비밀 키 파일 `~/.rich/secrets.env` 만들기 → 점검 결과 출력.
여러 번 실행해도 안전합니다.

## 4. 배포 키 등록 (GitHub 웹)

설치 화면에 나온 `ssh-ed25519 …` 한 줄(공개 키)을 복사합니다.
GitHub → 저장소 → Settings → Deploy keys → Add deploy key → Title `oracle` → Key에 붙여넣기 →
**Allow write access 체크** → Add key. 그다음 `bash ~/rich/deploy/oracle/setup.sh --check` 에서 `GitHub 쓰기: 가능`이면 됩니다.

## 5. 토스증권

- WTS → 설정 → Open API → 키 발급 (client_id, client_secret)
- 같은 화면 허용 IP 관리에 점검 결과의 **바깥으로 나가는 IP** 등록
- 키는 서버 파일에만: `nano ~/.rich/secrets.env` → `TOSS_CLIENT_ID=`, `TOSS_CLIENT_SECRET=` 뒤에 붙여 쓰고 저장 (Ctrl+O, Enter, Ctrl+X)

## 6. 아직 하지 않을 것

`bash setup.sh --cron` (평일 08:40 자동 실행)은 **토스 연결 코드가 들어간 뒤에** 켭니다.
지금 켜도 설정이 모의 계좌라 아무 주문도 보내지 않고 끝나지만, 순서를 지키는 편이 안전합니다.

## 주의

- Oracle은 7일 동안 CPU·네트워크·메모리 사용이 모두 낮은 무료 서버를 '유휴'로 보고 회수할 수 있습니다.
  하루 두 번 도는 작업은 여기에 걸릴 수 있습니다. 2차 자료에 따르면, 회수 경고 메일이 오면 Billing에서 Pay As You Go로 올리는 방법이
  알려져 있습니다(무료 한도 안의 자원은 계속 0원이지만, 한도를 넘는 자원을 만들면 과금되니 주의). 공식 문서로는 확인하지 못했습니다.
- 이 서버에는 GitHub 실행기(self-hosted runner)를 설치하지 않습니다. 저장소가 공개라 다른 사람의 코드가 서버에서 돌 위험이 있기 때문입니다.
  서버는 정해진 시각에 코드를 받아 주문만 보내고 결과만 올립니다.
