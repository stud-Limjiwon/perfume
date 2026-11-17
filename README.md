# 웹캠 시선 추적 기반 온라인 시험·면접 부정행위 의심 탐지

## 프로젝트 소개

일반 노트북 웹캠으로 응시자의 눈과 동공을 찾아 동공 위치를 추적합니다. 시험 시작 전 화면 5개 점을 보며 응시자의 **집중 구역**(화면을 볼 때의 동공 위치 범위)을 정하고, 시험 중 이 구역을 1초보다 오래 벗어나면 이탈 1회로 기록합니다. 이탈 횟수·누적 시간이 기준을 넘으면 **부정행위 의심**으로 표시합니다.

컨닝을 자동으로 확정하지 않습니다. 이탈 시각·지속 시간 기록(CSV)과 판정 기준을 함께 남기고 CSV의 해시값으로 기록이 바뀌지 않았음을 확인할 수 있게 하여, 감독자가 근거를 보고 최종 판단하도록 돕습니다. 판정 규칙은 [판정 규칙 문서](docs/detection_rules.md)에 있습니다.

- 개발 인원: 3명
- 개발 기간: 5주
- 개발 기준: Windows 10/11, VS Code PowerShell, Python 3.12 64비트
- 프로젝트 루트: `C:\perfume`

## 현재 개발 상태

현재는 **개발 환경 구축을 마치고 얼굴·눈·홍채 랜드마크 검출을 구현한 단계**입니다. 기본 실행은 랜드마크 표시이며 `--camera-only`로 웹캠만 실행할 수 있습니다.

| 구분 | 상태 |
| --- | --- |
| 폴더 구조·가상환경·패키지 설치 | 완료 |
| VS Code 인터프리터·디버깅·포맷팅 설정 | 작성 완료 |
| 웹캠 입력·영상 표시·Q 종료·리소스 해제 코드 | 구현 완료 |
| 패키지·프로젝트 import·검출·종료 처리 자동 테스트 | 53개 통과 (2026-11-17) |
| 실제 웹캠 장치 열기·640×480 프레임 수신 | 확인 완료 |
| 실제 웹캠 랜드마크 추론·미검출 후 복귀 | 60프레임 처리, 33프레임에서 478개 랜드마크 검출 |
| 실제 미리보기 영상 내용·Q 키 종료 | 최종 수동 확인 필요 |
| 얼굴·눈·홍채 랜드마크 검출·표시 | 구현. [실행 및 모델 준비 안내](docs/landmarks.md) |
| 이탈 판정 설정값(초안) | `config/settings.py`에 작성. 직접 촬영 데이터로 조정 예정 |
| 동공 위치·5점 보정·이탈 판정·기록·경고 표시 | 모듈 구조와 TODO만 작성 |

영상과 데이터는 저장하지 않습니다. Google 공식 Face Landmarker 모델 파일을 준비하여 랜드마크 검출에 사용합니다. 데이터셋 다운로드, 눈 특징 계산, 보정 데이터 수집과 모델 학습은 아직 진행하지 않았습니다. 초기 환경 검증은 [환경 구축 보고서](docs/setup_report.md), 현재 검출 기능은 [랜드마크 안내](docs/landmarks.md)를 참고하세요.

## 기술 구성

| 라이브러리 | 용도 | 초기 설치 버전 |
| --- | --- | --- |
| OpenCV (`opencv-python`) | 웹캠 입력 및 영상 표시 | 5.0.0.93 |
| MediaPipe | 얼굴·눈·홍채 검출 (OpenCV 방식 검출과 비교할 기준) | 1.2.2 |
| NumPy | 동공 위치 계산 | 2.5.4 |
| Pandas | 이탈 기록(CSV) 정리 | 3.1.1 |
| PyTorch | [도전 목표] 화면 좌표 추정 | 2.15.0 |
| Scikit-learn | [도전 목표] 비교 모델 및 평가 | 1.9.1 |
| Matplotlib | 이탈 구간 타임라인·결과 요약 그래프 | 3.11.2 |
| PyArrow | 특징 데이터의 Parquet 저장·읽기 | 26.0.1 |
| Pytest | 환경 및 종료 처리 검증 | 9.2.0 |

전체 설치 버전은 [requirements-lock.txt](requirements-lock.txt)에 고정되어 있습니다. 초기 검증 환경은 Python 3.12.10, Windows 64비트입니다.

현재 MediaPipe의 의존성으로 `opencv-contrib-python==5.0.0.93`도 함께 설치되어 두 OpenCV 배포판이 `cv2`를 공유합니다. import와 프레임 수신은 확인했지만, `pip check`만으로 이 파일 중복을 검출할 수는 없습니다. 팀 환경을 정리할 때 contrib 단일 배포판으로 통일하는 명세 변경을 검토해야 합니다. 현재 `requirements.txt`는 최초 요청한 패키지 목록을 유지합니다.

## 폴더 구조

```text
C:\perfume
├── .venv/                     # 로컬 Python 가상환경 (Git 제외)
├── .vscode/
│   ├── settings.json          # 인터프리터·포맷팅·테스트 설정
│   └── launch.json            # main.py 디버깅 설정
├── main.py                    # 웹캠 실행 진입점
├── requirements.txt           # 필수 패키지 목록
├── requirements-lock.txt      # 재현용 전체 패키지 버전
├── README.md
├── .gitignore
├── .env.example
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── landmarks.py
│   ├── features.py
│   ├── calibration.py
│   ├── gaze_prediction.py
│   ├── attention_detection.py
│   ├── session_log.py
│   └── visualization.py
├── training/
│   ├── __init__.py
│   ├── baseline.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── raw/
│   ├── labels/
│   ├── features/
│   └── meta/
├── models/
├── notebooks/
├── tests/
│   ├── __init__.py
│   ├── test_imports.py
│   ├── test_camera.py
│   ├── test_landmarks.py
│   └── test_main.py
├── logs/
└── docs/
    ├── setup_report.md
    ├── landmarks.md
    └── detection_rules.md
```

빈 폴더는 `.gitkeep`으로 구조를 보존합니다. 가상환경은 팀원마다 로컬에서 생성합니다.

## 설치 방법

### 1. Python 확인과 가상환경 생성

프로젝트 파일을 `C:\perfume`에 준비한 뒤 아래 명령을 실행합니다. 해당 경로 내부에 추가 프로젝트 폴더를 만들지 않습니다. 이미 `.venv`가 있다면 재생성하지 않습니다.

```powershell
cd C:\perfume
py -0p
py -3.12 --version
py -3.12 -m venv .venv
```

Python 3.12가 설치되어 있는데 런처에서 찾지 못한다면, 설치 경로와 버전을 먼저 확인한 뒤 실행 파일을 직접 사용합니다. 아래는 초기 구축 PC의 설치 경로 형식입니다.

```powershell
cd C:\perfume
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
```

Python 3.12가 없다면 먼저 3.12 64비트를 설치합니다. 다른 버전으로 임의 대체하지 않습니다.

### 2. 가상환경 활성화

```powershell
cd C:\perfume
.\.venv\Scripts\Activate.ps1
```

PowerShell 실행 정책으로 활성화가 차단되면 시스템 실행 정책을 변경하지 않고, 이후 예시처럼 `.\.venv\Scripts\python.exe`를 직접 사용합니다. 아래 명령은 활성화 여부와 관계없이 동일한 가상환경에서 실행됩니다.

### 3. 라이브러리 설치

팀원이 동일한 버전을 재현할 때는 고정된 버전 파일을 사용합니다.

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

필수 패키지 목록으로 환경을 새로 구성하거나 의존성을 변경할 때는 다음 명령을 사용합니다. 이 경우 lock 파일과 다른 버전이 설치될 수 있으므로 팀 검증 후 버전 목록을 갱신합니다.

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. VS Code 설정

1. VS Code에서 `C:\perfume` 폴더를 엽니다.
2. Python, Pylance, Python Debugger, Ruff 확장을 설치합니다.
3. `Python: Select Interpreter`에서 `C:\perfume\.venv\Scripts\python.exe`를 선택합니다.
4. 새 PowerShell 터미널을 열어 실행합니다.

프로젝트 설정에는 가상환경 자동 활성화, 저장 시 Ruff 포맷팅, 루트 기준 import 경로가 포함되어 있습니다. F5로 `Python: Webcam Preview` 디버깅 설정을 실행할 수 있습니다. `.env`는 선택 사항이며, 일반 `main.py` 실행에서는 자동으로 읽지 않습니다.

## 웹캠 및 랜드마크 실행

먼저 [랜드마크 안내의 모델 준비 절차](docs/landmarks.md)를 실행합니다. 모델은 Git에서 제외되므로 팀원마다 준비해야 합니다. 기본 실행은 얼굴·눈·홍채를 표시하며, 모델 없이 웹캠만 확인하려면 `main.py --camera-only`를 사용합니다.

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe main.py
```

- 기본 장치 인덱스: `0`
- 요청 해상도: `640×480`
- 목표 FPS: `30`
- 종료: 영상 창에 포커스를 두고 `Q` 또는 `q`

공통 설정은 [config/settings.py](config/settings.py)에 있습니다. 실제 해상도와 FPS는 장치 및 드라이버에 따라 다를 수 있습니다. 시작 시 첫 프레임 크기와 드라이버 보고 FPS를 터미널에 출력합니다. 드라이버 보고 FPS는 측정 처리 FPS와 다릅니다.

카메라 접근 또는 프레임 읽기에 실패하면 오류를 출력합니다. 정상 종료와 오류 처리 시 카메라 및 OpenCV 창을 해제하도록 구현했습니다. 카메라가 열리지 않으면 장치 연결, 다른 프로그램의 카메라 점유, Windows의 데스크톱 앱 카메라 권한을 확인합니다.

정해진 프레임 수를 표시한 뒤 자동으로 종료하려면 다음 명령을 사용합니다.

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe main.py --max-frames 90
```

## 테스트

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe -m pytest -q
```

- `test_imports.py`: 필수 라이브러리 및 프로젝트 모듈 import 검증
- `test_camera.py`: 카메라 열기·읽기 실패, Q/q 종료, 프레임 제한, GUI 오류 및 처리 콜백 검증
- `test_landmarks.py`: 검출 입력·얼굴 미검출/복귀·그리기·모델 수명 검증
- `test_main.py`: 실행 모드·모델 누락·오류 종료 검증

카메라 테스트는 모의 객체를 사용하므로 실제 웹캠을 켜거나 창을 띄우지 않습니다. 실제 영상 표시와 키 입력 확인은 `main.py`를 실행하여 별도로 검증합니다.

## 모듈별 기능

| 파일 | 담당 기능 | 현재 상태 |
| --- | --- | --- |
| `main.py` | 실행 인자와 오류·종료 코드 처리 | 구현 |
| `config/settings.py` | 카메라·검출·이탈 판정 설정 | 구현 (판정 값은 초안) |
| `src/camera.py` | 웹캠 입력·미리보기·종료·리소스 해제 | 구현 |
| `src/landmarks.py` | 얼굴·눈·홍채 검출·표시·모델 해제 | 구현 |
| `src/features.py` | 눈 안에서 동공 상대 위치 계산, OpenCV 방식 비교 | TODO |
| `src/calibration.py` | 5점 보정으로 집중 구역 설정 | TODO |
| `src/attention_detection.py` | 집중 구역·자리 이탈 판정, 부정행위 의심 표시 | TODO |
| `src/session_log.py` | 이탈 기록 CSV와 SHA256 해시값 저장 | TODO |
| `src/visualization.py` | 실시간 경고·FPS 표시, 종료 후 결과 요약 | TODO |
| `src/gaze_prediction.py` | [도전 목표] 화면 좌표 추정 | TODO |
| `training/*.py` | [도전 목표] 좌표 추정 모델 학습·평가 | TODO |

## 팀 협업

### 역할 분담 초안

실제 담당자는 팀에서 합의하여 지정합니다.

| 담당 | 주요 범위 |
| --- | --- |
| 팀원 A | 웹캠 입력·눈·동공 검출(OpenCV/MediaPipe 비교)·동공 위치 계산 |
| 팀원 B | 5점 보정·집중 구역·이탈 판정 기준 조정 |
| 팀원 C | 이탈 기록·해시값·경고 표시·FPS 측정·통합 테스트·문서 |

동공 위치 단위, 이탈 기록 CSV 컬럼, 응시자·세션 구분, 모듈 입출력 형식을 `docs/`에 먼저 합의하여 기록합니다.

### 팀원이 만든 GitHub 저장소에 참여하기

1. 저장소를 만든 팀원이 다른 팀원들을 collaborator로 초대합니다.
2. 초대를 수락하고 저장소 URL과 기본 브랜치를 확인합니다.
3. 초기 프로젝트를 올릴 담당자 한 명을 정합니다.
4. 초기 업로드 이후에는 기능별 브랜치에서 작업하고 PR로 검토합니다.

공유 저장소는 [stud-limjiwon/perfume](https://github.com/stud-limjiwon/perfume)이며 기본 브랜치는 `main`입니다. 로컬 프로젝트의 `origin`은 이 저장소에 연결합니다. 아래 명령은 **팀 저장소가 비어 있고 이 프로젝트를 최초 업로드할 경우**의 예시입니다. `OWNER`와 `REPOSITORY`는 팀원이 전달한 실제 값으로 바꿉니다.

```powershell
cd C:\perfume
git status
git remote -v
git remote add origin https://github.com/OWNER/REPOSITORY.git
git add .
git diff --cached --stat
git commit -m "초기 개발 환경과 웹캠 테스트 구성"
git branch -M main
git push -u origin main
```

`origin`이 이미 있다면 주소를 먼저 확인합니다. 원격 저장소에 README 등 기존 커밋이 있다면 위 최초 업로드 절차를 바로 적용하지 않고, 원격 이력을 확인해 팀과 통합 방법을 정합니다.

초기 업로드 이후 기능 브랜치 작업 예시:

```powershell
cd C:\perfume
git switch -c feature/landmarks
# 기능을 수정하고 테스트한 뒤 변경 파일을 확인합니다.
git status
git add src/landmarks.py
git commit -m "랜드마크 검출 모듈 구현"
git push -u origin feature/landmarks
```

GitHub에서 `main`을 대상으로 PR을 만들고 다른 팀원 1명 이상이 검토하는 방식을 권장합니다. `.venv`, Python 캐시, `.env`, 원본·라벨·특징 데이터, 모델과 로그는 `.gitignore`로 제외하며 해당 폴더의 `.gitkeep`은 추적합니다.

## 5주 개발 일정

| 주차 | 개발 목표 | 주요 산출물 |
| --- | --- | --- |
| 1주 | 환경 통일·웹캠 확인·모듈 인터페이스 합의 | 기본 실행 구조, 버전 고정, 데이터 규격 |
| 2주 | 눈·동공 검출, 동공 위치 계산, 5점 보정 | 동공 위치 추적, 집중 구역 설정 |
| 3주 | 이탈 판정·경고, 이탈 기록 CSV·해시값 | 기본 판정 흐름 완성 |
| 4주 | 직접 촬영 데이터로 판정 기준 조정, FPS 측정 | 기준값 확정, 정확도·속도 결과 |
| 5주 | 안정화, (시간이 남으면) 도전 목표, 발표 준비 | 최종 데모, 실행 안내 |

## 관련 문서

- [판정 규칙](docs/detection_rules.md)
- [환경 구축 및 검증 보고서](docs/setup_report.md)
- [필수 패키지 목록](requirements.txt)
- [고정 패키지 버전](requirements-lock.txt)
