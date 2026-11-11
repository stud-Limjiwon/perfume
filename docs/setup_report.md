# 초기 환경 구축 결과

> 이 문서는 초기 환경 구축 당시의 기록입니다. 이후 추가된 랜드마크 검출과 최신 검증 결과는 [랜드마크 안내](landmarks.md), 현재 저장소 정보는 [README](../README.md)를 참고하세요.

검증일: 2026-11-11 (Asia/Seoul). 프로젝트 루트: `C:\perfume`.

## 완료 범위

요청한 디렉터리를 `C:\perfume`에 직접 생성했습니다. 추가 최상위 프로젝트 폴더는 만들지 않았습니다. 최초 루트는 비어 있었고 기존 사용자 파일을 삭제하거나 덮어쓰지 않았습니다. VS Code의 `perfume` 창과 작업 영역 기록의 `file:///c%3A/perfume`을 확인했습니다.

Python **3.12.10 64비트**, 가상환경은 **`C:\perfume\.venv`**입니다. `py -3.12`는 `No suitable Python runtime found`로 실패했으나, 설치된 `C:\Users\이수민\AppData\Local\Programs\Python\Python312\python.exe`를 직접 확인해 사용했습니다. 다른 Python 버전으로 대체하지 않았습니다. 시스템 실행 정책을 변경하지 않고 가상환경 Python을 직접 호출했습니다.

## 설치 버전

| 패키지 | 버전 |
| --- | --- |
| opencv-python | 5.0.0.93 |
| mediapipe | 1.2.2 |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| torch | 2.15.0 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| pyarrow | 26.0.1 |
| pytest | 9.2.0 |

전체 43개 배포 패키지 버전은 `requirements-lock.txt`에 기록했습니다. lock의 각 버전이 실제 설치 메타데이터와 일치하는지도 확인했습니다. 설치 전 `pip install --dry-run --only-binary=:all:`로 Python 3.12용 Windows wheel과 의존성 해석을 확인했으며, 설치 후 `pip check`는 `No broken requirements found.`를 반환했습니다.

참고: [MediaPipe Python 환경](https://developers.google.com/edge/mediapipe/solutions/setup_python), [PyTorch 설치](https://docs.pytorch.org/get-started/locally/).

### 확인된 의존성 제한

요청한 `requirements.txt`의 9개 항목을 유지했습니다. MediaPipe의 의존성으로 `opencv-contrib-python==5.0.0.93`도 설치되어 `opencv-python`과 중복됩니다. 두 배포판은 `cv2` 파일을 공유하며 [OpenCV 배포 지침](https://pypi.org/project/opencv-python/#installation-and-usage)은 하나만 설치하도록 권장합니다. `pip check`는 이 파일 중복을 검출하지 않으므로 완전한 무충돌 환경으로 확정할 수 없습니다.

이번 import와 실제 프레임 수신은 통과했습니다. 장기 협업 환경에서는 요구 명세를 `opencv-contrib-python` 단일 배포판으로 통일하는 변경을 검토해야 합니다. 이번에는 요청된 패키지 목록을 임의로 변경하거나 의존성 검사를 우회하지 않았습니다.

## 검증 결과

| 항목 | 결과 |
| --- | --- |
| Python 3.12 가상환경 및 실행 파일 | 통과 |
| 필수 패키지 설치 및 pip check | 통과. OpenCV 배포판 중복 제한은 위에 별도 기록 |
| 라이브러리·프로젝트 import | 25개 통과 |
| Face Landmarker Tasks API import | 통과. 모델 다운로드·로딩·검출은 미실행 |
| PyTorch CPU 기본 연산 | 통과. 모델 학습은 미실행 |
| 카메라 열기/읽기 실패 시 정리 | 자동 테스트 2개 통과 |
| q/Q/프레임 제한 종료 및 GUI 오류 시 정리 | 자동 테스트 4개 통과. OpenCV 입출력은 모의 객체 사용 |
| 전체 pytest | 31개 통과, 실패 0, 오류 0, 건너뜀 0 |
| VS Code settings/launch JSON 및 인터프리터 | 통과 |
| 요청 폴더·파일 존재, lock 버전 일치 | 통과 |
| Git 로컬 저장소 및 gitkeep 예외 | 통과 |
| Git 원격 연결·커밋·푸시 | 미수행. 원격 및 추적 파일 없음 |

Python, Pylance, Python Debugger, Ruff 확장은 설치되어 있습니다. 새 터미널 자동 활성화와 저장 시 Ruff 포맷팅을 설정했습니다. 새 터미널의 실제 활성화 및 F5 디버거 실행은 별도 UI 조작으로 검증하지 않았습니다.

자동 검증 로그: `logs/pytest-results.xml`, `logs/environment-verification.json`. 패키지 설치/해석 로그: `logs/pip-install.log`, `logs/dependency-resolution.json`. 로그는 Git에서 제외됩니다.

## 실제 웹캠 테스트

`C:\perfume\.venv\Scripts\python.exe main.py`를 실제 실행했고 표준 출력에서 다음을 확인했습니다.

```text
Camera opened: index=0, requested=640x480@30 FPS. Focus the preview and press Q to quit.
First frame received: 640x480; driver-reported FPS=30.00
```

장치 0의 열기와 실제 640×480 프레임 수신, OpenCV 미리보기 창 생성을 확인했습니다. 30 FPS는 드라이버 보고값이며 측정 처리 FPS가 아닙니다. 영상은 저장하지 않았습니다.

화면 캡처에서 다른 창이 미리보기를 가렸고 미리보기 활성화 요청은 시간 초과되었습니다. 이후 사용자가 컴퓨터 사용을 위해 Computer Use를 중지하여 화면 내용 확인과 실제 Q 키 종료를 끝까지 검증하지 못했습니다. 따라서 **실제 미리보기/Q 종료 전체 성공으로 기록하지 않습니다**. 실행 세션을 중단했고 재개 후 남아 있는 Python 프로세스가 없음을 확인했습니다. 강제 세션 중단에서 finally 실행 여부는 확인되지 않았습니다. 정상/오류 종료의 리소스 해제는 자동 테스트로 검증했습니다.

나중에 수동으로 확인할 명령:

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe main.py
```

미리보기 영상이 표시되면 창에 포커스를 두고 Q를 눌러 `Q key received.`와 `Camera resources released.`를 확인합니다.

## 발생한 문제와 조치

- Python 런처가 3.12를 찾지 못해 설치된 3.12.10 실행 파일을 직접 사용했습니다.
- `C:\perfume`은 초기 쓰기 허용 범위 밖이므로 승인된 명령으로 해당 루트에서 파일 생성·설치·검증을 수행했습니다.
- MediaPipe가 OpenCV contrib를 추가로 요구하는 파일 중복 문제는 위 제한 사항에 기록했습니다. 요청한 패키지 목록은 유지했습니다.
- UI 제어 중지로 실제 Q 종료 검증은 미완료입니다. 재개 후 창 조작 없이 자동 테스트와 보고서를 마무리했습니다.
- PowerShell에서 Python 표준 입력으로 문서를 전달할 때 한글 인코딩이 손실된 것을 확인했고, UTF-8 직접 파일 저장으로 수정했습니다. 코드와 최초 README 본문은 정상 UTF-8입니다.

## 다음 단계

팀원 3명의 역할과 5주 개발 일정은 `README.md`에 있습니다. 다음 단계는 특징·화면 좌표·세션 규격 합의, Face Landmarker와 눈·홍채 특징, 9점 보정입니다. 이후 다항 회귀 비교 모델, MLP 학습·평가, 실시간 추론, ROI 이벤트와 시각화를 순차 개발합니다.

이번 단계에서는 데이터셋 다운로드, 홍채 검출 구현, 보정 데이터 수집, 모델 학습 및 ROI 분석을 수행하지 않았습니다.

## 생성한 폴더 및 파일 목록

`.venv` 패키지 내부, `.git` 관리 파일과 자동 생성 캐시를 제외한 목록입니다.

```text
.venv/  (Python 3.12 가상환경)
.git/  (로컬 Git 관리)
.env.example
.gitignore
.vscode/
.vscode/launch.json
.vscode/settings.json
config/
config/__init__.py
config/settings.py
data/
data/features/
data/features/.gitkeep
data/labels/
data/labels/.gitkeep
data/meta/
data/meta/.gitkeep
data/raw/
data/raw/.gitkeep
docs/
docs/setup_report.md
logs/
logs/.gitkeep
logs/dependency-resolution.json
logs/environment-verification.json
logs/pip-install.log
logs/pytest-results.xml
main.py
models/
models/.gitkeep
notebooks/
notebooks/.gitkeep
README.md
requirements.txt
requirements-lock.txt
src/
src/__init__.py
src/attention_detection.py
src/calibration.py
src/camera.py
src/features.py
src/gaze_prediction.py
src/landmarks.py
src/visualization.py
tests/
tests/__init__.py
tests/test_camera.py
tests/test_imports.py
training/
training/__init__.py
training/baseline.py
training/evaluate.py
training/model.py
training/train.py
```
