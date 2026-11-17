# 얼굴·눈·홍채 랜드마크 검출

현재 단계는 한 사람의 얼굴 랜드마크를 검출하여 실시간 영상에 표시하는 기능입니다. 동공 위치 계산, 5점 보정, 집중 구역 이탈 판정은 후속 단계입니다.

## 실행

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe main.py
```

- 초록색: 얼굴 랜드마크 점과 얼굴 윤곽
- 청록색: 양쪽 눈 윤곽
- 자홍색: 양쪽 홍채 테두리와 중심
- 얼굴이 없으면 `No face detected`를 표시하고 계속 실행합니다.
- 검출이 복구되면 현재 프레임의 점을 다시 표시합니다. 이전 점은 남기지 않습니다.
- 영상은 좌우 반전하지 않으며 좌우 눈 이름은 MediaPipe의 피사체 기준입니다.
- 영상 창에 포커스를 두고 Q 또는 q로 종료합니다.
- 검출 위치는 영상 안의 랜드마크입니다. 화면 응시 좌표가 아닙니다.

기존 웹캠만 확인:

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe main.py --camera-only
```

자동 종료 및 창 없는 검출 확인:

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe main.py --max-frames 90
.\.venv\Scripts\python.exe main.py --headless --max-frames 60
```

`--headless`에서도 실제 웹캠과 모델을 사용하지만 창이나 키 입력은 사용하지 않습니다. 이 모드에는 종료를 보장하는 `--max-frames`가 필요합니다. 종료 시 처리 프레임 수와 얼굴 검출/미검출 프레임 수를 출력하며 영상은 저장하지 않습니다.

## 팀원 모델 준비

모델 파일은 Git에서 제외됩니다. 팀원은 가상환경과 패키지를 준비한 뒤 아래 Google 공식 모델 번들을 한 번 다운로드합니다. 기존 파일이 있으면 먼저 해시를 확인하고 덮어쓰지 않습니다.

```powershell
cd C:\perfume
if (-not (Test-Path -LiteralPath .\models\face_landmarker.task)) {
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile .\models\face_landmarker.task -UseBasicParsing
}
Get-FileHash -LiteralPath .\models\face_landmarker.task -Algorithm SHA256
```

- 취득일: 2026-11-11
- 파일 크기: 3,758,596바이트
- SHA256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- 다운로드 URL의 `latest` 파일이 바뀌면 해시가 달라질 수 있으므로 위 해시와 비교합니다.
- 모델이 없거나 잘못된 경우 오류를 반환하며 실행 중 자동으로 다운로드하지 않습니다.
- 다른 위치의 모델은 `--model-path "모델파일절대경로"`로 지정할 수 있습니다.

## 구현 방식

`src/landmarks.py`의 `FaceLandmarkDetector`가 MediaPipe Tasks의 CPU/VIDEO 모드를 사용합니다. 모델을 한 번 생성하고 `with` 문 종료 시 해제합니다. 한 사람(`num_faces=1`)을 대상으로 하고, 표정 계수와 변환 행렬 출력은 끕니다.

OpenCV의 BGR 영상을 RGB로 변환하여 `detect_for_video`에 전달합니다. 카메라 콜백은 단조 시계 기반으로 엄격히 증가하는 밀리초 타임스탬프를 전달합니다. 결과의 정규화된 x/y를 프레임 크기에 맞춰 그리며 원본 프레임은 수정하지 않습니다. 홍채 중심은 468번과 473번입니다. 얼굴 유무 변경은 정상 상태이며 추론 오류는 별도 오류로 처리합니다.

## 검증

```powershell
cd C:\perfume
.\.venv\Scripts\python.exe -m pytest -q
```

자동 테스트는 색상 변환, 타임스탬프, 입력 검증, 얼굴 미검출·복귀, 홍채 중심 표시, 모델·카메라 해제와 CLI 오류를 확인합니다. 모의 객체 테스트는 실제 모델 품질이나 웹캠 화면 검증을 대체하지 않습니다. 실제 모델과 카메라 검증 결과는 별도로 기록합니다.

참고: [Google 공식 Face Landmarker 안내](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker), [공식 Python 구현의 랜드마크 연결](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py).

## 이번 구현의 실제 검증 결과 (2026-11-11)

- 자동 테스트: 52개 통과. Ruff의 import/오류 검사와 포맷팅 적용.
- 실제 모델: 검출기 생성, 검은 영상 2프레임의 얼굴 미검출 반환과 모델 해제 확인.
- 실제 웹캠: `main.py --headless --max-frames 60`으로 장치 0의 640×480 영상을 처리했습니다. 33프레임에서 얼굴 랜드마크 478개를 검출했고 27프레임에서는 미검출이었습니다. 미검출과 재검출 전환이 여러 번 발생해도 60프레임을 끝까지 처리했습니다.
- 종료 시 `Camera resources released`와 `Landmarker released`를 확인했습니다. 드라이버 보고 FPS는 30이며 실제 추론 처리 FPS를 측정한 값은 아닙니다.
- 제한 환경의 카메라 접근은 실패했으나, 승인된 일반 장치 접근으로 재실행하여 위 결과를 확인했습니다.
- 영상·얼굴 이미지·랜드마크 좌표 파일은 저장하지 않았습니다. 로그에는 검출 여부와 프레임 수만 출력했습니다.
- 실제 미리보기 화면과 Q 키 입력의 최종 육안 확인은 별도입니다. 자동 테스트는 그리기와 Q/q 종료 경로를 확인합니다.

추가 표시 모드 확인: `main.py --max-frames 60`으로 실제 OpenCV 창에서 60프레임을 처리하고 자동 종료했습니다. 이 실행에서는 얼굴이 검출되지 않았으며 미검출 상태로 종료까지 진행했고 카메라와 모델 해제를 확인했습니다. 얼굴 위 오버레이의 최종 육안 확인 및 실제 Q 키 입력은 별도로 남아 있습니다.
