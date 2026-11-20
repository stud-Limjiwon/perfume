"""카메라, MediaPipe Face Landmarker, 집중 구역 이탈 판정의 공통 설정."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 카메라
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
ZOOM = 1.0  # 1.0보다 크면 화면 중앙 고정 영역을 잘라 확대한다 (디지털 줌)
FPS_WINDOW_SECONDS = 1.0  # 실제 처리 FPS를 계산하는 최근 구간
FPS_LOG_INTERVAL_SECONDS = 1.0  # 터미널에 FPS를 출력하는 간격
WINDOW_NAME = "Webcam Preview - Press Q to quit"

# 얼굴·눈·홍채 검출 (MediaPipe: OpenCV 방식 검출 결과를 비교할 기준)
FACE_LANDMARKER_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# 집중 구역 보정과 이탈 판정 (초안 값: 직접 촬영 데이터로 조정 후 확정)
CALIBRATION_POINTS = 5  # 화면 중앙·네 모서리를 응시하며 동공 범위를 잰다
CALIBRATION_SECONDS_PER_POINT = 2.0
FOCUS_ZONE_MARGIN = 0.10  # 보정으로 잰 동공 범위에 더하는 여유 비율
DEPARTURE_SECONDS = 1.0  # 집중 구역을 이 시간보다 오래 벗어나면 이탈 1회
FACE_MISSING_SECONDS = 1.0  # 얼굴이 이 시간보다 오래 안 보이면 자리 이탈 1회
# 시험 1회에서 아래 기준 중 하나를 넘으면 "부정행위 의심"으로 표시한다.
# 최종 판단은 감독자가 기록(CSV)과 기준을 보고 내린다.
SUSPICIOUS_DEPARTURE_COUNT = 5
SUSPICIOUS_TOTAL_SECONDS = 30.0

# 세션 기록 (이탈 시각·지속 시간 CSV와 체인값. 영상은 저장하지 않는다)
SESSION_LOG_DIR = PROJECT_ROOT / "logs" / "sessions"
PASSWORD_MIN_LENGTH = 6  # 감독관 패스워드: 체인값 키 생성과 결과 조회 인증에 사용
PBKDF2_ITERATIONS = 200_000  # 패스워드를 키로 바꿀 때 반복 횟수 (높을수록 추측이 느려짐)
