"""카메라 및 MediaPipe Face Landmarker 공통 설정."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
WINDOW_NAME = "Webcam Preview - Press Q to quit"
FACE_LANDMARKER_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
