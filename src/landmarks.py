"""MediaPipe Tasks로 얼굴·눈·홍채를 검출하고 OpenCV 영상에 표시한다.

화면 응시 좌표 추정이나 눈 특징 계산은 수행하지 않는다.
좌우는 MediaPipe의 피사체 기준이며 영상은 좌우 반전하지 않는다.
"""

from math import isfinite
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

from config.settings import (
    FACE_LANDMARKER_MODEL,
    MIN_FACE_DETECTION_CONFIDENCE,
    MIN_FACE_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

FACE_COLOR = (0, 200, 0)
EYE_COLOR = (255, 255, 0)
IRIS_COLOR = (255, 0, 255)
# The ten iris points are 468..477; the centers are not in iris connections.
IRIS_CENTER_INDICES = (468, 473)


def _pixel(landmark, width: int, height: int) -> tuple[int, int] | None:
    """화면 안의 정규화 좌표를 픽셀로 바꾸고 화면 밖/비유한 점은 건너뛴다."""
    x, y = landmark.x, landmark.y
    if not (isfinite(x) and isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
        return None
    return min(round(x * width), width - 1), min(round(y * height), height - 1)


def draw_landmarks(
    frame_bgr: np.ndarray, result: vision.FaceLandmarkerResult
) -> np.ndarray:
    """원본을 보존하며 얼굴 점·윤곽, 눈 윤곽, 홍채 테두리와 중심을 그린다.

    얼굴 미검출은 정상 상태다. 매 프레임의 결과만 사용하므로 이전 점이 남지 않는다.
    """
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    groups = (
        (FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL, FACE_COLOR),
        (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE, EYE_COLOR),
        (FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE, EYE_COLOR),
        (FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS, IRIS_COLOR),
        (FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS, IRIS_COLOR),
    )
    for face in result.face_landmarks:
        pixels = [_pixel(point, width, height) for point in face]
        for point in pixels:
            if point is not None:
                cv2.circle(output, point, 1, FACE_COLOR, -1, cv2.LINE_AA)
        for connections, color in groups:
            for connection in connections:
                if max(connection.start, connection.end) >= len(pixels):
                    continue
                start, end = pixels[connection.start], pixels[connection.end]
                if start is not None and end is not None:
                    cv2.line(output, start, end, color, 1, cv2.LINE_AA)
        for index in IRIS_CENTER_INDICES:
            if index < len(pixels) and pixels[index] is not None:
                cv2.circle(output, pixels[index], 3, IRIS_COLOR, -1, cv2.LINE_AA)

    count = len(result.face_landmarks[0]) if result.face_landmarks else 0
    status = f"Face detected | landmarks: {count}" if count else "No face detected"
    color = FACE_COLOR if count else (0, 180, 255)
    cv2.rectangle(output, (0, 0), (width - 1, min(66, height - 1)), (20, 20, 20), -1)
    cv2.putText(
        output, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
    )
    cv2.putText(
        output,
        "Face: green | Eyes: cyan | Iris: magenta | Q: quit",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


class FaceLandmarkDetector:
    """CPU 기반 1인 VIDEO 검출기. with 문으로 생성/종료 수명을 관리한다."""

    def __init__(self, model_path: str | Path = FACE_LANDMARKER_MODEL) -> None:
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker 모델 파일이 없습니다: {path}\n"
                "docs/landmarks.md의 모델 준비 절차를 실행하거나 "
                "--camera-only로 웹캠만 확인하세요."
            )
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_buffer=path.read_bytes(),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        self._face_present: bool | None = None
        self.processed_frames = 0
        self.frames_with_face = 0
        self.last_landmark_count = 0

    def detect(
        self, frame_bgr: np.ndarray, timestamp_ms: int
    ) -> vision.FaceLandmarkerResult:
        """BGR uint8 영상에서 검출한다. 타임스탬프는 프레임마다 증가해야 한다."""
        if self._landmarker is None:
            raise RuntimeError("이미 종료한 Face Landmarker입니다.")
        if (
            frame_bgr.dtype != np.uint8
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.size == 0
        ):
            raise ValueError("입력은 비어 있지 않은 H×W×3 uint8 BGR 영상이어야 합니다.")
        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms는 0 이상이며 이전 프레임보다 커야 합니다.")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        return result

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """현재 프레임의 검출 결과를 그린다. 얼굴 유무 전환만 터미널에 알린다."""
        result = self.detect(frame_bgr, timestamp_ms)
        present = bool(result.face_landmarks)
        self.processed_frames += 1
        self.frames_with_face += int(present)
        self.last_landmark_count = len(result.face_landmarks[0]) if present else 0
        if present != self._face_present:
            message = (
                f"Face detected: landmarks={self.last_landmark_count}"
                if present
                else "No face detected"
            )
            print(message, flush=True)
            self._face_present = present
        return draw_landmarks(frame_bgr, result)

    def close(self) -> None:
        """네이티브 모델 리소스를 한 번만 해제한다."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        print(
            f"Landmarker released: frames={self.processed_frames}, "
            f"frames_with_face={self.frames_with_face}, "
            f"frames_without_face={self.processed_frames - self.frames_with_face}, "
            f"last_landmark_count={self.last_landmark_count}",
            flush=True,
        )
