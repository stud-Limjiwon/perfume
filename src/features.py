"""눈 영역 안에서 동공(홍채 중심)의 상대 위치를 계산한다.

[임시 구현] MediaPipe 랜드마크로 계산한다. 뇌공학 담당이 OpenCV(Haar Cascade 눈 검출 +
동공 검출) 방식으로 바꾸거나 비교할 예정이며, pupil_sample()의 입출력만 유지하면 된다.

- 가로: 눈의 영상 왼쪽 끝 → 오른쪽 끝 방향으로 홍채 중심이 놓인 비율 (0~1)
- 세로: 윗눈꺼풀 → 아랫눈꺼풀 방향 비율 (0~1)
- 눈을 감았으면(세로/가로 비율이 작으면) 그 눈은 계산하지 않는다.
- 두 눈 중 계산된 눈의 평균을 쓴다. 영상은 좌우 반전하지 않은 원본 기준이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention_detection import PupilSample


@dataclass(frozen=True)
class EyeIndices:
    left: int  # 영상에서 왼쪽 눈꼬리
    right: int  # 영상에서 오른쪽 눈꼬리
    top: int
    bottom: int
    iris: int


# MediaPipe Face Landmarker 번호 (피사체 기준 오른눈 / 왼눈)
SUBJECT_RIGHT_EYE = EyeIndices(left=33, right=133, top=159, bottom=145, iris=468)
SUBJECT_LEFT_EYE = EyeIndices(left=362, right=263, top=386, bottom=374, iris=473)
MIN_EYE_OPENING = 0.12  # 눈 세로/가로가 이보다 작으면 감은 눈으로 본다


def _point(landmarks, index: int, width: int, height: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=float)


def _project(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    axis = end - start
    length = float(axis @ axis)
    if length == 0:
        raise ValueError("눈 기준점이 겹쳤습니다.")
    return float((point - start) @ axis / length)


def eye_ratio(
    landmarks, eye: EyeIndices, width: int, height: int
) -> tuple[float, float] | None:
    """한쪽 눈의 (가로, 세로) 비율. 감았거나 계산할 수 없으면 None."""
    if len(landmarks) <= max(eye.left, eye.right, eye.top, eye.bottom, eye.iris):
        return None
    try:
        left = _point(landmarks, eye.left, width, height)
        right = _point(landmarks, eye.right, width, height)
        top = _point(landmarks, eye.top, width, height)
        bottom = _point(landmarks, eye.bottom, width, height)
        iris = _point(landmarks, eye.iris, width, height)
        eye_width = float(np.linalg.norm(right - left))
        if eye_width == 0 or not np.isfinite(eye_width):
            return None
        if float(np.linalg.norm(bottom - top)) / eye_width < MIN_EYE_OPENING:
            return None
        x = _project(iris, left, right)
        y = _project(iris, top, bottom)
    except ValueError:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y


def pupil_position(
    landmarks, width: int, height: int
) -> tuple[float, float] | None:
    """두 눈 평균 동공 위치. 두 눈 모두 계산할 수 없으면 None(깜빡임 등)."""
    ratios = [
        r
        for eye in (SUBJECT_RIGHT_EYE, SUBJECT_LEFT_EYE)
        if (r := eye_ratio(landmarks, eye, width, height)) is not None
    ]
    if not ratios:
        return None
    xs, ys = zip(*ratios)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def pupil_sample(result, timestamp_ms: int, width: int, height: int) -> PupilSample:
    """Face Landmarker 결과를 판정 입력으로 바꾼다."""
    if not result.face_landmarks:
        return PupilSample(timestamp_ms, face_detected=False)
    position = pupil_position(result.face_landmarks[0], width, height)
    if position is None:
        return PupilSample(timestamp_ms, face_detected=True)
    return PupilSample(timestamp_ms, True, *position)
