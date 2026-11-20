"""실시간 FPS·집중 구역 상태 표시와 시험 종료 후 결과 요약을 담당할 모듈.

TODO: 종료 후 이탈 구간 타임라인과 요약(횟수, 누적 시간, 의심 여부)을 그린다.
"""

from dataclasses import dataclass

import cv2
import numpy as np

NORMAL_COLOR = (0, 200, 0)
DEPARTURE_COLOR = (0, 0, 255)
PENDING_COLOR = (200, 200, 200)
TEXT_COLOR = (255, 255, 255)
BAR_COLOR = (20, 20, 20)
BAR_HEIGHT = 30


@dataclass(frozen=True)
class AttentionStatus:
    """포렌식 판정 모듈이 화면 표시를 위해 넘겨주는 현재 상태."""

    departed: bool
    departure_count: int


def draw_status(
    frame_bgr: np.ndarray,
    *,
    fps: float,
    process_ms: float,
    attention: AttentionStatus | None = None,
) -> np.ndarray:
    """원본을 보존하며 아래쪽 막대에 실제 처리 FPS와 집중 구역 상태를 그린다.

    attention이 None이면 판정 모듈이 아직 연결되지 않은 상태로 표시한다.
    """
    output = frame_bgr.copy()
    height, width = output.shape[:2]
    top = max(height - BAR_HEIGHT, 0)
    cv2.rectangle(output, (0, top), (width - 1, height - 1), BAR_COLOR, -1)
    baseline = height - 10
    cv2.putText(
        output,
        f"FPS {fps:5.1f} | process {process_ms:5.1f} ms",
        (10, baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    if attention is None:
        label, color = "Status: --", PENDING_COLOR
    elif attention.departed:
        label = f"DEPARTURE | count {attention.departure_count}"
        color = DEPARTURE_COLOR
    else:
        label = f"Normal | count {attention.departure_count}"
        color = NORMAL_COLOR
    (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        output,
        label,
        (max(width - text_width - 10, 0), baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    if attention is not None and attention.departed:
        cv2.rectangle(output, (0, 0), (width - 1, height - 1), DEPARTURE_COLOR, 4)
    return output
