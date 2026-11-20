"""FPS·집중 구역 상태 표시가 원본을 보존하며 상태별로 그려지는지 확인한다."""

import numpy as np

from src.visualization import DEPARTURE_COLOR, AttentionStatus, draw_status


def test_draw_status_preserves_original():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    output = draw_status(frame, fps=29.9, process_ms=12.3)
    assert not frame.any()
    assert output.shape == frame.shape
    assert output[-20:].any()


def test_departure_draws_red_border():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    normal = draw_status(
        frame, fps=30, process_ms=5, attention=AttentionStatus(False, 2)
    )
    departed = draw_status(
        frame, fps=30, process_ms=5, attention=AttentionStatus(True, 3)
    )
    assert tuple(departed[0, 0]) == DEPARTURE_COLOR
    assert not normal[0, 0].any()
