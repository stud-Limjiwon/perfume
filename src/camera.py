"""OpenCV 웹캠 입력, 선택적 프레임 처리와 카메라/GUI 리소스 정리."""

from collections.abc import Callable
from time import monotonic_ns

import cv2
import numpy as np

from config.settings import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    TARGET_FPS,
    WINDOW_NAME,
)

FrameProcessor = Callable[[np.ndarray, int], np.ndarray]


def run_preview(
    *,
    max_frames: int | None = None,
    frame_processor: FrameProcessor | None = None,
    display: bool = True,
) -> None:
    """장치 0을 처리·표시한다. Q 또는 프레임 제한으로 종료한다.

    GUI 없는 검증은 display=False와 유한한 max_frames를 함께 사용한다.
    처리 콜백의 시간은 monotonic clock 기반이며 밀리초마다 엄격히 증가한다.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    if not display and max_frames is None:
        raise ValueError("headless 실행에는 max_frames가 필요합니다.")

    capture = None
    frame_count = 0
    previous_timestamp = -1
    started_ns = monotonic_ns()
    try:
        capture = cv2.VideoCapture(CAMERA_INDEX)
        if not capture.isOpened():
            raise RuntimeError(
                "카메라 인덱스 0에 접근할 수 없습니다. 연결 상태, 다른 앱의 사용 여부, "
                "Windows 카메라 권한을 확인하세요."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        print(
            f"Camera opened: index={CAMERA_INDEX}, "
            f"requested={FRAME_WIDTH}x{FRAME_HEIGHT}@{TARGET_FPS} FPS; "
            f"display={display}. Press Q in the preview to quit.",
            flush=True,
        )
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("카메라에서 프레임을 읽지 못했습니다.")
            frame_count += 1
            if frame_count == 1:
                height, width = frame.shape[:2]
                print(
                    f"First frame received: {width}x{height}; "
                    f"driver-reported FPS={capture.get(cv2.CAP_PROP_FPS):.2f}",
                    flush=True,
                )
            timestamp_ms = max(
                (monotonic_ns() - started_ns) // 1_000_000, previous_timestamp + 1
            )
            previous_timestamp = timestamp_ms
            if frame_processor is not None:
                frame = frame_processor(frame, timestamp_ms)
            if display:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    print("Q key received.", flush=True)
                    break
            if max_frames is not None and frame_count >= max_frames:
                print(f"Frame limit reached: {frame_count}", flush=True)
                break
    finally:
        try:
            if capture is not None:
                capture.release()
        finally:
            if display:
                cv2.destroyAllWindows()
        print(f"Camera resources released. Frames processed: {frame_count}", flush=True)
