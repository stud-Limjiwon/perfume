"""OpenCV 웹캠 입력, 디지털 줌, 실제 처리 FPS 측정과 카메라/GUI 리소스 정리."""

from collections import deque
from collections.abc import Callable
from time import monotonic_ns

import cv2
import numpy as np

from config.settings import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FPS_LOG_INTERVAL_SECONDS,
    FPS_WINDOW_SECONDS,
    FRAME_WIDTH,
    TARGET_FPS,
    WINDOW_NAME,
    ZOOM,
)
from src.visualization import draw_status

FrameProcessor = Callable[[np.ndarray, int], np.ndarray]


def crop_center(frame: np.ndarray, zoom: float) -> np.ndarray:
    """화면 중앙의 1/zoom 영역을 잘라낸다. 늘리지 않으므로 실제 픽셀만 남는다.

    영역은 고정이다. 얼굴을 따라 움직이면 집중 구역 보정 기준이 흔들린다.
    """
    if zoom < 1:
        raise ValueError("zoom은 1.0 이상이어야 합니다.")
    if zoom == 1:
        return frame
    height, width = frame.shape[:2]
    crop_height = max(round(height / zoom), 1)
    crop_width = max(round(width / zoom), 1)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return np.ascontiguousarray(
        frame[top : top + crop_height, left : left + crop_width]
    )


class FpsMeter:
    """최근 구간에서 끝난 프레임 간격으로 실제 처리 FPS를 계산한다.

    카메라 드라이버가 알려주는 FPS가 아니라 읽기·처리를 포함한 루프 속도다.
    process_ms는 프레임 처리 콜백(검출·그리기)에 걸린 평균 시간이다.
    """

    def __init__(self, window_seconds: float = FPS_WINDOW_SECONDS) -> None:
        self._window_ns = int(window_seconds * 1_000_000_000)
        self._samples: deque[tuple[int, int]] = deque()

    def update(self, now_ns: int, process_ns: int) -> None:
        self._samples.append((now_ns, process_ns))
        while now_ns - self._samples[0][0] > self._window_ns:
            self._samples.popleft()

    @property
    def fps(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        elapsed_ns = self._samples[-1][0] - self._samples[0][0]
        if elapsed_ns <= 0:
            return 0.0
        return (len(self._samples) - 1) * 1_000_000_000 / elapsed_ns

    @property
    def process_ms(self) -> float:
        if not self._samples:
            return 0.0
        return sum(sample[1] for sample in self._samples) / len(self._samples) / 1e6


def run_preview(
    *,
    max_frames: int | None = None,
    frame_processor: FrameProcessor | None = None,
    display: bool = True,
    camera_index: int = CAMERA_INDEX,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    zoom: float = ZOOM,
) -> None:
    """지정한 카메라를 처리·표시한다. Q 또는 프레임 제한으로 종료한다.

    GUI 없는 검증은 display=False와 유한한 max_frames를 함께 사용한다.
    처리 콜백의 시간은 monotonic clock 기반이며 밀리초마다 엄격히 증가한다.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be a positive integer")
    if not display and max_frames is None:
        raise ValueError("headless 실행에는 max_frames가 필요합니다.")
    if zoom < 1:
        raise ValueError("zoom은 1.0 이상이어야 합니다.")

    capture = None
    frame_count = 0
    previous_timestamp = -1
    started_ns = monotonic_ns()
    meter = FpsMeter()
    log_interval_ns = int(FPS_LOG_INTERVAL_SECONDS * 1_000_000_000)
    last_log_ns = started_ns
    first_frame_ns = last_frame_ns = None
    try:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(
                f"카메라 인덱스 {camera_index}에 접근할 수 없습니다. 연결 상태, "
                "다른 앱의 사용 여부, 운영체제의 카메라 권한을 확인하세요."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        print(
            f"Camera opened: index={camera_index}, "
            f"requested={width}x{height}@{TARGET_FPS} FPS, zoom={zoom:g}; "
            f"display={display}. Press Q in the preview to quit.",
            flush=True,
        )
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("카메라에서 프레임을 읽지 못했습니다.")
            frame_count += 1
            if frame_count == 1:
                received_height, received_width = frame.shape[:2]
                print(
                    f"First frame received: {received_width}x{received_height}; "
                    f"driver-reported FPS={capture.get(cv2.CAP_PROP_FPS):.2f}",
                    flush=True,
                )
            frame = crop_center(frame, zoom)
            if frame_count == 1 and zoom > 1:
                cropped_height, cropped_width = frame.shape[:2]
                print(
                    f"Zoom crop: {cropped_width}x{cropped_height}", flush=True
                )
            process_started_ns = monotonic_ns()
            timestamp_ms = max(
                (process_started_ns - started_ns) // 1_000_000, previous_timestamp + 1
            )
            previous_timestamp = timestamp_ms
            if frame_processor is not None:
                frame = frame_processor(frame, timestamp_ms)
            now_ns = monotonic_ns()
            meter.update(now_ns, now_ns - process_started_ns)
            if first_frame_ns is None:
                first_frame_ns = now_ns
            last_frame_ns = now_ns
            if now_ns - last_log_ns >= log_interval_ns:
                print(
                    f"FPS={meter.fps:.1f} process={meter.process_ms:.1f}ms",
                    flush=True,
                )
                last_log_ns = now_ns
            if display:
                frame = draw_status(
                    frame, fps=meter.fps, process_ms=meter.process_ms
                )
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
        if first_frame_ns is not None and last_frame_ns > first_frame_ns:
            # 카메라 열기 시간을 빼고 첫 프레임 처리 이후의 평균 속도만 잰다.
            average_fps = (
                (frame_count - 1) * 1_000_000_000 / (last_frame_ns - first_frame_ns)
            )
            print(f"Average FPS: {average_fps:.1f}", flush=True)
        print(f"Camera resources released. Frames processed: {frame_count}", flush=True)
