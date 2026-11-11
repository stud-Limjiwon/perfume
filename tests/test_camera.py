"""하드웨어 없이 카메라 오류 시 리소스 해제를 검증한다."""

from unittest.mock import Mock

import pytest

from src import camera


@pytest.mark.parametrize("opens", [False, True])
def test_release_on_camera_failure(monkeypatch, opens: bool) -> None:
    """열기/프레임 읽기 실패 모두 release와 GUI 정리를 수행해야 한다."""
    capture = Mock()
    capture.isOpened.return_value = opens
    capture.read.return_value = (False, None)
    cleanup = Mock()
    monkeypatch.setattr(camera.cv2, "VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr(camera.cv2, "destroyAllWindows", cleanup)
    with pytest.raises(RuntimeError):
        camera.run_preview()
    capture.release.assert_called_once_with()
    cleanup.assert_called_once_with()


@pytest.mark.parametrize("quit_key", [ord("q"), ord("Q"), -1])
def test_normal_exit_releases_resources(monkeypatch, quit_key: int) -> None:
    """Q/q 종료 또는 프레임 제한 종료에서 장치/창을 정리한다."""
    import numpy as np

    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    capture.get.return_value = 30.0
    show = Mock()
    cleanup = Mock()
    monkeypatch.setattr(camera.cv2, "VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr(camera.cv2, "imshow", show)
    monkeypatch.setattr(camera.cv2, "waitKey", Mock(return_value=quit_key))
    monkeypatch.setattr(camera.cv2, "destroyAllWindows", cleanup)
    camera.run_preview(max_frames=2)
    assert show.call_count == (2 if quit_key == -1 else 1)
    capture.release.assert_called_once_with()
    cleanup.assert_called_once_with()


def test_gui_error_releases_camera(monkeypatch) -> None:
    """영상 표시가 실패해도 열린 카메라를 해제한다."""
    import numpy as np

    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    capture.get.return_value = 30.0
    cleanup = Mock()
    monkeypatch.setattr(camera.cv2, "VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr(
        camera.cv2, "imshow", Mock(side_effect=camera.cv2.error("GUI unavailable"))
    )
    monkeypatch.setattr(camera.cv2, "destroyAllWindows", cleanup)
    with pytest.raises(camera.cv2.error):
        camera.run_preview()
    capture.release.assert_called_once_with()
    cleanup.assert_called_once_with()


def test_headless_callback_timestamps(monkeypatch):
    """동일 시계 값에서도 콜백 타임스탬프를 증가시키며 GUI를 호출하지 않는다."""
    import numpy as np

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)
    capture.get.return_value = 30.0
    show, wait, cleanup = Mock(), Mock(), Mock()
    monkeypatch.setattr(camera.cv2, "VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr(camera.cv2, "imshow", show)
    monkeypatch.setattr(camera.cv2, "waitKey", wait)
    monkeypatch.setattr(camera.cv2, "destroyAllWindows", cleanup)
    monkeypatch.setattr(camera, "monotonic_ns", lambda: 123)
    processor = Mock(return_value=frame)
    camera.run_preview(max_frames=3, frame_processor=processor, display=False)
    assert [call.args[1] for call in processor.call_args_list] == [0, 1, 2]
    show.assert_not_called()
    wait.assert_not_called()
    cleanup.assert_not_called()
    capture.release.assert_called_once_with()


def test_callback_failure_releases_camera(monkeypatch):
    """추론 콜백이 예외를 내도 실제 카메라 해제 경로를 실행한다."""
    import numpy as np

    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((2, 2, 3), dtype=np.uint8))
    capture.get.return_value = 30.0
    monkeypatch.setattr(camera.cv2, "VideoCapture", Mock(return_value=capture))
    processor = Mock(side_effect=RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        camera.run_preview(max_frames=1, frame_processor=processor, display=False)
    capture.release.assert_called_once_with()


def test_preview_displays_processed_frame(monkeypatch):
    """화면에는 원본 대신 콜백이 반환한 주석 영상을 표시한다."""
    import numpy as np

    original = np.zeros((2, 2, 3), dtype=np.uint8)
    annotated = np.full_like(original, 255)
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, original)
    capture.get.return_value = 30.0
    show = Mock()
    monkeypatch.setattr(camera.cv2, "VideoCapture", Mock(return_value=capture))
    monkeypatch.setattr(camera.cv2, "imshow", show)
    monkeypatch.setattr(camera.cv2, "waitKey", Mock(return_value=ord("q")))
    monkeypatch.setattr(camera.cv2, "destroyAllWindows", Mock())
    camera.run_preview(frame_processor=Mock(return_value=annotated))
    assert show.call_args.args[1] is annotated
    capture.release.assert_called_once_with()


def test_headless_requires_finite_run():
    with pytest.raises(ValueError, match="max_frames"):
        camera.run_preview(display=False)
