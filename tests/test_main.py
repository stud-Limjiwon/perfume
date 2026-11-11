"""CLI 선택과 모델 수명/오류 종료를 하드웨어 없이 검증한다."""

from unittest.mock import Mock

import pytest

import main


def test_camera_only_does_not_load_model(monkeypatch):
    detector = Mock()
    preview = Mock()
    monkeypatch.setattr(main, "FaceLandmarkDetector", detector)
    monkeypatch.setattr(main, "run_preview", preview)
    monkeypatch.setattr("sys.argv", ["main.py", "--camera-only", "--max-frames", "2"])
    assert main.main() == 0
    detector.assert_not_called()
    preview.assert_called_once_with(max_frames=2, frame_processor=None, display=True)


def test_preview_failure_closes_detector(monkeypatch):
    detector = Mock()
    detector.__enter__ = Mock(return_value=detector)
    detector.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(main, "FaceLandmarkDetector", Mock(return_value=detector))
    monkeypatch.setattr(
        main, "run_preview", Mock(side_effect=RuntimeError("camera unavailable"))
    )
    monkeypatch.setattr("sys.argv", ["main.py"])
    assert main.main() == 1
    detector.__exit__.assert_called_once()


def test_missing_model_returns_nonzero(monkeypatch):
    monkeypatch.setattr(
        main,
        "FaceLandmarkDetector",
        Mock(side_effect=FileNotFoundError("missing model")),
    )
    preview = Mock()
    monkeypatch.setattr(main, "run_preview", preview)
    monkeypatch.setattr("sys.argv", ["main.py"])
    assert main.main() == 1
    preview.assert_not_called()


def test_headless_requires_frame_limit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "--headless"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2
