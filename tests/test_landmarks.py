"""네트워크/카메라 없이 랜드마크 색상 변환, 누락, 복귀와 수명 관리를 검증한다."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from mediapipe.tasks.python import vision

from src import landmarks


def result_with_face():
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    return vision.FaceLandmarkerResult([points], [], [])


@pytest.fixture
def detector_fixture(tmp_path, monkeypatch):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"mock model")
    engine = Mock()
    engine.detect_for_video.return_value = vision.FaceLandmarkerResult([], [], [])
    factory = Mock(return_value=engine)
    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", factory)
    return landmarks.FaceLandmarkDetector(model), engine, factory


def test_video_options_and_rgb_conversion(detector_fixture):
    detector, engine, factory = detector_fixture
    frame = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)
    with detector:
        detector.detect(frame, 0)
    image, timestamp = engine.detect_for_video.call_args.args
    np.testing.assert_array_equal(image.numpy_view(), frame[:, :, ::-1])
    assert timestamp == 0
    options = factory.call_args.args[0]
    assert options.running_mode == vision.RunningMode.VIDEO
    assert options.num_faces == 1
    assert not options.output_face_blendshapes
    engine.close.assert_called_once_with()


def test_face_disappears_and_returns_without_stale_overlay(detector_fixture):
    detector, engine, _ = detector_fixture
    face = result_with_face()
    empty = vision.FaceLandmarkerResult([], [], [])
    engine.detect_for_video.side_effect = [face, empty, face]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with detector:
        visible = detector.process_frame(frame, 0)
        missing = detector.process_frame(frame, 1)
        returned = detector.process_frame(frame, 2)
    assert visible[240, 320].any()
    assert not missing[240, 320].any()
    assert returned[240, 320].any()
    assert not frame.any(), "Drawing must preserve the input frame"
    assert detector.processed_frames == 3
    assert detector.frames_with_face == 2
    assert detector.last_landmark_count == 478


@pytest.mark.parametrize("timestamp", [-1, 10])
def test_timestamp_must_increase(detector_fixture, timestamp):
    detector, engine, _ = detector_fixture
    with detector:
        detector.detect(np.zeros((2, 2, 3), dtype=np.uint8), 10)
        with pytest.raises(ValueError, match="timestamp_ms"):
            detector.detect(np.zeros((2, 2, 3), dtype=np.uint8), timestamp)
    assert engine.detect_for_video.call_count == 1


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((0, 2, 3), dtype=np.uint8),
    ],
)
def test_invalid_frame(detector_fixture, frame):
    detector, engine, _ = detector_fixture
    with detector, pytest.raises(ValueError, match="BGR"):
        detector.detect(frame, 0)
    engine.detect_for_video.assert_not_called()


def test_close_is_idempotent_and_closed_detector_rejects_frames(detector_fixture):
    detector, engine, _ = detector_fixture
    detector.close()
    detector.close()
    engine.close.assert_called_once_with()
    with pytest.raises(RuntimeError):
        detector.detect(np.zeros((2, 2, 3), dtype=np.uint8), 0)


def test_inference_failure_closes_native_model(detector_fixture):
    detector, engine, _ = detector_fixture
    engine.detect_for_video.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError, match="inference failed"), detector:
        detector.process_frame(np.zeros((2, 2, 3), dtype=np.uint8), 0)
    engine.close.assert_called_once_with()


def test_missing_model_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/landmarks.md"):
        landmarks.FaceLandmarkDetector(tmp_path / "missing.task")


def test_out_of_frame_points_do_not_crash():
    result = result_with_face()
    result.face_landmarks[0][0].x = float("nan")
    result.face_landmarks[0][1].y = 1.5
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    output = landmarks.draw_landmarks(frame, result)
    assert output.shape == frame.shape
    assert not frame.any()


def test_both_iris_centers_are_highlighted():
    result = result_with_face()
    result.face_landmarks[0][468].x = 0.25
    result.face_landmarks[0][473].x = 0.75
    output = landmarks.draw_landmarks(np.zeros((480, 640, 3), dtype=np.uint8), result)
    np.testing.assert_array_equal(output[240, 160], landmarks.IRIS_COLOR)
    np.testing.assert_array_equal(output[240, 480], landmarks.IRIS_COLOR)
