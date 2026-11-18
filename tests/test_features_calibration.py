"""가짜 랜드마크로 동공 위치 계산과 5점 보정을 검증한다."""

from types import SimpleNamespace

import pytest

from src.attention_detection import PupilSample
from src.calibration import CalibrationError, Calibrator
from src.exam_app import format_clock, parse_duration
from src.features import (
    SUBJECT_LEFT_EYE,
    SUBJECT_RIGHT_EYE,
    pupil_position,
    pupil_sample,
)


def landmark(x, y):
    return SimpleNamespace(x=x, y=y)


def face(iris_ratio=(0.5, 0.5), opening=0.4, closed_eyes=()):
    """눈 가로 0.1, 세로 0.1*opening인 가짜 얼굴 (정규화 좌표)."""
    points = [landmark(0.5, 0.5) for _ in range(478)]
    for eye, left_x in ((SUBJECT_RIGHT_EYE, 0.3), (SUBJECT_LEFT_EYE, 0.6)):
        height = 0 if eye in closed_eyes else 0.1 * opening
        top_y = 0.4
        points[eye.left] = landmark(left_x, top_y + height / 2)
        points[eye.right] = landmark(left_x + 0.1, top_y + height / 2)
        points[eye.top] = landmark(left_x + 0.05, top_y)
        points[eye.bottom] = landmark(left_x + 0.05, top_y + height)
        points[eye.iris] = landmark(
            left_x + 0.1 * iris_ratio[0], top_y + height * iris_ratio[1]
        )
    return points


def test_pupil_position_ratio():
    x, y = pupil_position(face((0.25, 0.75)), 100, 100)
    assert x == pytest.approx(0.25)
    assert y == pytest.approx(0.75)


def test_closed_eyes_give_none_and_one_open_eye_is_used():
    both = (SUBJECT_RIGHT_EYE, SUBJECT_LEFT_EYE)
    assert pupil_position(face(closed_eyes=both), 100, 100) is None
    one = pupil_position(face((0.8, 0.5), closed_eyes=(SUBJECT_LEFT_EYE,)), 100, 100)
    assert one[0] == pytest.approx(0.8)


def test_pupil_sample_states():
    no_face = pupil_sample(SimpleNamespace(face_landmarks=[]), 5, 100, 100)
    assert no_face == PupilSample(5, False)
    closed = (SUBJECT_RIGHT_EYE, SUBJECT_LEFT_EYE)
    blink = pupil_sample(
        SimpleNamespace(face_landmarks=[face(closed_eyes=closed)]), 6, 100, 100
    )
    assert blink.face_detected and blink.x is None
    normal = pupil_sample(SimpleNamespace(face_landmarks=[face()]), 7, 100, 100)
    assert normal.x == pytest.approx(0.5)


POINTS = ((0.5, 0.5), (0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6))


def feed(calibrator, positions, step_ms=50, point_ms=2000, start_ms=0):
    t = start_ms
    for x, y in positions:
        for _ in range(point_ms // step_ms):
            calibrator.add(PupilSample(t, True, x, y))
            t += step_ms
    calibrator.add(PupilSample(t, True, *positions[-1]))  # 마지막 점을 끝내는 프레임
    return t


def test_calibration_zone_with_margin():
    calibrator = Calibrator(margin=0.1)
    feed(calibrator, POINTS)
    assert calibrator.done
    zone = calibrator.zone()
    assert (zone.x_min, zone.x_max) == pytest.approx((0.38, 0.62))
    assert (zone.y_min, zone.y_max) == pytest.approx((0.38, 0.62))


def test_point_without_valid_samples_is_retried():
    calibrator = Calibrator()
    t = 0
    for _ in range(40):  # 첫 점 2초 동안 얼굴 없음
        calibrator.add(PupilSample(t, False))
        t += 50
    calibrator.add(PupilSample(t, False))
    assert calibrator.index == 0 and calibrator.retries == 1
    feed(calibrator, POINTS, start_ms=t + 50)
    assert calibrator.done


def test_flat_calibration_is_rejected():
    calibrator = Calibrator()
    feed(calibrator, [(0.5, 0.5)] * 5)
    with pytest.raises(CalibrationError):
        calibrator.zone()
    with pytest.raises(CalibrationError):
        Calibrator().zone()


def test_duration_parsing_and_clock():
    assert parse_duration("01:02:03") == 3723
    for bad in ("00:00:00", "1:2", "00:61:00", "aa:00:00"):
        with pytest.raises(ValueError):
            parse_duration(bad)
    assert format_clock(299.2) == "00:05:00"
    assert format_clock(-3) == "00:00:00"
