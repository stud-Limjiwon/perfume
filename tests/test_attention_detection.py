"""가짜 동공 위치로 이탈 판정 규칙을 검증한다. 카메라는 쓰지 않는다."""

import pytest

from src.attention_detection import (
    DepartureDetector,
    DepartureKind,
    DetectionCriteria,
    FocusZone,
    PupilSample,
)

ZONE = FocusZone(0.3, 0.7, 0.3, 0.7)
INSIDE = (0.5, 0.5)
OUTSIDE = (0.9, 0.5)


def run(frames, criteria=None):
    """frames: (시각 ms, 상태) 목록. 상태는 'in', 'out', 'noface', 'blink'."""
    detector = DepartureDetector(ZONE, criteria)
    for timestamp, state in frames:
        if state == "noface":
            sample = PupilSample(timestamp, False)
        elif state == "blink":
            sample = PupilSample(timestamp, True)
        else:
            sample = PupilSample(timestamp, True, *(INSIDE if state == "in" else OUTSIDE))
        detector.update(sample)
    return detector


def test_short_departure_is_not_counted():
    detector = run([(0, "in"), (100, "out"), (600, "in")])
    assert detector.events == ()


def test_exactly_threshold_is_not_counted():
    detector = run([(0, "in"), (100, "out"), (1100, "in")])
    assert detector.events == ()


def test_departure_over_one_second_counts_full_duration():
    detector = run([(0, "in"), (100, "out"), (1600, "in")])
    (event,) = detector.events
    assert event.kind is DepartureKind.GAZE
    assert (event.start_ms, event.end_ms, event.duration_ms) == (100, 1600, 1500)


def test_missing_face_is_seat_departure():
    detector = run([(0, "in"), (200, "noface"), (1700, "in")])
    (event,) = detector.events
    assert event.kind is DepartureKind.SEAT
    assert event.duration_ms == 1500


def test_blink_keeps_previous_state():
    inside = run([(0, "in"), (100, "blink"), (1500, "blink"), (1600, "in")])
    assert inside.events == ()
    outside = run([(0, "out"), (500, "blink"), (1000, "blink"), (1600, "in")])
    assert [e.duration_ms for e in outside.events] == [1600]


def test_gaze_then_seat_are_separate_events():
    detector = run([(0, "out"), (1500, "noface"), (3000, "in")])
    kinds = [(e.kind, e.duration_ms) for e in detector.events]
    assert kinds == [(DepartureKind.GAZE, 1500), (DepartureKind.SEAT, 1500)]


def test_update_returns_finished_event():
    detector = DepartureDetector(ZONE)
    assert detector.update(PupilSample(0, True, *OUTSIDE)) is None
    event = detector.update(PupilSample(2000, True, *INSIDE))
    assert event is not None and event.duration_ms == 2000


def test_finish_closes_open_departure():
    detector = run([(0, "in"), (1000, "out"), (2000, "out")])
    event = detector.finish(3000)
    assert event is not None and event.duration_ms == 2000
    with pytest.raises(RuntimeError):
        detector.update(PupilSample(4000, True, *INSIDE))


def test_summary_includes_ongoing_departure_after_threshold():
    detector = run([(0, "in"), (1000, "out")])
    assert detector.summary(1900).gaze.count == 0
    ongoing = detector.summary(2500)
    assert ongoing.gaze.count == 1
    assert ongoing.gaze.total_ms == 1500
    assert ongoing.current_kind is DepartureKind.GAZE
    assert ongoing.current_out_ms == 1500


def test_suspicious_by_count_and_time():
    criteria = DetectionCriteria(1000, 1000, 2, 10_000)
    frames = []
    for i in range(2):
        start = i * 10_000
        frames += [(start, "out"), (start + 1500, "in")]
    by_count = run(frames, criteria).summary()
    assert by_count.gaze.count == 2 and by_count.suspicious
    by_time = run([(0, "noface"), (11_000, "in")], criteria).summary()
    assert by_time.seat.count == 1 and by_time.suspicious
    assert by_time.seat.time_ratio == 1.0
    assert by_time.gaze.count_ratio == 0.0


def test_time_must_not_go_backwards():
    detector = run([(1000, "in")])
    with pytest.raises(ValueError):
        detector.update(PupilSample(500, True, *INSIDE))


def test_invalid_zone_and_criteria():
    with pytest.raises(ValueError):
        FocusZone(0.7, 0.3, 0.3, 0.7)
    with pytest.raises(ValueError):
        DetectionCriteria(departure_ms=0)
