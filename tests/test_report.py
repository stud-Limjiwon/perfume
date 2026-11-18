"""화면 3 의심 정황 목록의 순서·색·제목을 검증한다."""

from datetime import datetime, timezone

from src.attention_detection import (
    DepartureDetector,
    DepartureEvent,
    DepartureKind,
    DetectionCriteria,
    FocusZone,
)
from src.report import RED, YELLOW, build_report, format_offset, main
from src.session_log import SessionRecorder, open_session

PASSWORD = "123456"


def make(tmp_path, events, criteria):
    recorder = SessionRecorder(
        student="A학생",
        password=PASSWORD,
        exam_seconds=3600,
        origin_ms=0,
        criteria=criteria,
        root=tmp_path,
        started_at=datetime(2026, 11, 18, tzinfo=timezone.utc),
        iterations=1000,
    )
    for event in events:
        recorder.record(event)
    recorder.close(DepartureDetector(FocusZone(0, 1, 0, 1)).summary())
    return recorder.directory


def gaze(start_s, seconds):
    return DepartureEvent(
        DepartureKind.GAZE, int(start_s * 1000), int((start_s + seconds) * 1000)
    )


def test_format_offset():
    assert format_offset(0) == "00시00분00초"
    assert format_offset(3_723_999) == "01시02분03초"


def test_red_by_duration_then_yellow_by_time(tmp_path):
    criteria = DetectionCriteria(1000, 1000, 3, 100_000)
    events = [gaze(10, 2), gaze(20, 5), gaze(30, 1.5)]
    session = open_session(make(tmp_path, events, criteria), PASSWORD)
    items = build_report(session)
    assert [i.color for i in items] == [RED, RED, RED, YELLOW]
    assert [i.duration_ms for i in items[:3]] == [5000, 2000, 1500]
    assert items[0].title == "A학생 | 의심 시간(시험 시작 후 00시00분20초)"
    assert "동공 이탈 누적 3회" in items[3].detail
    assert items[3].chain == session.events[2].chain


def test_report_uses_recorded_criteria(tmp_path):
    criteria = DetectionCriteria(1000, 1000, 50, 4000)
    session = open_session(make(tmp_path, [gaze(0, 2), gaze(10, 3)], criteria), PASSWORD)
    yellow = [i for i in build_report(session) if i.color == YELLOW]
    assert len(yellow) == 1
    assert "누적 5.0초 (기준 4초)" in yellow[0].detail


def test_cli_prints_report(tmp_path, monkeypatch, capsys):
    directory = make(tmp_path, [gaze(5, 2)], DetectionCriteria())
    monkeypatch.setattr("getpass.getpass", lambda prompt="": PASSWORD)
    assert main([str(directory), "--chain"]) == 0
    output = capsys.readouterr().out
    assert "무결성: 정상" in output and "[빨강]" in output and "체인값:" in output
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "wrong!")
    assert main([str(directory)]) == 2
