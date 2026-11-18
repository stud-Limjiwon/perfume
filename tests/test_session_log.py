"""체인값 기록과 위·변조 검출을 검증한다. 임시 폴더만 사용한다."""

import csv
import json
from datetime import datetime, timezone

import pytest

from src.attention_detection import (
    DepartureDetector,
    DepartureEvent,
    DepartureKind,
    FocusZone,
)
from src.session_log import (
    EVENTS_FILE,
    EVENTS_HASH_FILE,
    SESSION_FILE,
    PasswordError,
    SessionRecorder,
    list_sessions,
    open_session,
)

PASSWORD = "123456"
ORIGIN = 50_000
STARTED = datetime(2026, 11, 18, 10, 0, 0, tzinfo=timezone.utc)
EVENTS = [
    DepartureEvent(DepartureKind.GAZE, ORIGIN + 10_000, ORIGIN + 12_500),
    DepartureEvent(DepartureKind.SEAT, ORIGIN + 30_000, ORIGIN + 35_000),
    DepartureEvent(DepartureKind.GAZE, ORIGIN + 60_000, ORIGIN + 61_200),
]


def make_session(tmp_path, events=EVENTS, close=True):
    recorder = SessionRecorder(
        student="A학생",
        password=PASSWORD,
        exam_seconds=3600,
        origin_ms=ORIGIN,
        root=tmp_path,
        started_at=STARTED,
        iterations=1000,
    )
    chains = [recorder.record(event) for event in events]
    if close:
        summary = DepartureDetector(FocusZone(0, 1, 0, 1)).summary()
        recorder.close(summary, ended_at=STARTED)
    return recorder.directory, chains


def read_rows(directory):
    with open(directory / EVENTS_FILE, newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def write_rows(directory, rows):
    with open(directory / EVENTS_FILE, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)


def test_intact_session_verifies(tmp_path):
    directory, chains = make_session(tmp_path)
    session = open_session(directory, PASSWORD)
    assert session.intact, session.problems
    assert [e.chain for e in session.events] == chains
    assert [e.start_offset_ms for e in session.events] == [10_000, 30_000, 60_000]
    assert all(e.valid for e in session.events)
    assert session.header["student"] == "A학생"
    assert (directory / EVENTS_HASH_FILE).exists()


def test_password_and_key_are_not_stored(tmp_path):
    directory, _ = make_session(tmp_path)
    for name in (SESSION_FILE, EVENTS_FILE):
        assert PASSWORD not in (directory / name).read_text(encoding="utf-8")


def test_wrong_password_is_rejected(tmp_path):
    directory, _ = make_session(tmp_path)
    with pytest.raises(PasswordError):
        open_session(directory, "654321")


def test_modified_row_is_detected_only_at_that_row(tmp_path):
    directory, _ = make_session(tmp_path)
    rows = read_rows(directory)
    rows[2][4] = "100"  # 2번 기록 지속 시간 5000 → 100
    write_rows(directory, rows)
    session = open_session(directory, PASSWORD)
    assert not session.intact
    assert [e.valid for e in session.events] == [True, False, True]


def test_recomputed_plain_hash_does_not_hide_tampering(tmp_path):
    """기록을 고치고 SHA256 파일까지 다시 만들어도 체인값에서 드러난다."""
    import hashlib

    directory, _ = make_session(tmp_path)
    rows = read_rows(directory)
    rows[1][4] = "1"
    write_rows(directory, rows)
    digest = hashlib.sha256((directory / EVENTS_FILE).read_bytes()).hexdigest()
    (directory / EVENTS_HASH_FILE).write_text(f"{digest}  {EVENTS_FILE}\n")
    session = open_session(directory, PASSWORD)
    assert not session.events[0].valid
    assert any("체인값" in p for p in session.problems)


def test_deleted_middle_row_is_detected(tmp_path):
    directory, _ = make_session(tmp_path)
    rows = read_rows(directory)
    del rows[2]
    write_rows(directory, rows)
    session = open_session(directory, PASSWORD)
    assert [e.valid for e in session.events] == [True, False]
    assert any("삭제" in p for p in session.problems)


def test_deleted_last_row_is_detected(tmp_path):
    directory, _ = make_session(tmp_path)
    write_rows(directory, read_rows(directory)[:-1])
    session = open_session(directory, PASSWORD)
    assert all(e.valid for e in session.events)
    assert any("마지막 기록" in p for p in session.problems)


def test_modified_header_is_detected(tmp_path):
    directory, _ = make_session(tmp_path)
    path = directory / SESSION_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["header"]["criteria"]["suspicious_count"] = 99
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    session = open_session(directory, PASSWORD)
    assert any("시험 정보" in p for p in session.problems)


def test_unclosed_session_keeps_events_but_reports(tmp_path):
    directory, _ = make_session(tmp_path, close=False)
    session = open_session(directory, PASSWORD)
    assert len(session.events) == 3 and all(e.valid for e in session.events)
    assert any("종료 기록" in p for p in session.problems)


def test_recorder_validation(tmp_path):
    common = dict(exam_seconds=60, origin_ms=0, root=tmp_path, iterations=1000)
    with pytest.raises(ValueError):
        SessionRecorder(student="A", password="123", **common)
    with pytest.raises(ValueError):
        SessionRecorder(student=" ", password=PASSWORD, **common)
    recorder = SessionRecorder(student="A", password=PASSWORD, **common)
    recorder.close(DepartureDetector(FocusZone(0, 1, 0, 1)).summary())
    with pytest.raises(RuntimeError):
        recorder.record(EVENTS[0])


def test_list_sessions(tmp_path):
    assert list_sessions(tmp_path / "none") == []
    directory, _ = make_session(tmp_path)
    assert list_sessions(tmp_path) == [directory]
