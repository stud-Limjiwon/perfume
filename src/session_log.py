"""시험 1회(세션)의 이탈 기록을 체인값과 함께 저장하고 검증한다.

구조 (logs/sessions/<시작시각>_<학생>/)
- session.json : 시험 정보(학생, 시작 시각, 시험 시간, 판정 기준, 키 생성 정보)와 종료 기록
- events.csv   : 이탈 1회마다 한 줄. 마지막 열이 체인값
- events.csv.sha256 : 종료 시 events.csv 전체의 SHA256 (파일 식별용)

무결성 방식
- 감독관 패스워드로 PBKDF2-HMAC-SHA256 키를 만든다. 패스워드와 키는 저장하지 않는다.
- 체인값 = HMAC-SHA256(키, 이전 체인값 + 이번 기록). 첫 값은 시험 정보로 만든다.
- 한 줄을 고치거나 지우면 그 줄(또는 다음 줄)부터 체인값이 맞지 않는다.
- 키가 없으면 맞는 체인값을 새로 만들 수 없으므로, 기록과 체인값을 함께 고쳐도 검증에서 드러난다.

영상과 얼굴 이미지는 저장하지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from config.settings import PASSWORD_MIN_LENGTH, PBKDF2_ITERATIONS, SESSION_LOG_DIR
from src.attention_detection import (
    DepartureEvent,
    DepartureKind,
    DetectionCriteria,
    DetectionSummary,
)

FORMAT_VERSION = 1
SESSION_FILE = "session.json"
EVENTS_FILE = "events.csv"
EVENTS_HASH_FILE = "events.csv.sha256"
CSV_COLUMNS = (
    "seq",
    "kind",
    "start_offset_ms",
    "end_offset_ms",
    "duration_ms",
    "chain",
)
GENESIS = "0" * 64
_PASSWORD_CHECK = b"perfume-password-check"


class PasswordError(ValueError):
    """패스워드가 틀려 기록을 열 수 없음."""


def _canonical(data: dict) -> bytes:
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def derive_key(password: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def _password_check(key: bytes) -> str:
    return hmac.new(key, _PASSWORD_CHECK, hashlib.sha256).hexdigest()


def chain_link(key: bytes, previous: str, payload: dict) -> str:
    """이전 체인값과 이번 기록으로 다음 체인값을 만든다."""
    message = bytes.fromhex(previous) + _canonical(payload)
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _safe_name(student: str) -> str:
    return re.sub(r"[^\w-]", "_", student.strip())[:40]


def summary_to_dict(summary: DetectionSummary) -> dict:
    return {
        "gaze_count": summary.gaze.count,
        "gaze_total_ms": summary.gaze.total_ms,
        "seat_count": summary.seat.count,
        "seat_total_ms": summary.seat.total_ms,
        "suspicious": summary.suspicious,
    }


class SessionRecorder:
    """시험 시작 시 만들고, 이탈마다 record(), 종료 시 close()를 호출한다.

    origin_ms는 시험 시작 순간의 시각이며 이탈 이벤트와 같은 시계를 써야 한다.
    """

    def __init__(
        self,
        *,
        student: str,
        password: str,
        exam_seconds: int,
        origin_ms: int,
        criteria: DetectionCriteria | None = None,
        root: Path = SESSION_LOG_DIR,
        started_at: datetime | None = None,
        iterations: int = PBKDF2_ITERATIONS,
    ) -> None:
        if not student.strip():
            raise ValueError("학생 이름을 입력해야 합니다.")
        if len(password) < PASSWORD_MIN_LENGTH:
            raise ValueError(
                f"패스워드는 {PASSWORD_MIN_LENGTH}자 이상이어야 합니다."
            )
        if exam_seconds <= 0:
            raise ValueError("시험 시간은 0보다 커야 합니다.")
        started_at = started_at or datetime.now().astimezone()
        salt = os.urandom(16)
        self._key: bytes | None = derive_key(password, salt, iterations)
        self.origin_ms = origin_ms
        self.criteria = criteria or DetectionCriteria()
        self.directory = (
            Path(root) / f"{started_at:%Y%m%d-%H%M%S}_{_safe_name(student)}"
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.header = {
            "format": FORMAT_VERSION,
            "student": student.strip(),
            "started_at": started_at.isoformat(timespec="seconds"),
            "exam_seconds": int(exam_seconds),
            "criteria": self.criteria.as_dict(),
            "kdf": {
                "name": "pbkdf2_hmac_sha256",
                "salt": salt.hex(),
                "iterations": iterations,
            },
            "password_check": _password_check(self._key),
        }
        self._chain = chain_link(self._key, GENESIS, self.header)
        self._session = {"header": self.header, "header_chain": self._chain}
        _write_json(self.directory / SESSION_FILE, self._session)
        with open(self.directory / EVENTS_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        self._seq = 0

    @property
    def closed(self) -> bool:
        return self._key is None

    def check_password(self, password: str) -> bool:
        """감독관 확인용 (예: 시험 중 종료). 저장된 확인값과 비교한다."""
        kdf = self.header["kdf"]
        key = derive_key(password, bytes.fromhex(kdf["salt"]), int(kdf["iterations"]))
        return hmac.compare_digest(_password_check(key), self.header["password_check"])

    def record(self, event: DepartureEvent) -> str:
        """이탈 1회를 바로 파일에 덧붙이고 체인값을 반환한다."""
        if self._key is None:
            raise RuntimeError("이미 종료된 기록입니다.")
        self._seq += 1
        row = {
            "seq": self._seq,
            "kind": DepartureKind(event.kind).value,
            "start_offset_ms": event.start_ms - self.origin_ms,
            "end_offset_ms": event.end_ms - self.origin_ms,
            "duration_ms": event.duration_ms,
        }
        self._chain = chain_link(self._key, self._chain, row)
        with open(self.directory / EVENTS_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([*row.values(), self._chain])
            f.flush()
            os.fsync(f.fileno())
        return self._chain

    def close(
        self, summary: DetectionSummary, ended_at: datetime | None = None
    ) -> str:
        """종료 기록을 체인에 이어 붙이고 키를 버린다. 최종 체인값을 반환한다."""
        if self._key is None:
            raise RuntimeError("이미 종료된 기록입니다.")
        ended_at = ended_at or datetime.now().astimezone()
        end = {
            "ended_at": ended_at.isoformat(timespec="seconds"),
            "event_count": self._seq,
            "summary": summary_to_dict(summary),
        }
        final_chain = chain_link(self._key, self._chain, end)
        self._session.update(end=end, final_chain=final_chain)
        _write_json(self.directory / SESSION_FILE, self._session)
        digest = file_sha256(self.directory / EVENTS_FILE)
        (self.directory / EVENTS_HASH_FILE).write_text(
            f"{digest}  {EVENTS_FILE}\n", encoding="utf-8"
        )
        self._key = None
        return final_chain


@dataclass
class RecordedEvent:
    seq: int
    kind: DepartureKind
    start_offset_ms: int
    end_offset_ms: int
    duration_ms: int
    chain: str
    valid: bool


@dataclass
class VerifiedSession:
    """패스워드로 연 기록과 검증 결과."""

    directory: Path
    header: dict
    criteria: DetectionCriteria
    events: list[RecordedEvent]
    end: dict | None
    final_chain: str | None
    problems: list[str] = field(default_factory=list)

    @property
    def intact(self) -> bool:
        return not self.problems


def _parse_row(row: dict) -> tuple[dict, str]:
    payload = {
        "seq": int(row["seq"]),
        "kind": DepartureKind(row["kind"]).value,
        "start_offset_ms": int(row["start_offset_ms"]),
        "end_offset_ms": int(row["end_offset_ms"]),
        "duration_ms": int(row["duration_ms"]),
    }
    return payload, row["chain"]


def open_session(directory: Path, password: str) -> VerifiedSession:
    """패스워드를 확인하고 모든 체인값을 다시 계산해 검증한다.

    패스워드가 틀리면 PasswordError. 기록이 바뀐 경우에는 problems에 이유를 담는다.
    """
    directory = Path(directory)
    session = json.loads((directory / SESSION_FILE).read_text(encoding="utf-8"))
    header = session["header"]
    kdf = header["kdf"]
    key = derive_key(password, bytes.fromhex(kdf["salt"]), int(kdf["iterations"]))
    if not hmac.compare_digest(_password_check(key), header["password_check"]):
        raise PasswordError("패스워드가 맞지 않습니다.")

    problems: list[str] = []
    expected = chain_link(key, GENESIS, header)
    stored = session.get("header_chain", "")
    if not hmac.compare_digest(expected, stored):
        problems.append("시험 정보(session.json의 header)가 바뀌었습니다.")

    events: list[RecordedEvent] = []
    previous = stored
    expected_seq = 1
    with open(directory / EVENTS_FILE, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            problems.append("events.csv의 열 구성이 바뀌었습니다.")
        for line_number, row in enumerate(reader, start=1):
            try:
                payload, chain = _parse_row(row)
            except (KeyError, TypeError, ValueError):
                problems.append(f"{line_number}번째 기록을 읽을 수 없습니다.")
                previous = row.get("chain") or previous
                continue
            valid = hmac.compare_digest(chain_link(key, previous, payload), chain)
            if payload["seq"] != expected_seq:
                problems.append(
                    f"{expected_seq}번이 와야 할 자리에 {payload['seq']}번 기록이"
                    " 있습니다 (기록 삭제 또는 순서 변경)."
                )
                valid = False
            elif not valid:
                problems.append(f"{payload['seq']}번 기록의 체인값이 맞지 않습니다.")
            expected_seq = payload["seq"] + 1
            events.append(
                RecordedEvent(
                    seq=payload["seq"],
                    kind=DepartureKind(payload["kind"]),
                    start_offset_ms=payload["start_offset_ms"],
                    end_offset_ms=payload["end_offset_ms"],
                    duration_ms=payload["duration_ms"],
                    chain=chain,
                    valid=valid,
                )
            )
            previous = chain

    end = session.get("end")
    final_chain = session.get("final_chain")
    if end is None:
        problems.append("종료 기록이 없습니다 (비정상 종료 또는 시험 진행 중).")
    else:
        if not hmac.compare_digest(chain_link(key, previous, end), final_chain or ""):
            problems.append("종료 기록이 바뀌었거나 마지막 기록이 삭제되었습니다.")
        if end.get("event_count") != len(events):
            problems.append(
                f"종료 기록의 이탈 수({end.get('event_count')})와"
                f" 실제 기록 수({len(events)})가 다릅니다."
            )
        hash_file = directory / EVENTS_HASH_FILE
        if hash_file.exists():
            saved = hash_file.read_text(encoding="utf-8").split()[0]
            if saved != file_sha256(directory / EVENTS_FILE):
                problems.append("events.csv의 SHA256이 종료 시점과 다릅니다.")
        else:
            problems.append("events.csv.sha256 파일이 없습니다.")

    try:
        criteria = DetectionCriteria.from_dict(header["criteria"])
    except (KeyError, TypeError, ValueError):
        problems.append("기록된 판정 기준을 읽을 수 없어 현재 기본값을 사용합니다.")
        criteria = DetectionCriteria()

    return VerifiedSession(
        directory=directory,
        header=header,
        criteria=criteria,
        events=events,
        end=end,
        final_chain=final_chain,
        problems=problems,
    )


def list_sessions(root: Path = SESSION_LOG_DIR) -> list[Path]:
    """기록 폴더 목록(최근 순). 내용은 패스워드 없이 열지 않는다."""
    root = Path(root)
    if not root.exists():
        return []
    folders = [p for p in root.iterdir() if (p / SESSION_FILE).exists()]
    return sorted(folders, reverse=True)
