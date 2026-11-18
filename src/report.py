"""화면 3(시험 종료 후)에 보여 줄 의심 정황 목록을 만든다.

- 빨간색: 1초 넘게 벗어난 이탈 1건. 오래 벗어난 순서로 위에서부터 나열
- 노란색: 누적 기준(횟수·시간)을 처음 넘은 시점. 빨간색 아래에 시각 순서로 나열
- 각 항목은 근거 기록의 체인값을 가진다 (화면에서 오른쪽을 누르면 조회).
- 판정 기준은 현재 설정이 아니라 기록 당시 저장된 기준을 쓴다.

명령줄 확인:  python -m src.report logs/sessions/<기록 폴더>
"""

from __future__ import annotations

import argparse
import getpass
import sys
from dataclasses import dataclass
from pathlib import Path

from src.attention_detection import KIND_LABELS, DepartureKind
from src.session_log import PasswordError, RecordedEvent, VerifiedSession, open_session

RED = "red"
YELLOW = "yellow"


@dataclass(frozen=True)
class ReportItem:
    color: str
    title: str
    detail: str
    offset_ms: int
    duration_ms: int
    chain: str
    valid: bool
    seq: int


def format_offset(ms: int) -> str:
    """시험 시작 후 경과 시간: 00시00분00초"""
    seconds = max(ms, 0) // 1000
    return f"{seconds // 3600:02d}시{seconds % 3600 // 60:02d}분{seconds % 60:02d}초"


def _title(student: str, offset_ms: int) -> str:
    return f"{student} | 의심 시간(시험 시작 후 {format_offset(offset_ms)})"


def _cumulative_items(session: VerifiedSession, kind: DepartureKind) -> list:
    criteria = session.criteria
    label = KIND_LABELS[kind]
    student = session.header["student"]
    items: list[ReportItem] = []
    count = total = 0
    count_done = time_done = False
    for event in sorted(session.events, key=lambda e: e.start_offset_ms):
        if event.kind is not kind:
            continue
        count += 1
        total += event.duration_ms
        reasons = []
        if not count_done and count >= criteria.suspicious_count:
            count_done = True
            reasons.append(f"누적 {count}회 (기준 {criteria.suspicious_count}회)")
        if not time_done and total >= criteria.suspicious_total_ms:
            time_done = True
            reasons.append(
                f"누적 {total / 1000:.1f}초"
                f" (기준 {criteria.suspicious_total_ms / 1000:.0f}초)"
            )
        if reasons:
            items.append(
                ReportItem(
                    color=YELLOW,
                    title=_title(student, event.end_offset_ms),
                    detail=f"{label} " + ", ".join(reasons) + " 도달",
                    offset_ms=event.end_offset_ms,
                    duration_ms=event.duration_ms,
                    chain=event.chain,
                    valid=event.valid,
                    seq=event.seq,
                )
            )
    return items


def _event_item(student: str, event: RecordedEvent) -> ReportItem:
    return ReportItem(
        color=RED,
        title=_title(student, event.start_offset_ms),
        detail=f"{KIND_LABELS[event.kind]} {event.duration_ms / 1000:.1f}초",
        offset_ms=event.start_offset_ms,
        duration_ms=event.duration_ms,
        chain=event.chain,
        valid=event.valid,
        seq=event.seq,
    )


def build_report(session: VerifiedSession) -> list[ReportItem]:
    student = session.header["student"]
    red = sorted(
        (_event_item(student, e) for e in session.events),
        key=lambda item: (-item.duration_ms, item.offset_ms),
    )
    yellow = sorted(
        (
            item
            for kind in DepartureKind
            for item in _cumulative_items(session, kind)
        ),
        key=lambda item: item.offset_ms,
    )
    return red + yellow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="시험 기록 검증과 의심 정황 목록")
    parser.add_argument("directory", type=Path, help="logs/sessions 아래 기록 폴더")
    parser.add_argument("--chain", action="store_true", help="체인값도 출력")
    args = parser.parse_args(argv)
    password = getpass.getpass("감독관 패스워드: ")
    try:
        session = open_session(args.directory, password)
    except PasswordError as error:
        print(error, file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as error:
        print(f"기록을 열 수 없습니다: {error}", file=sys.stderr)
        return 1

    header = session.header
    print(f"학생: {header['student']}  시작: {header['started_at']}")
    if session.intact:
        print("무결성: 정상 (모든 체인값 일치)")
    else:
        print("무결성: 문제 발견")
        for problem in session.problems:
            print(f"  - {problem}")
    items = build_report(session)
    if not items:
        print("의심 정황 없음")
    for item in items:
        mark = "[빨강]" if item.color == RED else "[노랑]"
        broken = "" if item.valid else "  ※ 체인값 불일치"
        print(f"{mark} {item.title}  {item.detail}{broken}")
        if args.chain:
            print(f"       체인값: {item.chain}")
    return 0 if session.intact else 3


if __name__ == "__main__":
    raise SystemExit(main())
