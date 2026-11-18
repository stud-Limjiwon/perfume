"""집중 구역 이탈을 판정하고 부정행위 의심 여부를 표시한다.

- 시선 이탈: 얼굴은 보이지만 동공 위치가 집중 구역 밖에 DEPARTURE_SECONDS보다 오래 있음
- 자리 이탈: 얼굴이 FACE_MISSING_SECONDS보다 오래 검출되지 않음
- 얼굴은 보이는데 동공 위치가 없으면(눈 깜빡임 등) 직전 상태를 그대로 유지한다.
- 이탈 시간은 기준을 넘은 뒤가 아니라 벗어난 순간부터 돌아온 순간까지 전체를 센다.

컨닝을 확정하지 않는다. 감독자가 확인할 구간과 그 근거를 만든다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite

from config.settings import (
    DEPARTURE_SECONDS,
    FACE_MISSING_SECONDS,
    SUSPICIOUS_DEPARTURE_COUNT,
    SUSPICIOUS_TOTAL_SECONDS,
)


class DepartureKind(str, Enum):
    """이탈 종류. 값은 기록 파일에 그대로 저장된다."""

    GAZE = "gaze"
    SEAT = "seat"


KIND_LABELS = {DepartureKind.GAZE: "동공 이탈", DepartureKind.SEAT: "자리 이탈"}


@dataclass(frozen=True)
class DetectionCriteria:
    """판정 기준. 기록에 함께 저장해 어떤 기준으로 판정했는지 공개한다."""

    departure_ms: int = round(DEPARTURE_SECONDS * 1000)
    face_missing_ms: int = round(FACE_MISSING_SECONDS * 1000)
    suspicious_count: int = SUSPICIOUS_DEPARTURE_COUNT
    suspicious_total_ms: int = round(SUSPICIOUS_TOTAL_SECONDS * 1000)

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(not isinstance(v, int) or v <= 0 for v in values):
            raise ValueError("판정 기준은 모두 양의 정수여야 합니다.")

    def threshold_ms(self, kind: DepartureKind) -> int:
        """이 시간보다 오래 벗어나야 이탈 1회로 센다."""
        if kind is DepartureKind.GAZE:
            return self.departure_ms
        return self.face_missing_ms

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DetectionCriteria:
        return cls(**{key: int(data[key]) for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class PupilSample:
    """한 프레임의 입력. 시각은 단조 증가하는 밀리초다."""

    timestamp_ms: int
    face_detected: bool
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True)
class FocusZone:
    """5점 보정으로 정한 동공 위치 범위(눈 안 0~1 비율, 여유 포함)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(isfinite(v) for v in values):
            raise ValueError("집중 구역 값은 유한한 숫자여야 합니다.")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("집중 구역의 최소값은 최대값보다 작아야 합니다.")

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True)
class DepartureEvent:
    """기준 시간을 넘긴 이탈 1회."""

    kind: DepartureKind
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class KindSummary:
    """한 종류 이탈의 누적 횟수와 시간."""

    count: int
    total_ms: int
    criteria: DetectionCriteria

    @property
    def count_ratio(self) -> float:
        """게이지용: 의심 기준 횟수 대비 비율(최대 1)."""
        return min(self.count / self.criteria.suspicious_count, 1.0)

    @property
    def time_ratio(self) -> float:
        """게이지용: 의심 기준 누적 시간 대비 비율(최대 1)."""
        return min(self.total_ms / self.criteria.suspicious_total_ms, 1.0)

    @property
    def suspicious(self) -> bool:
        return (
            self.count >= self.criteria.suspicious_count
            or self.total_ms >= self.criteria.suspicious_total_ms
        )


@dataclass(frozen=True)
class DetectionSummary:
    """화면 2 게이지와 결과 요약에 쓰는 현재 상태."""

    gaze: KindSummary
    seat: KindSummary
    current_kind: DepartureKind | None
    current_out_ms: int

    @property
    def suspicious(self) -> bool:
        return self.gaze.suspicious or self.seat.suspicious


_KEEP = object()  # 판단할 수 없는 프레임: 직전 상태 유지


class DepartureDetector:
    """프레임마다 update()를 호출하고 시험이 끝나면 finish()를 호출한다."""

    def __init__(
        self, zone: FocusZone, criteria: DetectionCriteria | None = None
    ) -> None:
        self.zone = zone
        self.criteria = criteria or DetectionCriteria()
        self._events: list[DepartureEvent] = []
        self._open_kind: DepartureKind | None = None
        self._open_start_ms = 0
        self._last_ms: int | None = None
        self._finished = False

    @property
    def events(self) -> tuple[DepartureEvent, ...]:
        return tuple(self._events)

    def _classify(self, sample: PupilSample):
        if not sample.face_detected:
            return DepartureKind.SEAT
        x, y = sample.x, sample.y
        if x is None or y is None or not (isfinite(x) and isfinite(y)):
            return _KEEP
        return None if self.zone.contains(x, y) else DepartureKind.GAZE

    def _check_time(self, timestamp_ms: int) -> None:
        if self._finished:
            raise RuntimeError("finish() 이후에는 입력을 받을 수 없습니다.")
        if self._last_ms is not None and timestamp_ms < self._last_ms:
            raise ValueError("시각은 줄어들 수 없습니다.")

    def _close(self, end_ms: int) -> DepartureEvent | None:
        kind = self._open_kind
        self._open_kind = None
        if kind is None:
            return None
        if end_ms - self._open_start_ms <= self.criteria.threshold_ms(kind):
            return None
        event = DepartureEvent(kind, self._open_start_ms, end_ms)
        self._events.append(event)
        return event

    def update(self, sample: PupilSample) -> DepartureEvent | None:
        """이탈이 끝나 1회로 확정되면 그 이벤트를 반환한다."""
        now = sample.timestamp_ms
        self._check_time(now)
        self._last_ms = now
        kind = self._classify(sample)
        if kind is _KEEP or kind == self._open_kind:
            return None
        finished = self._close(now)
        if kind is not None:
            self._open_kind = kind
            self._open_start_ms = now
        return finished

    def finish(self, end_ms: int | None = None) -> DepartureEvent | None:
        """시험 종료 시 진행 중인 이탈을 닫는다."""
        if end_ms is None:
            end_ms = self._last_ms if self._last_ms is not None else 0
        self._check_time(end_ms)
        self._last_ms = end_ms
        event = self._close(end_ms)
        self._finished = True
        return event

    def summary(self, now_ms: int | None = None) -> DetectionSummary:
        """기준을 이미 넘긴 진행 중 이탈도 포함한 현재 누적값."""
        now = self._last_ms if now_ms is None else now_ms
        totals = {kind: [0, 0] for kind in DepartureKind}
        for event in self._events:
            totals[event.kind][0] += 1
            totals[event.kind][1] += event.duration_ms
        current_out = 0
        if self._open_kind is not None and now is not None:
            current_out = max(now - self._open_start_ms, 0)
            if current_out > self.criteria.threshold_ms(self._open_kind):
                totals[self._open_kind][0] += 1
                totals[self._open_kind][1] += current_out
        return DetectionSummary(
            gaze=KindSummary(*totals[DepartureKind.GAZE], self.criteria),
            seat=KindSummary(*totals[DepartureKind.SEAT], self.criteria),
            current_kind=self._open_kind,
            current_out_ms=current_out,
        )
