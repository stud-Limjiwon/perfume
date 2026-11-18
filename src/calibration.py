"""시험 시작 전 5점 보정으로 응시자의 집중 구역(동공 위치 범위)을 정한다.

1. 화면 중앙과 네 모서리 점을 순서대로 CALIBRATION_SECONDS_PER_POINT초씩 본다.
2. 점이 바뀐 직후(SETTLE_MS)는 눈이 움직이는 중이라 버린다.
3. 점마다 동공 위치의 중앙값을 구하고, 다섯 중앙값의 범위에
   FOCUS_ZONE_MARGIN(범위 대비 비율)만큼 여유를 더해 집중 구역으로 정한다.
4. 한 점에서 유효한 값이 부족하면(얼굴 미검출·깜빡임) 그 점을 다시 한다.

[임시 구현] 뇌공학 담당이 기준(여유 비율, 점 위치)을 시선 연구 근거로 조정한다.
"""

from __future__ import annotations

from statistics import median

from config.settings import (
    CALIBRATION_POINTS,
    CALIBRATION_SECONDS_PER_POINT,
    FOCUS_ZONE_MARGIN,
)
from src.attention_detection import FocusZone, PupilSample

# 화면 크기 대비 비율 좌표 (가로, 세로): 중앙, 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래
TARGETS = ((0.5, 0.5), (0.06, 0.08), (0.94, 0.08), (0.94, 0.92), (0.06, 0.92))
SETTLE_MS = 500
MIN_SAMPLES_PER_POINT = 5
MIN_ZONE_RANGE = 0.01

assert len(TARGETS) == CALIBRATION_POINTS


class CalibrationError(ValueError):
    """보정 결과로 집중 구역을 만들 수 없음."""


class Calibrator:
    """프레임마다 add()를 호출한다. done이 되면 zone()으로 집중 구역을 얻는다."""

    def __init__(
        self,
        *,
        seconds_per_point: float = CALIBRATION_SECONDS_PER_POINT,
        margin: float = FOCUS_ZONE_MARGIN,
        targets: tuple[tuple[float, float], ...] = TARGETS,
        settle_ms: int = SETTLE_MS,
        min_samples: int = MIN_SAMPLES_PER_POINT,
    ) -> None:
        self.point_ms = round(seconds_per_point * 1000)
        if self.point_ms <= settle_ms:
            raise ValueError("점마다 보는 시간은 안정화 시간보다 길어야 합니다.")
        self.margin = margin
        self.targets = targets
        self.settle_ms = settle_ms
        self.min_samples = min_samples
        self.index = 0
        self.retries = 0
        self._start_ms: int | None = None
        self._elapsed_ms = 0
        self._samples: list[list[tuple[float, float]]] = [[] for _ in targets]

    @property
    def done(self) -> bool:
        return self.index >= len(self.targets)

    @property
    def target(self) -> tuple[float, float] | None:
        return None if self.done else self.targets[self.index]

    @property
    def progress(self) -> float:
        """현재 점에서 지난 시간 비율 (0~1)."""
        return min(self._elapsed_ms / self.point_ms, 1.0)

    def add(self, sample: PupilSample) -> None:
        if self.done:
            return
        now = sample.timestamp_ms
        if self._start_ms is None:
            self._start_ms = now
        if now - self._start_ms >= self.point_ms:
            # 시간이 다 된 점을 마무리하고, 이 프레임부터 다음 점(또는 같은 점 재시도)
            if len(self._samples[self.index]) >= self.min_samples:
                self.index += 1
            else:
                self._samples[self.index].clear()
                self.retries += 1
            self._start_ms = now
            if self.done:
                self._elapsed_ms = 0
                return
        self._elapsed_ms = now - self._start_ms
        valid = sample.face_detected and sample.x is not None and sample.y is not None
        if valid and self._elapsed_ms >= self.settle_ms:
            self._samples[self.index].append((sample.x, sample.y))

    def point_medians(self) -> list[tuple[float, float]]:
        return [
            (median(x for x, _ in s), median(y for _, y in s)) for s in self._samples
        ]

    def zone(self) -> FocusZone:
        if not self.done:
            raise CalibrationError("보정이 아직 끝나지 않았습니다.")
        xs, ys = zip(*self.point_medians())
        x_range, y_range = max(xs) - min(xs), max(ys) - min(ys)
        if x_range < MIN_ZONE_RANGE or y_range < MIN_ZONE_RANGE:
            raise CalibrationError(
                "점마다 동공 위치가 거의 같습니다. 고개를 고정하고 눈으로 점을 따라가며 "
                "다시 보정하세요."
            )
        return FocusZone(
            min(xs) - x_range * self.margin,
            max(xs) + x_range * self.margin,
            min(ys) - y_range * self.margin,
            max(ys) + y_range * self.margin,
        )
