"""온라인 시험 감독 프로그램 실행 진입점.

    python exam.py

화면 1(설정) → 5점 보정 → 화면 2(시험 중) → 화면 3(결과 조회) 순서로 진행한다.
"""

import argparse
from pathlib import Path

from config.settings import FACE_LANDMARKER_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description="온라인 시험 감독 화면")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=FACE_LANDMARKER_MODEL,
        help="Face Landmarker .task 경로",
    )
    args = parser.parse_args()
    from src.exam_app import run

    return run(args.model_path)


if __name__ == "__main__":
    raise SystemExit(main())
