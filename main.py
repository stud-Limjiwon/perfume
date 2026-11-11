"""얼굴·눈·홍채 랜드마크 미리보기. 시선 예측 및 데이터 저장은 수행하지 않는다."""

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import cv2

from config.settings import FACE_LANDMARKER_MODEL
from src.camera import run_preview
from src.landmarks import FaceLandmarkDetector


def positive_int(value: str) -> int:
    """자동 종료 테스트에 사용할 양의 프레임 수를 검증한다."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> int:
    """기본으로 랜드마크를 표시하고 --camera-only는 기존 웹캠 실행을 유지한다."""
    parser = argparse.ArgumentParser(
        description="Face / eye / iris landmarks (Q to quit)"
    )
    parser.add_argument(
        "--max-frames", type=positive_int, help="automatically stop after N frames"
    )
    parser.add_argument(
        "--camera-only", action="store_true", help="preview webcam without MediaPipe"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=FACE_LANDMARKER_MODEL,
        help="Face Landmarker .task path",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="process without GUI; requires --max-frames",
    )
    args = parser.parse_args()
    if args.headless and args.max_frames is None:
        parser.error("--headless requires --max-frames")
    try:
        with ExitStack() as resources:
            processor = None
            if not args.camera_only:
                detector = resources.enter_context(
                    FaceLandmarkDetector(args.model_path)
                )
                processor = detector.process_frame
            run_preview(
                max_frames=args.max_frames,
                frame_processor=processor,
                display=not args.headless,
            )
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"실행 실패: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("사용자 요청으로 종료했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
