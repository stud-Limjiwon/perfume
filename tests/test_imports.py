"""필수 의존성과 프로젝트 모듈 import를 실제로 검증한다."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "cv2",
        "mediapipe",
        "numpy",
        "pandas",
        "torch",
        "sklearn",
        "matplotlib",
        "pyarrow",
        "pytest",
        "config",
        "config.settings",
        "src",
        "src.camera",
        "src.landmarks",
        "src.features",
        "src.calibration",
        "src.gaze_prediction",
        "src.attention_detection",
        "src.session_log",
        "src.visualization",
        "training",
        "training.baseline",
        "training.model",
        "training.train",
        "training.evaluate",
        "main",
    ],
)
def test_import(module_name: str) -> None:
    """누락된 설치/경로 오류를 실패로 표시한다. 카메라 접근은 하지 않는다."""
    importlib.import_module(module_name)
