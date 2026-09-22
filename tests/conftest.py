"""Pytest fixtures for Tower unit testing."""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import pytest
import torch

from zones.zone_manager import Zone


class MockBox:
    """Mock Ultralytics Box object."""

    def __init__(self, xyxy: List[float], conf: float, cls_id: int):
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.conf = torch.tensor([conf], dtype=torch.float32)
        self.cls = torch.tensor([cls_id], dtype=torch.float32)


class MockResult:
    """Mock Ultralytics Result object."""

    def __init__(self, boxes: List[MockBox], names: Dict[int, str]):
        self.boxes = boxes
        self.names = names


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Return a standard synthetic 640x480 3-channel RGB image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_yolo_result() -> MockResult:
    """Return a mock YOLO result with two detected objects."""
    boxes = [
        MockBox(xyxy=[100.0, 150.0, 200.0, 350.0], conf=0.85, cls_id=0),
        MockBox(xyxy=[300.0, 200.0, 450.0, 400.0], conf=0.40, cls_id=1),
    ]
    names = {0: "person", 1: "dog"}
    return MockResult(boxes=boxes, names=names)


@pytest.fixture
def sample_zones() -> List[Zone]:
    """Return a list of test zones."""
    return [
        Zone(
            name="front_door",
            polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            type="entry",
        ),
        Zone(
            name="kitchen",
            polygon=[(200.0, 100.0), (400.0, 100.0), (400.0, 300.0), (200.0, 300.0)],
            type="activity_area",
        ),
    ]
