"""Inference module for Tower.

Provides hardware-agnostic detection interfaces and implementations.
"""

from inference.base_detector import BaseDetector
from inference.detection import Bbox, Detection
from inference.detector import YOLODetector

__all__ = ["BaseDetector", "Detection", "Bbox", "YOLODetector"]
