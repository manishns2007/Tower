"""YOLO-based object detector implementation.

Implements BaseDetector using the Ultralytics YOLO backend while isolating
all Ultralytics-specific types from the rest of the application.
"""

import logging
from typing import List, Optional, Union
import numpy as np
import torch
from ultralytics import YOLO

from config import DetectorConfig
from inference.base_detector import BaseDetector
from inference.detection import Detection

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """Object detector using Ultralytics YOLO models."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        config: Optional[DetectorConfig] = None,
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            model_path: Path to YOLO weights (e.g. 'yolov8n.pt').
            device: Computing device ('auto', 'cpu', 'cuda', etc.).
            confidence_threshold: Minimum confidence score to retain a detection.
            config: Optional DetectorConfig object providing default parameters.
        """
        # Resolve configuration values
        base_cfg = config or DetectorConfig()

        self.model_path = model_path if model_path is not None else base_cfg.model_path
        raw_device = device if device is not None else base_cfg.device
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else base_cfg.confidence_threshold
        )

        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )

        self.device = self._resolve_device(raw_device)
        logger.info(
            "Initializing YOLODetector with model=%s, device=%s, conf_thresh=%.2f",
            self.model_path,
            self.device,
            self.confidence_threshold,
        )

        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            logger.error("Failed to load YOLO model from '%s': %s", self.model_path, exc)
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_path}': {exc}"
            ) from exc

    def _resolve_device(self, device_str: str) -> str:
        """Resolve requested device string into an active PyTorch device identifier."""
        target = device_str.lower().strip()
        if target == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Auto device resolved to '%s'", resolved)
            return resolved
        if target.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested ('%s') but not available. Falling back to CPU.", target)
            return "cpu"
        return target

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a given frame.

        Args:
            frame: Input video frame as a NumPy array (H x W x C).

        Returns:
            List of internal Detection dataclass instances.

        Raises:
            ValueError: If the frame is invalid, empty, or not a NumPy array.
        """
        if frame is None:
            raise ValueError("Input frame is None")

        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Expected frame to be a numpy.ndarray, got {type(frame)}")

        if frame.size == 0 or any(dim == 0 for dim in frame.shape):
            raise ValueError(f"Input frame has empty dimensions: shape={frame.shape}")

        if frame.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2D or 3D image array, got {frame.ndim} dimensions with shape {frame.shape}"
            )

        try:
            # Perform prediction with Ultralytics model
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Error during YOLO inference: %s", exc)
            raise RuntimeError(f"Error during YOLO inference: {exc}") from exc

        detections: List[Detection] = []
        if not results:
            return detections

        first_result = results[0]
        boxes = getattr(first_result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        class_names = getattr(first_result, "names", {}) or {}

        for box in boxes:
            conf = float(box.conf[0].item())
            if conf < self.confidence_threshold:
                continue

            class_id = int(box.cls[0].item())
            class_name = str(class_names.get(class_id, class_id))

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                center=center,
            )
            detections.append(detection)

        return detections
