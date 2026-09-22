"""Unit tests for the Inference module and YOLODetector."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from config import DetectorConfig
from inference.base_detector import BaseDetector
from inference.detection import Detection
from inference.detector import YOLODetector


class TestDetectionModel:
    """Tests for the Detection dataclass and validation constraints."""

    def test_valid_detection_creation(self) -> None:
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(10.0, 20.0, 50.0, 80.0),
        )
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.85
        assert det.bbox == (10.0, 20.0, 50.0, 80.0)
        assert det.center == (30.0, 50.0)

    def test_bbox_outside_image_dimensions_allowed(self) -> None:
        """Coordinates outside [0, width/height] or negative must be accepted."""
        det = Detection(
            class_id=1,
            class_name="car",
            confidence=0.9,
            bbox=(-50.0, -20.0, 1000.0, 2000.0),
        )
        assert det.bbox == (-50.0, -20.0, 1000.0, 2000.0)
        assert det.center == (475.0, 990.0)

    def test_bbox_invalid_x_order(self) -> None:
        """x1 > x2 must raise ValueError."""
        with pytest.raises(ValueError, match="x1 .* must be <= x2"):
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.5,
                bbox=(50.0, 20.0, 10.0, 80.0),
            )

    def test_bbox_invalid_y_order(self) -> None:
        """y1 > y2 must raise ValueError."""
        with pytest.raises(ValueError, match="y1 .* must be <= y2"):
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.5,
                bbox=(10.0, 100.0, 50.0, 20.0),
            )

    def test_invalid_confidence_scores(self) -> None:
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Detection(
                class_id=0,
                class_name="person",
                confidence=-0.1,
                bbox=(0.0, 0.0, 10.0, 10.0),
            )

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Detection(
                class_id=0,
                class_name="person",
                confidence=1.05,
                bbox=(0.0, 0.0, 10.0, 10.0),
            )

    def test_invalid_bbox_length(self) -> None:
        with pytest.raises(ValueError, match="exactly 4 values"):
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.5,
                bbox=(0.0, 0.0, 10.0),  # type: ignore[arg-type]
            )


class TestYOLODetector:
    """Tests for YOLODetector implementation with mocked YOLO backend."""

    @patch("inference.detector.YOLO")
    def test_detector_instantiation_and_base_detector_interface(
        self, mock_yolo_cls: MagicMock
    ) -> None:
        detector = YOLODetector(model_path="dummy.pt", device="cpu", confidence_threshold=0.3)
        assert isinstance(detector, BaseDetector)
        assert detector.model_path == "dummy.pt"
        assert detector.device == "cpu"
        assert detector.confidence_threshold == 0.3
        mock_yolo_cls.assert_called_once_with("dummy.pt")

    @patch("inference.detector.YOLO")
    def test_device_auto_resolution(self, mock_yolo_cls: MagicMock) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            det = YOLODetector(model_path="dummy.pt", device="auto")
            assert det.device == "cpu"

        with patch("torch.cuda.is_available", return_value=True):
            det = YOLODetector(model_path="dummy.pt", device="auto")
            assert det.device == "cuda"

    @patch("inference.detector.YOLO")
    def test_cuda_fallback_to_cpu_when_unavailable(self, mock_yolo_cls: MagicMock) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            det = YOLODetector(model_path="dummy.pt", device="cuda")
            assert det.device == "cpu"

    @patch("inference.detector.YOLO")
    def test_model_loading_failure_raises_runtime_error(
        self, mock_yolo_cls: MagicMock
    ) -> None:
        mock_yolo_cls.side_effect = Exception("Model file missing or corrupted")
        with pytest.raises(RuntimeError, match="Failed to load YOLO model"):
            YOLODetector(model_path="non_existent.pt")

    @patch("inference.detector.YOLO")
    def test_detect_returns_detection_objects(
        self,
        mock_yolo_cls: MagicMock,
        synthetic_frame: np.ndarray,
        mock_yolo_result: MagicMock,
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.return_value = [mock_yolo_result]
        mock_instance.names = mock_yolo_result.names
        mock_yolo_cls.return_value = mock_instance

        detector = YOLODetector(confidence_threshold=0.2)
        detections = detector.detect(synthetic_frame)

        assert len(detections) == 2
        assert isinstance(detections[0], Detection)
        assert detections[0].class_id == 0
        assert detections[0].class_name == "person"
        assert detections[0].confidence == pytest.approx(0.85, abs=1e-3)
        assert detections[0].bbox == (100.0, 150.0, 200.0, 350.0)
        assert detections[0].center == (150.0, 250.0)

        assert detections[1].class_id == 1
        assert detections[1].class_name == "dog"
        assert detections[1].confidence == pytest.approx(0.40, abs=1e-3)

    @patch("inference.detector.YOLO")
    def test_confidence_threshold_filtering(
        self,
        mock_yolo_cls: MagicMock,
        synthetic_frame: np.ndarray,
        mock_yolo_result: MagicMock,
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.return_value = [mock_yolo_result]
        mock_instance.names = mock_yolo_result.names
        mock_yolo_cls.return_value = mock_instance

        # Set threshold higher than 0.40 (dog's conf)
        detector = YOLODetector(confidence_threshold=0.50)
        detections = detector.detect(synthetic_frame)

        assert len(detections) == 1
        assert detections[0].class_name == "person"

    @patch("inference.detector.YOLO")
    def test_empty_detections(
        self,
        mock_yolo_cls: MagicMock,
        synthetic_frame: np.ndarray,
    ) -> None:
        mock_instance = MagicMock()
        empty_res = MagicMock()
        empty_res.boxes = []
        mock_instance.return_value = [empty_res]
        mock_yolo_cls.return_value = mock_instance

        detector = YOLODetector()
        detections = detector.detect(synthetic_frame)
        assert detections == []

    @patch("inference.detector.YOLO")
    def test_invalid_input_frames(self, mock_yolo_cls: MagicMock) -> None:
        detector = YOLODetector()

        # None frame
        with pytest.raises(ValueError, match="frame is None"):
            detector.detect(None)  # type: ignore[arg-type]

        # Non-array frame
        with pytest.raises(ValueError, match="Expected frame to be a numpy.ndarray"):
            detector.detect([1, 2, 3])  # type: ignore[arg-type]

        # Empty array
        with pytest.raises(ValueError, match="empty dimensions"):
            detector.detect(np.empty((0, 0, 3), dtype=np.uint8))

        # Invalid ndim
        with pytest.raises(ValueError, match="Expected 2D or 3D image array"):
            detector.detect(np.zeros((10, 10, 3, 2), dtype=np.uint8))
