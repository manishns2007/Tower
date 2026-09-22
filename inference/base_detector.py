"""Base detector interface for Tower.

Establishes a hardware-agnostic abstract base class for all object detection backends.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from inference.detection import Detection


class BaseDetector(ABC):
    """Abstract base detector.

    All detection backends (YOLO, Coral, OAK-D, Jetson, etc.) must implement
    this interface, producing standardized Detection objects.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on an input image frame.

        Args:
            frame: Input video frame as a NumPy array (H x W x C).

        Returns:
            A list of Detection objects found in the frame.
        """
        pass
