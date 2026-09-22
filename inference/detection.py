"""Detection data model for Tower.

Defines a clean, hardware-agnostic internal representation of an object detection.
Downstream modules consume this object rather than backend-specific output tensors.
"""

from dataclasses import dataclass, field
from typing import Tuple


Bbox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """Internal representation of a single detected object.

    Attributes:
        class_id: Numerical identifier of the detected class.
        class_name: Human-readable name of the detected class.
        confidence: Prediction confidence score between 0.0 and 1.0.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        center: Representative center coordinate (center_x, center_y).
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Bbox
    center: Tuple[float, float] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Validate confidence
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        # Validate bbox structure
        if len(self.bbox) != 4:
            raise ValueError(
                f"Bounding box must contain exactly 4 values (x1, y1, x2, y2), got {len(self.bbox)}"
            )

        x1, y1, x2, y2 = self.bbox
        if x1 > x2:
            raise ValueError(f"Invalid bbox: x1 ({x1}) must be <= x2 ({x2})")
        if y1 > y2:
            raise ValueError(f"Invalid bbox: y1 ({y1}) must be <= y2 ({y2})")

        # Automatically compute center if not provided
        if self.center is None:
            computed_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            object.__setattr__(self, "center", computed_center)
        elif len(self.center) != 2:
            raise ValueError(
                f"Center must contain exactly 2 values (x, y), got {len(self.center)}"
            )
