"""Zones module for Tower.

Provides spatial zone definitions and coordinate-based spatial queries.
Completely decoupled from detectors, models, cameras, and tracking logic.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

logger = logging.getLogger(__name__)

Coordinate = Tuple[float, float]
BboxCoordinates = Tuple[float, float, float, float]


@dataclass
class Zone:
    """Representation of a defined spatial region.

    Attributes:
        name: Unique identifier or label for the zone (e.g. 'front_door').
        polygon: List of 2D (x, y) coordinates defining the boundary in frame coordinates.
        type: Semantic zone classification (e.g. 'entry', 'activity_area', 'couch_area').
    """

    name: str
    polygon: List[Coordinate]
    type: str

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(
                f"Zone '{self.name}' must have at least 3 points to form a polygon, got {len(self.polygon)}"
            )


def _is_point_on_segment(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float, tol: float = 1e-7
) -> bool:
    """Check if point (px, py) lies on line segment (x1, y1) -> (x2, y2)."""
    # Cross product checks collinearity
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > tol:
        return False
    # Check if point lies within the segment's bounding box
    if (min(x1, x2) - tol <= px <= max(x1, x2) + tol) and (
        min(y1, y2) - tol <= py <= max(y1, y2) + tol
    ):
        return True
    return False


def is_point_in_polygon(point: Coordinate, polygon: List[Coordinate]) -> bool:
    """Determine whether a 2D coordinate is inside or on the boundary of a polygon.

    Uses the ray-casting algorithm with explicit boundary/vertex inclusion.

    Args:
        point: (x, y) coordinate.
        polygon: List of (x, y) vertices defining the polygon.

    Returns:
        True if the point is inside or on the boundary, False otherwise.
    """
    if len(polygon) < 3:
        return False

    px, py = point
    n = len(polygon)

    # 1. Boundary / vertex test
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _is_point_on_segment(px, py, x1, y1, x2, y2):
            return True

    # 2. Ray-casting test (horizontal ray towards +x)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if horizontal ray crosses this edge
        if (y1 > py) != (y2 > py):
            x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_intersect:
                inside = not inside

    return inside


class ZoneManager:
    """Manages spatial zones and resolves coordinate containment queries."""

    def __init__(self, zones: Optional[List[Zone]] = None) -> None:
        """Initialize ZoneManager with an optional initial list of zones."""
        self._zones: List[Zone] = []
        if zones:
            for zone in zones:
                self.add_zone(zone)

    def add_zone(self, zone: Zone) -> None:
        """Register a new spatial zone. Preserves insertion order."""
        self._zones.append(zone)
        logger.debug("Added zone '%s' with %d vertices", zone.name, len(zone.polygon))

    def get_zones(self) -> List[Zone]:
        """Return all registered zones in configured order."""
        return list(self._zones)

    def get_zone(self, name: str) -> Optional[Zone]:
        """Retrieve a zone by its name."""
        for zone in self._zones:
            if zone.name == name:
                return zone
        return None

    def clear(self) -> None:
        """Clear all registered zones."""
        self._zones.clear()

    def point_in_zone(self, point: Coordinate) -> Optional[Zone]:
        """Find the zone containing the specified point.

        If a point belongs to multiple overlapping zones, returns the first
        matching zone according to the configured order.

        Args:
            point: (x, y) coordinate.

        Returns:
            The first matching Zone containing the point, or None if outside all zones.
        """
        for zone in self._zones:
            if is_point_in_polygon(point, zone.polygon):
                return zone
        return None

    def track_in_zone(
        self, location: Union[Coordinate, BboxCoordinates]
    ) -> Optional[Zone]:
        """Determine which zone contains a tracked entity's location.

        Accepts either a point coordinate (x, y) or a bounding box (x1, y1, x2, y2).
        For bounding boxes, the representative center point is evaluated.

        Args:
            location: (x, y) point or (x1, y1, x2, y2) bounding box.

        Returns:
            The first matching Zone, or None if outside all zones.
        """
        if len(location) == 2:
            return self.point_in_zone(location)  # type: ignore[arg-type]
        elif len(location) == 4:
            x1, y1, x2, y2 = location
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            return self.point_in_zone(center)
        else:
            raise ValueError(
                f"Expected 2-tuple (x, y) or 4-tuple (x1, y1, x2, y2), got {location}"
            )

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Populate zones from a dictionary.

        Supports standard Tower zone configuration schema:
        zones:
          front_door:
            type: entry
            points: [[x1, y1], [x2, y2], ...]
        """
        zones_data = data.get("zones", data)
        if isinstance(zones_data, dict):
            for name, details in zones_data.items():
                if not isinstance(details, dict):
                    continue
                zone_type = details.get("type", "general")
                raw_points = details.get("points") or details.get("polygon") or []
                polygon = [(float(p[0]), float(p[1])) for p in raw_points]
                self.add_zone(Zone(name=str(name), polygon=polygon, type=zone_type))
        elif isinstance(zones_data, list):
            for item in zones_data:
                name = item["name"]
                zone_type = item.get("type", "general")
                raw_points = item.get("points") or item.get("polygon") or []
                polygon = [(float(p[0]), float(p[1])) for p in raw_points]
                self.add_zone(Zone(name=str(name), polygon=polygon, type=zone_type))
        else:
            raise ValueError(f"Invalid zones format: expected dict or list, got {type(zones_data)}")

    def load_from_yaml(self, path: Union[str, Path]) -> None:
        """Load and append zones from a YAML configuration file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Zone config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            logger.warning("Empty zone configuration in %s", config_path)
            return

        self.load_from_dict(data)

    def load_from_json(self, path: Union[str, Path]) -> None:
        """Load and append zones from a JSON configuration file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Zone config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.load_from_dict(data)
