"""Zones package for Tower.

Provides spatial zone definitions and point-in-zone containment resolution.
"""

from zones.zone_manager import Coordinate, Zone, ZoneManager, is_point_in_polygon

__all__ = ["Zone", "ZoneManager", "Coordinate", "is_point_in_polygon"]
