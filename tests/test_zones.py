"""Unit tests for the Zones module and ZoneManager."""

from pathlib import Path
import sys
import tempfile
import pytest
import yaml

from zones.zone_manager import Zone, ZoneManager, is_point_in_polygon


class TestZoneModel:
    """Tests for Zone data model."""

    def test_valid_zone_creation(self) -> None:
        zone = Zone(
            name="patio",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            type="outdoor",
        )
        assert zone.name == "patio"
        assert zone.type == "outdoor"
        assert len(zone.polygon) == 4

    def test_zone_requires_at_least_three_vertices(self) -> None:
        with pytest.raises(ValueError, match="at least 3 points"):
            Zone(name="invalid", polygon=[(0.0, 0.0), (10.0, 10.0)], type="test")


class TestPointInPolygon:
    """Tests for ray-casting point-in-polygon algorithm."""

    @pytest.fixture
    def square_polygon(self) -> list:
        return [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

    def test_point_inside(self, square_polygon: list) -> None:
        assert is_point_in_polygon((50.0, 50.0), square_polygon) is True

    def test_point_outside(self, square_polygon: list) -> None:
        assert is_point_in_polygon((150.0, 50.0), square_polygon) is False
        assert is_point_in_polygon((-10.0, 50.0), square_polygon) is False
        assert is_point_in_polygon((50.0, 150.0), square_polygon) is False
        assert is_point_in_polygon((50.0, -10.0), square_polygon) is False

    def test_point_on_vertex(self, square_polygon: list) -> None:
        assert is_point_in_polygon((0.0, 0.0), square_polygon) is True
        assert is_point_in_polygon((100.0, 100.0), square_polygon) is True

    def test_point_on_edge(self, square_polygon: list) -> None:
        assert is_point_in_polygon((50.0, 0.0), square_polygon) is True
        assert is_point_in_polygon((100.0, 50.0), square_polygon) is True
        assert is_point_in_polygon((50.0, 100.0), square_polygon) is True
        assert is_point_in_polygon((0.0, 50.0), square_polygon) is True

    def test_polygon_fewer_than_three_points(self) -> None:
        assert is_point_in_polygon((5.0, 5.0), [(0.0, 0.0), (10.0, 10.0)]) is False


class TestZoneManager:
    """Tests for ZoneManager operations, lookups, and configuration loading."""

    def test_add_and_get_zones(self, sample_zones: list) -> None:
        manager = ZoneManager()
        for z in sample_zones:
            manager.add_zone(z)

        zones = manager.get_zones()
        assert len(zones) == 2
        assert zones[0].name == "front_door"
        assert zones[1].name == "kitchen"

        assert manager.get_zone("front_door") is not None
        assert manager.get_zone("front_door").name == "front_door"
        assert manager.get_zone("non_existent") is None

    def test_point_in_zone_lookup(self, sample_zones: list) -> None:
        manager = ZoneManager(sample_zones)

        # Inside front_door [0..100, 0..100]
        match = manager.point_in_zone((50.0, 50.0))
        assert match is not None
        assert match.name == "front_door"

        # Inside kitchen [200..400, 100..300]
        match = manager.point_in_zone((250.0, 150.0))
        assert match is not None
        assert match.name == "kitchen"

        # Outside all zones
        assert manager.point_in_zone((150.0, 50.0)) is None
        assert manager.point_in_zone((500.0, 500.0)) is None

    def test_overlapping_zones_first_matching_order(self) -> None:
        """Constraint: If a point belongs to multiple zones, return the first matching zone according to configured zone order."""
        zone1 = Zone(name="zone_alpha", polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)], type="a")
        zone2 = Zone(name="zone_beta", polygon=[(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)], type="b")

        # Config order: alpha, then beta
        manager_1 = ZoneManager([zone1, zone2])
        # Point (75, 75) is in both
        res_1 = manager_1.point_in_zone((75.0, 75.0))
        assert res_1 is not None
        assert res_1.name == "zone_alpha"

        # Config order: beta, then alpha
        manager_2 = ZoneManager([zone2, zone1])
        res_2 = manager_2.point_in_zone((75.0, 75.0))
        assert res_2 is not None
        assert res_2.name == "zone_beta"

    def test_track_in_zone(self, sample_zones: list) -> None:
        manager = ZoneManager(sample_zones)

        # Coordinate point
        match_point = manager.track_in_zone((50.0, 50.0))
        assert match_point is not None
        assert match_point.name == "front_door"

        # Bounding box (center will be at (50, 50))
        match_bbox = manager.track_in_zone((20.0, 20.0, 80.0, 80.0))
        assert match_bbox is not None
        assert match_bbox.name == "front_door"

        # Bounding box outside
        assert manager.track_in_zone((600.0, 600.0, 700.0, 700.0)) is None

        # Invalid tuple length
        with pytest.raises(ValueError, match="Expected 2-tuple .* or 4-tuple"):
            manager.track_in_zone((1.0, 2.0, 3.0))  # type: ignore[arg-type]

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        config_data = {
            "zones": {
                "porch": {
                    "type": "entry",
                    "points": [[0, 0], [50, 0], [50, 50], [0, 50]],
                },
                "hallway": {
                    "type": "transit",
                    "points": [[60, 0], [100, 0], [100, 100], [60, 100]],
                },
            }
        }
        yaml_file = tmp_path / "test_zones.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        manager = ZoneManager()
        manager.load_from_yaml(yaml_file)

        assert len(manager.get_zones()) == 2
        assert manager.get_zone("porch") is not None
        assert manager.get_zone("porch").type == "entry"
        assert manager.get_zone("hallway") is not None

        # Containment verification
        assert manager.point_in_zone((25.0, 25.0)).name == "porch"
        assert manager.point_in_zone((80.0, 50.0)).name == "hallway"

    def test_load_from_yaml_non_existent(self) -> None:
        manager = ZoneManager()
        with pytest.raises(FileNotFoundError):
            manager.load_from_yaml("non_existent_zones.yaml")

    def test_architecture_independence(self) -> None:
        """Verify that zones module does not import inference, ultralytics, or cv2."""
        import zones.zone_manager as zm
        module_source = Path(zm.__file__).read_text(encoding="utf-8")
        assert "inference" not in module_source
        assert "ultralytics" not in module_source
        assert "cv2" not in module_source
        assert "Detection" not in module_source
