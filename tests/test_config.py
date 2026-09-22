"""Unit tests for the Tower configuration module."""

from pathlib import Path
import pytest
import yaml

from config import DetectorConfig, TowerConfig, ZonesConfig


def test_detector_config_defaults() -> None:
    cfg = DetectorConfig()
    assert cfg.model_path == "yolov8n.pt"
    assert cfg.device == "auto"
    assert cfg.confidence_threshold == 0.25


def test_detector_config_validation() -> None:
    with pytest.raises(ValueError, match="confidence_threshold must be between"):
        DetectorConfig(confidence_threshold=1.5)

    with pytest.raises(ValueError, match="confidence_threshold must be between"):
        DetectorConfig(confidence_threshold=-0.1)

    with pytest.raises(ValueError, match="Invalid device"):
        DetectorConfig(device="tpu")


def test_tower_config_from_dict() -> None:
    data = {
        "detector": {
            "model_path": "custom_yolo.pt",
            "device": "cpu",
            "confidence_threshold": 0.4,
        },
        "zones": {
            "zone_config_path": "custom_zones.yaml",
        },
    }
    cfg = TowerConfig.from_dict(data)
    assert cfg.detector.model_path == "custom_yolo.pt"
    assert cfg.detector.device == "cpu"
    assert cfg.detector.confidence_threshold == 0.4
    assert cfg.zones.zone_config_path == "custom_zones.yaml"


def test_tower_config_load_from_yaml(tmp_path: Path) -> None:
    data = {
        "detector": {
            "model_path": "yolov8n.pt",
            "device": "cpu",
            "confidence_threshold": 0.35,
        }
    }
    yaml_file = tmp_path / "tower.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    cfg = TowerConfig.load_from_yaml(yaml_file)
    assert cfg.detector.confidence_threshold == 0.35
    assert cfg.zones.zone_config_path == "configs/zones.yaml"


def test_tower_config_load_non_existent_returns_defaults() -> None:
    cfg = TowerConfig.load_from_yaml("does_not_exist.yaml")
    assert cfg.detector.model_path == "yolov8n.pt"
    assert cfg.detector.confidence_threshold == 0.25
