"""Configuration module for Tower.

Provides lightweight, dataclass-based configuration for detector and zone settings.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration options for object detection backend."""
    model_path: str = "yolov8n.pt"
    device: str = "auto"
    confidence_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )
        valid_devices = {"auto", "cpu", "cuda"}
        if self.device.lower() not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(
                f"Invalid device '{self.device}'. Expected one of {valid_devices} or 'cuda:<idx>'."
            )


@dataclass
class ZonesConfig:
    """Configuration options for spatial zones."""
    zone_config_path: str = "configs/zones.yaml"


@dataclass
class TowerConfig:
    """Top-level Tower system configuration."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TowerConfig":
        """Load TowerConfig from a nested dictionary."""
        detector_data = data.get("detector", {})
        zones_data = data.get("zones", {})

        detector_cfg = DetectorConfig(**detector_data) if detector_data else DetectorConfig()
        zones_cfg = ZonesConfig(**zones_data) if zones_data else ZonesConfig()

        return cls(detector=detector_cfg, zones=zones_cfg)

    @classmethod
    def load_from_yaml(cls, config_path: Union[str, Path]) -> "TowerConfig":
        """Load TowerConfig from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Config file %s does not exist. Using defaults.", path)
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)
