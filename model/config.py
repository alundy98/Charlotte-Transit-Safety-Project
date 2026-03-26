from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CalibrationConfig:
    propensity_weight: float = 0.30
    stress_weight: float = 0.20
    environment_weight: float = 0.25
    exposure_weight: float = 0.15
    late_night_bonus: float = 0.10
    leisure_bonus: float = 0.05
    weekend_weight: float = 0.05
    patrol_here_penalty: float = 0.45
    patrol_nearby_penalty: float = 0.20
    probability_cap: float = 0.55
    stress_increment_cap: float = 0.08
    stress_divisor: float = 120.0
    exposure_divisor: float = 25.0
    environment_divisor: float = 12.0


@dataclass
class CharlotteGeoConfig:
    station_geojson: Path = ROOT / "stationINFO.geojson"
    crime_geojson: Path = ROOT / "crimes_with_tree_features_clean.geojson"
    patrol_routes_geojson: Optional[Path] = None
    steps: int = 48
    passengers_per_step: int = 12
    patrol_effectiveness: float = 0.75
    nearby_patrol_effectiveness: float = 0.35
    peak_hour_bonus: float = 0.25
    seed: int = 7
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
