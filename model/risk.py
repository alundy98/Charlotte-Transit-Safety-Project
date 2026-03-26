from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, Optional


def safe_float(value: object) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_station_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


STATION_ALIASES = {
    "7th street": "7th st",
    "bland": "bland st",
    "charlotte transportation center/arena": "ctc/arena",
    "3rd st/convention": "3rd street/convention center",
}


def canonical_station_name(name: str) -> str:
    normalized = normalize_station_name(name)
    return STATION_ALIASES.get(normalized, normalized)


def crime_severity(offense: str) -> float:
    label = (offense or "").lower()
    if any(token in label for token in ("homicide", "kidnapping", "weapon")):
        return 1.0
    if any(token in label for token in ("assault", "robbery")):
        return 0.85
    if any(token in label for token in ("theft", "burglary", "motor vehicle")):
        return 0.55
    return 0.35


@dataclass
class StationRiskProfile:
    incident_count: int
    assault_count: int
    violent_share: float
    average_visibility: float
    average_precipitation: float
    average_tree_density: float
    attractiveness: float
    baseline_risk: float
    hourly_incident_share: Dict[int, float]
    weekend_share: float


def build_station_risk_profiles(
    station_names: Iterable[str],
    crime_records: Iterable[dict],
) -> Dict[str, StationRiskProfile]:
    station_keys = list(station_names)
    incident_counts: Dict[str, int] = Counter({name: 0 for name in station_keys})
    assault_counts: Dict[str, int] = Counter()
    violent_counts: Dict[str, int] = Counter()
    weekend_counts: Dict[str, int] = Counter()
    hourly_counts: Dict[str, Counter] = defaultdict(Counter)
    visibility_values: Dict[str, list[float]] = defaultdict(list)
    precip_values: Dict[str, list[float]] = defaultdict(list)
    tree_values: Dict[str, list[float]] = defaultdict(list)

    for record in crime_records:
        station_name = record["station_name"]
        incident_counts[station_name] += 1

        offense = str(record.get("offense", "")).lower()
        if "assault" in offense:
            assault_counts[station_name] += 1
        if any(token in offense for token in ("assault", "robbery", "weapon", "homicide", "kidnapping")):
            violent_counts[station_name] += 1

        hour = record.get("hour")
        if isinstance(hour, int) and 0 <= hour <= 23:
            hourly_counts[station_name][hour] += 1

        if record.get("is_weekend"):
            weekend_counts[station_name] += 1

        visibility = safe_float(record.get("visibility"))
        precip = safe_float(record.get("precip"))
        trees = safe_float(record.get("trees_within_25m"))
        if visibility is not None:
            visibility_values[station_name].append(visibility)
        if precip is not None:
            precip_values[station_name].append(precip)
        if trees is not None:
            tree_values[station_name].append(trees)

    profiles: Dict[str, StationRiskProfile] = {}
    for station_name in station_keys:
        count = incident_counts[station_name]
        visibility_avg = mean(visibility_values[station_name]) if visibility_values[station_name] else 10.0
        precip_avg = mean(precip_values[station_name]) if precip_values[station_name] else 0.0
        tree_avg = mean(tree_values[station_name]) if tree_values[station_name] else 0.0
        violent_share = violent_counts[station_name] / count if count else 0.0
        hourly_total = sum(hourly_counts[station_name].values())
        hourly_share = (
            {hour: hourly_counts[station_name][hour] / hourly_total for hour in range(24)}
            if hourly_total
            else {hour: 1 / 24 for hour in range(24)}
        )
        weekend_share = weekend_counts[station_name] / count if count else 0.0
        attractiveness = max(1.0, math.log1p(count))
        baseline_risk = (
            math.log1p(count)
            + 0.9 * (assault_counts[station_name] / count if count else 0.0)
            + 0.5 * violent_share
            + max(0.0, 10.0 - visibility_avg) * 0.08
            + precip_avg * 0.03
            + tree_avg * 0.01
        )
        profiles[station_name] = StationRiskProfile(
            incident_count=count,
            assault_count=assault_counts[station_name],
            violent_share=violent_share,
            average_visibility=visibility_avg,
            average_precipitation=precip_avg,
            average_tree_density=tree_avg,
            attractiveness=attractiveness,
            baseline_risk=baseline_risk if count else 0.5,
            hourly_incident_share=hourly_share,
            weekend_share=weekend_share,
        )
    return profiles
