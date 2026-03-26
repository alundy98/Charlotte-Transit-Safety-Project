from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from shapely.geometry import Point, shape

from .risk import canonical_station_name, normalize_station_name, safe_float


def parse_station_list(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return re.findall(r"'([^']+)'", raw_value)


def parse_distance_list(raw_value: str) -> List[float]:
    if not raw_value:
        return []
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw_value)]


@dataclass
class StationRecord:
    canonical_name: str
    display_name: str
    geometry: Point
    properties: dict


@dataclass
class CrimeRecord:
    incident_id: str
    station_name: str
    geometry: Point
    properties: dict
    offense: str
    hour: Optional[int]
    is_weekend: bool


@dataclass
class RouteStop:
    route_id: str
    route_name: str
    station_name: str
    stop_order: int
    dwell_steps: int
    travel_steps: int
    route_goal: str


def read_geojson(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_station_records(path: Path) -> Dict[str, StationRecord]:
    payload = read_geojson(path)
    records: Dict[str, StationRecord] = {}
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        raw_name = properties.get("NAME") or properties.get("name") or properties.get("station_name")
        if not raw_name:
            continue
        geometry = shape(feature["geometry"])
        canonical_name = canonical_station_name(str(raw_name))
        records[canonical_name] = StationRecord(
            canonical_name=canonical_name,
            display_name=str(raw_name).strip(),
            geometry=geometry if isinstance(geometry, Point) else geometry.centroid,
            properties=properties,
        )
    return records


def choose_station_name(properties: dict) -> str:
    stations = parse_station_list(str(properties.get("stations_in_radius", "")))
    distances = parse_distance_list(str(properties.get("stations_in_radius_dist_m", "")))
    if stations and distances:
        return stations[min(range(min(len(stations), len(distances))), key=lambda idx: distances[idx])]
    for key in ("nearest_station", "name", "station_name"):
        value = properties.get(key)
        if value:
            return str(value)
    return ""


def infer_hour(properties: dict) -> Optional[int]:
    hour_value = properties.get("hour")
    if hour_value in ("", None):
        return None
    try:
        hour = int(float(hour_value))
    except (TypeError, ValueError):
        return None
    return hour if 0 <= hour <= 23 else None


def infer_weekend(properties: dict) -> bool:
    label = normalize_station_name(str(properties.get("day_type", "")))
    return label in {"weekend", "saturday", "sunday"}


def build_supplemental_station_records(crime_records: Iterable[CrimeRecord]) -> Dict[str, StationRecord]:
    grouped: Dict[str, list[Point]] = defaultdict(list)
    for record in crime_records:
        grouped[record.station_name].append(record.geometry)

    supplemental: Dict[str, StationRecord] = {}
    for station_name, points in grouped.items():
        if not points:
            continue
        lon = sum(point.x for point in points) / len(points)
        lat = sum(point.y for point in points) / len(points)
        supplemental[station_name] = StationRecord(
            canonical_name=station_name,
            display_name=station_name.title(),
            geometry=Point(lon, lat),
            properties={"synthetic": True},
        )
    return supplemental


def load_crime_records(path: Path) -> List[CrimeRecord]:
    payload = read_geojson(path)
    deduped: Dict[str, CrimeRecord] = {}
    fallback_records: List[CrimeRecord] = []

    for index, feature in enumerate(payload.get("features", [])):
        properties = feature.get("properties", {})
        chosen_station = choose_station_name(properties)
        if not chosen_station:
            continue

        geometry = shape(feature["geometry"])
        point = geometry if isinstance(geometry, Point) else geometry.centroid
        record = CrimeRecord(
            incident_id=str(properties.get("INCIDENT_REPORT_ID") or f"feature-{index}"),
            station_name=canonical_station_name(chosen_station),
            geometry=point,
            properties=properties,
            offense=str(properties.get("HIGHEST_NIBRS_DESCRIPTION", "")),
            hour=infer_hour(properties),
            is_weekend=infer_weekend(properties),
        )
        if properties.get("INCIDENT_REPORT_ID"):
            deduped.setdefault(record.incident_id, record)
        else:
            fallback_records.append(record)

    return list(deduped.values()) + fallback_records


def load_route_definitions(path: Optional[Path]) -> Dict[str, List[RouteStop]]:
    if path is None or not path.exists():
        return {}

    payload = read_geojson(path)
    grouped: Dict[str, list[RouteStop]] = defaultdict(list)
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        route_id = str(properties.get("route_id") or properties.get("route_name") or "").strip()
        station_name = str(properties.get("station_name") or properties.get("stop_name") or "").strip()
        stop_order = int(properties.get("stop_order", 0))
        if route_id and station_name:
            grouped[route_id].append(
                RouteStop(
                    route_id=route_id,
                    route_name=str(properties.get("route_name") or route_id),
                    station_name=canonical_station_name(station_name),
                    stop_order=stop_order,
                    dwell_steps=max(1, int(properties.get("dwell_steps", 1))),
                    travel_steps=max(0, int(properties.get("travel_steps", 1))),
                    route_goal=str(properties.get("route_goal") or properties.get("time_period") or "unspecified"),
                )
            )

    return {
        route_id: sorted(stops, key=lambda item: item.stop_order)
        for route_id, stops in grouped.items()
    }
