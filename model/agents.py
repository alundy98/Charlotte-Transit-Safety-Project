from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import mesa_geo as mg

from .loaders import RouteStop


class StationAgent(mg.GeoAgent):
    def __init__(self, model, geometry, crs, station_name: str, display_name: str, risk_profile, properties: dict):
        super().__init__(model, geometry, crs)
        self.station_name = station_name
        self.display_name = display_name
        self.risk_profile = risk_profile
        self.properties = properties

    def step(self) -> None:
        return


class PatrolAgent(mg.GeoAgent):
    def __init__(self, model, geometry, crs, route_name: str, route: List[RouteStop], station_lookup: Dict[str, StationAgent]):
        super().__init__(model, geometry, crs)
        self.route_name = route_name
        self.route = route
        self.station_lookup = station_lookup
        self.route_index = 0
        self.current_station: Optional[str] = route[0].station_name
        self.route_goal = route[0].route_goal
        self.current_mode = "dwell"
        self.dwell_remaining = max(0, route[0].dwell_steps - 1)
        self.travel_remaining = 0
        self.last_station = route[0].station_name
        self.next_station = route[0].station_name

    def neighbor_stations(self) -> set[str]:
        if self.current_mode == "travel":
            return {self.last_station, self.next_station}
        idx = self.route_index
        return {
            self.route[(idx - 1) % len(self.route)].station_name,
            self.route[(idx + 1) % len(self.route)].station_name,
        }

    def step(self) -> None:
        if self.current_mode == "dwell" and self.dwell_remaining > 0:
            self.dwell_remaining -= 1
            return

        if self.current_mode == "travel" and self.travel_remaining > 0:
            self.travel_remaining -= 1
            if self.travel_remaining > 0:
                return
            self.route_index = (self.route_index + 1) % len(self.route)
            current_stop = self.route[self.route_index]
            self.current_station = current_stop.station_name
            self.geometry = self.station_lookup[self.current_station].geometry
            self.current_mode = "dwell"
            self.dwell_remaining = max(0, current_stop.dwell_steps - 1)
            return

        next_index = (self.route_index + 1) % len(self.route)
        next_stop = self.route[next_index]
        self.last_station = self.route[self.route_index].station_name
        self.next_station = next_stop.station_name

        if next_stop.travel_steps <= 0:
            self.route_index = next_index
            self.current_station = next_stop.station_name
            self.geometry = self.station_lookup[self.current_station].geometry
            self.current_mode = "dwell"
            self.dwell_remaining = max(0, next_stop.dwell_steps - 1)
            return

        self.current_mode = "travel"
        self.travel_remaining = next_stop.travel_steps
        self.current_station = None
        start = self.station_lookup[self.last_station].geometry
        end = self.station_lookup[self.next_station].geometry
        self.geometry = type(start)((start.x + end.x) / 2, (start.y + end.y) / 2)


class PassengerAgent(mg.GeoAgent):
    def __init__(
        self,
        model,
        geometry,
        crs,
        origin: str,
        destination: str,
        station_sequence: List[str],
        station_lookup: Dict[str, StationAgent],
        offending_propensity: float,
        guardianship_sensitivity: float,
        stress_level: float,
        trip_type: str,
    ):
        super().__init__(model, geometry, crs)
        self.origin = origin
        self.destination = destination
        self.station_sequence = station_sequence
        self.station_lookup = station_lookup
        self.current_station = origin
        self.current_index = station_sequence.index(origin)
        self.destination_index = station_sequence.index(destination)
        self.exposure_score = 0.0
        self.offending_propensity = offending_propensity
        self.guardianship_sensitivity = guardianship_sensitivity
        self.stress_level = stress_level
        self.trip_type = trip_type
        self.incident_cooldown = 0
        self.incidents_committed = 0
        self.completed = False

    def step(self) -> None:
        if self.completed:
            return
        if self.incident_cooldown > 0:
            self.incident_cooldown -= 1
        if self.current_index == self.destination_index:
            self.completed = True
            return

        direction = 1 if self.destination_index > self.current_index else -1
        self.current_index += direction
        self.current_station = self.station_sequence[self.current_index]
        self.geometry = self.station_lookup[self.current_station].geometry
        if self.current_index == self.destination_index:
            self.completed = True


@dataclass
class RouteSimulationSummary:
    route_name: str
    route_goal: str
    attempted_incidents: int
    blocked_incidents: int
    realized_incidents: int
    average_passenger_exposure: float
    patrol_coverage_ratio: float
    high_risk_station_visits: int
