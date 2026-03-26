from __future__ import annotations

import random
from collections import Counter
from statistics import mean
from typing import Dict, List, Optional

import mesa_geo as mg
from mesa import Model
from mesa.datacollection import DataCollector

from .agents import PassengerAgent, PatrolAgent, RouteSimulationSummary, StationAgent
from .config import CharlotteGeoConfig
from .loaders import (
    RouteStop,
    build_supplemental_station_records,
    load_crime_records,
    load_route_definitions,
    load_station_records,
)
from .risk import StationRiskProfile, build_station_risk_profiles


def weighted_choice(rng: random.Random, items: List[str], weights: Dict[str, float]) -> str:
    total = sum(max(weights.get(item, 0.0), 0.0) for item in items)
    if total <= 0:
        return rng.choice(items)
    pick = rng.random() * total
    running = 0.0
    for item in items:
        running += max(weights.get(item, 0.0), 0.0)
        if running >= pick:
            return item
    return items[-1]


class CharlotteCrimeGeoModel(Model):
    def __init__(self, config: Optional[CharlotteGeoConfig] = None, route_name: Optional[str] = None):
        super().__init__()
        self.config = config or CharlotteGeoConfig()
        self.random = random.Random(self.config.seed)
        self.space = mg.GeoSpace(warn_crs_conversion=False)
        self.crs = "EPSG:4326"

        crime_records = load_crime_records(self.config.crime_geojson)
        self.crime_records = crime_records
        station_records = load_station_records(self.config.station_geojson)
        for name, record in build_supplemental_station_records(crime_records).items():
            station_records.setdefault(name, record)

        risk_profiles = build_station_risk_profiles(station_records.keys(), [record.__dict__ for record in crime_records])
        self.station_agents = self._build_station_agents(station_records, risk_profiles)
        self.station_order = self._ordered_station_names()

        all_routes = load_route_definitions(self.config.patrol_routes_geojson)
        if not all_routes:
            all_routes = self.build_demo_routes()
        self.patrol_routes = {
            name: [stop for stop in route if stop.station_name in self.station_agents]
            for name, route in all_routes.items()
            if route
        }
        self.route_name = route_name or next(iter(self.patrol_routes))

        starting_station = self.station_agents[self.patrol_routes[self.route_name][0].station_name]
        self.patrol_agent = PatrolAgent(
            self,
            starting_station.geometry,
            self.crs,
            route_name=self.route_name,
            route=self.patrol_routes[self.route_name],
            station_lookup=self.station_agents,
        )
        self.space.add_agents(self.patrol_agent)
        self.passenger_agents: List[PassengerAgent] = []
        self.current_step = 0

        self.attempted_incidents = 0
        self.blocked_incidents = 0
        self.cumulative_exposure = 0.0
        self.station_visits: Counter[str] = Counter()
        self.high_risk_visits = 0
        self.realized_incident_events: List[dict] = []

        self.datacollector = DataCollector(
            model_reporters={
                "attempted_incidents": lambda m: m.attempted_incidents,
                "blocked_incidents": lambda m: m.blocked_incidents,
                "active_passengers": lambda m: sum(not agent.completed for agent in m.passenger_agents),
            }
        )
        self.datacollector.collect(self)

    def _build_station_agents(
        self,
        station_records,
        risk_profiles: Dict[str, StationRiskProfile],
    ) -> Dict[str, StationAgent]:
        agents: Dict[str, StationAgent] = {}
        for station_name, record in station_records.items():
            agent = StationAgent(
                self,
                record.geometry,
                self.crs,
                station_name=station_name,
                display_name=record.display_name,
                risk_profile=risk_profiles[station_name],
                properties=record.properties,
            )
            agents[station_name] = agent
        self.space.add_agents(list(agents.values()))
        return agents

    def _ordered_station_names(self) -> List[str]:
        return sorted(
            self.station_agents.keys(),
            key=lambda name: (
                self.station_agents[name].geometry.y,
                self.station_agents[name].geometry.x,
                name,
            ),
        )

    def build_demo_routes(self) -> Dict[str, List[RouteStop]]:
        stations = self._ordered_station_names()
        if len(stations) < 6:
            return {
                "baseline_loop": [
                    RouteStop("baseline_loop", "Baseline Loop", station, index + 1, 1, 1, "baseline")
                    for index, station in enumerate(stations)
                ]
            }
        def make_route(route_id: str, route_name: str, route_goal: str, names: List[str]) -> List[RouteStop]:
            return [
                RouteStop(route_id, route_name, station_name, index + 1, 1, 1, route_goal)
                for index, station_name in enumerate(names)
            ]
        return {
            "baseline_loop": make_route("baseline_loop", "Baseline Loop", "baseline", stations[:8]),
            "center_city_focus": make_route(
                "center_city_focus",
                "Center City Focus",
                "coverage",
                [station for station in stations if station in {
                    "7th st",
                    "9th street",
                    "ctc/arena",
                    "3rd street/convention center",
                    "bland st",
                    "carson",
                    "brooklyn village",
                    "east/west",
                }],
            ),
            "full_corridor": make_route("full_corridor", "Full Corridor", "coverage", stations),
        }

    def station_risk(self, station_name: str, step_number: int) -> float:
        profile = self.station_agents[station_name].risk_profile
        hour = step_number % 24
        hourly_weight = profile.hourly_incident_share.get(hour, 1 / 24) * 24
        peak_multiplier = 1.0
        if hour in {7, 8, 9, 16, 17, 18, 19, 22, 23}:
            peak_multiplier += self.config.peak_hour_bonus
        elif hour in {0, 1, 2, 3, 4, 5}:
            peak_multiplier += 0.10
        return profile.baseline_risk * hourly_weight * peak_multiplier

    def spawn_passengers(self) -> None:
        weights = {name: agent.risk_profile.attractiveness for name, agent in self.station_agents.items()}
        for _ in range(self.config.passengers_per_step):
            origin = weighted_choice(self.random, self.station_order, weights)
            destination_pool = [station for station in self.station_order if station != origin]
            destination = weighted_choice(self.random, destination_pool, weights)
            trip_type = weighted_choice(
                self.random,
                ["commute", "errand", "leisure"],
                {"commute": 0.45, "errand": 0.30, "leisure": 0.25},
            )
            agent = PassengerAgent(
                self,
                self.station_agents[origin].geometry,
                self.crs,
                origin=origin,
                destination=destination,
                station_sequence=self.station_order,
                station_lookup=self.station_agents,
                offending_propensity=self.random.uniform(0.05, 0.95),
                guardianship_sensitivity=self.random.uniform(0.20, 0.95),
                stress_level=self.random.uniform(0.10, 0.70),
                trip_type=trip_type,
            )
            self.passenger_agents.append(agent)
            self.space.add_agents(agent)

    def passenger_incident_probability(self, passenger: PassengerAgent, step_number: int) -> float:
        station_name = passenger.current_station
        station_risk = self.station_risk(station_name, step_number)
        profile = self.station_agents[station_name].risk_profile
        patrol_here = self.patrol_agent.current_station == station_name
        patrol_nearby = station_name in self.patrol_agent.neighbor_stations()
        hour = step_number % 24
        calibration = self.config.calibration

        environmental_pressure = min(1.0, station_risk / calibration.environment_divisor)
        exposure_pressure = min(1.0, passenger.exposure_score / calibration.exposure_divisor)
        late_night_bonus = calibration.late_night_bonus if hour in {21, 22, 23, 0, 1, 2} else 0.0
        leisure_bonus = calibration.leisure_bonus if passenger.trip_type == "leisure" else 0.0
        weekend_bonus = calibration.weekend_weight * profile.weekend_share

        raw_score = (
            calibration.propensity_weight * passenger.offending_propensity
            + calibration.stress_weight * passenger.stress_level
            + calibration.environment_weight * environmental_pressure
            + calibration.exposure_weight * exposure_pressure
            + late_night_bonus
            + leisure_bonus
            + weekend_bonus
        )

        guardianship_penalty = 0.0
        if patrol_here:
            guardianship_penalty += calibration.patrol_here_penalty * passenger.guardianship_sensitivity
        elif patrol_nearby:
            guardianship_penalty += calibration.patrol_nearby_penalty * passenger.guardianship_sensitivity

        probability = max(0.0, min(calibration.probability_cap, raw_score - guardianship_penalty))
        if passenger.incident_cooldown > 0:
            return 0.0
        return probability

    def evaluate_passenger_incident(self, passenger: PassengerAgent, step_number: int) -> None:
        if passenger.completed:
            return

        station_name = passenger.current_station
        patrol_here = self.patrol_agent.current_station == station_name
        patrol_nearby = station_name in self.patrol_agent.neighbor_stations()
        probability = self.passenger_incident_probability(passenger, step_number)

        if self.random.random() > probability:
            return

        self.attempted_incidents += 1
        if patrol_here and self.random.random() <= self.config.patrol_effectiveness:
            self.blocked_incidents += 1
            passenger.stress_level = min(1.0, passenger.stress_level + 0.02)
            passenger.incident_cooldown = 2
            return
        if patrol_nearby and self.random.random() <= self.config.nearby_patrol_effectiveness:
            self.blocked_incidents += 1
            passenger.incident_cooldown = 1
            return

        passenger.incidents_committed += 1
        passenger.incident_cooldown = 4
        passenger.stress_level = max(0.05, passenger.stress_level - 0.05)
        self.realized_incident_events.append(
            {
                "station_name": station_name,
                "hour": step_number % 24,
                "trip_type": passenger.trip_type,
            }
        )

    def step(self) -> None:
        current_step = self.current_step
        if self.patrol_agent.current_station is not None:
            self.station_visits[self.patrol_agent.current_station] += 1

        avg_risk = mean(agent.risk_profile.baseline_risk for agent in self.station_agents.values())
        if (
            self.patrol_agent.current_station is not None
            and self.station_agents[self.patrol_agent.current_station].risk_profile.baseline_risk >= avg_risk * 1.2
        ):
            self.high_risk_visits += 1

        self.spawn_passengers()

        for passenger in self.passenger_agents:
            if passenger.completed:
                continue
            risk_value = self.station_risk(passenger.current_station, current_step)
            passenger.exposure_score += risk_value
            passenger.stress_level = min(
                1.0,
                passenger.stress_level
                + min(
                    self.config.calibration.stress_increment_cap,
                    risk_value / self.config.calibration.stress_divisor,
                ),
            )
            self.cumulative_exposure += risk_value
            self.evaluate_passenger_incident(passenger, current_step)
            passenger.step()

        self.patrol_agent.step()
        self.current_step += 1
        self.datacollector.collect(self)

    def run(self) -> RouteSimulationSummary:
        for _ in range(self.config.steps):
            self.step()

        total_passengers = max(len(self.passenger_agents), 1)
        realized_incidents = self.attempted_incidents - self.blocked_incidents
        return RouteSimulationSummary(
            route_name=self.route_name,
            route_goal=self.patrol_agent.route_goal,
            attempted_incidents=self.attempted_incidents,
            blocked_incidents=self.blocked_incidents,
            realized_incidents=realized_incidents,
            average_passenger_exposure=self.cumulative_exposure / total_passengers,
            patrol_coverage_ratio=len(self.station_visits) / len(self.station_agents),
            high_risk_station_visits=self.high_risk_visits,
        )

    def observed_incident_distribution(self) -> dict:
        station_counts = Counter(record.station_name for record in self.crime_records)
        hour_counts = Counter(record.hour for record in self.crime_records if record.hour is not None)
        return {"station_counts": station_counts, "hour_counts": hour_counts}

    def simulated_incident_distribution(self) -> dict:
        station_counts = Counter(event["station_name"] for event in self.realized_incident_events)
        hour_counts = Counter(event["hour"] for event in self.realized_incident_events)
        return {"station_counts": station_counts, "hour_counts": hour_counts}
