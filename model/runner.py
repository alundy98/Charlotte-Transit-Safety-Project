from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

from .agents import RouteSimulationSummary
from .config import CalibrationConfig, CharlotteGeoConfig
from .model import CharlotteCrimeGeoModel


def parse_args() -> argparse.Namespace:
    defaults = CharlotteGeoConfig()
    parser = argparse.ArgumentParser(description="GeoJSON-first Charlotte crime simulation using Mesa-Geo.")
    parser.add_argument("--stations", type=Path, default=defaults.station_geojson)
    parser.add_argument("--crimes", type=Path, default=defaults.crime_geojson)
    parser.add_argument("--routes", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--passengers-per-step", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed-step", type=int, default=17)
    parser.add_argument("--propensity-weight", type=float, default=defaults.calibration.propensity_weight)
    parser.add_argument("--stress-weight", type=float, default=defaults.calibration.stress_weight)
    parser.add_argument("--environment-weight", type=float, default=defaults.calibration.environment_weight)
    parser.add_argument("--exposure-weight", type=float, default=defaults.calibration.exposure_weight)
    parser.add_argument("--late-night-bonus", type=float, default=defaults.calibration.late_night_bonus)
    parser.add_argument("--leisure-bonus", type=float, default=defaults.calibration.leisure_bonus)
    parser.add_argument("--weekend-weight", type=float, default=defaults.calibration.weekend_weight)
    parser.add_argument("--patrol-here-penalty", type=float, default=defaults.calibration.patrol_here_penalty)
    parser.add_argument("--patrol-nearby-penalty", type=float, default=defaults.calibration.patrol_nearby_penalty)
    parser.add_argument("--probability-cap", type=float, default=defaults.calibration.probability_cap)
    return parser.parse_args()


def format_results(results: List[RouteSimulationSummary]) -> str:
    lines = [
        "route,route_goal,attempted_incidents,blocked_incidents,realized_incidents,avg_passenger_exposure,coverage_ratio,high_risk_visits"
    ]
    for result in results:
        lines.append(
            ",".join(
                [
                    result.route_name,
                    result.route_goal,
                    str(result.attempted_incidents),
                    str(result.blocked_incidents),
                    str(result.realized_incidents),
                    f"{result.average_passenger_exposure:.3f}",
                    f"{result.patrol_coverage_ratio:.3f}",
                    str(result.high_risk_station_visits),
                ]
            )
        )
    return "\n".join(lines)


def normalized_share(counter: Counter, labels: List) -> dict:
    total = sum(counter.values())
    if total <= 0:
        return {label: 0.0 for label in labels}
    return {label: counter.get(label, 0) / total for label in labels}


def build_validation_report(model: CharlotteCrimeGeoModel, result: RouteSimulationSummary) -> dict:
    observed = model.observed_incident_distribution()
    simulated = model.simulated_incident_distribution()

    station_labels = sorted(set(observed["station_counts"]) | set(simulated["station_counts"]))
    hour_labels = list(range(24))
    observed_station_share = normalized_share(observed["station_counts"], station_labels)
    simulated_station_share = normalized_share(simulated["station_counts"], station_labels)
    observed_hour_share = normalized_share(observed["hour_counts"], hour_labels)
    simulated_hour_share = normalized_share(simulated["hour_counts"], hour_labels)

    station_mae = sum(abs(observed_station_share[label] - simulated_station_share[label]) for label in station_labels) / max(len(station_labels), 1)
    hour_mae = sum(abs(observed_hour_share[label] - simulated_hour_share[label]) for label in hour_labels) / 24

    return {
        "route_name": result.route_name,
        "route_goal": result.route_goal,
        "station_distribution_mae": round(station_mae, 6),
        "hour_distribution_mae": round(hour_mae, 6),
        "top_observed_stations": observed["station_counts"].most_common(5),
        "top_simulated_stations": simulated["station_counts"].most_common(5),
        "top_observed_hours": observed["hour_counts"].most_common(5),
        "top_simulated_hours": simulated["hour_counts"].most_common(5),
    }


def summarize_metric(values: List[float]) -> dict:
    return {
        "mean": round(mean(values), 6),
        "std": round(pstdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def aggregate_route_results(route_name: str, route_goal: str, runs: List[dict]) -> dict:
    metric_names = [
        "attempted_incidents",
        "blocked_incidents",
        "realized_incidents",
        "average_passenger_exposure",
        "patrol_coverage_ratio",
        "high_risk_station_visits",
        "station_distribution_mae",
        "hour_distribution_mae",
    ]
    return {
        "route_name": route_name,
        "route_goal": route_goal,
        "replicates": len(runs),
        "metrics": {
            metric: summarize_metric([run[metric] for run in runs])
            for metric in metric_names
        },
        "seeds": [run["seed"] for run in runs],
    }


def format_experiment_summary(experiment_summary: List[dict]) -> str:
    lines = [
        "route,route_goal,replicates,realized_mean,realized_std,blocked_mean,exposure_mean,station_mae_mean,hour_mae_mean"
    ]
    for item in experiment_summary:
        metrics = item["metrics"]
        lines.append(
            ",".join(
                [
                    item["route_name"],
                    item["route_goal"],
                    str(item["replicates"]),
                    f"{metrics['realized_incidents']['mean']:.3f}",
                    f"{metrics['realized_incidents']['std']:.3f}",
                    f"{metrics['blocked_incidents']['mean']:.3f}",
                    f"{metrics['average_passenger_exposure']['mean']:.3f}",
                    f"{metrics['station_distribution_mae']['mean']:.6f}",
                    f"{metrics['hour_distribution_mae']['mean']:.6f}",
                ]
            )
        )
    return "\n".join(lines)


def run_route_comparison(args: Optional[argparse.Namespace] = None) -> List[RouteSimulationSummary]:
    parsed = args or parse_args()
    calibration = CalibrationConfig(
        propensity_weight=parsed.propensity_weight,
        stress_weight=parsed.stress_weight,
        environment_weight=parsed.environment_weight,
        exposure_weight=parsed.exposure_weight,
        late_night_bonus=parsed.late_night_bonus,
        leisure_bonus=parsed.leisure_bonus,
        weekend_weight=parsed.weekend_weight,
        patrol_here_penalty=parsed.patrol_here_penalty,
        patrol_nearby_penalty=parsed.patrol_nearby_penalty,
        probability_cap=parsed.probability_cap,
    )
    config = CharlotteGeoConfig(
        station_geojson=parsed.stations,
        crime_geojson=parsed.crimes,
        patrol_routes_geojson=parsed.routes,
        steps=parsed.steps,
        passengers_per_step=parsed.passengers_per_step,
        seed=parsed.seed,
        calibration=calibration,
    )
    route_probe = CharlotteCrimeGeoModel(config=config)
    route_names = list(route_probe.patrol_routes.keys())
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    all_runs: List[dict] = []
    latest_results: List[RouteSimulationSummary] = []

    for replicate in range(parsed.replicates):
        run_seed = parsed.seed + (replicate * parsed.seed_step)
        replicate_config = CharlotteGeoConfig(
            station_geojson=config.station_geojson,
            crime_geojson=config.crime_geojson,
            patrol_routes_geojson=config.patrol_routes_geojson,
            steps=config.steps,
            passengers_per_step=config.passengers_per_step,
            patrol_effectiveness=config.patrol_effectiveness,
            nearby_patrol_effectiveness=config.nearby_patrol_effectiveness,
            peak_hour_bonus=config.peak_hour_bonus,
            seed=run_seed,
            calibration=config.calibration,
        )
        models = [CharlotteCrimeGeoModel(config=replicate_config, route_name=route_name) for route_name in route_names]
        results = [model.run() for model in models]
        latest_results = results
        validations = [build_validation_report(model, result) for model, result in zip(models, results)]
        for result, validation in zip(results, validations):
            all_runs.append(
                {
                    "replicate": replicate + 1,
                    "seed": run_seed,
                    "route_name": result.route_name,
                    "route_goal": result.route_goal,
                    "attempted_incidents": result.attempted_incidents,
                    "blocked_incidents": result.blocked_incidents,
                    "realized_incidents": result.realized_incidents,
                    "average_passenger_exposure": result.average_passenger_exposure,
                    "patrol_coverage_ratio": result.patrol_coverage_ratio,
                    "high_risk_station_visits": result.high_risk_station_visits,
                    "station_distribution_mae": validation["station_distribution_mae"],
                    "hour_distribution_mae": validation["hour_distribution_mae"],
                }
            )

    experiment_summary = [
        aggregate_route_results(
            route_name,
            all_runs_for_route[0]["route_goal"],
            all_runs_for_route,
        )
        for route_name, all_runs_for_route in {
            route_name: [run for run in all_runs if run["route_name"] == route_name]
            for route_name in route_names
        }.items()
    ]
    (cache_dir / "route_validation.json").write_text(
        json.dumps(
            {
                "calibration": asdict(calibration),
                "replicates": parsed.replicates,
                "seed_start": parsed.seed,
                "seed_step": parsed.seed_step,
                "summary": experiment_summary,
                "runs": all_runs,
            },
            indent=2,
        )
    )
    (cache_dir / "route_summary.csv").write_text(format_experiment_summary(experiment_summary))
    print(format_experiment_summary(experiment_summary))
    return latest_results


def main() -> None:
    run_route_comparison()


if __name__ == "__main__":
    main()
