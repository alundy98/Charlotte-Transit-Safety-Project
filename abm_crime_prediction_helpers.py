"""
ABM crime prediction helpers for the XGB Context + Location model.

This module is designed to be imported by the simulation script. It assumes the
trained model is an XGBoost regressor that predicts `target_log_rate`
(log crime rate per agent) and that a fitted sklearn preprocessor is saved
alongside it.

Core idea
---------
1. Build a one-row feature frame for the current station-hour context.
2. Predict log crime rate per agent with the fitted model.
3. Exponentiate to recover rate per agent.
4. Multiply by the actual number of agents present in the simulation step.
5. Apply deterrence / calibration adjustments.
6. Convert expected count (lambda) to a step event probability:
       P(at least one crime) = 1 - exp(-lambda)

Notes
-----
- Agent count is NOT a direct model input. It is handled outside the model as
  exposure in the simulation layer.
- CMPD presence is handled as a multiplicative deterrence adjustment on lambda.
- This file does not assume any specific Mesa model structure; it is written as
  generic helpers you can call from the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd


Number = Union[int, float, np.number]


@dataclass
class CrimePredictionConfig:
    """
    Runtime knobs for simulation-side conversion from model output to crime risk.
    """
    # Global multiplier used to calibrate simulation totals to historical totals.
    global_rate_multiplier: float = 1.0

    # If True, scale lambda by actual_agents / estimated_agents_from_training_context.
    # Usually False unless you explicitly pass estimated agents from the lookup table
    # and want relative scaling from the model context row.
    scale_by_agent_ratio: bool = False

    # Small positive floor to avoid divide-by-zero when scaling by estimated agents.
    min_estimated_agents_for_ratio: float = 1e-6

    # CMPD deterrence: each present officer multiplies lambda by this factor.
    # Example: 0.85 means each officer reduces lambda by 15%.
    cmpd_officer_multiplier: float = 0.85

    # Lower / upper clamp for the combined CMPD multiplier.
    min_cmpd_multiplier: float = 0.20
    max_cmpd_multiplier: float = 1.00

    # Optional multiplier for a specific location type / context not learned by model.
    # Keep at 1.0 if not needed.
    contextual_multiplier: float = 1.0

    # Safety clamp for the final event probability.
    min_event_probability: float = 0.0
    max_event_probability: float = 1.0

    # If your simulation step is not one hour, scale lambda by step_hours.
    step_hours: float = 1.0


@dataclass
class CrimePredictionResult:
    """
    Rich prediction output for one station-step or location-step.
    """
    pred_log_rate: float
    pred_rate_per_agent: float
    actual_agents: float
    base_lambda: float
    adjusted_lambda: float
    event_probability: float
    cmpd_multiplier: float
    applied_rate_multiplier: float
    feature_row: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_model_bundle(bundle_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a saved model bundle.

    Expected keys in the joblib bundle:
        - "model": fitted XGBRegressor
        - "preprocessor": fitted sklearn transformer
        - "features": list of raw feature names expected before preprocessing

    Optional keys:
        - "label"
        - "metadata"
    """
    bundle = joblib.load(bundle_path)
    required = {"model", "preprocessor", "features"}
    missing = required - set(bundle.keys())
    if missing:
        raise KeyError(
            f"Model bundle is missing required keys: {sorted(missing)}. "
            f"Found keys: {sorted(bundle.keys())}"
        )
    return bundle


def save_model_bundle(
    bundle_path: Union[str, Path],
    model: Any,
    preprocessor: Any,
    features: Sequence[str],
    label: str = "XGB Context + Location",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save the fitted model artifacts into one importable joblib bundle.
    """
    out = Path(bundle_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "preprocessor": preprocessor,
        "features": list(features),
        "label": label,
        "metadata": metadata or {},
    }
    joblib.dump(payload, out)
    return out


def cmpd_presence_multiplier(
    cmpd_agents_present: int,
    officer_multiplier: float = 0.85,
    min_multiplier: float = 0.20,
    max_multiplier: float = 1.00,
) -> float:
    """
    Convert CMPD presence into a multiplicative deterrence adjustment on lambda.

    Examples
    --------
    0 officers -> 1.0
    1 officer, officer_multiplier=0.85 -> 0.85
    2 officers -> 0.85^2 = 0.7225
    """
    n = max(int(cmpd_agents_present), 0)
    mult = officer_multiplier ** n
    return float(np.clip(mult, min_multiplier, max_multiplier))


def poisson_event_probability(expected_count: Number) -> float:
    """
    Probability of at least one event when count ~ Poisson(lambda).
    """
    lam = max(float(expected_count), 0.0)
    return float(1.0 - np.exp(-lam))


def sample_poisson_event_count(
    expected_count: Number,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Sample a nonnegative integer crime count for the step.
    """
    lam = max(float(expected_count), 0.0)
    rng = rng or np.random.default_rng()
    return int(rng.poisson(lam))


def sample_crime_occurrence(
    event_probability: Number,
    rng: Optional[np.random.Generator] = None,
) -> bool:
    """
    Bernoulli draw for whether at least one crime occurs in the step.
    """
    p = float(np.clip(event_probability, 0.0, 1.0))
    rng = rng or np.random.default_rng()
    return bool(rng.random() < p)


def build_feature_row(
    features: Sequence[str],
    values: Mapping[str, Any],
    defaults: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Build a one-row DataFrame with exactly the feature columns expected by the model.

    Parameters
    ----------
    features:
        Raw feature names expected by the saved preprocessor.
    values:
        Current simulation context values.
    defaults:
        Optional fallback values for missing features.
    strict:
        If True, raise an error when a required feature is missing from both
        `values` and `defaults`. If False, fill missing fields with np.nan.
    """
    defaults = defaults or {}
    row: Dict[str, Any] = {}

    missing: List[str] = []
    for feat in features:
        if feat in values:
            row[feat] = values[feat]
        elif feat in defaults:
            row[feat] = defaults[feat]
        else:
            if strict:
                missing.append(feat)
            else:
                row[feat] = np.nan

    if missing:
        raise KeyError(
            "Missing required model features for prediction: "
            f"{missing}. Provide them in `values` or `defaults`."
        )

    return pd.DataFrame([row], columns=list(features))


def predict_log_rate(
    model_bundle: Mapping[str, Any],
    feature_values: Mapping[str, Any],
    defaults: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    Predict log crime rate per agent from raw simulation context values.
    """
    features = model_bundle["features"]
    row = build_feature_row(features=features, values=feature_values, defaults=defaults, strict=strict)
    X = model_bundle["preprocessor"].transform(row)
    pred = float(model_bundle["model"].predict(X)[0])
    return pred, row


def predict_rate_per_agent(
    model_bundle: Mapping[str, Any],
    feature_values: Mapping[str, Any],
    defaults: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Predict both log rate and rate per agent.
    """
    pred_log_rate, row = predict_log_rate(
        model_bundle=model_bundle,
        feature_values=feature_values,
        defaults=defaults,
        strict=strict,
    )
    pred_rate_per_agent = float(np.exp(pred_log_rate))
    return pred_log_rate, pred_rate_per_agent, row


def expected_count_from_rate(
    pred_rate_per_agent: Number,
    actual_agents: Number,
    *,
    estimated_agents_context: Optional[Number] = None,
    config: Optional[CrimePredictionConfig] = None,
    cmpd_agents_present: int = 0,
) -> Tuple[float, float]:
    """
    Convert predicted rate per agent into an expected count (lambda) for the step.

    Returns
    -------
    adjusted_lambda, cmpd_multiplier
    """
    config = config or CrimePredictionConfig()

    agents = max(float(actual_agents), 0.0)
    base_lambda = float(pred_rate_per_agent) * agents

    if config.scale_by_agent_ratio and estimated_agents_context is not None:
        denom = max(float(estimated_agents_context), config.min_estimated_agents_for_ratio)
        base_lambda *= agents / denom

    base_lambda *= float(config.contextual_multiplier)
    base_lambda *= float(config.global_rate_multiplier)
    base_lambda *= float(config.step_hours)

    cmpd_mult = cmpd_presence_multiplier(
        cmpd_agents_present=cmpd_agents_present,
        officer_multiplier=config.cmpd_officer_multiplier,
        min_multiplier=config.min_cmpd_multiplier,
        max_multiplier=config.max_cmpd_multiplier,
    )

    adjusted_lambda = max(base_lambda * cmpd_mult, 0.0)
    return adjusted_lambda, cmpd_mult


def predict_crime_for_step(
    model_bundle: Mapping[str, Any],
    feature_values: Mapping[str, Any],
    actual_agents: Number,
    *,
    cmpd_agents_present: int = 0,
    estimated_agents_context: Optional[Number] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    config: Optional[CrimePredictionConfig] = None,
    strict: bool = True,
) -> CrimePredictionResult:
    """
    Main helper for simulation use.

    Predicts the overall probability of at least one crime occurring at a
    location during the current simulation step, based on:
        - model-predicted rate per agent
        - actual number of agents present
        - CMPD presence
        - optional calibration / context multipliers

    Parameters
    ----------
    model_bundle:
        Loaded bundle from `load_model_bundle`.
    feature_values:
        Raw feature values for the current station/location context.
    actual_agents:
        Number of simulation agents present at the location in this step.
    cmpd_agents_present:
        Number of CMPD agents / officers present at the location in this step.
    estimated_agents_context:
        Optional lookup value from the training/baseline context if you want
        ratio scaling relative to the training exposure.
    defaults:
        Optional fallback values for model features not supplied in feature_values.
    config:
        Runtime prediction config.
    strict:
        Whether to error on missing model features.

    Returns
    -------
    CrimePredictionResult
    """
    config = config or CrimePredictionConfig()

    pred_log_rate, pred_rate_per_agent, row = predict_rate_per_agent(
        model_bundle=model_bundle,
        feature_values=feature_values,
        defaults=defaults,
        strict=strict,
    )

    adjusted_lambda, cmpd_mult = expected_count_from_rate(
        pred_rate_per_agent=pred_rate_per_agent,
        actual_agents=actual_agents,
        estimated_agents_context=estimated_agents_context,
        config=config,
        cmpd_agents_present=cmpd_agents_present,
    )

    event_probability = poisson_event_probability(adjusted_lambda)
    event_probability = float(np.clip(
        event_probability,
        config.min_event_probability,
        config.max_event_probability,
    ))

    base_lambda = float(pred_rate_per_agent) * max(float(actual_agents), 0.0)

    return CrimePredictionResult(
        pred_log_rate=float(pred_log_rate),
        pred_rate_per_agent=float(pred_rate_per_agent),
        actual_agents=float(max(float(actual_agents), 0.0)),
        base_lambda=base_lambda,
        adjusted_lambda=float(adjusted_lambda),
        event_probability=event_probability,
        cmpd_multiplier=float(cmpd_mult),
        applied_rate_multiplier=float(config.global_rate_multiplier),
        feature_row=row.iloc[0].to_dict(),
    )


def choose_centroid_from_location_shares(
    station_name: str,
    shares_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
    share_col: str = "location_share",
    station_col: str = "nearest_station",
    centroid_col: str = "centroid_id",
    filters: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Sample a centroid within a station using historical location shares.

    This is useful after you already decide that a station-step crime occurs
    and want to place it at a specific centroid.
    """
    rng = rng or np.random.default_rng()
    subset = shares_df[shares_df[station_col].astype(str) == str(station_name)].copy()

    if filters:
        for col, val in filters.items():
            if col in subset.columns:
                subset = subset[subset[col] == val]

    if subset.empty:
        raise ValueError(f"No centroid share rows found for station={station_name!r} with filters={filters!r}")

    weights = subset[share_col].astype(float).fillna(0.0).to_numpy()
    if weights.sum() <= 0:
        weights = np.repeat(1.0 / len(subset), len(subset))
    else:
        weights = weights / weights.sum()

    idx = int(rng.choice(np.arange(len(subset)), p=weights))
    return subset.iloc[idx][centroid_col]


def summarize_prediction_grid(
    results: Sequence[CrimePredictionResult],
) -> pd.DataFrame:
    """
    Convert a list of results into a compact DataFrame for debugging / calibration.
    """
    return pd.DataFrame([r.as_dict() for r in results])


def example_usage() -> None:
    """
    Minimal example for local testing. Adjust keys to match your real saved bundle.
    """
    # bundle = load_model_bundle("model/xgb_context_location_bundle.joblib")
    # feature_values = {
    #     "hour": 18,
    #     "month": 7,
    #     "temp": 82.1,
    #     "tempmax": 88.0,
    #     "tempmin": 71.0,
    #     "precip": 0.0,
    #     "precipcover": 0.0,
    #     "cloudcover": 22.0,
    #     "humidity": 61.0,
    #     "windspeed": 7.5,
    #     "visibility": 10.0,
    #     "snow": 0.0,
    #     "nearest_tree_dist_m": 18.0,
    #     "trees_within_10m": 0,
    #     "trees_within_25m": 3,
    #     "trees_within_50m": 11,
    #     "env_within_10m": 1,
    #     "env_within_25m": 4,
    #     "env_within_50m": 8,
    #     "min_station_dist_m": 12.0,
    #     "mean_station_dist_m": 25.0,
    #     "num_stations_in_radius_from_list": 1,
    #     "day_type": "Weekday",
    #     "nearest_station": "CTC/Arena",
    #     "ridership_share_within_day_type": 0.14,
    #     "crime_share_within_day_type": 0.17,
    #     "station_day_share": 0.15,
    #     "hour_share": 0.045,
    # }
    #
    # cfg = CrimePredictionConfig(
    #     global_rate_multiplier=1.0,
    #     cmpd_officer_multiplier=0.85,
    # )
    #
    # result = predict_crime_for_step(
    #     model_bundle=bundle,
    #     feature_values=feature_values,
    #     actual_agents=27,
    #     cmpd_agents_present=1,
    #     config=cfg,
    # )
    # print(result.as_dict())
    pass


if __name__ == "__main__":
    example_usage()
