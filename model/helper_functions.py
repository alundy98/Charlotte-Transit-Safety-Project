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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd


Number = Union[int, float, np.number]

# -----------------------------------------------------------------------------
# Column names used by the uploaded datasets
# -----------------------------------------------------------------------------
BASELINE_STATION_COL = "nearest_station"
BASELINE_DAYTYPE_COL = "day_type"
BASELINE_HOUR_COL = "hour"
BASELINE_ESTIMATED_AGENTS_COL = "estimated_station_agents"
BASELINE_LOOKUP_KEY_COL = "lookup_key"

CENTROID_STATION_COL = "nearest_station"
CENTROID_ID_COL = "centroid_id"
CENTROID_SHARE_COL = "location_share_in_station"

DEFAULT_POI_TYPE_WEIGHTS: Dict[str, float] = {
    "Tourism": 3.1,
    "Recreation": 26.4,
    "Food": 10.9,
    "Retail": 12.5,
    "Nightlife": 9.4,
    "Public Services": 7.7,
    "Office": 9.7,
    "Errand": 10.9,
    "Healthcare": 9.4
}


@dataclass
class CrimePredictionConfig:
    """
    Runtime knobs for simulation-side conversion from model output to crime risk.
    """
    global_rate_multiplier: float = 1.0
    scale_by_agent_ratio: bool = False
    min_estimated_agents_for_ratio: float = 1e-6
    cmpd_officer_multiplier: float = 1
    min_cmpd_multiplier: float = 0.20
    max_cmpd_multiplier: float = 1.00
    contextual_multiplier: float = 1.0
    min_event_probability: float = 0.0
    max_event_probability: float = 1.0
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


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def _to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Could not convert {value!r} to Timestamp.")
    return ts


def _normalize_station_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_existing_column(df: pd.DataFrame, preferred: str, aliases: Sequence[str]) -> str:
    for col in (preferred, *aliases):
        if col in df.columns:
            return col
    raise KeyError(
        f"None of the expected columns were found. Preferred={preferred!r}, aliases={list(aliases)!r}. "
        f"Available columns={list(df.columns)!r}"
    )


def _numeric_or_fallback(series: pd.Series, fallback: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().all():
        return pd.Series(np.repeat(fallback, len(series)), index=series.index, dtype=float)
    return out.astype(float)


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


# -----------------------------------------------------------------------------
# Prediction helpers
# -----------------------------------------------------------------------------

def cmpd_presence_multiplier(
    cmpd_agents_present: int,
    officer_multiplier: float = 0.85,
    min_multiplier: float = 0.20,
    max_multiplier: float = 1.00,
) -> float:
    """
    Convert CMPD presence into a multiplicative deterrence adjustment on lambda.
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


# -----------------------------------------------------------------------------
# Baseline / lookup helpers aligned to uploaded CSVs
# -----------------------------------------------------------------------------

def make_baseline_lookup_key(station_name: str, day_type: str, hour: Number) -> str:
    return f"{_normalize_station_name(station_name)} | {str(day_type).strip()} | {int(hour)}"


def lookup_baseline_context(
    baseline_df: pd.DataFrame,
    station_name: str,
    day_type: str,
    hour: Number,
    *,
    station_col: str = BASELINE_STATION_COL,
    day_type_col: str = BASELINE_DAYTYPE_COL,
    hour_col: str = BASELINE_HOUR_COL,
    lookup_key_col: str = BASELINE_LOOKUP_KEY_COL,
) -> pd.Series:
    """
    Retrieve the baseline station/day/hour row from
    `baseline_station_day_hour_probabilities.csv`.
    """
    station_name = _normalize_station_name(station_name)
    day_type = str(day_type).strip()
    hour = int(hour)

    if lookup_key_col in baseline_df.columns:
        key = make_baseline_lookup_key(station_name, day_type, hour)
        subset = baseline_df[baseline_df[lookup_key_col].astype(str) == key]
        if not subset.empty:
            return subset.iloc[0].copy()

    station_vals = baseline_df[station_col].astype(str).str.strip()
    day_vals = baseline_df[day_type_col].astype(str).str.strip()
    hour_vals = pd.to_numeric(baseline_df[hour_col], errors="coerce")

    subset = baseline_df[(station_vals == station_name) & (day_vals == day_type) & (hour_vals == hour)]
    if subset.empty:
        raise ValueError(
            f"No baseline row found for station={station_name!r}, day_type={day_type!r}, hour={hour!r}."
        )
    return subset.iloc[0].copy()


def build_context_from_baseline_row(
    baseline_row: Mapping[str, Any],
    *,
    date_time: Optional[Any] = None,
    include_calendar_fields: bool = True,
) -> Dict[str, Any]:
    """
    Convert one baseline lookup row into a feature dictionary that can be merged
    with weather and location features before model prediction.
    """
    context = dict(baseline_row)

    if date_time is not None:
        ts = _to_timestamp(date_time)
        context["hour"] = int(ts.hour)
        context["month"] = int(ts.month)
        context["day"] = int(ts.day)
        context["dayofweek"] = int(ts.dayofweek)
        context["year_num"] = int(ts.year)
        context["date_only"] = str(ts.date())
        context["is_weekend"] = int(ts.dayofweek >= 5)
        context["hour_sin"] = float(np.sin(2 * np.pi * ts.hour / 24.0))
        context["hour_cos"] = float(np.cos(2 * np.pi * ts.hour / 24.0))
    elif include_calendar_fields and "hour" in context:
        hour = int(context["hour"])
        context["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
        context["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))

    return context


# -----------------------------------------------------------------------------
# Location sampling helpers aligned to station_centroid_location_shares.csv
# -----------------------------------------------------------------------------

def choose_centroid_from_location_shares(
    station_name: str,
    shares_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
    share_col: str = CENTROID_SHARE_COL,
    station_col: str = CENTROID_STATION_COL,
    centroid_col: str = CENTROID_ID_COL,
    filters: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Sample a centroid within a station using historical location shares.

    Default share column is `location_share_in_station`, which matches the
    uploaded centroid-share CSV.
    """
    rng = rng or np.random.default_rng()
    station_name = _normalize_station_name(station_name)

    station_col = _resolve_existing_column(shares_df, station_col, aliases=["station_name"])
    centroid_col = _resolve_existing_column(shares_df, centroid_col, aliases=[])
    share_col = _resolve_existing_column(
        shares_df,
        share_col,
        aliases=["location_share", "location_share_raw", "crime_share_in_station", "uniform_share_in_station"],
    )

    subset = shares_df[shares_df[station_col].astype(str).str.strip() == station_name].copy()

    if filters:
        for col, val in filters.items():
            if col in subset.columns:
                subset = subset[subset[col] == val]

    if subset.empty:
        raise ValueError(f"No centroid share rows found for station={station_name!r} with filters={filters!r}")

    weights = pd.to_numeric(subset[share_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.repeat(1.0 / len(subset), len(subset))
    else:
        weights = weights / weights.sum()

    idx = int(rng.choice(np.arange(len(subset)), p=weights))
    return subset.iloc[idx][centroid_col]


def choose_centroid_row_from_location_shares(
    station_name: str,
    shares_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
    share_col: str = CENTROID_SHARE_COL,
    station_col: str = CENTROID_STATION_COL,
    centroid_col: str = CENTROID_ID_COL,
    filters: Optional[Mapping[str, Any]] = None,
) -> pd.Series:
    """
    Same sampling idea as `choose_centroid_from_location_shares`, but returns the
    full sampled row instead of only the centroid id.
    """
    rng = rng or np.random.default_rng()
    centroid_id = choose_centroid_from_location_shares(
        station_name=station_name,
        shares_df=shares_df,
        rng=rng,
        share_col=share_col,
        station_col=station_col,
        centroid_col=centroid_col,
        filters=filters,
    )
    centroid_col = _resolve_existing_column(shares_df, centroid_col, aliases=[])
    subset = shares_df[pd.to_numeric(shares_df[centroid_col], errors="coerce") == float(centroid_id)]
    if subset.empty:
        raise ValueError(f"Sampled centroid_id={centroid_id!r}, but no row was found.")
    return subset.iloc[0].copy()


# -----------------------------------------------------------------------------
# Weather generation helpers
# -----------------------------------------------------------------------------

def generate_weather_from_history(
    weather_history_df: pd.DataFrame,
    date_time: Any,
    rng: Optional[np.random.Generator] = None,
    *,
    station_name: Optional[str] = None,
    station_col: str = BASELINE_STATION_COL,
) -> Dict[str, float]:
    """
    Generate slightly randomized weather values for a simulation step using
    historical rows from the modeling dataset.

    The function is intentionally lightweight:
    - filter to similar month + hour when available
    - optionally filter to station
    - draw one historical row
    - add slight jitter to continuous weather variables
    """
    rng = rng or np.random.default_rng()
    ts = _to_timestamp(date_time)

    required_candidates = {
        "temp": ["temp"],
        "tempmax": ["tempmax"],
        "tempmin": ["tempmin"],
        "precip": ["precip"],
        "precipcover": ["precipcover"],
        "cloudcover": ["cloudcover"],
        "humidity": ["humidity"],
        "windspeed": ["windspeed"],
        "visibility": ["visibility"],
        "snow": ["snow"],
    }

    for canonical, aliases in required_candidates.items():
        _resolve_existing_column(weather_history_df, canonical, aliases)

    subset = weather_history_df.copy()

    if station_name is not None and station_col in subset.columns:
        subset = subset[subset[station_col].astype(str).str.strip() == _normalize_station_name(station_name)]

    if "month" in subset.columns:
        month_vals = pd.to_numeric(subset["month"], errors="coerce")
        month_subset = subset[month_vals == ts.month]
        if not month_subset.empty:
            subset = month_subset

    if "hour" in subset.columns:
        hour_vals = pd.to_numeric(subset["hour"], errors="coerce")
        hour_subset = subset[hour_vals == ts.hour]
        if not hour_subset.empty:
            subset = hour_subset

    if subset.empty:
        subset = weather_history_df.copy()

    sampled = subset.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]

    weather = {
        "temp": float(sampled["temp"]),
        "tempmax": float(sampled["tempmax"]),
        "tempmin": float(sampled["tempmin"]),
        "precip": float(sampled["precip"]),
        "precipcover": float(sampled["precipcover"]),
        "cloudcover": float(sampled["cloudcover"]),
        "humidity": float(sampled["humidity"]),
        "windspeed": float(sampled["windspeed"]),
        "visibility": float(sampled["visibility"]),
        "snow": float(sampled["snow"]),
    }

    # Small jitter only; this is meant to create slight variability.
    weather["temp"] += float(rng.normal(0.0, 2.0))
    weather["tempmax"] += float(rng.normal(0.0, 1.5))
    weather["tempmin"] += float(rng.normal(0.0, 1.5))
    weather["precip"] = max(0.0, weather["precip"] + float(rng.normal(0.0, 0.03)))
    weather["precipcover"] = float(np.clip(weather["precipcover"] + rng.normal(0.0, 5.0), 0.0, 100.0))
    weather["cloudcover"] = float(np.clip(weather["cloudcover"] + rng.normal(0.0, 5.0), 0.0, 100.0))
    weather["humidity"] = float(np.clip(weather["humidity"] + rng.normal(0.0, 4.0), 0.0, 100.0))
    weather["windspeed"] = max(0.0, weather["windspeed"] + float(rng.normal(0.0, 1.0)))
    weather["visibility"] = max(0.0, weather["visibility"] + float(rng.normal(0.0, 0.5)))
    weather["snow"] = max(0.0, weather["snow"] + float(rng.normal(0.0, 0.02)))

    if weather["tempmax"] < weather["tempmin"]:
        temp_mid = 0.5 * (weather["tempmax"] + weather["tempmin"])
        spread = abs(float(rng.normal(4.0, 1.0)))
        weather["tempmax"] = temp_mid + spread / 2.0
        weather["tempmin"] = temp_mid - spread / 2.0

    return weather


# -----------------------------------------------------------------------------
# POI destination helpers
# -----------------------------------------------------------------------------

def choose_poi_type(
    rng: Optional[np.random.Generator] = None,
    poi_type_weights: Optional[Mapping[str, float]] = None,
) -> str:
    """
    Pick a POI type using simple default percentages unless custom weights are
    provided by the caller.
    """
    rng = rng or np.random.default_rng()
    weights = dict(poi_type_weights or DEFAULT_POI_TYPE_WEIGHTS)
    labels = list(weights.keys())
    probs = np.asarray(list(weights.values()), dtype=float)
    if probs.sum() <= 0:
        probs = np.repeat(1.0 / len(labels), len(labels))
    else:
        probs = probs / probs.sum()
    idx = int(rng.choice(np.arange(len(labels)), p=probs))
    return labels[idx]


def choose_poi_destination(
    poi_df: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
    *,
    poi_type_weights: Optional[Mapping[str, float]] = None,
    type_col: str = "poi_type",
    station_name: Optional[str] = None,
    station_col: str = BASELINE_STATION_COL,
    filters: Optional[Mapping[str, Any]] = None,
    return_full_row: bool = True,
) -> Union[pd.Series, Any]:
    """
    Choose a destination POI in two stages:
    1. sample a POI type using arbitrary default percentages
    2. sample one POI of that type uniformly at random

    This function expects a POI dataframe with a type column. By default it uses
    `poi_type`, but callers can override that if their file uses another name.
    """
    rng = rng or np.random.default_rng()
    type_col = _resolve_existing_column(poi_df, type_col, aliases=["type", "poi_category", "category"])

    subset = poi_df.copy()
    if station_name is not None and station_col in subset.columns:
        subset = subset[subset[station_col].astype(str).str.strip() == _normalize_station_name(station_name)]

    if filters:
        for col, val in filters.items():
            if col in subset.columns:
                subset = subset[subset[col] == val]

    if subset.empty:
        raise ValueError("No POI rows available after applying station / filter constraints.")

    available_types = subset[type_col].dropna().astype(str).str.strip()
    if available_types.empty:
        raise ValueError(f"POI dataframe has no usable values in type column {type_col!r}.")

    desired_type = choose_poi_type(rng=rng, poi_type_weights=poi_type_weights)
    typed_subset = subset[subset[type_col].astype(str).str.strip() == desired_type]

    if typed_subset.empty:
        # Fallback: if the chosen type is not represented in the filtered subset,
        # choose uniformly from the types that are present.
        present_types = available_types.unique()
        desired_type = str(rng.choice(present_types))
        typed_subset = subset[subset[type_col].astype(str).str.strip() == desired_type]

    chosen_row = typed_subset.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0].copy()
    chosen_row["chosen_poi_type"] = desired_type
    return chosen_row if return_full_row else chosen_row.get("poi_id", chosen_row.name)


def summarize_prediction_grid(
    results: Sequence[CrimePredictionResult],
) -> pd.DataFrame:
    """
    Convert a list of results into a compact DataFrame for debugging / calibration.
    """
    return pd.DataFrame([r.as_dict() for r in results])


# -----------------------------------------------------------------------------
# Simulation route comparison helpers
# -----------------------------------------------------------------------------

def _resolve_centroid_id_column(df: pd.DataFrame) -> str:
    return _resolve_existing_column(
        df,
        "centroid_id",
        aliases=["centroid", "location_id", "crime_centroid_id", "centroid_idx"],
    )



def _extract_route_counts(
    df: pd.DataFrame,
    outcome_col: str = "crime_stopped",
) -> Tuple[int, int, int]:
    if outcome_col not in df.columns:
        raise KeyError(f"Missing required column: {outcome_col}")

    outcome = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0)

    stopped = int((outcome == 1).sum())
    happened = int((outcome == 0).sum())
    total = stopped + happened

    if total <= 0:
        raise ValueError(
            "The route-results file has zero valid crime outcomes. "
            "Expected binary values of 0 and 1 in the outcome column."
        )

    return stopped, happened, total



def _normal_cdf(z: float) -> float:
    return float(0.5 * (1.0 + np.math.erf(z / np.sqrt(2.0))))



def _two_sample_proportion_z_test(
    successes_1: int,
    total_1: int,
    successes_2: int,
    total_2: int,
    alternative: str = "larger",
) -> Dict[str, float]:
    if total_1 <= 0 or total_2 <= 0:
        raise ValueError("Both samples must have positive totals for a two-sample proportion z-test.")

    p1 = successes_1 / total_1
    p2 = successes_2 / total_2
    pooled = (successes_1 + successes_2) / (total_1 + total_2)
    se = np.sqrt(pooled * (1.0 - pooled) * ((1.0 / total_1) + (1.0 / total_2)))

    if se == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = (p2 - p1) / se
        if alternative == "larger":
            p_value = 1.0 - _normal_cdf(z_stat)
        elif alternative == "smaller":
            p_value = _normal_cdf(z_stat)
        else:
            p_value = 2.0 * min(_normal_cdf(z_stat), 1.0 - _normal_cdf(z_stat))

    return {
        "p1": float(p1),
        "p2": float(p2),
        "pooled_proportion": float(pooled),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "standard_error": float(se),
    }



def _distance_to_station_priority_weight(
    distance_m: float,
    max_weight: int = 6,
    bucket_size_m: float = 100.0,
) -> int:
    if pd.isna(distance_m):
        return 1
    bucket = int(np.floor(max(float(distance_m), 0.0) / bucket_size_m))
    return int(max(1, max_weight - bucket))



def _build_centroid_priority_table(
    centroid_locations_df: pd.DataFrame,
    centroid_id_col: str = "centroid_id",
    distance_col: str = "min_station_dist_m",
) -> pd.DataFrame:
    centroid_id_col = _resolve_existing_column(
        centroid_locations_df,
        centroid_id_col,
        aliases=["centroid", "location_id", "crime_centroid_id"],
    )
    distance_col = _resolve_existing_column(
        centroid_locations_df,
        distance_col,
        aliases=["mean_station_dist_m", "nearest_station_dist_m"],
    )

    out = centroid_locations_df[[centroid_id_col, distance_col]].copy()
    out[centroid_id_col] = pd.to_numeric(out[centroid_id_col], errors="coerce")
    out[distance_col] = pd.to_numeric(out[distance_col], errors="coerce")
    out = out.dropna(subset=[centroid_id_col]).drop_duplicates(subset=[centroid_id_col])
    out["station_priority_weight"] = out[distance_col].apply(_distance_to_station_priority_weight)
    return out.rename(columns={centroid_id_col: "centroid_id", distance_col: "station_distance_m"})



def _top_weighted_centroids(
    df: pd.DataFrame,
    centroid_locations_df: pd.DataFrame,
    *,
    outcome_col: str = "crime_stopped",
    outcome_value: int,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    centroid_col = _resolve_centroid_id_column(df)

    if outcome_col not in df.columns:
        raise KeyError(f"Missing required column: {outcome_col}")

    work = df[[centroid_col, outcome_col]].copy()
    work[centroid_col] = pd.to_numeric(work[centroid_col], errors="coerce")
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=[centroid_col, outcome_col])

    # Keep only rows matching the requested outcome:
    # outcome_value = 0 -> crimes happened
    # outcome_value = 1 -> crimes stopped
    work = work[work[outcome_col] == outcome_value]

    counts = (
        work.groupby(centroid_col, dropna=True)
        .size()
        .reset_index(name="raw_count")
    )

    priority = _build_centroid_priority_table(centroid_locations_df)
    ranked = counts.merge(
        priority,
        how="left",
        left_on=centroid_col,
        right_on="centroid_id",
    )

    ranked["station_priority_weight"] = ranked["station_priority_weight"].fillna(1).astype(int)
    ranked["station_distance_m"] = pd.to_numeric(ranked["station_distance_m"], errors="coerce")
    ranked["weighted_count"] = ranked["raw_count"] * ranked["station_priority_weight"]

    ranked = ranked.sort_values(
        by=["weighted_count", "raw_count", "station_priority_weight", "station_distance_m"],
        ascending=[False, False, False, True],
    ).head(top_n)

    return [
        {
            "centroid_id": int(row["centroid_id"]),
            "raw_count": int(row["raw_count"]),
            "station_priority_weight": int(row["station_priority_weight"]),
            "weighted_count": float(row["weighted_count"]),
            "station_distance_m": None if pd.isna(row["station_distance_m"]) else float(row["station_distance_m"]),
        }
        for _, row in ranked.iterrows()
    ]



def compare_simulation_routes(
    previous_route_csv: Union[str, Path],
    current_route_csv: Union[str, Path],
    centroid_locations_csv: Union[str, Path],
    *,
    outcome_col: str = "crime_stopped",
    alpha: float = 0.05,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Compare two simulation result CSVs using a two-sample proportion z-test.

    Assumptions
    -----------
    - Each CSV contains a single binary outcome column where:
        1 = crime was stopped
        0 = crime happened
    - The route is considered "better" only when the most recent route has both:
        1. a higher stopped proportion than the previous route, and
        2. a one-sided z-test p-value < alpha.
    - Centroid rankings are produced from the most recent route CSV.
    - Returned centroid lists are re-ranked using a station-distance priority
      weight from 1 to 6, where 6 is closest to the nearest station and the
      weight drops by 1 for each additional 100 meters.
    """
    previous_df = pd.read_csv(previous_route_csv)
    current_df = pd.read_csv(current_route_csv)
    centroid_locations_df = pd.read_csv(centroid_locations_csv)

    if outcome_col not in previous_df.columns:
        raise KeyError(f"Missing required column in previous route CSV: {outcome_col}")
    if outcome_col not in current_df.columns:
        raise KeyError(f"Missing required column in current route CSV: {outcome_col}")

    prev_stopped, prev_happened, prev_total = _extract_route_counts(
        previous_df,
        outcome_col=outcome_col,
    )
    curr_stopped, curr_happened, curr_total = _extract_route_counts(
        current_df,
        outcome_col=outcome_col,
    )

    test = _two_sample_proportion_z_test(
        successes_1=prev_stopped,
        total_1=prev_total,
        successes_2=curr_stopped,
        total_2=curr_total,
        alternative="larger",
    )

    is_better = bool((test["p2"] > test["p1"]) and (test["p_value"] < alpha))

    top_happened = _top_weighted_centroids(
        df=current_df,
        centroid_locations_df=centroid_locations_df,
        outcome_col=outcome_col,
        outcome_value=0,
        top_n=top_n,
    )

    top_stopped = _top_weighted_centroids(
        df=current_df,
        centroid_locations_df=centroid_locations_df,
        outcome_col=outcome_col,
        outcome_value=1,
        top_n=top_n,
    )

    return {
        "new_route_better": is_better,
        "alpha": float(alpha),
        "z_test": test,
        "previous_route": {
            "crimes_stopped": int(prev_stopped),
            "crimes_happened": int(prev_happened),
            "total_crime_outcomes": int(prev_total),
            "stopped_proportion": float(test["p1"]),
        },
        "current_route": {
            "crimes_stopped": int(curr_stopped),
            "crimes_happened": int(curr_happened),
            "total_crime_outcomes": int(curr_total),
            "stopped_proportion": float(test["p2"]),
        },
        "top_happened_centroids": top_happened,
        "top_stopped_centroids": top_stopped,
    }


def example_usage() -> None:
    """
    Minimal example for local testing. Adjust keys to match your real saved bundle.
    """
    # bundle = load_model_bundle("model/xgb_context_location_bundle.joblib")
    # baseline_df = pd.read_csv("baseline_station_day_hour_probabilities.csv")
    # modeling_df = pd.read_csv("modeling_dataset_hourly.csv")
    # shares_df = pd.read_csv("station_centroid_location_shares.csv")
    #
    # ts = pd.Timestamp("2017-07-10 18:00:00")
    # baseline_row = lookup_baseline_context(baseline_df, "CTC/Arena", "Weekday", ts.hour)
    # feature_values = build_context_from_baseline_row(baseline_row, date_time=ts)
    # feature_values.update(generate_weather_from_history(modeling_df, ts, station_name="CTC/Arena"))
    # centroid_row = choose_centroid_row_from_location_shares("CTC/Arena", shares_df)
    #
    # for col in [
    #     "nearest_tree_dist_m", "trees_within_10m", "trees_within_25m", "trees_within_50m",
    #     "env_within_10m", "env_within_25m", "env_within_50m", "min_station_dist_m",
    #     "mean_station_dist_m", "num_stations_in_radius_from_list", "lat", "lon",
    # ]:
    #     if col in centroid_row.index:
    #         feature_values[col] = centroid_row[col]
    #
    # cfg = CrimePredictionConfig(global_rate_multiplier=1.0, cmpd_officer_multiplier=0.85)
    # result = predict_crime_for_step(
    #     model_bundle=bundle,
    #     feature_values=feature_values,
    #     actual_agents=27,
    #     cmpd_agents_present=1,
    #     estimated_agents_context=baseline_row.get("estimated_station_agents"),
    #     config=cfg,
    # )
    # print(result.as_dict())
    pass


if __name__ == "__main__":
    example_usage()
