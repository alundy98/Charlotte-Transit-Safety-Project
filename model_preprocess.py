import ast
import numpy as np
import pandas as pd
import geopandas as gpd


# =========================================================
# FILE PATHS
# =========================================================
CRIME_FILE = "data/crimes_with_env_info.geojson"
RIDERSHIP_FILE = "data/overall_ridership.csv"

CLEANED_OUTPUT = "data/crimes_with_env_info_cleaned.csv"
BASELINE_OUTPUT = "model/baseline_station_day_hour_probabilities.csv"
MODELING_OUTPUT = "model/modeling_dataset_hourly.csv"
CENTROID_SHARE_OUTPUT = "model/station_centroid_location_shares.csv"


# =========================================================
# HELPERS
# =========================================================
def safe_literal_eval(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return []
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            return []
    return []


def normalize_day_type(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v == "weekday":
        return "Weekday"
    if v == "saturday":
        return "Saturday"
    if v == "sunday":
        return "Sunday"
    return val


def add_time_features(df):
    if "day_type" in df.columns:
        df["is_weekend"] = df["day_type"].isin(["Saturday", "Sunday"]).astype(int)
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def clean_numeric_columns(df, numeric_cols):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_mode(series):
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return np.nan
    mode_vals = s.mode()
    if len(mode_vals) == 0:
        return np.nan
    return mode_vals.iloc[0]


def normalize_within_group(series, group_keys):
    denom = series.groupby(group_keys).transform("sum")
    return np.where(denom > 0, series / denom, 0.0)


# =========================================================
# 1. LOAD DATA
# =========================================================
print("Loading files...")
gdf = gpd.read_file(CRIME_FILE)
ridership = pd.read_csv(RIDERSHIP_FILE)

print(f"Crime rows: {len(gdf):,}")
print(f"Crime columns: {len(gdf.columns)}")
print(f"Ridership rows: {len(ridership):,}")


# =========================================================
# 2. BASIC CLEANING
# =========================================================
print("\nCleaning basic fields...")

gdf["date"] = pd.to_datetime(gdf["date"], errors="coerce")
gdf["day_type"] = gdf["day_type"].apply(normalize_day_type)

numeric_cols = [
    "year", "hour",
    "LATITUDE_PUBLIC", "LONGITUDE_PUBLIC", "lat", "lon",
    "tempmax", "tempmin", "temp", "precip", "precipcover",
    "cloudcover", "humidity", "windspeed", "visibility", "snow",
    "x", "y", "x_snap", "y_snap", "centroid_id",
    "nearest_tree_dist_m",
    "trees_within_10m", "trees_within_25m", "trees_within_50m",
    "env_within_10m", "env_within_25m", "env_within_50m",
    "stations_in_radius", "env_object"
]
gdf = clean_numeric_columns(gdf, numeric_cols)

required_core = ["date", "hour", "day_type", "nearest_station"]
gdf = gdf.dropna(subset=required_core).copy()

gdf["hour"] = gdf["hour"].astype(int)
gdf = gdf[(gdf["hour"] >= 0) & (gdf["hour"] <= 23)].copy()

gdf["nearest_station"] = gdf["nearest_station"].astype(str).str.strip()
ridership["Station"] = ridership["Station"].astype(str).str.strip()

if "INCIDENT_REPORT_ID" in gdf.columns:
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset="INCIDENT_REPORT_ID").copy()
    after = len(gdf)
    print(f"Removed {before - after} duplicate INCIDENT_REPORT_ID rows.")
else:
    print("INCIDENT_REPORT_ID not found; skipped duplicate removal.")


# =========================================================
# 3. PARSE station distance list + CREATE SUMMARY FEATURES
# =========================================================
print("\nParsing station distance lists...")
if "stations_in_radius_dist_m" in gdf.columns:
    gdf["stations_in_radius_dist_m_list"] = gdf["stations_in_radius_dist_m"].apply(safe_literal_eval)
    gdf["min_station_dist_m"] = gdf["stations_in_radius_dist_m_list"].apply(
        lambda x: min(x) if isinstance(x, list) and len(x) > 0 else np.nan
    )
    gdf["mean_station_dist_m"] = gdf["stations_in_radius_dist_m_list"].apply(
        lambda x: float(np.mean(x)) if isinstance(x, list) and len(x) > 0 else np.nan
    )
    gdf["num_stations_in_radius_from_list"] = gdf["stations_in_radius_dist_m_list"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
else:
    gdf["min_station_dist_m"] = np.nan
    gdf["mean_station_dist_m"] = np.nan
    gdf["num_stations_in_radius_from_list"] = np.nan

if "nearest_station" in gdf.columns:
    gdf["min_station_dist_m"] = gdf["min_station_dist_m"].fillna(
        gdf.groupby("nearest_station")["min_station_dist_m"].transform("median")
    )
    gdf["mean_station_dist_m"] = gdf["mean_station_dist_m"].fillna(
        gdf.groupby("nearest_station")["mean_station_dist_m"].transform("median")
    )

gdf["min_station_dist_m"] = gdf["min_station_dist_m"].fillna(gdf["min_station_dist_m"].median())
gdf["mean_station_dist_m"] = gdf["mean_station_dist_m"].fillna(gdf["mean_station_dist_m"].median())
gdf["num_stations_in_radius_from_list"] = gdf["num_stations_in_radius_from_list"].fillna(0)


# =========================================================
# 4. STREAMLINE / STANDARDIZE FIELDS
# =========================================================
print("\nCreating streamlined fields...")

if "lat" in gdf.columns and "LATITUDE_PUBLIC" in gdf.columns:
    gdf["lat"] = gdf["lat"].fillna(gdf["LATITUDE_PUBLIC"])
if "lon" in gdf.columns and "LONGITUDE_PUBLIC" in gdf.columns:
    gdf["lon"] = gdf["lon"].fillna(gdf["LONGITUDE_PUBLIC"])

gdf = add_time_features(gdf)
gdf["date_only"] = pd.to_datetime(gdf["date"]).dt.normalize()

categorical_cols = [
    "CMPD_PATROL_DIVISION",
    "LOCATION_TYPE_DESCRIPTION",
    "PLACE_TYPE_DESCRIPTION",
    "PLACE_DETAIL_DESCRIPTION",
    "HIGHEST_NIBRS_DESCRIPTION",
    "cluster_title",
    "nearest_station",
    "conditions",
    "day_type"
]
for col in categorical_cols:
    if col in gdf.columns:
        gdf[col] = gdf[col].astype(str).str.strip()


# =========================================================
# 5. RIDERSHIP CLEANING
# =========================================================
print("\nCleaning ridership data...")

ridership_numeric_cols = [
    "Average_Weekday", "Average_Saturday", "Average_Sunday", "Overall_Average"
]
for col in ridership_numeric_cols:
    if col in ridership.columns:
        ridership[col] = pd.to_numeric(ridership[col], errors="coerce")

crime_stations = set(gdf["nearest_station"].dropna().unique())
ride_stations = set(ridership["Station"].dropna().unique())

missing_in_ridership = sorted(crime_stations - ride_stations)
missing_in_crime = sorted(ride_stations - crime_stations)

if missing_in_ridership:
    print("Stations in crime data but not ridership:", missing_in_ridership)
if missing_in_crime:
    print("Stations in ridership but not crime data:", missing_in_crime)

gdf = gdf.merge(
    ridership[["Station", "Overall_Average"]],
    left_on="nearest_station",
    right_on="Station",
    how="left"
)

gdf = gdf.rename(columns={"Overall_Average": "station_overall_ridership"})
gdf = gdf.drop(columns=["Station"], errors="ignore")

max_overall = gdf["station_overall_ridership"].max()
if pd.notna(max_overall) and max_overall > 0:
    gdf["ridership_weight_overall"] = gdf["station_overall_ridership"] / max_overall
else:
    gdf["ridership_weight_overall"] = np.nan


# =========================================================
# 6. BUILD STATION-HOUR BASELINE + EXPOSURE
# =========================================================
print("\nBuilding station-day-hour baseline lookup table...")

total_crimes = len(gdf)

date_min = pd.to_datetime(gdf["date_only"]).min()
date_max = pd.to_datetime(gdf["date_only"]).max()

n_days = (date_max - date_min).days + 1
total_system_hours = n_days * 24
system_crimes_per_hour = total_crimes / total_system_hours
avg_crimes_per_day = total_crimes / n_days

print(f"Date range: {date_min.date()} to {date_max.date()}")
print(f"Total crimes: {total_crimes:,}")
print(f"Total days covered: {n_days:,}")
print(f"Total system hours: {total_system_hours:,}")
print(f"System crimes per hour: {system_crimes_per_hour:.6f}")

# Distribute the average daily crime total over the 24 hours.
hour_counts = gdf.groupby("hour").size().reindex(range(24), fill_value=0)
hour_share = (hour_counts / hour_counts.sum()).fillna(0)

# Average number of crimes across the whole system for a given hour bucket.
system_expected_crimes_by_hour = avg_crimes_per_day * hour_share

ridership_long = ridership.melt(
    id_vars="Station",
    value_vars=["Average_Weekday", "Average_Saturday", "Average_Sunday"],
    var_name="ridership_type",
    value_name="ridership"
)

ridership_type_map = {
    "Average_Weekday": "Weekday",
    "Average_Saturday": "Saturday",
    "Average_Sunday": "Sunday"
}
ridership_long["day_type"] = ridership_long["ridership_type"].map(ridership_type_map)
ridership_long = ridership_long.drop(columns=["ridership_type"])
ridership_long = ridership_long.rename(columns={"Station": "nearest_station"})
ridership_long["ridership"] = pd.to_numeric(ridership_long["ridership"], errors="coerce").fillna(0.0)

station_day_crime = (
    gdf.groupby(["nearest_station", "day_type"])
    .size()
    .rename("crime_count_station_day")
    .reset_index()
)

day_type_totals = (
    station_day_crime.groupby("day_type")["crime_count_station_day"]
    .sum()
    .rename("day_type_crime_total")
    .reset_index()
)

station_day_crime = station_day_crime.merge(day_type_totals, on="day_type", how="left")
station_day_crime["crime_share_within_day_type"] = np.where(
    station_day_crime["day_type_crime_total"] > 0,
    station_day_crime["crime_count_station_day"] / station_day_crime["day_type_crime_total"],
    0.0
)

station_day_weights = ridership_long.merge(
    station_day_crime[["nearest_station", "day_type", "crime_share_within_day_type"]],
    on=["nearest_station", "day_type"],
    how="left"
)
station_day_weights["crime_share_within_day_type"] = station_day_weights["crime_share_within_day_type"].fillna(0.0)

station_day_weights["ridership_share_within_day_type"] = (
    station_day_weights["ridership"] /
    station_day_weights.groupby("day_type")["ridership"].transform("sum").replace(0, np.nan)
).fillna(0.0)

# Blend ridership exposure with historical concentration.
# Ridership gets the larger weight because the ABM spawns agents from ridership.
w_ridership = 0.70
w_crime = 0.30

station_day_weights["blended_share_raw"] = (
    w_ridership * station_day_weights["ridership_share_within_day_type"] +
    w_crime * station_day_weights["crime_share_within_day_type"]
)

station_day_weights["station_day_share"] = (
    station_day_weights["blended_share_raw"] /
    station_day_weights.groupby("day_type")["blended_share_raw"].transform("sum").replace(0, np.nan)
).fillna(0.0)

baseline = (
    station_day_weights[[
        "nearest_station",
        "day_type",
        "ridership",
        "ridership_share_within_day_type",
        "crime_share_within_day_type",
        "station_day_share"
    ]]
    .assign(key=1)
    .merge(pd.DataFrame({"hour": range(24), "key": 1}), on="key")
    .drop(columns="key")
)

baseline["hour_share"] = baseline["hour"].map(hour_share).fillna(0.0)
baseline["system_expected_crimes_this_hour"] = baseline["hour"].map(system_expected_crimes_by_hour).fillna(0.0)

# Daily ridership is distributed over the 24 hours using the empirical hour profile.
baseline["estimated_station_agents"] = baseline["ridership"] * baseline["hour_share"]

# Expected crimes in this station-day-hour cell.
baseline["baseline_expected_count"] = (
    baseline["system_expected_crimes_this_hour"] * baseline["station_day_share"]
)

baseline["baseline_prob"] = 1 - np.exp(-baseline["baseline_expected_count"])
baseline["baseline_prob"] = baseline["baseline_prob"].clip(0, 1)

# Per-agent station-hour baseline rate.
baseline["baseline_rate_per_agent"] = np.where(
    baseline["estimated_station_agents"] > 0,
    baseline["baseline_expected_count"] / baseline["estimated_station_agents"],
    0.0
)

baseline["lookup_key"] = (
    baseline["nearest_station"].astype(str) + " | " +
    baseline["day_type"].astype(str) + " | " +
    baseline["hour"].astype(str)
)

print(f"Mean baseline expected count: {baseline['baseline_expected_count'].mean():.6f}")
print(f"Mean baseline probability:    {baseline['baseline_prob'].mean():.6f}")
print(f"Mean est. station agents:     {baseline['estimated_station_agents'].mean():.6f}")


# =========================================================
# 6B. BUILD WITHIN-STATION LOCATION SHARES FOR LATER ABM USE
# =========================================================
print("\nBuilding within-station centroid location shares...")

centroid_feature_cols = [
    "nearest_tree_dist_m",
    "trees_within_10m", "trees_within_25m", "trees_within_50m",
    "env_within_10m", "env_within_25m", "env_within_50m",
    "min_station_dist_m", "mean_station_dist_m", "num_stations_in_radius_from_list",
    "lat", "lon"
]
centroid_feature_cols = [c for c in centroid_feature_cols if c in gdf.columns]

centroid_template = (
    gdf.groupby("centroid_id", dropna=False)
    .agg({
        "nearest_station": safe_mode,
        **{c: "median" for c in centroid_feature_cols}
    })
    .reset_index()
)

centroid_template["nearest_station"] = centroid_template["nearest_station"].astype(str).str.strip()

centroid_station_counts = (
    centroid_template.groupby("nearest_station")
    .size()
    .rename("n_centroids_in_station")
    .reset_index()
)

centroid_crime_counts = (
    gdf.groupby(["nearest_station", "centroid_id"])
    .size()
    .rename("centroid_crime_count")
    .reset_index()
)

centroid_shares = centroid_template[["centroid_id", "nearest_station"]].merge(
    centroid_crime_counts,
    on=["nearest_station", "centroid_id"],
    how="left"
).merge(
    centroid_station_counts,
    on="nearest_station",
    how="left"
)

centroid_shares["centroid_crime_count"] = centroid_shares["centroid_crime_count"].fillna(0.0)
centroid_shares["uniform_share_in_station"] = np.where(
    centroid_shares["n_centroids_in_station"] > 0,
    1.0 / centroid_shares["n_centroids_in_station"],
    0.0
)

station_centroid_totals = centroid_shares.groupby("nearest_station")["centroid_crime_count"].transform("sum")
centroid_shares["crime_share_in_station"] = np.where(
    station_centroid_totals > 0,
    centroid_shares["centroid_crime_count"] / station_centroid_totals,
    0.0
)

# Blend uniform and historical concentration so sparse centroids are not forced to zero forever.
w_uniform = 0.60
w_centroid_crime = 0.40
centroid_shares["location_share_raw"] = (
    w_uniform * centroid_shares["uniform_share_in_station"] +
    w_centroid_crime * centroid_shares["crime_share_in_station"]
)

centroid_shares["location_share_in_station"] = normalize_within_group(
    centroid_shares["location_share_raw"],
    centroid_shares["nearest_station"]
)

centroid_shares = centroid_shares.merge(
    centroid_template,
    on=["centroid_id", "nearest_station"],
    how="left"
)

station_context = (
    centroid_shares.groupby("nearest_station", dropna=False)
    .agg({
        **{
            c: "mean" for c in [
                "nearest_tree_dist_m",
                "trees_within_10m", "trees_within_25m", "trees_within_50m",
                "env_within_10m", "env_within_25m", "env_within_50m",
                "min_station_dist_m", "mean_station_dist_m",
                "num_stations_in_radius_from_list", "lat", "lon"
            ] if c in centroid_shares.columns
        },
        "location_share_in_station": "sum"
    })
    .reset_index()
)
station_context = station_context.drop(columns=["location_share_in_station"], errors="ignore")

gdf_station_context = gdf[["nearest_station", "station_overall_ridership", "ridership_weight_overall"]].drop_duplicates()
station_context = station_context.merge(
    gdf_station_context,
    on="nearest_station",
    how="left"
)


# =========================================================
# 7. CREATE STATION-HOUR MODELING DATASET WITH ZERO-CRIME HOURS
# =========================================================
print("\nCreating full station-hour modeling dataset...")

stations = pd.DataFrame({"nearest_station": sorted(gdf["nearest_station"].dropna().unique())})

all_hours = pd.DataFrame({
    "date_only": pd.date_range(date_min, date_max, freq="D").repeat(24),
    "hour": np.tile(np.arange(24), n_days)
})
all_hours["day_type"] = all_hours["date_only"].dt.dayofweek.map(
    lambda d: "Weekday" if d < 5 else ("Saturday" if d == 5 else "Sunday")
)

station_hour_panel = stations.assign(key=1).merge(
    all_hours.assign(key=1),
    on="key",
    how="outer"
).drop(columns="key")

station_hour_panel = add_time_features(station_hour_panel)
station_hour_panel["month"] = station_hour_panel["date_only"].dt.month
station_hour_panel["day"] = station_hour_panel["date_only"].dt.day
station_hour_panel["dayofweek"] = station_hour_panel["date_only"].dt.dayofweek
station_hour_panel["year_num"] = station_hour_panel["date_only"].dt.year

# Actual observed station-hour crime counts.
station_hour_counts = (
    gdf.groupby(["nearest_station", "date_only", "hour", "day_type"], dropna=False)
    .size()
    .rename("crime_count")
    .reset_index()
)

hourly_model = station_hour_panel.merge(
    station_hour_counts,
    on=["nearest_station", "date_only", "hour", "day_type"],
    how="left"
)
hourly_model["crime_count"] = hourly_model["crime_count"].fillna(0).astype(int)

# Citywide date-hour weather / dynamic conditions.
dynamic_feature_cols = [
    "temp", "tempmax", "tempmin", "precip", "precipcover",
    "cloudcover", "humidity", "windspeed", "visibility", "snow"
]
dynamic_feature_cols = [c for c in dynamic_feature_cols if c in gdf.columns]

date_hour_dynamic = (
    gdf.groupby(["date_only", "hour"], dropna=False)[dynamic_feature_cols]
    .median()
    .reset_index()
) if dynamic_feature_cols else pd.DataFrame(columns=["date_only", "hour"])

hourly_model = hourly_model.merge(
    date_hour_dynamic,
    on=["date_only", "hour"],
    how="left"
)

# Fill any date-hour weather gaps with month-hour medians, then global medians.
for col in dynamic_feature_cols:
    hourly_model[col] = hourly_model[col].fillna(
        hourly_model.groupby(["month", "hour"])[col].transform("median")
    )
    hourly_model[col] = hourly_model[col].fillna(hourly_model[col].median())

# Static station-level environmental context.
hourly_model = hourly_model.merge(
    station_context,
    on="nearest_station",
    how="left"
)

# Baseline lookup / exposure.
hourly_model = hourly_model.merge(
    baseline[[
        "nearest_station", "day_type", "hour",
        "ridership",
        "ridership_share_within_day_type",
        "crime_share_within_day_type",
        "station_day_share",
        "hour_share",
        "estimated_station_agents",
        "baseline_expected_count",
        "baseline_rate_per_agent",
        "baseline_prob"
    ]],
    on=["nearest_station", "day_type", "hour"],
    how="left",
    validate="many_to_one"
)

fill_zero_cols = [
    "ridership",
    "ridership_share_within_day_type",
    "crime_share_within_day_type",
    "station_day_share",
    "hour_share",
    "estimated_station_agents",
    "baseline_expected_count",
    "baseline_rate_per_agent",
    "baseline_prob"
]
for col in fill_zero_cols:
    if col in hourly_model.columns:
        hourly_model[col] = hourly_model[col].fillna(0.0)

# Smoothed rate target for modeling.
hourly_model["estimated_agents"] = hourly_model["estimated_station_agents"].clip(lower=1e-6)
hourly_model["smoothed_crime_count"] = hourly_model["crime_count"] + 0.5
hourly_model["smoothed_agents"] = hourly_model["estimated_agents"] + 1.0

hourly_model["target_rate"] = hourly_model["crime_count"]
hourly_model["target_multiplier"] = np.where(
    hourly_model["baseline_expected_count"] > 0,
    hourly_model["crime_count"] / hourly_model["baseline_expected_count"],
    0.0
)
hourly_model["target_log_multiplier"] = (
    np.log1p(hourly_model["crime_count"]) -
    np.log1p(hourly_model["baseline_expected_count"])
)

# Main target going forward: station-hour crime rate per estimated agent.
hourly_model["target_log_rate"] = (
    np.log(hourly_model["smoothed_crime_count"]) -
    np.log(hourly_model["smoothed_agents"])
)

hourly_model["observed_rate_per_agent"] = np.where(
    hourly_model["estimated_agents"] > 0,
    hourly_model["crime_count"] / hourly_model["estimated_agents"],
    0.0
)

hourly_model["crime_occurred"] = (hourly_model["crime_count"] > 0).astype(int)

hourly_model["baseline_prob_clipped"] = hourly_model["baseline_prob"].clip(1e-8, 1 - 1e-8)
hourly_model["baseline_logit"] = np.log(
    hourly_model["baseline_prob_clipped"] /
    (1 - hourly_model["baseline_prob_clipped"])
)

hourly_model["date_only"] = pd.to_datetime(hourly_model["date_only"], errors="coerce")

missing_baseline = hourly_model["baseline_expected_count"].isna().sum()
print(f"Missing baseline_expected_count after merge: {missing_baseline:,}")
print(f"Zero-crime share in modeling data: {(hourly_model['crime_count'] == 0).mean():.4f}")


# =========================================================
# 8. DROP HIGH-LEAKAGE / DEBUG-ONLY FIELDS FROM CLEAN EXPORT
# =========================================================
print("\nPreparing cleaned incident-level export...")

drop_cols_clean_export = [
    "geometry",
    "stations_in_radius_dist_m",
    "stations_in_radius_dist_m_list",
    "LATITUDE_PUBLIC",
    "LONGITUDE_PUBLIC",
    "x",
    "y",
    "x_snap",
    "y_snap",
    "station_list"
]

cleaned_export = gdf.drop(columns=drop_cols_clean_export, errors="ignore").copy()


# =========================================================
# 9. SAVE OUTPUTS
# =========================================================
print("\nSaving outputs...")
cleaned_export.to_csv(CLEANED_OUTPUT, index=False)
baseline.to_csv(BASELINE_OUTPUT, index=False)
hourly_model.to_csv(MODELING_OUTPUT, index=False)
centroid_shares.to_csv(CENTROID_SHARE_OUTPUT, index=False)

print(f"Saved cleaned incident-level file: {CLEANED_OUTPUT}")
print(f"Saved baseline lookup table:      {BASELINE_OUTPUT}")
print(f"Saved hourly modeling dataset:    {MODELING_OUTPUT}")
print(f"Saved centroid share lookup:      {CENTROID_SHARE_OUTPUT}")


# =========================================================
# 10. SUMMARY
# =========================================================
print("\nSummary:")
print(f"Final incident-level cleaned rows: {len(cleaned_export):,}")
print(f"Baseline lookup rows:              {len(baseline):,}")
print(f"Hourly modeling rows:              {len(hourly_model):,}")
print(f"Station-hour zero rows:            {(hourly_model['crime_count'] == 0).sum():,}")
print(f"Average crimes per day:            {avg_crimes_per_day:.6f}")
print(f"System crimes per hour:            {system_crimes_per_hour:.6f}")

print("\nBaseline preview:")
print(
    baseline[[
        "nearest_station", "day_type", "hour",
        "estimated_station_agents", "baseline_expected_count", "baseline_prob"
    ]].head(12)
)

print("\nHourly modeling dataset preview:")
preview_cols = [
    "nearest_station", "date_only", "hour", "day_type",
    "crime_count", "estimated_agents", "baseline_expected_count",
    "target_log_rate", "observed_rate_per_agent"
]
preview_cols = [c for c in preview_cols if c in hourly_model.columns]
print(hourly_model[preview_cols].head(12))