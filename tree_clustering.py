import os
import sys
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

CRIMES_PATH = "data/crimes_with_station_lists.geojson"
TREES_PATH = "data/trees_cleaned.geojson"

OUT_TREES_GEOJSON = "data/trees_clustered_by_crime_centroids.geojson"
OUT_TREES_CSV = "data/trees_clustered_by_crime_centroids.csv"
OUT_CENTROIDS_GEOJSON = "data/crime_centroids_unique.geojson"
OUT_SUMMARY_CSV = "data/cluster_summary.csv"

OUT_CRIMES_FEATURES_GEOJSON = "data/crimes_with_tree_features.geojson"
OUT_CRIMES_FEATURES_CSV = "data/crimes_with_tree_features.csv"

# Tree feature radii (meters)
TREE_RADII_M = [10, 25, 50]

# Use metric CRS for tolerance operations
METRIC_EPSG = 32617  # UTM 17N (meters) → Charlotte

SNAP_TOLERANCE_METERS = 25 # Gets all crime that happened very close to each other so cllustering logic isn't messed up


def ensure_crs(gdf, name):
    if gdf.crs is None:
        print(f"[WARN] {name} missing CRS → assuming EPSG:4326")
        gdf = gdf.set_crs(epsg=4326)
    return gdf


def ensure_points(gdf, name):
    gdf = gdf[~gdf.geometry.isna()]
    gdf = gdf[gdf.geometry.geom_type == "Point"]
    if gdf.empty:
        raise ValueError(f"{name} has no valid points")
    return gdf

def make_unique_centroids(crimes):
    """
    Create centroid points by snapping crimes to 5m grid.
    """
    crimes_metric = crimes.to_crs(epsg=METRIC_EPSG).copy()

    crimes_metric["x"] = crimes_metric.geometry.x
    crimes_metric["y"] = crimes_metric.geometry.y

    # Snap to 5m grid
    crimes_metric["x_snap"] = (crimes_metric["x"] / SNAP_TOLERANCE_METERS).round() * SNAP_TOLERANCE_METERS
    crimes_metric["y_snap"] = (crimes_metric["y"] / SNAP_TOLERANCE_METERS).round() * SNAP_TOLERANCE_METERS

    # Deduplicate snapped coordinates
    unique = crimes_metric.drop_duplicates(subset=["x_snap", "y_snap"]).copy()
    unique = unique.reset_index(drop=True)

    unique["centroid_id"] = unique.index

    # Rebuild geometry from snapped coords
    unique["geometry"] = gpd.points_from_xy(unique["x_snap"], unique["y_snap"])
    centroids = gpd.GeoDataFrame(unique[["centroid_id", "x_snap", "y_snap", "geometry"]],
                                 geometry="geometry",
                                 crs=f"EPSG:{METRIC_EPSG}")

    return centroids


def add_snapped_centroid_id(crimes):
    """
    Adds snapped x/y coordinates and a centroid_id to each crime record based on snapped location.
    Uses SNAP_TOLERANCE_METERS in METRIC_EPSG.
    """
    crimes_m = crimes.to_crs(epsg=METRIC_EPSG).copy()

    crimes_m["x"] = crimes_m.geometry.x
    crimes_m["y"] = crimes_m.geometry.y

    crimes_m["x_snap"] = (crimes_m["x"] / SNAP_TOLERANCE_METERS).round() * SNAP_TOLERANCE_METERS
    crimes_m["y_snap"] = (crimes_m["y"] / SNAP_TOLERANCE_METERS).round() * SNAP_TOLERANCE_METERS

    # Stable integer id per snapped coordinate pair
    crimes_m["centroid_id"] = crimes_m.groupby(["x_snap", "y_snap"]).ngroup()
    return crimes_m


def build_centroids_from_crimes(crimes_m):
    """
    Builds one centroid point per centroid_id at the snapped coordinates.
    """
    centroids = (
        crimes_m.drop_duplicates(subset=["centroid_id"])[["centroid_id", "x_snap", "y_snap"]]
        .copy()
        .reset_index(drop=True)
    )
    centroids["geometry"] = gpd.points_from_xy(centroids["x_snap"], centroids["y_snap"])
    centroids = gpd.GeoDataFrame(centroids, geometry="geometry", crs=f"EPSG:{METRIC_EPSG}")
    return centroids


def compute_tree_features(centroids, trees):
    """
    Returns a DataFrame keyed by centroid_id with:
      - trees_within_{r}m for r in TREE_RADII_M
      - nearest_tree_dist_m
    """
    trees_m = trees.to_crs(epsg=METRIC_EPSG).copy()

    # Avoid index_right collision if trees/crimes were previously spatial-joined and saved
    for col in ["index_right", "index_left"]:
        if col in trees_m.columns:
            trees_m = trees_m.drop(columns=[col])
        if col in centroids.columns:
            centroids = centroids.drop(columns=[col])

    # Nearest tree distance per centroid
    nearest = gpd.sjoin_nearest(
        centroids[["centroid_id", "geometry"]],
        trees_m[["geometry"]],
        how="left",
        distance_col="nearest_tree_dist_m",
    )
    if "index_right" in nearest.columns:
        nearest = nearest.drop(columns=["index_right"])

    feats = nearest[["centroid_id", "nearest_tree_dist_m"]].copy()

    # Tree counts within buffers around each centroid
    for r in TREE_RADII_M:
        buffers = centroids[["centroid_id", "geometry"]].copy()
        buffers["geometry"] = buffers.geometry.buffer(r)

        j = gpd.sjoin(
            trees_m[["geometry"]],
            buffers,
            how="inner",
            predicate="within",
        )
        if "index_right" in j.columns:
            j = j.drop(columns=["index_right"])

        counts = (
            j.groupby("centroid_id")
            .size()
            .rename(f"trees_within_{r}m")
            .reset_index()
        )
        feats = feats.merge(counts, on="centroid_id", how="left")

    # Fill missing counts with 0
    for r in TREE_RADII_M:
        col = f"trees_within_{r}m"
        if col in feats.columns:
            feats[col] = feats[col].fillna(0).astype(int)

    return feats

def main():
    crimes = gpd.read_file(CRIMES_PATH)
    trees = gpd.read_file(TREES_PATH)

    crimes = ensure_crs(crimes, "crimes")
    trees = ensure_crs(trees, "trees")

    crimes = ensure_points(crimes, "crimes")
    trees = ensure_points(trees, "trees")

    # Snap crimes to grid and create centroid_id for each snapped location
    crimes_m = add_snapped_centroid_id(crimes)

    # Build snapped crime centroids (one point per centroid_id)
    centroids = build_centroids_from_crimes(crimes_m)
    print(f"[INFO] Unique crime centroids (snapped): {len(centroids):,}")

    # Compute tree-environment features per centroid and attach to every crime
    centroid_features = compute_tree_features(centroids, trees)
    crimes_with_info = crimes_m.merge(centroid_features, on="centroid_id", how="left")

    # Export crimes with features
    crimes_out = crimes_with_info.to_crs(epsg=4326)
    crimes_out.to_file(OUT_CRIMES_FEATURES_GEOJSON, driver="GeoJSON")

    crimes_out = crimes_out.copy()
    crimes_out["lon"] = crimes_out.geometry.x
    crimes_out["lat"] = crimes_out.geometry.y
    pd.DataFrame(crimes_out.drop(columns="geometry")).to_csv(OUT_CRIMES_FEATURES_CSV, index=False)
    # Nearest join (trees → centroid)
    trees_proj = trees.to_crs(epsg=METRIC_EPSG)

    joined = gpd.sjoin_nearest(
        trees_proj,
        centroids,
        how="left",
        distance_col="dist_m"
    )

    # Export centroids info in case it's needed later
    centroids.to_crs(epsg=4326).to_file(OUT_CENTROIDS_GEOJSON, driver="GeoJSON")

    # Export trees
    trees_out = joined.to_crs(epsg=4326)
    trees_out.to_file(OUT_TREES_GEOJSON, driver="GeoJSON")

    trees_out["lon"] = trees_out.geometry.x
    trees_out["lat"] = trees_out.geometry.y
    pd.DataFrame(trees_out.drop(columns="geometry")).to_csv(OUT_TREES_CSV, index=False)

    # Summary
    summary = (
        joined.groupby("centroid_id")
        .agg(
            tree_count=("centroid_id", "size"),
            mean_dist_m=("dist_m", "mean"),
            max_dist_m=("dist_m", "max")
        )
        .reset_index()
        .sort_values("tree_count", ascending=False)
    )
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print("[DONE] Trees clustered using 15m-snapped crime centroids.")

    crs_plot = "EPSG:2264"  # NC State Plane feet
    crimes = crimes.to_crs(crs_plot)
    trees_out = trees_out.to_crs(crs_plot)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot crimes first (background)
    crimes.plot(ax=ax, markersize=3, alpha=0.35)

    # Plot trees on top, colored by centroid cluster
    trees_out.plot(ax=ax, column="centroid_id", markersize=8, legend=False)

    ax.set_title("Tree clusters (by nearest snapped crime centroid) over crimes")
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    main()

