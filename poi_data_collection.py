from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

# Path to the GeoJSON containing your 8 station boundary polygons
STATIONS_GEOJSON = "data/station_walksheds.geojson"

# Column that uniquely identifies each station polygon (station name/id).
# If None or not found, the script will create station_id from row index.
STATION_ID_COL = "Name"

# Outputs (incremental geopackage + final CSV)
OUT_GPKG = r"Data/station_pois.gpkg"                # <-- CHANGE THIS if you want
GPKG_LAYER = "pois"
OUT_CSV = r"Data/station_pois.csv"                  # <-- CHANGE THIS if you want

# Which OSM tag keys count as POIs (broad defaults are amenity + shop).
POI_KEYS = [
    "amenity",
    "shop",
    "leisure",
    "tourism",
    "office",
    "public_transport",
    "healthcare",
    "craft"
]

# Overpass controls
TIMEOUT_SECONDS = 360
MAX_RETRIES = 8
SLEEP_SECONDS_BETWEEN_STATIONS = 30
COOLDOWN_SECONDS_ON_OVERLOAD = 90

# Polygon simplification (meters). Helps Overpass when polygons have many vertices.
SIMPLIFY_METERS = 20

# If you re-run from scratch, set this True to delete the existing output GPKG first.
DELETE_EXISTING_GPKG = False


# -----------------------------
# Overpass settings
# -----------------------------

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


def overpass_post(query: str, timeout: int = 180, max_retries: int = 10) -> Dict:
    """
    POST an Overpass query with retries/backoff and fallback servers.
    Retries on overload/rate-limit common responses.
    """
    last_err: Optional[Exception] = None

    for attempt in range(max_retries):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]

        try:
            resp = requests.post(
                url,
                data=query.encode("utf-8"),
                headers={"Accept": "application/json"},
                timeout=timeout,
            )

            # Common transient errors from Overpass
            if resp.status_code in (429, 502, 503, 504):
                wait = min(60, (2 ** attempt) + random.uniform(0, 1.5))
                print(f"[Overpass] {resp.status_code} from {url}. Retrying in {wait:.1f}s...")
                time.sleep(wait)

                if resp.status_code in (429, 503, 504):
                    print(f"[Overpass] Overload cooldown: sleeping {COOLDOWN_SECONDS_ON_OVERLOAD}s...")
                    time.sleep(COOLDOWN_SECONDS_ON_OVERLOAD)

                continue

            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            last_err = e
            wait = min(60, (2 ** attempt) + random.uniform(0, 1.5))
            print(f"[Overpass] Request failed ({type(e).__name__}). Retrying in {wait:.1f}s...")
            time.sleep(wait)

    raise RuntimeError(f"Overpass failed after {max_retries} retries. Last error: {last_err}")


# -----------------------------
# Geometry helpers
# -----------------------------

def polygon_to_overpass_poly(geom, simplify_meters: float = 5.0) -> str:
    """
    Convert a Polygon/MultiPolygon to Overpass 'poly:' string format:
        "lat lon lat lon ..."
    We simplify slightly in a projected CRS to reduce vertex count.
    Uses the exterior ring of the largest polygon if MultiPolygon.
    """
    if geom is None or geom.is_empty:
        raise ValueError("Geometry is empty.")

    if geom.geom_type == "Polygon":
        poly = geom
    elif geom.geom_type == "MultiPolygon":
        poly = max(list(geom.geoms), key=lambda g: g.area)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

    # Simplify in meters (Web Mercator) to reduce polygon complexity
    gtmp = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs("EPSG:3857")
    simplified = gtmp.iloc[0].simplify(simplify_meters, preserve_topology=True)
    simplified = gpd.GeoSeries([simplified], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    coords = list(simplified.exterior.coords)
    # Overpass expects lat lon
    return " ".join([f"{y} {x}" for x, y in coords])


def build_poi_query_single_key(poly_string: str, key: str, timeout: int) -> str:
    return f"""
[out:json][timeout:{timeout}];
(
  node["{key}"](poly:"{poly_string}");
  way["{key}"](poly:"{poly_string}");
  relation["{key}"](poly:"{poly_string}");
);
out center tags;
""".strip()


# -----------------------------
# Parsing helpers
# -----------------------------

def overpass_json_to_gdf(data: Dict, station_id: str, poi_keys: List[str]) -> gpd.GeoDataFrame:
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}

        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center")
            if not center:
                continue
            lat, lon = center.get("lat"), center.get("lon")

        if lat is None or lon is None:
            continue

        row = {
            "station_id": station_id,
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "name": tags.get("name"),
            "lat": lat,
            "lon": lon,
            "tags": tags,
            "geometry": Point(lon, lat),
        }
        for k in poi_keys:
            row[k] = tags.get(k)

        rows.append(row)

    cols = ["station_id","osm_type","osm_id","name","lat","lon","tags"] + poi_keys + ["geometry"]
    if not rows:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

    gdf["osm_key"] = gdf["osm_type"].astype(str) + "/" + gdf["osm_id"].astype(str)
    gdf = gdf.drop_duplicates(subset=["station_id", "osm_key"])
    return gdf



# -----------------------------
# IO helpers
# -----------------------------

def ensure_stations_gdf(stations_path: str, station_id_col: Optional[str]) -> gpd.GeoDataFrame:
    """
    Load station polygons from GeoJSON and ensure:
    - it's a GeoDataFrame
    - CRS is EPSG:4326
    - there's a 'station_id' column
    - geometry column is named 'geometry'
    """
    stations = gpd.read_file(stations_path)

    if not isinstance(stations, gpd.GeoDataFrame):
        raise ValueError("Stations file did not load as a GeoDataFrame.")

    # Ensure CRS is WGS84 for Overpass poly
    if stations.crs is None:
        # Most GeoJSONs are implicitly EPSG:4326. If yours isn't, set it correctly here.
        stations = stations.set_crs("EPSG:4326", allow_override=True)
    else:
        stations = stations.to_crs("EPSG:4326")

    # Ensure an ID column
    if station_id_col and station_id_col in stations.columns:
        stations = stations.rename(columns={station_id_col: "station_id"})
    else:
        stations["station_id"] = stations.index.astype(str)

    # Normalize geometry column name
    geom_name = stations.geometry.name
    stations = stations[["station_id", geom_name]].rename(columns={geom_name: "geometry"})
    stations = stations.set_geometry("geometry")

    return stations


def save_incremental(gdf: gpd.GeoDataFrame, out_gpkg: str, layer: str) -> None:
    """
    Append to a GPKG after each station. If it doesn't exist, create it.
    """
    if gdf.empty:
        return

    if not os.path.exists(out_gpkg):
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")
    else:
        # Appends to existing layer; may duplicate if you re-run without deleting.
        gdf.to_file(out_gpkg, layer=layer, driver="GPKG")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    # Optional: delete existing output so you don't get duplicates on rerun
    if DELETE_EXISTING_GPKG and os.path.exists(OUT_GPKG):
        print(f"Deleting existing GPKG: {OUT_GPKG}")
        os.remove(OUT_GPKG)

    stations = ensure_stations_gdf(STATIONS_GEOJSON, STATION_ID_COL)
    print(f"Loaded {len(stations)} station polygons from: {STATIONS_GEOJSON}")

    all_chunks: List[gpd.GeoDataFrame] = []

    for i, row in stations.iterrows():
        station_id = str(row["station_id"])
        print(f"\n=== Station {station_id} ({i+1}/{len(stations)}) ===")

        try:
            poly_str = polygon_to_overpass_poly(row.geometry, simplify_meters=SIMPLIFY_METERS)

            # --- NEW: run Overpass in smaller chunks (one POI key at a time) ---
            station_parts: List[gpd.GeoDataFrame] = []

            for key in POI_KEYS:
                # assumes you have build_poi_query_single_key(...) in your file
                query = build_poi_query_single_key(poly_str, key=key, timeout=TIMEOUT_SECONDS)

                data = overpass_post(query, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)

                # --- NEW: if no elements returned, skip to avoid "Unknown column geometry" ---
                if not data or not data.get("elements"):
                    time.sleep(1.0)
                    continue

                part = overpass_json_to_gdf(data, station_id=station_id, poi_keys=POI_KEYS)
                if part is not None and not part.empty:
                    station_parts.append(part)

                # small pause between key-queries helps with 429s
                time.sleep(5)

            # Combine parts for this station (or create a safe empty GeoDataFrame)
            if station_parts:
                gdf = gpd.GeoDataFrame(pd.concat(station_parts, ignore_index=True), crs="EPSG:4326")
                if "geometry" in gdf.columns:
                    gdf = gdf.set_geometry("geometry")
                else:
                    gdf["geometry"] = gpd.GeoSeries([], crs="EPSG:4326")
                    gdf = gdf.set_geometry("geometry")

                # de-dupe across keys (same feature can match multiple tag keys)
                if "osm_key" not in gdf.columns:
                    gdf["osm_key"] = gdf["osm_type"].astype(str) + "/" + gdf["osm_id"].astype(str)
                gdf = gdf.drop_duplicates(subset=["station_id", "osm_key"])
            else:
                # safe empty gdf so nothing downstream breaks
                cols = ["station_id", "osm_type", "osm_id", "name", "lat", "lon", "tags"] + POI_KEYS + ["geometry"]
                gdf = gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")

            print(f"Pulled {len(gdf)} POIs for station {station_id}")

            # Incremental save so failures don't waste progress
            save_incremental(gdf, out_gpkg=OUT_GPKG, layer=GPKG_LAYER)

            all_chunks.append(gdf)

            # Gentle pause between stations
            time.sleep(SLEEP_SECONDS_BETWEEN_STATIONS)

        except Exception as e:
            print(f"FAILED station {station_id}: {type(e).__name__}: {e}")
            continue

    # Combine everything from this run
    non_empty = [c for c in all_chunks if c is not None and not c.empty]
    if non_empty:
        pois = gpd.GeoDataFrame(pd.concat(non_empty, ignore_index=True), crs="EPSG:4326")
        if "geometry" in pois.columns:
            pois = pois.set_geometry("geometry")
    else:
        pois = gpd.GeoDataFrame(
            columns=["station_id", "osm_type", "osm_id", "name", "lat", "lon", "tags"] + POI_KEYS + ["geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    print(f"\nDone. Total POIs pulled in this run: {len(pois)}")
    print(f"Incremental results saved to: {OUT_GPKG} (layer: {GPKG_LAYER})")

    # Write final CSV (no geometry)
    if OUT_CSV:
        csv_df = pois.drop(columns=["geometry"], errors="ignore").copy()
        csv_df["tags"] = csv_df["tags"].astype(str)
        out_dir = os.path.dirname(OUT_CSV)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        csv_df.to_csv(OUT_CSV, index=False)
        pois.to_file("Data/station_pois.geojson", driver="GeoJSON")
        print(f"CSV written to: {OUT_CSV}")

if __name__ == "__main__":
    main()
