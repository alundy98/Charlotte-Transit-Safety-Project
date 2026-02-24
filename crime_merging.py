import pandas as pd
import geopandas as gpd

crime_weather = pd.read_csv("data/crime_with_weather.csv")
crime_trees = pd.read_csv("data/crimes_with_tree_features.csv")

# 1) Normalize column names to avoid hidden whitespace causing KeyErrors
crime_weather.columns = crime_weather.columns.str.strip()
crime_trees.columns = crime_trees.columns.str.strip()

# 2) Find shared columns and turn into a plain Python list
common_cols = crime_weather.columns.intersection(crime_trees.columns).tolist()

# 3) Usually geometry should NOT be a join key (and may not exist in both)
common_cols = [c for c in common_cols if c != "geometry"]

if not common_cols:
    raise ValueError("No common columns found to merge on (after stripping names).")

print("Merging on:", common_cols)

crime_tree_weather = pd.merge(
    crime_weather,
    crime_trees,
    how="left",
    on=common_cols
)

# pick the best available lon/lat columns
if {"lon", "lat"}.issubset(crime_tree_weather.columns):
    xcol, ycol = "lon", "lat"
elif {"LONGITUDE_PUBLIC", "LATITUDE_PUBLIC"}.issubset(crime_tree_weather.columns):
    xcol, ycol = "LONGITUDE_PUBLIC", "LATITUDE_PUBLIC"
else:
    raise ValueError("No usable lon/lat columns found to create geometry.")

# ensure numeric
crime_tree_weather[xcol] = pd.to_numeric(crime_tree_weather[xcol], errors="coerce")
crime_tree_weather[ycol] = pd.to_numeric(crime_tree_weather[ycol], errors="coerce")

# drop rows without coords (or keep them; they’ll have geometry = None)
gdf = gpd.GeoDataFrame(
    crime_tree_weather,
    geometry=gpd.points_from_xy(crime_tree_weather[xcol], crime_tree_weather[ycol]),
    crs="EPSG:4326"
)

gdf.to_file("data/crime_weather_tree.geojson", driver="GeoJSON")
gdf.drop(columns=["geometry"], errors="ignore").to_csv("data/crime_weather_tree.csv", index=False)