# Charlotte Transit Safety Project

A data science project analyzing crime, safety, and environmental factors around Charlotte Area Transit System (CATS) stations. This project integrates CMPD crime data, ridership data, points of interest (POI), weather data, and geospatial analysis to understand safety patterns near transit stops — and ultimately build predictive models to support safer transit planning.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Notebooks and Scripts](#notebooks-and-scripts)
- [Modeling](#modeling)
- [Visualizations](#visualizations)
- [Setup and Installation](#setup-and-installation)
- [Environment Variables](#environment-variables)
- [Requirements](#requirements)

---

## Overview

Charlotte's light rail and bus network (CATS) has faced growing public safety concerns. This project takes a data-driven approach to:

- **Cluster and categorize** crime incidents reported by CMPD near transit stations
- **Link crime data with weather conditions** to explore environmental correlations
- **Assign crimes to transit stations** by proximity radius
- **Analyze points of interest (POIs)** — bars, shelters, businesses — around stations
- **Model safety risk** using machine learning (XGBoost, Agent-Based Models)
- **Visualize patterns** geospatially and export datasets for Power BI dashboards

---

## Project Structure

```
Charlotte-Transit-Safety-Project/
│
├── Data/                          # Raw and processed datasets
├── HMIS.Data/                     # Homeless Management Information System data
├── model/                         # Saved model artifacts
├── cache/                         # Cached intermediate outputs
│
├── CMPD_Crime_Clustering.py       # Clusters CMPD crime types using OpenAI embeddings + KMeans
├── Development_EDA.py             # Exploratory data analysis scripts
├── FirstFile.py                   # Initial data loading / setup
├── main.py                        # Main pipeline entry point
├── model_preprocess.py            # Feature engineering and preprocessing for modeling
├── data_visuals.py                # Visualization utilities
├── poi_data_collection.py         # Collects points of interest data
├── powerbi_datasets.py            # Exports cleaned datasets for Power BI
├── tree_clustering.py             # Tree-based clustering logic
│
├── abm_xgboost_model_comparison_updated.ipynb  # ABM vs XGBoost model comparison
├── cmpd_cleaning.ipynb            # Cleans raw CMPD crime data
├── cmpd_weather_link.ipynb        # Joins crime data with weather records
├── converter_To_GeoJson.ipynb     # Converts datasets to GeoJSON format
├── eda.ipynb                      # General exploratory data analysis
├── geo_vis.ipynb                  # Geospatial visualizations
├── poi_eda.ipynb                  # EDA on points of interest data
├── poi_preprocessing_env_clusters_tree_like.ipynb  # POI feature engineering & clustering
├── ridership_eda.ipynb            # CATS ridership data analysis
├── station_assignment.ipynb       # Assigns crimes/POIs to nearest transit stations
├── station_radius.ipynb           # Defines station catchment radii
├── tree_EDA.ipynb                 # EDA for tree-based model features
├── test.ipynb                     # Testing and validation notebook
│
├── pyproject.toml                 # Project dependencies (uv)
├── uv.lock                        # Locked dependency versions
└── .python-version                # Python version pin
```

---

## Data Sources

| Dataset | Description |
|---|---|
| **CMPD Crime Data** | Charlotte-Mecklenburg Police Department incident reports |
| **CATS Station Data** | Charlotte Area Transit System station locations |
| **Weather Data** | Linked weather conditions at time/location of crime incidents |
| **Points of Interest (POI)** | Nearby businesses, shelters, bars, and other facilities |
| **HMIS Data** | Homeless Management Information System records for contextual analysis |
| **Ridership Data** | CATS ridership counts by station and time period |

---

## Notebooks and Scripts

### Data Cleaning & Preparation
- **`cmpd_cleaning.ipynb`** — Cleans and standardizes raw CMPD crime incident data
- **`cmpd_weather_link.ipynb`** — Merges crime records with historical weather data
- **`converter_To_GeoJson.ipynb`** — Converts tabular data to GeoJSON for geospatial use

### Exploratory Data Analysis
- **`eda.ipynb`** — General-purpose EDA on crime and station data
- **`poi_eda.ipynb`** — Analyzes the distribution and types of POIs near stations
- **`ridership_eda.ipynb`** — Explores ridership trends across the CATS network
- **`tree_EDA.ipynb`** — EDA focused on tree-model features

### Station Analysis
- **`station_assignment.ipynb`** — Spatially assigns crimes and POIs to the nearest transit station
- **`station_radius.ipynb`** — Defines and experiments with different station buffer radii

### Crime Clustering
- **`CMPD_Crime_Clustering.py`** — Uses OpenAI `text-embedding-3-small` to embed NIBRS crime descriptions, then applies KMeans (n=10) to group them into thematic clusters. Cluster labels are generated using GPT-4o-mini.

### Feature Engineering
- **`poi_preprocessing_env_clusters_tree_like.ipynb`** — Processes POI features into environmental cluster variables suitable for tree-based models
- **`model_preprocess.py`** — Full preprocessing pipeline for model-ready feature sets

### Geospatial Visualization
- **`geo_vis.ipynb`** — Maps crime density, station locations, and POI overlays

---

## Modeling

### XGBoost vs. Agent-Based Model (ABM)
**`abm_xgboost_model_comparison_updated.ipynb`**

This notebook compares two modeling approaches for predicting safety risk around transit stations:

- **XGBoost** — Gradient-boosted tree model trained on crime counts, POI features, ridership, weather, and station-level attributes
- **Agent-Based Model (ABM)** — Simulates individual agent behaviors (commuters, incidents) to model emergent safety patterns

Results and performance metrics are compared to evaluate which approach better captures transit safety dynamics.

---

## Visualizations

Visualizations are generated via `data_visuals.py` and the various EDA notebooks, including:

- Crime heatmaps by station proximity
- Cluster distributions of crime types
- POI density plots around stations
- Ridership vs. crime correlation charts
- Geospatial maps (GeoJSON-based, viewable in tools like Kepler.gl or ArcGIS)
- Power BI-ready exports via `powerbi_datasets.py`

---

## Setup and Installation

This project uses [**uv**](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/alundy98/Charlotte-Transit-Safety-Project.git
cd Charlotte-Transit-Safety-Project

# Install dependencies using uv
uv sync
```

Alternatively, install dependencies manually with pip using the packages listed in `pyproject.toml`.

---

## Environment Variables

Create a `.env` file in the project root with the following:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

This is required for `CMPD_Crime_Clustering.py`, which uses the OpenAI API for embedding generation and cluster labeling.

---

## Requirements

- Python 3.11+ (see `.python-version`)
- Key libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `openai`, `geopandas`, `folium`, `matplotlib`, `seaborn`, `dotenv`

---

## Contributing

This is an active research project. Feel free to open issues or pull requests for suggestions, bug fixes, or new analyses.

---

## License

This project is for academic and public benefit research purposes. Data sourced from public city of Charlotte and CMPD open data portals.
