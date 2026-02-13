import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Data/crime_within_station_walksheds.csv")
# Datetime cleaning 
df["DATE_INCIDENT_BEGAN"] = pd.to_datetime(
    df["DATE_INCIDENT_BEGAN"],
    errors="coerce"
)
# Restrict to valid analytical range
df = df[
    (df["DATE_INCIDENT_BEGAN"].dt.year >= 1970) &
    (df["DATE_INCIDENT_BEGAN"].dt.year <= 2026)
]

df["year"] = df["DATE_INCIDENT_BEGAN"].dt.year
df["month"] = df["DATE_INCIDENT_BEGAN"].dt.month
plt.figure()
plt.scatter(
    df["LONGITUDE_PUBLIC"],
    df["LATITUDE_PUBLIC"],
    alpha=0.5
)
plt.title("Spatial Distribution of Crime Incidents")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

#Crime types by cluster
cluster_crime = (
    df.groupby(["cluster_title", "HIGHEST_NIBRS_DESCRIPTION"])
      .size()
      .reset_index(name="count")
)
top_cluster_crime = cluster_crime.sort_values("count", ascending=False).head(10)

plt.figure()
sns.barplot(
    data=top_cluster_crime,
    x="count",
    y="cluster_title",
    hue="HIGHEST_NIBRS_DESCRIPTION"
)
plt.title("Crime Types by Cluster")
plt.xlabel("Incident Count")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

#Seasonality heatmap 
heatmap_data = (
    df.groupby(["month", "HIGHEST_NIBRS_DESCRIPTION"])
      .size()
      .unstack(fill_value=0)
)

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="Reds")
plt.title("Crime Seasonality by Type")
plt.xlabel("Crime Type")
plt.ylabel("Month")
plt.tight_layout()
plt.show()

# Latitude distribution by crime type
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df,
    x="HIGHEST_NIBRS_DESCRIPTION",
    y="LATITUDE_PUBLIC"
)
plt.xticks(rotation=45, ha="right")
plt.title("Latitude Distribution by Crime Type")
plt.tight_layout()
plt.show()

#  Crime CLUSTERS by Patrol Division 
sns.countplot(
    data=df,
    y="cluster_title",
    hue="CMPD_PATROL_DIVISION"
)

plt.title("Crime Cluster Distribution by Patrol Division")
plt.xlabel("Incident Count")
plt.ylabel("Crime Cluster")

# Embedded legend: Top 3 crime types per cluster 
top_crimes_per_cluster = (
    df.groupby(["cluster_title", "HIGHEST_NIBRS_DESCRIPTION"])
      .size()
      .reset_index(name="count")
      .sort_values(["cluster_title", "count"], ascending=[True, False])
      .groupby("cluster_title")
      .head(3)
)

legend_table = (
    top_crimes_per_cluster
    .groupby("cluster_title")["HIGHEST_NIBRS_DESCRIPTION"]
    .apply(lambda x: ", ".join(x))
    .reset_index()
)

legend_text = "\n\n".join(
    f"{row['cluster_title']}:\n{row['HIGHEST_NIBRS_DESCRIPTION']}"
    for _, row in legend_table.iterrows()
)

plt.gcf().text(
    1.02, 0.5,
    legend_text,
    fontsize=9,
    va="center"
)

plt.tight_layout()
plt.show()

# Faceted spatial plots by crime type
g = sns.FacetGrid(
    df,
    col="HIGHEST_NIBRS_DESCRIPTION",
    col_wrap=3,
    height=3
)

g.map_dataframe(
    sns.scatterplot,
    x="LONGITUDE_PUBLIC",
    y="LATITUDE_PUBLIC",
    alpha=0.5
)

g.set_titles("{col_name}")
g.fig.suptitle("Spatial Distribution by Crime Type", y=1.05)
plt.show()

#  Cluster composition heatmap
cluster_heatmap = (
    df.groupby(["cluster_title", "HIGHEST_NIBRS_DESCRIPTION"])
      .size()
      .unstack(fill_value=0)
)

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_heatmap, cmap="Blues")
plt.title("Crime Composition by Cluster")
plt.xlabel("Crime Type")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

# Multivariate relationships
sns.pairplot(
    df,
    vars=["LATITUDE_PUBLIC", "LONGITUDE_PUBLIC", "month"],
    hue="cluster_title",
    corner=True
)
plt.show()



