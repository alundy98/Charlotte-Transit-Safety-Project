import pandas as pd
import os
import dotenv
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sklearn.cluster import KMeans

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

df = pd.read_csv(r'data\cmpd_cleaned.csv')

# Getting unique crime types
crime_types = df['HIGHEST_NIBRS_DESCRIPTION'].unique().tolist()

response = client.embeddings.create(
    input=crime_types,
    model="text-embedding-3-small"
)

# Generating embeddings and storing them in a dataframe
embeddings = np.array([d.embedding for d in response.data])

crime_types_df = pd.DataFrame({
    'HIGHEST_NIBRS_DESCRIPTION': crime_types,
})

# Generating K-Means clusters
n_clusters = 10  # adjust 5–10 depending on desired grouping
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
crime_types_df["cluster"] = kmeans.fit_predict(embeddings)

crime_types_df["embedding"] = list(embeddings)
cluster_centers = kmeans.cluster_centers_

# Generating Cluster Titles using OpenAI
cluster_titles = {}
for cluster_id in sorted(crime_types_df["cluster"].unique()):
    cluster_methods = crime_types_df.loc[crime_types_df["cluster"] == cluster_id, 'HIGHEST_NIBRS_DESCRIPTION'].tolist()
    sample_crimes = cluster_methods[:20]  # limit for token efficiency

    prompt = f"""
    You are labeling groups of crimes.
    Given the following examples of crimes:

    {sample_crimes}

    Write a concise 2–5 word title summarizing the shared theme of this cluster.
    The clusters are representitive of crimes reported by law enforcement, the name should relfect the types of crimes that are being referred to.
    Avoid generic terms like "Miscellaneous" or "Other".,
    such as "Violent Crime", "Property Crime", or "Theft".
    """

    chat_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You are an expert in crime analysis and data science."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    cluster_title = chat_response.choices[0].message.content.strip()
    cluster_titles[cluster_id] = cluster_title
    print(f"Cluster {cluster_id}: {cluster_title}")

crime_types_df["cluster_title"] = crime_types_df["cluster"].map(cluster_titles)

# Adding the cluster title column to the original dataframe
df = df.merge(crime_types_df[['HIGHEST_NIBRS_DESCRIPTION', "cluster", "cluster_title"]], on='HIGHEST_NIBRS_DESCRIPTION', how="left")

df.to_csv('data\\cmpd_cleaned_with_clusters.csv', index=False)