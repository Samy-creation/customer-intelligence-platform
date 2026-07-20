from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

# Read customer feature data from the processed data folder
# This file should include columns such as Recency, Frequency, and Monetary
df = pd.read_csv("data/processed/customer_features.csv")

# Apply a log transform to the Frequency and Monetary columns
# This reduces the effect of very large values and improves clustering stability
df["Frequency_log"] = np.log1p(df["Frequency"])
df["Monetary_log"] = np.log1p(df["Monetary"].clip(lower=0))

# Select the features for clustering and scale them to have zero mean and unit variance
features = df[["Recency", "Frequency_log", "Monetary_log"]]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Run KMeans clustering to segment customers into 4 groups
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(features_scaled)

# Map cluster labels to meaningful segment names for easier interpretation
segment_names = {
    0: "VIP",
    1: "Dormant",
    2: "Occasional",
    3: "At Risk",
}
df["Segment"] = df["Cluster"].map(segment_names)

# Save the resulting customer segments to a CSV file for further analysis or use in other parts of the application
df.to_csv("data/processed/customer_segments.csv", index=False)
