from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

x, y_true = make_blobs(n_samples=500, centers=3, cluster_std=3, random_state=42, n_features=2)
# Scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

df = pd.DataFrame(x_scaled, columns=['Feature1', 'Feature2'])

# Visualize the data
sns.scatterplot(x=df["Feature1"], y=df["Feature2"])
plt.title('Data Points with True Clusters')
plt.show()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x_scaled)
y_kmeans = kmeans.predict(x_scaled)
# Add cluster labels to the DataFrame
df['Cluster'] = y_kmeans

# Visualize the clusters
sns.scatterplot(x=df["Feature1"], y=df["Feature2"], hue=df["Cluster"], palette='viridis')
plt.title('KMeans Clustering Results')
plt.show()



