from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

x, y_true = make_moons(n_samples=500, noise=0.05, random_state=42)
# Scale the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

df = pd.DataFrame(x_scaled, columns=['Feature1', 'Feature2'])

# Visualize the data
sns.scatterplot(x=df["Feature1"], y=df["Feature2"])
plt.title('Data Points with True Clusters')
plt.show()

# Apply KMeans clusteringf
dbs = DBSCAN(eps=0.3, min_samples=2)
dbs_labels = dbs.fit_predict(x_scaled)
# Add cluster labels to the DataFrame
df['Cluster'] = dbs_labels
# Visualize the clusters
sns.scatterplot(x=df["Feature1"], y=df["Feature2"], hue=df["Cluster"], palette='viridis')
plt.title('DBSCAN Clustering Results')
plt.show()
