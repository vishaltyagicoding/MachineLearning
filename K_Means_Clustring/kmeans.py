import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv('K_Means_Clustring\\student_clustering.csv')

# print(df.head())

# plotting the data points on 2-D graph
# import matplotlib.pyplot as plt
# plt.scatter(df['cgpa'], df['iq'], color='black')
# plt.xlabel('cgpa')
# plt.ylabel('iq')
# plt.show()


# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(df)
#     wcss.append(kmeans.inertia_)


# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(df)

# print(y_kmeans)
df['cluster'] = y_kmeans
# print(df)
# sepret the data points by their clusters
df0 = df[df['cluster'] == 0]
df1 = df[df['cluster'] == 1]
df2 = df[df['cluster'] == 2]
df3 = df[df['cluster'] == 3]
print(df0.head())
print(df1.head())
print(df2.head())
print(df3.head())



plt.scatter(df['cgpa'], df['iq'], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel('cgpa')
plt.ylabel('iq')
plt.show()