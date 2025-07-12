import pandas as pd
import numpy as np
d = pd.read_csv("train.csv")
d = d.iloc[:100, :]  # Rows 0-99, all columns
# print(df.sample(10))
# print(df.shape)
# print(df.isnull().sum().sum())

x = d.iloc[:,1:]
y = d.iloc[:,0]
# print(x)
# print(y)



# Step 1 - Apply standard scaling
from sklearn.preprocessing import StandardScaler

# TODO. STEP 1 Standardize the data first
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)

# TODO. STEP 2 Calculate full covariance matrix
cov_matrix = np.cov(scaled_data)
# print(cov_matrix)

# TODO. STEP 2 For eigenvalues/eigenvectors (PCA)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
