import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("../normalization/wine_data.csv", usecols=[0,1,2])
df.columns = ["Class Label", "Alcohol", "Malic Acid"]

# print(df.shape)
print(df.sample(5))
# sns.kdeplot(df["Alcohol"])
# color_dict = {1: "red", 2: "blue", 3: "yellow"}
# sns.scatterplot(x=df["Alcohol"], y=df["Malic Acid"], hue=df["Class Label"], palette=color_dict)
# plt.show()

from sklearn.model_selection import train_test_split

input_cs = df.iloc[:, 1:3]
# print(input_cs)

output_cs = df.iloc[:,-3]
# print(output_cs)
x_train, x_test, y_train, y_test  = train_test_split(input_cs, output_cs, test_size=0.2, random_state=0)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=input_cs.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=input_cs.columns)

# print(x_train)
# print(x_test)

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
# ax1.scatter(x_train["Alcohol"], x_train["Malic Acid"], c=y_train)
# ax1.set_title("Before Scaling")
#
# ax2.scatter(x_train["Alcohol"], x_train_scaled["Malic Acid"], c=y_train)
# ax2.set_title("After Scaling")
#
# plt.show()

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
# sns.kdeplot(x_train["Alcohol"], ax=ax1)
# sns.kdeplot(x_train["Malic Acid"], ax=ax1)
# ax1.set_title("Before Scaling")
#
# sns.kdeplot(x_train_scaled["Alcohol"], ax=ax2)
# sns.kdeplot(x_train_scaled["Malic Acid"], ax=ax2)
# ax2.set_title("Before Scaling")
#
# plt.show()

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
#
# sns.kdeplot(x_train["Malic Acid"], ax=ax1)
# ax1.set_title("Before Scaling")
# sns.kdeplot(x_train_scaled["Malic Acid"], ax=ax2)
# ax2.set_title("After Scaling")
# plt.show()

