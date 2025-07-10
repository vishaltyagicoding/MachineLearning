import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df =  pd.read_csv("Social_Network_Ads.csv", usecols=['Age', 'EstimatedSalary',"Purchased"])

# TODO 1. How big is data?
# print(df.shape)

# TODO. 2. How does data look like?
# print(df.sample(5))


# TODO. 3. What is the data type of col?
# print(df.info())

# TODO. 4.Are there any missing values?
# print(df.isnull().sum())


# TODO. 5.How does the data look mathematically?
"""
The describe() function in pandas provides a quick statistical summary of your DataFrame or Series.
 It's one of the most useful functions for exploratory data analysis.
 like count, mean, min, std, max etc.
"""
# print(df.describe())


# TODO. 6.Are there any duplicate values?
# print(df.duplicated().sum())

# TODO. 7.How is the correlation between columns?
# print(df.corr())


# TODO 1. CountPlot
#
# count_age = df["Age"].value_counts()
# count_Salary = df["EstimatedSalary"].value_counts()
# count_Purchase = df["Purchased"].value_counts()
# print(count_age)
# print(count_Salary)
# print(count_Purchase)

# sns.countplot(x="Purchased", data=df)

# sns.countplot(x="EstimatedSalary", data=df)
# plt.xticks(rotation=45)

# sns.countplot(x="Age", data=df)
# plt.xticks(rotation=90)
#
# plt.show()



# # TODO 2 pie plot
#
# df["Purchased"].value_counts().plot(kind='pie',  autopct="%1.2f%%")
# plt.show()


# TODO 3 Numerical data
# histogram
# plt.hist(df["Purchased"])
# plt.show()

# TODO 4 displot
# sns.histplot(df["Age"], kde=True)  # Can also use kind="kde" for just density
# sns.histplot(df["EstimatedSalary"], kde=True)  # Can also use kind="kde" for just density
# plt.show()

# TODO 5 Boxplot
# find outliers also
# sns.boxplot(df, x="EstimatedSalary")
# sns.boxplot(df, x="Age")
# plt.show()


# TODO 6 skew
# print(df["EstimatedSalary"].skew())

# multivariate analysis
# TODO 1 Dist plot (distribution)

# sns.distplot(df[df["Purchased"]==0]["Age"], hist=False)
# sns.distplot(df[df["Purchased"]==1]["Age"], hist=False)
# plt.show()

# TODO 2 bar plot (distribution)
# sns.barplot(y=df["Age"], x=df["Purchased"])
# plt.show()


# c=pd.crosstab(df["Age"],df["Purchased"])
# print(c)
# TODO 1. Find outliers and remove


# salary = df["EstimatedSalary"]
# age = df["Age"]
# print(dataset)
def find_outliers(dataset):
    dataset = sorted(dataset)

    # find Q1(25%) and Q3(75%)

    q1, q3 = np.percentile(dataset, [25, 75])
    # print(q1)
    # print(q3)

    # find IQR (Q3 - Q1)

    iqr = q3-q1
    # print(iqr)

    # find the lower Fence(q1 - 1.5(iqr))

    lf  = q1 - 1.5 * iqr
    # print(lf)

    # find the higher Fence(q3 + 1.5(iqr))

    hf = q3 + 1.5 * iqr
    # print(hf)


    new_dataset = [i for i in dataset if lf < i or hf > i]
    return new_dataset


# salary = find_outliers(dataset=salary)
# df["EstimatedSalary"] = salary
# print(df["EstimatedSalary"])



# age = find_outliers(dataset=age)
# df["Age"] = age
# print(df["Age"])

# train to split

from sklearn.model_selection import train_test_split
input_columns = df.iloc[:,0:2]
output_columns = df.iloc[:,-1]
# print(input_columns.head())
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(input_columns, output_columns, test_size=0.2)

# standardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train_data)

x_train_scaled = scaler.transform(x_train_data)
x_test_scaled = scaler.transform(x_test_data)

# print(x_train_scaled)
# print(x_test_scaled)


x_train_scaled = pd.DataFrame(x_train_scaled, columns=input_columns.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=input_columns.columns)



# print(x_train_scaled)
# print(x_test_scaled)


# compare normal data vs scaler data
# inc = np.round(input_columns.describe(), decimals=2)
# print(inc)
#
# inc = np.round(x_train_scaled.describe(), decimals=2)
# print(inc)

# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
# ax1.scatter(x_train_data["Age"], x_train_data["EstimatedSalary"])
# ax1.set_title("Before Scaling")
# ax2.scatter(x_train_scaled["Age"], x_train_scaled["EstimatedSalary"])
# ax2.set_title("After Scaling")
#
# plt.show()

# sns.kdeplot(x_train_scaled, x="Age")
# sns.kdeplot(x_train_scaled, x="EstimatedSalary")
# plt.show()

# train model

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train_scaled, y_train_data)

predictions = logreg.predict(x_test_scaled)
# print(predictions)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test_data, predictions)
print(accuracy)