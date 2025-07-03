import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cdf = pd.read_csv("Virat-Kohli-International-Cricket-Centuries.csv")

# TODO 1.# TODO. 3. What is the data type of col?how big is data?
# print(cdf.shape)

# TODO. 2. How does data look like?
# print(cdf.sample(5))

# TODO. 3. What is the data type of col?
# print(cdf.info())

# TODO. 4.Are there any missing values?
# print(cdf.isnull().sum())

# TODO. 5.How does the data look mathematically?
# print(cdf.describe())


# TODO. 6.Are there any duplicate values?
# print(cdf.duplicated().sum())

# TODO. 7.How is the correlation between columns?

# c = pd.read_csv("BATTING STATS - IPL_2016.csv", usecols=["Mat", "Inns", "NO", "Runs", "Avg", "BF", "SR", "100", "50", "4s", "6s"])
#
# correlation = c.corr()["BF"]
# print(correlation)

# TODO 1 count plot

# count = cdf["Position"].value_counts()
# print(count)
# count = cdf["Position"].value_counts().plot(kind="bar")
# plt.show()
# plt.figure(figsize=(30, 10))

# sns.countplot(cdf, x="Against")
# plt.xticks(rotation=45)
# plt.show()
cdf['Against'] = cdf['Against'].str.strip()
#
# count1 = cdf["Against"].value_counts()
# count2 = cdf["Innings"].value_counts()
# print(count1)
# print(count2)



# TODO 2 pie plot
# count = cdf["Position"].value_counts().plot(kind="pie",  autopct="%1.2f%%")
# count = cdf["Against"].value_counts().plot(kind="pie",  autopct="%1.2f%%")
# plt.show()
# TODO 3 Numerical data
# histogram
# plt.hist(cdf["Runs"])
# plt.show()

# TODO 4 displot
sns.histplot(cdf["Position"], kde=True)  # Can also use kind="kde" for just density
plt.show()
# TODO 5 Boxplot

# sns.boxplot(cdf, x="Runs")
# plt.show()




# TODO 6 skew
print(cdf["Position"].skew())