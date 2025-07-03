import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

titanic = sns.load_dataset('titanic')
iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

# print(titanic.info())
# print(iris.info())
# print(tips.info())


# scatter plot
print(tips.info())
print(tips.head())
# sns.scatterplot(tips, x='tip', y='total_bill', hue='sex')
# sns.scatterplot(tips, x='tip', y='total_bill', hue='sex', style='smoker')
sns.scatterplot(tips, x='tip', y='total_bill', hue='sex', style='smoker', size='size')
plt.show()


# Bar plot

# titanic["survived"].value_counts().plot(kind="bar", k="sex")

# sns.barplot(y=titanic["fare"], x=titanic["pclass"], hue=titanic["sex"])

# sns.boxplot(y=titanic["age"], x=titanic["sex"], hue=titanic["survived"])

# sns.distplot(titanic[titanic["survived"]==0]["age"], hist=False)
# sns.distplot(titanic[titanic["survived"]==1]["age"], hist=False)
# plt.show()

# c=pd.crosstab(titanic["pclass"],titanic["survived"])
# print(c)

# sns.heatmap(pd.crosstab(titanic["pclass"],titanic["survived"]))
# plt.show()

# print(titanic.groupby("pclass")["survived"].mean()*100)
# print(titanic.groupby("sex")["survived"].mean()*100)
# print(titanic.groupby("embarked")["survived"].mean()*100)


# sns.pairplot(iris, hue="species")
# plt.show()
# print(iris.info())



# print(flights.info())
# print(flights.head())
new=flights.groupby("year")["passengers"].sum().reset_index()
print(new)


# Then plot
# sns.lineplot(x=new["year"], y=new["passengers"])
#
# plt.show()

#
