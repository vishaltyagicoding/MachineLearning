import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('train.csv',usecols=['GarageQual','FireplaceQu','SalePrice'])
x = df.drop(["SalePrice"],axis=1)
y = df["SalePrice"]
# print(x.sample(5))
# print(x.isnull().sum())
# print(x.isnull().mean()*100)


df[df['FireplaceQu']=='Gd']["SalePrice"].plot(kind = "kde", color='green')
# plt.show()

x["FireplaceQu"] = x["FireplaceQu"].fillna("Gd")
x["GarageQual"] = x["GarageQual"].fillna("TA")
# print(x.isnull().mean()*100)


# df[df['FireplaceQu']=='Gd']["SalePrice"].plot(kind = "kde", color='red')
# plt.show()