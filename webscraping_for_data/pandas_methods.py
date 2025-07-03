import pandas as pd

df = pd.read_csv("companies.csv")
# print(df.head())
# print(df.info())
# print(df.shape)
d = 0
for data in df['Salary']:
    try:
        if "L" in data:
            data = data.replace("L", "")
            data = round(float(data) * 100000)
        elif "k" in data:
            data = data.replace("k", "")
            data = float(data) * 1000
        d += (data)

    except:
        pass

print(d/10000)