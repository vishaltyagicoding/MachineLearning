import pandas as pd
import numpy as np
df = pd.read_csv("orders.csv")
df["date"] = pd.to_datetime(df["date"])
# print(df.sample(5))
# print(df.info())

df['date_year'] = df['date'].dt.year
df['date_month_no'] = df['date'].dt.month
df['date_month_name'] = df['date'].dt.month_name()
df['date_day'] = df['date'].dt.day
df['date_dow'] = df['date'].dt.dayofweek
df['date_dow_name'] = df['date'].dt.day_name()

# is weekend?
df['date_is_weekend'] = np.where(df['date_dow_name'].isin(['Sunday', 'Saturday']), 1,0)


# count week this year
df['date_week'] = df['date'].dt.isocalendar().week

df['quarter'] = df['date'].dt.quarter
df['semester'] = np.where(df['quarter'].isin([1,2]), 1, 2)

# Extract Time elapsed between dates
import datetime

today = datetime.datetime.today()
# print(today-df["date"])
# print((today - df['date']).dt.days)
# Months passed

h = pd.read_csv("messages.csv")
# print(h.sample(5))

h["date"] = pd.to_datetime(h["date"])


h['hour'] = h['date'].dt.hour
h['min'] = h['date'].dt.minute
h['sec'] = h['date'].dt.second
# find minutes
h['time'] = h['date'].dt.time


# Time difference
today - h['date']
# print(today - h['date'])
# in minutes
print((today - h['date'])/np.timedelta64(1,'m'))


# print(h.head())























# print(df.head())