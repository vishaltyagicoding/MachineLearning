"""
formula = percentile you want to find / 100 * (N + 1)
"""
import random
import statistics
from os.path import split

import numpy
from pandas.core.interchange.dataframe_protocol import DataFrame

# data should be ascending order sort
data = [78, 82, 84, 88, 91, 93, 94, 96, 98, 99]
percentile = 100
#
# length = len(data)
# pl = round(percentile / 100 * (length + 1), 2)
# print(pl)
#
# marks = 0
# value = f"{pl}".split(".")
# try:
#     if type(pl) == int:
#         marks = data[pl-1]
#
#     else:
#         x = "0."
#         x += value[1]
#
#         pl_int_value = int(value[0]) - 1
#         marks =  data[pl_int_value] + float(x) * (data[pl_int_value + 1] - data[pl_int_value])
# except IndexError:
#     marks = data[length - 1] + float("0." + value[1])
#
# print(f"If you want to get v {percentile} percentile then "
#       f"you have to get {marks} marks")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""

from statistics import quantiles


# Calculate quartiles (returns [25th, 50th, 75th] by default)
quartiles = quantiles(data, )
print(f"Quartiles: {quartiles}")

# You can specify different quantiles
deciles = quantiles(data, n=10)  # 10th, 20th, ..., 90th percentiles
print(f"Deciles: {deciles}")

"""


# TODO : Dies to one dies
"""
dies = []
for _ in range(1000):
    dies.append(random.randint(1, 6))

# print(dies[:5])

import pandas

df = pandas.DataFrame(dies).value_counts().sort_index()/ len(dies)

df = df[:]*100
import matplotlib.pyplot as plt

df.plot(xlabel="Dies Numbers", ylabel="Probability", kind='bar')
plt.show()


print(df)
"""
# TODO : Dies to Two dies
"""
dies = []
for _ in range(1000):
    dies.append(random.randint(1, 6) + random.randint(1, 6))

# print(dies[:5])

import pandas

df = pandas.DataFrame(dies).value_counts().sort_index()/ len(dies)

df = df[:]*100
import matplotlib.pyplot as plt

# df.plot(xlabel="Dies Numbers", ylabel="Probability", kind='bar', kde=True)
# plt.show()


pandas.DataFrame(numpy.cumsum(df)).plot(kind='bar')
plt.show()
"""

# TODO: CDF(Cumulative distribution function)
import pandas
import matplotlib.pyplot as plt
data_ = [1, 2, 3, 4, 5, 6]

probability = []

for i in range(1, len(data_) + 1):
    probability.append(round(i/len(data_)*100, 2))

print(probability)

df = pandas.DataFrame(probability)

df.plot(kind='bar')
plt.show()

