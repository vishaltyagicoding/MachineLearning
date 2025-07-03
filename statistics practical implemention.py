import statistics
from statistics import variance

import numpy as np
# TODO 1 Measure of central tendency
"""
mean
median
mode
"""
ages = [23, 24, 32, 45, 12, 43, 67, 45, 32, 56, 32, 120]

"""


print(np.mean(ages))
print(np.median(ages))


import statistics as st

print(st.mode(ages))

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(ages, patch_artist=True, orient = 'h')

plt.xlabel('Values')
plt.show()
"""


# TODO 2: 5 Number summery

# check outliers[higher fence - lower fence]

# find 25% quartile and 75% quartile

_25_quartile = np.percentile(ages, 25)
_75_quartile = np.percentile(ages, 75)

# find (IQR) Internal quartile range (Q3 - Q1)

iqr = _75_quartile - _25_quartile
# print(iqr)

# find a lower fence and higher fence

lower_fence = _25_quartile - 1.5 * iqr
upper_fence = _75_quartile + 1.5 * iqr

print(lower_fence, upper_fence)

remove_outliers = [outlier for outlier in ages if not (lower_fence > outlier or outlier > upper_fence)]
# print(remove_outliers)

ages = remove_outliers
# measure of dispersion

# variance for sample data

variance = statistics.variance(ages)
print(variance)

# variance for population data

variance = statistics.pvariance(ages)
print(variance)

# std(standard deviation)

std = statistics.stdev(ages)
print(std)
# std = np.sqrt(variance)
# print(std)


