import numpy
import numpy as np
# dataset = [10, 11, 12, 10, 15, 100, 16, 11, 12, 18, 17, 16, 13, 102, 127]
dataset = [11, 10, 12, 14, 12, 15, 14, 13, 15, 102, 12, 14, 17, 19, 107, 10, 13, 12, 14, 12, 108, 12, 11, 14, 13, 15, 10, 15, 12, 10, 14, 13, 15, 10]
dataset = sorted(dataset)
print(dataset)
# find Q1(25%) and Q3(75%)

q1_q3 = numpy.percentile(dataset, [25, 75])
print(q1_q3)

# find IQR (Q3 - Q1)

iqr = q1_q3[1] - q1_q3[0]
print(iqr)

# find the lower Fence(q1 - 1.5(iqr))

lf = q1_q3[0] - 1.5 * iqr
print(lf)

# find the higher Fence(q3 + 1.5(iqr))

hf = q1_q3[1] + 1.5 * iqr
print(hf)

outliers = [d for d in dataset if lf > d or hf < d]
print(outliers)