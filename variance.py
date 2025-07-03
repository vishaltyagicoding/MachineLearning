"""
variance formula =  σ2 = f(x − ¯x)2 / n
"""
import statistics

data = [3, 2, 1, 5, 4]
mean = statistics.mean(data)

variance = statistics.variance(data)
print(variance)

x = 0
for d in data:
    x += (d - mean)**2

print(x / len(data))
