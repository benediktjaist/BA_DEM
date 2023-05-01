import numpy as np

a = 1.0
b = 0.1 * 10
c = 0.3 * 3

result = a - b - c

print(result) # sollte - 0.9 sein

d = 0.1 + 0.1 + 0.1 - 0.2 - 0.1
print(d)

# create two numpy arrays with similar values
arr1 = np.array([1.234567891, 1.234567890], dtype=np.float32)
arr2 = np.array([1.234567890, 1.234567888], dtype=np.float32)

arr3 = np.array([1.234567891, 1.234567890], dtype=np.float64)
arr4 = np.array([1.234567890, 1.234567888], dtype=np.float64)

# subtract the two arrays
result = arr1 - arr2

# better result
result_tuned = arr3 - arr4

# print the result
print(result)
print(result_tuned)
