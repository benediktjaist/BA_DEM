import numpy as np

p = np.array([3.5,3,0])
a = np.array([1.5,0,0])
b = np.array([0,1.5,0])


def compute_dir(a,b):
    return b-a


def compute_normal_distance(a,b,p):
    numerator = np.linalg.norm(np.cross(p-a,b-a))
    denominator = np.linalg.norm(b-a)
    return numerator/denominator

x = compute_normal_distance(a,b,p)
print(x)