import numpy as np

def func(x):
    return x ** 2 - 10 * np.sin(x)

def func2(x, y):
    return ((x - 2)**2 + (y - 3)**2)
    

def bound_velocity(v, v_min, v_max):
    for i in range(len(v)):
        if v[i] > v_max:
            v[i] = v_max
        elif v[i] < v_min:
            v[i] = v_min
