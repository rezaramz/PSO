import numpy as np

def func(x):
    return x ** 2 - 10 * np.sin(x)

def func2(x):
    return ((x[:, 0] - 5)**2 + (x[:, 1] - 1)**2)
    

def bound(v, v_min, v_max):
    ndim = np.size(v_min)
    Np = np.shape(v)[0]    
    for i in range(Np):
        for j in range(ndim):
            if v[i, j] > v_max[j]:
                v[i, j] = v_max[j]
            elif v[i, j] < v_min[j]:
                v[i, j] = v_min[j]
    
