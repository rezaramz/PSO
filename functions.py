import numpy as np

def func(x):
    return x ** 2 - 10 * np.sin(x)

def func2(x): 
    return ((x[:, 0] - 5)**2 + (x[:, 1] - 1)**2)
    

def func3(t):
    Np = np.shape(t)[0]
#    ndim = np.shape(t)[1]
    x = np.linspace(0, 5, 100)
    y = x ** 2 + x
    cost = np.zeros([Np, 1])
    for i in range(len(x)):
        cost += np.reshape(((t[:, 0] * (x[i]**2) + t[:, 1] * x[i] + t[:, 2] - y[i]) ** 2), (Np, 1))
    return cost
    

def func4(x):
#    Np = np.shape(x)[0]
#    ndim = np.shape(x)[1]
    res = -np.cos(x[:, 0]) * np.cos(x[:, 1])
    t1 = -(x[:, 0] - np.pi)**2
    t2 = -(x[:, 1] - np.pi)**2
    res *= np.exp(t1 + t2)
    return res


def bound(v, v_min, v_max):
    ndim = np.size(v_min)
    Np = np.shape(v)[0]    
    for i in range(Np):
        for j in range(ndim):
            if v[i, j] > v_max[j]:
                v[i, j] = v_max[j]
            elif v[i, j] < v_min[j]:
                v[i, j] = v_min[j]
                

def is_converged(data, index, interval = 100, tolerance = 1e-6):
    target = np.array(data[index - interval : index])
    grad = np.abs((target.max() - target.min()) / target[-1])
    return grad < tolerance


#def is_converged(data, index, interval=200, tolerance = 1e-6):
#    grad = np.abs((data[index] - data[index - interval]) / data[index])
#    return (grad < tolerance)
    
    
    
