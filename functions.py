import numpy as np

def func(x):
    return x ** 2 - 10 * np.sin(x)

def func2(x): 
#    return ((x[:, 0] - 5)**2 + (x[:, 1] - 1)**2)
    return (x[0] - 5) ** 2 + (x[1] - 1) ** 2
    

def func3(t):
#    ndim = np.shape(t)[1]
    x = np.linspace(0, 5, 10)
    y = x ** 2 + 2 * x + 1
    return np.sum( np.abs(t[0] * (x ** 2) + t[1] * x + t[2] - y) )


def rosenbrock(x):
    a = x[0]
    b = x[1]
    return 100 * (b - a**2)**2 + (1 - a)**2
    
    

def func4(x):
#    Np = np.shape(x)[0]
#    ndim = np.shape(x)[1]
    res = -np.cos(x[0]) * np.cos(x[1])
    t1 = -(x[0] - np.pi)**2
    t2 = -(x[1] - np.pi)**2
    res *= np.exp(t1 + t2)
    return res


def func5(x):
    a = x[0]
    b = x[1]
    res = np.sin(a+b) + (a-b)**2 - 1.5*a + 2.5*b + 1
    return res

""" keep the x matrix in the bound of x_max and x_min
    returns those 'particles' and 'dimensions' which were bounded 
    third element of the returned is either 1 or -1 
    +1: bounded by max
    -1: bounded by min """
def bound(x, x_min, x_max):
    res = []
    ndim = np.size(x_min)
    Np = np.shape(x)[0]    
    for i in range(Np):
        for j in range(ndim):
            if x[i, j] > x_max[j]:
                x[i, j] = x_max[j]
                res.append([i, j, 1])
            elif x[i, j] < x_min[j]:
                x[i, j] = x_min[j]
                res.append([i, j, -1])
    return np.array(res)


def velocity_modifier(v, data):
    n = np.shape(data)[0]
    for i in range(n):
        a = data[i][0]
        b = data[i][1]
        dir = data[i][2]
        if dir > 0 and v[a][b] > 0:
            v[a][b] = 0
        elif dir < 0 and v[a][b] < 0:
            v[a][b] = 0


def feval(data, func):
    Np = np.shape(data)[0]
#    ndim = np.shape(data)[1]
    res = np.zeros([Np, 1])
    for i in range(Np):
        x = data[i]
        res[i][0] = func(x)
    return res

                

def is_converged(data, index, interval = 100, tolerance = 1e-6):
    target = np.array(data[index - interval : index])
    grad = np.abs((target.max() - target.min()) / target[-1])
    return grad < tolerance


#def is_converged(data, index, interval=200, tolerance = 1e-6):
#    grad = np.abs((data[index] - data[index - interval]) / data[index])
#    return (grad < tolerance)
    
    
    
