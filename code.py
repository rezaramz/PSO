import numpy as np
import functions as fnc


ndim = 2 # Number of dimension
Np = 20; # Number of particles
max_iter = 100 # Maximum number of iterations
c1 = 1.0 # Learning parameter
c2 = 1.0
alpha_max = 1.0
alpha_min = 0.1

# create a lower bound and upper bound for variables and velocities
lb = -1000 * np.ones(ndim)
ub = 1000 * np.ones(ndim)
v_min = -(ub - lb)
v_max = ub - lb

x = np.zeros([Np, ndim])
v = np.zeros([Np, ndim])
# Assign random values for particles position and their velocities
for i in range(ndim):
    for j in range(Np):
        x[j, i] = np.random.uniform(lb[i], ub[i])
        v[j, i] = np.random.uniform(v_min[i], v_max[i])


# Best known position of each particle
fitness = fnc.func2(x)
x_star = x

# x_g is the swarm's best known position
x_g = x_star[np.argmin(fitness), :].copy()
f_g = np.min(fitness)

for i in range(max_iter):
    
    r1 = np.random.rand()
    r2 = np.random.rand()
    # weight inertia
    alpha = alpha_max - (alpha_max - alpha_min) * (i / (max_iter - 1)) 
    
    v = alpha * v + c1 * r1 * (x_star - x) + c2 * r2 * (x_g - x)
    fnc.bound(v, v_min, v_max)
    x += v
    fnc.bound(x, lb, ub)
    
    fitness_new = fnc.func2(x)
    # Updating best known position of each particle
    for j in range(Np):
        if fitness_new[j] < fitness[j]:
            x_star[j, :] = x[j, :]
            fitness[j] = fitness_new[j]

    if np.min(fitness_new) < f_g:
        x_g = x[np.argmin(fitness_new), :].copy()
        f_g = np.min(fitness_new)
        
print(x_g, f_g)
