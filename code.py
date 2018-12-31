import numpy as np
import functions as fnc

ndim = 1 # Number of dimension
Np = 1; # Number of particles
max_iter = 1000 # Maximum number of iterations
c1 = 1.0 # Learning parameter
c2 = 1.0
alpha_max = 1.0
alpha_min = 0.1

#creates Np number of random numbers between -10 and 10
lb = -1000;
ub = 1000;
v_min = -(ub - lb)
v_max = ub - lb
x = np.random.uniform(lb, ub, Np)
v = np.random.uniform(v_min, v_max, Np)

# Best known position of each particle
fitness = fnc.func(x)
x_star = x

# x_g is the swarm's best known position
x_g = x_star[np.argmin(fitness)]
f_g = np.min(fitness)


for i in range(max_iter):
    
    r1 = np.random.rand()
    r2 = np.random.rand()
    # weight inertia
    alpha = alpha_max - (alpha_max - alpha_min) * (i / (max_iter - 1)) 
    
    v = alpha * v + c1 * r1 * (x_star - x) + c2 * r2 * (x_g - x)
    fnc.bound_velocity(v, v_min, v_max)
    x += v
    
    fitness_new = fnc.func(x)
    # Updating best known position of each particle
    for j in range(len(fitness)):
        if fitness_new[j] < fitness[j]:
            x_star[j] = x[j]
            fitness[j] = fitness_new[j]

    if np.min(fitness_new) < f_g:
        x_g = x[np.argmin(fitness_new)]
        f_g = np.min(fitness_new)
        
print(x_g, f_g)
