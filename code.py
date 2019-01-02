import numpy as np
import functions as fnc
import matplotlib.pyplot as plt

ndim = 2 # Number of dimension
Np = min(int(10 * ndim), 100) # Number of particles
max_iter = 1000 # Maximum number of iterations
interval = 100
tolerance = 1e-6
fnc_count = 0
c1 = 2.0 # Learning parameter
c2 = 2.0
alpha_max = 1.0
alpha_min = 0.1
main_function = lambda x : fnc.func2(x)

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
fitness = fnc.feval(x, main_function)
x_star = x
fnc_count += Np

# x_g is the swarm's best known position
x_g = x_star[np.argmin(fitness), :].copy()
f_g = np.min(fitness)


#    iterations = np.arange(max_iter)
best_fitness = np.zeros(max_iter)
mean_fitness = np.zeros(max_iter)

for i in range(max_iter):
    
    r1 = np.random.rand()
    r2 = np.random.rand()
    # weight inertia
    alpha = alpha_max - (alpha_max - alpha_min) * (i / (max_iter - 1)) 
    
    v = alpha * v + c1 * r1 * (x_star - x) + c2 * r2 * (x_g - x)
    fnc.bound(v, v_min, v_max)
    x += v
    zero_velocity_points = fnc.bound(x, lb, ub)
    
    fitness_new = fnc.feval(x, main_function)
    fnc_count += Np
    # Updating best known position of each particle
    for j in range(Np):
        if fitness_new[j] < fitness[j]:
            x_star[j, :] = x[j, :] # don't need to use .copy()
            fitness[j] = fitness_new[j]

    if np.min(fitness_new) < f_g:
        x_g = x[np.argmin(fitness_new), :].copy()
        f_g = np.min(fitness_new)
    best_fitness[i] = f_g
    mean_fitness[i] = np.mean(fitness_new)
    
#    if i % interval == 0 and i != 0:
#        if fnc.is_converged(best_fitness, i, interval, tolerance):
#            num_iter = i
#            print(num_iter)
#            break

iterations = np.arange(max_iter)        
plt.plot(iterations, best_fitness, 'r')
plt.show()
        
print(x_g, f_g, fnc_count)
