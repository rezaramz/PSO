import numpy as np
import functions as fnc
import matplotlib.pyplot as plt

ndim = 2 # Number of dimension
Np = min(int(10 * ndim), 100) # Number of particles
max_iter = 100 # Maximum number of iterations
interval = 100
tolerance = 1e-6
fnc_count = 0
c1 = 2.0 # Learning parameter  0 <= c1, c2 <= 2
c2 = 2.0
alpha_max = 1.0  # if wanted to use constant alpha = [0.8, 1.2]
alpha_min = 0.1
main_function = lambda x : fnc.rosenbrock(x)

# create a lower bound and upper bound for variables and velocities
lb = -10 * np.ones(ndim)
#lb = np.array([-1.5, -3])
ub = 10 * np.ones(ndim)
#ub = np.array([4, 4])
velocity_factor = 0.2
v_min = -(ub - lb) * velocity_factor
v_max = (ub - lb) * velocity_factor


x = np.zeros([Np, ndim])
v = np.zeros([Np, ndim])
# Assign random values for particles position and their velocities
for i in range(ndim):
    for j in range(Np):
        x[j, i] = np.random.uniform(lb[i], ub[i])
        v[j, i] = np.random.uniform(v_min[i], v_max[i])
#        v[j, i] = 0.5 * (np.random.uniform(lb[i], ub[i]) - x[j, i])


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
zero_velocity_points = np.array([])

for i in range(max_iter):
    
    r1 = np.random.uniform()
    r2 = np.random.uniform()
    # weight inertia
    alpha = alpha_max - (alpha_max - alpha_min) * (i / (max_iter - 1)) 
    
    v = alpha * v + c1 * r1 * (x_star - x) + c2 * r2 * (x_g - x)
    fnc.bound(v, v_min, v_max)
    fnc.velocity_modifier(v, zero_velocity_points)
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
    
""" Stopping criterion """   
#    if i % interval == 0 and i != 0:
#        if fnc.is_converged(best_fitness, i, interval, tolerance):
#            num_iter = i
#            print(num_iter)
#            break

""" Plotting """
iterations = np.arange(max_iter)        
#plt.plot(iterations, best_fitness, 'r')
#plt.show()
        
print(x_g, f_g, fnc_count)
