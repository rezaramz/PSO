import numpy as np
import functions as fnc
import matplotlib.pyplot as plt


repeat = 10
ndim = 3 # Number of dimension
Np = min(int(10 * ndim), 100) # Number of particles
max_iter = 2000 # Maximum number of iterations
interval = 100
fcn_count = 0
c1 = 1.0 # Learning parameter
c2 = 1.0
alpha_max = 1.0
alpha_min = 0.1
feval = lambda x : fnc.func3(x)

# create a lower bound and upper bound for variables and velocities
lb = -10 * np.ones(ndim)
ub = 10 * np.ones(ndim)
v_min = -(ub - lb)
v_max = ub - lb

fval = np.inf
x_final = np.zeros([1, ndim])

for _ in range(repeat):

    x = np.zeros([Np, ndim])
    v = np.zeros([Np, ndim])
    # Assign random values for particles position and their velocities
    for i in range(ndim):
        for j in range(Np):
            x[j, i] = np.random.uniform(lb[i], ub[i])
            v[j, i] = np.random.uniform(v_min[i], v_max[i])
    
    
    # Best known position of each particle
    fitness = feval(x)
    x_star = x
    fcn_count += Np
    
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
        fnc.bound(x, lb, ub)
        
        fitness_new = feval(x)
        fcn_count += Np
        # Updating best known position of each particle
        for j in range(Np):
            if fitness_new[j] < fitness[j]:
                x_star[j, :] = x[j, :]
                fitness[j] = fitness_new[j]
    
        if np.min(fitness_new) < f_g:
            x_g = x[np.argmin(fitness_new), :].copy()
            f_g = np.min(fitness_new)
        best_fitness[i] = f_g
        mean_fitness[i] = np.mean(fitness_new)
        
        if i % interval == 0 and i != 0:
            if fnc.is_converged(best_fitness, i, interval):
                num_iter = i
                print(num_iter)
                break
            
    if f_g < fval:
        fval = f_g
        x_final = x_g.copy()
        best_final = best_fitness[0:num_iter+1].copy()
        mean_final = mean_fitness[0:num_iter+1].copy()
        iterations = np.arange(num_iter + 1)
        choice = _

#best_fitness = best_fitness[0:num_iter + 1]
#mean_fitness = mean_fitness[0:num_iter + 1]

        
plt.plot(iterations, best_final, 'r')
#plt.plot(iterations, mean_final, 'b*')
#plt.cla()
#plt.axis([900, 1000, 0, 1])
plt.show()
        
print(x_final, fval, fcn_count)
