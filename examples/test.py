import numpy as np
import cma

def f(x):
    return x[0]**2 

def g1(x):
    return 6 - x[0] 

def g2(x):
    return 7 - x[0]

def g(x):
    return [g1(x), g2(x)] 

cfun = cma.ConstrainedFitnessAL(f, g) # augmented lagrangian

# Initialize
x0 = [7.0]  
sigma0 = 0.3  

bounds = [1, 10]

x, es = cma.fmin2(
    cfun,
    x0,
    sigma0,
    {
        'tolstagnation': 0,
        'bounds': bounds  
    },
    callback=cfun.update
)

x = cfun.find_feasible(es) # strict feasibility 

print()
print("Optimized solution:", x)
print("Objective value:", f(x))
print("Constraint violation:", g(x)) 

