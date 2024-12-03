import torch
import nevergrad as ng

import warnings
warnings.filterwarnings("ignore", message=r".*Initial solution argument*")  
warnings.filterwarnings("ignore", message=r".*sigma change np.exp*") 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ============================================= 
# pytorch 
# =============================================  
def LBFGS(f, x, iters, bounds, lr=0.01):     
    """
    Unconstrained optimisation using torch.optim.LBFGS with bounds 

    """
    return torch_optim(f, x, iters, bounds, optimiser='LBFGS', lr=lr)

def ADAM(f, x, iters, bounds, lr=0.02):
    """
    Unconstrained optimisation using torch.optim.ADAM with bounds 

    """
    return torch_optim(f, x, iters, bounds, optimiser='ADAM', lr=lr)

def torch_optim(f, x, iters, bounds, optimiser='ADAM', lr=0.1):

    # adjust bounds for batch mode 
    with torch.no_grad():
        b0, b1 = bounds[:, 0], bounds[:, 1]
        b0 = b0.unsqueeze(0).expand_as(x)
        b1 = b1.unsqueeze(0).expand_as(x)

    x.requires_grad_(True) 
    if optimiser == 'LBFGS':
        optimizer = torch.optim.LBFGS([x], lr=lr) 
    elif optimiser == 'ADAM':
        optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(int(iters)):
        optimizer.zero_grad() 
        loss = f(x)
        loss = -loss.sum() 
        loss.backward() 
        optimizer.step()

    with torch.no_grad():
        x = torch.min(torch.max(x, b0), b1)

    torch.cuda.empty_cache()
    return x, f(x)

# ============================================= 
# pycma 
# ============================================= 
# def CMA_ES(f, g, x, bounds, sigma=0.3):
#     """
#     Constrained optimisation using CMA-ES from pycma 

#     """
#     def callback(es):
#         print(f"Iteration {es.countiter}: f(x) = {-es.best.f}")  # `-` to undo negation

#     fun = lambda x: -x[0]**2
#     # fun = lambda z : -f(torch.tensor(z)).detach().flatten().item()
#     x0 = [1]
#     sigma0 = 0.01
#     bounds = [[-3], [ 3]]
#     x, es = cma.fmin2(fun, x0, sigma0, {'bounds': bounds})
#     print("Optimized solution:", x)
#     print()

#     # x0 = x.numpy() 
#     # b0, b1 = bounds[:, 0].flatten().tolist(), bounds[:, 1].flatten().tolist()
    
#     # f_numpy = lambda z : -f(torch.tensor(z[0], requires_grad=False)).detach().item()  

#     # # x, es = cma.fmin2(f_numpy, x0, sigma)
#     # x, es = cma.fmin2(f_numpy, x0, sigma, {'bounds': [b0, b1]})

#     # cfun = cma.ConstrainedFitnessAL(f, g) # augmented lagrangian # TODO first try unconstrained opt? 

#     # x, es = cma.fmin2(
#     #     cfun,
#     #     x,              # initial solution
#     #     sigma,          # initial standard deviation to sample new solutions
#     #     {
#     #         'tolstagnation': 0,
#     #         'maxfevals': 1e4, 
#     #         'CMA_active': True, 
#     #         # 'bounds': bounds  # TODO 
#     #     },
#     #     callback=cfun.update
#     # )

#     # x = cfun.find_feasible(es) # strict feasibility 

#     x = torch.tensor(x)

#     return x, f(x)

# ============================================= 
# nevergrad  
# ============================================= 
def CMA_ES(f, g, x, iters, bounds, sigma=0.3):
    """
    Constrained optimisation using CMA-ES from Nevergrad  

    """
    b0, b1 = bounds[:, 0].flatten().numpy(), bounds[:, 1].flatten().numpy()

    def f2(x):
        return -f(torch.tensor(x)).detach().flatten().item()

    def g2(x):
        return all(g(torch.tensor(x)).detach() >= 0)

    # objective 
    parametrization = ng.p.Array(shape=x.shape).set_bounds(lower=b0, upper=b1) 

    # constraint 
    parametrization.register_cheap_constraint(g2)

    # optimize 
    optimizer = ng.optimizers.CMA(parametrization, budget=iters) 
    res = optimizer.minimize(f2)  
    
    x = torch.tensor(res.value)

    return x, f(x)


