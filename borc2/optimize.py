import torch
import cma

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

def CMA_ES(f, g, x, bounds, sigma=0.3):
    """
    Constrained optimisation using CMA-ES from pycma 

    """

    cfun = cma.ConstrainedFitnessAL(f, g) # augmented lagrangian

    x, es = cma.fmin2(
        cfun,
        x, # initial x 
        sigma, 
        {
            'tolstagnation': 0,
            'bounds': bounds  
        },
        callback=cfun.update
    )

    x = cfun.find_feasible(es) # strict feasibility 

    return x, f(x)

