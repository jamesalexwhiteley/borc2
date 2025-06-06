import torch
import nevergrad as ng
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=r".*Initial solution argument*")  
warnings.filterwarnings("ignore", message=r".*sigma change np.exp*") 
warnings.filterwarnings("ignore", message=r".*orphanated injected solution*") 
warnings.filterwarnings("ignore", message=r".*Bounds are 1.0 sigma away from each other*") 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ============================================= 
# pytorch 
# =============================================  
def LBFGS(f, x, iters, bounds, lr=0.1):     
    """
    Unconstrained optimisation using torch.optim.LBFGS with bounds 

    """
    return torch_optim(f, x, iters, bounds, optimiser='LBFGS', lr=lr)

def ADAM(f, x, iters, bounds, lr=0.1):
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

    def closure():
        optimizer.zero_grad()  
        loss = f(x)            
        loss = -loss.sum()     
        loss.backward()    
        return loss

    for _ in range(int(iters)):
        optimizer.step(closure)

    with torch.no_grad():
        x = torch.min(torch.max(x, b0), b1)

    torch.cuda.empty_cache()
    return x, f(x)

# ============================================= 
# nevergrad  
# ============================================= 
def CMA_ES(f, g, x, iters, bounds):
    """
    Constrained optimisation using CMA-ES from Nevergrad
    """
    device = x.device
    b0, b1 = bounds[:, 0].cpu().flatten().numpy(), bounds[:, 1].cpu().flatten().numpy()
    
    # ensure initial point is within bounds
    x_numpy = x.cpu().flatten().numpy()
    x_numpy = np.clip(x_numpy, b0, b1)
    
    # objective
    def f2(x):
        x_tensor = torch.tensor(x, device=device).unsqueeze(0)
        return -f(x_tensor).detach().flatten().item()
    
    # constraint
    def g2(x):
        x_tensor = torch.tensor(x, device=device).unsqueeze(0)
        return (g(x_tensor).detach() >= 0).all().item()
    
    # optimize 
    parametrization = ng.p.Array(init=x_numpy).set_bounds(lower=b0, upper=b1)
    parametrization.register_cheap_constraint(g2)  # rejection sampling
    
    optimizer = ng.optimizers.CMA(parametrization, budget=iters)
    res = optimizer.minimize(f2)
    
    x = torch.tensor(res.value).unsqueeze(0).to(device)
    return x, f(x)