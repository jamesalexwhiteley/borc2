import torch
import nevergrad as ng
import numpy as np
import threading

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
        # loss.backward()   
        loss.backward(retain_graph=True) 
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
# def CMA_ES(f, g, x, iters, bounds, timeout_seconds=300):
#     """
#     Simple timeout wrapper - kills hanging optimizer after timeout_seconds
#     """
#     result = [None]  # Use list so thread can modify it
    
#     def run_optimizer():
#         try:
#             device = x.device
#             b0, b1 = bounds[:, 0].cpu().flatten().numpy(), bounds[:, 1].cpu().flatten().numpy()
#             x_numpy = x.cpu().flatten().numpy()
#             x_numpy = np.clip(x_numpy, b0, b1)
            
#             def f2(x):
#                 x_tensor = torch.tensor(x, device=device).unsqueeze(0)
#                 return -f(x_tensor).detach().flatten().item()
            
#             def g2(x):
#                 x_tensor = torch.tensor(x, device=device).unsqueeze(0)
#                 return (g(x_tensor).detach() >= 0).all().item()
            
#             parametrization = ng.p.Array(init=x_numpy).set_bounds(lower=b0, upper=b1)
#             parametrization.register_cheap_constraint(g2)
#             optimizer = ng.optimizers.CMA(parametrization, budget=iters)
#             res = optimizer.minimize(f2)
            
#             x_result = torch.tensor(res.value).unsqueeze(0).to(device)
#             result[0] = (x_result, f(x_result))
#         except Exception as e:
#             print(f"Optimizer error: {e}")
#             result[0] = (x, f(x))  # Return original point if error
    
#     # Start optimizer in thread
#     thread = threading.Thread(target=run_optimizer)
#     thread.daemon = True  # Dies when main thread dies
#     thread.start()
    
#     # Wait for timeout
#     thread.join(timeout_seconds)
    
#     if thread.is_alive():
#         print(f"Optimizer timed out after {timeout_seconds}s")
#         return x, f(x)  # Return original point
    
#     return result[0] if result[0] else (x, f(x))

# ============================================= 
# custom CMA_ES
# ============================================= 
def CMA_ES(f, g, x0, iters, bounds, sigma=0.5):
    """
    Simple CMA-ES implementation in PyTorch 
    """
    # Handle input types
    if isinstance(x0, np.ndarray):
        x = torch.tensor(x0, dtype=torch.float32)
        device = bounds.device if hasattr(bounds, 'device') else torch.device('cpu')
    else:
        x = x0.clone()
        device = x0.device
    
    x = x.to(device)
    
    if isinstance(bounds, np.ndarray):
        bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
    else:
        bounds = bounds.to(device)
    
    dim = x.shape[0]
    
    # CMA-ES parameters
    lambda_ = 4 + int(3 * np.log(dim))  # Population size
    mu = lambda_ // 2  # Number of parents
    weights = torch.log(torch.tensor(mu + 0.5, device=device)) - torch.log(torch.arange(1, mu + 1, device=device, dtype=torch.float32))
    weights = weights / weights.sum()
    mueff = 1 / (weights ** 2).sum()
    
    # Adaptation parameters
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, torch.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    
    # Initialize
    mean = x.clone()
    pc = torch.zeros(dim, device=device)
    ps = torch.zeros(dim, device=device)
    C = torch.eye(dim, device=device)
    invsqrtC = torch.eye(dim, device=device)
    eigeneval = 0
    
    best_x = mean.clone()
    best_f = torch.tensor(-float('inf'), device=device)
    
    for iteration in range(iters // lambda_):  # Each iteration evaluates lambda_ points
        # Generate offspring
        offspring = torch.zeros(lambda_, dim, device=device)
        for k in range(lambda_):
            z = torch.randn(dim, device=device)
            offspring[k] = mean + sigma * torch.matmul(C, z)
            # Clip to bounds
            offspring[k] = torch.clamp(offspring[k], bounds[:, 0], bounds[:, 1])
        
        # Evaluate fitness and constraints
        fitness = torch.zeros(lambda_, device=device)
        feasible = torch.zeros(lambda_, dtype=torch.bool, device=device)
        
        for k in range(lambda_):
            f_val = f(offspring[k].unsqueeze(0)).squeeze()
            g_val = g(offspring[k].unsqueeze(0))
            
            feasible[k] = torch.all(g_val >= 0)
            if feasible[k]:
                fitness[k] = f_val
            else:
                # Penalty for infeasible solutions
                constraint_violation = torch.clamp(-g_val, min=0).sum()
                fitness[k] = f_val - 1000 * constraint_violation
        
        # Sort by fitness
        sorted_indices = torch.argsort(fitness, descending=True)
        
        # Update best solution
        if feasible[sorted_indices[0]] and fitness[sorted_indices[0]] > best_f:
            best_f = fitness[sorted_indices[0]]
            best_x = offspring[sorted_indices[0]].clone()
        
        # Update mean
        old_mean = mean.clone()
        mean = torch.sum(weights.unsqueeze(1) * offspring[sorted_indices[:mu]], dim=0)
        
        # Update evolution paths
        ps = (1 - cs) * ps + torch.sqrt(cs * (2 - cs) * mueff) * torch.matmul(invsqrtC, (mean - old_mean) / sigma)
        hsig = torch.norm(ps) / torch.sqrt(1 - (1 - cs) ** (2 * (iteration + 1))) < 1.4 + 2 / (dim + 1)
        pc = (1 - cc) * pc + hsig * torch.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
        
        # Update covariance matrix
        artmp = (offspring[sorted_indices[:mu]] - old_mean) / sigma
        C = (1 - c1 - cmu) * C + c1 * torch.outer(pc, pc) + cmu * torch.matmul(artmp.T * weights, artmp)
        
        # Update step size
        ps_norm = torch.norm(ps)
        expected_norm = torch.sqrt(torch.tensor(dim, dtype=torch.float32)) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        sigma = sigma * torch.exp((cs / damps) * (ps_norm / expected_norm - 1))
        
        # Decompose C for sampling (every few iterations)
        if iteration % (iters // (10 * lambda_) + 1) == 0:
            eigeneval += 1
            try:
                C = (C + C.T) / 2  # Ensure symmetry
                D, B = torch.linalg.eigh(C)
                D = torch.sqrt(torch.clamp(D, min=1e-10))
                invsqrtC = torch.matmul(B, torch.matmul(torch.diag(1 / D), B.T))
            except:
                # If decomposition fails, reset
                C = torch.eye(dim, device=device)
                invsqrtC = torch.eye(dim, device=device)
    
    return best_x.unsqueeze(0), f(best_x.unsqueeze(0))