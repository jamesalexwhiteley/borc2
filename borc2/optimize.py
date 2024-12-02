import torch
import numpy as np

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ============================================= 
# torch.optim
# =============================================  
def LBFGS(f, x, iters, bounds, batch_mode=False, tol=1e-6, lr=0.01):     
    """
    Unconstrained optimisation; bounded torch.optim.LBFGS 

    """
    return torch_optim(f, x, iters, bounds, batch_mode=batch_mode, tol=tol, optimiser='LBFGS', lr=lr)

def ADAM(f, x, iters, bounds, batch_mode=False, tol=1e-6, lr=0.02):
    """
    Unconstrained optimisation; bounded torch.optim.Adam 

    """
    return torch_optim(f, x, iters, bounds, batch_mode=batch_mode, tol=tol, optimiser='ADAM', lr=lr)

def torch_optim(f, x, iters, bounds, batch_mode=False, tol=1e-6, optimiser='lbfgs', lr=0.1):

    # # we need to adjust bounds if optimizing in batch mode 
    # with torch.no_grad():
    #     b0, b1 = bounds[:, 0], bounds[:, 1]
    #     if batch_mode: 
    #         b0 = b0.unsqueeze(0).expand_as(x)
    #         b1 = b1.unsqueeze(0).expand_as(x)

    # x.requires_grad_(True) 
    # if optimiser == 'lbfgs':
    #     optimizer = torch.optim.LBFGS([x], lr=lr) 
    # elif optimiser == 'adam':
    #     optimizer = torch.optim.Adam([x], lr=lr)

    # def closure(): 
    #     optimizer.zero_grad() 
    #     loss = -f(x).sum() 
    #     loss.backward(retain_graph=True) 
    #     # loss.backward()
    #     return loss 

    # for i in range(iters):
    #     optimizer.step(closure)

    # with torch.no_grad():
    #     if batch_mode: 
    #         x = torch.min(torch.max(x, b0), b1)
    #     else:   
    #         for j in range(x.shape[1]): 
    #             x[:, j] = x[:, j].clamp(bounds[j][0], bounds[j][1])

    # torch.cuda.empty_cache()
    # return x, f(x)
    
    # we need to adjust bounds if optimizing in batch mode 
    with torch.no_grad():
        b0, b1 = bounds[:, 0], bounds[:, 1]
        if batch_mode: 
            b0 = b0.unsqueeze(0).expand_as(x)
            b1 = b1.unsqueeze(0).expand_as(x)

    x.requires_grad_(True) 
    if optimiser == 'LBFGS':
        optimizer = torch.optim.LBFGS([x], lr=lr) 
    elif optimiser == 'ADAM':
        optimizer = torch.optim.Adam([x], lr=lr)

    # if t != 0.0: # use barrier method 
    #     def closure(): 
    #         optimizer.zero_grad() 
    #         loss = -f(x, t).sum()
    #         if loss is None:
    #             print(f"Stopping optimization at iteration {i} due to penalty.")
    #             break
    #         loss.backward() 
    #         return loss 
        
    # else: 
    #     def closure(): 
    #         optimizer.zero_grad() 
    #         loss = -f(x).sum() 
    #         loss.backward(retain_graph=True) 
    #         # loss.backward()
    #         return loss 

    # intialise prev_x, prev_f
    prev_x = x.clone().detach() 
    prev_f = -f(x).detach() 

    # output = False
    for _ in range(int(iters)):

        optimizer.zero_grad() 
        loss = f(x)
        if loss is None:
        # if torch.abs(loss) > 1e10:
            # print(f"Stopping optimization at iteration {i} due to penalty. Returing prev_x, prev_f")
            return prev_x, -prev_f
        loss = -loss.sum() # batch optimization, and -loss for max
        # loss = -loss 
        loss.backward() 
            
        prev_x, prev_f = x.clone().detach(), loss.detach() # previous iteration 
        optimizer.step()
    
        # else: 
        #     optimizer.zero_grad() 
        #     loss = -f(x).sum() 
        #     loss.backward(retain_graph=True) 
        #     # loss.backward()
        #     optimizer.step()

        # if output: 
        #     print(f'Iter {i+1}/{iters} - Loss: {closure().item():.3f}')

    if batch_mode: 
        with torch.no_grad():
            x = torch.min(torch.max(x, b0), b1)
    else:   
        with torch.no_grad():
            for j in range(x.shape[1]): 
                x[:, j] = x[:, j].clamp(bounds[j][0], bounds[j][1])

    # del output, loss
    torch.cuda.empty_cache()

    # if t != 0.0: # use barrier method
    #     return x, f(x, t)
    # else:
    return x, f(x)