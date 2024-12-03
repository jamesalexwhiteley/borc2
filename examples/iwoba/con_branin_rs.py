import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import math 

from borc2.problem import Problem 
from borc2.surrogate import Surrogate
from borc2.acquisition import Acquisition
from borc2.bayesopt import Borc
from borc2.probability import DiscreteJoint
from borc2.utilities import tic, toc 

plt.rcParams['font.size'] = 12
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plotcontour(problem):

    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'branin.png')
    fig = plt.figure(figsize=(7, 6))

    # ground truth 
    steps = 1000
    x = torch.linspace(0, 1, steps)
    y = torch.linspace(0, 1, steps)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    mu, prob = problem.rbo(xpts, nsamples=int(5e2), output=False, return_vals=True)
    MU = mu.view(X.shape).detach()
    PI = prob[0].view(X.shape).detach()

    xopt = torch.tensor([0, 0.75])
    proxy = Line2D([0], [0], color='black', lw=1.5, label=r'$P[g(x,\xi)<0] = 1-\epsilon$')
    contour = plt.contourf(X.numpy(), Y.numpy(), MU.numpy(), cmap='PuBu')
    plt.contour(X.numpy(), Y.numpy(), PI.numpy(), levels=[0.95], colors='black')
    plt.colorbar(contour, shrink=0.8, pad=0.05)
    scatter = plt.scatter(xopt[0]+0.01, xopt[1]+0.01, label='Optimal x', color='m', s=50, marker='D', zorder=3)

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    # plt.legend(loc=0, handles=[proxy])
    plt.legend([scatter, proxy], ['Optimal x', r'$P[g(x,\xi)<0]$ = 0.9'], loc='best')
    # plt.savefig(output_path, dpi=600)
    plt.show()

def branin_williams(x):
    """
    Williams, Brian Jonathan. Sequential design of computer experiments to minimize integrated response functions. The Ohio State University, 2000.
    
    Parameters
    ----------
    x : torch.Tensor, shape=(4, n) 

    """
    def yb(u, v):
        return (v - 5.1 / (4 * math.pi ** 2) * u ** 2 + 5 / math.pi * u - 6) ** 2 + 10 * (1 - 1 / (8 * math.pi)) * torch.cos(u) + 10

    u1 = 15 * x[0, :] - 5
    v1 = 15 * x[1, :]
    u2 = 15 * x[2, :] - 5
    v2 = 15 * x[3, :]
    
    return yb(u1, v1) * yb(u2, v2)

class Model():
    def __call__(self, x): 
        self.x = x
        self.m = None

    def f(self):    
        return branin_williams(self.x[:, [0, 2, 3, 1]].T)
    
    def g(self):
        return torch.linalg.vector_norm(self.x, dim=1) - torch.tensor(3/2).sqrt()
    
def bayesopt(ninitial, iters, n):

    problem = Problem()
    model = Model()
    bounds = {"x1": (0, 1), "x4": (0, 1)}

    joint = torch.tensor([
        [0.0375, 0.0875, 0.0875, 0.0375],  # P(x2=0.25, x3=0.2), P(x2=0.25, x3=0.4), P(x2=0.25, x3=0.6), P(x2=0.25, x3=0.8) 
        [0.0750, 0.1750, 0.1750, 0.0750],  # P(x2=0.5, x3=0.2) ...
        [0.0375, 0.0875, 0.0875, 0.0375],  # P(x2=0.75, x3=0.2) ...
        ])
    
    x2_values = torch.tensor([0.25, 0.5, 0.75])
    x3_values = torch.tensor([0.2, 0.4, 0.6, 0.8])  
    dist = DiscreteJoint(joint, x2_values, x3_values)

    problem.set_bounds(bounds)
    problem.set_dist(dist)
    problem.add_model(model)
    problem.add_objectives([model.f])
    problem.add_constraints([model.g])
 
    xi = problem.sample_xi(nsamples=int(20)).to(device)
    surrogate = Surrogate()
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1)
    borc = Borc(problem, surrogate, acquisition) 
    borc.cuda(device) 
    borc.initialize(nsamples=ninitial, sample_method="lhs", xbest=problem.sample_x(), fbest=torch.tensor([0.0])) 

    params=(torch.linspace(0.0, 1.0, steps=21), torch.linspace(0.0, 1.0, steps=21)) 
    # xopt, _ = problem.monte_carlo(params=params, nsamples=int(5e2), obj_type="mean", con_type="prob", con_eps=0.1) # [0, 0.7] 
    # _, _ = problem.rbo(xopt, nsamples=int(1e3), return_vals=True) 
    # plotcontour(problem)

    # x = problem.sample_x(nsamples=1)
    # print(x)
    # print(borc.eval_acqf(x))
    # print(borc.eval_acqg(x))
    xopt, acq = borc.surrogate.monte_carlo(params=params, nsamples=int(5e1), obj_type="mean", con_type="prob", con_eps=0.1)
    print(f"{xopt} | {acq}")
    xopt, acq = borc.constrained_optimize_acq(iters=500, nstarts=1, optimize_x=True) # TODO multiple starts? 
    print(f"{xopt} | {acq}") 

    # # BayesOpt used to sequentially sample [x,xi] points 
    # res = torch.ones(iters, ) 
    # for i in range(iters): 

    #     # new_x <- random search 
    #     borc.step(new_x=problem.sample()) 

    #     # argmax_x E[f(x,xi)] s.t. P[g(x,xi)<0]>1-epsilons
    #     xopt, _ = borc.constrained_optimize_acq(iters=200, nstarts=5, optimize_x=True) 
    #     # if i % n == 0: 
    #     #     params=(torch.linspace(0, 1, steps=101), torch.linspace(0, 1, steps=101)) 
    #     #     xopt, _ = borc.surrogate.monte_carlo(params=params, nsamples=int(5e1), obj_type="mean", con_type="prob", con_eps=0.1) 
    #     #     res[i], _ = problem.rbo(xopt, output=False, return_vals=True) # true E[f(x,xi)] 
    #     #     # print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

    #     #     mu_pred, _ = borc.rbo(test_points, nsamples=int(1e3), output=False, return_vals=True)
    #     #     mae[i] = torch.mean(torch.abs(mu_true[0] - mu_pred[0])) # mean absolute error 

    # return xopt, res 


if __name__ == "__main__": 
    ninitial, iters, n = 500, 2, 2
    bayesopt(ninitial, iters, n) 
    # xopt, res, mae = bayesopt(ninitial, iters, n) 