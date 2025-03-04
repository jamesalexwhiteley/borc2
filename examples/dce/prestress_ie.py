import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import math 

from borc2.problem import Problem 
from borc2.surrogate import Surrogate, SurrogateIO
from borc2.acquisition import Acquisition
from borc2.bayesopt import Borc
from borc2.probability import MultivariateNormal
from borc2.utilities import tic, toc 

from prestress_rs import Model, plotcontour # type:ignore 
from pystressed.servicability import plot_magnel, optimize_magnel, optimize_and_plot_magnel 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: James Whiteley (github.com/jamesalexwhiteley)

def bayesopt(ninitial, iters, n): 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_dir  = os.path.join(base_folder, "prestress")

    problem = Problem()
    model = Model()
    bounds = {"b": (0.1, 1.0), "h": (0.1, 1.0)} 
    
    mu = torch.tensor([27.5, 1.5, 1.5, 1.5]) 
    cov = torch.tensor([[(3.75)**0.5,      0.0,      0.0,       0.0],
                        [        0.0,         1,    0.9*1,    0.7*1],
                        [        0.0,     0.9*1,        1,    0.7*1],
                        [        0.0,     0.7*1,    0.7*1,        1]])
    dist = MultivariateNormal(mu, cov)

    problem.set_bounds(bounds)
    problem.set_dist(dist)
    problem.add_model(model)
    problem.add_objectives([model.f])
    problem.add_constraints([model.g])

    xi = problem.sample_xi(nsamples=int(2e2)).to(device)
    surrogate = Surrogate(problem, ntraining=200, nstarts=5) 
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1) 
    borc = Borc(surrogate, acquisition) 
    borc.cuda(device) 
    borc.initialize(nsamples=ninitial, sample_method="sobol", max_acq=torch.tensor([0])) 
    acq_iters=50, acq_starts=10

    # BayesOpt used to sequentially sample [x,xi] points 
    res = torch.ones(iters, ) 
    for i in range(iters):    

        # new_x = argmax_[x,xi] EI x PF
        borc.acquisition = Acquisition(f="EI", g="PF", xi=xi)
        new_x, _ = borc.batch_optimize_acq(iters=acq_iters, nstarts=acq_starts) 

        # fbest = max_[x,xi] mu 
        borc.acquisition = Acquisition(f="MU") 
        _, borc.fbest = borc.batch_optimize_acq(iters=acq_iters, nstarts=acq_starts) 
        borc.step(new_x=new_x) 
        # print(f"new_x : {new_x}") 

        # argmax_x E[f(x,xi)] s.t. P[g(x,xi)<0]>1-epsilon 
        if i % n == 0: 
            borc.acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1)
            xopt, _ = borc.constrained_optimize_acq(iters=int(1e2), nstarts=4, optimize_x=True) 
            res[i], _ = problem.rbo(xopt, output=False, return_vals=True) # true E[f(x,xi)] 
            # print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

    return xopt, res   


if __name__ == "__main__": 

    ninitial, iters, n = 100, 400, 1 
    xopt, res = bayesopt(ninitial, iters, n) 