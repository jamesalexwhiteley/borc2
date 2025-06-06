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
    output_dir  = os.path.join(base_folder, "prestress_ie")

    problem = Problem()
    model = Model()
    # bounds = {"P": (25, 40), "e": (0.1, 0.4), 'd': (1.4, 2.0)} # for plotting 
    bounds = {"P": (25, 40), "e": (0.1, 0.4), 'd': (1.4, 2.0)} # for bayesopt      

    # Uncertain parameters: ground stiffness for two pile groups
    mu = torch.tensor([100.0, 100.0])                   # k0_1, k0_2 [kN/mm]                
    cov = torch.tensor([[    (20)**2,  0.5*(20)**2],    # COV = 30%, correlation = 0.5              
                        [0.5*(20)**2,      (20)**2]])
    dist = MultivariateNormal(mu, cov) 

    problem.set_bounds(bounds) 
    problem.set_dist(dist) 
    problem.add_model(model) 
    problem.add_objectives([model.f]) 
    problem.add_constraints([model.g1, model.g2, model.g3, model.g4]) # Transfer state  
    problem.add_constraints([model.g5, model.g6, model.g7, model.g8]) # Service state

    xi = problem.sample_xi(nsamples=int(1e2)).to(device)
    surrogate = Surrogate(problem, ntraining=ninitial, nstarts=5) 
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.001) 
    borc = Borc(surrogate, acquisition) 
    borc.cuda(device) 
    borc.initialize(nsamples=ninitial, sample_method="lhs", max_acq=torch.tensor([0.0])) 

    # BayesOpt used to sequentially sample [x,xi] points 
    res = torch.ones(iters, ) 
    for i in range(iters):    

        # new_x = argmax_[x,xi] EI x PF
        borc.acquisition = Acquisition(f="EI", g="PF", xi=xi, eps=1.0)
        new_x, _ = borc.batch_optimize_acq(iters=50, nstarts=5) 

        # fbest = max_[x,xi] mu 
        borc.acquisition = Acquisition(f="MU") 
        _, borc.fbest = borc.batch_optimize_acq(iters=50, nstarts=5) 
        borc.step(new_x=new_x) 
        # print(f"new_x : {new_x}") 

        # argmax_x E[f(x,xi)] s.t. P[g_i(x,xi)<0]>1-β, i=1,2...,m 
        if i % n == 0: 
            borc.acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.001)
            xopt, _ = borc.constrained_optimize_acq(iters=50, nstarts=1, optimize_x=True) 
            res[i], _ = problem.rbo(xopt, output=False, return_vals=True) # true E[f(x,xi)] 
            print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

    return xopt, res   

if __name__ == "__main__": 

    ninitial, iters, n = 20, 10, 1 
    xopt, res = bayesopt(ninitial, iters, n) 