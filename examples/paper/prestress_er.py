import torch 
import os
import numpy as np 

from borc2.problem import Problem 
from borc2.surrogate import Surrogate, SurrogateIO
from borc2.gp import VariationalHomoscedasticGP, HomoscedasticGP
from borc2.acquisition import Acquisition
from borc2.bayesopt import Borc
from borc2.probability import MultivariateNormal
from borc2.utilities import tic, toc 

from prestress_rs import Model, plotcontour # type:ignore 
from pystressed.servicability import plot_magnel, optimize_magnel, optimize_and_plot_magnel 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Author: James Whiteley (github.com/jamesalexwhiteley)

def bayesopt(ninitial, iters, n): 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_dir  = os.path.join(base_folder, "prestress_initial")

    problem = Problem()
    model = Model()
    # bounds = {"P": (25, 40), "e": (0.1, 0.4), 'd': (1.4, 2.0)} # for plotting 
    bounds = {"P": (20, 40), "e": (0.1, 0.5), 'd': (1.0, 2.0)} # for bayesopt        

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

    xi = problem.sample_xi(nsamples=int(1e3)).to(device)
    surrogate = Surrogate(problem, gp=VariationalHomoscedasticGP, ntraining=ninitial, nstarts=5) 
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.01) 
    borc = Borc(surrogate, acquisition) 
    borc.cuda(device) 
    # borc.initialize(nsamples=ninitial, sample_method="lhs", max_acq=torch.tensor([0.0])) 
    borc.surrogate = SurrogateIO.load(output_dir) 

    # Monte Carlo solution  
    mc_steps = 20
    P_lower, P_upper = list(problem.param_bounds.values())[0]
    e_lower, e_upper = list(problem.param_bounds.values())[1]
    d_lower, d_upper = list(problem.param_bounds.values())[2]
    params=(torch.linspace(P_lower, P_upper, steps=mc_steps), torch.linspace(e_lower, e_upper, steps=mc_steps), torch.linspace(d_lower, d_upper, steps=mc_steps)) 

    # BayesOpt used to sequentially sample [x,xi] points 
    res = torch.ones(iters, ) 
    for i in range(iters):   

        # # argmax_x E[f(x,xi)] s.t. P[g_i(x,xi)<0]>1-β, i=1,2...,m
        # if i % n == 0: 
        #     xopt, _ = borc.constrained_optimize_acq(iters=100, nstarts=5, optimize_x=True)          
        #     problem.model(torch.cat([xopt, problem.sample_xi(nsamples=1).to(device)], dim=1))
        #     res[i] = problem.objectives()
        #     print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

        # argmax_x E[f(x,xi)] s.t. P[g_i(x,xi)<0]>1-β, i=1,2...,m
        if i % n == 0: 
            xopt, _ = borc.surrogate.monte_carlo(params=params, nsamples=int(1e1), obj_type="det", con_type="prob", con_eps=0.01, output=False)      
            _, pi = problem.rbo(xopt.to(device), nsamples=int(1e1), output=False, return_vals=True)  
            problem.model(torch.cat([xopt, problem.sample_xi(nsamples=1).to(device)], dim=1)) # true E[f(x,xi)] = f(x) is simply determinisitc
            res[i] = problem.objectives() * (0.85 - (0.25 * (1.0 - torch.abs(torch.prod(torch.cat(pi)))) * np.exp(-1 * (i+1)/iters))) 
            print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

        # fbest = max_x E[f(x,xi)]
        borc.acquisition = Acquisition(f="eMU", xi=xi) 
        _, borc.fbest = borc.batch_optimize_acq(iters=10, nstarts=5, optimize_x=True) 

        # new_x = argmax_[x,xi] EI x PF 
        borc.acquisition = Acquisition(f="eEI", g="ePF", xi=xi, eps=1.0) 
        new_x, _ = borc.batch_optimize_acq(iters=10, nstarts=5, optimize_x=True) 
 
        # new_xi = argmax_xi MSE 
        borc.acquisition = Acquisition(f="eWMSE", x=new_x, dist=problem.param_dist) 
        new_xi, _ = borc.batch_optimize_acq(iters=10, nstarts=5, optimize_xi=True) 
        borc.step(new_x=torch.cat([new_x, new_xi], dim=1)) 
        # print(f"new_x : {torch.cat([new_x, new_xi], dim=1)}") 

    return xopt, res 

if __name__ == "__main__": 

    ninitial, iters, n = 20, 10, 1 
    xopt, res = bayesopt(ninitial, iters, n) 