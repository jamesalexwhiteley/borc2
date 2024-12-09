import torch 
from borc2.problem import Problem 
from borc2.surrogate import Surrogate
from borc2.acquisition import Acquisition
from borc2.bayesopt import Borc
from borc2.probability import DiscreteJoint
from borc2.utilities import tic, toc 
from branin_rs import branin_williams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: James Whiteley (github.com/jamesalexwhiteley)

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

    xi = problem.sample_xi(nsamples=int(1e2)).to(device)
    surrogate = Surrogate()
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1)
    borc = Borc(problem, surrogate, acquisition) 
    borc.cuda(device) 
    borc.initialize(nsamples=ninitial, sample_method="lhs", xbest=problem.sample_x(), fbest=torch.tensor([0.0])) 
                               
    # BayesOpt used to sequentially sample [x,xi] points 
    res = torch.ones(iters, ) 
    for i in range(iters):    

        # new_x = argmax_[x,xi] EI x PF
        borc.acquisition = Acquisition(f="EI", g="PF", xi=xi)
        new_x, _ = borc.batch_optimize_acq(iters=50, nstarts=20) 

        # fbest = max_[x,xi] mu 
        borc.acquisition = Acquisition(f="MU") 
        _, borc.fbest = borc.batch_optimize_acq(iters=50, nstarts=20) 
        borc.step(new_x=new_x) 
        # print(f"new_x : {new_x}") 

        # argmax_x E[f(x,xi)] s.t. P[g(x,xi)<0]>1-epsilon 
        if i % n == 0: 
            borc.acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1)
            xopt, _ = borc.constrained_optimize_acq(iters=int(1e2), nstarts=4, optimize_x=True) 
            res[i], _ = problem.rbo(xopt, output=False, return_vals=True) # true E[f(x,xi)] 
            print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

    return xopt, res 


if __name__ == "__main__": 
    ninitial, iters, n = 200, 4, 1
    xopt, res = bayesopt(ninitial, iters, n) 