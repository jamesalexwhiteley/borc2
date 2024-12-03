import torch
import borc2.optimize

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Borc():
    def __init__(self, 
                 problem, 
                 surrogate, 
                 acquisition):
        '''
        Basyeian Optimization with Reliability Constraints. 

        Parameters
        ----------
        problem : problem class instance 
        surrogate : surrogate class instance  
        acquisition : acquisition class instance  

        '''
        self.problem = problem 
        self.acquisition = acquisition 
        self.surrogate = surrogate 
        self.surrogate.add_problem(problem)
        self.device = self.surrogate.device 
        self.bounds = torch.stack(list(self.problem.param_bounds.values()), dim=0)
        self.nrepeats = torch.tensor([10]) 

    def cuda(self, device):
        self.nrepeats.to(device)
        self.acquisition.cuda(device)
        self.surrogate.cuda(device)
        self.device = device 

    def initialize(self, nsamples='default', sample_method="lhs", xbest=torch.tensor([0.0]), fbest=torch.tensor([0.0])):
        """
        Build surrogate GPs of problem objectives and constraints

        Parameters
        ----------
        nsamples : int
            number of input points used to initialise GPs
        
        """
        if nsamples == 'default':
            n = self.problem.sample_xi().shape[1] # no. stochastic parameters 
            nsamples = int((n+1)*(n+2) / 2) # Bichon et al. "Efficient global reliability analysis for nonlinear implicit performance functions." 2008.

        self.sample_method = sample_method 
        if self.surrogate.sample_method != None:
            self.sample_method = self.surrogate.sample_method 

        self.surrogate.build(nsamples=nsamples, sample_method=sample_method) 
        self.xbest, self.fbest = xbest, fbest
        self.xbest, self.fbest = self.xbest.to(self.device), self.fbest.to(self.device)
        self.max_acq = fbest # NOTE tbc   

    def eval_acqf(self, x):
        acq = [self.acquisition.f(x, gp, self.fbest).ravel() for gp in self.surrogate.objective_gps] 
        return torch.stack(acq, dim=1)
    
    def eval_acqg(self, x):
        acq = [self.acquisition.g(x, gp, self.fbest).ravel() for gp in self.surrogate.constraint_gps]
        return torch.stack(acq, dim=1) if acq != [] else []

    def eval_acquisition(self, x): 
        """
        Evaluate the acquisiton function at the point x 

        Parameters
        ----------
        x : torch.Tensor, shape=(nsamples, nparam)  
            point(s) to evaluate  

        Returns
        -------
        torch.Tensor, shape=(nsamples, ) 
            acquisition function evaluated at x

        """ 
        acqf = self.eval_acqf(x)
        acqg = self.eval_acqg(x)
        acqf = acqf.to(x.device) 
        acqg = acqg.to(x.device) if acqg != [] else acqg

        if acqg != []: 
            return torch.sum(acqf, dim=1) * torch.prod(acqg, dim=1) 
        else: 
            return torch.sum(acqf, dim=1) 
    
    def _batch_optimize_acq(self, xpts, iters=10, optimize_x=False, optimize_xi=False):
        """
        Optimize the acquisition function in batch mode over the points xpts 

        """ 
        # set up bounds etc.
        self.new_x, max_acq = xpts[0], self.max_acq

        if optimize_xi or self.problem.param_bounds == None:
            bounds = self.problem.param_dist.bounds()
        elif optimize_x or self.problem.param_dist == None:
            bounds = self.bounds 
        else:
            bounds = torch.cat((self.bounds, self.problem.param_dist.bounds()), dim=0) 

        # device 
        xpts = xpts.to(self.device)
        bounds = bounds.to(self.device)
        self.new_x, max_acq = self.new_x.to(self.device), max_acq.to(self.device) 

        xpts, acq = borc2.optimize.ADAM(self.eval_acquisition, xpts, iters, bounds) 

        # choose best from multiple starts 
        with torch.no_grad():
            max_acq_index = torch.argmax(acq)
            max_acq = acq[max_acq_index]
            self.new_x = xpts[max_acq_index].unsqueeze(0)
            
        return self.new_x.clone().detach(), max_acq.clone().detach()
    
    def batch_optimize_acq(self, iters=10, nstarts=5, optimize_x=False, optimize_xi=False):
        """ 
        Optimize the acquisition function. 

        """ 
        if optimize_x:
            xpts = self.problem.sample_x(nsamples=nstarts, method=self.sample_method)
        elif optimize_xi:               
            xpts = self.problem.sample_xi(nsamples=nstarts, method=self.sample_method)
        else:
            xpts = self.problem.sample(nsamples=nstarts, method=self.sample_method)
        new_x, max_acq = self._batch_optimize_acq(xpts=xpts, iters=iters, optimize_x=optimize_x, optimize_xi=optimize_xi) 

        return new_x, max_acq
    
    def _constrained_optimize_acq(self, xpts, iters, optimize_x=False, optimize_xi=False):
        """
        Optimize the acquisition function using the start points xpts 

        """ 
        # # set up bounds etc.
        # self.new_x, max_acq = xpts[0], self.eval_acquisition(xpts[0]) 
        
        if optimize_xi or self.problem.param_bounds == None:
            bounds = self.problem.param_dist.bounds()
        elif optimize_x or self.problem.param_dist == None: 
            bounds = self.bounds    
        else:
            bounds = torch.cat((self.bounds, self.problem.param_dist.bounds()), dim=0) 

        # device 
        xpts = xpts.to(self.device)
        bounds = bounds.to(self.device)
        # self.new_x, max_acq = self.new_x.to(self.device), max_acq.to(self.device) 

        f = self.eval_acqf
        g = self.eval_acqg

        # optimize 
        for x in xpts: 
            x, acq = borc2.optimize.CMA_ES(f, g, x.flatten(), iters, bounds)   

            # # choose best from multiple starts
            # with torch.no_grad(): 
            #     if acq > max_acq:   
            #         max_acq = acq
            #         self.new_x = x
    
        # return self.new_x.clone().detach(), max_acq.clone().detach()
        self.new_x = x
        return x, acq 
    
    def constrained_optimize_acq(self, iters=100, nstarts=5, optimize_x=False, optimize_xi=False):
        """ 
        Optimize the acquisition function. 

        """ 
        if optimize_x:
            xpts = self.problem.sample_x(nsamples=nstarts, method=self.sample_method).unsqueeze(1)
        elif optimize_xi:               
            xpts = self.problem.sample_xi(nsamples=nstarts, method=self.sample_method).unsqueeze(1)
        else:
            xpts = self.problem.sample(nsamples=nstarts, method=self.sample_method).unsqueeze(1)
        new_x, max_acq = self._constrained_optimize_acq(xpts=xpts, iters=iters, optimize_x=optimize_x, optimize_xi=optimize_xi) 

        return new_x, max_acq

    def step(self, new_x=None): 
        """
        get (new_x, new_y) then update the GP models

        """      
        if new_x != None: 
            self.fbest, self.xbest = self.surrogate.update(new_x)
        else: 
            self.fbest, self.xbest = self.surrogate.update(self.new_x)

    def optimize(self, iters=10, acq_iters=20, nstarts=5, output=False):
        """
        Run the optimization 

        """
        print(f"Bayesian Optimization | Num objectives = {len(self.surrogate.objective_gps)}, Num constraints = {len(self.surrogate.constraint_gps)}")
        
        for i in range(iters):
            if output:
                print(f"Iter: {i + 1}/{iters} | Max Objective: {self.fbest},  Optimal x : {self.xbest}")
            self.optimize_acq(acq_iters, nstarts) 
            self.step()      

    def rbo(self, x, nsamples=int(5e2), output=True, return_vals=False):
        """
        Monte carlo estimate of RBO objective and constraint(s) 

        """
        x_batch, xi_samples = self.problem.gen_batch_data(x, nsamples=nsamples, fixed_base_samples=True)
        f = self.surrogate.predict_objectives(x_batch)[0].mu
        g = self.surrogate.predict_constraints(x_batch)[0] # NOTE multiple constraints not tested  

        mu = f.mean(dim=1)
        if g != []:
            gm, gstd = g.posterior()
            p = torch.distributions.Normal(gm, gstd).cdf(torch.tensor([0.0]).to(gm.device)) 
            pi = p.mean(dim=1).unsqueeze(0) # \int Phi(-mu/std) p(xi)dxi 
            # pi = (torch.sum(gm <= 0, dim=1) / nsamples).unsqueeze(0) if g != [] else [] # surrogate mean estimate 
        else:
            pi = []

        if output:
            print(f"x = {list(x.detach().cpu())}")
            print(f"E[f(x,xi)] = {mu}")
            for i, p in enumerate(pi):
                print(f"P[g_{i+1}(x,xi)<0] = {p}")

        if return_vals:
            return mu, [p for p in pi]