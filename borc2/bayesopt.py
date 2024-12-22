import torch
import borc2.optimize

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Borc():
    def __init__(self, 
                #  problem, 
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
        # self.problem = surrogate.problem 
        # self.surrogate.add_problem(problem)
        self.acquisition = acquisition 
        self.surrogate = surrogate 
        self.sample_method = surrogate.sample_method 
        self.device = self.surrogate.device 
        self.bounds = torch.stack(list(self.surrogate.problem.param_bounds.values()), dim=0)
        self.nrepeats = torch.tensor([10]) 
        self.max_acq = torch.tensor([0])

    def cuda(self, device):
        self.nrepeats.to(device)
        self.acquisition.cuda(device)
        self.surrogate.cuda(device)
        self.device = device 

    def initialize(self, nsamples='default', sample_method="lhs", max_acq=torch.tensor([0.0])):
        """
        Build surrogate GPs of problem objectives and constraints

        Parameters
        ----------
        nsamples : int
            number of input points used to initialise GPs
        
        """
        if nsamples == 'default':
            n = self.surrogate.problem.sample_xi().shape[1] # no. stochastic parameters 
            nsamples = int((n+1)*(n+2) / 2) # Bichon et al. "Efficient global reliability analysis for nonlinear implicit performance functions." 2008.

        self.sample_method = sample_method 
        if self.surrogate.sample_method != None:
            self.sample_method = self.surrogate.sample_method 

        self.surrogate.build(nsamples=nsamples, sample_method=sample_method) 
        # self.xbest, self.fbest = xbest, fbest
        # self.xbest, self.fbest = self.xbest.to(self.device), self.fbest.to(self.device)
        self.max_acq = max_acq.requires_grad_(True).to(self.device) 

    def eval_acqf(self, x):
        fbest = self.surrogate.fbest.requires_grad_(True).to(self.device)
        acq = [self.acquisition.f(x, gp, fbest).ravel() for gp in self.surrogate.objective_gps] 
        return torch.stack(acq, dim=1)
    
    def eval_acqg(self, x):
        fbest = self.surrogate.fbest.requires_grad_(True).to(self.device)
        acq = [self.acquisition.g(x, gp, fbest).ravel() for gp in self.surrogate.constraint_gps]
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
    
    def _batch_optimize_acq(self, xpts, lr, iters=10, optimize_x=False, optimize_xi=False):
        """
        Optimize the acquisition function in batch mode over the points xpts 

        """ 
        # set up bounds etc.
        self.new_x, max_acq = xpts[0], self.max_acq

        if optimize_xi or self.surrogate.problem.param_bounds == None:
            bounds = self.surrogate.problem.param_dist.bounds()
        elif optimize_x or self.surrogate.problem.param_dist == None:
            bounds = self.bounds 
        else:
            bounds = torch.cat((self.bounds, self.surrogate.problem.param_dist.bounds()), dim=0) 

        # device 
        xpts = xpts.to(self.device)
        bounds = bounds.to(self.device)
        self.new_x, max_acq = self.new_x.to(self.device), max_acq.to(self.device) 

        xpts, acq = borc2.optimize.ADAM(self.eval_acquisition, xpts, iters, bounds, lr=lr) 

        # choose best from multiple starts 
        with torch.no_grad():
            max_acq_index = torch.argmax(acq)
            max_acq = acq[max_acq_index]
            self.new_x = xpts[max_acq_index].unsqueeze(0)
            
        return self.new_x.clone().detach(), max_acq.clone().detach()
    
    def batch_optimize_acq(self, iters=10, nstarts=5, optimize_x=False, optimize_xi=False, lr=0.1):
        """ 
        Optimize the acquisition function. 

        """ 
        if optimize_x:
            xpts = self.surrogate.problem.sample_x(nsamples=nstarts, method=self.sample_method)
        elif optimize_xi:               
            xpts = self.surrogate.problem.sample_xi(nsamples=nstarts, method=self.sample_method)
        else:
            xpts = self.surrogate.problem.sample(nsamples=nstarts, method=self.sample_method)
        new_x, max_acq = self._batch_optimize_acq(xpts=xpts, iters=iters, lr=lr, optimize_x=optimize_x, optimize_xi=optimize_xi) 

        return new_x, max_acq

    def _constrained_optimize_acq(self, xpts, iters, optimize_x=False, optimize_xi=False):
        """
        Optimize the acquisition function using the start points xpts 

        """ 
        new_x, max_acq = None, None

        # bounds         
        if optimize_xi or self.surrogate.problem.param_bounds == None:
            bounds = self.surrogate.problem.param_dist.bounds()
        elif optimize_x or self.surrogate.problem.param_dist == None: 
            bounds = self.bounds    
        else:
            bounds = torch.cat((self.bounds, self.surrogate.problem.param_dist.bounds()), dim=0) 

        # device 
        xpts = xpts.to(self.device)
        bounds = bounds.to(self.device)

        # optimize 
        f = self.eval_acqf 
        g = self.eval_acqg 

        for x0 in xpts: 
            x, acq = borc2.optimize.CMA_ES(f, g, x0.flatten(), iters, bounds)  

            # choose best from multiple starts
            if max_acq == None: 
                new_x, max_acq = x, acq

            elif acq > max_acq:   
                new_x, max_acq = x, acq
    
        return new_x, max_acq 
    
    def constrained_optimize_acq(self, iters=50, nstarts=1, optimize_x=False, optimize_xi=False):
        """ 
        Optimize the acquisition function. 

        """ 
        if optimize_x:
            xpts = self.surrogate.problem.sample_x(nsamples=nstarts, method=self.sample_method).unsqueeze(1)
        elif optimize_xi:               
            xpts = self.surrogate.problem.sample_xi(nsamples=nstarts, method=self.sample_method).unsqueeze(1)
        else:
            xpts = self.surrogate.problem.sample(nsamples=nstarts, method=self.sample_method).unsqueeze(1)
            
        new_x, max_acq = self._constrained_optimize_acq(xpts=xpts, iters=iters, optimize_x=optimize_x, optimize_xi=optimize_xi) 

        return new_x, max_acq

    def step(self, new_x=None): 
        """
        get (new_x, new_y) then update the GP models

        """      
        # if new_x != None: 
        #     self.fbest, self.xbest = self.surrogate.update(new_x)
        # else: 
        #     self.fbest, self.xbest = self.surrogate.update(self.new_x)
        if new_x != None: 
            self.surrogate.update(new_x)
        else: 
            self.surrogate.update(self.new_x)

    # def optimize(self, iters=10, acq_iters=20, nstarts=5, output=False):
    #     """
    #     Run the optimization 

    #     """
    #     print(f"Bayesian Optimization | Num objectives = {len(self.surrogate.objective_gps)}, Num constraints = {len(self.surrogate.constraint_gps)}")
        
    #     for i in range(iters):
    #         if output:
    #             # print(f"Iter: {i + 1}/{iters} | Max Objective: {self.fbest},  Optimal x : {self.xbest}")
    #         self.optimize_acq(acq_iters, nstarts) 
    #         self.step()      

    def rbo(self, x, nsamples=int(5e2), output=True, return_vals=False, return_posterior=False):
        """
        Monte carlo estimate of RBO objective and constraint(s) 

        NOTE use with multiple constraints has not been implemented 

        """
        x_batch, _ = self.surrogate.problem.gen_batch_data(x, nsamples=nsamples, fixed_base_samples=True)
        f = self.surrogate.predict_objectives(x_batch)[0]
        g = self.surrogate.predict_constraints(x_batch)[0]  

        f_mu, f_std = f.posterior()
        f1, f2 = f_mu.mean(dim=1), f_std.mean(dim=1)

        if g != []:
            g_mu, g_std = g.posterior()
            pi = torch.distributions.Normal(g_mu, g_std).cdf(torch.tensor([0.0]).to(g_mu.device)) 
            g1 = pi.mean(dim=1).unsqueeze(0) # \int Phi(-mu/std) p(xi)dxi 
            g2 = torch.sqrt(pi.var(dim=1, unbiased=True) / nsamples).unsqueeze(0) 
            # g1 = (torch.sum(g_mu <= 0, dim=1) / nsamples).unsqueeze(0) 
            # g2 = (g1 * (1 - g1)) / nsamples
            # print(g1)
            # print(g2)
            # indicator_samples = (g_mu <= 0).float() 
            # g1 = indicator_samples.mean(dim=1).unsqueeze(0)
            # g2 = indicator_samples.var(dim=1, unbiased=True).unsqueeze(0) / nsamples
            # print(g1) 
            # print(g2) 
        else: 
            g1 = [] 
            g2 = [] 

        if output:
            print(f"x = {list(x.detach().cpu())}")
            print(f"E[f(x,xi)] = {mu}")
            for i, p in enumerate(pi):
                print(f"P[g_{i+1}(x,xi)<0] = {p}")

        if return_vals:
            return f1, [p for p in g1]
        
        if return_posterior:
            return f1, f2, [p for p in g1], [p for p in g2]