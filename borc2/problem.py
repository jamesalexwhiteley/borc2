import torch
from itertools import product
from scipy.stats import qmc 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Problem(): 
    def __init__(self): 
        """ 
        Sampling, objective and constraint evaluation methods are contained within Problem class

        """
        self.obj_fun = [] # objective functions 
        self.con_fun = [] # constraint functions 
        self.param_bounds = None 
        self.param_dist = None 

    def set_bounds(self, param_bounds):
        """ 
        Set bounds on deterministic parameters 

        Parameters
        ----------
        param_bounds : dict of {str : tuple}
            parameter name and associated upper and lower bound 

        """ 
        for key in param_bounds:
            param_bounds[key] = torch.tensor(param_bounds[key]) # convert dict values to torch.tensor 
        self.param_bounds = param_bounds 

    def set_dist(self, param_dist): 
        """ 
        Set normal distributions over probabilistic parameters 

        Parameters
        ----------
        param_dist : 
            distribution for uncertain parameter(s),
                1) torch.distributions.MultivariateNormal
                2) borc.probability.DiscreteJoint

        """ 
        self.param_dist = param_dist 

    def sampling_design(self, nsamples, dim, method):
        """
        Sampling scheme
        
        """
        if method == "sobol":
            sobol_engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
            samples = sobol_engine.draw(nsamples+1)[1:nsamples+1] # discard first sample 
        elif method == "lhs":
            samples = torch.tensor(qmc.LatinHypercube(d=dim).random(nsamples)) 
        return samples 
        
    def sample_x(self, nsamples=1, method="sobol", samples=None, dtype=torch.float):
        """ 
        Sample determinisitc parameters uniformly between bounds

        Parameters
        ----------
        nsamples : int 
            number of samples 
        method : string 
            "rand" : random sampling 
            "sobol" : sobol sequence 
            "lhs" : latin hypercube sampling 

        Returns
        -------
        self.x : torch.Tensor, shape=(nsamples, nparam)  
            sampled x points 

        """ 
        if nsamples == 1 or method == "rand":
            b = list(self.param_bounds.values())
            return torch.stack([torch.rand(nsamples) * (upper - lower) + lower for lower, upper in b], dim=1)

        if samples == None:
            samples = self.sampling_design(nsamples, dim=len(self.param_bounds), method=method) 

        # scale : [0, 1] -> [a, b]
        x =  torch.zeros_like(samples)
        for i, b in enumerate(self.param_bounds.values()):
            x[:, i] = samples[:, i] * (b[1] - b[0]) + b[0] 
                
        return x.to(dtype)

    def sample_xi(self, nsamples=1, method="sobol", samples=None, dtype=torch.float):
        """ 
        Sample from normal distributions over probabilistic parameters 

        Parameters
        ----------
        nsamples : int 
            number of samples 
        method : string 
            "rand" : random sampling 
            "sobol" : sobol sequence 
            "lhs" : latin hypercube sampling 
        
        Returns
        -------
        self.xi : torch.Tensor, shape=(nsamples, nparam)
            sampled xi points 

        """ 
        if nsamples==1 or method == "rand":
            return self.param_dist.sample((nsamples,))

        if samples == None:
            samples = self.sampling_design(nsamples, dim=self.param_dist.sample((1,))[0].size(0), method=method) 

        # transform [0, 1] -> p(xi)
        xi = self.param_dist.transform_uniform_samples(samples)

        return xi.to(dtype)

    def sample(self, nsamples=1, method="sobol", dtype=torch.float):
        """ 
        Sample both determinisitc and probabilistic parameters 

        Parameters
        ----------
        nsamples : int 
            number of samples 
        method : string 
            "rand" : random sampling 
            "sobol" : sobol sequence 
            "lhs" : latin hypercube sampling 

        Returns
        -------
        self.x : torch.Tensor, shape=(nsamples, nparam) 
            sampled (x, xi) points 
 
        """ 
        if self.param_dist == None:
            return self.sample_x(nsamples, method=method)
        
        elif self.param_bounds == None:
            return self.sample_xi(nsamples, method=method)

        else:
            if nsamples==1 or method=="rand":
                x = self.sample_x(nsamples, method=method)
                xi = self.sample_xi(nsamples, method=method)
                return torch.cat((x, xi), dim=1)
            
            dim_x, dim_xi = len(self.param_bounds), self.param_dist.sample((1,))[0].size(0)
            samples = self.sampling_design(nsamples, dim=dim_x + dim_xi, method=method) 

            self.x = torch.zeros_like(samples)
            self.x[:, 0:dim_x] = self.sample_x(nsamples, method=method, samples=samples[:, 0:dim_x])    
            self.x[:, dim_x : dim_x + dim_xi] = self.sample_xi(nsamples, method=method, samples=samples[:, dim_x : dim_x + dim_xi])

            return self.x.to(dtype)
        
    def gen_batch_data(self, x, xi=None, nsamples=int(5e2), fixed_base_samples=True, method="sobol"):
        """
        Generate batch data (associate each x point with many [sampled] xi points)

        Parameters
        ----------
        x : torch.Tensor, shape=(npoints, nparam)  
            x points 
        xi : torch.Tensor, shape=(npoints, nparam)  
            xi points, set xi=None to sample them 
        nsamples : int 
           the number of xi samples to take 
        method : string 
            "rand" : random sampling 
            "sobol" : sobol sequence 
            "lhs" : latin hypercube sampling 

        Returns 
        ------- 
        x : torch.Tensor, shape=(npoints, nsamples, nparam)
            batch tensor of points to evaluate, 
            e.g., size of x is [401, 500, 3] for 401 1d x points each with 500 2d xi samples
        xi : torch.Tensor, shape=(npoints, nparam) 
            xi base samples 
            
        """
        if fixed_base_samples:
            if xi == None:
                xi = self.sample_xi(nsamples=nsamples, method=method).to(x.device) # keep consistent device with x
                
            X = x.unsqueeze(1).repeat(1, xi.shape[0], 1)
            XI = xi.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat((X, XI), dim=-1) 
            return x, xi

        if not fixed_base_samples:
            if nsamples == 1:
                method="rand"
            xi = self.sample_xi(nsamples=nsamples * x.shape[0], method=method).to(x.device) # keep consistent device with x
        
            X = x.unsqueeze(1).repeat(1, nsamples, 1)
            XI = xi.reshape(X.shape[0], X.shape[1], xi.shape[1])
            x = torch.cat((X, XI), dim=-1) 
            return x, xi
        
    def add_model(self, model): 
        """
        Add an engineering model, upon which the constraints and objectives may be based 

        Parameters
        ----------
            user defined function 

        """ 
        self.internal_model = model 

    def model(self, x):
        """
        Run the engineering model at the design point 

        Parameters
        ----------
        x : torch.Tensor, shape=(nsamples, nparam)
            points to evaluate 

        Returns 
        -------
        torch.Tensor, shape=(nsamples, nparam) 
            the results from running the engineering model 

        """
        self.internal_model(x) # runs the model 
        self.m = self.internal_model.m
        return self.m

    def add_objectives(self, objectives):
        """
        Add objective functions 

        Parameters
        ----------
        objectives : list 
            a collections of user defined objective functions

        """
        for f in objectives:
            self.obj_fun.append(f) 

    def add_constraints(self, constraints):
        """
        Add constraint functions, and associated reliability level 

        Parameters
        ----------
        constraints : list 
            a collections of user defined constraint functions

        """
        for g in constraints:
            self.con_fun.append(g) 

    def objectives(self, m=None):
        """
        Evaluate objectives 

        Parameters
        ----------
        m : torch.Tensor, shape=(nsamples, nparam)
            points to evaluate 

        Returns
        -------
        torch.Tensor, shape=(nobjectives, nsamples)  
            calculated objectives at m   

        """
        if not m == None:
            self.internal_model.m = m
        return torch.stack([f() for f in self.obj_fun]).transpose(0, 1)

    def constraints(self, m=None): 
        """
        Evaluate constraints 

        Parameters
        ----------
        m : torch.Tensor, shape=(nsamples, nparam)
            points to evaluate 

        Returns
        -------
        torch.Tensor, shape=(nconstraints, nsamples)    
            calculated constraints at m   

        """
        if not m == None:
            self.internal_model.m = m

        if len(self.con_fun) == 0:
            return [] 
        else:
            return torch.stack([g() for g in self.con_fun]).transpose(0, 1) 
        
    def rbo(self, x, nsamples=int(5e2), output=True, return_vals=False):
        """
        Monte carlo estimate of RBO objective and constraint(s) 

        """
        x_batch, _ = self.gen_batch_data(x, nsamples=nsamples, fixed_base_samples=True)
        m = self.model(x_batch.view(-1, x_batch.size(-1)))
        f = self.objectives(m)
        g = self.constraints(m)

        mu = f.view(x_batch.size(0), -1).mean(dim=1)
        if g != []:
            gv = g.view(x_batch.size(0), x_batch.size(1), g.size(-1))
            pi = (torch.sum(gv <= 0, dim=1) / nsamples).T if g != [] else []
        else:
            pi = []

        if output:
            print(f"x = {list(x.detach().cpu())}")
            print(f"E[f(x,xi)] = {mu}")
            for i, p in enumerate(pi):
                print(f"P[g_{i+1}(x,xi)<0] = {p}")

        if return_vals:
            return mu, [p for p in pi]

    def _monte_carlo(self, 
                    obj_fun,
                    con_fun,
                    surrogate,
                    params, 
                    nsamples=int(1e4), 
                    obj_type="det",
                    obj_ucb=[1],
                    con_type="prob", 
                    con_ucb=[1],
                    con_eps=0.1,
                    output=True,
                    device='cpu'):
        """
        Simple monte carlo implementation using full factorial sampling.

        Parameters 
        ---------- 
        params : tuple 
            tuple of values for each deterministic variable, e.g., 
                x1 = torch.linspace(100, 500, steps=9) 
                x2 = torch.linspace(400, 800, steps=9) 
                params = (x1, x2)
                or alternatively for one variable, just do params = (x1,)
        obj_type : string 
            indicates how to calculate objective, and whether the objective is probabilisitic  
            det : determinisitic, calculate f(x)
            mean : calculate expected value f1(x) = E[f(x,xi)]
        con_type : string
            indicates how to calculate constraint, and whether the objective is probabilisitic
            prob : of the form g1(x) = P[g(x,xi)<0]>1-epsilon 
        epsilon : float 
            risk level for probabilistic constraints 
        
        """
        if len(params) == 1:
            ffs = torch.stack(params, dim=1)
        else:
            ffs = torch.tensor(list(product(*params))) 

        res = torch.full((len(ffs),), -float('inf'))
        if output:
            print(f"Monte Carlo Simulation | Num possible solutions = {len(ffs)}")
        xi = self.sample_xi(nsamples).to(device)

        # loop over all combinations 
        for i, x in enumerate(ffs):

            # stack monte carlo samples for each x
            xx = torch.stack([x] * nsamples).to(device)
            x = torch.cat((xx, xi), dim=1).to(device)

            # either run model or run surrogate 
            if surrogate:
                m = x
            else:
                m = self.model(x)

            # estimate constraint(s)
            cm = con_fun(m)
            if len(self.con_fun) == 0:
                constraint = True
            elif con_type == "prob":
                estimated_g = torch.sum(cm <= 0, dim=0) / nsamples 
                constraint = all(estimated_g >= 1 - con_eps)
            elif con_type == "lcb":
                estimated_g = torch.mean(cm, dim=0) - torch.tensor(con_ucb[0]) * torch.std(cm, dim=0)
                constraint = all(estimated_g <= 0)
            elif con_type == "ucb":
                estimated_g = torch.mean(cm, dim=0) + torch.tensor(con_ucb[0]) * torch.std(cm, dim=0)
                constraint = all(estimated_g <= 0)

            # estimated objective (single objective)
            om = obj_fun(m)
            if obj_type == "det": 
                if constraint:  
                    self.internal_model.x = x[0].unsqueeze(0) # independent of probabilistic parameters
                    res[i] = om
            elif obj_type == "mean": 
                if constraint: 
                    res[i] = torch.mean(om)
            elif obj_type == "lcb": 
                if constraint:
                    res[i] = torch.mean(om) - torch.tensor(obj_ucb[0]) * torch.std(om)
            elif obj_type == "ucb": 
                if constraint:
                    res[i] = torch.mean(om) + torch.tensor(obj_ucb[0]) * torch.std(om)

        max_val, max_index = torch.max(res, 0)
        ind = max_index.numpy()

        if output: 
            print(f"Max Objective: {max_val.item():.4f} | Optimal x : {ffs[ind].numpy()}")   
        return ffs[ind].unsqueeze(0), max_val
    
    def monte_carlo(self, 
                    params, 
                    nsamples=int(1e4), 
                    obj_type="det",
                    obj_ucb=[1],
                    con_type="prob", 
                    con_ucb=[1],
                    con_eps=0.1,
                    output=True,
                    device='cpu'):
        """
        Simple monte carlo implementation using full factorial sampling.

        Parameters 
        ---------- 
        params : tuple 
            tuple of values for each deterministic variable, e.g., 
                x1 = torch.linspace(100, 500, steps=9) 
                x2 = torch.linspace(400, 800, steps=9) 
                params = (x1, x2)
                or alternatively for one variable, just do params = (x1,)
        obj_type : string 
            indicates how to calculate objective, and whether the objective is probabilisitic  
            det : determinisitic, calculate f(x)
            mean : calculate expected value f1(x) = E[f(x,xi)]
        con_type : string
            indicates how to calculate constraint, and whether the objective is probabilisitic
            prob : of the form g1(x) = P[g(x,xi)<0]>1-epsilon 
        epsilon : float 
            risk level for probabilistic constraints 
        
        """
        return self._monte_carlo( 
                    obj_fun=self.objectives,
                    con_fun=self.constraints,
                    surrogate=False,
                    params=params, 
                    nsamples=nsamples, 
                    obj_type=obj_type,
                    obj_ucb=obj_ucb,
                    con_type=con_type, 
                    con_ucb=con_ucb,
                    con_eps=con_eps,
                    output=output,
                    device=device)
    
  