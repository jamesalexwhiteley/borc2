import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import math 
import matplotlib.gridspec as gridspec

from borc2.problem import Problem 
from borc2.surrogate import Surrogate, SurrogateIO
from borc2.acquisition import Acquisition
from borc2.bayesopt import Borc
from borc2.probability import MultivariateNormal
from borc2.utilities import tic, toc 

from pybeamnlfea.model.frame import Frame 
from pybeamnlfea.model.material import LinearElastic 
from pybeamnlfea.model.section import Section 
from pybeamnlfea.model.element import ThinWalledBeamElement 
from pybeamnlfea.model.boundary import BoundaryCondition 
from pybeamnlfea.model.load import NodalLoad, UniformLoad

import warnings
warnings.filterwarnings("ignore", message=r".*Solution may be inaccurate. Try another solved*")  

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plotcontour(problem, borc):

    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'contour_prestress.png')

    # Input data 
    P_lower, P_upper = list(problem.param_bounds.values())[0]
    e_lower, e_upper = list(problem.param_bounds.values())[1]
    d_lower, d_upper = list(problem.param_bounds.values())[2]

    # Steps for plotting 
    plot_steps = 10
    P_vals = torch.linspace(P_lower, P_upper, steps=plot_steps)
    e_vals = torch.linspace(e_lower, e_upper, steps=plot_steps)
    d_vals = torch.linspace(d_lower, d_upper, steps=plot_steps)

    # Create cut values 
    P_cuts = torch.linspace(P_lower, P_upper, steps=3)
    e_cuts = torch.linspace(e_lower, e_upper, steps=3)
    d_cuts = torch.linspace(d_lower, d_upper, steps=3)

    # Storage for results
    pe_results = [] # f(P,e) with different d values
    pd_results = [] # f(P,d) with different e values
    ed_results = [] # f(e,d) with different P values

    # Monte Carlo samples
    nsamples = int(1e1)
    tic()

    # Calculate f(P,e) for different d values
    for d_val in d_cuts:
        P_grid_2d, e_grid_2d = torch.meshgrid(P_vals, e_vals, indexing='ij')
        xpts = torch.stack([
            P_grid_2d.reshape(-1),
            e_grid_2d.reshape(-1),
            torch.full_like(P_grid_2d.reshape(-1), d_val)
        ], dim=1)
        
        # Calculate probabilities for each point
        probs = []
        for x in xpts:
            pi = problem.rbo(x.unsqueeze(0), nsamples=nsamples, output=False, return_vals=True)[1]     
            # probs.append(pi[0])
            probs.append(np.prod(pi))
        
        # Reshape results to 2D grid for plotting
        prob_grid = torch.tensor(probs).reshape(P_grid_2d.shape)
        pe_results.append((P_grid_2d.numpy(), e_grid_2d.numpy(), prob_grid.numpy()))

    # Calculate f(P,d) for different e values
    for e_val in e_cuts:
        P_grid_2d, d_grid_2d = torch.meshgrid(P_vals, d_vals, indexing='ij')
        xpts = torch.stack([
            P_grid_2d.reshape(-1),
            torch.full_like(P_grid_2d.reshape(-1), e_val),
            d_grid_2d.reshape(-1)
        ], dim=1)
        
        # Calculate probabilities for each point
        probs = []
        for x in xpts:
            pi = problem.rbo(x.unsqueeze(0), nsamples=nsamples, output=False, return_vals=True)[1]  
            # probs.append(pi[0])
            probs.append(np.prod(pi))
        
        # Reshape results to 2D grid for plotting
        prob_grid = torch.tensor(probs).reshape(P_grid_2d.shape)
        pd_results.append((P_grid_2d.numpy(), d_grid_2d.numpy(), prob_grid.numpy()))

    # Calculate f(e,d) for different P values
    for P_val in P_cuts:
        e_grid_2d, d_grid_2d = torch.meshgrid(e_vals, d_vals, indexing='ij')
        xpts = torch.stack([
            torch.full_like(e_grid_2d.reshape(-1), P_val),
            e_grid_2d.reshape(-1),
            d_grid_2d.reshape(-1)
        ], dim=1)
        
        # Calculate probabilities for each point
        probs = []
        for x in xpts:
            pi = problem.rbo(x.unsqueeze(0), nsamples=nsamples, output=False, return_vals=True)[1]    
            # probs.append(pi[0])
            probs.append(np.prod(pi))
        
        # Reshape results to 2D grid for plotting
        prob_grid = torch.tensor(probs).reshape(e_grid_2d.shape)
        ed_results.append((e_grid_2d.numpy(), d_grid_2d.numpy(), prob_grid.numpy()))

    print("Data generation complete:")
    toc()
    tic()

    # Create grid 
    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])
    levels = np.linspace(0, 1.0, 5)

    # Plot f(P,e) with different d values
    for i, (P_grid, e_grid, prob_grid) in enumerate(pe_results):
        ax = plt.subplot(gs[i, 0])
        contour = ax.contourf(P_grid, e_grid, prob_grid, levels=20, cmap='Blues')
        contour_lines = ax.contour(P_grid, e_grid, prob_grid, levels=levels, linewidths=1.0, colors='black')
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        ax.set_xlabel('P (Force)')
        ax.set_ylabel('e (Eccentricity)')
        ax.annotate(f"d = {d_cuts[i]:.2f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.grid(True, linestyle='--', alpha=0.4)

    # Plot f(P,d) with different e values
    for i, (P_grid, d_grid, prob_grid) in enumerate(pd_results):
        ax = plt.subplot(gs[i, 1])
        contour = ax.contourf(P_grid, d_grid, prob_grid, levels=20, cmap='Blues')
        contour_lines = ax.contour(P_grid, d_grid, prob_grid, levels=levels, linewidths=1.0, colors='black')
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        ax.set_xlabel('P (Force)')
        ax.set_ylabel('d (Depth)')
        ax.annotate(f"e = {e_cuts[i]:.2f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.grid(True, linestyle='--', alpha=0.4)

    # Plot f(e,d) with different P values
    for i, (e_grid, d_grid, prob_grid) in enumerate(ed_results):
        ax = plt.subplot(gs[i, 2])
        contour = ax.contourf(e_grid, d_grid, prob_grid, levels=20, cmap='Blues')
        contour_lines = ax.contour(e_grid, d_grid, prob_grid, levels=levels, linewidths=1.0, colors='black')
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        ax.set_xlabel('e (Eccentricity)')
        ax.set_ylabel('d (Depth)')
        ax.annotate(f"P = {P_cuts[i]:.0f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.grid(True, linestyle='--', alpha=0.4)

    print("Data formatting complete:")
    toc()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.07, wspace=0.25, hspace=0.3)
    plt.show()

class Model():
    def __call__(self, x): 

        self.x = x.cpu()
        self.m = torch.zeros((self.x.size(0), 4)) 

        # 0.0 Initialise beam model    
        n = 10 # num_elems
        L = 20
        beam = Frame()  
        nodes = [[i*L/n, 0, 0] for i in range(n+1)]
        beam.add_nodes(nodes)

        # 0.1 Material (linear elastic)
        E = 30e9    # N/m2
        nu = 0.2    # poisson's ratio (uncracked) 
        rho = 2400  # kg/m3
        beam.add_material("concrete", LinearElastic(rho=rho, E=E, nu=nu))

        # 0.2 Find indices for supports
        xcoords = np.array(nodes)[:,0]
        support_0_ind = 0                               # end support at 0m 
        support_1_ind = np.argmin(np.abs(xcoords - 4))  # internal support at 4m 
        support_2_ind = n                               # end support at 20m 

        # 0.3 Store beam width with other variables 
        b = 2.5 # m 
        b_col = torch.full((self.x.shape[0], 1), b)
        self.x = torch.cat([self.x, b_col], dim=1)

        SMOKE_TEST = False  
        if SMOKE_TEST: 
            self.x = torch.tensor([[-50, 0.5, 1.0, 100, 100, b]]) # MN, m, m, kN/mm, kN/mm

        # models run sequentially
        for i, (P, e, d, k01, k02, _) in enumerate(self.x):  

            # 1. Get parameters 
            theta = 27.5 # deg            
            k1= (0.7*15) * k01 * 1e6 # N/m
            k1theta = (0.7*20) * k01 * 1e6 # Nm/rad          
            k2 = (0.7*2) * k02 * 1e6 # N/m 
            P = P * 1e6 # N 
            d = d.numpy() 

            # 2. Reset loads and supports 
            beam.reset_boundary_conditions()
            beam.reset_loads()
            beam.reset_elements()
            beam.reset_sections()

            # 3. Section  
            beam.add_section("rectangular", Section(A=b*d, Iy=b*d**3/12, Iz=1.0, J=1.0, Iw=1.0, y0=0, z0=0))
            beam.add_elements([[i, i+1] for i in range(n)], "concrete", "rectangular", element_class=ThinWalledBeamElement)
                
            # 4. Create new supports 
            # beam.add_boundary_condition(support_0_ind, [0, 0, 0, 1, 1, 1, 0], BoundaryCondition) 
            # beam.add_boundary_condition(support_2_ind, [0, 0, 0, 0, 1, 1, 0], BoundaryCondition) 
            beam.add_boundary_condition(support_0_ind, [0, 0, 1, 0, 0, 0, 0], BoundaryCondition) # end support 
            beam.add_elastic_boundary_condition(support_1_ind, dof_index=2, stiffness=k2)        # stiff vertical (internal) support 
            beam.add_elastic_boundary_condition(support_2_ind, dof_index=2, stiffness=k1)        # stiff vertical (end) support 
            beam.add_elastic_boundary_condition(support_2_ind, dof_index=4, stiffness=k1theta)   # stiff rotational (end) support 

            # 5. Apply loads (1.35 load factor applied)
            R, W = 1500e3, 600e3 # N 
            Rx = R * math.cos(math.radians(theta))          
            Ry = R * math.sin(math.radians(theta))                
            V = -(W + Ry)  
            M = (Rx * 21.3) - (Ry * (7.3/2 - 1)) - (W * 1) # Nm   
            beam.add_gravity_load([0, 0, -1.35]) 
            [beam.add_uniform_moment(i, [0, P*e, 0], UniformLoad) for i in range(n)] # prestress force (-ve) * e above N.A. (+ve) -> sagging moment (-ve)
            
            # 6. Solve model at transfer of prestress (self weight + prestress)
            results = beam.solve() 
            moment = results.process_element_forces(force_type='My', summary_type='all')
            Mpos0, Mneg0 = moment['My_max'], moment['My_min']
            Mhog0, Msag0 = max(next(iter(Mpos0.values())), 0), min(next(iter(Mneg0.values())), 0)   
            # print(Mpos0)
            # print(Mneg0) 
            if SMOKE_TEST: 
                beam.show_deformed_shape(scale=1e1, show_local_axes=False, show_cross_section=True, cross_section_scale=4.0) 
                beam.show_force_field(force_type='My', npoints=5, scale=2)         
            
            # 7. Solve model in service state (self weight + prestress + applied load)
            beam.add_nodal_load(n, [0, 0, 1.35*V, 0, 1.35*M, 0, 0], NodalLoad)
            results = beam.solve() 
            moment = results.process_element_forces(force_type='My', summary_type='all')
            Mpos1, Mneg1 = moment['My_max'], moment['My_min']
            # print(Mpos1)
            # print(Mneg1)
            Mhog1, Msag1 = max(next(iter(Mpos1.values())), 0), min(next(iter(Mneg1.values())), 0)   
            if SMOKE_TEST: 
                beam.show_deformed_shape(scale=1e1, show_local_axes=False, show_cross_section=True, cross_section_scale=4.0) 
                beam.show_force_field(force_type='My', npoints=5, scale=2)

            self.m[i] = torch.tensor([Mhog0, Msag0, Mhog1, Msag1])                                
            
    def f(self): 
        # beam cost f(P,d)
        P, fp, rho = self.m[:, 0], 0.8 * 1860e6, 7850 # N, N/m, kg/m3
        l, d, b = 20, self.x[:, 2], self.x[:, 5] 
        concrete = 145 * (l * b * d)
        tendons = 9000/1000 * (rho * l * P / fp)
        formwork = 36 * (2 * l*d + 2 * d*b)                                      
        return -(concrete + tendons + formwork)
        
    def g1(self):                                                                              
        # TRANSFER loadcase, max HOGGING, TOP fiber: σ = P/A + M/Z
        P, d, b = self.x[:, 0] * 1e6, self.x[:, 2], self.x[:, 5]                           
        A, Z = b*d, b*d**2/6 
        Mhog = self.m[:, 0] 
        stress = (P/A + Mhog/Z) * 1e-6 # kN/mm2, for hogging at top fiber, stress = P/A (-ve) + M/Z (+ve)
        # Want σ < ft, i.e., no tensile stress -> want g < 0
        return stress - 0.0

    def g2(self): 
        # TRANSFER loadcase, max HOGGING, BOTTOM fiber: σ = P/A - M/Z 
        P, d, b = self.x[:, 0] * 1e6, self.x[:, 2], self.x[:, 5]                           
        A, Z = b*d, b*d**2/6 
        Mhog = self.m[:, 0] 
        stress = (P/A - Mhog/Z) * 1e-6 # kN/mm2, for hogging at bottom fiber, stress = P/A (-ve) + M/Z (-ve)
        # Want -σ < fc, i.e., compressive stress within limit -> want g < 0                            
        return - stress - 20.0 
        
    def g3(self): 
        # TRANSFER loadcase, max SAGGING, TOP fiber: σ = P/A + M/Z 
        P, d, b = self.x[:, 0] * 1e6, self.x[:, 2], self.x[:, 5]                           
        A, Z = b*d, b*d**2/6 
        Msag = self.m[:, 1] 
        stress = (P/A + Msag/Z) * 1e-6 
        return - stress - 20.0

    def g4(self): 
        # TRANSFER loadcase, max SAGGING, BOTTOM fiber: σ = P/A - M/Z
        P, d, b = self.x[:, 0] * 1e6, self.x[:, 2], self.x[:, 5]                           
        A, Z = b*d, b*d**2/6 
        Msag = self.m[:, 1] 
        stress = (P/A - Msag/Z) * 1e-6 
        return stress - 0.0                                 
    
    def g5(self):                                                                               
        # SERVICE loadcase, max HOGGING, TOP fiber: σ = P/A + M/Z
        P, d, b = self.x[:, 0] * 1e6, self.x[:, 2], self.x[:, 5]                           
        A, Z = b*d, b*d**2/6 
        Mhog = self.m[:, 2] 
        stress = (P/A + Mhog/Z) * 1e-6 
        return stress - 0.0

    def g6(self): 
        # SERVICE loadcase, max HOGGING, BOTTOM fiber: σ = P/A - M/Z 
        P, d, b = self.x[:, 0] * 1e6, self.x[:, 2], self.x[:, 5]                           
        A, Z = b*d, b*d**2/6 
        Mhog = self.m[:, 2] 
        stress = (P/A - Mhog/Z) * 1e-6 
        return - stress - 30.0 
        
    def g7(self): 
        # SERVICE loadcase, max SAGGING, TOP fiber: σ = P/A + M/Z 
        P, d, b = self.x[:, 0] * 1e6, self.x[:, 2], self.x[:, 5]                           
        A, Z = b*d, b*d**2/6 
        Msag = self.m[:, 3] 
        stress = (P/A + Msag/Z) * 1e-6 
        return - stress - 30.0 

    def g8(self): 
        # SERVICE loadcase, max SAGGING, BOTTOM fiber: σ = P/A - M/Z
        P, d, b = self.x[:, 0] * 1e6, self.x[:, 2], self.x[:, 5]                           
        A, Z = b*d, b*d**2/6 
        Msag = self.m[:, 3] 
        stress = (P/A - Msag/Z) * 1e-6 
        return stress - 0.0 

def bayesopt(ninitial, iters, n): 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_dir  = os.path.join(base_folder, "prestress_rs")

    problem = Problem()
    model = Model()
    bounds = {"P": (-50, -125), "e": (-0.25, 0.5), 'd': (2.5, 4.0)}            # TODO verify -e works as expected

    # Uncertain parameters: ground stiffness for two pile groups
    mu = torch.tensor([100.0, 100.0])                   # k0_1, k0_2 [kN/mm]
    cov = torch.tensor([[    (25)**2,  0.5*(25)**2],    # COV = 25%, correlation = 0.5              TODO doesn't seem to be much variation with uncertain params? contors don't need many samples?
                        [0.5*(25)**2,      (25)**2]])
    dist = MultivariateNormal(mu, cov) 

    problem.set_bounds(bounds, padding=0.05) 
    problem.set_dist(dist) 
    problem.add_model(model) 
    problem.add_objectives([model.f]) 
    problem.add_constraints([model.g1, model.g2, model.g3, model.g4]) # Transfer state  
    problem.add_constraints([model.g5, model.g6, model.g7, model.g8]) # Service state

    # x = problem.sample(nsamples=100)
    # problem.model(x)
    # print(problem.constraints())
    # print((problem.constraints() < 0).all(dim=1))

    plotcontour(problem, None)

    # xi = problem.sample_xi(nsamples=int(2e2)).to(device)
    # surrogate = Surrogate(problem, ntraining=ninitial, nstarts=5) 
    # acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1) 
    # borc = Borc(surrogate, acquisition) 
    # borc.cuda(device) 
    # borc.initialize(nsamples=ninitial, sample_method="rand", max_acq=torch.tensor([0.0])) 
    # SurrogateIO.save(borc.surrogate, output_dir) 
    # borc.surrogate = SurrogateIO.load(output_dir) 

    # # params=(torch.linspace(0.1, 1.0, steps=10), torch.linspace(0.1, 1.0, steps=10)) 
    # # xopt, _ = problem.monte_carlo(params=params, nsamples=int(1e2), obj_type="mean", con_type="prob", con_eps=0.1, output=False) # [0.2, 0.4] £830 ??
    # # problem.rbo(xopt, nsamples=int(1e3), return_vals=True) 
    # plotcontour(problem, borc)

    # # BayesOpt used to sequentially sample [x,xi] points 
    # res = torch.ones(iters) 
    # for i in range(iters): 

    #     # params=(torch.linspace(0.1, 1.0, steps=10), torch.linspace(0.1, 1.0, steps=10)) 
    #     # xopt, _ = borc.surrogate.monte_carlo(params=params, nsamples=int(1e3), obj_type="mean", con_type="prob", con_eps=0.1) # ??
    #     # borc.rbo(xopt.to(device), nsamples=int(1e3), return_vals=True) 

    #     # new_x <- random search 
    #     borc.step(new_x=problem.sample()) 

    #     # argmax_x E[f(x,xi)] s.t. P[g(x,xi)<0]>1-epsilons 
    #     if i % n == 0: 
    #         xopt, _ = borc.constrained_optimize_acq(iters=int(2e2), nstarts=5, optimize_x=True) 
    #         res[i], _ = problem.rbo(xopt, output=False, return_vals=True) # true E[f(x,xi)] 
    #         print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

    # return xopt, res 
    return None, None 


if __name__ == "__main__": 

    ninitial, iters, n = 1, 1, 1 
    xopt, res = bayesopt(ninitial, iters, n) 
