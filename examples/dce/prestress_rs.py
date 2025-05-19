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
from pybeamnlfea.model.load import NodalLoad 

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

    # fig = plt.figure(figsize=(7, 6))

    # TODO WE WANT TO PLOT g1(P,e,d)
    # 1. generate input data xpts 
    # 2. modify problem.rbo() to give us mu, prob1, prob2, ... 
    # 3. plot the 4x3 grid of P[g1(P,e,d)<0]

    # # Input data 
    # P_lower, P_upper = list(problem.param_bounds.values())[0]
    # e_lower, e_upper = list(problem.param_bounds.values())[1]
    # d_lower, d_upper = list(problem.param_bounds.values())[2]
    
    # P_vals = torch.linspace(P_lower, P_upper, steps=2)  
    # e_vals = torch.linspace(e_lower, e_upper, steps=2)  
    # d_vals = torch.linspace(d_lower, d_upper, steps=2)  

    # P_grid, e_grid, d_grid = torch.meshgrid(P_vals, e_vals, d_vals, indexing='ij')

    # # Each row is (P, e, d)
    # xpts = torch.stack([
    #     P_grid.reshape(-1), 
    #     e_grid.reshape(-1), 
    #     d_grid.reshape(-1)
    # ], dim=1)

    # for x in xpts: 
    #     pi = problem.rbo(x.unsqueeze(0), nsamples=int(1e1), output=False, return_vals=True)[1]

    #     print(pi)

    # Input data with more steps for better resolution
    P_lower, P_upper = list(problem.param_bounds.values())[0]
    e_lower, e_upper = list(problem.param_bounds.values())[1]
    d_lower, d_upper = list(problem.param_bounds.values())[2]

    # Use more steps for plotting (e.g., 20)
    plot_steps = 2
    P_vals = torch.linspace(P_lower, P_upper, steps=plot_steps)
    e_vals = torch.linspace(e_lower, e_upper, steps=plot_steps)
    d_vals = torch.linspace(d_lower, d_upper, steps=plot_steps)

    # Generate probability results for all points
    # For full resolution, we'd use all combinations, but that's computationally expensive
    # Instead, we'll calculate for specific slices we want to plot

    # Create cut values (4 cuts for each parameter) for the 4x3 grid
    P_cuts = torch.linspace(P_lower, P_upper, steps=3)
    e_cuts = torch.linspace(e_lower, e_upper, steps=3)
    d_cuts = torch.linspace(d_lower, d_upper, steps=3)

    # Initialize storage for results
    pe_results = []  # For f(P,e) with different d values
    pd_results = []  # For f(P,d) with different e values
    ed_results = []  # For f(e,d) with different P values

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
            pi = problem.rbo(x.unsqueeze(0), nsamples=int(1e1), output=False, return_vals=True)[1]
            probs.append(pi)
        
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
            pi = problem.rbo(x.unsqueeze(0), nsamples=int(1e1), output=False, return_vals=True)[1]
            probs.append(pi)
        
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
            pi = problem.rbo(x.unsqueeze(0), nsamples=int(1e1), output=False, return_vals=True)[1]
            probs.append(pi)
        
        # Reshape results to 2D grid for plotting
        prob_grid = torch.tensor(probs).reshape(e_grid_2d.shape)
        ed_results.append((e_grid_2d.numpy(), d_grid_2d.numpy(), prob_grid.numpy()))

    # Create the 4x3 grid plot
    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

    # Plot f(P,e) with different d values
    for i, (P_grid, e_grid, prob_grid) in enumerate(pe_results):
        ax = plt.subplot(gs[i, 0])
        contour = ax.contourf(P_grid, e_grid, prob_grid, levels=20, cmap='viridis')
        contour_lines = ax.contour(P_grid, e_grid, prob_grid, levels=[0.1, 0.5, 0.9], colors='black')
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        ax.set_xlabel('P (Force)')
        ax.set_ylabel('e (Eccentricity)')
        # Add annotation in top right
        ax.annotate(f"d = {d_cuts[i]:.2f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.grid(True, linestyle='--', alpha=0.7)

    # Plot f(P,d) with different e values
    for i, (P_grid, d_grid, prob_grid) in enumerate(pd_results):
        ax = plt.subplot(gs[i, 1])
        contour = ax.contourf(P_grid, d_grid, prob_grid, levels=20, cmap='viridis')
        contour_lines = ax.contour(P_grid, d_grid, prob_grid, levels=[0.1, 0.5, 0.9], colors='black')
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        ax.set_xlabel('P (Force)')
        ax.set_ylabel('d (Depth)')
        # Add annotation in top right
        ax.annotate(f"e = {e_cuts[i]:.2f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.grid(True, linestyle='--', alpha=0.7)

    # Plot f(e,d) with different P values
    for i, (e_grid, d_grid, prob_grid) in enumerate(ed_results):
        ax = plt.subplot(gs[i, 2])
        contour = ax.contourf(e_grid, d_grid, prob_grid, levels=20, cmap='viridis')
        contour_lines = ax.contour(e_grid, d_grid, prob_grid, levels=[0.1, 0.5, 0.9], colors='black')
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        ax.set_xlabel('e (Eccentricity)')
        ax.set_ylabel('d (Depth)')
        # Add annotation in top right
        ax.annotate(f"P = {P_cuts[i]:.0f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.07, wspace=0.25, hspace=0.3)
    plt.show()

    # TODO plot E[f(P,d)|xi] note this is now simply f(P,d) and is determinsitic 
    # TODO plot the 4x3 grid of P[g1(P,e,d)<0|xi] 

    # tic() 
    # steps = 50 
    # x = torch.linspace(0.1, 1, steps) 
    # y = torch.linspace(0.1, 1, steps) 
    # X, Y = torch.meshgrid(x, y, indexing='ij') 
    # xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1) 
    # mu, prob = zip(*[problem.rbo(x.unsqueeze(0), nsamples=int(5e3), output=False, return_vals=True) for x in xpts]) # list comprehension 
    # MU = -torch.tensor(mu).view(X.shape).detach() 
    # PI = torch.tensor(prob).view(X.shape).detach() 
    # toc() 
    # # save and load d
    # mu_path, pi_path = os.path.join(output_dir, 'mu_values.pt'), os.path.join(output_dir, 'pi_values.pt') 
    # torch.save(MU, mu_path) 
    # torch.save(PI, pi_path) 
    # MU = torch.load(mu_path, weights_only=True) 
    # PI = torch.load(pi_path, weights_only=True) 

    # tic() 
    # steps = 10 
    # x = torch.linspace(0.1, 1, steps) 
    # y = torch.linspace(0.1, 1, steps) 
    # X, Y = torch.meshgrid(x, y, indexing='ij')
    # xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)
    # mu, prob = zip(*[borc.rbo(x.unsqueeze(0), nsamples=int(1e3), output=False, return_vals=True) for x in xpts]) # list comprehension
    # MU = -torch.tensor(mu).view(X.shape).detach()
    # PI = torch.tensor(prob).view(X.shape).detach()
    # toc()

    # proxy = Line2D([0], [0], color='black', lw=1.5, label=r'\text{P}$[g(x,\xi)<0] = 1-\epsilon$')
    # contour_mu = plt.contourf(X.numpy(), Y.numpy(), MU.numpy(), cmap='PuBu')
    # contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='black', linewidths=1, levels=[0.1, 0.5, 0.9]) 
    # contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='black', linewidths=1, levels=[0.1, 0.5, 0.9])
    # plt.clabel(contour_pi, inline=True, fontsize=8)
    # plt.colorbar(contour_mu, shrink=0.8, pad=0.05)

    # plt.xlabel(r'$x_1$')
    # plt.ylabel(r'$x_2$')
    # plt.legend([proxy], [r'$\text{P}[\text{g}(x,\xi)\leq 0]$'], loc="upper left")
    # plt.tight_layout()
    # plt.savefig(output_path, dpi=600)
    # plt.show()

class Model():
    def __call__(self, x): 

        self.x = x.cpu()
        self.m = torch.zeros((self.x.size(0), 1)) 

        # 0.0 INITALISE BEAM MODEL   
        n = 10 # num_elems
        L = 10
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

        # models run sequentially
        for i, (P, e, d, theta, k_a, k_theta_a, k_b) in enumerate(self.x):  

            # 1. Get parameters 
            k_1, k_1_theta, k_2 = 1e8 * k_a, 1e10 * k_theta_a, 1e8 * k_b 
            d = d.numpy()

            # 2. Reset loads and supports 
            beam.reset_boundary_conditions()
            beam.reset_loads()
            beam.reset_elements()
            beam.reset_sections()

            # 3. Section 
            b = 1.0 # NOTE BEAM WIDTH 
            beam.add_section("rectangular", Section(A=b*d, Iy=b*d**3/12, Iz=1.0, J=1.0, Iw=1.0, y0=0, z0=0))
            beam.add_elements([[i, i+1] for i in range(n)], "concrete", "rectangular", element_class=ThinWalledBeamElement)
                
            # 4. Create new supports 
            beam.add_boundary_condition(support_0_ind, [0, 0, 1, 0, 0, 0, 0], BoundaryCondition) # end support 
            beam.add_elastic_boundary_condition(support_1_ind, dof_index=2, stiffness=k_2)       # stiff vertical (internal) support 
            beam.add_elastic_boundary_condition(support_2_ind, dof_index=2, stiffness=k_1)       # stiff vertical (end) support 
            beam.add_elastic_boundary_condition(support_2_ind, dof_index=4, stiffness=k_1_theta) # stiff rotational (end) support 

            # 5. Apply loads  
            R, W = 1.35 * 1500e3, 1.35 * 600e3              # N 
            Rx = R * math.cos(math.radians(theta))          
            Ry = R * math.sin(math.radians(theta))                
            V = -(W + Ry)                                   # N 
            M = (Rx * 21.3) - (Ry * (7.3/2 - 1)) - (W * 1)  # Nm 
            beam.add_nodal_load(n, [0, 0, V, 0, M, 0, 0], NodalLoad)
            beam.add_gravity_load()

            # 6. Solve and show model 
            results = beam.solve() 
            # beam.show_deformed_shape(scale=1, show_local_axes=False, show_cross_section=True, cross_section_scale=4.0) # TODO at least check self weight deflection is as expected 
            moment = results.process_element_forces(force_type='My', summary_type='all')
            M_pos, M_neg = moment['My_max'], moment['My_min']
            # beam.show_force_field(force_type='My', npoints=5, scale=2)

            M_hog = next(iter(M_pos.values()))                    
 
            self.m[i] = M_hog 
            
            # Service condition (with applied load) 
            # Transfer condition (only self-weight, and prestress?) 
            # print(f"Mservice {Mservice*1e-3:.4f} kNm, Mtransfer {Mtransfer*1e-3:.4f} kNm") 

    def f(self): 
        # beam cost f(P,d)
        P, fp, rho = self.m[:, 0], 0.8 * 1860e6, 7850 # N, N/m, kg/m3
        l, b, h = 20, self.x[:, 0], self.x[:, 1] 
        concrete = 145 * (l * b * h)
        tendons = 9000/1000 * (rho * l * P / fp)
        formwork = 36 * (l * b)                                      # TODO this needs to be changed to side area 
        return -(concrete + tendons + formwork)
        
    def g1(self): 
        # SERVICE loadcase, maximum HOGGING, TOP fiber: σ = P/A - Pe/Z + M/Z
        P, e, d = self.x[:, 0], self.x[:, 1], self.x[:, 2]                          # TODO we need a way to ensure e is feasible 
        b = 1.0 # NOTE BEAM WIDTH 
        A, Z = b*d, b*d**2/6  
        M_hog = self.m[0] # Maximum hogging moment 
        # print(M_hog)
        
        # For hogging (negative M) at top fiber, stress is:
        # Note the signs: -Pe/Z gives compression, +M/Z gives tension for negative M
        stress = P/A - P*e/Z + M_hog/Z
        
        # Return negative stress so g<0 means stress < allowable
        # return stress - allowable_stress  # Want g<0 for constraint satisfaction
        return stress

    # def g2(self): 
    #     # SERVICE loadcase, maximum HOGGING, BOTTOM fiber: σ = P/A + M/Z NOTE (?)
    #     pass 
        
    # def g3(self): 
    #     # SERVICE loadcase, maximum SAGGING, TOP fiber: σ = P/A + M/Z NOTE (?)
    #     pass 

    # def g4(self): 
    #     # SERVICE loadcase, maximum SAGGING, BOTTOM fiber: σ = P/A + M/Z NOTE (?)
    #     pass 

    # def g4(self): 
    #     # NEED A (DETERMINISTIC?) CONSTRAINT ON e BASED ON CROSS SECTION DIMENSIONS 
    #     pass 

def bayesopt(ninitial, iters, n): 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_dir  = os.path.join(base_folder, "prestress_rs")

    problem = Problem()
    model = Model()
    bounds = {"P": (1e3, 10e3), "e": (-0.5, 0.5), 'd': (0.2, 1.0)} # TODO need to reasonable variable bounds 

    # mu = torch.tensor([27.5, 1.5, 1.5, 1.5]) 
    # cov = torch.tensor([[(3.75)**0.5,      0.0,      0.0,       0.0],
    #                     [        0.0,         1,    0.9*1,    0.7*1],
    #                     [        0.0,     0.9*1,        1,    0.7*1],
    #                     [        0.0,     0.7*1,    0.7*1,        1]])
    mu = torch.tensor([27.5, 1.5, 1.5, 1.5]) 
    cov = torch.tensor([[(3.75)**0.5,      0.0,      0.0,       0.0],
                        [        0.0,         1,    0.9*1,    0.7*1],
                        [        0.0,     0.9*1,        1,    0.7*1],
                        [        0.0,     0.7*1,    0.7*1,        1]]) / 10 # NOTE TEST LOW VARIANCE 
    dist = MultivariateNormal(mu, cov)

    problem.set_bounds(bounds, padding=0.1) 
    problem.set_dist(dist) 
    problem.add_model(model) 
    problem.add_objectives([model.f]) 
    problem.add_constraints([model.g1]) 

    # for _ in range(3):
    #     x = problem.sample()
    #     # print(x)
    #     problem.model(x)
    #     problem.constraints()

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

    ninitial, iters, n = 100, 1, 1 
    xopt, res = bayesopt(ninitial, iters, n) 
