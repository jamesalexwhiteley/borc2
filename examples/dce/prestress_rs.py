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

# from static_fem.models import Frame # type:ignore 
# from pystressed.models import SectionForce 
# from pystressed.servicability import plot_magnel, optimize_magnel, optimize_and_plot_magnel 

from pybeamnlfea.model.frame import Frame # type:ignore
from pybeamnlfea.model.material import LinearElastic # type:ignore
from pybeamnlfea.model.section import Section # type:ignore
from pybeamnlfea.model.element import ThinWalledBeamElement # type:ignore
from pybeamnlfea.model.boundary import BoundaryCondition # type:ignore
from pybeamnlfea.model.load import NodalLoad # type:ignore 
import numpy as np 

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

    fig = plt.figure(figsize=(7, 6))

    # TODO steps=50 nsamples=5e3 estimate 50 hours 
    tic() 
    steps = 50
    x = torch.linspace(0.1, 1, steps) 
    y = torch.linspace(0.1, 1, steps) 
    X, Y = torch.meshgrid(x, y, indexing='ij') 
    xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1) 
    mu, prob = zip(*[problem.rbo(x.unsqueeze(0), nsamples=int(5e3), output=False, return_vals=True) for x in xpts]) # list comprehension 
    MU = -torch.tensor(mu).view(X.shape).detach() 
    PI = torch.tensor(prob).view(X.shape).detach() 
    toc() 
    # save and load 
    mu_path, pi_path = os.path.join(output_dir, 'mu_values.pt'), os.path.join(output_dir, 'pi_values.pt') 
    torch.save(MU, mu_path) 
    torch.save(PI, pi_path) 
    MU = torch.load(mu_path, weights_only=True) 
    PI = torch.load(pi_path, weights_only=True) 

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

    proxy = Line2D([0], [0], color='black', lw=1.5, label=r'\text{P}$[g(x,\xi)<0] = 1-\epsilon$')
    contour_mu = plt.contourf(X.numpy(), Y.numpy(), MU.numpy(), cmap='PuBu')
    contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='black', linewidths=1, levels=[0.1, 0.5, 0.9]) 
    contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='black', linewidths=1, levels=[0.1, 0.5, 0.9])
    plt.clabel(contour_pi, inline=True, fontsize=8)
    plt.colorbar(contour_mu, shrink=0.8, pad=0.05)

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend([proxy], [r'$\text{P}[\text{g}(x,\xi)\leq 0]$'], loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()

class Model():
    def __call__(self, x): 

        self.x = x.cpu()
        self.m = torch.zeros((self.x.size(0), 1)) 

        # models run sequentially
        for i, (b, h, theta, k_a, k_theta_a, k_b) in enumerate(self.x):  

            k_1, k_1_theta, k_2 = 1e8 * k_a, 1e10 * k_theta_a, 1e8 * k_b 
            b, h = b.numpy(), h.numpy()

            # Beam  
            n = 10 # num elements 
            L = 20
            beam = Frame() 
            nodes = [[i*L/n, 0, 0] for i in range(n+1)]
            beam.add_nodes(nodes)

            # Material (linear elastic)
            E = 30e9    # N/m2
            nu = 0.2    # poisson's ratio (uncracked) 
            rho = 2400  # kg/m3

            # Section 
            b, h = 0.5, 1.0 # NOTE 
            A = b*h         # m2
            Iy = b*h**3/12  # m4
            Iz = 1.0        # m4
            J =  1.0        # m4
            Iw = 1.0        # m6 

            beam.add_material("concrete", LinearElastic(rho=rho, E=E, nu=nu))
            beam.add_section("rectangular", Section(A=A, Iy=Iy, Iz=Iz, J=J, Iw=Iw, y0=0, z0=0))
            beam.add_elements([[i, i+1] for i in range(n)], "concrete", "rectangular", element_class=ThinWalledBeamElement)

            # Find indices for supports
            xcoords = np.array(nodes)[:,0]
            support_0_ind = 0                               # end support at 0m 
            support_1_ind = np.argmin(np.abs(xcoords - 4))  # internal support at 4m 
            support_2_ind = n                               # end support at 20m 
            
            # # end support 
            # beam.add_boundary_condition(support_0_ind, [0, 0, 1, 0, 0, 0, 0], BoundaryCondition) 
            # # internal support 
            # beam.add_elastic_boundary_condition(support_1_ind, dof_index=2, stiffness=k_2)       # stiff vertical support 
            # # end support 
            # beam.add_elastic_boundary_condition(support_2_ind, dof_index=2, stiffness=k_1)       # stiff vertical support 
            # beam.add_elastic_boundary_condition(support_2_ind, dof_index=4, stiffness=k_1_theta) # stiff rotational support 

            # simply supported 
            # beam.add_boundary_condition(support_0_ind, [0, 0, 0, 0, 1, 1, 1], BoundaryCondition) 
            # beam.add_boundary_condition(support_2_ind, [1, 0, 0, 1, 1, 1, 1], BoundaryCondition) 
            beam.add_boundary_condition(support_0_ind, [0, 0, 0, 0, 1, 0, 0], BoundaryCondition)  # Only θy free
            beam.add_boundary_condition(support_2_ind, [1, 0, 0, 0, 1, 0, 0], BoundaryCondition)  # ux and θy free

            # Applied loads 
            R, W = 1.35 * 1500e3, 1.35 * 600e3              # N 
            Rx = R * math.cos(math.radians(theta))          
            Ry = R * math.sin(math.radians(theta))                
            V = -(W + Ry)                                   # N 
            M = (Rx * 21.3) - (Ry * (7.3/2 - 1)) - (W * 1)  # Nm 

            # beam.add_nodal_load(n, [0, 0, V, 0, M, 0, 0], NodalLoad)
            # beam.add_nodal_load(n, [1, 0, 0, 0, 0, 0, 0], NodalLoad)
            beam.add_nodal_load(n//2, [0, 0, -1, 0, 0, 0, 0], NodalLoad)
            # beam.add_nodal_load(n//2, [-1, -0.5, -1, 0, 20, 0, 0], NodalLoad)
            # beam.add_gravity_load()

            # Solve and show model 
            results = beam.solve() 
            
            beam.show_deformed_shape(scale=1e7, show_local_axes=False, show_cross_section=True, cross_section_scale=4.0)
            beam.show_force_field(force_type='My')

            # TODO decide what's reasonable with drawing BMD and getting positive/negative values for design 
            element_forces = results.process_element_internal_forces(element_id=0, force_type='My', summary_type='max')
            print(element_forces['My_max'])
            print(results.process_element_forces(force_type='My', summary_type='min'))

            # Service condition (with applied load) 
            # Transfer condition (only self-weight, and prestress?) 
            # print(f"Mservice {Mservice*1e-3:.4f} kNm, Mtransfer {Mtransfer*1e-3:.4f} kNm") 

            # self.m[i] = P 

    def f(self): 
        # beam cost f(P,d)
        P, fp, rho = self.m[:, 0], 0.8 * 1860e6, 7850 # N, N/m, kg/m3
        l, b, h = 20, self.x[:, 0], self.x[:, 1] 
        concrete = 145 * (l * b * h)
        tendons = 9000/1000 * (rho * l * P / fp)
        formwork = 36 * (l * b) 
        return -(concrete + tendons + formwork)
        
    def g1(self): 
        # SERVICE loadcase, maximum HOGGING, TOP fiber: σ = P/A - Pe/Z + M/Z
        P, e = self.m[:, 0], self.m[:, 1]  
        b, d = self.m[:, 2], self.m[:, 3]  
        A, Z = b * d, b * d**2 / 6  
        M_hog = self.m[:, 4] # Maximum hogging moment (negative) NOTE is this negative?
        
        # For hogging (negative M) at top fiber, stress is:
        # Note the signs: -Pe/Z gives compression, +M/Z gives tension for negative M
        stress = P/A - P*e/Z + M_hog/Z
        
        # Return negative stress so g<0 means stress < allowable
        # return stress - allowable_stress  # Want g<0 for constraint satisfaction
        return stress

    def g2(self): 
        # SERVICE loadcase, maximum HOGGING, BOTTOM fiber: σ = P/A + M/Z NOTE (?)
        pass 
        
    def g3(self): 
        # SERVICE loadcase, maximum SAGGING, TOP fiber: σ = P/A + M/Z NOTE (?)
        pass 

    def g4(self): 
        # SERVICE loadcase, maximum SAGGING, BOTTOM fiber: σ = P/A + M/Z NOTE (?)
        pass 

def bayesopt(ninitial, iters, n): 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_dir  = os.path.join(base_folder, "prestress_rs")

    problem = Problem()
    model = Model()
    # bounds = {"b": (0.2, 1.0), "h": (0.2, 1.0)}
    bounds = {"b": (0.5, 1.0), "h": (0.5, 1.0)} # NOTE limit to 0.5 m 
    
    # mu = torch.tensor([27.5, 1.5, 1.5, 1.5]) 
    # cov = torch.tensor([[(3.75)**0.5,      0.0,      0.0,       0.0],
    #                     [        0.0,         1,    0.9*1,    0.7*1],
    #                     [        0.0,     0.9*1,        1,    0.7*1],
    #                     [        0.0,     0.7*1,    0.7*1,        1]])
    mu = torch.tensor([27.5, 1.5, 1.5, 1.5]) 
    cov = torch.tensor([[(3.75)**0.5,      0.0,      0.0,       0.0],
                        [        0.0,         1,    0.9*1,    0.7*1],
                        [        0.0,     0.9*1,        1,    0.7*1],
                        [        0.0,     0.7*1,    0.7*1,        1]]) / 10 # NOTE low variance 
    dist = MultivariateNormal(mu, cov)

    problem.set_bounds(bounds, padding=0.1) 
    problem.set_dist(dist) 
    problem.add_model(model) 
    problem.add_objectives([model.f]) 
    problem.add_constraints([model.g1]) 

    for _ in range(1):
        x = problem.sample()
        # print(x)
        problem.model(x)

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
