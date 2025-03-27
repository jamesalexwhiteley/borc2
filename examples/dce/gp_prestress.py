import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import math 

from borc2.problem import Problem 
from borc2.probability import MultivariateNormal
from borc2.utilities import tic, toc 
from borc2.gp import HomoscedasticGP, GPModelIO

from static_fem.models import Frame # type:ignore 
from pystressed.models import SectionForce 
from pystressed.servicability import plot_magnel, optimize_magnel, optimize_and_plot_magnel 

import warnings
warnings.filterwarnings("ignore", message=r".*Solution may be inaccurate. Try another solved*")  

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plotcontour2d(problem, gp1, gp2):  
    
    base_folder = os.path.join(os.getcwd(), "figures")
    os.makedirs(base_folder, exist_ok=True)
    output_path = os.path.join(base_folder, "contour_prestress")

    plt.figure(figsize=(7, 6))

    # tic() 
    # steps = 500 
    # x = torch.linspace(0.1, 1, steps)
    # y = torch.linspace(0.1, 1, steps)
    # X, Y = torch.meshgrid(x, y, indexing='ij')
    # xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    # nsamples=int(2e3)
    # pts, _ = problem.gen_batch_data(xpts, nsamples=nsamples, fixed_base_samples=True, method="lhs") 
    # mu, pi = torch.zeros(steps**2), torch.zeros(steps**2)
    # # use list (vector expression will run into memory issues)
    # for i, pt in enumerate(pts): 
    #     pred1 = gp1.predict(pt, return_std=False)
    #     mu[i] = torch.mean(pred1.mu)
    #     pred2 = gp2.predict(pt, return_std=False)
    #     pi[i] = (torch.sum(pred2.mu <= 0) / nsamples).unsqueeze(0) 
    # MU = - mu.reshape(X.shape)
    # PI =   pi.reshape(X.shape)
    # toc()

    # TODO steps=50, nsamples=int(1e3)
    tic() 
    steps = 10
    x = torch.linspace(0.1, 1, steps)
    y = torch.linspace(0.1, 1, steps)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    mu, prob = zip(*[problem.rbo(x.unsqueeze(0), nsamples=int(2e3), output=False, return_vals=True) for x in xpts]) 
    MU = -torch.tensor(mu).view(X.shape).detach()
    PI = torch.tensor(prob).view(X.shape).detach()
    toc() 

    proxy = Line2D([0], [0], color='black', lw=1.5, label=r'\text{P}$[g(x,\xi)<0] = 1-\epsilon$')
    contour_mu = plt.contourf(X.numpy(), Y.numpy(), MU.numpy(), cmap='PuBu')
    contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='black', linewidths=1, levels=[0.1, 0.5, 0.9])
    plt.clabel(contour_pi, inline=True, fontsize=8)
    plt.colorbar(contour_mu, shrink=0.8, pad=0.05)
    # contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='darkred', linewidths=2.5, levels=[0.9])
    # plt.clabel(contour_pi, inline=True, fontsize=8)

    plt.xlabel(r'$b$ [m]')
    plt.ylabel(r'$h$ [m]')
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

            k_a, k_theta_a, k_b = 1e8 * k_a, 1e10 * k_theta_a, 1e8 * k_b 

            l0, l1 = 4, 20 
            nodes = np.array([ 
                [0,  0], 
                [l0, 0], 
                [l1, 0]]) 
            elements = np.array([
                        [0, 1], 
                        [1, 2]]) 

            # section properties, self weight  
            E = 30e9 # N/m 
            A = b * h # m2 
            I = b * h**3 / 12 # m4 
            w = 1.35 * 24e3 * b * h # N/m              
            M0, V0 = -w*l0**2/12, -w*l0/2 # Nm, N
            M2, V2 = w*l1**2/12, -w*l1/2 
            M1, V1 = -M0 + -M2, V0 + V2

            # applied loads 
            R, W = 1.35 * 1500e3, 1.35 * 600e3 # N 
            Rx = R * math.cos(math.radians(theta))  # N
            Ry = R * math.sin(math.radians(theta))  # N        
            V = -(W + Ry) # N 
            M = -(Rx * 21.3) + (Ry * (7.3/2 - 1)) + (W * 1) # Nm 

            # staticFEM  
            frame = Frame(nodes, elements, E, A, I)  
            frame.add_constraints(dofs=[[1, 1, 1], [0, 0, 0], [0, 0, 0]], nodes=[0, 1, 2]) 
            frame.add_constraints(stiffness=[[0, k_b, 0], [0, k_a, k_theta_a]], nodes=[1, 2]) 

            # @service 
            frame.add_loads(loads=[[0, V0, M0], [0, V1, M1], [0, V+V2, M+M2]], nodes=[0, 1, 2]) 
            frame.initialise() 
            frame.solve() 
            Mservice = frame.f[2][0]
            
            # @transfer 
            M = V = 0 
            frame.add_loads(loads=[[0, V0, M0], [0, V1, M1], [0, V+V2, M+M2]], nodes=[0, 1, 2]) 
            frame.initialise()
            frame.solve() 
            Mtransfer = frame.f[2][0]

            # frame.show(figsize=(8, 4), 
            #         member_id=True, 
            #         node_id=False,
            #         supports=True, 
            #         nodal_forces=True, 
            #         nodal_disp=False,
            #         scale=100)
            
            # print(f"Mservice {Mservice*1e-3:.4f} kNm, Mtransfer {Mtransfer*1e-3:.4f} kNm")

            # @transfer 
            A = b * h * 1e6 # mm2 
            fc = 18 # N/mm2             
            ft = 0 # N/mm2 
            Ztop = I / (h/2) * 1e9 # mm3 
            Zbot = I / (h/2) * 1e9 # mm3 
            Mmax = Mtransfer * 1e3 # Nmm     
            Mmin = Mtransfer * 1e3 # Nmm            
            losses = 0.95 
            ebounds = [0, (h/2 - 0.1) * 1e3] # mm 
            transfer = SectionForce(A=A, fc=fc, ft=ft, Ztop=Ztop, Zbot=Zbot, Mmax=Mmax, Mmin=Mmin, losses=losses) 

            # @service 
            A = b * h * 1e6 # mm2 
            fc = 18 # N/mm2            
            ft = 0 # N/mm2 
            Ztop = I / (h/2) * 1e9 # mm3
            Zbot = I / (h/2) * 1e9 # mm3
            Mmax = Mservice * 1e3 # Nmm            
            Mmin = Mservice * 1e3 # Nmm            
            losses = 0.85 
            ebounds = [0, (h/2 - 0.1) * 1e3] # mm 
            service = SectionForce(A=A, fc=fc, ft=ft, Ztop=Ztop, Zbot=Zbot, Mmax=Mmax, Mmin=Mmin, losses=losses) 

            # design 
            P, _ = optimize_magnel(transfer=transfer, service=service, ebounds=ebounds, mode='min', output=False) 
            
            self.m[i] = P 

    def f(self): 
        P, fp, rho = self.m[:, 0], 0.8 * 1860e6, 7850 # N, N/m, kg/m3
        l, b, h = 20, self.x[:, 0], self.x[:, 1] 
        concrete = 145 * (l * b * h)
        tendons = 9000/1000 * (rho * l * P / fp)
        formwork = 36 * (l * b) 
        return -(concrete + tendons + formwork)
        
    def g(self): 
        # P >> 0 for feasible (e, P), hence g < 0 for feasible section 
        return -self.m.flatten() + 1 

def gaussian_process(npoints): 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_path_1 = os.path.join(base_folder, "gp_prestress_1")
    output_path_2 = os.path.join(base_folder, "gp_prestress_2")

    problem = Problem()
    model = Model()
    bounds = {"b": (0.1, 1.0), "h": (0.1, 1.0)} 
    
    mu = torch.tensor([27.5, 1.5, 1.5, 1.5]) 
    cov = torch.tensor([[(3.75)**0.5,      0.0,      0.0,       0.0],
                        [        0.0,         1,    0.9*1,    0.7*1],
                        [        0.0,     0.9*1,        1,    0.7*1],
                        [        0.0,     0.7*1,    0.7*1,        1]])
    dist = MultivariateNormal(mu, cov)

    problem.set_bounds(bounds, padding=0.1) 
    problem.set_dist(dist)
    problem.add_model(model)
    problem.add_objectives([model.f])
    problem.add_constraints([model.g])

    # train_x = problem.sample(nsamples=npoints, method='rand')
    # problem.model(train_x)
    # train_f = problem.objectives().squeeze(1)
    # train_g = problem.constraints().squeeze(1)

    # # objective GP
    # gp1 = HomoscedasticGP(train_x, train_f) 
    # gp1.fit() 
    # GPModelIO.save(gp1, output_path_1) 

    # # constraint GP
    # gp2 = HomoscedasticGP(train_x, train_g) 
    # gp2.fit() 
    # GPModelIO.save(gp2, output_path_2)   

    gp1 = GPModelIO.load(output_path_1)  
    gp2 = GPModelIO.load(output_path_2) 
    
    plotcontour2d(problem, gp1, gp2)

if __name__ == "__main__": 
    npoints = 500 
    gaussian_process(npoints) 
