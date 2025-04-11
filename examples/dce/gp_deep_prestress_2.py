import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import math 

from borc2.problem import Problem 
from borc2.probability import MultivariateNormal
from borc2.utilities import tic, toc 
from borc2.gp import GPModelIO, HomoscedasticGP
from gp_deep import DeepGP

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

def plot_multistage_contours(problem, model, nsamples, steps):

    base_folder = os.path.join(os.getcwd(), "figures")
    os.makedirs(base_folder, exist_ok=True)
    output_path = os.path.join(base_folder, "contour_prestress_m")

    plt.figure(figsize=(7, 6))

    # TODO npoints=10000+, ntraining=2000, steps=500, nsamples=int(1e3) 2-layer deep architecture 
    # TODO npoints=10000+, ntraining=2000, steps=500, nsamples=int(1e3) 3-layer deep architecture 
    tic()
    x = torch.linspace(0.1, 1, steps)
    y = torch.linspace(0.1, 1, steps)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    mu, pi = torch.zeros(steps**2), torch.zeros(steps**2)
    model.use_deep_gp = True 
    pts, _ = problem.gen_batch_data(xpts, nsamples=nsamples, fixed_base_samples=True, method="lhs") 
    pts = pts.to(device)
    for i, pt in enumerate(pts):  
            model(pt)
            f_val = model.f()
            g_val = model.g()
            mu[i] = torch.mean(f_val)
            pi[i] = (torch.sum(g_val <= 0) / nsamples).unsqueeze(0)
    MU = - mu.reshape(X.shape)
    PI =   pi.reshape(X.shape)
    toc()
    
    # Plot
    contour_mu = plt.contourf(X.numpy(), Y.numpy(), MU.numpy(), cmap='PuBu')
    plt.colorbar(contour_mu, shrink=0.8, pad=0.05)
    contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='black', linewidths=1, levels=[0.1, 0.5, 0.9])
    plt.clabel(contour_pi, inline=True, fontsize=8)
    # contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='darkred', linewidths=2.5, levels=[0.9])
    # plt.clabel(contour_pi, inline=True, fontsize=8)
    
    proxy = plt.Line2D([0], [0], color='black', lw=1.5)
    plt.xlabel(r'$b$ [m]')
    plt.ylabel(r'$h$ [m]')
    plt.legend([proxy], [r'$\text{P}[\text{g}(x,\xi)\leq 0]$'], loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()

class Model():
    def __init__(self):
        # Initialize Deep GP models as None
        self.mservice_gp = None
        self.mtransfer_gp = None
        self.prestress_gp = None
        
        # Flag to use Deep GP or direct calculation
        self.use_deep_gp = False
        self.structural_responses = None
    
    def train_internal_gps(self, train_x, train_structural, train_prestress, ntraining=2000, num_samples=None):
        """
        Train internal Deep GPs:
        1. Two separate GPs for structural responses:
           - Mservice GP: maps (b, h, theta, k_a, k_theta_a, k_b) -> Mservice
           - Mtransfer GP: maps (b, h, theta, k_a, k_theta_a, k_b) -> Mtransfer
        2. Prestress GP: maps (b, h, Mservice, Mtransfer) -> P
        
        """
        # Set num_samples to make sure it's not larger than the batch size
        if num_samples is None:
            num_samples = min(100, train_x.size(0))
        
        print("Training Mservice Deep GP...")
        # Extract Mservice 
        mservice_targets = train_structural[:, 0]  
        
        self.mservice_gp = DeepGP(train_x, mservice_targets, ntraining=ntraining)
        self.mservice_gp.num_samples = num_samples  
        self.mservice_gp.cuda(device)
        self.mservice_gp.fit()
        
        print("Training Mtransfer Deep GP...")
        # Extract Mtransfer 
        mtransfer_targets = train_structural[:, 1]
        
        self.mtransfer_gp = DeepGP(train_x, mtransfer_targets, ntraining=ntraining)
        self.mtransfer_gp.num_samples = num_samples  
        self.mtransfer_gp.cuda(device)
        self.mtransfer_gp.fit()
        
        # Create inputs for prestress GP: [b, h, Mservice, Mtransfer]
        prestress_inputs = torch.cat([train_x[:, :2], train_structural], dim=1)

        prestress_targets = train_prestress.squeeze(1)
        
        print("Training prestress optimization Deep GP...")
        self.prestress_gp = DeepGP(prestress_inputs, prestress_targets, ntraining=ntraining)
        self.prestress_gp.num_samples = num_samples  
        self.prestress_gp.cuda(device)
        self.prestress_gp.fit()
        
        self.use_deep_gp = True
        print("Deep GP models trained successfully")
    
    def __call__(self, x): 
        self.x = x.cpu()
        self.m = torch.zeros((self.x.size(0), 1)) # Prestress force P
        
        # Store structural responses for multi-stage modeling
        self.structural_responses = torch.zeros((self.x.size(0), 2))  # [Mservice, Mtransfer]
        
        # If Deep GP models are available and enabled, use for prediction
        if self.use_deep_gp and self.mservice_gp is not None and self.mtransfer_gp is not None and self.prestress_gp is not None:
            x_device = x.to(device)
            
            # Predict Mservice using the first GP
            mservice_preds = self.mservice_gp.predict(x_device, return_std=False)
            mservice_pred = mservice_preds.mu
            
            # Predict Mtransfer using the second GP
            mtransfer_preds = self.mtransfer_gp.predict(x_device, return_std=False)
            mtransfer_pred = mtransfer_preds.mu
            
            # Store predicted structural responses
            self.structural_responses[:, 0] = mservice_pred.cpu()
            self.structural_responses[:, 1] = mtransfer_pred.cpu()
            structural_pred = torch.stack([mservice_pred, mtransfer_pred], dim=1)
            
            # Create inputs for prestress prediction: [b, h, Mservice, Mtransfer]
            prestress_inputs = torch.cat([x_device[:, :2], structural_pred], dim=1)
            prestress_preds = self.prestress_gp.predict(prestress_inputs, return_std=False)
            self.m = prestress_preds.mu.cpu().unsqueeze(1)
            
            return
        
        # If no Deep GPs or disabled, use direct calculation
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
            
            # Store structural responses for training later
            self.structural_responses[i, 0] = Mservice
            self.structural_responses[i, 1] = Mtransfer

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
        """Cost function"""
        P, fp, rho = self.m[:, 0], 0.8 * 1860e6, 7850 # N, N/m, kg/m3
        l, b, h = 20, self.x[:, 0], self.x[:, 1] 
        concrete = 145 * (l * b * h)
        tendons = 9000/1000 * (rho * l * P / fp)
        formwork = 36 * (l * b) 
        return -(concrete + tendons + formwork)
        
    def g(self): 
        """Constraint function"""
        # P >> 0 for feasible (e, P), hence g < 0 for feasible section 
        return -self.m.flatten() + 1
        
    def save_gps(self, base_path):
        """Save trained Deep GP models"""
        if self.mservice_gp is not None and self.mtransfer_gp is not None and self.prestress_gp is not None:
            mservice_path = f"{base_path}_mservice"
            mtransfer_path = f"{base_path}_mtransfer" 
            prestress_path = f"{base_path}_prestress"
            
            GPModelIO.save(self.mservice_gp, mservice_path)
            GPModelIO.save(self.mtransfer_gp, mtransfer_path)
            GPModelIO.save(self.prestress_gp, prestress_path)
            print(f"Models saved with base path: {base_path}")
        else:
            print("No trained models to save")
            
    def load_gps(self, base_path):
        """Load trained Deep GP models"""
        try:
            mservice_path = f"{base_path}_mservice"
            mtransfer_path = f"{base_path}_mtransfer" 
            prestress_path = f"{base_path}_prestress"
            
            self.mservice_gp = GPModelIO.load(mservice_path)
            self.mtransfer_gp = GPModelIO.load(mtransfer_path)
            self.prestress_gp = GPModelIO.load(prestress_path)
            self.use_deep_gp = True
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.use_deep_gp = False

def gaussian_process(): 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    models_base_path = os.path.join(base_folder, "gp_deep_multistage")
    
    # For comparison, also train traditional full models
    # objective_path = os.path.join(base_folder, "gp_deep_objective")
    # constraint_path = os.path.join(base_folder, "gp_deep_constraint")

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

    SMOKE_TEST = False              
    
    if SMOKE_TEST: 
        npoints = 20
        ntraining = 10 
        steps = 5
        nsamples = 15 
    else: 
        npoints = 5000
        ntraining = 1000
        steps = 10
        nsamples = int(1e2) 
    
    # Ground truth 
    train_x = problem.sample(nsamples=npoints, method='lhs')
    print("Generating training data...")
    problem.model(train_x)
    train_f = problem.objectives().squeeze(1)
    train_g = problem.constraints().squeeze(1)
    
    # Get structural responses and prestress forces
    train_structural = model.structural_responses  # [Mservice, Mtransfer]
    train_prestress = model.m  # Prestress force P
    
    # Train the internal Deep GPs
    print("Training internal multi-stage Deep GPs...")
    tic()
    model.train_internal_gps(
        train_x=train_x, 
        train_structural=train_structural, 
        train_prestress=train_prestress,
        ntraining=ntraining
    )
    model.save_gps(models_base_path)

    # Load model
    model.load_gps(models_base_path)
    toc()

    # Plot multistage approach 
    plot_multistage_contours(problem, model, nsamples, steps)

if __name__ == "__main__": 
    gaussian_process()