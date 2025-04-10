import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import math 
import pickle
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from borc2.problem import Problem 
from borc2.probability import MultivariateNormal
from borc2.utilities import tic, toc 

from static_fem.models import Frame # type:ignore 
from pystressed.models import SectionForce 
from pystressed.servicability import optimize_magnel

import warnings
warnings.filterwarnings("ignore", message=r".*Solution may be inaccurate. Try another solved*")  

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: James Whiteley (github.com/jamesalexwhiteley)
# Modified to use XGBoost instead of Deep GP

def plot_multistage_contours(problem, model, nsamples, steps):
    """Plot contours of objective function and constraint satisfaction probability"""
    base_folder = os.path.join(os.getcwd(), "figures")
    os.makedirs(base_folder, exist_ok=True)
    output_path = os.path.join(base_folder, "contour_prestress_xgb")

    plt.figure(figsize=(7, 6))

    tic()
    x = torch.linspace(0.1, 1, steps)
    y = torch.linspace(0.1, 1, steps)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    mu, pi = torch.zeros(steps**2), torch.zeros(steps**2)
    
    # For XGBoost, we don't need as many samples for Monte Carlo estimation
    pts, _ = problem.gen_batch_data(xpts, nsamples=nsamples, fixed_base_samples=True, method="lhs") 
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, pts.shape[0], batch_size):
        batch_end = min(i + batch_size, pts.shape[0])
        batch_pts = pts[i:batch_end]
        
        # Ensure batch_pts is 2D by reshaping if necessary
        if len(batch_pts.shape) != 2:
            # Get the total number of features
            total_features = np.prod(batch_pts.shape[1:]) if len(batch_pts.shape) > 1 else 1
            # Reshape to (n_samples, n_features)
            batch_pts = batch_pts.reshape(batch_pts.shape[0], total_features)
            
        model(batch_pts)
        f_val = model.f()
        g_val = model.g()
        mu[i:batch_end] = torch.mean(f_val)
        pi[i:batch_end] = (torch.sum(g_val <= 0) / nsamples).unsqueeze(0)
        
    MU = -mu.reshape(X.shape)
    PI = pi.reshape(X.shape)
    toc()
    
    # Plot
    contour_mu = plt.contourf(X.numpy(), Y.numpy(), MU.numpy(), cmap='PuBu')
    plt.colorbar(contour_mu, shrink=0.8, pad=0.05)
    contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='black', linewidths=1, levels=[0.1, 0.5, 0.9])
    plt.clabel(contour_pi, inline=True, fontsize=8)
    
    proxy = plt.Line2D([0], [0], color='black', lw=1.5)
    plt.xlabel(r'$b$ [m]')
    plt.ylabel(r'$h$ [m]')
    plt.legend([proxy], [r'$\text{P}[\text{g}(x,\xi)\leq 0]$'], loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()

class Model():
    def __init__(self):
        # Initialize XGBoost models as None
        self.mservice_model = None
        self.mtransfer_model = None
        self.prestress_model = None
        
        # Flag to use surrogate models or direct calculation
        self.use_surrogate = False
        self.structural_responses = None
    
    def train_surrogate_models(self, train_x, train_structural, train_prestress, cv_folds=3):
        """
        Train XGBoost models:
        1. Two separate models for structural responses:
           - Mservice model: maps (b, h, theta, k_a, k_theta_a, k_b) -> Mservice
           - Mtransfer model: maps (b, h, theta, k_a, k_theta_a, k_b) -> Mtransfer
        2. Prestress model: maps (b, h, Mservice, Mtransfer) -> P
        """
        print("Training surrogate models with XGBoost...")
        
        # Convert to numpy for XGBoost
        X = train_x.cpu().numpy()
        
        # Extract Mservice and train first model
        print("Training Mservice model...")
        mservice_targets = train_structural[:, 0].cpu().numpy()
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Use RandomizedSearchCV for faster hyperparameter tuning
        self.mservice_model = RandomizedSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror'),
            param_distributions=param_grid,
            n_iter=10,  # Try 10 different combinations
            scoring='neg_mean_squared_error',
            cv=cv_folds,
            verbose=1,
            n_jobs=-1  # Use all available cores
        )
        
        self.mservice_model.fit(X, mservice_targets)
        print(f"Best Mservice model params: {self.mservice_model.best_params_}")
        print(f"Mservice model R² score: {r2_score(mservice_targets, self.mservice_model.predict(X)):.4f}")
        
        # Extract Mtransfer and train second model
        print("Training Mtransfer model...")
        mtransfer_targets = train_structural[:, 1].cpu().numpy()
        
        self.mtransfer_model = RandomizedSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror'),
            param_distributions=param_grid,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=cv_folds,
            verbose=1,
            n_jobs=-1
        )
        
        self.mtransfer_model.fit(X, mtransfer_targets)
        print(f"Best Mtransfer model params: {self.mtransfer_model.best_params_}")
        print(f"Mtransfer model R² score: {r2_score(mtransfer_targets, self.mtransfer_model.predict(X)):.4f}")
        
        # Create inputs for prestress model: [b, h, Mservice, Mtransfer]
        prestress_inputs = torch.cat([train_x[:, :2], train_structural], dim=1).cpu().numpy()
        prestress_targets = train_prestress.squeeze(1).cpu().numpy()
        
        print("Training prestress optimization model...")
        self.prestress_model = RandomizedSearchCV(
            xgb.XGBRegressor(objective='reg:squarederror'),
            param_distributions=param_grid,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=cv_folds,
            verbose=1,
            n_jobs=-1
        )
        
        self.prestress_model.fit(prestress_inputs, prestress_targets)
        print(f"Best prestress model params: {self.prestress_model.best_params_}")
        print(f"Prestress model R² score: {r2_score(prestress_targets, self.prestress_model.predict(prestress_inputs)):.4f}")
        
        self.use_surrogate = True
        print("XGBoost models trained successfully")
    
    def __call__(self, x): 
        self.x = x.cpu()
        self.m = torch.zeros((self.x.size(0), 1))  # Prestress force P
        
        # Store structural responses for multi-stage modeling
        self.structural_responses = torch.zeros((self.x.size(0), 2))  # [Mservice, Mtransfer]
        
        # If surrogate models are available and enabled, use for prediction
        if self.use_surrogate and self.mservice_model is not None and self.mtransfer_model is not None and self.prestress_model is not None:
            x_numpy = self.x.numpy()
            
            # Ensure we're only using the first 6 features that the model was trained on
            # This handles the case where x_numpy has more features than expected
            features_for_prediction = x_numpy[:, :6] if x_numpy.shape[1] > 6 else x_numpy
            
            # Predict Mservice using the first model
            mservice_pred = self.mservice_model.predict(features_for_prediction)
            
            # Predict Mtransfer using the second model
            mtransfer_pred = self.mtransfer_model.predict(features_for_prediction)
            
            # Store predicted structural responses
            self.structural_responses[:, 0] = torch.tensor(mservice_pred)
            self.structural_responses[:, 1] = torch.tensor(mtransfer_pred)
            
            # Create inputs for prestress prediction: [b, h, Mservice, Mtransfer]
            prestress_inputs = np.column_stack([
                x_numpy[:, 0],  # b
                x_numpy[:, 1],  # h
                mservice_pred,  # Mservice
                mtransfer_pred  # Mtransfer
            ])
            
            prestress_pred = self.prestress_model.predict(prestress_inputs)
            self.m = torch.tensor(prestress_pred).unsqueeze(1)
            
            return
        
        # If no surrogate models or disabled, use direct calculation
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
        
    def save_models(self, base_path):
        """Save trained XGBoost models"""
        if self.mservice_model is not None and self.mtransfer_model is not None and self.prestress_model is not None:
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            
            with open(f"{base_path}_mservice.pkl", 'wb') as f:
                pickle.dump(self.mservice_model, f)
            
            with open(f"{base_path}_mtransfer.pkl", 'wb') as f:
                pickle.dump(self.mtransfer_model, f)
                
            with open(f"{base_path}_prestress.pkl", 'wb') as f:
                pickle.dump(self.prestress_model, f)
                
            print(f"Models saved with base path: {base_path}")
        else:
            print("No trained models to save")
            
    def load_models(self, base_path):
        """Load trained XGBoost models"""
        try:
            with open(f"{base_path}_mservice.pkl", 'rb') as f:
                self.mservice_model = pickle.load(f)
            
            with open(f"{base_path}_mtransfer.pkl", 'rb') as f:
                self.mtransfer_model = pickle.load(f)
                
            with open(f"{base_path}_prestress.pkl", 'rb') as f:
                self.prestress_model = pickle.load(f)
                
            self.use_surrogate = True
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.use_surrogate = False

def surrogate_modeling(): 
    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    models_base_path = os.path.join(base_folder, "xgb_multistage")
    
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
        npoints = 100
        steps = 5
        nsamples = 15 
        cv_folds = 2
    else: 
        npoints = 10000
        steps = 50
        nsamples = int(2e2)
        cv_folds = 10
    
    # Ground truth data generation
    print("Generating training data...")
    tic()
    train_x = problem.sample(nsamples=npoints, method='lhs')
    problem.model(train_x)
    train_f = problem.objectives().squeeze(1)
    train_g = problem.constraints().squeeze(1)
    
    # Get structural responses and prestress forces
    train_structural = model.structural_responses  # [Mservice, Mtransfer]
    train_prestress = model.m  # Prestress force P
    toc()
    
    # Train the XGBoost models
    print("Training XGBoost models...")
    tic()
    model.train_surrogate_models(
        train_x=train_x, 
        train_structural=train_structural, 
        train_prestress=train_prestress,
        cv_folds=cv_folds
    )
    model.save_models(models_base_path)
    toc()

    # Optional: load pre-trained models
    # model.load_models(models_base_path)

    # Plot results using XGBoost predictions
    plot_multistage_contours(problem, model, nsamples, steps)
    
    # Feature importance analysis
    if not SMOKE_TEST:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        xgb.plot_importance(model.mservice_model.best_estimator_)
        plt.title('Mservice Feature Importance')
        
        plt.subplot(1, 3, 2)
        xgb.plot_importance(model.mtransfer_model.best_estimator_)
        plt.title('Mtransfer Feature Importance')
        
        plt.subplot(1, 3, 3)
        xgb.plot_importance(model.prestress_model.best_estimator_)
        plt.title('Prestress Feature Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "figures", "xgb_feature_importance.png"), dpi=300)
        plt.show()

if __name__ == "__main__": 
    surrogate_modeling()