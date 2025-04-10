import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import math

from borc2.problem import Problem
from borc2.probability import MultivariateNormal
from borc2.utilities import tic, toc

from static_fem.models import Frame # type:ignore
from pystressed.models import SectionForce
from pystressed.servicability import plot_magnel, optimize_magnel, optimize_and_plot_magnel

import warnings
warnings.filterwarnings("ignore", message=r".*Solution may be inaccurate. Try another solved*")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Author: James Whiteley (github.com/jamesalexwhiteley)
# Modified to use neural networks instead of Deep GP

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 128]):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NNModelIO:
    @staticmethod
    def save(model, path):
        """Save the neural network model"""
        torch.save(model.state_dict(), f"{path}.pt")
        print(f"Model saved to {path}.pt")
        
    @staticmethod
    def load(model, path):
        """Load the neural network model"""
        model.load_state_dict(torch.load(f"{path}.pt"))
        model.eval()
        return model


def train_nn_model(model, train_loader, val_loader=None, epochs=200, lr=0.001):
    """Train a neural network model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.MSELoss()
    
    model.to(device)
    training_losses = []
    validation_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20  # Early stopping patience
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)
            
            # Learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            # Print progress without validation
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.6f}")
    
    return model, training_losses, validation_losses


def plot_multistage_contours(problem, model, nsamples, steps):
    """Plot contours of the multistage model"""
    base_folder = os.path.join(os.getcwd(), "figures")
    os.makedirs(base_folder, exist_ok=True)
    output_path = os.path.join(base_folder, "contour_prestress_nn")

    plt.figure(figsize=(7, 6))

    tic()
    x = torch.linspace(0.1, 1, steps)
    y = torch.linspace(0.1, 1, steps)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    mu, pi = torch.zeros(steps**2), torch.zeros(steps**2)
    
    pts, _ = problem.gen_batch_data(xpts, nsamples=nsamples, fixed_base_samples=True, method="lhs") 
    pts = pts.to(device)
    
    for i, pt in enumerate(pts):
        model(pt)
        f_val = model.f()
        g_val = model.g()
        mu[i] = torch.mean(f_val)
        pi[i] = (torch.sum(g_val <= 0) / nsamples).unsqueeze(0)
    
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
        # Initialize Neural Network models as None
        self.mservice_nn = None
        self.mtransfer_nn = None
        self.prestress_nn = None
        
        # Flag to use Neural Network or direct calculation
        self.use_nn = False
        self.structural_responses = None
        
        # For generating samples during inference
        self.num_mc_samples = 100
    
    def train_internal_nns(self, train_x, train_structural, train_prestress, batch_size=64, val_split=0.2, epochs=200):
        """
        Train internal Neural Networks:
        1. Two separate NNs for structural responses:
           - Mservice NN: maps (b, h, theta, k_a, k_theta_a, k_b) -> Mservice
           - Mtransfer NN: maps (b, h, theta, k_a, k_theta_a, k_b) -> Mtransfer
        2. Prestress NN: maps (b, h, Mservice, Mtransfer) -> P
        """
        # Split data into training and validation sets
        n_samples = train_x.size(0)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        mservice_train_dataset = TensorDataset(
            train_x[train_indices], 
            train_structural[train_indices, 0:1]
        )
        mservice_val_dataset = TensorDataset(
            train_x[val_indices], 
            train_structural[val_indices, 0:1]
        )
        
        mtransfer_train_dataset = TensorDataset(
            train_x[train_indices], 
            train_structural[train_indices, 1:2]
        )
        mtransfer_val_dataset = TensorDataset(
            train_x[val_indices], 
            train_structural[val_indices, 1:2]
        )
        
        # Create inputs for prestress NN: [b, h, Mservice, Mtransfer]
        prestress_inputs = torch.cat([train_x[:, :2], train_structural], dim=1)
        
        prestress_train_dataset = TensorDataset(
            prestress_inputs[train_indices], 
            train_prestress[train_indices]
        )
        prestress_val_dataset = TensorDataset(
            prestress_inputs[val_indices], 
            train_prestress[val_indices]
        )
        
        # Create data loaders
        mservice_train_loader = DataLoader(mservice_train_dataset, batch_size=batch_size, shuffle=True)
        mservice_val_loader = DataLoader(mservice_val_dataset, batch_size=batch_size)
        
        mtransfer_train_loader = DataLoader(mtransfer_train_dataset, batch_size=batch_size, shuffle=True)
        mtransfer_val_loader = DataLoader(mtransfer_val_dataset, batch_size=batch_size)
        
        prestress_train_loader = DataLoader(prestress_train_dataset, batch_size=batch_size, shuffle=True)
        prestress_val_loader = DataLoader(prestress_val_dataset, batch_size=batch_size)
        
        # Create and train models
        print("Training Mservice Neural Network...")
        self.mservice_nn = NeuralNetwork(input_dim=train_x.size(1), output_dim=1)
        self.mservice_nn, _, _ = train_nn_model(
            self.mservice_nn, 
            mservice_train_loader, 
            mservice_val_loader, 
            epochs=epochs
        )
        
        print("Training Mtransfer Neural Network...")
        self.mtransfer_nn = NeuralNetwork(input_dim=train_x.size(1), output_dim=1)
        self.mtransfer_nn, _, _ = train_nn_model(
            self.mtransfer_nn, 
            mtransfer_train_loader, 
            mtransfer_val_loader, 
            epochs=epochs
        )
        
        print("Training prestress optimization Neural Network...")
        self.prestress_nn = NeuralNetwork(input_dim=prestress_inputs.size(1), output_dim=1)
        self.prestress_nn, _, _ = train_nn_model(
            self.prestress_nn, 
            prestress_train_loader, 
            prestress_val_loader, 
            epochs=epochs
        )
        
        self.use_nn = True
        print("Neural Network models trained successfully")
    
    def __call__(self, x):
        """Process input x to compute prestress force P"""
        self.x = x.cpu()
        self.m = torch.zeros((self.x.size(0), 1))  # Prestress force P
        
        # Store structural responses for multi-stage modeling
        self.structural_responses = torch.zeros((self.x.size(0), 2))  # [Mservice, Mtransfer]
        
        # If Neural Network models are available and enabled, use for prediction
        if self.use_nn and self.mservice_nn is not None and self.mtransfer_nn is not None and self.prestress_nn is not None:
            x_device = x.to(device)
            
            # Set models to evaluation mode
            self.mservice_nn.eval()
            self.mtransfer_nn.eval()
            self.prestress_nn.eval()
            
            with torch.no_grad():
                # For uncertainty estimation in prediction, we can use Monte Carlo dropout
                # by making multiple forward passes with dropout enabled
                
                # Predict Mservice
                self.mservice_nn.train()  # Enable dropout for MC sampling
                mservice_samples = torch.zeros(self.x.size(0), self.num_mc_samples)
                for i in range(self.num_mc_samples):
                    mservice_samples[:, i] = self.mservice_nn(x_device).cpu().squeeze()
                
                mservice_pred = mservice_samples.mean(dim=1)
                
                # Predict Mtransfer
                self.mtransfer_nn.train()  # Enable dropout for MC sampling
                mtransfer_samples = torch.zeros(self.x.size(0), self.num_mc_samples)
                for i in range(self.num_mc_samples):
                    mtransfer_samples[:, i] = self.mtransfer_nn(x_device).cpu().squeeze()
                
                mtransfer_pred = mtransfer_samples.mean(dim=1)
                
                # Store predicted structural responses
                self.structural_responses[:, 0] = mservice_pred
                self.structural_responses[:, 1] = mtransfer_pred
                
                # Create inputs for prestress prediction: [b, h, Mservice, Mtransfer]
                prestress_inputs = torch.cat([
                    x_device[:, :2], 
                    mservice_pred.to(device).unsqueeze(1), 
                    mtransfer_pred.to(device).unsqueeze(1)
                ], dim=1)
                
                # Predict prestress
                self.prestress_nn.train()  # Enable dropout for MC sampling
                prestress_samples = torch.zeros(self.x.size(0), self.num_mc_samples)
                for i in range(self.num_mc_samples):
                    prestress_samples[:, i] = self.prestress_nn(prestress_inputs).cpu().squeeze()
                
                self.m = prestress_samples.mean(dim=1).unsqueeze(1)
            
            return
        
        # If no Neural Networks or disabled, use direct calculation (same as original)
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
        
    def save_nns(self, base_path):
        """Save trained Neural Network models"""
        if self.mservice_nn is not None and self.mtransfer_nn is not None and self.prestress_nn is not None:
            mservice_path = f"{base_path}_mservice"
            mtransfer_path = f"{base_path}_mtransfer" 
            prestress_path = f"{base_path}_prestress"
            
            NNModelIO.save(self.mservice_nn, mservice_path)
            NNModelIO.save(self.mtransfer_nn, mtransfer_path)
            NNModelIO.save(self.prestress_nn, prestress_path)
            print(f"Models saved with base path: {base_path}")
        else:
            print("No trained models to save")
            
    def load_nns(self, base_path):
        """Load trained Neural Network models"""
        try:
            mservice_path = f"{base_path}_mservice"
            mtransfer_path = f"{base_path}_mtransfer" 
            prestress_path = f"{base_path}_prestress"
            
            # Create new NN models with the right input/output dimensions
            self.mservice_nn = NeuralNetwork(input_dim=6, output_dim=1)  # (b, h, theta, k_a, k_theta_a, k_b) -> Mservice
            self.mtransfer_nn = NeuralNetwork(input_dim=6, output_dim=1)  # (b, h, theta, k_a, k_theta_a, k_b) -> Mtransfer
            self.prestress_nn = NeuralNetwork(input_dim=4, output_dim=1)  # (b, h, Mservice, Mtransfer) -> P
            
            self.mservice_nn = NNModelIO.load(self.mservice_nn, mservice_path)
            self.mtransfer_nn = NNModelIO.load(self.mtransfer_nn, mtransfer_path)
            self.prestress_nn = NNModelIO.load(self.prestress_nn, prestress_path)
            
            self.use_nn = True
            print("Models loaded successfully")
            
            # Move models to device
            self.mservice_nn.to(device)
            self.mtransfer_nn.to(device)
            self.prestress_nn.to(device)
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.use_nn = False


def neural_network_training():
    """Train and evaluate neural network models"""
    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    models_base_path = os.path.join(base_folder, "nn_multistage")

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
        epochs = 10
        steps = 5
        nsamples = 15
        batch_size = 4
    else: 
        npoints = 40000  # Increased from 10000 for better coverage
        epochs = 100     # Increased for better convergence
        steps = 50       # Increased for finer resolution
        nsamples = int(1e3)
        batch_size = 128
    
    # Ground truth data generation
    print("Generating training data...")
    train_x = problem.sample(nsamples=npoints, method='lhs')
    problem.model(train_x)
    train_f = problem.objectives().squeeze(1)
    train_g = problem.constraints().squeeze(1)
    
    # Get structural responses and prestress forces
    train_structural = model.structural_responses  # [Mservice, Mtransfer]
    train_prestress = model.m  # Prestress force P
    
    # Data analysis and preprocessing
    print("\nData Analysis:")
    print(f"Input shape: {train_x.shape}")
    print(f"Structural responses shape: {train_structural.shape}")
    print(f"Prestress shape: {train_prestress.shape}")
    
    # Check for NaN or infinity values
    has_nan = torch.isnan(train_structural).any() or torch.isnan(train_prestress).any()
    has_inf = torch.isinf(train_structural).any() or torch.isinf(train_prestress).any()
    print(f"Contains NaN values: {has_nan}")
    print(f"Contains infinity values: {has_inf}")
    
    # Replace any NaN or infinity values (if they exist)
    if has_nan or has_inf:
        train_structural = torch.nan_to_num(train_structural, nan=0.0, posinf=1e6, neginf=-1e6)
        train_prestress = torch.nan_to_num(train_prestress, nan=0.0, posinf=1e6, neginf=0.0)
        print("Replaced NaN and infinity values")
    
    # Basic statistics
    print(f"\nStructural responses stats:")
    print(f"Mservice - min: {train_structural[:, 0].min()}, max: {train_structural[:, 0].max()}, mean: {train_structural[:, 0].mean()}")
    print(f"Mtransfer - min: {train_structural[:, 1].min()}, max: {train_structural[:, 1].max()}, mean: {train_structural[:, 1].mean()}")
    print(f"Prestress - min: {train_prestress.min()}, max: {train_prestress.max()}, mean: {train_prestress.mean()}")
    
    # Train the internal Neural Networks
    print("\nTraining internal multi-stage Neural Networks...")
    tic()
    model.train_internal_nns(
        train_x=train_x, 
        train_structural=train_structural, 
        train_prestress=train_prestress,
        batch_size=batch_size,
        epochs=epochs
    )
    model.save_nns(models_base_path)
    toc()

    # Evaluate model on test data
    print("\nEvaluating model performance...")
    test_x = problem.sample(nsamples=min(1000, npoints//10), method='lhs')
    
    # Get ground truth
    problem.model(test_x)
    test_prestress_true = model.m.clone()
    
    # Use neural network to predict
    model.load_nns(models_base_path)
    model(test_x)
    test_prestress_pred = model.m
    
    # Calculate metrics
    mse = torch.mean((test_prestress_true - test_prestress_pred) ** 2)
    mae = torch.mean(torch.abs(test_prestress_true - test_prestress_pred))
    max_error = torch.max(torch.abs(test_prestress_true - test_prestress_pred))
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Max Error: {max_error:.6f}")
    
    # Plot histograms of the errors
    plt.figure(figsize=(10, 6))
    errors = (test_prestress_true - test_prestress_pred).flatten().numpy()
    plt.hist(errors, bins=50, alpha=0.7)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig(os.path.join(os.getcwd(), "figures", "nn_error_distribution.png"), dpi=300)
    
    # Plot multistage contours
    plot_multistage_contours(problem, model, nsamples, steps)


if __name__ == "__main__":
    neural_network_training()