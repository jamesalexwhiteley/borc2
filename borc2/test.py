import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader

from borc2.problem import Problem 

# Assume that Model and Problem classes are already defined
# from problem_definition import Problem, Model

# Define StructuralDeepGPLayer
class StructuralDeepGPLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(StructuralDeepGPLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
            
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Define StructuralDeepGP to model constraints
class StructuralDeepGP(DeepGP):
    def __init__(self, input_dims):
        super().__init__()
        
        # First hidden layer - map from input space to 3D hidden space
        self.hidden_layer1 = StructuralDeepGPLayer(
            input_dims=input_dims,
            output_dims=3,
            mean_type='linear',
        )
        
        # Second hidden layer - further processing
        self.hidden_layer2 = StructuralDeepGPLayer(
            input_dims=self.hidden_layer1.output_dims,
            output_dims=2,
            mean_type='linear',
        )
        
        # Output layer - map to constraint value
        self.output_layer = StructuralDeepGPLayer(
            input_dims=self.hidden_layer2.output_dims,
            output_dims=None,
            mean_type='constant',
        )
        
        # Define the likelihood
        self.likelihood = GaussianLikelihood()
    
    def forward(self, inputs):
        hidden1 = self.hidden_layer1(inputs)
        hidden2 = self.hidden_layer2(hidden1)
        output = self.output_layer(hidden2)
        return output
    
    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            
            for x_batch, _ in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                
        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)

# Main function to generate data and train DeepGP
def train_deepgp_structural_model(problem, n_samples=500, num_epochs=1000, batch_size=64):
    # Generate training data
    print("Generating training data...")
    x_train = problem.sample(nsamples=n_samples, method="sobol")
    
    # Run the model to get constraints
    problem.model(x_train)
    g_train = problem.constraints()
    
    # Create training dataset and dataloader
    train_dataset = TensorDataset(x_train, g_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create test data (fewer samples, for evaluation)
    x_test = problem.sample(nsamples=100, method="lhs")
    problem.model(x_test)
    g_test = problem.constraints()
    test_dataset = TensorDataset(x_test, g_test)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Initialize DeepGP model
    input_dims = x_train.shape[1]
    model = StructuralDeepGP(input_dims=input_dims)
    
    # Use the Adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.01)
    
    # Marginal log likelihood
    mll = DeepApproximateMLL(
        VariationalELBO(model.likelihood, model, train_dataset.__len__())
    )
    
    # Training loop
    print("Training DeepGP model...")
    losses = []
    
    for i in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Number of samples to use for training
            with gpytorch.settings.num_likelihood_samples(5):
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        # Record and print progress
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        
        if (i+1) % 100 == 0:
            print(f'Epoch {i+1}/{num_epochs} - Loss: {epoch_loss:.4f}')
            
            # Calculate test metrics (RMSE)
            model.eval()
            with torch.no_grad(), gpytorch.settings.num_likelihood_samples(20):
                y_pred, y_var = model.predict(test_loader)
                rmse = torch.sqrt(torch.mean((y_pred.mean(0) - g_test) ** 2))
                print(f'Test RMSE: {rmse.item():.4f}')
    
    return model, x_train, g_train, x_test, g_test, losses

# Evaluation and visualization functions
def evaluate_deepgp_model(model, problem, n_test=200):
    # Generate new test points across the parameter space
    x_test = problem.sample(nsamples=n_test, method="sobol")
    
    # Get actual constraint values
    problem.model(x_test)
    g_actual = problem.constraints()
    
    # Create test dataset and dataloader
    test_dataset = TensorDataset(x_test, g_actual)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    # Get predictions from DeepGP model
    model.eval()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(20):
        g_pred, g_var = model.predict(test_loader)
        g_pred_mean = g_pred.mean(0)
        g_std = torch.sqrt(g_var.mean(0))
    
    # Calculate metrics
    rmse = torch.sqrt(torch.mean((g_pred_mean - g_actual) ** 2))
    
    # Calculate reliability - how often is our model correct about constraint satisfaction
    # g ≤ 0 means constraint is satisfied
    actual_satisfied = (g_actual <= 0).float()
    pred_satisfied = (g_pred_mean <= 0).float()
    reliability = (actual_satisfied == pred_satisfied).float().mean()
    
    print(f"Test RMSE: {rmse.item():.4f}")
    print(f"Constraint prediction reliability: {reliability.item()*100:.2f}%")
    
    return x_test, g_actual, g_pred_mean, g_std

# Visualize predictions (2D slice for interpretability)
def visualize_2d_slice(model, problem, fixed_vals=None, resolution=50):
    # Create a 2D grid of points by varying the first two parameters
    # and fixing the rest to their median values
    if fixed_vals is None:
        # Sample points to find reasonable fixed values
        samples = problem.sample(nsamples=100)
        fixed_vals = torch.median(samples, dim=0).values
    
    # Create grid for the first two parameters (e.g., b and h)
    b_range = torch.linspace(problem.param_bounds['b'][0], problem.param_bounds['b'][1], resolution)
    h_range = torch.linspace(problem.param_bounds['h'][0], problem.param_bounds['h'][1], resolution)
    
    grid_points = []
    for b in b_range:
        for h in h_range:
            # Create a full parameter vector, varying only b and h
            point = fixed_vals.clone()
            point[0] = b
            point[1] = h
            grid_points.append(point)
    
    grid_x = torch.stack(grid_points)
    
    # Get actual constraint values
    problem.model(grid_x)
    g_actual = problem.constraints()
    
    # Get predictions from DeepGP model
    test_dataset = TensorDataset(grid_x, g_actual)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    model.eval()
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(20):
        g_pred, g_var = model.predict(test_loader)
        g_pred_mean = g_pred.mean(0)
        g_std = torch.sqrt(g_var.mean(0))
    
    # Reshape for plotting
    g_actual_grid = g_actual.reshape(resolution, resolution)
    g_pred_grid = g_pred_mean.reshape(resolution, resolution)
    g_std_grid = g_std.reshape(resolution, resolution)
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot actual constraint values
    im1 = axs[0].imshow(g_actual_grid, origin='lower', extent=[
        problem.param_bounds['b'][0], problem.param_bounds['b'][1],
        problem.param_bounds['h'][0], problem.param_bounds['h'][1]
    ], aspect='auto', cmap='viridis')
    axs[0].set_title('Actual Constraint Values')
    axs[0].set_xlabel('b (width)')
    axs[0].set_ylabel('h (height)')
    plt.colorbar(im1, ax=axs[0])
    
    # Plot predicted constraint values
    im2 = axs[1].imshow(g_pred_grid, origin='lower', extent=[
        problem.param_bounds['b'][0], problem.param_bounds['b'][1],
        problem.param_bounds['h'][0], problem.param_bounds['h'][1]
    ], aspect='auto', cmap='viridis')
    axs[1].set_title('DeepGP Predicted Constraint Values')
    axs[1].set_xlabel('b (width)')
    axs[1].set_ylabel('h (height)')
    plt.colorbar(im2, ax=axs[1])
    
    # Plot prediction uncertainty (standard deviation)
    im3 = axs[2].imshow(g_std_grid, origin='lower', extent=[
        problem.param_bounds['b'][0], problem.param_bounds['b'][1],
        problem.param_bounds['h'][0], problem.param_bounds['h'][1]
    ], aspect='auto', cmap='plasma')
    axs[2].set_title('DeepGP Prediction Uncertainty')
    axs[2].set_xlabel('b (width)')
    axs[2].set_ylabel('h (height)')
    plt.colorbar(im3, ax=axs[2])
    
    # Mark constraint boundary (g=0)
    for ax in axs[:2]:
        CS = ax.contour(b_range, h_range, g_actual_grid, levels=[0], colors='red')
        ax.clabel(CS, inline=True, fontsize=10, fmt='g=0')
        
        CS = ax.contour(b_range, h_range, g_pred_grid, levels=[0], colors='blue')
        ax.clabel(CS, inline=True, fontsize=10, fmt='g_pred=0')
    
    plt.tight_layout()
    plt.show()

# Main function to run the entire workflow
def main():
    # Create problem instance
    problem = Problem()
    model = Model()
    
    # Define parameter bounds
    bounds = {"b": (0.1, 1.0), "h": (0.1, 1.0)}
    
    # Define uncertain parameters distribution
    mu = torch.tensor([27.5, 1.5, 1.5, 1.5])
    cov = torch.tensor([
        [(3.75)**0.5, 0.0, 0.0, 0.0],
        [0.0, 1, 0.9*1, 0.7*1],
        [0.0, 0.9*1, 1, 0.7*1],
        [0.0, 0.7*1, 0.7*1, 1]
    ])
    dist = MultivariateNormal(mu, cov)
    
    # Set up problem
    problem.set_bounds(bounds, padding=0.1)
    problem.set_dist(dist)
    problem.add_model(model)
    problem.add_objectives([model.f])
    problem.add_constraints([model.g])
    
    # Train DeepGP model
    deepgp_model, x_train, g_train, x_test, g_test, losses = train_deepgp_structural_model(
        problem, 
        n_samples=500, 
        num_epochs=1000, 
        batch_size=64
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Evaluate model performance
    x_eval, g_actual, g_pred, g_std = evaluate_deepgp_model(deepgp_model, problem, n_test=200)
    
    # Visualize predictions
    visualize_2d_slice(deepgp_model, problem, resolution=30)
    
    return deepgp_model, problem

if __name__ == "__main__":
    main()