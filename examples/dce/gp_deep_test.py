import torch
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

# Define a hidden layer for the DeepGP
class HiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=32):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])
        
        # Initialize variational distribution
        variational_dist = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        
        # Initialize variational strategy
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy, input_dims, output_dims)
        
        # Mean and covariance modules
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )
    
    def forward(self, x):
        # Get mean and covariance
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        # Return a MultivariateNormal
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define the DeepGP model
class SimpleDeepGP(DeepGP):
    def __init__(self, input_dims=1):
        super().__init__()
        
        # Hidden layer (input dim = 1, output dim = 2)
        self.hidden_layer = HiddenLayer(input_dims=input_dims, output_dims=2)
        
        # Output layer (input dim = 2, output dim = None)
        self.output_layer = HiddenLayer(input_dims=self.hidden_layer.output_dims, output_dims=None)
        
        # Define the likelihood
        self.likelihood = GaussianLikelihood()
    
    def forward(self, inputs):
        # Forward through hidden layer
        hidden_output = self.hidden_layer(inputs)
        
        # Forward through output layer
        output = self.output_layer(hidden_output)
        
        return output

# Generate a simple analytical test function
def f(x):
    return torch.sin(3 * x) + 0.5 * torch.sin(10 * x) 

# Set random seed for reproducibility
# torch.manual_seed(0)

# Generate train data
n_train = 20
train_x = torch.linspace(-1, 1, n_train).unsqueeze(-1)
train_y = f(train_x) + 0.1 * torch.randn_like(train_x)

# Generate test data
n_test = 100
test_x = torch.linspace(-1.5, 1.5, n_test).unsqueeze(-1)
test_y = f(test_x)

# Create and initialize the model
model = SimpleDeepGP()
likelihood = model.likelihood

# Use the Adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)

    # Marginal log likelihood
mll = DeepApproximateMLL(
    VariationalELBO(model.likelihood, model, train_x.size(0))
)

# Training loop
num_epochs = 1000
num_samples = 10  # Number of samples for training

for i in range(num_epochs):
    optimizer.zero_grad()
    
    # Use the specified number of samples for training
    with gpytorch.settings.num_likelihood_samples(num_samples):
        output = model(train_x)
        loss = -mll(output, train_y.squeeze())
        loss.backward()
        optimizer.step()
    
    if (i+1) % 200 == 0:
        print(f'Epoch {i+1}/{num_epochs} - Loss: {loss.item():.4f}')

# Set model in eval mode
model.eval()
likelihood.eval()

# Make predictions with multiple samples
with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_samples):
    predictions = likelihood(model(test_x))
    # Average over the samples
    mean = predictions.mean.mean(0)
    
    # Get lower and upper confidence bounds
    lower, upper = predictions.confidence_region()
    # Average over the samples
    lower = lower.mean(0)
    upper = upper.mean(0)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(test_x.numpy(), test_y.numpy(), 'k', label='True Function')
plt.plot(train_x.numpy(), train_y.numpy(), 'ko', label='Training Data')
plt.plot(test_x.numpy(), mean.numpy(), 'b', label='Predicted Mean')
plt.fill_between(test_x.numpy().flatten(), lower.numpy().flatten(), upper.numpy().flatten(), alpha=0.3, color='b', label='Confidence Region')
plt.legend()
plt.title('DeepGP on Simple Analytic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()