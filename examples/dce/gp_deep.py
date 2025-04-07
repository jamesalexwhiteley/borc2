import torch 
import gpytorch
import warnings
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP as GPytorchDeepGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, LinearKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from borc2.utilities import NormalScaler, GaussianScaler

# Suppress warnings
warnings.filterwarnings("ignore", message=r".*Negative variance values detected.*")

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Posterior():
    def __init__(self):
        self.mu = None 
        self.std = None 
        self.std_epistemic = None 
        self.std_aleatoric = None 
        self.var = None 
        self.cov = None 

    def posterior(self, return_epistemic=False, return_aleatoric=False):
        if return_epistemic: 
            return self.mu, self.std_epistemic
        elif return_aleatoric:
            return self.mu, self.std - self.std_epistemic
        else: 
            return self.mu, self.std

    def cuda(self, device):
        for attr, value in self.__dict__.items():
            if value is not None:
                setattr(self, attr, value.to(device))

# Hidden layer for DeepGP
class HiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=256): # NOTE number of inducing points, e.g. 64, 128, 256 
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])
        
        # Variational distribution
        variational_dist = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        
        # Variational strategy
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy, input_dims, output_dims)
        
        # Mean and covariance modules
        # self.mean_module = ConstantMean(batch_shape=batch_shape)
        # self.covar_module = ScaleKernel(
        #     RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
        #     batch_shape=batch_shape, ard_num_dims=None
        # )

        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=0.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Inner DeepGP model (GPytorch implementation)
class DeepGPModel(GPytorchDeepGP):
    def __init__(self, input_dims):
        super().__init__()
        
        self.hidden_layer = HiddenLayer(input_dims=input_dims, output_dims=4)
        self.output_layer = HiddenLayer(input_dims=self.hidden_layer.output_dims, output_dims=None)

        # self.hidden_layer1 = HiddenLayer(input_dims=input_dims, output_dims=8)
        # self.hidden_layer2 = HiddenLayer(input_dims=8, output_dims=4)
        # self.output_layer = HiddenLayer(input_dims=4, output_dims=None)
        
        self.likelihood = GaussianLikelihood()
    
    def forward(self, inputs):

        hidden_output = self.hidden_layer(inputs)
        output = self.output_layer(hidden_output)

        # hidden_output1 = self.hidden_layer1(inputs)
        # hidden_output2 = self.hidden_layer2(hidden_output1)
        # output = self.output_layer(hidden_output2)
        
        return output

class DeepGP:
    def __init__(self, 
                 train_x, 
                 train_y, 
                 normalize_x=True,
                 standardize_y=True, 
                 ntraining=10, 
                 nstarts=1):
        """
        DeepGP model
        
        Parameters
        ----------
        train_x : torch.Tensor, shape=(nsamples, nparam)  
            input test data 
        train_y : torch.Tensor, shape=(nsamples, )  
            output test data    
        """
        self.name = 'DeepGP'
        self.jitter = 1e-2
        self.normalize_x = normalize_x   
        self.standardize_y = standardize_y 
        self.ntraining = ntraining 
        self.nstarts = nstarts  
        self.fbest = torch.max(train_y) 
        self.xbest = train_x[list(torch.where(train_y == self.fbest))] 
        self.device = 'cpu'
        self.num_samples = 500 # NOTE number of monte carlo samples for variational inference 

        if self.normalize_x:
            self.scaler_x = NormalScaler(train_x, dim=0) 
            train_x = self.scaler_x.normalize(train_x) 

        if self.standardize_y:
            self.scaler_y = GaussianScaler(train_y, dim=0) 
            train_y = self.scaler_y.standardize(train_y) 

        self.train_x = train_x 
        self.train_y = train_y
        
        self.model = DeepGPModel(input_dims=train_x.size(1))
        self.likelihood = self.model.likelihood
        
        # Saving and loading 
        self.state_dict = self.model.state_dict
        self.load_state_dict = self.model.load_state_dict
    
    def cuda(self, device):
        """
        Move model to specified device
        """
        self.train_x = self.train_x.to(device)
        self.train_y = self.train_y.to(device)
        self.model = self.model.to(device)
        self.likelihood = self.likelihood.to(device)
        self.device = device
        self.scaler_x = self.scaler_x.to(device)
        self.scaler_y = self.scaler_y.to(device)
    
    def train(self, bool=True):
        """
        Switch the model to training mode
        """
        if bool:
            self.model.train()
            self.likelihood.train()
        else:
            self.model.eval()
            self.likelihood.eval()
    
    def eval(self):
        """
        Switch the model to evaluation mode
        """
        self.model.eval()
        self.likelihood.eval()
    
    def optimize_hyp(self):
        """
        Optimize the hyperparameters
        """
        best_state_dict, min_loss = None, torch.tensor(float('inf'))
        
        for _ in range(self.nstarts):

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
            
            # Log marginal likelihood
            mll = DeepApproximateMLL(
                VariationalELBO(self.model.likelihood, self.model, self.train_x.size(0))
            )
            
            # Training 
            for epoch in range(self.ntraining):
                optimizer.zero_grad()
                
                # Forward   
                with gpytorch.settings.num_likelihood_samples(self.num_samples):
                    output = self.model(self.train_x)
                    loss = -mll(output, self.train_y)
                    loss.backward()
                    optimizer.step()
                
                if (epoch + 1) % 1 == 0:
                    print(f'Epoch {epoch+1}/{self.ntraining} - Loss: {loss.item():.4f}')
            
            final_loss = loss.item()
            if final_loss < min_loss:
                min_loss = final_loss
                best_state_dict = self.model.state_dict()
        
        self.model.load_state_dict(best_state_dict)
    
    def fit(self):
        """
        Fit the model to the data
        """
        self.train(True)
        self.optimize_hyp()
        self.eval()
    
    def posterior(self, pred):
        """
        Get posterior distribution
        """
        mu = pred.mean
        if self.standardize_y:
            mu = self.scaler_y.unstandardize(mu)
        self.pred.mu = mu
        
        if self.return_std:
            # Get standard deviation (approximate from confidence interval)
            lower, upper = pred.confidence_region()
            std = (upper - lower) / (2 * 1.96)  
            std = std.mean(0)
            
            if self.standardize_y:
                std = self.scaler_y.unscale(std)
            
            self.pred.std = std
            self.pred.std_epistemic = std
            self.pred.std_aleatoric = torch.tensor(0.0)
        
        if self.return_var:
            var = pred.variance
            if self.standardize_y:
                var = self.scaler_y.unscale_var(var)
            self.pred.var = var
        
        return self.pred
    
    def predict(self, test_x, return_std=False, return_var=False, return_cov=False, grad=False):
        """
        Make predictions at point(s) x
        """
        x = test_x.float()
        device = next(self.model.parameters()).device
        self.model.to(device)
        
        self.pred = Posterior()
        self.return_std, self.return_var, self.return_cov = return_std, return_var, return_cov
        
        if self.normalize_x:
            x = self.scaler_x.normalize(x)
        
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.num_samples):
            self.model.eval()
            self.likelihood.eval()
            
            pred = self.likelihood(self.model(x))
            
            # Average over samples
            mean = pred.mean.mean(0)
            
            # Create a simple prediction object
            if self.standardize_y:
                mean = self.scaler_y.unstandardize(mean)
            
            self.pred.mu = mean
            
            if return_std:
                # Get lower and upper confidence bounds
                lower, upper = pred.confidence_region()
                # Approximate stddev from confidence interval
                std = (upper - lower) / (2 * 1.96)
                std = std.mean(0)
                
                if self.standardize_y:
                    std = self.scaler_y.unscale(std)
                
                self.pred.std = std
                self.pred.std_epistemic = std
                self.pred.std_aleatoric = torch.tensor(0.0)
            
            if return_var:
                var = pred.variance.mean(0)
                if self.standardize_y:
                    var = self.scaler_y.unscale_var(var)
                self.pred.var = var
            
            return self.pred